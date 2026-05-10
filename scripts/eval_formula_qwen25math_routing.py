"""F-formula Mode B: Qwen2.5-Math-7B encoder for formula passages, RRF-fused with Qwen3-Embedding-4B baseline.

Mode B routing:
  - Qwen3-Embedding-4B handles all 2809 passages (existing baseline ranking).
  - Qwen2.5-Math-7B re-encodes ONLY the 1253 formula passages.
  - For each query: queries are encoded by both models; formula passages get
    two ranks (Q3 + Math), non-formula passages get one rank (Q3 only).
  - Reciprocal-rank fusion combines them. Sweep RRF k in {10, 20, 60}.

Outputs (under --output-dir):
  - formula_embeddings.npy / formula_passage_ids.json   : Math embeddings of formula passages
  - query_embeddings.npy / query_ids.json               : Math embeddings of queries
  - ranking_rrf_k{10,20,60}.jsonl                       : fused top-K rankings
  - eval_report.json                                    : R@K + MRR (overall + per-modality), formula bucket on M4query_v1 + smoke50

Run inside minerU env. Qwen2.5-Math-7B: hidden_size=3584, max_position_embeddings=4096.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
from torch.utils.data import DataLoader
from transformers import AutoModel, AutoTokenizer


# ---------- I/O ----------

def read_jsonl(path: Path) -> Iterable[dict]:
    with path.open() as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def write_jsonl(path: Path, rows: Iterable[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


# ---------- Encoder ----------

class Qwen25MathEncoder:
    """Qwen2.5-Math-7B as encoder via mean-pooled last hidden state, L2-normalized."""

    def __init__(self, model_path: str, device: str = "cuda", dtype: torch.dtype = torch.bfloat16) -> None:
        print(f"[encoder] loading {model_path} ...", flush=True)
        t0 = time.time()
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = AutoModel.from_pretrained(
            model_path,
            torch_dtype=dtype,
            trust_remote_code=True,
        ).to(device).eval()
        self.device = device
        self.hidden_size = self.model.config.hidden_size
        print(f"[encoder] loaded in {time.time()-t0:.1f}s, hidden_size={self.hidden_size}", flush=True)

    @torch.inference_mode()
    def encode(self, texts: list[str], batch_size: int = 4, max_length: int = 512) -> np.ndarray:
        out = []
        n = len(texts)
        for i in range(0, n, batch_size):
            batch = texts[i:i+batch_size]
            enc = self.tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors="pt",
            ).to(self.device)
            hidden = self.model(**enc).last_hidden_state            # (B, L, H)
            mask = enc.attention_mask.unsqueeze(-1).to(hidden.dtype)
            summed = (hidden * mask).sum(dim=1)
            denom = mask.sum(dim=1).clamp(min=1.0)
            pooled = summed / denom                                  # (B, H)
            pooled = torch.nn.functional.normalize(pooled.float(), p=2, dim=-1)
            out.append(pooled.cpu().numpy())
            if (i // batch_size) % 25 == 0:
                print(f"[encoder]   {i+len(batch)}/{n}", flush=True)
        return np.concatenate(out, axis=0)


# ---------- Fusion + metrics ----------

def rrf_fuse(
    baseline_top_k: list[str],
    math_formula_rank: dict[str, int],
    formula_set: set[str],
    k: int,
    output_top: int = 100,
) -> list[str]:
    """Reciprocal-rank fusion.

    - baseline_top_k: Q3 top-100 (ordered).
    - math_formula_rank: passage_id -> 1-indexed rank within math formula ranking
      (only contains formula passages).
    - formula_set: set of all formula passage_ids.
    - k: RRF constant.
    """
    scores: dict[str, float] = {}
    # Baseline contribution (all top_k passages)
    for r, pid in enumerate(baseline_top_k, start=1):
        scores[pid] = scores.get(pid, 0.0) + 1.0 / (k + r)
    # Math contribution (formula passages only, may add new pids not in baseline)
    for pid, r in math_formula_rank.items():
        if pid in formula_set:
            scores[pid] = scores.get(pid, 0.0) + 1.0 / (k + r)
    # Sort
    sorted_pids = sorted(scores.keys(), key=lambda p: -scores[p])
    return sorted_pids[:output_top]


def compute_metrics(ranking: list[str], gold: set[str], ks: list[int]) -> dict:
    out = {}
    for K in ks:
        topk = ranking[:K]
        hits = sum(1 for p in topk if p in gold)
        out[f"R@{K}"] = hits / len(gold) if gold else 0.0
    # MRR @ first hit in entire ranking
    mrr = 0.0
    for r, pid in enumerate(ranking, start=1):
        if pid in gold:
            mrr = 1.0 / r
            break
    out["MRR"] = mrr
    return out


def aggregate(per_query: list[dict], ks: list[int]) -> dict:
    if not per_query:
        return {f"R@{K}": 0.0 for K in ks} | {"MRR": 0.0, "n": 0}
    out = {"n": len(per_query)}
    for K in ks:
        out[f"R@{K}"] = sum(q[f"R@{K}"] for q in per_query) / len(per_query)
    out["MRR"] = sum(q["MRR"] for q in per_query) / len(per_query)
    return out


# ---------- Main ----------

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", required=True, type=Path,
                    help="corpus_v1_enriched.jsonl (passage_id, type, text, ...)")
    ap.add_argument("--queries", required=True, type=Path)
    ap.add_argument("--qrels", required=True, type=Path)
    ap.add_argument("--baseline-ranking", required=True, type=Path,
                    help="ranking_v1_enriched.jsonl from Qwen3-Embedding-4B (top-100 per query)")
    ap.add_argument("--smoke50-queries", type=Path, default=None,
                    help="optional smoke50 source: either a .jsonl with query_id field or a txt with one id per line")
    ap.add_argument("--model-path", default="Qwen/Qwen2.5-Math-7B",
                    help="HF id or local path (will auto-download to HF cache)")
    ap.add_argument("--output-dir", required=True, type=Path)
    ap.add_argument("--passage-max-length", type=int, default=384)
    ap.add_argument("--query-max-length", type=int, default=128)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--rrf-ks", default="10,20,60",
                    help="comma-separated RRF k values to sweep")
    ap.add_argument("--output-top", type=int, default=100)
    args = ap.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Load corpus, isolate formula passages
    print("[main] loading corpus ...", flush=True)
    formula_passages: list[dict] = []
    formula_ids: list[str] = []
    type_by_pid: dict[str, str] = {}
    for row in read_jsonl(args.corpus):
        type_by_pid[row["passage_id"]] = row["type"]
        if row["type"] == "formula":
            formula_ids.append(row["passage_id"])
            formula_passages.append(row)
    formula_set = set(formula_ids)
    print(f"[main] formula passages: {len(formula_passages)}", flush=True)

    # Load queries
    print("[main] loading queries ...", flush=True)
    queries: list[dict] = list(read_jsonl(args.queries))
    print(f"[main] queries: {len(queries)}", flush=True)

    # Load qrels
    qrels_by_qid: dict[str, set[str]] = {}
    for row in read_jsonl(args.qrels):
        if row.get("relevance", 0) > 0:
            qrels_by_qid.setdefault(row["query_id"], set()).add(row["passage_id"])

    # Load baseline ranking
    baseline_by_qid: dict[str, list[str]] = {}
    for row in read_jsonl(args.baseline_ranking):
        baseline_by_qid[row["query_id"]] = row["top_k"]

    # Smoke50 slice
    smoke50: set[str] = set()
    if args.smoke50_queries and args.smoke50_queries.exists():
        if args.smoke50_queries.suffix == ".jsonl":
            smoke50 = {row["query_id"] for row in read_jsonl(args.smoke50_queries)}
        else:
            smoke50 = {ln.strip() for ln in args.smoke50_queries.read_text().splitlines() if ln.strip()}
        print(f"[main] smoke50 queries: {len(smoke50)}", flush=True)

    # ---- Encode ----
    encoder = Qwen25MathEncoder(args.model_path)

    print("[main] encoding formula passages ...", flush=True)
    t0 = time.time()
    formula_emb = encoder.encode(
        [p["text"] for p in formula_passages],
        batch_size=args.batch_size,
        max_length=args.passage_max_length,
    )
    print(f"[main] formula encode: {formula_emb.shape}, {time.time()-t0:.1f}s", flush=True)
    np.save(args.output_dir / "formula_embeddings.npy", formula_emb)
    (args.output_dir / "formula_passage_ids.json").write_text(json.dumps(formula_ids))

    print("[main] encoding queries ...", flush=True)
    t0 = time.time()
    query_texts = [q["query"] for q in queries]
    query_ids = [q["query_id"] for q in queries]
    query_emb = encoder.encode(
        query_texts,
        batch_size=args.batch_size,
        max_length=args.query_max_length,
    )
    print(f"[main] query encode: {query_emb.shape}, {time.time()-t0:.1f}s", flush=True)
    np.save(args.output_dir / "query_embeddings.npy", query_emb)
    (args.output_dir / "query_ids.json").write_text(json.dumps(query_ids))

    # Free GPU
    del encoder
    torch.cuda.empty_cache()

    # ---- Score formula per query (cosine on already-normalized embeddings) ----
    print("[main] computing math formula rankings ...", flush=True)
    sim = query_emb @ formula_emb.T                  # (Q, F)
    math_rank_per_query: dict[str, dict[str, int]] = {}
    formula_top_K = min(200, len(formula_ids))
    order = np.argsort(-sim, axis=1)[:, :formula_top_K]
    for qi, qid in enumerate(query_ids):
        ranks = {formula_ids[order[qi, r]]: r + 1 for r in range(formula_top_K)}
        math_rank_per_query[qid] = ranks

    # ---- RRF fusion across k ----
    rrf_ks = [int(k) for k in args.rrf_ks.split(",")]
    overall_metrics = {}
    smoke_metrics = {}
    formula_bucket_metrics_overall = {}
    formula_bucket_metrics_smoke = {}

    metric_ks = [1, 2, 5, 10, 100]

    for k in rrf_ks:
        print(f"[main] RRF fusion k={k} ...", flush=True)
        per_query: list[dict] = []
        per_query_smoke: list[dict] = []
        per_query_formula: list[dict] = []
        per_query_formula_smoke: list[dict] = []
        out_rows = []
        for q in queries:
            qid = q["query_id"]
            base = baseline_by_qid.get(qid, [])
            math_ranks = math_rank_per_query.get(qid, {})
            fused = rrf_fuse(base, math_ranks, formula_set, k=k, output_top=args.output_top)
            gold = qrels_by_qid.get(qid, set())
            metrics = compute_metrics(fused, gold, metric_ks)
            metrics["qid"] = qid
            per_query.append(metrics)
            if qid in smoke50:
                per_query_smoke.append(metrics)
            # formula bucket: gold contains at least one formula passage
            gold_has_formula = any(type_by_pid.get(p) == "formula" for p in gold)
            if gold_has_formula:
                # Formula bucket metric: only count formula gold
                formula_gold = {p for p in gold if type_by_pid.get(p) == "formula"}
                formula_metrics = compute_metrics(fused, formula_gold, metric_ks)
                formula_metrics["qid"] = qid
                per_query_formula.append(formula_metrics)
                if qid in smoke50:
                    per_query_formula_smoke.append(formula_metrics)
            out_rows.append({"query_id": qid, "top_k": fused})
        write_jsonl(args.output_dir / f"ranking_rrf_k{k}.jsonl", out_rows)
        overall_metrics[f"k{k}"] = aggregate(per_query, metric_ks)
        smoke_metrics[f"k{k}"] = aggregate(per_query_smoke, metric_ks)
        formula_bucket_metrics_overall[f"k{k}"] = aggregate(per_query_formula, metric_ks)
        formula_bucket_metrics_smoke[f"k{k}"] = aggregate(per_query_formula_smoke, metric_ks)
        print(f"[main]   overall: {overall_metrics[f'k{k}']}", flush=True)
        print(f"[main]   formula bucket overall: {formula_bucket_metrics_overall[f'k{k}']}", flush=True)
        if smoke50:
            print(f"[main]   smoke50: {smoke_metrics[f'k{k}']}", flush=True)
            print(f"[main]   formula bucket smoke50: {formula_bucket_metrics_smoke[f'k{k}']}", flush=True)

    report = {
        "model": args.model_path,
        "mode": "B_formula_routing_rrf",
        "n_formula_passages": len(formula_ids),
        "n_queries": len(query_ids),
        "n_smoke50": len(smoke50),
        "rrf_ks": rrf_ks,
        "overall": overall_metrics,
        "formula_bucket_overall": formula_bucket_metrics_overall,
        "smoke50": smoke_metrics,
        "formula_bucket_smoke50": formula_bucket_metrics_smoke,
        "baselines_for_compare": {
            "dense_qwen3_4B_R@10": 0.6195,
            "graph_explicit_only_R@10": 0.6913,
            "formula_bucket_ceiling_R@10": 0.5600,
        },
    }
    (args.output_dir / "eval_report.json").write_text(json.dumps(report, indent=2))
    print(f"[main] report -> {args.output_dir / 'eval_report.json'}", flush=True)


if __name__ == "__main__":
    main()
