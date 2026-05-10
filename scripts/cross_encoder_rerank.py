#!/usr/bin/env python3
"""
cross_encoder_rerank.py
=======================
Cross-encoder rerank on dense top-K, with optional graph-static-prior fusion.

Triggered by: refine-logs/CEILING_DECISION_20260503.md (rule R2).

Workflow:
  1. Load dense ranking (e.g. ranking_v1_enriched.jsonl with top_k pids per query).
     If --slice-from-full and the input has > rerank_topn pids, slice to top-N.
  2. Load corpus (pid -> text) and queries (qid -> text).
  3. For each query, score (query, candidate) pairs with a HuggingFace
     CrossEncoder. Truncate candidates to --max-length tokens.
  4. Sort candidates by CE score descending. Optionally fuse with a
     pre-computed graph static prior (linear combination after min-max
     normalization on the rerank slice).
  5. Output:
       - reranked ranking jsonl (same schema as dense ranking)
       - metrics json (R@K / NDCG@K / MRR for K in {1,5,10,100})
"""
from __future__ import annotations

import argparse
import inspect
import json
import math
import statistics
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Sequence, Tuple


# ---------- IO helpers ----------

def load_jsonl(path: Path) -> List[dict]:
    out: List[dict] = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def load_corpus(path: Path) -> Dict[str, str]:
    out: Dict[str, str] = {}
    with path.open(encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            pid = r.get("passage_id") or r.get("pid") or r.get("id")
            text = r.get("text", "")
            out[pid] = text
    return out


def load_queries(path: Path) -> Dict[str, str]:
    out: Dict[str, str] = {}
    with path.open(encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            qid = r.get("query_id") or r.get("qid") or r.get("id")
            text = r.get("text") or r.get("query") or ""
            out[qid] = text
    return out


def load_qrels(path: Path) -> Dict[str, set]:
    out: Dict[str, set] = defaultdict(set)
    with path.open(encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            if r.get("relevance", 1) >= 1:
                out[r["query_id"]].add(r["passage_id"])
    return out


# ---------- Metrics ----------

def compute_metrics(rankings: Dict[str, List[str]], qrels: Dict[str, set],
                    ks: Sequence[int] = (1, 5, 10, 100)) -> dict:
    recall_at = {k: [] for k in ks}
    ndcg_at = {k: [] for k in ks}
    hit_at = {k: [] for k in ks}
    rrs: List[float] = []
    n_eval = 0
    for qid, gold in qrels.items():
        if not gold:
            continue
        ranked = rankings.get(qid, [])
        n_eval += 1
        rr = 0.0
        for i, pid in enumerate(ranked, start=1):
            if pid in gold:
                rr = 1.0 / i
                break
        rrs.append(rr)
        for k in ks:
            topk = ranked[:k]
            hits = sum(1 for p in topk if p in gold)
            recall_at[k].append(hits / len(gold))
            hit_at[k].append(1.0 if hits > 0 else 0.0)
            dcg = sum((1.0 / math.log2(i + 2)) for i, p in enumerate(topk) if p in gold)
            idcg = sum(1.0 / math.log2(i + 2) for i in range(min(len(gold), k)))
            ndcg_at[k].append(dcg / idcg if idcg > 0 else 0.0)

    def avg(xs: List[float]) -> float:
        return sum(xs) / len(xs) if xs else 0.0

    out = {"num_queries": n_eval, "mrr": avg(rrs)}
    for k in ks:
        out[f"recall@{k}"] = avg(recall_at[k])
        out[f"ndcg@{k}"] = avg(ndcg_at[k])
        out[f"hit@{k}"] = avg(hit_at[k])
    return out


# ---------- Reranking ----------

def rerank_one(model, query: str, candidates: List[Tuple[str, str]],
               batch_size: int, max_length: int) -> List[Tuple[str, float]]:
    pairs = [(query, text or "") for _, text in candidates]
    scores = model.predict(
        pairs,
        batch_size=batch_size,
        show_progress_bar=False,
        convert_to_numpy=True,
    )
    pid_score = list(zip([pid for pid, _ in candidates], [float(s) for s in scores]))
    pid_score.sort(key=lambda x: x[1], reverse=True)
    return pid_score


def minmax_normalize(scores: Dict[str, float]) -> Dict[str, float]:
    if not scores:
        return scores
    vs = list(scores.values())
    lo, hi = min(vs), max(vs)
    if hi - lo < 1e-12:
        return {k: 0.0 for k in scores}
    return {k: (v - lo) / (hi - lo) for k, v in scores.items()}


# ---------- Main ----------

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ranking-in", required=True, type=Path)
    ap.add_argument("--corpus", required=True, type=Path)
    ap.add_argument("--queries", required=True, type=Path)
    ap.add_argument("--qrels", required=True, type=Path)
    ap.add_argument("--model", default="BAAI/bge-reranker-v2-m3")
    ap.add_argument("--rerank-topn", type=int, default=500,
                    help="Number of dense candidates to rerank per query.")
    ap.add_argument("--max-length", type=int, default=2048,
                    help="Max tokens per (query, passage) pair (passage truncated). "
                         "Corpus p99 is ~128 tokens; 2048 covers all but ~1.8% outliers.")
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--limit-queries", type=int, default=0,
                    help="Optional dry-run limit on the number of query rows.")
    ap.add_argument("--prompt", default="",
                    help="Optional SentenceTransformers CrossEncoder prompt/instruction. "
                         "Useful for Qwen3 rerankers whose default prompt targets web search.")
    ap.add_argument("--fp16", action="store_true", default=True,
                    help="Use fp16 inference (default True on cuda).")
    ap.add_argument("--ranking-out", required=True, type=Path)
    ap.add_argument("--metrics-out", required=True, type=Path)
    ap.add_argument("--scores-out", default=None, type=Path,
                    help="Optional jsonl dump of reranker scores for the rerank slice.")
    ap.add_argument("--graph-static-prior", default=None, type=Path,
                    help="Optional jsonl/json: pid -> static prior float "
                         "(for fusion experiments).")
    ap.add_argument("--fuse-weights", default="",
                    help="Comma-separated linear weights for graph prior fusion, "
                         "e.g. '0.0,0.1,0.2,0.3'. If non-empty, emits one metrics "
                         "block per weight under metrics_out.")
    args = ap.parse_args()

    print(f"[info] loading queries from {args.queries}", flush=True)
    queries = load_queries(args.queries)
    print(f"[info] loading corpus from {args.corpus}", flush=True)
    corpus = load_corpus(args.corpus)
    print(f"[info] loading qrels from {args.qrels}", flush=True)
    qrels = load_qrels(args.qrels)
    print(f"[info] loading dense ranking from {args.ranking_in}", flush=True)
    dense = load_jsonl(args.ranking_in)
    if args.limit_queries and args.limit_queries > 0:
        dense = dense[: args.limit_queries]
        print(f"[info] dry-run limit: using first {len(dense)} dense rows", flush=True)

    print(f"[info] queries={len(queries)} corpus={len(corpus)} qrels-queries={len(qrels)} "
          f"dense-rows={len(dense)}", flush=True)

    print(f"[info] loading CrossEncoder {args.model} (device={args.device}, "
          f"max_length={args.max_length})", flush=True)
    from sentence_transformers import CrossEncoder
    ce_kwargs = {"max_length": args.max_length, "device": args.device}
    if args.prompt:
        sig = inspect.signature(CrossEncoder.__init__)
        if "prompts" in sig.parameters and "default_prompt_name" in sig.parameters:
            ce_kwargs["prompts"] = {"dpt_scientific_evidence": args.prompt}
            ce_kwargs["default_prompt_name"] = "dpt_scientific_evidence"
            print("[info] using custom CrossEncoder prompt", flush=True)
        else:
            print("[warn] this sentence-transformers CrossEncoder does not expose "
                  "prompts/default_prompt_name; continuing without custom prompt",
                  flush=True)
    model = CrossEncoder(args.model, **ce_kwargs)
    try:
        model.model.to(args.device)
    except Exception:
        pass
    if args.fp16 and args.device.startswith("cuda"):
        try:
            model.model.half()
            print("[info] cast model to fp16", flush=True)
        except Exception as e:
            print(f"[warn] fp16 cast failed: {e}", flush=True)

    reranked: Dict[str, List[str]] = {}
    ce_scores: Dict[str, Dict[str, float]] = {}

    t0 = time.time()
    for i, row in enumerate(dense):
        qid = row.get("query_id")
        pids = (row.get("top_k") or row.get("ranking") or [])[: args.rerank_topn]
        q_text = queries.get(qid, "")
        if not q_text or not pids:
            reranked[qid] = pids
            continue
        cands = [(pid, corpus.get(pid, "")) for pid in pids]
        ranked = rerank_one(model, q_text, cands,
                            batch_size=args.batch_size,
                            max_length=args.max_length)
        reranked[qid] = [pid for pid, _ in ranked]
        ce_scores[qid] = dict(ranked)
        if (i + 1) % 25 == 0:
            elapsed = time.time() - t0
            print(f"[info] {i+1}/{len(dense)} queries reranked, elapsed={elapsed:.1f}s",
                  flush=True)

    print(f"[info] cross-encoder reranking done in {time.time() - t0:.1f}s", flush=True)

    # Write reranked ranking jsonl (top_k = up to args.rerank_topn)
    args.ranking_out.parent.mkdir(parents=True, exist_ok=True)
    with args.ranking_out.open("w", encoding="utf-8") as f:
        for qid, pids in reranked.items():
            f.write(json.dumps({"query_id": qid, "top_k": pids}) + "\n")
    print(f"[info] wrote reranked ranking -> {args.ranking_out}", flush=True)

    if args.scores_out:
        args.scores_out.parent.mkdir(parents=True, exist_ok=True)
        with args.scores_out.open("w", encoding="utf-8") as f:
            for qid, scores in ce_scores.items():
                ranked_scores = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
                f.write(json.dumps({
                    "query_id": qid,
                    "scores": [{"passage_id": pid, "score": score}
                               for pid, score in ranked_scores],
                }) + "\n")
        print(f"[info] wrote scores -> {args.scores_out}", flush=True)

    # Compute headline metrics
    metrics_blocks: Dict[str, dict] = {}
    metrics_blocks["dense_baseline"] = compute_metrics(
        {r["query_id"]: r.get("top_k") or [] for r in dense}, qrels)
    metrics_blocks["cross_encoder"] = compute_metrics(reranked, qrels)

    # Optional fusion
    if args.graph_static_prior and args.fuse_weights:
        prior_raw = json.loads(args.graph_static_prior.read_text())
        # Expect dict pid -> float
        prior: Dict[str, float] = {k: float(v) for k, v in prior_raw.items()}
        for w_str in args.fuse_weights.split(","):
            w_str = w_str.strip()
            if not w_str:
                continue
            w = float(w_str)
            fused: Dict[str, List[str]] = {}
            for qid, pids in reranked.items():
                ce = ce_scores.get(qid, {})
                ce_norm = minmax_normalize(ce)
                # Min-max prior over the SAME slice
                slice_prior = {pid: prior.get(pid, 0.0) for pid in pids}
                p_norm = minmax_normalize(slice_prior)
                combined = {pid: ce_norm.get(pid, 0.0) + w * p_norm.get(pid, 0.0)
                            for pid in pids}
                fused[qid] = sorted(pids, key=lambda p: combined[p], reverse=True)
            metrics_blocks[f"ce_plus_graph_w{w_str}"] = compute_metrics(fused, qrels)

    args.metrics_out.parent.mkdir(parents=True, exist_ok=True)
    args.metrics_out.write_text(json.dumps({
        "config": {
            "model": args.model,
            "rerank_topn": args.rerank_topn,
            "max_length": args.max_length,
            "batch_size": args.batch_size,
            "ranking_in": str(args.ranking_in),
        },
        "metrics": metrics_blocks,
    }, indent=2))
    print(f"[info] wrote metrics -> {args.metrics_out}", flush=True)

    # Print summary
    print("\n── Summary (R@10 / R@100 / MRR) ──")
    for name, m in metrics_blocks.items():
        print(f"  {name:30s}  R@10={m['recall@10']:.4f}  "
              f"R@100={m['recall@100']:.4f}  MRR={m['mrr']:.4f}")


if __name__ == "__main__":
    main()
