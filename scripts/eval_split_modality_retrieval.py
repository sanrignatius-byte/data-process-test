#!/usr/bin/env python3
"""Split-modality dense retrieval: separate FAISS index per element type.

Hypothesis: text embeddings crowd different modalities into overlapping regions.
By building separate indices per type (figure/table/formula/text) and merging
with proportional top-K allocation, we give each modality fair representation.

Compares:
  A. Baseline — single mixed index (all passages together)
  B. Split-index — separate indices, proportional merge

Usage:
  python scripts/eval_split_modality_retrieval.py \
    --data-dir data/03_queries/M4query_v1 \
    --model-path models/Qwen3-Embedding-4B \
    --output-dir data/05_eval/dense_retrieval/split_modality
"""

from __future__ import annotations

import argparse, json, math, sys, time
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
from sentence_transformers import SentenceTransformer


def load_jsonl(path: Path) -> list[dict]:
    rows = []
    with open(path, encoding="utf-8-sig") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def encode_batch(model, texts: list[str], batch_size: int = 32, desc: str = "") -> np.ndarray:
    """Encode texts to normalized embeddings."""
    embs = model.encode(
        texts, batch_size=batch_size, show_progress_bar=True,
        normalize_embeddings=True,
    )
    return np.array(embs)


def build_faiss_index(embeddings: np.ndarray) -> Any:
    """Build flat IP (inner product) index for normalized embeddings."""
    import faiss
    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)  # inner product = cosine for normalized vecs
    index.add(embeddings.astype(np.float32))
    return index


def search_index(index, query_emb: np.ndarray, top_k: int) -> tuple[np.ndarray, np.ndarray]:
    """Search index, return (scores, indices)."""
    return index.search(query_emb.astype(np.float32), top_k)


def evaluate(qrels_by_qid: dict, ranking_by_qid: dict, ks: list[int] = [1, 5, 10, 100]):
    """Compute Recall@k and MRR."""
    metrics = {}
    for k in ks:
        recall_sum = 0.0
        for qid, qrels in qrels_by_qid.items():
            if qid not in ranking_by_qid:
                continue
            top_k = set(ranking_by_qid[qid][:k])
            hits = sum(1 for qr in qrels if qr["passage_id"] in top_k)
            recall_sum += hits / len(qrels) if qrels else 0
        metrics[f"recall@{k}"] = recall_sum / max(len(qrels_by_qid), 1)

    # MRR
    mrr_sum = 0.0
    for qid, qrels in qrels_by_qid.items():
        if qid not in ranking_by_qid:
            continue
        qrel_pids = {qr["passage_id"] for qr in qrels}
        for rank, pid in enumerate(ranking_by_qid[qid], 1):
            if pid in qrel_pids:
                mrr_sum += 1.0 / rank
                break
    metrics["mrr"] = mrr_sum / max(len(qrels_by_qid), 1)
    metrics["num_queries"] = len(qrels_by_qid)
    return metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="data/03_queries/M4query_v1")
    parser.add_argument("--model-path", default="models/Qwen3-Embedding-4B")
    parser.add_argument("--output-dir", default="data/05_eval/dense_retrieval/split_modality")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--top-k", type=int, default=100)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--limit-queries", type=int, default=0)
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    print("Loading data...")
    corpus = load_jsonl(data_dir / "corpus.jsonl")
    queries = load_jsonl(data_dir / "queries.jsonl")
    qrels = load_jsonl(data_dir / "qrels.jsonl")

    if args.limit_queries > 0:
        queries = queries[:args.limit_queries]
        qids_keep = {q["query_id"] for q in queries}
        qrels = [qr for qr in qrels if qr["query_id"] in qids_keep]

    qrels_by_qid = defaultdict(list)
    for qr in qrels:
        qrels_by_qid[qr["query_id"]].append(qr)

    print(f"  Corpus: {len(corpus)} passages")
    print(f"  Queries: {len(queries)}")
    print(f"  Qrels: {len(qrels)}")

    # Split corpus by type
    type_passages: dict[str, list[dict]] = defaultdict(list)
    for p in corpus:
        ptype = p.get("type", "text")
        type_passages[ptype].append(p)

    print("\nCorpus by type:")
    for t, plist in sorted(type_passages.items()):
        print(f"  {t:<10s}: {len(plist):>5d} passages")

    # Load model
    print(f"\nLoading model: {args.model_path}")
    model = SentenceTransformer(args.model_path, device=args.device)
    model.max_seq_length = getattr(args, "max_length", 512)
    print(f"  max_seq_length: {model.max_seq_length}")

    # Encode queries
    print("\nEncoding queries...")
    query_texts = []
    for q in queries:
        qt = q.get("query", q.get("question", q.get("text", "")))
        query_texts.append(qt)
    query_embs = encode_batch(model, query_texts, args.batch_size, "queries")

    # =========================================================
    # A. BASELINE: single mixed index
    # =========================================================
    print("\n" + "=" * 60)
    print("A. BASELINE — Single Mixed Index")
    print("=" * 60)

    all_texts = [p.get("text", "") for p in corpus]
    all_ids = [p["passage_id"] for p in corpus]
    all_embs = encode_batch(model, all_texts, args.batch_size, "corpus")
    mixed_index = build_faiss_index(all_embs)

    t0 = time.time()
    scores, indices = search_index(mixed_index, query_embs, args.top_k)
    elapsed = time.time() - t0

    baseline_ranking = {}
    for qi, q in enumerate(queries):
        qid = q["query_id"]
        ranked = [all_ids[idx] for idx in indices[qi] if idx >= 0]
        baseline_ranking[qid] = ranked

    baseline_metrics = evaluate(qrels_by_qid, baseline_ranking)
    print(f"  Search: {elapsed:.2f}s")
    for k, v in baseline_metrics.items():
        if k.startswith("recall") or k == "mrr":
            print(f"  {k}: {v:.4f}")

    # Save baseline ranking
    with open(out_dir / "ranking_baseline.jsonl", "w") as f:
        for q in queries:
            qid = q["query_id"]
            f.write(json.dumps({"query_id": qid, "top_k": baseline_ranking.get(qid, [])}) + "\n")

    # =========================================================
    # B. SPLIT-INDEX: separate index per type + merge
    # =========================================================
    print("\n" + "=" * 60)
    print("B. SPLIT-INDEX — Per-Modality Indices")
    print("=" * 60)

    type_indices: dict[str, tuple[Any, list[str]]] = {}
    for ptype, plist in type_passages.items():
        texts = [p.get("text", "") for p in plist]
        ids = [p["passage_id"] for p in plist]
        embs = encode_batch(model, texts, args.batch_size, f"corpus/{ptype}")
        idx = build_faiss_index(embs)
        type_indices[ptype] = (idx, ids)
        print(f"  {ptype}: {len(ids)} passages, dim={embs.shape[1]}")

    # Merge strategies to test
    merge_configs = [
        # (name, allocation dict: type -> fraction of top_k)
        ("equal_split", {t: 1.0 / len(type_indices) for t in type_indices}),
        ("prop_to_corpus", {t: len(pl) / len(corpus) for t, pl in type_passages.items()}),
        # Heuristic: give more slots to formula (the bottleneck)
        ("boost_formula", {"figure": 0.30, "table": 0.25, "formula": 0.35, "text": 0.10}),
        # Heuristic: give more slots to figure (most common qrel type)
        ("boost_figure", {"figure": 0.40, "table": 0.25, "formula": 0.25, "text": 0.10}),
    ]

    all_merge_results = {}

    for merge_name, allocation in merge_configs:
        print(f"\n  Merge: {merge_name}")
        print(f"    Allocation: {allocation}")

        ranking = {}
        for qi, q in enumerate(queries):
            qid = q["query_id"]
            q_emb = query_embs[qi:qi + 1]

            merged: list[str] = []
            seen: set[str] = set()

            # Search each type index with allocated budget
            for ptype, (idx, ids) in type_indices.items():
                budget = max(1, int(args.top_k * allocation.get(ptype, 0.25)))
                _, idxs = search_index(idx, q_emb, budget)
                for i in idxs[0]:
                    if i >= 0 and i < len(ids):
                        pid = ids[i]
                        if pid not in seen:
                            seen.add(pid)
                            merged.append(pid)

            # If we didn't fill top_k, add more from all types
            if len(merged) < args.top_k:
                for ptype, (idx, ids) in type_indices.items():
                    if len(merged) >= args.top_k:
                        break
                    _, idxs = search_index(idx, q_emb, args.top_k)
                    for i in idxs[0]:
                        if i >= 0 and i < len(ids):
                            pid = ids[i]
                            if pid not in seen:
                                seen.add(pid)
                                merged.append(pid)
                                if len(merged) >= args.top_k:
                                    break

            ranking[qid] = merged[:args.top_k]

        merge_metrics = evaluate(qrels_by_qid, ranking)
        all_merge_results[merge_name] = merge_metrics

        for k, v in merge_metrics.items():
            if k.startswith("recall") or k == "mrr":
                delta = v - baseline_metrics.get(k, 0)
                print(f"    {k}: {v:.4f}  (Δ={delta:+.4f})")

        # Save ranking
        with open(out_dir / f"ranking_{merge_name}.jsonl", "w") as f:
            for q in queries:
                qid = q["query_id"]
                f.write(json.dumps({"query_id": qid, "top_k": ranking.get(qid, [])}) + "\n")

    # =========================================================
    # C. Per-modality recall analysis for best merge
    # =========================================================
    print("\n" + "=" * 60)
    print("C. Per-Modality Recall Comparison")
    print("=" * 60)

    best_name = max(all_merge_results, key=lambda n: all_merge_results[n].get("mrr", 0))
    best_ranking = {}
    with open(out_dir / f"ranking_{best_name}.jsonl") as f:
        for line in f:
            r = json.loads(line)
            best_ranking[r["query_id"]] = r["top_k"]

    pid2type = {p["passage_id"]: p.get("type", "?") for p in corpus}

    print(f"  Best merge: {best_name}")
    print(f"  {'Type':<10s} {'Baseline R@10':>15s} {'Split R@10':>15s} {'Delta':>10s}")
    print("  " + "-" * 52)

    for ptype in ["figure", "table", "formula"]:
        base_hit, base_total = 0, 0
        split_hit, split_total = 0, 0
        for qid, qrs in qrels_by_qid.items():
            base_top = baseline_ranking.get(qid, [])[:10]
            split_top = best_ranking.get(qid, [])[:10]
            for qr in qrs:
                if pid2type.get(qr["passage_id"]) == ptype:
                    base_total += 1
                    split_total += 1
                    if qr["passage_id"] in base_top:
                        base_hit += 1
                    if qr["passage_id"] in split_top:
                        split_hit += 1
        base_r = base_hit / max(base_total, 1)
        split_r = split_hit / max(split_total, 1)
        delta = split_r - base_r
        print(f"  {ptype:<10s} {base_r:>15.4f} {split_r:>15.4f} {delta:>+10.4f}")

    # =========================================================
    # D. Save summary
    # =========================================================
    report = {
        "baseline": baseline_metrics,
        "splits": all_merge_results,
        "best_merge": best_name,
        "config": {
            "model": args.model_path,
            "data_dir": str(data_dir),
            "top_k": args.top_k,
            "num_queries": len(queries),
            "num_passages": len(corpus),
        },
    }
    with open(out_dir / "eval_report.json", "w") as f:
        json.dump(report, f, indent=2, default=str)

    print(f"\nResults saved to {out_dir}")
    print(f"  Baseline MRR: {baseline_metrics['mrr']:.4f}  R@10: {baseline_metrics['recall@10']:.4f}")
    best = all_merge_results[best_name]
    print(f"  Best ({best_name}): MRR: {best['mrr']:.4f}  R@10: {best['recall@10']:.4f}")


if __name__ == "__main__":
    main()
