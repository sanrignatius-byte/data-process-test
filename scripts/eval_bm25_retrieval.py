#!/usr/bin/env python3
"""
eval_bm25_retrieval.py
======================

Lightweight BM25 retrieval evaluation for JSONL query / corpus / qrels files.
The interface mirrors `eval_dense_retrieval.py` so chunk corpora can be
evaluated under the same query / qrels remapping setup.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.retrieval import BM25Lite, coverage_at_k, ndcg_at_k, reciprocal_rank_binary
from src.utils.text_utils import tokenize_for_retrieval


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8-sig") as f:
        for ln, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as e:
                print(f"[warn] {path}:{ln} bad json: {e}", file=sys.stderr)
    return rows


def extract_query_text(row: Dict[str, Any]) -> str:
    for key in ("query", "question", "text"):
        val = row.get(key)
        if isinstance(val, str) and val.strip():
            return val.strip()
    sq = row.get("standard_query") or {}
    if isinstance(sq, dict):
        for key in ("query", "question", "text"):
            val = sq.get(key)
            if isinstance(val, str) and val.strip():
                return val.strip()
    raise KeyError(f"no query text in row, keys={list(row.keys())[:10]}")


def extract_query_id(row: Dict[str, Any]) -> str:
    for key in ("query_id", "qid", "id"):
        if key in row:
            return str(row[key])
    raise KeyError(f"no query_id in row, keys={list(row.keys())[:10]}")


def extract_passage_text(row: Dict[str, Any]) -> str:
    for key in ("text", "content", "passage", "enriched_content"):
        val = row.get(key)
        if isinstance(val, str) and val.strip():
            return val.strip()
    return ""


def extract_passage_id(row: Dict[str, Any]) -> str:
    for key in ("passage_id", "pid", "id", "element_id"):
        if key in row:
            return str(row[key])
    raise KeyError(f"no passage_id in row, keys={list(row.keys())[:10]}")


def compute_metrics(
    ranking: Dict[str, List[str]],
    qrels: Dict[str, Dict[str, int]],
    ks: tuple[int, ...] = (1, 2, 5, 10, 100),
) -> Dict[str, float]:
    out: Dict[str, float] = {}
    valid_qids = [qid for qid in ranking if qrels.get(qid)]
    if not valid_qids:
        return out

    mrr = 0.0
    for k in ks:
        out[f"recall@{k}"] = 0.0
        out[f"ndcg@{k}"] = 0.0

    for qid in valid_qids:
        ranked = ranking[qid]
        relevant = {pid for pid, rel in qrels[qid].items() if rel > 0}
        mrr += reciprocal_rank_binary(ranked, relevant)
        for k in ks:
            out[f"recall@{k}"] += coverage_at_k(ranked, relevant, k)
            out[f"ndcg@{k}"] += ndcg_at_k(ranked, relevant, k)

    n = float(len(valid_qids))
    out["num_queries_evaluated"] = n
    out["mrr"] = mrr / n
    for k in ks:
        out[f"recall@{k}"] /= n
        out[f"ndcg@{k}"] /= n
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="BM25 retrieval baseline for JSONL corpora")
    ap.add_argument("--data-dir", required=True, help="Directory containing default queries/corpus/qrels")
    ap.add_argument("--queries", default=None, help="Override queries path")
    ap.add_argument("--corpus", default=None, help="Override corpus path")
    ap.add_argument("--qrels", default=None, help="Override qrels path")
    ap.add_argument("--output", default="eval_report.json")
    ap.add_argument("--save-ranking", default=None, help="Optional ranking.jsonl output path")
    ap.add_argument("--top-k", type=int, default=100)
    ap.add_argument("--limit-queries", type=int, default=0)
    args = ap.parse_args()

    data_dir = Path(args.data_dir)
    queries_path = Path(args.queries) if args.queries else data_dir / "queries.jsonl"
    corpus_path = Path(args.corpus) if args.corpus else data_dir / "corpus.jsonl"
    qrels_path = Path(args.qrels) if args.qrels else data_dir / "qrels.jsonl"

    for p in (queries_path, corpus_path):
        if not p.exists():
            print(f"[fatal] missing file: {p}", file=sys.stderr)
            sys.exit(1)
    has_qrels = qrels_path.exists()

    t0 = time.time()
    queries_raw = load_jsonl(queries_path)
    corpus_raw = load_jsonl(corpus_path)
    qrels_raw = load_jsonl(qrels_path) if has_qrels else []
    print(
        f"[info] loaded {len(queries_raw)} queries / {len(corpus_raw)} passages / "
        f"{len(qrels_raw)} qrels in {time.time() - t0:.1f}s"
    )

    if args.limit_queries:
        queries_raw = queries_raw[: args.limit_queries]
        print(f"[info] limit-queries -> {len(queries_raw)}")

    query_ids: List[str] = []
    query_texts: List[str] = []
    for row in queries_raw:
        try:
            query_ids.append(extract_query_id(row))
            query_texts.append(extract_query_text(row))
        except KeyError as e:
            print(f"[warn] skip query: {e}", file=sys.stderr)

    passage_ids: List[str] = []
    passage_texts: List[str] = []
    for row in corpus_raw:
        try:
            passage_ids.append(extract_passage_id(row))
            passage_texts.append(extract_passage_text(row))
        except KeyError as e:
            print(f"[warn] skip passage: {e}", file=sys.stderr)

    qrels: Dict[str, Dict[str, int]] = defaultdict(dict)
    for row in qrels_raw:
        qid = str(row.get("query_id") or row.get("qid") or "")
        pid = str(row.get("passage_id") or row.get("pid") or "")
        if not qid or not pid:
            continue
        try:
            rel = int(row.get("relevance", 1))
        except (TypeError, ValueError):
            rel = 1
        qrels[qid][pid] = rel

    covered_q = sum(1 for qid in query_ids if qrels.get(qid))
    print(f"[info] {covered_q}/{len(query_ids)} queries have qrels")
    if covered_q == 0 and has_qrels and len(qrels_raw) > 0:
        print("[fatal] no qrels overlap any query_id -- check schema", file=sys.stderr)
        sys.exit(1)

    print("[info] tokenizing corpus ...")
    corpus_tokens = [tokenize_for_retrieval(text) for text in passage_texts]
    print("[info] building BM25 index ...")
    bm25 = BM25Lite(corpus_tokens)

    ranking: Dict[str, List[str]] = {}
    ranking_scores: Dict[str, List[float]] = {}
    top_k = min(int(args.top_k), len(passage_ids))

    print("[info] scoring queries ...")
    for qid, qtext in zip(query_ids, query_texts):
        q_tokens = tokenize_for_retrieval(qtext)
        scored = [(idx, bm25.score(q_tokens, idx)) for idx in range(len(passage_ids))]
        scored.sort(key=lambda x: x[1], reverse=True)
        top = scored[:top_k]
        ranking[qid] = [passage_ids[idx] for idx, _ in top]
        ranking_scores[qid] = [float(score) for _, score in top]

    metrics = compute_metrics(ranking, qrels, ks=(1, 2, 5, 10, 100))
    metrics.update({
        "num_queries_total": float(len(query_ids)),
        "num_queries_with_qrels": float(covered_q),
        "num_passages": float(len(passage_ids)),
        "model": "BM25Lite",
        "top_k": float(args.top_k),
    })

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)

    if args.save_ranking:
        rp = Path(args.save_ranking)
        rp.parent.mkdir(parents=True, exist_ok=True)
        with open(rp, "w", encoding="utf-8") as f:
            for qid in query_ids:
                f.write(json.dumps({
                    "query_id": qid,
                    "top_k": ranking.get(qid, []),
                    "scores": ranking_scores.get(qid, []),
                }) + "\n")
        print(f"[info] wrote ranking to {rp}")

    print()
    print("── Eval summary ──")
    for k in (1, 2, 5, 10, 100):
        print(f"  Recall@{k:<3}  = {metrics.get(f'recall@{k}', 0.0):.4f}")
    for k in (1, 2, 5, 10, 100):
        print(f"  NDCG@{k:<3}    = {metrics.get(f'ndcg@{k}', 0.0):.4f}")
    print(f"  MRR         = {metrics.get('mrr', 0.0):.4f}")
    print(f"  Covered     = {int(covered_q)}/{len(query_ids)}")
    print(f"\n[info] wrote {out_path}")


if __name__ == "__main__":
    main()
