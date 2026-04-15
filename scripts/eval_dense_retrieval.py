#!/usr/bin/env python3
"""
eval_dense_retrieval.py — Qwen3-Embedding-0.6B baseline on M4query_v1
======================================================================
Self-contained dense-retrieval baseline. Given the three JSONL files that
ship in M4query_v1 (queries / corpus / qrels), it encodes everything with
Qwen3-Embedding-0.6B, runs top-K cosine retrieval, and prints
Recall@{1,5,10,100}, NDCG@{1,5,10,100}, and MRR.

Expected files in --data-dir:
    queries.jsonl    each line: {"query_id": ..., "query": ...}
                     (script also accepts top-level "question"/"text"
                      or nested `standard_query.query`)
    corpus.jsonl     each line: {"passage_id": ..., "text": ..., ...}
    qrels.jsonl      each line: {"query_id": ..., "passage_id": ...,
                                 "relevance": 1}

Dependencies (CPU or single GPU):
    pip install torch sentence-transformers>=3.0 numpy
    # model download path 1: via modelscope (default on server)
    pip install modelscope
    # model download path 2: pre-download and pass --model-path

Usage:
    # A. let modelscope fetch the weights at runtime
    python eval_dense_retrieval.py --data-dir ./M4query_v1

    # B. use a pre-downloaded model dir (offline)
    python eval_dense_retrieval.py \\
        --data-dir ./M4query_v1 \\
        --model-path /path/to/Qwen3-Embedding-0.6B

    # C. debug on 10 queries
    python eval_dense_retrieval.py --data-dir ./M4query_v1 --limit-queries 10

Output:
    eval_report.json  (metrics + config; next to --output or CWD)
    ranking.jsonl     (optional, --save-ranking)
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np


# ──────────────────────────────────────────────────────────
# IO helpers
# ──────────────────────────────────────────────────────────
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


# ──────────────────────────────────────────────────────────
# Model loading (Qwen3-Embedding-0.6B)
# ──────────────────────────────────────────────────────────
def resolve_model_path(
    model_name: str,
    model_path: str | None,
    download_root: str | None,
) -> str:
    """Return a local directory path to the embedding model.

    Priority:
      1. --model-path (offline / pre-downloaded)
      2. modelscope snapshot_download of --model-name
    """
    if model_path:
        p = Path(model_path).expanduser()
        if not p.is_dir():
            print(f"[fatal] --model-path {p} is not a directory", file=sys.stderr)
            sys.exit(2)
        return str(p)

    try:
        from modelscope import snapshot_download
    except ImportError:
        print(
            "[fatal] modelscope not installed and no --model-path provided.\n"
            "        pip install modelscope\n"
            "        or pass --model-path /path/to/Qwen3-Embedding-0.6B",
            file=sys.stderr,
        )
        sys.exit(2)

    kwargs: Dict[str, Any] = {}
    if download_root:
        kwargs["cache_dir"] = download_root
    print(f"[info] downloading {model_name} via modelscope ...", flush=True)
    return snapshot_download(model_name, **kwargs)


def load_embedding_model(model_dir: str, device: str):
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError:
        print(
            "[fatal] sentence-transformers not installed.\n"
            "        pip install 'sentence-transformers>=3.0'",
            file=sys.stderr,
        )
        sys.exit(2)
    print(f"[info] loading SentenceTransformer from {model_dir}", flush=True)
    model = SentenceTransformer(model_dir, device=device, trust_remote_code=True)
    return model


# ──────────────────────────────────────────────────────────
# Encoding
# ──────────────────────────────────────────────────────────
def encode_texts(
    model,
    texts: List[str],
    *,
    batch_size: int,
    is_query: bool,
    max_length: int,
) -> np.ndarray:
    """Encode a list of strings into L2-normalized embeddings.

    For Qwen3-Embedding we pass prompt_name="query" on the query side so
    sentence-transformers prepends the official retrieval instruction.
    """
    try:
        model.max_seq_length = max_length
    except Exception:
        pass

    kwargs: Dict[str, Any] = {
        "batch_size": batch_size,
        "convert_to_numpy": True,
        "normalize_embeddings": True,
        "show_progress_bar": True,
    }
    if is_query:
        # Qwen3-Embedding ships a "query" prompt in its modules config.
        # If the loaded model does not have it, fall back to raw encode.
        try:
            return model.encode(texts, prompt_name="query", **kwargs)
        except (ValueError, KeyError):
            print("[warn] model has no 'query' prompt, encoding raw", file=sys.stderr)
    return model.encode(texts, **kwargs)


# ──────────────────────────────────────────────────────────
# Metrics
# ──────────────────────────────────────────────────────────
def compute_metrics(
    ranking: Dict[str, List[str]],
    qrels: Dict[str, Dict[str, int]],
    ks: Tuple[int, ...] = (1, 5, 10, 100),
) -> Dict[str, float]:
    recalls: Dict[int, List[float]] = {k: [] for k in ks}
    ndcgs: Dict[int, List[float]] = {k: [] for k in ks}
    mrrs: List[float] = []

    for qid, pids in ranking.items():
        gold = qrels.get(qid) or {}
        gold_set = {pid for pid, rel in gold.items() if rel > 0}
        if not gold_set:
            continue

        rr = 0.0
        for i, pid in enumerate(pids):
            if pid in gold_set:
                rr = 1.0 / (i + 1)
                break
        mrrs.append(rr)

        for k in ks:
            hits = len(set(pids[:k]) & gold_set)
            recalls[k].append(hits / len(gold_set))

            dcg = 0.0
            for i, pid in enumerate(pids[:k]):
                if pid in gold_set:
                    dcg += 1.0 / math.log2(i + 2)
            ideal = sum(1.0 / math.log2(i + 2) for i in range(min(k, len(gold_set))))
            ndcgs[k].append(dcg / ideal if ideal > 0 else 0.0)

    out: Dict[str, float] = {
        "num_queries_evaluated": float(len(mrrs)),
        "mrr": float(np.mean(mrrs)) if mrrs else 0.0,
    }
    for k in ks:
        out[f"recall@{k}"] = float(np.mean(recalls[k])) if recalls[k] else 0.0
        out[f"ndcg@{k}"] = float(np.mean(ndcgs[k])) if ndcgs[k] else 0.0
    return out


# ──────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────
def main() -> None:
    ap = argparse.ArgumentParser(
        description="Dense retrieval baseline with Qwen3-Embedding-0.6B",
    )
    ap.add_argument("--data-dir", required=True,
                    help="Directory containing queries.jsonl / corpus.jsonl / qrels.jsonl")
    ap.add_argument("--queries", default=None, help="Override queries path")
    ap.add_argument("--corpus", default=None, help="Override corpus path")
    ap.add_argument("--qrels", default=None, help="Override qrels path")
    ap.add_argument("--model-name", default="Qwen/Qwen3-Embedding-0.6B",
                    help="ModelScope model id (default: Qwen/Qwen3-Embedding-0.6B)")
    ap.add_argument("--model-path", default=None,
                    help="Local pre-downloaded model dir (offline mode)")
    ap.add_argument("--download-root", default=None,
                    help="ModelScope cache dir (if downloading)")
    ap.add_argument("--output", default="eval_report.json")
    ap.add_argument("--save-ranking", default=None,
                    help="Optional path to write per-query top-K ranking as JSONL")
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--max-length", type=int, default=512)
    ap.add_argument("--top-k", type=int, default=100)
    ap.add_argument("--device", default=None,
                    help="'cuda' / 'cpu' / None (auto)")
    ap.add_argument("--limit-queries", type=int, default=0,
                    help="Debug: run only N queries")
    args = ap.parse_args()

    data_dir = Path(args.data_dir)
    queries_path = Path(args.queries) if args.queries else data_dir / "queries.jsonl"
    corpus_path = Path(args.corpus) if args.corpus else data_dir / "corpus.jsonl"
    qrels_path = Path(args.qrels) if args.qrels else data_dir / "qrels.jsonl"

    for p in (queries_path, corpus_path, qrels_path):
        if not p.exists():
            print(f"[fatal] missing file: {p}", file=sys.stderr)
            sys.exit(1)

    # ── Load data ──
    t0 = time.time()
    queries_raw = load_jsonl(queries_path)
    corpus_raw = load_jsonl(corpus_path)
    qrels_raw = load_jsonl(qrels_path)
    print(
        f"[info] loaded {len(queries_raw)} queries / "
        f"{len(corpus_raw)} passages / {len(qrels_raw)} qrels "
        f"in {time.time() - t0:.1f}s"
    )

    if args.limit_queries:
        queries_raw = queries_raw[: args.limit_queries]
        print(f"[info] limit-queries → {len(queries_raw)}")

    # Build tables
    query_ids: List[str] = []
    query_texts: List[str] = []
    for row in queries_raw:
        try:
            qid = extract_query_id(row)
            qtext = extract_query_text(row)
        except KeyError as e:
            print(f"[warn] skip query: {e}", file=sys.stderr)
            continue
        query_ids.append(qid)
        query_texts.append(qtext)

    passage_ids: List[str] = []
    passage_texts: List[str] = []
    for row in corpus_raw:
        try:
            pid = extract_passage_id(row)
        except KeyError as e:
            print(f"[warn] skip passage: {e}", file=sys.stderr)
            continue
        ptext = extract_passage_text(row)
        passage_ids.append(pid)
        passage_texts.append(ptext)

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
    if covered_q == 0:
        print("[fatal] no qrels overlap any query_id — check schema", file=sys.stderr)
        sys.exit(1)

    # ── Load model ──
    try:
        import torch
    except ImportError:
        print("[fatal] torch not installed", file=sys.stderr)
        sys.exit(2)
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[info] device = {device}")

    model_dir = resolve_model_path(args.model_name, args.model_path, args.download_root)
    model = load_embedding_model(model_dir, device)

    # ── Encode ──
    print("[info] encoding corpus ...")
    t0 = time.time()
    corpus_emb = encode_texts(
        model, passage_texts,
        batch_size=args.batch_size,
        is_query=False,
        max_length=args.max_length,
    )
    print(f"[info] corpus encoded in {time.time() - t0:.1f}s, shape={corpus_emb.shape}")

    print("[info] encoding queries ...")
    t0 = time.time()
    query_emb = encode_texts(
        model, query_texts,
        batch_size=args.batch_size,
        is_query=True,
        max_length=args.max_length,
    )
    print(f"[info] queries encoded in {time.time() - t0:.1f}s, shape={query_emb.shape}")

    # ── Retrieval (cosine via dot product, embeddings are L2-normalized) ──
    print("[info] scoring ...")
    t0 = time.time()
    scores = query_emb @ corpus_emb.T  # [Q, P]
    top_k = int(min(args.top_k, scores.shape[1]))
    topk_idx = np.argpartition(-scores, top_k - 1, axis=1)[:, :top_k]
    ranking: Dict[str, List[str]] = {}
    for i, qid in enumerate(query_ids):
        row_scores = scores[i, topk_idx[i]]
        order = np.argsort(-row_scores)
        sorted_idx = topk_idx[i][order]
        ranking[qid] = [passage_ids[j] for j in sorted_idx]
    print(f"[info] retrieval done in {time.time() - t0:.1f}s")

    # ── Metrics ──
    metrics = compute_metrics(ranking, qrels, ks=(1, 5, 10, 100))
    metrics.update({
        "num_queries_total": float(len(query_ids)),
        "num_queries_with_qrels": float(covered_q),
        "num_passages": float(len(passage_ids)),
        "model": args.model_name if not args.model_path else args.model_path,
        "top_k": float(args.top_k),
        "batch_size": float(args.batch_size),
        "max_length": float(args.max_length),
        "device": device,
    })

    # ── Save ──
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)

    if args.save_ranking:
        rp = Path(args.save_ranking)
        rp.parent.mkdir(parents=True, exist_ok=True)
        with open(rp, "w", encoding="utf-8") as f:
            for qid, pids in ranking.items():
                f.write(json.dumps({"query_id": qid, "top_k": pids}) + "\n")
        print(f"[info] wrote ranking to {rp}")

    # ── Print summary ──
    print()
    print("── Eval summary ──")
    for k in (1, 5, 10, 100):
        print(f"  Recall@{k:<3}  = {metrics[f'recall@{k}']:.4f}")
    for k in (1, 5, 10, 100):
        print(f"  NDCG@{k:<3}    = {metrics[f'ndcg@{k}']:.4f}")
    print(f"  MRR         = {metrics['mrr']:.4f}")
    print(f"  Covered     = {int(covered_q)}/{len(query_ids)}")
    print(f"\n[info] wrote {out_path}")


if __name__ == "__main__":
    main()
