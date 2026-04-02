#!/usr/bin/env python3
"""
Run lexical retrieval baselines on dual-evidence triplet datasets.

Experiments:
1) Local ranking (positive + provided negatives per triplet)
2) Global ranking over all unique positive bundles

Metrics:
- Accuracy@1
- MRR
- Recall@k (configurable)
"""

from __future__ import annotations

import argparse
import json
import math
import random
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Tuple, Any

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.utils.text_utils import tokenize_for_retrieval as tokenize
from src.retrieval import BM25Lite

# TOKEN_RE, tokenize(), BM25Lite moved to src.utils.text_utils / src.retrieval


def jaccard_score(query_tokens: List[str], doc_tokens: List[str]) -> float:
    q = set(query_tokens)
    d = set(doc_tokens)
    if not q or not d:
        return 0.0
    return len(q & d) / len(q | d)


def reciprocal_rank(ranked_ids: List[str], target_id: str) -> float:
    for i, rid in enumerate(ranked_ids, start=1):
        if rid == target_id:
            return 1.0 / i
    return 0.0


def recall_at_k(ranked_ids: List[str], target_id: str, k: int) -> float:
    return 1.0 if target_id in ranked_ids[:k] else 0.0


def load_triplets(path: Path) -> List[Dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def pick_text(obj: Dict[str, Any], text_field: str) -> str:
    txt = str(obj.get(text_field, "") or "").strip()
    if txt:
        return txt
    return str(obj.get("text", "") or "")


def run_local_eval(
    triplets: List[Dict[str, Any]],
    bm25: BM25Lite,
    text_to_doc_idx: Dict[str, int],
    recall_ks: List[int],
    seed: int,
    text_field: str,
) -> Dict[str, Any]:
    rng = random.Random(seed)
    methods = ["random", "jaccard", "bm25"]
    stats = {
        m: {
            "n": 0,
            "acc_at_1": 0.0,
            "mrr": 0.0,
            **{f"recall_at_{k}": 0.0 for k in recall_ks},
        }
        for m in methods
    }

    for t in triplets:
        query = str(t.get("query", ""))
        q_toks = tokenize(query)

        pos = t.get("positive", {})
        pos_text = pick_text(pos, text_field)
        pos_id = "POS"

        cands = [{"cid": pos_id, "text": pos_text}]
        for i, n in enumerate(t.get("negatives", []) or []):
            cands.append({"cid": f"NEG{i}", "text": pick_text(n, text_field)})

        # random
        cids = [c["cid"] for c in cands]
        rng.shuffle(cids)
        ranked_random = cids

        # jaccard
        scored_j = []
        for c in cands:
            s = jaccard_score(q_toks, tokenize(c["text"]))
            scored_j.append((c["cid"], s))
        ranked_j = [cid for cid, _ in sorted(scored_j, key=lambda x: x[1], reverse=True)]

        # bm25
        scored_b = []
        for c in cands:
            didx = text_to_doc_idx.get(c["text"])
            s = bm25.score(q_toks, didx) if didx is not None else -1e9
            scored_b.append((c["cid"], s))
        ranked_b = [cid for cid, _ in sorted(scored_b, key=lambda x: x[1], reverse=True)]

        rankings = {"random": ranked_random, "jaccard": ranked_j, "bm25": ranked_b}
        for m, ranked in rankings.items():
            stats[m]["n"] += 1
            stats[m]["acc_at_1"] += 1.0 if ranked and ranked[0] == pos_id else 0.0
            stats[m]["mrr"] += reciprocal_rank(ranked, pos_id)
            for k in recall_ks:
                stats[m][f"recall_at_{k}"] += recall_at_k(ranked, pos_id, k)

    for m in methods:
        n = max(1, stats[m]["n"])
        stats[m]["acc_at_1"] = round(stats[m]["acc_at_1"] / n, 4)
        stats[m]["mrr"] = round(stats[m]["mrr"] / n, 4)
        for k in recall_ks:
            key = f"recall_at_{k}"
            stats[m][key] = round(stats[m][key] / n, 4)
    return stats


def run_global_eval(
    triplets: List[Dict[str, Any]],
    bm25: BM25Lite,
    text_to_doc_idx: Dict[str, int],
    recall_ks: List[int],
    text_field: str,
) -> Dict[str, Any]:
    # build unique positive corpus
    positive_key_to_text: Dict[str, str] = {}
    for t in triplets:
        pmd = t.get("positive", {}).get("metadata", {}) or {}
        key = f"{pmd.get('doc_id','')}::{pmd.get('pair_id','')}"
        text = pick_text(t.get("positive", {}) or {}, text_field)
        if key and text:
            positive_key_to_text[key] = text

    pos_keys = list(positive_key_to_text.keys())
    pos_doc_indices = {}
    for k, txt in positive_key_to_text.items():
        pos_doc_indices[k] = text_to_doc_idx.get(txt)

    metrics = {
        "n_queries": 0,
        "acc_at_1": 0.0,
        "mrr": 0.0,
        **{f"recall_at_{k}": 0.0 for k in recall_ks},
    }

    for t in triplets:
        query = str(t.get("query", ""))
        q_toks = tokenize(query)
        pmd = t.get("positive", {}).get("metadata", {}) or {}
        gold_key = f"{pmd.get('doc_id','')}::{pmd.get('pair_id','')}"
        gold_idx = pos_doc_indices.get(gold_key)
        if gold_idx is None:
            continue

        scored = []
        for key in pos_keys:
            didx = pos_doc_indices.get(key)
            if didx is None:
                continue
            s = bm25.score(q_toks, didx)
            scored.append((key, s))
        ranked = [k for k, _ in sorted(scored, key=lambda x: x[1], reverse=True)]

        metrics["n_queries"] += 1
        metrics["acc_at_1"] += 1.0 if ranked and ranked[0] == gold_key else 0.0
        metrics["mrr"] += reciprocal_rank(ranked, gold_key)
        for k in recall_ks:
            metrics[f"recall_at_{k}"] += recall_at_k(ranked, gold_key, k)

    n = max(1, metrics["n_queries"])
    metrics["acc_at_1"] = round(metrics["acc_at_1"] / n, 4)
    metrics["mrr"] = round(metrics["mrr"] / n, 4)
    for k in recall_ks:
        key = f"recall_at_{k}"
        metrics[key] = round(metrics[key] / n, 4)
    metrics["unique_positive_corpus"] = len(pos_keys)
    return metrics


def main() -> None:
    ap = argparse.ArgumentParser(description="Run retrieval baselines on dual-evidence triplets")
    ap.add_argument(
        "--triplets",
        default="data/l1_dual_evidence_triplets_v1_pass.jsonl",
        help="Input triplet JSONL",
    )
    ap.add_argument(
        "--report-json",
        default="data/l1_dual_evidence_triplets_v1_pass_baseline_report.json",
        help="Output report JSON",
    )
    ap.add_argument(
        "--seed",
        type=int,
        default=42,
    )
    ap.add_argument(
        "--recall-ks",
        nargs="*",
        type=int,
        default=[1, 2, 5, 10],
    )
    ap.add_argument(
        "--text-field",
        choices=["text", "text_short"],
        default="text",
        help="Which passage field to evaluate",
    )
    args = ap.parse_args()

    triplets = load_triplets(Path(args.triplets))

    # Build global text corpus from all positive+negative texts.
    corpus_texts: List[str] = []
    for t in triplets:
        corpus_texts.append(pick_text(t.get("positive", {}) or {}, args.text_field))
        for n in t.get("negatives", []) or []:
            corpus_texts.append(pick_text(n, args.text_field))
    # Deduplicate while preserving order.
    seen = set()
    dedup_texts = []
    for txt in corpus_texts:
        if txt in seen:
            continue
        seen.add(txt)
        dedup_texts.append(txt)

    doc_tokens = [tokenize(t) for t in dedup_texts]
    bm25 = BM25Lite(doc_tokens)
    text_to_doc_idx = {txt: i for i, txt in enumerate(dedup_texts)}

    local_stats = run_local_eval(
        triplets=triplets,
        bm25=bm25,
        text_to_doc_idx=text_to_doc_idx,
        recall_ks=args.recall_ks,
        seed=args.seed,
        text_field=args.text_field,
    )

    global_stats = run_global_eval(
        triplets=triplets,
        bm25=bm25,
        text_to_doc_idx=text_to_doc_idx,
        recall_ks=args.recall_ks,
        text_field=args.text_field,
    )

    pair_type_dist = Counter(
        (t.get("positive", {}).get("metadata", {}) or {}).get("pair_type", "unknown")
        for t in triplets
    )
    query_type_dist = Counter(t.get("query_type", "unknown") for t in triplets)
    neg_type_dist = Counter()
    for t in triplets:
        for n in t.get("negatives", []) or []:
            neg_type_dist[n.get("negative_type", "unknown")] += 1

    report = {
        "input_triplets": args.triplets,
        "n_triplets": len(triplets),
        "pair_type_distribution": dict(pair_type_dist),
        "query_type_distribution": dict(query_type_dist),
        "negative_type_distribution": dict(neg_type_dist),
        "text_field": args.text_field,
        "global_corpus_size": len(dedup_texts),
        "local_ranking": local_stats,
        "global_ranking_bm25": global_stats,
    }

    out = Path(args.report_json)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print("=" * 60)
    print("Dual-Evidence Retrieval Baseline Summary")
    print("=" * 60)
    print(f"Input triplets:      {args.triplets}")
    print(f"Text field:          {args.text_field}")
    print(f"Triplets:            {len(triplets)}")
    print(f"Global corpus size:  {len(dedup_texts)}")
    print("Local ranking (Acc@1 / MRR):")
    for name, m in local_stats.items():
        print(f"  {name:8s}  Acc@1={m['acc_at_1']:.4f}  MRR={m['mrr']:.4f}")
    print("Global BM25 (Acc@1 / MRR):")
    print(f"  Acc@1={global_stats['acc_at_1']:.4f}  MRR={global_stats['mrr']:.4f}")
    print(f"Report: {out}")
    print("=" * 60)


if __name__ == "__main__":
    main()
