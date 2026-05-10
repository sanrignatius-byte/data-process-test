#!/usr/bin/env python3
"""Per-query: count real elements actually recalled via top-K chunks.

Mentor (录音 60, 2026-05-02): "你实际把你 recalled 的唱客的 recall at 一去算个
平均值算一下，实际 recall 几个 element... recall at 十呢？"

Authoritative chunk→element mapping comes from the EVAL-TIME chunk-mapped qrels
(`chunk_corpus_*/qrels.jsonl` row format
`{query_id, passage_id=chunk_id, source_element_id, relevance}`),
NOT from the `chunk_contains_element` edges in `paragraph_chunks_*_v2.json`
(which were found to be inconsistent on 2026-05-03 — they map element_2 to a
different chunk than the eval pipeline uses).

For each query Q:
  - eval_qrels gives R_pair(Q) = list of (chunk_id, element_id) pairs that the
    eval pipeline considers relevant.
  - Take top-K retrieved chunks for Q from a chunk-retrieval `ranking.jsonl`.
  - hit = { element_id | (chunk_id, element_id) in R_pair(Q) AND chunk_id in topK }
  - Report:
      * |hit|                     = real # relevant elements surfaced
      * |hit| / |distinct(R(Q))|  = element-level recall
      * # queries with full / zero element recall

Usage:
  python scripts/analyze_chunk_query_element_recall.py \
    --eval-qrels data/05_eval/chunk_corpus_n400_partial_overlay/qrels.jsonl \
    --ranking data/05_eval/chunk_corpus_n400_partial_overlay/ranking.jsonl \
    --ks 1,2,5,10,15,20 \
    --out data/05_eval/chunk_corpus_n400_partial_overlay/chunk_query_element_recall.json
"""

from __future__ import annotations

import argparse
import json
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Set


def load_eval_qrels(eval_qrels_path: Path):
    """Read chunk-mapped qrels.

    Returns:
      query_to_relevant_pairs: query_id -> list[(chunk_id, element_id)]
      query_to_relevant_elements: query_id -> set[element_id]
    """
    pairs: Dict[str, list] = defaultdict(list)
    elems: Dict[str, Set[str]] = defaultdict(set)
    with open(eval_qrels_path) as f:
        for line in f:
            r = json.loads(line)
            if r.get("relevance", 0) <= 0:
                continue
            qid = r["query_id"]
            cid = r["passage_id"]
            eid = r.get("source_element_id")
            if eid is None:
                # legacy row: passage IS the element
                eid = cid
            pairs[qid].append((cid, eid))
            elems[qid].add(eid)
    return pairs, elems


def load_ranking(ranking_path: Path) -> Dict[str, List[str]]:
    """query_id -> [chunk_id, ...]"""
    rank: Dict[str, List[str]] = {}
    with open(ranking_path) as f:
        for line in f:
            r = json.loads(line)
            rank[r["query_id"]] = r["top_k"]
    return rank


def percentile(xs: List[float], p: float) -> float:
    if not xs:
        return 0.0
    s = sorted(xs)
    k = int(round((p / 100) * (len(s) - 1)))
    return s[k]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval-qrels", required=True, type=Path,
                    help="chunk-mapped qrels.jsonl (query_id, passage_id=chunk_id, source_element_id, relevance)")
    ap.add_argument("--ranking", required=True, type=Path,
                    help="ranking.jsonl from chunk retrieval (query_id, top_k=[chunk_id])")
    ap.add_argument("--ks", default="1,5,10",
                    help="comma-separated top-K values, default 1,5,10")
    ap.add_argument("--out", type=Path, required=True,
                    help="output JSON with summary + per-K stats")
    args = ap.parse_args()

    ks = [int(k) for k in args.ks.split(",")]

    relevant_pairs, relevant_elems = load_eval_qrels(args.eval_qrels)
    ranking = load_ranking(args.ranking)

    print(f"Loaded {len(relevant_pairs)} queries from chunk-mapped qrels, "
          f"{len(ranking)} queries with ranking")

    eval_qids = sorted(set(relevant_pairs) & set(ranking))
    print(f"Evaluating on {len(eval_qids)} queries (qrels ∩ ranking)")

    # Sanity: how many distinct elements per query?
    elem_counts = [len(relevant_elems[q]) for q in eval_qids]
    print(f"Distinct elements per query: "
          f"mean={statistics.mean(elem_counts):.2f} "
          f"median={statistics.median(elem_counts)} "
          f"min={min(elem_counts)} max={max(elem_counts)}")

    results = {
        "config": {
            "eval_qrels": str(args.eval_qrels),
            "ranking": str(args.ranking),
            "ks": ks,
            "n_eval_queries": len(eval_qids),
            "mean_distinct_elements_per_query": statistics.mean(elem_counts),
        },
        "per_k": {},
    }

    for k in ks:
        n_relevant_per_q: List[int] = []   # |distinct elements relevant to Q|
        n_recalled_per_q: List[int] = []   # # of those elements found via top-K chunks
        elem_recall_per_q: List[float] = []
        n_relevant_chunks_per_q: List[int] = []
        chunk_recall_per_q: List[float] = []  # check vs reported chunk R@K

        for qid in eval_qids:
            R_pairs = relevant_pairs[qid]
            R_elems = relevant_elems[qid]
            R_chunks = set(cid for cid, _ in R_pairs)
            top = set(ranking[qid][:k])
            hit_chunks = top & R_chunks
            hit_elems = set()
            for cid, eid in R_pairs:
                if cid in top:
                    hit_elems.add(eid)
            n_relevant_per_q.append(len(R_elems))
            n_recalled_per_q.append(len(hit_elems))
            elem_recall_per_q.append(len(hit_elems) / max(len(R_elems), 1))
            n_relevant_chunks_per_q.append(len(R_chunks))
            chunk_recall_per_q.append(1.0 if hit_chunks else 0.0)

        results["per_k"][k] = {
            "k": k,
            "mean_relevant_elements_per_query": statistics.mean(n_relevant_per_q),
            "mean_recalled_elements_per_query": statistics.mean(n_recalled_per_q),
            "mean_element_recall": statistics.mean(elem_recall_per_q),
            "median_element_recall": statistics.median(elem_recall_per_q),
            "mean_relevant_chunks_per_query": statistics.mean(n_relevant_chunks_per_q),
            "chunk_hit_rate": statistics.mean(chunk_recall_per_q),
            "n_queries_full_elem_recall": sum(1 for r in elem_recall_per_q if r >= 0.999),
            "n_queries_zero_elem_recall": sum(1 for r in elem_recall_per_q if r < 1e-9),
        }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(results, indent=2))

    # Print compact table
    print()
    print("=" * 78)
    print(" Mentor 录音60: chunk top-K → real element recall (eval-qrels grounded)")
    print("=" * 78)
    print(f"  {'K':>3s} {'avg|R_e|':>9s} {'recalled':>10s} {'elem R@K':>10s} "
          f"{'med':>6s} {'chk hit':>8s} {'full':>5s} {'zero':>5s}")
    for k in ks:
        s = results["per_k"][k]
        print(f"  {k:>3d} "
              f"{s['mean_relevant_elements_per_query']:>9.2f} "
              f"{s['mean_recalled_elements_per_query']:>10.3f} "
              f"{s['mean_element_recall']:>10.4f} "
              f"{s['median_element_recall']:>6.2f} "
              f"{s['chunk_hit_rate']:>8.4f} "
              f"{s['n_queries_full_elem_recall']:>5d} "
              f"{s['n_queries_zero_elem_recall']:>5d}")
    print()
    print("Legend:")
    print("  avg|R_e|       : avg # distinct relevant elements per query")
    print("  recalled       : avg # relevant elements actually found via top-K chunks")
    print("  elem R@K       : recalled / |R_e|, averaged over queries")
    print("  chk hit        : fraction of queries with ≥1 relevant chunk in top-K "
          "(matches reported chunk R@K)")
    print("  full / zero    : # queries with 100% / 0% element recall at this K")
    print()
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
