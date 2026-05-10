#!/usr/bin/env python3
"""Analyze per-modality retrieval performance on M4query_v1.

Answers: which element types (figure/table/formula/text) are under-retrieved?
Are certain modality pairs harder than others?

Usage:
  python scripts/analyze_per_modality_retrieval.py \
    --data-dir data/03_queries/M4query_v1 \
    --ranking data/05_eval/dense_retrieval/rebuilt_20260417/ranking_v1_enriched.jsonl
"""

from __future__ import annotations

import argparse, json, sys
from collections import defaultdict, Counter
from pathlib import Path
from typing import Any


def load_jsonl(path: str) -> list[dict]:
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="data/03_queries/M4query_v1")
    parser.add_argument("--ranking", help="Existing ranking.jsonl to analyze")
    parser.add_argument("--top-k", type=int, default=10)
    args = parser.parse_args()

    data_dir = Path(args.data_dir)

    # Load data
    corpus = load_jsonl(str(data_dir / "corpus.jsonl"))
    queries = load_jsonl(str(data_dir / "queries.jsonl"))
    qrels = load_jsonl(str(data_dir / "qrels.jsonl"))

    pid2type = {p["passage_id"]: p.get("type", "unknown") for p in corpus}
    qid2query = {q["query_id"]: q for q in queries}

    # Qrel stats
    qrel_by_qid = defaultdict(list)
    for qr in qrels:
        qrel_by_qid[qr["query_id"]].append(qr)

    # === 1. Per-modality qrel distribution ===
    print("=" * 60)
    print("1. Qrel Passage Type Distribution")
    print("=" * 60)
    type_counts = Counter(pid2type.get(qr["passage_id"], "?") for qr in qrels)
    for t, c in type_counts.most_common():
        print(f"  {t:<12s}: {c:>5d}")

    # === 2. Per-query pair type analysis ===
    print()
    print("=" * 60)
    print("2. Query Pair Type Breakdown")
    print("=" * 60)
    pair_stats = defaultdict(lambda: {"count": 0, "qids": []})
    for q in queries:
        pt = q.get("pair_type", "unknown")
        pair_stats[pt]["count"] += 1
        pair_stats[pt]["qids"].append(q["query_id"])

    for pt in sorted(pair_stats.keys()):
        print(f"  {pt:<20s}: {pair_stats[pt]['count']:>4d} queries")

    # === 3. If ranking provided, analyze per-modality recall ===
    if args.ranking:
        ranking = load_jsonl(args.ranking)
        # Ranking format: {query_id, top_k: [passage_ids...]}
        rank_by_qid: dict[str, list[str]] = {}
        for r in ranking:
            qid = r["query_id"]
            top_k_list = r.get("top_k", r.get("ranking", []))
            rank_by_qid[qid] = top_k_list

        print()
        print("=" * 60)
        print(f"3. Per-Modality Recall@{args.top_k} (from ranking)")
        print("=" * 60)

        # Track recall per modality type
        mod_recall: dict[str, dict[str, int]] = defaultdict(lambda: {"hit": 0, "total": 0})
        pair_mod_recall: dict[str, dict[str, dict[str, int]]] = defaultdict(
            lambda: defaultdict(lambda: {"hit": 0, "total": 0}))

        for qid, qrs in qrel_by_qid.items():
            if qid not in rank_by_qid:
                continue
            top_k = rank_by_qid[qid][:args.top_k]
            query = qid2query.get(qid, {})
            pair_type = query.get("pair_type", "unknown")

            for qr in qrs:
                pid = qr["passage_id"]
                ptype = pid2type.get(pid, "unknown")
                mod_recall[ptype]["total"] += 1
                pair_mod_recall[pair_type][ptype]["total"] += 1
                if pid in top_k:
                    mod_recall[ptype]["hit"] += 1
                    pair_mod_recall[pair_type][ptype]["hit"] += 1

        print(f"  {'Type':<12s} {'Recall':>10s} {'Hit/Total':>12s}")
        print("  " + "-" * 36)
        for ptype in ["figure", "table", "formula", "text"]:
            s = mod_recall[ptype]
            if s["total"] == 0:
                continue
            recall = s["hit"] / s["total"]
            print(f"  {ptype:<12s} {recall:>9.4f}  {s['hit']:>4d}/{s['total']:<4d}")

        # By pair type × modality
        print()
        print(f"  Per pair_type × evidence modality:")
        print(f"  {'Pair Type':<20s} {'Modality':<10s} {'Recall@10':>10s} {'Hit/Total':>12s}")
        print("  " + "-" * 55)
        for pt in sorted(pair_mod_recall.keys()):
            for mod in ["figure", "table", "formula"]:
                s = pair_mod_recall[pt][mod]
                if s["total"] == 0:
                    continue
                recall = s["hit"] / s["total"]
                print(f"  {pt:<20s} {mod:<10s} {recall:>9.4f}  {s['hit']:>4d}/{s['total']:<4d}")

        # === 4. Per-query: which modality is the bottleneck? ===
        print()
        print("=" * 60)
        print("4. Bottleneck Analysis: which evidence gets missed?")
        print("=" * 60)

        bottleneck_stats = Counter()
        for qid, qrs in qrel_by_qid.items():
            if qid not in rank_by_qid:
                continue
            top_k = rank_by_qid[qid][:args.top_k]
            for qr in qrs:
                pid = qr["passage_id"]
                ptype = pid2type.get(pid, "unknown")
                if pid not in top_k:
                    bottleneck_stats[ptype] += 1

        total_missed = sum(bottleneck_stats.values())
        print(f"  Total missed qrel passages in top-{args.top_k}: {total_missed}")
        for ptype, count in bottleneck_stats.most_common():
            pct = count / total_missed * 100 if total_missed else 0
            print(f"  {ptype:<12s}: {count:>4d} missed ({pct:5.1f}%)")

        # === 5. Queries where BOTH qrels are missed (complete failure) ===
        print()
        print("=" * 60)
        print(f"5. Complete Failures (0/{len(qrel_by_qid.get(list(qrel_by_qid.keys())[0], []))} qrels in top-{args.top_k})")
        print("=" * 60)
        complete_fails = []
        for qid, qrs in qrel_by_qid.items():
            if qid not in rank_by_qid:
                continue
            top_k = rank_by_qid[qid][:args.top_k]
            hits = sum(1 for qr in qrs if qr["passage_id"] in top_k)
            if hits == 0 and len(qrs) > 0:
                query = qid2query.get(qid, {})
                qrel_types = [pid2type.get(qr["passage_id"], "?") for qr in qrs]
                complete_fails.append({
                    "qid": qid,
                    "pair_type": query.get("pair_type", "?"),
                    "qrel_types": qrel_types,
                })

        print(f"  Complete failures: {len(complete_fails)}/{len(qrel_by_qid)} queries")
        by_pair = Counter(cf["pair_type"] for cf in complete_fails)
        for pt, c in by_pair.most_common():
            total_pt = pair_stats.get(pt, {}).get("count", 1)
            print(f"  {pt:<20s}: {c:>4d}/{total_pt} ({c/total_pt*100:.1f}%)")

        # Show examples
        print()
        print("  Example complete failures (first 5):")
        for cf in complete_fails[:5]:
            q = qid2query.get(cf["qid"], {})
            print(f"  {cf['qid']} [{cf['pair_type']}] qrels={cf['qrel_types']}")
            qtext = q.get("query", "")[:120]
            print(f"    query: {qtext}...")

    print()
    print("=" * 60)
    print("Key findings:")
    print("=" * 60)
    print("  - Which modality is the retrieval bottleneck?")
    print("  - Do figure evidence passages get retrieved worse than table/formula?")
    print("  - Which pair types are hardest?")


if __name__ == "__main__":
    main()
