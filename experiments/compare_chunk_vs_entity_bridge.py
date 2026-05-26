#!/usr/bin/env python3
"""Compare chunk-bridge judge results vs entity-bridge judge results.

Produces:
1. Head-to-head verdict comparison (where same paper pairs exist in both)
2. Quality distribution comparison
3. Coverage vs precision trade-off analysis
"""

from __future__ import annotations

import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

ENTITY_DIR = ROOT / "data/05_eval/entity_bridge_judge_20260521T113000Z"
CHUNK_DIR = None  # set from CLI


def load_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            rows.append(json.loads(line))
    return rows


def compare(entity_path: Path, chunk_path: Path):
    entity_judgments = load_jsonl(entity_path / "judgments.jsonl")
    chunk_judgments = load_jsonl(chunk_path / "judgments.jsonl")

    entity_summary = json.loads((entity_path / "summary.json").read_text())
    chunk_summary = json.loads((chunk_path / "summary.json").read_text())

    # --- 1. Overall comparison ---
    print("=" * 70)
    print("CHUNK-BRIDGE vs ENTITY-BRIDGE: JUDGE COMPARISON")
    print("=" * 70)

    print(f"\n{'Metric':<35} {'Entity-Bridge':>15} {'Chunk-Bridge':>15}")
    print("-" * 65)
    print(f"{'Total judged':<35} {entity_summary['total']:>15} {chunk_summary['total']:>15}")
    print(f"{'strong_chain':<35} {entity_summary['strong_chain']:>15} {chunk_summary['strong_chain']:>15}")
    print(f"{'strong_rate':<35} {entity_summary['strong_rate']:>15.1%} {chunk_summary['strong_rate']:>15.1%}")

    ew = entity_summary.get("verdict_counts", {}).get("weak_but_related", 0)
    cw = chunk_summary.get("verdict_counts", {}).get("weak_but_related", 0)
    er = entity_summary.get('strong_rate', 0)
    cr = chunk_summary.get('strong_rate', 0)
    e_usable = er + ew / entity_summary['total']
    c_usable = cr + cw / chunk_summary['total']
    print(f"{'weak_but_related':<35} {ew:>15} {cw:>15}")
    print(f"{'usable (strong+weak)':<35} {e_usable:>15.1%} {c_usable:>15.1%}")

    # Paper coverage
    e_papers = set()
    for j in entity_judgments:
        e_papers.add(j.get("source_doc", ""))
        e_papers.add(j.get("target_doc", ""))
    c_papers = set()
    for j in chunk_judgments:
        c_papers.add(j.get("source_doc", ""))
        c_papers.add(j.get("target_doc", ""))

    print(f"\n{'Unique papers covered':<35} {len(e_papers):>15} {len(c_papers):>15}")

    # Paper pair overlap
    e_pairs = set()
    for j in entity_judgments:
        pair = tuple(sorted([j.get("source_doc", ""), j.get("target_doc", "")]))
        e_pairs.add(pair)
    c_pairs = set()
    for j in chunk_judgments:
        pair = tuple(sorted([j.get("source_doc", ""), j.get("target_doc", "")]))
        c_pairs.add(pair)

    overlap_pairs = e_pairs & c_pairs
    e_only = e_pairs - c_pairs
    c_only = c_pairs - e_pairs
    print(f"{'Unique doc pairs':<35} {len(e_pairs):>15} {len(c_pairs):>15}")
    print(f"{'Overlapping pairs':<35} {len(overlap_pairs):>15}")
    print(f"{'Entity-only pairs':<35} {len(e_only):>15}")
    print(f"{'Chunk-only pairs':<35} {len(c_only):>15}")

    # --- 2. Verdict distribution ---
    print(f"\n{'='*70}")
    print("VERDICT DISTRIBUTIONS")
    print(f"{'='*70}")
    print(f"\n{'Verdict':<30} {'Entity':>10} {'Chunk':>10}")
    print("-" * 50)
    all_verdicts = set(entity_summary.get("verdict_counts", {}).keys()) | set(chunk_summary.get("verdict_counts", {}).keys())
    for v in sorted(all_verdicts):
        ec = entity_summary.get("verdict_counts", {}).get(v, 0)
        cc = chunk_summary.get("verdict_counts", {}).get(v, 0)
        print(f"{v:<30} {ec:>10} {cc:>10}")

    # --- 3. By pair type ---
    print(f"\n{'='*70}")
    print("BY PAIR TYPE — strong_chain RATE")
    print(f"{'='*70}")
    e_by_pt = entity_summary.get("by_pair_type", {})
    c_by_pt = chunk_summary.get("by_pair_type", {})

    all_pts = set(e_by_pt.keys()) | set(c_by_pt.keys())
    print(f"\n{'Pair Type':<30} {'Entity':>10} {'Chunk':>10}")
    print("-" * 50)
    for pt in sorted(all_pts):
        e_total = sum(e_by_pt.get(pt, {}).values())
        c_total = sum(c_by_pt.get(pt, {}).values())
        e_strong = e_by_pt.get(pt, {}).get("strong_chain", 0)
        c_strong = c_by_pt.get(pt, {}).get("strong_chain", 0)
        e_rate = f"{e_strong}/{e_total} ({e_strong/e_total:.1%})" if e_total else "N/A"
        c_rate = f"{c_strong}/{c_total} ({c_strong/c_total:.1%})" if c_total else "N/A"
        print(f"{pt:<30} {e_rate:>10} {c_rate:>10}")

    # --- 4. Score bucket analysis (chunk only) ---
    if "by_score_bucket" in chunk_summary:
        print(f"\n{'='*70}")
        print("CHUNK-BRIDGE: strong_chain rate by TF-IDF similarity bucket")
        print(f"{'='*70}")
        print(f"\n{'Score bucket':<20} {'Total':>8} {'Strong':>8} {'Rate':>8}")
        print("-" * 50)
        for bucket, vc in sorted(chunk_summary["by_score_bucket"].items()):
            total = sum(vc.values())
            strong = vc.get("strong_chain", 0)
            rate = f"{strong/total:.1%}" if total else "N/A"
            print(f"{bucket:<20} {total:>8} {strong:>8} {rate:>8}")

    # --- 5. Overlap pair head-to-head ---
    if overlap_pairs:
        print(f"\n{'='*70}")
        print(f"HEAD-TO-HEAD ON {len(overlap_pairs)} OVERLAPPING PAPER PAIRS")
        print(f"{'='*70}")

        # Build lookup
        e_map: dict[tuple, list[dict]] = defaultdict(list)
        for j in entity_judgments:
            pair = tuple(sorted([j.get("source_doc", ""), j.get("target_doc", "")]))
            v = j["judgment"].get("verdict") if isinstance(j.get("judgment"), dict) else "parse_failed"
            e_map[pair].append(v)

        c_map: dict[tuple, list[dict]] = defaultdict(list)
        for j in chunk_judgments:
            pair = tuple(sorted([j.get("source_doc", ""), j.get("target_doc", "")]))
            v = j["judgment"].get("verdict") if isinstance(j.get("judgment"), dict) else "parse_failed"
            c_map[pair].append(v)

        comparison = Counter()
        for pair in overlap_pairs:
            e_best = _best_verdict(e_map[pair])
            c_best = _best_verdict(c_map[pair])
            comparison[(e_best, c_best)] += 1

        print(f"\n{'Entity \\ Chunk':<30} {'count':>8}")
        print("-" * 40)
        for (e, c), cnt in sorted(comparison.items()):
            print(f"{str(e)+' -> '+str(c):<30} {cnt:>8}")

        # Upgrade: chunk finds strong where entity found only weak
        upgrade = sum(cnt for (e, c), cnt in comparison.items()
                      if c == "strong_chain" and e != "strong_chain")
        downgrade = sum(cnt for (e, c), cnt in comparison.items()
                        if e == "strong_chain" and c != "strong_chain")
        print(f"\nUpgrades (chunk finds strong, entity didn't): {upgrade}")
        print(f"Downgrades (entity found strong, chunk didn't): {downgrade}")

    # --- 6. Key takeaway ---
    print(f"\n{'='*70}")
    print("KEY TAKEAWAY")
    print(f"{'='*70}")
    e_strong_total = entity_summary["strong_chain"]
    c_strong_total = chunk_summary["strong_chain"]
    e_paper_cov = len(e_papers)
    c_paper_cov = len(c_papers)

    print(f"""
Entity-bridge: {e_strong_total} strong chains from {entity_summary['total']} judged ({entity_summary['strong_rate']:.1%})
  → Covers {e_paper_cov} unique papers across {len(e_pairs)} doc pairs
  → Method: shared enriched-element keywords → precise but sparse coverage

Chunk-bridge:  {c_strong_total} strong chains from {chunk_summary['total']} judged ({chunk_summary['strong_rate']:.1%})
  → Covers {c_paper_cov} unique papers across {len(c_pairs)} doc pairs
  → Method: TF-IDF paragraph matching → broad coverage but noisy

Net: chunk-bridge {"improves" if c_strong_total > e_strong_total else "reduces"} strong_chain count by {abs(c_strong_total - e_strong_total)}.
     chunk-bridge covers {c_paper_cov - e_paper_cov} more papers ({c_paper_cov} vs {e_paper_cov}).
     chunk-bridge has {len(c_pairs) - len(e_pairs)} more unique doc pairs.
""")


def _best_verdict(verdicts: list[str]) -> str:
    priority = ["strong_chain", "weak_but_related", "topic_only",
                "wrong_target", "wrong_source", "insufficient_context"]
    for v in priority:
        if v in verdicts:
            return v
    return verdicts[0] if verdicts else "unknown"


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Compare chunk vs entity bridge judge results")
    parser.add_argument("--chunk-dir", required=True, help="Path to chunk bridge judge output dir")
    parser.add_argument("--entity-dir", default=str(ENTITY_DIR))
    args = parser.parse_args()

    chunk_path = Path(args.chunk_dir)
    if not chunk_path.is_absolute():
        chunk_path = ROOT / chunk_path
    entity_path = Path(args.entity_dir)
    if not entity_path.is_absolute():
        entity_path = ROOT / entity_path

    compare(entity_path, chunk_path)


if __name__ == "__main__":
    main()
