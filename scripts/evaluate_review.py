#!/usr/bin/env python3
"""
Evaluate a manually-annotated citation_review.csv and suggest a confidence threshold.

Review_Label convention:
    1   = accept  (match is correct)
    0   = reject  (match is wrong / dubious)
    ?   = borderline (treated conservatively as reject)

Usage:
    python scripts/evaluate_review.py
    python scripts/evaluate_review.py --input data/citation_review.csv
"""

import argparse
import csv
from pathlib import Path
from collections import defaultdict


def main(csv_path: str) -> None:
    p = Path(csv_path)
    if not p.exists():
        raise FileNotFoundError(f"Review CSV not found: {p.resolve()}")

    rows = []
    with open(p, encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)

    if not rows:
        print("CSV is empty.")
        return

    # Filter to labelled rows only
    valid = []
    skipped = 0
    for row in rows:
        label = row.get("Review_Label", "").strip()
        if label in ("1", "0", "?"):
            valid.append(row)
        else:
            skipped += 1

    print(f"Total rows: {len(rows)}")
    print(f"Labelled:   {len(valid)}  (skipped unlabelled: {skipped})")

    if not valid:
        print("\nNo labelled rows found. Fill Review_Label (1/0/?) and re-run.")
        return

    # Aggregate stats
    accepts     = sum(1 for r in valid if r["Review_Label"].strip() == "1")
    rejects     = sum(1 for r in valid if r["Review_Label"].strip() in ("0", "?"))
    reject_rate = rejects / len(valid)

    print(f"\nAccept:      {accepts} ({accepts/len(valid):.1%})")
    print(f"Reject+?:    {rejects} ({reject_rate:.1%})")

    # Per-method breakdown
    method_stats: dict = defaultdict(lambda: {"accept": 0, "reject": 0})
    for row in valid:
        method = row.get("Match_Method", "unknown").strip()
        label  = row["Review_Label"].strip()
        if label == "1":
            method_stats[method]["accept"] += 1
        else:
            method_stats[method]["reject"] += 1

    print("\nPer match_method breakdown:")
    for method, counts in sorted(method_stats.items()):
        total = counts["accept"] + counts["reject"]
        acc   = counts["accept"]
        print(
            f"  {method:<25}  accept={acc}/{total} "
            f"({acc/total:.1%})"
        )

    # Per-confidence-bucket breakdown
    buckets = {
        "high (<0.65)":    [],
        "mid  (0.65-0.75)": [],
        "low  (>=0.75)":   [],
    }
    for row in valid:
        try:
            conf = float(row.get("Confidence", 0.0))
        except ValueError:
            continue
        label = row["Review_Label"].strip()
        is_ok = 1 if label == "1" else 0
        if conf < 0.65:
            buckets["high (<0.65)"].append(is_ok)
        elif conf < 0.75:
            buckets["mid  (0.65-0.75)"].append(is_ok)
        else:
            buckets["low  (>=0.75)"].append(is_ok)

    print("\nPer confidence bucket:")
    for bname, vals in buckets.items():
        if not vals:
            print(f"  {bname}: no samples")
            continue
        acc_rate = sum(vals) / len(vals)
        print(f"  {bname}: n={len(vals):3d}  accept={acc_rate:.1%}")

    # Threshold recommendation
    print("\n" + "=" * 50)
    print("THRESHOLD RECOMMENDATION")
    print("=" * 50)

    if reject_rate > 0.30:
        rec = 0.70
        reason = f"High reject rate ({reject_rate:.0%}) → raise threshold to 0.70"
    elif reject_rate > 0.20:
        rec = 0.65
        reason = f"Moderate reject rate ({reject_rate:.0%}) → keep threshold at 0.65"
    elif reject_rate > 0.10:
        rec = 0.60
        reason = f"Low reject rate ({reject_rate:.0%}) → can lower threshold to 0.60"
    else:
        rec = 0.55
        reason = f"Very low reject rate ({reject_rate:.0%}) → current 0.55 is safe"

    print(f"  Recommended threshold: {rec}")
    print(f"  Reason: {reason}")
    print()
    print("Next steps:")
    print(f"  • Set JACCARD_THRESHOLD = {rec} in build_citation_graph.py")
    print("  • Re-run build_citation_graph.py to rebuild with new threshold")
    print("  • Consider upgrading to BM25/trigram candidate generation")
    print("    before scaling beyond the current 73-paper corpus")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(
        description="Evaluate annotated citation review CSV and suggest threshold"
    )
    ap.add_argument(
        "--input", default="data/citation_review.csv",
        help="Path to annotated citation_review.csv",
    )
    args = ap.parse_args()
    main(args.input)
