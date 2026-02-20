#!/usr/bin/env python3
"""
Re-evaluate QC for existing l1_multihop_queries_v2.jsonl with v2.1 thresholds.

v2.1 changes vs v2:
  A) MIN_OVERLAP_PER_ELEMENT:  2 → 1   (token overlap with captions is a noisy proxy)
  B) ANSWER_BALANCE_THRESHOLD: 0.25 → 0.15
  C) bridge_entity_leakage:   ≥3 tokens → ≥4 tokens
  D) cross_reading exempt from weak_reasoning_connector (it's a lookup type, not causal)

Re-uses the qc_metrics already stored in the v2 file — no API calls needed.
"""

import argparse
import json
from pathlib import Path

# ── v2.1 thresholds ──────────────────────────────────────────
ANCHOR_LEAK_THRESHOLD = 0.15          # unchanged
ANSWER_BALANCE_THRESHOLD = 0.15       # was 0.25
MIN_OVERLAP_PER_ELEMENT = 1           # was 2
BRIDGE_ENTITY_COPY_THRESHOLD = 4      # was 3

EXPLANATORY_TYPES = {                 # cross_reading removed from this set
    "trend_explanation",
    "anomaly_investigation",
    "bridge_reasoning",
    "theory_vs_experiment",
    "data_formula_consistency",
}


def reeval_record(record: dict) -> dict:
    """Return a new record with updated qc_issues and qc_pass under v2.1 rules."""
    m = record.get("qc_metrics", {})
    old_issues = record.get("qc_issues", [])
    qtype = str(record.get("query_type", "")).lower()

    new_issues = []

    # ── Keep checks that are NOT being tuned ─────────────────
    for iss in old_issues:
        if iss in ("single_element_answer", "bridge_entity_leakage", "weak_reasoning_connector"):
            continue  # will re-evaluate below
        new_issues.append(iss)

    # ── A+B: re-eval single_element_answer ───────────────────
    ov_a = m.get("answer_overlap_a", 0)
    ov_b = m.get("answer_overlap_b", 0)
    total = ov_a + ov_b
    if total == 0:
        new_issues.append("single_element_answer")
    else:
        balance = min(ov_a / total, ov_b / total)
        if (
            ov_a < MIN_OVERLAP_PER_ELEMENT
            or ov_b < MIN_OVERLAP_PER_ELEMENT
            or balance < ANSWER_BALANCE_THRESHOLD
        ):
            new_issues.append("single_element_answer")

    # ── C: re-eval bridge_entity_leakage ─────────────────────
    if m.get("anchor_token_copy_count", 0) >= BRIDGE_ENTITY_COPY_THRESHOLD:
        new_issues.append("bridge_entity_leakage")

    # ── D: re-eval weak_reasoning_connector ──────────────────
    if "weak_reasoning_connector" in old_issues:
        if qtype in EXPLANATORY_TYPES:
            new_issues.append("weak_reasoning_connector")
        # else: cross_reading and others are exempt — drop the issue

    out = dict(record)
    out["qc_issues"] = new_issues
    out["qc_pass"] = len(new_issues) == 0
    return out


def main():
    parser = argparse.ArgumentParser(description="Re-evaluate QC with v2.1 thresholds")
    parser.add_argument(
        "--input",
        default="data/l1_multihop_queries_v2.jsonl",
        help="Input v2 JSONL file",
    )
    parser.add_argument(
        "--output",
        default="data/l1_multihop_queries_v2_1.jsonl",
        help="Output re-evaluated JSONL",
    )
    parser.add_argument(
        "--pass-output",
        default="data/l1_multihop_queries_v2_1_pass.jsonl",
        help="Output JSONL containing only QC-pass records",
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)
    pass_path = Path(args.pass_output)

    records = []
    with input_path.open() as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))

    print(f"Loaded {len(records)} records from {input_path}")

    updated = [reeval_record(r) for r in records]

    v2_pass = sum(1 for r in records if r.get("qc_pass"))
    v21_pass = sum(1 for r in updated if r.get("qc_pass"))
    total = len(updated)

    print(f"\n=== QC comparison ===")
    print(f"  v2  pass: {v2_pass}/{total}  ({v2_pass/total*100:.1f}%)")
    print(f"  v2.1 pass: {v21_pass}/{total} ({v21_pass/total*100:.1f}%)")
    print(f"  Delta: +{v21_pass - v2_pass}")

    # Issue breakdown after v2.1
    from collections import Counter
    issue_counts: Counter = Counter()
    for r in updated:
        if not r.get("qc_pass"):
            for iss in r.get("qc_issues", []):
                issue_counts[iss] += 1
    fails = total - v21_pass
    print(f"\n=== v2.1 fail breakdown ({fails} fails) ===")
    for iss, cnt in issue_counts.most_common():
        print(f"  {iss}: {cnt} ({cnt/fails*100:.1f}%)")

    # Write outputs
    with output_path.open("w") as f:
        for r in updated:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"\nWrote {len(updated)} records → {output_path}")

    pass_records = [r for r in updated if r.get("qc_pass")]
    with pass_path.open("w") as f:
        for r in pass_records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"Wrote {len(pass_records)} pass records → {pass_path}")


if __name__ == "__main__":
    main()
