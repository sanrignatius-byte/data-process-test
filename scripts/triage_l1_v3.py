#!/usr/bin/env python3
"""
L1 v3 Triage: classify 974 queries into A (keep) / B (clean) / C (drop).

Gates (applied in order):
  1. Evidence completeness  — truncated or <50 chars → B or C
  2. Leakage                — query contains answer-like decimals → B
  3. Visual necessity       — anchor is OCR-only (no geometric/color/position) → B
  4. Ungrounded why         — why + no visual anchor at all → C
  5. Everything else        → A

Output:
  - Annotated JSONL with "triage" field (A/B/C) and "triage_reasons"
  - Summary report JSON
"""

import json
import re
import sys
from collections import Counter
from pathlib import Path

# ── Gate helpers ──────────────────────────────────────────────────────────

# Words that indicate *genuine* visual features (geometry, color, position, shape)
VISUAL_WORDS = {
    # Colors
    "red", "blue", "green", "black", "gray", "grey", "orange", "purple",
    "yellow", "white", "brown", "pink", "cyan", "magenta",
    # Line styles
    "dashed", "dotted", "solid", "thick", "thin",
    # Shapes / geometry
    "curve", "line", "bar", "circle", "rectangle", "box", "arrow",
    "node", "edge", "block", "layer", "module", "arc",
    # Spatial / positional
    "top", "bottom", "left", "right", "upper", "lower", "center",
    "above", "below", "horizontal", "vertical", "diagonal",
    "peak", "valley", "spike", "drop", "plateau", "crossover",
    "intersection", "slope", "trend", "inflection", "knee", "elbow",
    # Plot-specific
    "axis", "x-axis", "y-axis", "subplot", "panel", "legend",
    "marker", "shaded", "highlighted", "gradient", "heatmap",
    "contour", "region", "area", "gap", "overlap",
}

# Truncation signals: evidence ends mid-word or mid-sentence
TRUNCATION_PATTERNS = [
    r'\w-$',        # hyphenated break at end
    r'\w{2,}\.$',   # not a truncation (normal sentence end) — skip
    r'[,;:]\s*$',   # ends with comma/semicolon
    r'\s\w{1,3}$',  # ends with a very short dangling word
]


def has_visual_words(text: str) -> int:
    """Count genuine visual-feature words in text."""
    words = set(re.findall(r'\b[a-z]+\b', text.lower()))
    return len(words & VISUAL_WORDS)


def is_truncated(text: str) -> bool:
    """Heuristic: does evidence look truncated?"""
    text = text.strip()
    if not text:
        return True
    # Ends with comma, semicolon, or colon (mid-sentence)
    if re.search(r'[,;:]\s*$', text):
        return True
    # Ends with a hyphenated word break
    if re.search(r'\w-$', text):
        return True
    # Very short (likely fragment)
    if len(text) < 30:
        return True
    return False


def has_leakage(query: str, answer: str) -> bool:
    """Check if query contains decimal numbers that also appear in the answer."""
    q_decimals = set(re.findall(r'\d+\.\d+', query))
    a_decimals = set(re.findall(r'\d+\.\d+', answer))
    if not q_decimals:
        return False
    # Leakage = query decimals are a subset of answer decimals
    overlap = q_decimals & a_decimals
    return len(overlap) > 0


def triage_entry(entry: dict) -> tuple[str, list[str]]:
    """
    Classify one L1 query.
    Returns (grade, [reasons]).
    """
    reasons = []
    query = entry.get("query", "")
    answer = entry.get("answer", "")
    visual_anchor = entry.get("visual_anchor", "")
    text_evidence = entry.get("text_evidence", "")
    query_lower = query.lower().strip()

    # ── Gate 1: Evidence completeness ──
    if not text_evidence or len(text_evidence) < 20:
        reasons.append("evidence_missing_or_tiny")
        return "C", reasons
    if is_truncated(text_evidence):
        reasons.append("evidence_truncated")

    # ── Gate 2: Leakage ──
    if has_leakage(query, answer):
        reasons.append("value_leakage")

    # ── Gate 3: Visual necessity ──
    anchor_visual_score = has_visual_words(visual_anchor)
    query_visual_score = has_visual_words(query)
    combined_visual = anchor_visual_score + query_visual_score

    if not visual_anchor or len(visual_anchor.strip()) < 5:
        reasons.append("no_visual_anchor")
    elif combined_visual == 0:
        reasons.append("ocr_only_anchor")

    # ── Gate 4: Ungrounded why ──
    if query_lower.startswith("why") and combined_visual == 0:
        reasons.append("ungrounded_why")

    # ── Classify ──
    if "evidence_missing_or_tiny" in reasons:
        return "C", reasons
    if "ungrounded_why" in reasons and "no_visual_anchor" in reasons:
        return "C", reasons

    # B: has issues but salvageable
    if reasons:
        return "B", reasons

    # A: clean
    return "A", reasons


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Triage L1 v3 queries into A/B/C")
    parser.add_argument("--input", default="data/l1_cross_modal_queries_v3.jsonl",
                        help="Input JSONL")
    parser.add_argument("--output", default="data/l1_triage_v3.jsonl",
                        help="Output annotated JSONL")
    parser.add_argument("--report", default="data/l1_triage_report_v3.json",
                        help="Summary report JSON")
    args = parser.parse_args()

    entries = []
    with open(args.input) as f:
        for line in f:
            line = line.strip()
            if line:
                entries.append(json.loads(line))

    grade_counts = Counter()
    reason_counts = Counter()
    grade_by_type = {"A": Counter(), "B": Counter(), "C": Counter()}
    results = []

    for entry in entries:
        grade, reasons = triage_entry(entry)
        grade_counts[grade] += 1
        for r in reasons:
            reason_counts[r] += 1
        grade_by_type[grade][entry.get("query_type", "unknown")] += 1

        entry["triage"] = grade
        entry["triage_reasons"] = reasons
        results.append(entry)

    # Write annotated JSONL
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        for entry in results:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    total = len(entries)
    report = {
        "total": total,
        "grade_distribution": {
            "A": {"count": grade_counts["A"],
                  "pct": f"{grade_counts['A']/total*100:.1f}%"},
            "B": {"count": grade_counts["B"],
                  "pct": f"{grade_counts['B']/total*100:.1f}%"},
            "C": {"count": grade_counts["C"],
                  "pct": f"{grade_counts['C']/total*100:.1f}%"},
        },
        "reason_counts": dict(reason_counts.most_common()),
        "grade_A_by_query_type": dict(grade_by_type["A"].most_common()),
        "grade_B_by_query_type": dict(grade_by_type["B"].most_common()),
        "grade_C_by_query_type": dict(grade_by_type["C"].most_common()),
    }

    with open(args.report, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    # Print summary
    print(f"{'='*50}")
    print(f"L1 v3 Triage Report")
    print(f"{'='*50}")
    print(f"Total: {total}")
    print()
    for g in ["A", "B", "C"]:
        print(f"  Grade {g}: {grade_counts[g]:>4}  ({grade_counts[g]/total*100:.1f}%)")
    print()
    print("Reason breakdown:")
    for reason, count in reason_counts.most_common():
        print(f"  {reason:<30} {count:>4}  ({count/total*100:.1f}%)")
    print()
    print(f"Output: {args.output}")
    print(f"Report: {args.report}")


if __name__ == "__main__":
    main()
