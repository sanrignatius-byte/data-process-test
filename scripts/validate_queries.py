#!/usr/bin/env python3
"""
Validate and audit L1 cross-modal queries.

Checks:
1. Schema validation (types, required fields)
2. Visual anchor specificity (not generic)
3. Cross-modal constraint (blindfold test heuristics)
4. Banned pattern detection
5. Distribution reporting
"""

import json
import re
import sys
from collections import Counter
from pathlib import Path


# Visual anchor vocabulary — specific elements you'd need to SEE
VISUAL_ANCHOR_WORDS = {
    # Lines & curves
    "line", "curve", "dashed", "dotted", "solid", "red", "blue", "green",
    "black", "gray", "orange", "purple", "yellow", "color",
    # Plot elements
    "axis", "x-axis", "y-axis", "bar", "histogram", "scatter", "peak",
    "valley", "drop", "spike", "plateau", "crossover", "intersection",
    "slope", "trend", "inflection", "knee", "elbow",
    # Values & positions
    "value", "point", "coordinate", "pixel", "cell", "row", "column",
    # Diagram elements
    "node", "arrow", "box", "rectangle", "circle", "edge", "module",
    "layer", "block", "connection",
    # Regions
    "region", "area", "shaded", "highlighted", "subplot", "panel",
    "top", "bottom", "left", "right", "upper", "lower",
    # Data
    "legend", "label", "marker", "error", "whisker", "outlier",
    "heatmap", "gradient", "contour",
}

# Banned query prefixes
BANNED_PREFIXES = [
    "according to the text",
    "as mentioned in the text",
    "as described in the text",
    "as stated in the text",
    "how does figure",
    "how does the figure",
    "what does figure",
    "what does the figure",
    "what is shown in figure",
    "what is depicted in figure",
]

# Generic evidence phrases that indicate weak visual grounding
GENERIC_EVIDENCE = [
    "visual content",
    "the figure shows",
    "the figure displays",
    "the figure depicts",
    "as shown in the figure",
    "the figure illustrates",
    "figure content",
    "visual representation",
]


def count_visual_anchors(text: str) -> int:
    """Count visual anchor words in text."""
    words = set(re.findall(r'\b\w+\b', text.lower()))
    return len(words & VISUAL_ANCHOR_WORDS)


def has_specific_value(text: str) -> bool:
    """Check if text contains a specific numeric value or coordinate."""
    return bool(re.search(r'\d+\.?\d*', text))


def check_banned_prefix(query: str) -> str | None:
    """Return matched banned prefix or None."""
    q = query.lower().strip()
    for bp in BANNED_PREFIXES:
        if q.startswith(bp):
            return bp
    return None


def check_generic_evidence(evidence: str) -> bool:
    """Check if evidence is generic/weak."""
    ev_lower = evidence.lower()
    return any(g in ev_lower for g in GENERIC_EVIDENCE) or len(evidence) < 10


def validate_entry(entry: dict, idx: int) -> list[dict]:
    """Validate a single query entry. Returns list of issues found."""
    issues = []

    # Schema checks
    for field in ["query", "answer", "doc_id", "figure_id", "image_path"]:
        if not entry.get(field):
            issues.append({"severity": "error", "type": "missing_field", "field": field})

    # New schema fields
    visual_anchor = entry.get("visual_anchor", "")
    text_evidence = entry.get("text_evidence", "")

    # Legacy schema fields (v1)
    if "evidence_from_figure" in entry and "visual_anchor" not in entry:
        visual_anchor = entry.get("evidence_from_figure", "")
        text_evidence = entry.get("evidence_from_text", "")

    # Type checks
    for bool_field in ["requires_figure", "requires_text"]:
        val = entry.get(bool_field)
        if val is not None and not isinstance(val, bool):
            issues.append({
                "severity": "error", "type": "type_error",
                "field": bool_field, "got": type(val).__name__,
            })

    # Banned prefix
    query = entry.get("query", "")
    banned = check_banned_prefix(query)
    if banned:
        issues.append({
            "severity": "warning", "type": "banned_prefix",
            "matched": banned,
        })

    # Visual anchor quality
    anchor_count = count_visual_anchors(query)
    if anchor_count == 0:
        # Also check visual_anchor field
        if count_visual_anchors(visual_anchor) == 0:
            issues.append({
                "severity": "warning", "type": "no_visual_anchor",
                "query": query[:80],
            })

    # Visual anchor specificity
    if visual_anchor and check_generic_evidence(visual_anchor):
        issues.append({
            "severity": "warning", "type": "generic_visual_evidence",
            "evidence": visual_anchor[:80],
        })

    # Text evidence quality
    if text_evidence and len(text_evidence) < 15:
        issues.append({
            "severity": "warning", "type": "weak_text_evidence",
            "evidence": text_evidence,
        })

    # Pseudo cross-modal: query starts with text-referencing patterns
    q_lower = query.lower()
    text_first_patterns = [
        "the text states", "the text mentions", "the text describes",
        "the text explains", "the paper states", "the paper describes",
        "the caption states", "the caption mentions",
    ]
    if any(p in q_lower for p in text_first_patterns):
        issues.append({
            "severity": "info", "type": "text_first_pattern",
            "query": query[:80],
        })

    # Check for "why" without visual anchor
    if q_lower.strip().startswith("why") and anchor_count == 0:
        issues.append({
            "severity": "warning", "type": "ungrounded_why",
            "query": query[:80],
        })

    return issues


def validate_file(path: str, verbose: bool = False) -> dict:
    """Validate entire JSONL file and return report."""
    entries = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                entries.append(json.loads(line))

    total = len(entries)
    all_issues = []
    entries_with_issues = 0
    severity_counts = Counter()
    type_counts = Counter()

    # Distribution tracking
    figure_types = Counter()
    query_types = Counter()
    query_lengths = []
    answer_lengths = []
    visual_anchor_counts = []

    for i, entry in enumerate(entries):
        issues = validate_entry(entry, i)
        if issues:
            entries_with_issues += 1
        for issue in issues:
            severity_counts[issue["severity"]] += 1
            type_counts[issue["type"]] += 1
            if verbose:
                all_issues.append({"index": i, **issue})

        # Stats
        figure_types[entry.get("figure_type", "unknown")] += 1
        query_types[entry.get("query_type", "unknown")] += 1
        query_lengths.append(len(entry.get("query", "").split()))
        answer_lengths.append(len(entry.get("answer", "").split()))
        visual_anchor_counts.append(count_visual_anchors(entry.get("query", "")))

    # Compute stats
    avg_q = sum(query_lengths) / max(len(query_lengths), 1)
    avg_a = sum(answer_lengths) / max(len(answer_lengths), 1)
    anchored = sum(1 for c in visual_anchor_counts if c > 0)
    has_value = sum(1 for e in entries if has_specific_value(e.get("query", "")))

    report = {
        "total_queries": total,
        "entries_with_issues": entries_with_issues,
        "clean_entries": total - entries_with_issues,
        "clean_rate": f"{(total - entries_with_issues) / max(total, 1) * 100:.1f}%",
        "severity_counts": dict(severity_counts),
        "issue_type_counts": dict(type_counts.most_common()),
        "avg_query_length_words": round(avg_q, 1),
        "avg_answer_length_words": round(avg_a, 1),
        "queries_with_visual_anchors": f"{anchored}/{total} ({anchored/max(total,1)*100:.1f}%)",
        "queries_with_specific_values": f"{has_value}/{total} ({has_value/max(total,1)*100:.1f}%)",
        "figure_type_distribution": dict(figure_types.most_common()),
        "query_type_distribution": dict(query_types.most_common()),
    }

    if verbose:
        report["issues"] = all_issues

    return report


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Validate L1 cross-modal queries")
    parser.add_argument("input", help="Path to JSONL file")
    parser.add_argument("--verbose", "-v", action="store_true", help="Show all issues")
    parser.add_argument("--output", "-o", help="Save report to JSON file")
    args = parser.parse_args()

    report = validate_file(args.input, verbose=args.verbose)

    print(json.dumps(report, indent=2, ensure_ascii=False))

    if args.output:
        with open(args.output, 'w') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        print(f"\nReport saved to {args.output}")


if __name__ == "__main__":
    main()
