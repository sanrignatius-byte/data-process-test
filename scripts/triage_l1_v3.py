#!/usr/bin/env python3
"""Triage L1 v3 queries into A/B/C buckets with lightweight heuristics."""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Tuple

VISUAL_WORDS = {
    "red", "blue", "green", "orange", "purple", "black", "gray", "grey",
    "left", "right", "upper", "lower", "top", "bottom", "center", "middle",
    "curve", "line", "bar", "point", "scatter", "cluster", "peak", "slope",
    "trend", "distribution", "outlier", "correlation", "axis", "histogram",
    "arrow", "dashed", "solid", "region", "quadrant", "node", "edge",
}
TEXTY_VISUAL_WORDS = {"word", "text", "caption", "label", "token", "appearing", "near"}
CAUSAL_WORDS = {"because", "since", "due to", "therefore", "thus", "as a result", "resulting"}


def tokenize(s: str) -> List[str]:
    return re.findall(r"[a-zA-Z][a-zA-Z0-9_-]*", s.lower())


def is_truncated(text: str) -> bool:
    if not text:
        return True
    t = text.strip()
    if len(t) < 40:
        return True
    # likely cut tails
    return bool(re.search(r"\b(thoug|becaus|therefor|compar|approa)$", t.lower()))


def triage_one(obj: Dict[str, Any]) -> Tuple[str, List[str], int]:
    reasons: List[str] = []
    score = 0

    query = obj.get("query", "")
    answer = obj.get("answer", "")
    visual_anchor = obj.get("visual_anchor", "")
    text_evidence = obj.get("text_evidence", "")

    vtoks = set(tokenize(visual_anchor))
    qtoks = set(tokenize(query))
    etoks = set(tokenize(text_evidence))

    if vtoks & VISUAL_WORDS:
        score += 2
    else:
        reasons.append("weak_visual_anchor")
        score -= 2

    if vtoks & TEXTY_VISUAL_WORDS:
        reasons.append("textual_visual_anchor")
        score -= 2

    new_info = len(etoks - qtoks)
    if new_info < 4:
        reasons.append("low_evidence_novelty")
        score -= 2

    if "why" in query.lower() and not any(w in text_evidence.lower() for w in CAUSAL_WORDS):
        reasons.append("ungrounded_why")
        score -= 3

    if is_truncated(text_evidence):
        reasons.append("truncated_evidence")
        score -= 3

    if re.search(r"\d+\.\d+", query):
        reasons.append("numeric_leakage")
        score -= 1

    if len(tokenize(answer)) < 6:
        reasons.append("short_answer")
        score -= 1

    if score >= 2:
        klass = "A"
    elif score >= -2:
        klass = "B"
    else:
        klass = "C"

    return klass, reasons, score


def main() -> None:
    parser = argparse.ArgumentParser(description="Triage L1 v3 query quality")
    parser.add_argument("--input", default="data/l1_cross_modal_queries_v3.jsonl")
    parser.add_argument("--output", default="data/l1_triage_v3.jsonl")
    parser.add_argument("--report", default="data/l1_triage_report_v3.json")
    args = parser.parse_args()

    in_path = Path(args.input)
    out_path = Path(args.output)
    report_path = Path(args.report)

    counts = Counter()
    reason_counts = Counter()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with in_path.open("r", encoding="utf-8") as fin, out_path.open("w", encoding="utf-8") as fout:
        for line in fin:
            if not line.strip():
                continue
            obj = json.loads(line)
            klass, reasons, score = triage_one(obj)
            obj["triage_class"] = klass
            obj["triage_score"] = score
            obj["triage_reasons"] = reasons
            counts[klass] += 1
            reason_counts.update(reasons)
            fout.write(json.dumps(obj, ensure_ascii=False) + "\n")

    report = {
        "input": str(in_path),
        "output": str(out_path),
        "class_counts": dict(counts),
        "reason_counts": dict(reason_counts.most_common()),
        "total": sum(counts.values()),
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
