#!/usr/bin/env python3
"""
Postprocess dual-evidence JSONL with additional hard guardrails.

Default behavior:
1) Detect and fail high-confidence premise-answer contradictions.

Optional behavior:
2) Fail templated openings (for diversity control), e.g. "Given that", "What causes".
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, List, Tuple


def has_premise_answer_contradiction(query: str, answer: str) -> bool:
    q = (query or "").lower()
    a = (answer or "").lower()

    if "drop below" in q and re.search(r"\b(?:does|do|did)\s+not\s+drop below\b|\bnot drop below\b", a):
        return True

    if "drop more unlabeled users" in q and "fewer unlabeled users" in a:
        return True

    if "with fewer" in q and "with more" in q:
        nums = [float(n.replace(",", "")) for n in re.findall(r"\b\d+(?:\.\d+)?\b", answer or "")]
        if len(nums) >= 2 and nums[0] > nums[1]:
            return True

    for verb in ("increase", "decrease", "drop", "rise", "fall", "exceed", "improve"):
        if re.search(rf"\bwhy\s+do(?:es)?\b.*\b{verb}\b", q) and re.search(
            rf"\b(?:does|do|did)\s+not\s+{verb}\b", a
        ):
            return True

    return False


def has_templated_opening(query: str, banned_openings: Tuple[str, ...]) -> bool:
    q = (query or "").strip().lower()
    return any(q.startswith(prefix) for prefix in banned_openings)


def dedupe_keep_order(items: Iterable[str]) -> List[str]:
    out: List[str] = []
    seen = set()
    for it in items:
        if it in seen:
            continue
        seen.add(it)
        out.append(it)
    return out


def process_record(
    record: Dict,
    banned_openings: Tuple[str, ...],
    enforce_template_opening: bool,
) -> Tuple[Dict, Counter]:
    out = dict(record)
    issues = list(out.get("qc_issues") or [])
    metrics = dict(out.get("qc_metrics") or {})
    added = Counter()

    q = out.get("query", "")
    a = out.get("answer", "")

    if has_premise_answer_contradiction(q, a):
        if "premise_answer_contradiction" not in issues:
            issues.append("premise_answer_contradiction")
            added["premise_answer_contradiction"] += 1
        metrics["premise_contradiction_warn"] = True

    if enforce_template_opening and has_templated_opening(q, banned_openings):
        if "templated_opening" not in issues:
            issues.append("templated_opening")
            added["templated_opening"] += 1
        metrics["templated_opening_warn"] = True

    out["qc_issues"] = dedupe_keep_order(issues)
    out["qc_metrics"] = metrics
    out["qc_pass"] = len(out["qc_issues"]) == 0
    return out, added


def load_jsonl(path: Path) -> List[Dict]:
    rows: List[Dict] = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def dump_jsonl(path: Path, rows: List[Dict]) -> None:
    with path.open("w") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def default_outputs(input_path: Path) -> Tuple[Path, Path]:
    stem = input_path.stem
    suffix = input_path.suffix or ".jsonl"
    out_full = input_path.with_name(f"{stem}_guarded{suffix}")
    out_pass = input_path.with_name(f"{stem}_guarded_pass{suffix}")
    return out_full, out_pass


def main() -> None:
    parser = argparse.ArgumentParser(description="Postprocess dual-evidence JSONL with guardrails.")
    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="Input full JSONL (with qc_issues/qc_pass fields).",
    )
    parser.add_argument(
        "--output-full",
        type=Path,
        default=None,
        help="Output full JSONL path. Default: <input>_guarded.jsonl",
    )
    parser.add_argument(
        "--output-pass",
        type=Path,
        default=None,
        help="Output pass-only JSONL path. Default: <input>_guarded_pass.jsonl",
    )
    parser.add_argument(
        "--enforce-template-opening",
        action="store_true",
        help="Fail queries starting with templated openings.",
    )
    parser.add_argument(
        "--banned-opening",
        action="append",
        default=["given that", "what causes"],
        help="Banned opening prefix (lowercase). Can be passed multiple times.",
    )
    args = parser.parse_args()

    output_full, output_pass = default_outputs(args.input)
    if args.output_full:
        output_full = args.output_full
    if args.output_pass:
        output_pass = args.output_pass

    rows = load_jsonl(args.input)
    original_pass = sum(1 for r in rows if r.get("qc_pass") is True)
    banned_openings = tuple(x.strip().lower() for x in args.banned_opening if x.strip())

    processed: List[Dict] = []
    added_counts = Counter()
    for row in rows:
        out, added = process_record(
            row,
            banned_openings=banned_openings,
            enforce_template_opening=args.enforce_template_opening,
        )
        processed.append(out)
        added_counts.update(added)

    passed = [r for r in processed if r.get("qc_pass") is True]
    dump_jsonl(output_full, processed)
    dump_jsonl(output_pass, passed)

    print(f"input: {args.input}")
    print(f"output_full: {output_full}")
    print(f"output_pass: {output_pass}")
    print(f"counts: total={len(processed)} pass_before={original_pass} pass_after={len(passed)}")
    print(f"added_issues: {dict(added_counts)}")


if __name__ == "__main__":
    main()
