#!/usr/bin/env python3
"""Create a standardized failure-case JSONL from phase0 report."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--report", type=Path, default=Path("data/phase0_eval_report.json"))
    ap.add_argument("--method", choices=["bm25", "dense", "graph_hub_rerank"], default="graph_hub_rerank")
    ap.add_argument("--min-cases", type=int, default=20)
    ap.add_argument("--output", type=Path, default=Path("data/phase0_failure_cases.jsonl"))
    args = ap.parse_args()

    report = json.loads(args.report.read_text(encoding="utf-8"))
    details = report.get("details", {}).get(args.method, [])

    failed = [d for d in details if float(d.get("hit_at_10", 0.0)) < 1.0]
    if len(failed) < args.min_cases:
        print(f"[phase0] WARNING: only {len(failed)} failure cases found (min_cases={args.min_cases})")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as f:
        for r in failed:
            obj = {
                "query_id": r.get("query_id"),
                "failure_type": "retrieval_bias",
                "failure_description": "auto-generated: no hit in top-10 under current method",
                "fix_action": "改图结构",
                "priority": "P0",
            }
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")

    print(f"[phase0] wrote {len(failed)} failure cases to {args.output}")


if __name__ == "__main__":
    main()
