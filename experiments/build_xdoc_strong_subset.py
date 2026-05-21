#!/usr/bin/env python3
"""Build a high-precision cross-document subset from xdoc resolver judge results.

The subset is intentionally conservative:
  - promotable: verdict == strong_chain
  - exploratory: weak_but_related only, kept separately

For the v1 judge pack, this may produce an empty promotable subset. That is a
valid result and should block downstream cross-doc M4 triplet generation rather
than silently widening the filter.
"""

from __future__ import annotations

import argparse
import json
import os
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_JUDGE_DIR = ROOT / "data/05_eval/xdoc_resolver_judge_latest"
DEFAULT_OUT_PARENT = ROOT / "data/05_eval"


def utc_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def summarize(rows: list[dict[str, Any]]) -> tuple[Counter[str], dict[str, Counter[str]]]:
    verdict_counts: Counter[str] = Counter()
    by_stratum: dict[str, Counter[str]] = defaultdict(Counter)
    for row in rows:
        verdict = (row.get("judgment") or {}).get("verdict", "missing")
        stratum = row.get("target_stratum", "unknown")
        verdict_counts[verdict] += 1
        by_stratum[stratum][verdict] += 1
    return verdict_counts, by_stratum


def atomic_latest_symlink(target: Path, latest: Path) -> None:
    tmp = latest.with_name(latest.name + ".tmp")
    if tmp.exists() or tmp.is_symlink():
        tmp.unlink()
    os.symlink(target, tmp)
    os.replace(tmp, latest)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--judge-dir", type=Path, default=DEFAULT_JUDGE_DIR)
    ap.add_argument("--out-dir", type=Path, default=None)
    ap.add_argument("--min-confidence", type=float, default=0.0)
    args = ap.parse_args()

    judge_dir = args.judge_dir.resolve()
    judgments_path = judge_dir / "judgments.jsonl"
    if not judgments_path.exists():
        raise FileNotFoundError(f"Missing judgments file: {judgments_path}")

    out_dir = args.out_dir
    if out_dir is None:
        out_dir = DEFAULT_OUT_PARENT / f"xdoc_resolver_strong_subset_{utc_stamp()}"
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = read_jsonl(judgments_path)
    strong = [
        row for row in rows
        if (row.get("judgment") or {}).get("verdict") == "strong_chain"
        and float((row.get("judgment") or {}).get("confidence", 0.0)) >= args.min_confidence
    ]
    exploratory_weak = [
        row for row in rows
        if (row.get("judgment") or {}).get("verdict") == "weak_but_related"
    ]

    verdict_counts, by_stratum = summarize(rows)
    strong_by_stratum = Counter(row.get("target_stratum", "unknown") for row in strong)
    weak_by_stratum = Counter(row.get("target_stratum", "unknown") for row in exploratory_weak)

    write_jsonl(out_dir / "strong_chain_subset.jsonl", strong)
    write_jsonl(out_dir / "exploratory_weak_but_related.jsonl", exploratory_weak)

    status = "ok" if strong else "blocked_no_strong_chain"
    summary = {
        "status": status,
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "judge_dir": str(judge_dir.relative_to(ROOT) if judge_dir.is_relative_to(ROOT) else judge_dir),
        "judgments": str(judgments_path.relative_to(ROOT) if judgments_path.is_relative_to(ROOT) else judgments_path),
        "min_confidence": args.min_confidence,
        "total_judged": len(rows),
        "strong_chain": len(strong),
        "exploratory_weak_but_related": len(exploratory_weak),
        "verdict_counts": dict(verdict_counts),
        "by_stratum": {k: dict(v) for k, v in sorted(by_stratum.items())},
        "strong_by_stratum": dict(strong_by_stratum),
        "weak_by_stratum": dict(weak_by_stratum),
        "decision": (
            "No promotable high-precision cross-doc subset; downstream cross-doc "
            "M4 material/triplet generation should be blocked for resolver v1."
            if not strong else
            "Use only strong_chain rows for high-precision cross-doc M4 material conversion."
        ),
        "claim_scope": {
            "A_hard_title_window": "hard explicit precision claim stratum",
            "B_C_D": "exploratory explicit-resolution signals only",
            "E_F": "overlap baselines only",
        },
        "files": {
            "strong_subset": str((out_dir / "strong_chain_subset.jsonl").relative_to(ROOT)),
            "exploratory_weak": str((out_dir / "exploratory_weak_but_related.jsonl").relative_to(ROOT)),
            "summary": str((out_dir / "summary.json").relative_to(ROOT)),
            "summary_md": str((out_dir / "summary.md").relative_to(ROOT)),
        },
    }
    (out_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    md_lines = [
        "# XDoc Strong Subset Summary",
        "",
        f"- status: **{status}**",
        f"- total judged: **{len(rows)}**",
        f"- strong_chain: **{len(strong)}**",
        f"- exploratory weak_but_related: **{len(exploratory_weak)}**",
        f"- judge dir: `{summary['judge_dir']}`",
        "",
        "## Verdict Counts",
        "",
    ]
    for verdict, count in sorted(verdict_counts.items()):
        md_lines.append(f"- `{verdict}`: {count}")
    md_lines.extend(["", "## By Stratum", ""])
    for stratum, counts in sorted(by_stratum.items()):
        joined = ", ".join(f"{k}={v}" for k, v in sorted(counts.items()))
        md_lines.append(f"- `{stratum}`: {joined}")
    md_lines.extend(["", "## Decision", "", summary["decision"], ""])
    (out_dir / "summary.md").write_text("\n".join(md_lines), encoding="utf-8")

    latest = DEFAULT_OUT_PARENT / "xdoc_resolver_strong_subset_latest"
    atomic_latest_symlink(out_dir, latest)

    print(f"Output: {out_dir}")
    print(f"Status: {status}")
    print(f"strong_chain={len(strong)}/{len(rows)}")
    print(f"exploratory_weak_but_related={len(exploratory_weak)}")
    print(f"Latest: {latest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
