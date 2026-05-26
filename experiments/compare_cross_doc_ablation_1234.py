#!/usr/bin/env python3
"""Compare cross-document ablation 1-4 judge outputs."""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data/05_eval/cross_doc_ablation_1234"

RUNS = [
    ("1_path_baseline_fixed", ROOT / "data/05_eval/cross_doc_chain_judge_fixed/summary.json"),
    ("2_entity_cluster", OUT / "judge_entity_cluster/summary.json"),
    ("3_gated_path", OUT / "judge_gated_path/summary.json"),
    ("4_entity_cluster_enriched", OUT / "judge_entity_cluster_enriched/summary.json"),
]


def load(path: Path) -> dict:
    if not path.exists():
        return {"missing": True}
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def pct(value: float) -> str:
    return f"{value:.1%}"


def main() -> None:
    rows = []
    for name, path in RUNS:
        s = load(path)
        if s.get("missing"):
            rows.append({
                "strategy": name, "status": "missing", "total": 0,
                "strong": 0, "strong_rate": 0, "usable": 0, "usable_rate": 0,
                "keep": 0, "keep_rate": 0, "review": 0, "drop": 0,
            })
            continue
        prod = s.get("production_counts", {})
        rows.append({
            "strategy": name,
            "status": s.get("status", "unknown"),
            "total": s.get("total", 0),
            "strong": s.get("strong_chain", 0),
            "strong_rate": s.get("strong_rate", 0),
            "usable": s.get("usable_chain", 0),
            "usable_rate": s.get("usable_rate", 0),
            "keep": s.get("production_keep", 0),
            "keep_rate": s.get("production_keep_rate", 0),
            "review": prod.get("review", 0),
            "drop": prod.get("drop", 0),
            "main_failures": s.get("main_failure_counts", {}),
            "verdicts": s.get("verdict_counts", {}),
            "path": str(path.relative_to(ROOT)),
        })

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "ablation_comparison.json").write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")

    lines = [
        "# Cross-Document Ablation 1-4 Comparison",
        "",
        "| Strategy | Total | Strong | Usable | Keep | Review | Drop |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for r in rows:
        lines.append(
            f"| {r['strategy']} | {r['total']} | "
            f"{r['strong']} ({pct(r['strong_rate'])}) | "
            f"{r['usable']} ({pct(r['usable_rate'])}) | "
            f"{r['keep']} ({pct(r['keep_rate'])}) | "
            f"{r['review']} | {r['drop']} |"
        )
    lines.extend(["", "## Files", ""])
    for r in rows:
        if "path" in r:
            lines.append(f"- `{r['strategy']}`: `{r['path']}`")
    (OUT / "ablation_comparison.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
