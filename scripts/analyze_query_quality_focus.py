#!/usr/bin/env python3
"""Analyze query quality focus areas: length diversity and architecture failures."""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Tuple


SHORT_MIN = 8
SHORT_MAX = 14
LONG_MIN = 18
LONG_MAX = 30

ARCHITECTURE_KEYWORDS = {
    "architecture", "framework", "pipeline", "overview", "module", "modules",
    "component", "components", "encoder", "decoder", "branch", "branches",
    "backbone", "head", "heads", "block", "blocks", "topology", "fusion",
    "stream", "pathway", "network",
}


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def word_count(text: str) -> int:
    return len(re.findall(r"[A-Za-z0-9][A-Za-z0-9_-]*", text or ""))


def length_bucket(query: str) -> str:
    wc = word_count(query)
    if wc < SHORT_MIN:
        return "too_short"
    if wc <= SHORT_MAX:
        return "short"
    if wc < LONG_MIN:
        return "medium"
    if wc <= LONG_MAX:
        return "long"
    return "too_long"


def has_architecture_text(text: str) -> bool:
    t = (text or "").lower()
    return any(re.search(rf"\b{re.escape(k)}\b", t) for k in ARCHITECTURE_KEYWORDS)


def build_element_index(elements_path: Path) -> Dict[str, Dict[str, Any]]:
    if not elements_path.exists():
        return {}
    data = json.loads(elements_path.read_text(encoding="utf-8"))
    out: Dict[str, Dict[str, Any]] = {}
    for _, doc in (data.get("documents", {}) or {}).items():
        for eid, elem in (doc.get("elements", {}) or {}).items():
            out[eid] = elem
    return out


def row_is_architecture(row: Dict[str, Any], element_index: Dict[str, Dict[str, Any]]) -> bool:
    for eid in (row.get("element_ids", []) or []):
        elem = element_index.get(str(eid), {})
        if str(elem.get("element_type", "")) != "figure":
            continue
        fig_text = " ".join(
            [
                str(elem.get("caption", "") or ""),
                str(elem.get("content", "") or ""),
                str(elem.get("context_before", "") or ""),
                str(elem.get("context_after", "") or ""),
            ]
        )
        if has_architecture_text(fig_text):
            return True
    return False


def summarize_pair_length_mix(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    by_pair: Dict[str, List[str]] = defaultdict(list)
    for r in rows:
        pid = str(r.get("pair_id", ""))
        if not pid:
            continue
        by_pair[pid].append(length_bucket(str(r.get("query", ""))))

    mix_ok = 0
    no_short = 0
    no_long = 0
    for buckets in by_pair.values():
        has_short = "short" in buckets
        has_long = "long" in buckets
        if has_short and has_long:
            mix_ok += 1
        if not has_short:
            no_short += 1
        if not has_long:
            no_long += 1

    total_pairs = len(by_pair)
    return {
        "total_pairs": total_pairs,
        "pairs_with_length_mix": mix_ok,
        "length_mix_ratio": round(mix_ok / max(1, total_pairs), 4),
        "pairs_missing_short": no_short,
        "pairs_missing_long": no_long,
    }


def top_issues(rows: List[Dict[str, Any]], limit: int = 12) -> List[Tuple[str, int]]:
    c = Counter()
    for r in rows:
        for iss in (r.get("qc_issues", []) or []):
            c[str(iss)] += 1
    return c.most_common(limit)


def main() -> None:
    ap = argparse.ArgumentParser(description="Analyze query quality focus dimensions.")
    ap.add_argument("--queries", default="data/l1_dual_evidence_queries_slurm_img150_tuned_v4_official.jsonl")
    ap.add_argument("--elements", default="data/multimodal_elements.json")
    ap.add_argument("--output", default="data/query_quality_focus_report.json")
    ap.add_argument("--sample-limit", type=int, default=20)
    args = ap.parse_args()

    query_path = Path(args.queries)
    elements_path = Path(args.elements)
    if not query_path.exists():
        raise FileNotFoundError(f"Missing --queries: {query_path}")

    rows = read_jsonl(query_path)
    element_index = build_element_index(elements_path)

    length_dist = Counter(length_bucket(str(r.get("query", ""))) for r in rows)
    length_by_type: Dict[str, Counter] = defaultdict(Counter)
    for r in rows:
        ptype = str(r.get("pair_type", "unknown"))
        length_by_type[ptype][length_bucket(str(r.get("query", "")))] += 1

    ff_rows = [r for r in rows if str(r.get("pair_type", "")) == "figure+formula"]
    arch_rows = [r for r in rows if row_is_architecture(r, element_index)]
    arch_ff_rows = [r for r in ff_rows if row_is_architecture(r, element_index)]

    report = {
        "metadata": {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "queries": str(query_path),
            "elements": str(elements_path),
            "short_range": [SHORT_MIN, SHORT_MAX],
            "long_range": [LONG_MIN, LONG_MAX],
        },
        "summary": {
            "total_queries": len(rows),
            "figure_formula_queries": len(ff_rows),
            "architecture_queries": len(arch_rows),
            "architecture_figure_formula_queries": len(arch_ff_rows),
        },
        "length_distribution": dict(length_dist),
        "length_distribution_by_pair_type": {
            k: dict(v) for k, v in sorted(length_by_type.items())
        },
        "pair_length_mix": summarize_pair_length_mix(rows),
        "qc_issues_global_top": top_issues(rows),
        "qc_issues_figure_formula_top": top_issues(ff_rows),
        "qc_issues_architecture_top": top_issues(arch_rows),
        "architecture_failure_samples": [
            {
                "query_id": r.get("query_id", ""),
                "pair_id": r.get("pair_id", ""),
                "pair_type": r.get("pair_type", ""),
                "query": r.get("query", ""),
                "answer": r.get("answer", "")[:220],
                "qc_issues": r.get("qc_issues", []),
                "qc_pass": bool(r.get("qc_pass", False)),
            }
            for r in arch_ff_rows[: max(0, args.sample_limit)]
        ],
    }

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print("=" * 68)
    print("Query quality focus analysis complete")
    print("=" * 68)
    print(f"Total queries:        {report['summary']['total_queries']}")
    print(f"Length distribution:  {dict(length_dist)}")
    print(f"Pair length mix:      {report['pair_length_mix']}")
    print(f"Architecture rows:    {report['summary']['architecture_queries']}")
    print(f"Figure+formula rows:  {report['summary']['figure_formula_queries']}")
    print(f"Output:               {out_path}")
    print("=" * 68)


if __name__ == "__main__":
    main()
