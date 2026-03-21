#!/usr/bin/env python3
"""Package existing query data into M2 difficulty levels.

Level 1: Single-element queries (l1_cross_modal_queries_v3.jsonl)
Level 2: Dual-evidence queries (v3_pass + v4_4_run1_pass, deduped)
Level 3: 3-step reasoning chain queries (generated separately)

Outputs unified JSONL files to data/m2/ with a difficulty_level field.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
M2_DIR = DATA_DIR / "m2"


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    if not path.exists():
        print(f"  WARN: {path} not found, skipping")
        return rows
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def save_jsonl(rows: List[Dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"  Wrote {len(rows)} rows -> {path}")


def package_level1() -> List[Dict[str, Any]]:
    """Level 1: Single-element cross-modal queries."""
    path = DATA_DIR / "l1_cross_modal_queries_v3.jsonl"
    rows = load_jsonl(path)
    for r in rows:
        r["difficulty_level"] = 1
        r["difficulty_label"] = "single_element"
        # Ensure evidence structure is uniform
        if "required_evidence_spans" not in r:
            r["required_evidence_spans"] = []
            # Build from existing fields
            if r.get("visual_anchor") and r.get("figure_id"):
                r["required_evidence_spans"].append({
                    "element_id": r["figure_id"],
                    "span": r.get("text_evidence", r.get("visual_anchor", "")),
                    "evidence_type": "observation",
                })
    print(f"  Level 1: {len(rows)} single-element queries")
    return rows


def package_level2() -> List[Dict[str, Any]]:
    """Level 2: Dual-evidence queries (deduped union of v3_pass + v4.4_pass)."""
    v3 = load_jsonl(DATA_DIR / "l1_dual_evidence_queries_v3_pass.jsonl")
    v44 = load_jsonl(DATA_DIR / "l1_dual_evidence_queries_v4_4_run1_pass.jsonl")

    # Deduplicate by pair_id (prefer v4.4 which is newer)
    seen_pairs: set = set()
    deduped: List[Dict[str, Any]] = []
    for r in v44:
        pid = r.get("pair_id", "")
        if pid and pid in seen_pairs:
            continue
        seen_pairs.add(pid)
        r["difficulty_level"] = 2
        r["difficulty_label"] = "dual_evidence"
        deduped.append(r)
    for r in v3:
        pid = r.get("pair_id", "")
        if pid and pid in seen_pairs:
            continue
        seen_pairs.add(pid)
        r["difficulty_level"] = 2
        r["difficulty_label"] = "dual_evidence"
        deduped.append(r)

    print(f"  Level 2: {len(deduped)} dual-evidence queries (v3={len(v3)}, v4.4={len(v44)}, deduped)")
    return deduped


def package_level3() -> List[Dict[str, Any]]:
    """Level 3: 3-step reasoning chain queries (if already generated)."""
    path = DATA_DIR / "m2" / "l3_reasoning_chain_queries_pass.jsonl"
    if not path.exists():
        print(f"  Level 3: not yet generated ({path})")
        return []
    rows = load_jsonl(path)
    for r in rows:
        r["difficulty_level"] = 3
        r["difficulty_label"] = "reasoning_chain"
    print(f"  Level 3: {len(rows)} reasoning-chain queries")
    return rows


def repackage_exp_b() -> Dict[str, Any]:
    """Repackage Phase0 eval report for Experiment B."""
    report_paths = [
        DATA_DIR / "phase0_eval_report_v3_tuned.json",
        DATA_DIR / "phase0_eval_report_bugfix.json",
        DATA_DIR / "phase0_eval_report_tuned.json",
    ]
    report: Dict[str, Any] = {}
    for p in report_paths:
        if p.exists():
            report = json.loads(p.read_text(encoding="utf-8"))
            print(f"  Exp B: loaded {p.name}")
            break
    if not report:
        print("  Exp B: no phase0 eval report found")
        return {}

    # Extract key numbers for experiment B
    # Phase0 report structure: config.n_queries/n_chunks, metrics.{method}.{recall_at_10,mrr}
    cfg = report.get("config", {})
    methods = report.get("metrics", report.get("methods", report.get("results", {})))
    exp_b = {
        "experiment": "B_retrieval_enhancement",
        "description": "Document Graph vs BM25 baseline retrieval",
        "source_report": str(report_paths[0].name),
        "query_count": cfg.get("n_queries", report.get("query_count", 0)),
        "chunk_count": cfg.get("n_chunks", report.get("chunk_count", 0)),
        "results": {},
    }

    if isinstance(methods, dict):
        for method_name, method_data in methods.items():
            if isinstance(method_data, dict):
                exp_b["results"][method_name] = {
                    "recall_at_10": method_data.get("recall_at_10", method_data.get("recall@10")),
                    "mrr": method_data.get("mrr", method_data.get("MRR")),
                    "n": method_data.get("n"),
                }

    # Also include layered results if available
    layered = report.get("layered", {})
    if layered:
        exp_b["hub_overlap_pct"] = layered.get("hub_overlap_pct")
        exp_b["layered_results"] = {}
        for k, v in layered.items():
            if isinstance(v, dict) and "recall_at_10" in v:
                exp_b["layered_results"][k] = {
                    "recall_at_10": v.get("recall_at_10"),
                    "mrr": v.get("mrr"),
                    "n": v.get("n"),
                }

    return exp_b


def main() -> None:
    M2_DIR.mkdir(parents=True, exist_ok=True)
    print("Packaging M2 difficulty levels...")
    print()

    # Package each level
    l1 = package_level1()
    l2 = package_level2()
    l3 = package_level3()

    # Save individual level files
    if l1:
        save_jsonl(l1, M2_DIR / "level1_single_element.jsonl")
    if l2:
        save_jsonl(l2, M2_DIR / "level2_dual_evidence.jsonl")
    if l3:
        save_jsonl(l3, M2_DIR / "level3_reasoning_chain.jsonl")

    # Save combined file
    combined = l1 + l2 + l3
    save_jsonl(combined, M2_DIR / "all_levels_combined.jsonl")

    # Repackage experiment B
    exp_b = repackage_exp_b()
    if exp_b:
        exp_b_path = M2_DIR / "exp_b_retrieval_enhancement.json"
        exp_b_path.write_text(json.dumps(exp_b, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"  Wrote exp B report -> {exp_b_path}")

    # Summary
    print(f"\n{'='*50}")
    print(f"M2 Data Summary")
    print(f"{'='*50}")
    print(f"  Level 1 (single-element):    {len(l1):>5}")
    print(f"  Level 2 (dual-evidence):     {len(l2):>5}")
    print(f"  Level 3 (reasoning-chain):   {len(l3):>5}")
    print(f"  Total:                       {len(combined):>5}")
    print(f"  Output directory:            {M2_DIR}")
    print(f"{'='*50}")


if __name__ == "__main__":
    main()
