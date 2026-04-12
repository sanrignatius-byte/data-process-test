#!/usr/bin/env python3
"""Analyze Method C old-data experiment and auto-launch the next-wave run.

Current policy:
  1. Summarize the finished old-data bridge1 / bridge2 experiment.
  2. Detect whether bridge2 on old data is a valid "true 2-bridge" comparison.
  3. If not, snapshot the current partial scale-up enrich file.
  4. Build partial long-chain feasibility candidates.
  5. Filter to pairs whose compressed bridge count is truly >= 2.
  6. Auto-submit the next-wave experiment on that filtered subset.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.pilot_method_c import build_method_c_view


def _load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _utc_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H%M%SZ")


def summarize_run(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {"exists": False, "path": str(path)}

    data = _load_json(path)
    rows = [r for r in data.get("results", []) if "qc" in r]
    issues = Counter()
    compressed_counts = Counter()
    for row in rows:
        compressed_counts[int(row.get("method_c", {}).get("compressed_bridge_count", 0))] += 1
        for issue in row["qc"].get("all_issues", []):
            issues[issue] += 1

    passed = sum(1 for r in rows if r["qc"].get("passed"))
    return {
        "exists": True,
        "path": str(path),
        "rows": len(rows),
        "passed": passed,
        "failed": len(rows) - passed,
        "issue_counts": dict(issues),
        "compressed_bridge_count_dist": dict(compressed_counts),
        "metadata": data.get("metadata", {}),
    }


def stable_snapshot_json(src: Path, dst: Path) -> Dict[str, Any]:
    data = _load_json(src)
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    return data


def build_partial_candidates(elements_snapshot: Path, output_path: Path) -> Dict[str, Any]:
    cmd = [
        sys.executable,
        "scripts/build_method_c_feasibility_candidates.py",
        "--elements", str(elements_snapshot.relative_to(PROJECT_ROOT)),
        "--min-hop", "4",
        "--max-hop", "5",
        "--output", str(output_path.relative_to(PROJECT_ROOT)),
    ]
    subprocess.run(cmd, cwd=PROJECT_ROOT, check=True)
    return _load_json(output_path)


def filter_true_two_bridge_candidates(
    candidate_data: Dict[str, Any],
    output_path: Path,
) -> Dict[str, Any]:
    kept: List[Dict[str, Any]] = []
    counts = Counter()
    for pair in candidate_data.get("pairs", []):
        view = build_method_c_view(pair, max_bridge_nodes=2)
        cnt = int(view.get("compressed_bridge_count", 0))
        counts[cnt] += 1
        if cnt >= 2:
            kept.append(pair)

    out = {
        "metadata": {
            **candidate_data.get("metadata", {}),
            "selection": "method_c_true_two_bridge_partial_scaleup",
        },
        "summary": {
            **candidate_data.get("summary", {}),
            "true_two_bridge_pairs": len(kept),
            "compressed_bridge_count_dist": dict(counts),
        },
        "pairs": kept,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    return out


def submit_followup_job(input_candidates: Path, sample_limit: int) -> str:
    out_b1 = PROJECT_ROOT / "data/03_queries/pilot_method_c_v3_true2_bridge1.json"
    out_b2 = PROJECT_ROOT / "data/03_queries/pilot_method_c_v3_true2_bridge2.json"
    cmd = [
        "sbatch",
        "--parsable",
        "--job-name=method_c_true2",
        "--output=logs/method_c_true2_%j.out",
        "--error=logs/method_c_true2_%j.err",
        "--export=ALL,"
        f"OLD_ENRICHED={input_candidates.relative_to(PROJECT_ROOT)},"
        f"MIN_HOP=4,MAX_HOP=5,NUM_SAMPLES={sample_limit},PER_DOC_CAP=0,"
        f"OUT_B1={out_b1.relative_to(PROJECT_ROOT)},"
        f"OUT_B2={out_b2.relative_to(PROJECT_ROOT)}",
        "slurm_scripts/10_run_method_c_old_full.sh",
    ]
    completed = subprocess.run(cmd, cwd=PROJECT_ROOT, check=True, capture_output=True, text=True)
    return completed.stdout.strip()


def write_report(md_path: Path, payload: Dict[str, Any]) -> None:
    old = payload["old_experiment"]
    follow = payload["followup"]
    lines = [
        "# Method C Auto Follow-up Report",
        "",
        f"- Timestamp: `{payload['timestamp']}`",
        f"- Source job: `{payload['source_job_id']}`",
        "",
        "## Old Experiment",
        f"- bridge1 rows/pass/fail: `{old['bridge1'].get('rows', 0)}/{old['bridge1'].get('passed', 0)}/{old['bridge1'].get('failed', 0)}`",
        f"- bridge2 rows/pass/fail: `{old['bridge2'].get('rows', 0)}/{old['bridge2'].get('passed', 0)}/{old['bridge2'].get('failed', 0)}`",
        f"- bridge1 compressed dist: `{old['bridge1'].get('compressed_bridge_count_dist', {})}`",
        f"- bridge2 compressed dist: `{old['bridge2'].get('compressed_bridge_count_dist', {})}`",
        f"- next-wave rationale: {payload['rationale']}",
        "",
        "## Follow-up",
        f"- snapshot: `{follow.get('snapshot')}`",
        f"- partial candidates: `{follow.get('partial_candidates')}`",
        f"- true-two-bridge candidates: `{follow.get('true_two_bridge_candidates')}`",
        f"- true-two count: `{follow.get('true_two_count')}`",
        f"- submitted job: `{follow.get('submitted_job_id')}`",
    ]
    md_path.parent.mkdir(parents=True, exist_ok=True)
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--source-job-id", required=True)
    ap.add_argument("--bridge1", default="data/03_queries/pilot_method_c_v3_old_full_bridge1.json")
    ap.add_argument("--bridge2", default="data/03_queries/pilot_method_c_v3_old_full_bridge2.json")
    ap.add_argument(
        "--current-elements",
        default="data/02_enriched/method_c_long_chain_elements_enriched_v1.json",
    )
    ap.add_argument("--sample-limit", type=int, default=48)
    ap.add_argument(
        "--report-json",
        default="data/03_queries/method_c_auto_followup_latest.json",
    )
    ap.add_argument(
        "--report-md",
        default="data/03_queries/method_c_auto_followup_latest.md",
    )
    args = ap.parse_args()

    bridge1 = summarize_run(PROJECT_ROOT / args.bridge1)
    bridge2 = summarize_run(PROJECT_ROOT / args.bridge2)

    old_exp = {"bridge1": bridge1, "bridge2": bridge2}
    bridge2_dist = bridge2.get("compressed_bridge_count_dist", {}) if bridge2.get("exists") else {}
    max_bridge2 = max((int(k) for k in bridge2_dist.keys()), default=0)
    rationale = (
        "Old enriched pairs do not form a valid true-2-bridge comparison; "
        "auto-launching a partial long-chain follow-up restricted to compressed_bridge_count >= 2."
        if max_bridge2 < 2
        else
        "Old enriched bridge2 run already contains true 2-bridge samples; "
        "still launching partial long-chain follow-up for harder 4-5 hop validation."
    )

    stamp = _utc_stamp()
    snapshot_path = PROJECT_ROOT / f"data/02_enriched/method_c_long_chain_elements_enriched_snapshot_{stamp}.json"
    stable_snapshot_json(PROJECT_ROOT / args.current_elements, snapshot_path)

    partial_candidates_path = PROJECT_ROOT / f"data/03_queries/method_c_partial_candidates_{stamp}.json"
    partial_data = build_partial_candidates(snapshot_path, partial_candidates_path)

    true2_candidates_path = PROJECT_ROOT / f"data/03_queries/method_c_true2_candidates_{stamp}.json"
    true2_data = filter_true_two_bridge_candidates(partial_data, true2_candidates_path)
    true2_count = len(true2_data.get("pairs", []))
    sample_limit = min(args.sample_limit, true2_count) if true2_count > 0 else 0

    submitted_job_id = ""
    if sample_limit > 0:
        submitted_job_id = submit_followup_job(true2_candidates_path, sample_limit)

    payload = {
        "timestamp": stamp,
        "source_job_id": args.source_job_id,
        "rationale": rationale,
        "old_experiment": old_exp,
        "followup": {
            "snapshot": str(snapshot_path.relative_to(PROJECT_ROOT)),
            "partial_candidates": str(partial_candidates_path.relative_to(PROJECT_ROOT)),
            "true_two_bridge_candidates": str(true2_candidates_path.relative_to(PROJECT_ROOT)),
            "true_two_count": true2_count,
            "sample_limit": sample_limit,
            "submitted_job_id": submitted_job_id,
        },
    }

    report_json = PROJECT_ROOT / args.report_json
    report_json.parent.mkdir(parents=True, exist_ok=True)
    report_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    write_report(PROJECT_ROOT / args.report_md, payload)

    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
