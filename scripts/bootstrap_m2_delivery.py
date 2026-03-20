#!/usr/bin/env python3
"""Bootstrap the M2 delivery workspace from the repo's current artifacts.

This script does two things:
1. Audits the current repo status against the proposed M2 five-phase plan.
2. Writes a machine-readable manifest that maps existing assets to the planned
   deliverables (Level 1 / Level 2 / Level 3 / general queries / experiments).

It is intentionally lightweight: no data regeneration, no LLM calls, only local
inspection so it can be used as the first execution step before longer jobs.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]


@dataclass
class ArtifactStatus:
    logical_name: str
    path: str
    exists: bool
    record_count: int | None
    notes: str


def count_records(path: Path) -> int | None:
    if not path.exists():
        return None
    if path.suffix == ".jsonl":
        with path.open("r", encoding="utf-8") as f:
            return sum(1 for line in f if line.strip())
    if path.suffix == ".json":
        obj = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(obj, list):
            return len(obj)
        if isinstance(obj, dict):
            for key in ("queries", "pairs", "hubs", "documents", "rows"):
                value = obj.get(key)
                if isinstance(value, (list, dict)):
                    return len(value)
    return None


def build_artifact(logical_name: str, rel_path: str, notes: str) -> ArtifactStatus:
    path = ROOT / rel_path
    return ArtifactStatus(
        logical_name=logical_name,
        path=rel_path,
        exists=path.exists(),
        record_count=count_records(path),
        notes=notes,
    )


def load_phase0_metrics(rel_path: str) -> dict[str, Any]:
    path = ROOT / rel_path
    if not path.exists():
        return {"path": rel_path, "exists": False}
    obj = json.loads(path.read_text(encoding="utf-8"))
    metrics = obj.get("metrics", {})
    bm25 = metrics.get("bm25", {})
    graph_full = metrics.get("graph_full", {})
    return {
        "path": rel_path,
        "exists": True,
        "bm25": {
            "recall_at_10": bm25.get("recall_at_10"),
            "mrr": bm25.get("mrr"),
            "n": bm25.get("n"),
        },
        "graph_full": {
            "recall_at_10": graph_full.get("recall_at_10"),
            "mrr": graph_full.get("mrr"),
            "n": graph_full.get("n"),
        },
        "delta_vs_bm25": {
            "recall_at_10": round((graph_full.get("recall_at_10", 0) or 0) - (bm25.get("recall_at_10", 0) or 0), 4),
            "mrr": round((graph_full.get("mrr", 0) or 0) - (bm25.get("mrr", 0) or 0), 4),
        },
    }


def build_manifest() -> dict[str, Any]:
    current_assets = {
        "level1_single": [
            build_artifact(
                "l1_multihop_v3",
                "data/l1_multihop_queries_v3.jsonl",
                "Current best single-element/L1-style pool mentioned in status docs.",
            ),
        ],
        "level2_dual_evidence": [
            build_artifact(
                "dual_evidence_queries_v3_pass",
                "data/l1_dual_evidence_queries_v3_pass.jsonl",
                "QC-passed dual-evidence queries; conservative baseline for M2 Level 2.",
            ),
            build_artifact(
                "dual_evidence_triplets_v2_pass",
                "data/l1_dual_evidence_triplets_v2_pass.jsonl",
                "Triplet training/eval view over the same dual-evidence core set.",
            ),
            build_artifact(
                "dual_evidence_queries_v4_4_run1",
                "data/l1_dual_evidence_queries_v4_4_run1.jsonl",
                "Larger but not fully passed set; useful expansion pool once re-QC'd.",
            ),
        ],
        "level3_reasoning_chain": [
            build_artifact(
                "long_chain_iterative_v2",
                "data/l1_dual_evidence_long_chain_queries_v2_iterative.jsonl",
                "Nearest existing precursor to 3-step reasoning data; currently small and not yet true native Level 3.",
            ),
            build_artifact(
                "l2_queries_v3",
                "data/l2_queries_v3.jsonl",
                "Cross-doc/L2 experiments; useful schema reference, not yet final M2 Level 3.",
            ),
        ],
        "retrieval_eval": [
            build_artifact(
                "phase0_eval_report_tuned",
                "data/phase0_eval_report_v3_tuned.json",
                "Best current retrieval enhancement report for experiment B packaging.",
            ),
            build_artifact(
                "phase0_eval_report_fixed",
                "data/phase0_eval_report_v3_fixed.json",
                "Conservative locked-protocol report for comparison.",
            ),
        ],
        "embedding_bridge_probe": [
            build_artifact(
                "crossdoc_embedding_matches",
                "data/mineru_crossdoc_embedding_matches_Qwen3-Embedding-4B.jsonl",
                "Raw embedding-based element matching candidates.",
            ),
            build_artifact(
                "crossdoc_embedding_audit",
                "data/mineru_crossdoc_embedding_matches_Qwen3-Embedding-4B_v2_rerank_audit.json",
                "Existing audit evidence for embedding-based virtual edges.",
            ),
        ],
    }

    target_outputs = {
        "data": [
            "data/m2/m2_level1_single.jsonl",
            "data/m2/m2_level2_dual_evidence.jsonl",
            "data/m2/m2_level3_3step.jsonl",
            "data/m2/m2_general_queries.jsonl",
        ],
        "experiments": [
            "experiments/exp_A_difficulty_gradient.json",
            "experiments/exp_B_retrieval_enhancement.json",
            "experiments/exp_C_qa_triangle.json",
        ],
        "docs": [
            "docs/GRAPH_ARCHITECTURE.md",
        ],
    }

    output_status = []
    for group, paths in target_outputs.items():
        for rel_path in paths:
            path = ROOT / rel_path
            output_status.append({
                "group": group,
                "path": rel_path,
                "exists": path.exists(),
            })

    return {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "repo_root": str(ROOT),
        "current_assets": {
            key: [asdict(item) for item in items]
            for key, items in current_assets.items()
        },
        "phase0_best_report": load_phase0_metrics("data/phase0_eval_report_v3_tuned.json"),
        "target_output_status": output_status,
        "execution_notes": [
            "Stage 1 can start immediately from current artifacts without waiting for new long jobs.",
            "Level 2 already has a conservative passed subset plus a larger re-QC candidate pool.",
            "Level 3 is the biggest gap: current long-chain artifacts are precursors, not yet the final native 3-step benchmark.",
            "Experiment B can be packaged immediately from the tuned Phase-0 report, while experiments A/C still need dedicated outputs.",
            "Embedding-based virtual edges already have raw matches + audit artifacts, so Stage 4 can start from a small-doc probe instead of a cold start.",
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Bootstrap the M2 delivery workspace and manifest")
    parser.add_argument(
        "--output",
        default="experiments/m2_execution_manifest.json",
        help="Manifest output path relative to repo root.",
    )
    args = parser.parse_args()

    manifest = build_manifest()
    out_path = ROOT / args.output
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({
        "output": str(out_path.relative_to(ROOT)),
        "phase0_best_report": manifest["phase0_best_report"],
        "target_outputs_ready": sum(1 for item in manifest["target_output_status"] if item["exists"]),
        "target_outputs_total": len(manifest["target_output_status"]),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
