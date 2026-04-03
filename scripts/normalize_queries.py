#!/usr/bin/env python3
"""Normalize historical L1/L2/L3 query files into a unified StandardQuery schema.

This script is idempotent: re-running it produces the same output given the
same inputs (deterministic field mapping, sorted output by query_id).

Usage
-----
::

    python scripts/normalize_queries.py \\
        --output data/queries_normalized.jsonl

By default it reads from the canonical ``data/m2/`` level files.  Use
``--l1``, ``--l2``, ``--l3`` to override individual paths.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List

# ── path bootstrap (until setup.py / pip install -e . is standardised) ────────
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

from src.models.training import (  # noqa: E402
    DifficultyLevel,
    EvidenceSpan,
    ReasoningStep,
    StandardQuery,
)

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")


# ── Converters ────────────────────────────────────────────────────────────────

def _convert_l1(raw: Dict[str, Any]) -> StandardQuery:
    """Convert a Level-1 single-element record."""
    # Build evidence span from legacy fields
    evidence: list[EvidenceSpan] = []
    for span_raw in raw.get("required_evidence_spans", []):
        evidence.append(
            EvidenceSpan(
                element_id=span_raw.get("element_id", raw.get("figure_id", "")),
                doc_id=raw.get("doc_id", ""),
                element_type=raw.get("figure_type", "figure"),
                span_text=span_raw.get("span", ""),
                evidence_type=span_raw.get("evidence_type", ""),
                image_path=raw.get("image_path"),
            )
        )

    # Fallback: if no evidence_spans, create one from figure_id
    if not evidence and raw.get("figure_id"):
        evidence.append(
            EvidenceSpan(
                element_id=raw["figure_id"],
                doc_id=raw.get("doc_id", ""),
                element_type=raw.get("figure_type", "figure"),
                span_text=raw.get("text_evidence", ""),
                image_path=raw.get("image_path"),
            )
        )

    image_paths: list[str] = []
    if raw.get("image_path"):
        image_paths = [raw["image_path"]]

    return StandardQuery(
        query_id=raw.get("query_id", ""),
        query_text=raw.get("query", ""),
        answer_text=raw.get("answer", ""),
        difficulty_level=DifficultyLevel.L1,
        evidence_spans=evidence,
        doc_id=raw.get("doc_id", ""),
        element_ids=[raw["figure_id"]] if raw.get("figure_id") else [],
        image_paths=image_paths,
        query_type=raw.get("query_type", ""),
        qc_pass=raw.get("qc_pass", True),
        qc_issues=raw.get("qc_issues", []),
        extra={
            k: raw[k]
            for k in ("visual_anchor", "caption", "figure_number",
                       "difficulty_label", "text_evidence")
            if k in raw
        },
    )


def _convert_l2(raw: Dict[str, Any]) -> StandardQuery:
    """Convert a Level-2 dual-evidence record."""
    return _convert_multihop(raw, DifficultyLevel.L2)


def _convert_l3(raw: Dict[str, Any]) -> StandardQuery:
    """Convert a Level-3 reasoning-chain record."""
    return _convert_multihop(raw, DifficultyLevel.L3)


def _convert_multihop(raw: Dict[str, Any], level: DifficultyLevel) -> StandardQuery:
    """Shared converter for L2/L3 (same upstream schema, 34 fields)."""
    # Evidence spans
    evidence: list[EvidenceSpan] = []
    for span_raw in raw.get("required_evidence_spans", []):
        eid = span_raw.get("element_id", "")
        evidence.append(
            EvidenceSpan(
                element_id=eid,
                doc_id=raw.get("doc_id", ""),
                element_type=_guess_type(eid),
                span_text=span_raw.get("span", ""),
                evidence_type=span_raw.get("evidence_type", ""),
            )
        )

    # Reasoning steps (L3)
    steps: list[ReasoningStep] = []
    for step_raw in raw.get("reasoning_steps", []):
        steps.append(
            ReasoningStep(
                step_id=step_raw.get("step_id", 0),
                evidence_element_id=step_raw.get("evidence_element_id", ""),
                evidence_type=step_raw.get("evidence_type", ""),
                evidence_span=step_raw.get("evidence_span", ""),
                reasoning_role=step_raw.get("reasoning_role", ""),
                depends_on_steps=step_raw.get("depends_on_steps", []),
                produces_claim=step_raw.get("produces_claim", ""),
            )
        )

    element_ids: list[str] = raw.get("element_ids", [])
    image_paths: list[str] = raw.get("image_paths", [])
    qc_issues_raw = raw.get("qc_issues", [])
    if isinstance(qc_issues_raw, dict):
        qc_issues_raw = list(qc_issues_raw.keys())

    return StandardQuery(
        query_id=raw.get("query_id", ""),
        query_text=raw.get("query", ""),
        answer_text=raw.get("answer", ""),
        difficulty_level=level,
        evidence_spans=evidence,
        reasoning_steps=steps,
        doc_id=raw.get("doc_id", ""),
        pair_id=raw.get("pair_id", ""),
        pair_type=raw.get("pair_type", ""),
        element_ids=element_ids,
        image_paths=image_paths,
        query_style=raw.get("query_style", "academic"),
        query_type=raw.get("query_type", ""),
        persona=raw.get("persona", ""),
        hop_distance=raw.get("hop_distance", 0),
        qc_pass=raw.get("qc_pass", True),
        qc_issues=qc_issues_raw,
        extra={
            k: raw[k]
            for k in (
                "bridge_quality", "graph_path", "path", "reasoning_chain",
                "reasoning_depth", "reasoning_structure", "quality_tier",
                "cross_modal", "dual_evidence", "visual_anchors",
                "text_evidence", "difficulty_label", "query_intent",
                "query_length_bucket", "qc_metrics",
            )
            if k in raw
        },
    )


def _guess_type(element_id: str) -> str:
    """Infer element type from its ID convention (e.g. 'docid_figure_1')."""
    lower = element_id.lower()
    if "figure" in lower or "fig" in lower:
        return "figure"
    if "table" in lower or "tbl" in lower:
        return "table"
    if "formula" in lower or "eq" in lower:
        return "formula"
    if "bridge" in lower or "paragraph" in lower:
        return "text"
    return "unknown"


# ── I/O ───────────────────────────────────────────────────────────────────────

def _load_jsonl(path: Path) -> List[Dict[str, Any]]:
    records: list[dict] = []
    if not path.exists():
        logger.warning("File not found, skipping: %s", path)
        return records
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def normalize(
    l1_path: Path,
    l2_path: Path,
    l3_path: Path,
    output_path: Path,
) -> Dict[str, int]:
    """Run the full normalisation pipeline.  Returns counts per level."""
    all_queries: list[StandardQuery] = []
    counts: Dict[str, int] = {}

    for label, path, converter in [
        ("L1", l1_path, _convert_l1),
        ("L2", l2_path, _convert_l2),
        ("L3", l3_path, _convert_l3),
    ]:
        records = _load_jsonl(path)
        converted = 0
        for rec in records:
            try:
                sq = converter(rec)
                all_queries.append(sq)
                converted += 1
            except Exception as exc:  # noqa: BLE001
                logger.warning("[%s] Skipping %s: %s",
                               label, rec.get("query_id", "?"), exc)
        counts[label] = converted
        logger.info("%s: %d / %d converted", label, converted, len(records))

    # sort for determinism
    all_queries.sort(key=lambda q: q.query_id)

    # write
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as fh:
        for q in all_queries:
            fh.write(q.model_dump_json() + "\n")

    logger.info("Wrote %d normalised queries → %s", len(all_queries), output_path)
    counts["total"] = len(all_queries)
    return counts


# ── CLI ───────────────────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser(
        description="Normalize L1/L2/L3 queries into StandardQuery schema",
    )
    data_dir = _PROJECT_ROOT / "data" / "m2"
    ap.add_argument("--l1", type=Path, default=data_dir / "level1_single_element.jsonl")
    ap.add_argument("--l2", type=Path, default=data_dir / "level2_dual_evidence.jsonl")
    ap.add_argument("--l3", type=Path, default=data_dir / "level3_reasoning_chain.jsonl")
    ap.add_argument(
        "--output", "-o", type=Path,
        default=_PROJECT_ROOT / "data" / "queries_normalized.jsonl",
    )
    args = ap.parse_args()
    normalize(args.l1, args.l2, args.l3, args.output)


if __name__ == "__main__":
    main()
