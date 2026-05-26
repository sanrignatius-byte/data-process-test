#!/usr/bin/env python3
"""Convert high-precision xdoc resolver subset rows into M4 material records.

This is the bridge from P1 (judged strong cross-doc chains) to P2 (multi-turn
triplet generation). It refuses to invent material when the strong subset is
empty, and writes an explicit blocked summary instead.
"""

from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SUBSET_DIR = ROOT / "data/05_eval/xdoc_resolver_strong_subset_latest"
DEFAULT_JUDGE_PACK = ROOT / "data/05_eval/xdoc_element_resolver_v1_latest/judge_pack_120.jsonl"
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


def atomic_latest_symlink(target: Path, latest: Path) -> None:
    tmp = latest.with_name(latest.name + ".tmp")
    if tmp.exists() or tmp.is_symlink():
        tmp.unlink()
    os.symlink(target, tmp)
    os.replace(tmp, latest)


def element_record(row: dict[str, Any], side: str) -> dict[str, Any]:
    prefix = "source" if side == "source" else "target"
    elem_id = row.get(f"{prefix}_element_id", "")
    return {
        "element_id": elem_id,
        "doc_id": row.get(f"{prefix}_doc", ""),
        "element_type": row.get(f"{prefix}_element_type", "element"),
        "caption": row.get(f"{prefix}_caption_or_content", ""),
        "image_path": row.get("element_a_image_path" if side == "source" else "element_b_image_path", ""),
        "enriched_title": row.get(f"{prefix}_caption_or_content", "")[:180],
        "enriched_content": row.get(f"{prefix}_caption_or_content", ""),
        "enrichment_issues": [],
    }


def material_from_row(row: dict[str, Any], judgment: dict[str, Any]) -> dict[str, Any]:
    candidate_id = row.get("candidate_id", "xdoc_unknown")
    source_id = row.get("source_element_id", "")
    target_id = row.get("target_element_id", "")
    source_doc = row.get("source_doc", "")
    target_doc = row.get("target_doc", "")
    bridge_id = f"{source_doc}::citation_bridge::{candidate_id}"
    target_context_id = f"{target_doc}::target_context::{target_id}"

    bridge_text = row.get("citation_bridge_text", "")
    evidence = judgment.get("evidence") or {}
    judge_rationale = judgment.get("rationale", "")
    target_context = (
        f"Target element context from {target_doc}: "
        f"{row.get('target_caption_or_content', '')} "
        f"Judge target cue: {evidence.get('target_cue', '')} "
        f"Rationale: {judge_rationale}"
    ).strip()

    source_type = row.get("source_element_type", "element")
    target_type = row.get("target_element_type", "element")
    return {
        "material_id": f"m4_xdoc_material_{candidate_id}",
        "pair_id": candidate_id,
        "doc_id": source_doc,
        "hop_distance": 4,
        "pair_type": row.get("pair_type", f"{source_type}+{target_type}"),
        "path": [source_id, bridge_id, target_context_id, target_id],
        "element_a": element_record(row, "source"),
        "element_b": element_record(row, "target"),
        "method_c": {
            "full_path_ids": [source_id, bridge_id, target_context_id, target_id],
            "compressed_bridge_count": 2,
            "compressed_chain_ids": [source_id, bridge_id, target_context_id, target_id],
            "compressed_chain_types": [source_type, "text", "bridge", target_type],
            "compressed_bridge_summaries": [bridge_text, target_context],
            "compression_summary": (
                "Cross-document citation chain accepted by xdoc resolver judge "
                f"as {judgment.get('verdict')} with confidence {judgment.get('confidence')}."
            ),
        },
        "xdoc_resolver": {
            "candidate_id": candidate_id,
            "target_stratum": row.get("target_stratum"),
            "target_anchor_reason": row.get("target_anchor_reason"),
            "citation_probability": row.get("citation_probability"),
            "target_resolution_score": row.get("target_resolution_score"),
            "quality_score": row.get("quality_score"),
            "judge_verdict": judgment.get("verdict"),
            "judge_confidence": judgment.get("confidence"),
        },
        "checks": {
            "cross_doc": source_doc != target_doc,
            "judge_strong_chain": judgment.get("verdict") == "strong_chain",
            "source_and_target_present": bool(source_id and target_id),
        },
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--subset-dir", type=Path, default=DEFAULT_SUBSET_DIR)
    ap.add_argument("--judge-pack", type=Path, default=DEFAULT_JUDGE_PACK)
    ap.add_argument("--out-dir", type=Path, default=None)
    args = ap.parse_args()

    subset_dir = args.subset_dir.resolve()
    strong_path = subset_dir / "strong_chain_subset.jsonl"
    if not strong_path.exists():
        raise FileNotFoundError(f"Missing strong subset: {strong_path}")

    out_dir = args.out_dir
    if out_dir is None:
        out_dir = DEFAULT_OUT_PARENT / f"m4_xdoc_materials_{utc_stamp()}"
    out_dir.mkdir(parents=True, exist_ok=True)

    strong_rows = read_jsonl(strong_path)
    pack_rows = {row.get("candidate_id"): row for row in read_jsonl(args.judge_pack.resolve())}

    materials: list[dict[str, Any]] = []
    missing_from_pack: list[str] = []
    for judged_row in strong_rows:
        cid = judged_row.get("candidate_id")
        pack_row = pack_rows.get(cid)
        if not pack_row:
            missing_from_pack.append(cid or "missing_candidate_id")
            continue
        materials.append(material_from_row(pack_row, judged_row.get("judgment") or {}))

    write_jsonl(out_dir / "m4_material_pack.jsonl", materials)

    status = "ok" if materials else "blocked_empty_strong_subset"
    summary = {
        "status": status,
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "subset_dir": str(subset_dir.relative_to(ROOT) if subset_dir.is_relative_to(ROOT) else subset_dir),
        "judge_pack": str(args.judge_pack.resolve().relative_to(ROOT)),
        "strong_rows": len(strong_rows),
        "materials": len(materials),
        "missing_from_pack": missing_from_pack,
        "decision": (
            "No cross-doc M4 material generated because high-precision subset is empty."
            if not materials else
            "Generated cross-doc M4 material pack; safe to pass to multiturn generator."
        ),
        "files": {
            "material_pack": str((out_dir / "m4_material_pack.jsonl").relative_to(ROOT)),
            "summary": str((out_dir / "summary.json").relative_to(ROOT)),
        },
    }
    (out_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    (out_dir / "summary.md").write_text(
        "\n".join([
            "# XDoc M4 Material Conversion",
            "",
            f"- status: **{status}**",
            f"- strong rows: **{len(strong_rows)}**",
            f"- materials: **{len(materials)}**",
            "",
            "## Decision",
            "",
            summary["decision"],
            "",
        ]),
        encoding="utf-8",
    )

    latest = DEFAULT_OUT_PARENT / "m4_xdoc_materials_latest"
    atomic_latest_symlink(out_dir, latest)

    print(f"Output: {out_dir}")
    print(f"Status: {status}")
    print(f"materials={len(materials)} from strong_rows={len(strong_rows)}")
    print(f"Latest: {latest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
