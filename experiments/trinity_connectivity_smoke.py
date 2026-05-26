#!/usr/bin/env python3
"""Experimental connectivity smoke test for the 2026-05-19 M4 trinity plans.

This is deliberately kept under ``experiments/`` rather than ``src/`` because it
only validates whether the proposed experiment inputs connect. It does not write
production artifacts or modify any production pipeline.

Outputs:
  data/05_eval/trinity_connectivity_smoke/report.md
  data/05_eval/trinity_connectivity_smoke/summary.json
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "data/05_eval/trinity_connectivity_smoke"

CROSSDOC = ROOT / "archive/data_legacy/embedding_probes/mineru_crossdoc_embedding_matches_Qwen3-Embedding-4B_v2b_cap10.jsonl"
ELEMENTS = ROOT / "data/01_graphs/multimodal_elements.json"
CITATION = ROOT / "data/01_graphs/citation_graph.json"
L3_FILES = [
    ROOT / "data/03_queries/l3_enriched_v3_rerun2_pass.jsonl",
    ROOT / "data/03_queries/l3_enriched_v3_new82_rerun2_pass.jsonl",
]

MODAL_TYPES = {"figure", "table", "formula"}


def element_detail(element: dict[str, Any]) -> dict[str, Any]:
    return {
        "element_id": element.get("element_id", ""),
        "doc_id": element.get("doc_id", ""),
        "element_type": element.get("element_type", ""),
        "caption": element.get("caption", "") or "",
        "content": element.get("content", "") or "",
        "image_path": element.get("image_path", "") or "",
        "context_before": element.get("context_before", "") or "",
        "context_after": element.get("context_after", "") or "",
        "enriched_title": element.get("enriched_title", "") or "",
        "enriched_content": element.get("enriched_content", "") or "",
        "enriched_metadata": element.get("enriched_metadata", {}) or {},
    }


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def doc_of(node_id: str) -> str:
    if "::" in node_id:
        return node_id.split("::", 1)[0]
    return node_id.split("_", 1)[0]


def node_kind(node_id: str) -> str:
    if "::p::" in node_id:
        return "paragraph"
    if "_figure_" in node_id:
        return "figure"
    if "_table_" in node_id:
        return "table"
    if "_formula_" in node_id:
        return "formula"
    return "other"


def load_element_index(path: Path) -> dict[str, dict[str, Any]]:
    data = json.load(path.open(encoding="utf-8"))
    index: dict[str, dict[str, Any]] = {}
    for doc in data.get("documents", {}).values():
        for element_id, element in doc.get("elements", {}).items():
            index[element_id] = element
    return index


def audit_crossdoc(element_index: dict[str, dict[str, Any]]) -> dict[str, Any]:
    rows = read_jsonl(CROSSDOC)
    total_matches = 0
    cross_doc_matches = 0
    modal_matches = 0
    source_covered = 0
    target_covered = 0
    score_values: list[float] = []
    source_type_counts: Counter[str] = Counter()
    pair_type_counts: Counter[str] = Counter()
    source_to_targets: dict[str, list[str]] = {}

    for row in rows:
        source_id = row.get("source_element_id", "")
        source_type = row.get("source_type", "")
        source_type_counts[source_type] += 1
        if source_id in element_index:
            source_covered += 1
        targets: list[str] = []
        for rank, match in enumerate(row.get("matches", []), 1):
            target_id = match.get("target_element_id", "")
            target_type = match.get("target_type", "")
            target_doc = match.get("target_doc_id", "")
            total_matches += 1
            targets.append(target_id)
            if target_id in element_index:
                target_covered += 1
            if target_doc and target_doc != row.get("source_doc_id"):
                cross_doc_matches += 1
            if source_type in MODAL_TYPES or target_type in MODAL_TYPES:
                modal_matches += 1
            pair_type_counts[f"{source_type}->{target_type}"] += 1
            if isinstance(match.get("score"), (int, float)):
                score_values.append(float(match["score"]))
        source_to_targets[source_id] = targets

    reciprocal_top10 = 0
    reciprocal_any = 0
    comparable = 0
    for source_id, targets in source_to_targets.items():
        for target_id in targets:
            if target_id in source_to_targets:
                comparable += 1
                back = source_to_targets[target_id]
                if source_id in back:
                    reciprocal_any += 1
                    if back.index(source_id) < 10:
                        reciprocal_top10 += 1

    return {
        "rows": len(rows),
        "total_matches": total_matches,
        "cross_doc_matches": cross_doc_matches,
        "modal_matches": modal_matches,
        "source_coverage_in_multimodal_elements": source_covered,
        "target_coverage_in_multimodal_elements": target_covered,
        "source_type_counts": dict(source_type_counts),
        "top_pair_type_counts": dict(pair_type_counts.most_common(10)),
        "score_min": min(score_values) if score_values else None,
        "score_max": max(score_values) if score_values else None,
        "score_mean": sum(score_values) / len(score_values) if score_values else None,
        "reciprocal_comparable_matches": comparable,
        "reciprocal_any": reciprocal_any,
        "reciprocal_top10": reciprocal_top10,
    }


def audit_l3() -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for path in L3_FILES:
        rows.extend(read_jsonl(path))

    path_len_counts: Counter[int] = Counter()
    path_shape_counts: Counter[tuple[str, ...]] = Counter()
    reasoning_steps_len_counts: Counter[int] = Counter()
    element_id_len_counts: Counter[int] = Counter()
    hop_distance_counts: Counter[str] = Counter()
    endpoint_type_counts: Counter[str] = Counter()
    cross_doc_count = 0
    paragraph_bridge_count = 0
    candidate_2turn_sessions = 0
    sample_cross_doc: dict[str, Any] | None = None

    for row in rows:
        path = row.get("path") or []
        element_ids = row.get("element_ids") or []
        path_len_counts[len(path)] += 1
        path_shape_counts[tuple(node_kind(node_id) for node_id in path)] += 1
        reasoning_steps_len_counts[len(row.get("reasoning_steps") or [])] += 1
        element_id_len_counts[len(element_ids)] += 1
        hop_distance_counts[str(row.get("hop_distance"))] += 1
        endpoint_type_counts[f"{row.get('element_a_type')}->{row.get('element_b_type')}"] += 1
        if len({doc_of(node_id) for node_id in path}) > 1:
            cross_doc_count += 1
            if sample_cross_doc is None:
                sample_cross_doc = {
                    "query_id": row.get("query_id"),
                    "query": row.get("query"),
                    "path": path,
                    "element_ids": element_ids,
                    "reasoning_chain_head": (row.get("reasoning_chain") or "")[:500],
                }
        if any("::p::" in node_id for node_id in path):
            paragraph_bridge_count += 1
        if len(element_ids) == 2 and len(path) in {3, 4} and row.get("reasoning_chain"):
            candidate_2turn_sessions += 1

    return {
        "rows": len(rows),
        "path_len_counts": dict(path_len_counts),
        "path_shape_top10": {" → ".join(k): v for k, v in path_shape_counts.most_common(10)},
        "reasoning_steps_len_counts": dict(reasoning_steps_len_counts),
        "element_id_len_counts": dict(element_id_len_counts),
        "hop_distance_counts": dict(hop_distance_counts),
        "endpoint_type_top10": dict(endpoint_type_counts.most_common(10)),
        "cross_doc_count": cross_doc_count,
        "paragraph_bridge_count": paragraph_bridge_count,
        "candidate_2turn_sessions": candidate_2turn_sessions,
        "sample_cross_doc": sample_cross_doc,
    }


def write_report(summary: dict[str, Any]) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    crossdoc = summary["crossdoc"]
    l3 = summary["l3"]
    lines = [
        "# Trinity Connectivity Smoke Report",
        "",
        "Scope: experimental-only connectivity check. No production code or production artifacts were written.",
        "",
        "## Inputs",
        f"- crossdoc matches: `{CROSSDOC.relative_to(ROOT)}`",
        f"- multimodal elements: `{ELEMENTS.relative_to(ROOT)}`",
        f"- citation graph: `{CITATION.relative_to(ROOT)}`",
        "- L3 pass files:",
    ]
    for path in L3_FILES:
        lines.append(f"  - `{path.relative_to(ROOT)}`")
    lines += [
        "",
        "## Cross-doc Pairing Connectivity",
        f"- rows: **{crossdoc['rows']}**",
        f"- matches: **{crossdoc['total_matches']}**",
        f"- cross-doc matches: **{crossdoc['cross_doc_matches']}**",
        f"- modal-constrained matches: **{crossdoc['modal_matches']}**",
        f"- source coverage in `multimodal_elements`: **{crossdoc['source_coverage_in_multimodal_elements']} / {crossdoc['rows']}**",
        f"- target coverage in `multimodal_elements`: **{crossdoc['target_coverage_in_multimodal_elements']} / {crossdoc['total_matches']}**",
        f"- score range / mean: **{crossdoc['score_min']:.4f} – {crossdoc['score_max']:.4f} / {crossdoc['score_mean']:.4f}**",
        f"- reciprocal comparable matches: **{crossdoc['reciprocal_comparable_matches']}**; reciprocal_any=**{crossdoc['reciprocal_any']}**, reciprocal_top10=**{crossdoc['reciprocal_top10']}**",
        f"- source type counts: `{crossdoc['source_type_counts']}`",
        f"- top pair types: `{crossdoc['top_pair_type_counts']}`",
        "",
        "## Chain-to-Session Connectivity",
        f"- L3 pass rows: **{l3['rows']}**",
        f"- path length distribution: `{l3['path_len_counts']}`",
        f"- reasoning_steps length distribution: `{l3['reasoning_steps_len_counts']}`",
        f"- element_ids length distribution: `{l3['element_id_len_counts']}`",
        f"- hop_distance distribution: `{l3['hop_distance_counts']}`",
        f"- cross-doc chains by path: **{l3['cross_doc_count']} / {l3['rows']}**",
        f"- paragraph-bridge chains: **{l3['paragraph_bridge_count']} / {l3['rows']}**",
        f"- v1-compatible 2-turn candidates: **{l3['candidate_2turn_sessions']} / {l3['rows']}**",
        f"- top path shapes: `{l3['path_shape_top10']}`",
        "",
        "## Start Verdict",
        "- `exp:20260519_xdoc_pairing_module`: **connects at input level**. The path must use the archive location or a trial-stage symlink/copy; do not write a production `src/` module until the filter is validated under `experiments/`.",
        "- `exp:20260519_chain_to_session`: **connects only after design calibration**. Current data supports a 2-turn endpoint→endpoint projection plus a verbalization step; the original `reasoning_steps[]`-based 3-turn rule is not executable because all 146 rows have empty `reasoning_steps`.",
        "- `idea:007` trinity benchmark: **defer until the 2-turn projection smoke passes**; otherwise view-factorization metrics will be testing a broken projection rather than M4 difficulty.",
        "- Existing `scripts/generate_multihop_l1_queries.py` is **not directly connected** for cross-doc pairs: it unconditionally applies `filter_intra_doc_pairs()`, so a schema-compatible cross-doc pair file is still dropped before prompt rendering. This confirms the blue-team warning: validate in the experimental layer first, then promote a guarded production change later.",
    ]

    sample = l3.get("sample_cross_doc")
    if sample:
        lines += [
            "",
            "## Sample Cross-doc L3 Row",
            f"- query_id: `{sample.get('query_id')}`",
            f"- path: `{sample.get('path')}`",
            f"- element_ids: `{sample.get('element_ids')}`",
            f"- query: {sample.get('query')}",
            f"- reasoning_chain head: {sample.get('reasoning_chain_head')}",
        ]

    (OUT_DIR / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_crossdoc_pair_sample(element_index: dict[str, dict[str, Any]], limit: int = 20) -> Path:
    rows = read_jsonl(CROSSDOC)
    pairs: list[dict[str, Any]] = []
    for row in rows:
        source_id = row.get("source_element_id", "")
        source = element_index.get(source_id)
        if not source:
            continue
        for match in row.get("matches", []):
            target_id = match.get("target_element_id", "")
            target = element_index.get(target_id)
            if not target:
                continue
            if row.get("source_doc_id") == match.get("target_doc_id"):
                continue
            source_type = row.get("source_type", source.get("element_type", ""))
            target_type = match.get("target_type", target.get("element_type", ""))
            pair_id = f"xdoc_smoke_{len(pairs)+1:04d}"
            pairs.append(
                {
                    "pair_id": pair_id,
                    "doc_id": row.get("source_doc_id", ""),
                    "element_a_id": source_id,
                    "element_b_id": target_id,
                    "element_a_type": source_type,
                    "element_b_type": target_type,
                    "pair_type": "+".join(sorted([source_type, target_type])),
                    "hop_distance": 1,
                    "path": [source_id, target_id],
                    "quality_score": float(match.get("score", 0.0) or 0.0),
                    "is_cross_doc": True,
                    "element_a": element_detail(source),
                    "element_b": element_detail(target),
                    "node_group": [element_detail(source), element_detail(target)],
                    "edge_contexts": [],
                    "hub_semantic_summary": "",
                    "hub_metadata": {
                        "is_cross_doc": True,
                        "source": "trinity_connectivity_smoke",
                        "cross_doc_metadata": {
                            "source_doc_id": row.get("source_doc_id", ""),
                            "target_doc_id": match.get("target_doc_id", ""),
                            "score": match.get("score"),
                            "model": row.get("model", ""),
                        },
                    },
                }
            )
            if len(pairs) >= limit:
                out = OUT_DIR / "crossdoc_pairs_smoke20.json"
                out.write_text(
                    json.dumps(
                        {
                            "metadata": {
                                "source": str(CROSSDOC.relative_to(ROOT)),
                                "scope": "experimental_smoke_only",
                                "note": "Do not use as production candidate pairs.",
                            },
                            "pairs": pairs,
                        },
                        ensure_ascii=False,
                        indent=2,
                    ),
                    encoding="utf-8",
                )
                return out
    raise RuntimeError("Could not build any cross-doc pair sample")


def main() -> None:
    missing = [path for path in [CROSSDOC, ELEMENTS, CITATION, *L3_FILES] if not path.exists()]
    if missing:
        raise SystemExit("Missing inputs: " + ", ".join(str(path) for path in missing))
    element_index = load_element_index(ELEMENTS)
    summary = {
        "element_index_size": len(element_index),
        "crossdoc": audit_crossdoc(element_index),
        "l3": audit_l3(),
    }
    write_report(summary)
    sample_path = write_crossdoc_pair_sample(element_index)
    print(f"[ok] wrote {OUT_DIR / 'report.md'}")
    print(f"[ok] wrote {OUT_DIR / 'summary.json'}")
    print(f"[ok] wrote {sample_path}")


if __name__ == "__main__":
    main()