#!/usr/bin/env python3
"""Build experimental PDF-first cross-doc + cross-modal chain candidates.

Direct archive cross-doc matches are same-modality only (figure→figure,
table→table, formula→formula). To satisfy both axes, treat the direct match as a
cross-document semantic bridge, then attach a nearby cross-modal element from the
source or target document using PDF/MinerU layout/context fields.

No production code or production artifacts are written.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
CROSSDOC = ROOT / "archive/data_legacy/embedding_probes/mineru_crossdoc_embedding_matches_Qwen3-Embedding-4B_v2b_cap10.jsonl"
ELEMENTS = ROOT / "data/01_graphs/multimodal_elements.json"
MODAL_TYPES = {"figure", "table", "formula"}


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def richness(element: dict[str, Any]) -> int:
    fields = ["caption", "content", "context_before", "context_after"]
    return sum(len((element.get(field) or "").strip()) for field in fields)


def detail(element: dict[str, Any]) -> dict[str, Any]:
    return {
        "element_id": element.get("element_id", ""),
        "doc_id": element.get("doc_id", ""),
        "element_type": element.get("element_type", ""),
        "caption": element.get("caption", "") or "",
        "content": element.get("content", "") or "",
        "image_path": element.get("image_path", "") or "",
        "page_idx": element.get("page_idx"),
        "position_idx": element.get("position_idx"),
        "context_before": element.get("context_before", "") or "",
        "context_after": element.get("context_after", "") or "",
        "richness": richness(element),
    }


def layout_distance(anchor: dict[str, Any], other: dict[str, Any]) -> float:
    ap = anchor.get("page_idx")
    bp = other.get("page_idx")
    ai = anchor.get("position_idx")
    bi = other.get("position_idx")
    page_dist = abs((ap or 0) - (bp or 0)) if ap is not None and bp is not None else 10
    pos_dist = abs((ai or 0) - (bi or 0)) if ai is not None and bi is not None else 10
    return page_dist * 10 + pos_dist


def load_elements() -> tuple[dict[str, dict[str, Any]], dict[str, list[dict[str, Any]]]]:
    data = json.loads(ELEMENTS.read_text(encoding="utf-8"))
    index: dict[str, dict[str, Any]] = {}
    by_doc: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for doc_id, doc in data.get("documents", {}).items():
        for eid, element in doc.get("elements", {}).items():
            index[eid] = element
            by_doc[doc_id].append(element)
    return index, by_doc


def best_crossmodal_neighbor(anchor: dict[str, Any], by_doc: dict[str, list[dict[str, Any]]]) -> dict[str, Any] | None:
    anchor_type = anchor.get("element_type")
    doc_id = anchor.get("doc_id")
    candidates = [
        element for element in by_doc.get(doc_id, [])
        if element.get("element_id") != anchor.get("element_id")
        and element.get("element_type") in MODAL_TYPES
        and element.get("element_type") != anchor_type
        and richness(element) >= 80
    ]
    if not candidates:
        return None
    candidates.sort(key=lambda element: (layout_distance(anchor, element), -richness(element)))
    return candidates[0]


def build_candidates(limit_per_base_type: int) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    element_index, by_doc = load_elements()
    rows = load_jsonl(CROSSDOC)
    direct_counts: Counter[str] = Counter()
    chain_counts: Counter[str] = Counter()
    selected_counts: Counter[str] = Counter()
    used_source_docs: set[tuple[str, str]] = set()
    selected: list[dict[str, Any]] = []

    for row in rows:
        source_id = row.get("source_element_id", "")
        source = element_index.get(source_id)
        if not source:
            continue
        source_type = row.get("source_type", source.get("element_type", ""))
        matches = sorted(row.get("matches", []), key=lambda match: float(match.get("score", 0) or 0), reverse=True)
        for match in matches:
            target_id = match.get("target_element_id", "")
            target = element_index.get(target_id)
            if not target:
                continue
            target_type = match.get("target_type", target.get("element_type", ""))
            base_type = f"{source_type}->{target_type}"
            direct_counts[base_type] += 1
            if row.get("source_doc_id") == match.get("target_doc_id"):
                continue
            if source_type != target_type:
                continue

            expansions = []
            source_neighbor = best_crossmodal_neighbor(source, by_doc)
            target_neighbor = best_crossmodal_neighbor(target, by_doc)
            if source_neighbor:
                expansions.append(("source_side", source_neighbor))
            if target_neighbor:
                expansions.append(("target_side", target_neighbor))

            for expansion_side, neighbor in expansions:
                companion_type = neighbor.get("element_type", "")
                chain_shape = f"{base_type}+{expansion_side}:{companion_type}"
                chain_counts[chain_shape] += 1
                source_doc_key = (base_type, row.get("source_doc_id", ""))
                if selected_counts[base_type] >= limit_per_base_type:
                    continue
                if source_doc_key in used_source_docs:
                    continue
                pair_id = f"pdf_xdoc_xmodal_{len(selected)+1:04d}"
                selected.append(
                    {
                        "candidate_id": pair_id,
                        "base_pair_type": base_type,
                        "chain_shape": chain_shape,
                        "cross_doc_bridge": [source_id, target_id],
                        "cross_modal_attachment_side": expansion_side,
                        "path": [source_id, target_id, neighbor.get("element_id")]
                        if expansion_side == "target_side"
                        else [neighbor.get("element_id"), source_id, target_id],
                        "source": detail(source),
                        "target": detail(target),
                        "cross_modal_neighbor": detail(neighbor),
                        "cross_doc_metadata": {
                            "source_doc_id": row.get("source_doc_id"),
                            "target_doc_id": match.get("target_doc_id"),
                            "score": match.get("score"),
                            "model": row.get("model"),
                        },
                        "pdf_first_metadata": {
                            "layout_distance": layout_distance(target if expansion_side == "target_side" else source, neighbor),
                            "neighbor_richness": richness(neighbor),
                            "source_richness": richness(source),
                            "target_richness": richness(target),
                        },
                    }
                )
                selected_counts[base_type] += 1
                used_source_docs.add(source_doc_key)
                break
    summary = {
        "direct_counts": dict(direct_counts),
        "chain_shape_top20": dict(chain_counts.most_common(20)),
        "selected_counts": dict(selected_counts),
        "selected_total": len(selected),
    }
    return selected, summary


def write_report(out_dir: Path, candidates: list[dict[str, Any]], summary: dict[str, Any]) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "xdoc_xmodal_candidates.json").write_text(
        json.dumps({"metadata": {"scope": "experimental_pdf_first", "source": str(CROSSDOC.relative_to(ROOT))}, "candidates": candidates}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    lines = [
        "# PDF-first Cross-doc + Cross-modal Candidate Design",
        "",
        "Direct Qwen3 cross-doc matches are same-modality only. This experiment composes:",
        "",
        "1. a cross-document same-modality semantic bridge (`source → target`);",
        "2. a PDF/MinerU local cross-modal neighbor attached on source or target side;",
        "3. a 3-node chain that is both cross-document and cross-modal.",
        "",
        "## Counts",
        f"- direct pair types: `{summary['direct_counts']}`",
        f"- selected counts: `{summary['selected_counts']}`",
        f"- selected total: **{summary['selected_total']}**",
        f"- top chain shapes: `{summary['chain_shape_top20']}`",
        "",
        "## Sample Candidates",
    ]
    for cand in candidates[:12]:
        lines.extend([
            "",
            f"### {cand['candidate_id']} — {cand['chain_shape']}",
            f"- path: `{cand['path']}`",
            f"- score: `{cand['cross_doc_metadata']['score']}`",
            f"- source: `{cand['source']['element_id']}` ({cand['source']['element_type']})",
            f"- target: `{cand['target']['element_id']}` ({cand['target']['element_type']})",
            f"- neighbor: `{cand['cross_modal_neighbor']['element_id']}` ({cand['cross_modal_neighbor']['element_type']})",
            f"- source caption: {cand['source']['caption'][:220]}",
            f"- target caption: {cand['target']['caption'][:220]}",
            f"- neighbor caption: {cand['cross_modal_neighbor']['caption'][:220]}",
        ])
    lines.extend([
        "",
        "## Design Decision",
        "",
        "This should remain experimental. The old production cross-doc entry is intentionally blocked. The new route needs a dedicated prompt family for `same-modality cross-doc bridge + local cross-modal attachment`, not the old L1 cross-modal pair templates.",
    ])
    (out_dir / "design.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build PDF-first cross-doc + cross-modal candidates")
    parser.add_argument("--output-dir", default="")
    parser.add_argument("--limit-per-base-type", type=int, default=4)
    args = parser.parse_args()

    if args.output_dir:
        out_dir = Path(args.output_dir)
        if not out_dir.is_absolute():
            out_dir = ROOT / out_dir
    else:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        out_dir = ROOT / f"data/05_eval/pdf_first_xdoc_xmodal_design_{stamp}"
    candidates, summary = build_candidates(args.limit_per_base_type)
    write_report(out_dir, candidates, summary)
    latest = ROOT / "data/05_eval/pdf_first_xdoc_xmodal_design_latest"
    try:
        if latest.is_symlink() or latest.exists():
            latest.unlink()
        latest.symlink_to(out_dir.resolve())
    except OSError:
        pass
    print(f"[ok] selected {len(candidates)} candidates")
    print(f"[ok] wrote {out_dir / 'design.md'}")


if __name__ == "__main__":
    main()