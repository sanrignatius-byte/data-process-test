#!/usr/bin/env python3
"""Prompt dry-run for PDF-first cross-doc + cross-modal 3-node chains."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_IN = ROOT / "data/05_eval/pdf_first_xdoc_xmodal_design_latest/xdoc_xmodal_candidates.json"
DEFAULT_OUT = ROOT / "data/05_eval/pdf_first_xdoc_xmodal_design_latest/xdoc_xmodal_prompt_dryrun.jsonl"


TEMPLATE = """You are generating one PDF-grounded M4 query from a 3-node evidence chain.

The chain must test BOTH axes:
1. cross-document: the bridge connects element A in one paper to element B in another paper;
2. cross-modal: element C is a nearby PDF/MinerU element of a different modality in the same local document as A or B.

Return only valid JSON with exactly these fields:
- query: a natural paper-domain question that cannot be answered from one document or one modality alone.
- answer: a concise grounded answer using A, B, and C.
- reasoning_chain: 3-4 sentences; explicitly say how A→B is cross-document and how C adds cross-modal evidence.
- required_evidence_spans: exactly three strings, one for A, one for B, one for C.
- chain_roles: an object with keys cross_doc_source, cross_doc_target, cross_modal_companion.
- qc_notes: explain why removing B or removing C would make the answer incomplete.

Cross-document semantic bridge:
Element A (source):
- id: {source_id}
- doc: {source_doc}
- type: {source_type}
- caption: {source_caption}
- context/content: {source_context}

Element B (target):
- id: {target_id}
- doc: {target_doc}
- type: {target_type}
- caption: {target_caption}
- context/content: {target_context}

Local cross-modal companion:
Element C:
- id: {neighbor_id}
- doc: {neighbor_doc}
- type: {neighbor_type}
- attached_side: {attachment_side}
- caption: {neighbor_caption}
- context/content: {neighbor_context}

Metadata:
- chain_shape: {chain_shape}
- cross_doc_score: {score}
- PDF-first rule: use PDF/MinerU caption/context/layout evidence; do not rely on LaTeX ref labels.
"""


def load_candidates(path: Path) -> list[dict[str, Any]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    return data.get("candidates", [])


def context(element: dict[str, Any], max_chars: int = 900) -> str:
    parts = [
        element.get("content", "") or "",
        element.get("context_before", "") or "",
        element.get("context_after", "") or "",
    ]
    text = " ".join(part.strip() for part in parts if part and part.strip())
    return text[:max_chars] if text else "(no context)"


def build_prompt(candidate: dict[str, Any]) -> str:
    source = candidate["source"]
    target = candidate["target"]
    neighbor = candidate["cross_modal_neighbor"]
    meta = candidate["cross_doc_metadata"]
    return TEMPLATE.format(
        source_id=source["element_id"],
        source_doc=source["doc_id"],
        source_type=source["element_type"],
        source_caption=(source.get("caption") or "")[:500],
        source_context=context(source),
        target_id=target["element_id"],
        target_doc=target["doc_id"],
        target_type=target["element_type"],
        target_caption=(target.get("caption") or "")[:500],
        target_context=context(target),
        neighbor_id=neighbor["element_id"],
        neighbor_doc=neighbor["doc_id"],
        neighbor_type=neighbor["element_type"],
        attachment_side=candidate["cross_modal_attachment_side"],
        neighbor_caption=(neighbor.get("caption") or "")[:500],
        neighbor_context=context(neighbor),
        chain_shape=candidate["chain_shape"],
        score=meta.get("score"),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Dry-run prompts for xdoc+xmodal chains")
    parser.add_argument("--candidates", default=str(DEFAULT_IN))
    parser.add_argument("--output", default=str(DEFAULT_OUT))
    parser.add_argument("--limit", type=int, default=6)
    args = parser.parse_args()

    input_path = Path(args.candidates)
    output_path = Path(args.output)
    if not input_path.is_absolute():
        input_path = ROOT / input_path
    if not output_path.is_absolute():
        output_path = ROOT / output_path
    candidates = load_candidates(input_path)[: args.limit]
    output_path.parent.mkdir(parents=True, exist_ok=True)

    rows = []
    for candidate in candidates:
        prompt = build_prompt(candidate)
        rows.append(
            {
                "candidate_id": candidate["candidate_id"],
                "chain_shape": candidate["chain_shape"],
                "path": candidate["path"],
                "prompt_chars": len(prompt),
                "prompt": prompt,
            }
        )
    with output_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    md_path = output_path.with_suffix(".md")
    lines = ["# XDoc + XModal Prompt Dry-run", "", f"- rendered: **{len(rows)}**", ""]
    for row in rows[:3]:
        lines.extend([
            f"## {row['candidate_id']} — {row['chain_shape']}",
            f"- path: `{row['path']}`",
            f"- prompt chars: `{row['prompt_chars']}`",
            "",
            "```text",
            row["prompt"][:2500],
            "```",
            "",
        ])
    md_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"[ok] rendered {len(rows)} prompts")
    print(f"[ok] wrote {output_path}")
    print(f"[ok] wrote {md_path}")


if __name__ == "__main__":
    main()