#!/usr/bin/env python3
"""Experimental cross-doc prompt dry-run.

Production ``scripts/generate_multihop_l1_queries.py`` intentionally filters all
cross-document pairs through ``filter_intra_doc_pairs()``. This script validates
the *new* cross-doc pairing idea without touching that production guard: it loads
schema-compatible cross-doc pairs and directly calls the existing prompt builder.

Outputs:
  data/05_eval/trinity_connectivity_smoke/crossdoc_prompt_dryrun.jsonl
  data/05_eval/trinity_connectivity_smoke/crossdoc_prompt_dryrun.md
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import scripts.generate_multihop_l1_queries as gen  # noqa: E402


DEFAULT_CANDIDATES = ROOT / "data/05_eval/trinity_connectivity_smoke/crossdoc_pairs_smoke20.json"
DEFAULT_OUT = ROOT / "data/05_eval/trinity_connectivity_smoke/crossdoc_prompt_dryrun.jsonl"
DEFAULT_TOPOLOGY = ROOT / "data/05_eval/mineru_topology_graph_v1_latest/mineru_topology_graph_v1.json"
ELEMENT_NODE_TYPES = {"figure", "table", "formula"}


GENERIC_CROSSDOC_TEMPLATE = """You are generating one cross-document academic retrieval query from two multimodal evidence elements.

The two elements are intentionally cross-document. They may also be same-modality (e.g. figure→figure), because the new cross-doc match layer is not the old cross-modal L1 pair layer.

Return a JSON object with exactly these fields:
- query: a natural paper-domain question that requires comparing or connecting both elements.
- answer: a concise grounded answer using both evidence elements.
- reasoning_chain: 2-3 sentences explaining why element A and element B must both be used.
- required_evidence_spans: a two-item list with one short quote/paraphrase from each element.
- qc_notes: one sentence about why this is not answerable from only one document.

Element A:
- id: {element_a_id}
- doc: {element_a_doc}
- type: {element_a_type}
- caption: {element_a_caption}
- context: {element_a_context}

Element B:
- id: {element_b_id}
- doc: {element_b_doc}
- type: {element_b_type}
- caption: {element_b_caption}
- context: {element_b_context}

Cross-doc match metadata:
- similarity_score: {score}
- pair_type: {pair_type}
"""


def load_topology_nodes(path: Path) -> dict[str, dict[str, Any]]:
    if not path.exists():
        return {}
    data = json.loads(path.read_text(encoding="utf-8"))
    return {str(node.get("node_id")): node for node in data.get("nodes", []) if node.get("node_id")}


def node_to_prompt_element(node: dict[str, Any]) -> dict[str, Any]:
    metadata = node.get("metadata") if isinstance(node.get("metadata"), dict) else {}
    context_parts = [
        node.get("text_snippet") or "",
        metadata.get("content_preview") or "",
        metadata.get("context_before") or "",
        metadata.get("context_after") or "",
    ]
    return {
        "element_id": node.get("element_id") or node.get("mapped_element_id") or node.get("node_id", ""),
        "node_id": node.get("node_id", ""),
        "doc_id": node.get("doc_id", ""),
        "element_type": node.get("node_type", ""),
        "caption": metadata.get("caption") or node.get("label") or "",
        "content": " ".join(str(part).strip() for part in context_parts if part and str(part).strip()),
        "context_before": metadata.get("context_before") or "",
        "context_after": metadata.get("context_after") or "",
    }


def candidate_to_pair(candidate: dict[str, Any], nodes: dict[str, dict[str, Any]]) -> dict[str, Any] | None:
    path_nodes = [nid for nid in candidate.get("path", []) if nid in nodes]
    if len(path_nodes) < 2:
        return None
    element_nodes = [nodes[nid] for nid in path_nodes if nodes[nid].get("node_type") in ELEMENT_NODE_TYPES]
    left = nodes[path_nodes[0]]
    right = nodes[path_nodes[-1]]
    if len(element_nodes) >= 2:
        left = element_nodes[0]
        right = element_nodes[-1]
        if candidate.get("is_cross_doc"):
            for candidate_right in element_nodes[1:]:
                if candidate_right.get("doc_id") != left.get("doc_id"):
                    right = candidate_right
                    break
    element_a = node_to_prompt_element(left)
    element_b = node_to_prompt_element(right)
    return {
        "pair_id": candidate.get("candidate_id", ""),
        "pair_type": "mineru_multihop_candidate",
        "path": candidate.get("path", []),
        "quality_score": candidate.get("score", 0.0),
        "element_a_id": element_a.get("element_id", ""),
        "element_b_id": element_b.get("element_id", ""),
        "element_a_type": element_a.get("element_type", ""),
        "element_b_type": element_b.get("element_type", ""),
        "element_a": element_a,
        "element_b": element_b,
        "hub_metadata": {
            "cross_doc_metadata": {
                "score": candidate.get("score", 0.0),
                "edge_types": candidate.get("edge_types", []),
                "hop_count": candidate.get("hop_count"),
                "is_cross_doc": candidate.get("is_cross_doc", False),
            }
        },
    }


def load_pairs(path: Path) -> list[dict[str, Any]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data.get("pairs"), list):
        return data.get("pairs", [])
    if isinstance(data.get("candidates"), list):
        topo_path = Path((data.get("metadata") or {}).get("source_topology") or DEFAULT_TOPOLOGY)
        if not topo_path.is_absolute():
            topo_path = ROOT / topo_path
        nodes = load_topology_nodes(topo_path)
        candidates = data.get("candidates", [])
        cross_doc_candidates = [candidate for candidate in candidates if candidate.get("is_cross_doc")]
        if cross_doc_candidates:
            candidates = cross_doc_candidates
        pairs = [candidate_to_pair(candidate, nodes) for candidate in candidates]
        return [pair for pair in pairs if pair]
    return []


def compact_context(element: dict[str, Any], max_chars: int = 800) -> str:
    parts = [
        element.get("enriched_content", "") or "",
        element.get("content", "") or "",
        element.get("context_before", "") or "",
        element.get("context_after", "") or "",
    ]
    text = " ".join(part.strip() for part in parts if part and part.strip())
    return text[:max_chars] if text else "(no context)"


def build_generic_crossdoc_prompt(pair: dict[str, Any]) -> str:
    element_a = pair.get("element_a", {}) or {}
    element_b = pair.get("element_b", {}) or {}
    metadata = ((pair.get("hub_metadata") or {}).get("cross_doc_metadata") or {})
    return GENERIC_CROSSDOC_TEMPLATE.format(
        element_a_id=pair.get("element_a_id", element_a.get("element_id", "")),
        element_a_doc=element_a.get("doc_id", ""),
        element_a_type=pair.get("element_a_type", element_a.get("element_type", "")),
        element_a_caption=(element_a.get("caption", "") or "")[:500],
        element_a_context=compact_context(element_a),
        element_b_id=pair.get("element_b_id", element_b.get("element_id", "")),
        element_b_doc=element_b.get("doc_id", ""),
        element_b_type=pair.get("element_b_type", element_b.get("element_type", "")),
        element_b_caption=(element_b.get("caption", "") or "")[:500],
        element_b_context=compact_context(element_b),
        score=metadata.get("score", pair.get("quality_score", "")),
        pair_type=pair.get("pair_type", ""),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Experimental cross-doc prompt dry-run")
    parser.add_argument("--candidates", default=str(DEFAULT_CANDIDATES))
    parser.add_argument("--output", default=str(DEFAULT_OUT))
    parser.add_argument("--limit", type=int, default=5)
    parser.add_argument("--query-style", default="academic", choices=["academic", "real_user", "mixed"])
    args = parser.parse_args()

    candidates = Path(args.candidates)
    output = Path(args.output)
    if not candidates.is_absolute():
        candidates = ROOT / candidates
    if not output.is_absolute():
        output = ROOT / output
    pairs = load_pairs(candidates)[: args.limit]

    output.parent.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    failures: list[dict[str, str]] = []
    for pair in pairs:
        try:
            render_mode = "legacy_build_prompt"
            try:
                prompt = gen.build_prompt(pair, query_style=args.query_style, use_persona=False)
                if not prompt:
                    raise ValueError("legacy build_prompt returned empty prompt")
            except Exception:
                prompt = build_generic_crossdoc_prompt(pair)
                render_mode = "generic_crossdoc_fallback"
            rows.append(
                {
                    "pair_id": pair.get("pair_id"),
                    "pair_type": pair.get("pair_type"),
                    "path": pair.get("path"),
                    "render_mode": render_mode,
                    "prompt_chars": len(prompt),
                    "prompt_head": prompt[:2000],
                }
            )
        except Exception as exc:  # noqa: BLE001 - experimental diagnostic
            failures.append({"pair_id": str(pair.get("pair_id")), "error": repr(exc)})

    with output.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    md = output.with_suffix(".md")
    lines = [
        "# Cross-doc Prompt Dry-run",
        "",
        "Scope: experimental bypass of the intentional production intra-doc filter.",
        "",
        f"- candidates: `{candidates.relative_to(ROOT)}`",
        f"- requested limit: `{args.limit}`",
        f"- rendered prompts: **{len(rows)}**",
        f"- failures: **{len(failures)}**",
        "",
    ]
    if failures:
        lines.append("## Failures")
        for failure in failures:
            lines.append(f"- `{failure['pair_id']}`: `{failure['error']}`")
        lines.append("")
    lines.append("## Rendered Prompt Samples")
    for row in rows[:3]:
        lines.extend(
            [
                "",
                f"### {row['pair_id']} ({row['pair_type']})",
                f"- path: `{row['path']}`",
                f"- render mode: `{row['render_mode']}`",
                f"- prompt chars: `{row['prompt_chars']}`",
                "",
                "```text",
                row["prompt_head"],
                "```",
            ]
        )
    md.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"[ok] rendered={len(rows)} failures={len(failures)}")
    print(f"[ok] wrote {output}")
    print(f"[ok] wrote {md}")
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
