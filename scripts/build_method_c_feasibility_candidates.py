#!/usr/bin/env python3
"""Build feasibility-test candidates for Method C from long-chain paths.

This script converts `long_chain_candidates_v2.json` into the enriched-pair
schema consumed by `scripts/pilot_method_c.py`.

Intended use:
  - use a stable snapshot of partially enriched elements
  - keep long-chain topology from the v2 graph outputs
  - test the revised Method C prompt/logic before the full scale-up enrich finishes
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.analyze_latex_graph_topology import build_multimodal_index, map_label_to_element
from src.utils.text_utils import normalize_label_type


MODAL_TYPES = {"figure", "table", "equation"}


def _load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _clean_text(text: Any) -> str:
    return " ".join(str(text or "").split()).strip()


def _build_label_ordinals(doc: Dict[str, Any]) -> Dict[str, int]:
    labels = doc.get("labels", {}) or {}
    counters = {"figure": 0, "table": 0, "formula": 0}
    out: Dict[str, int] = {}
    sorted_labels = sorted(
        labels.items(),
        key=lambda kv: (
            int(kv[1].get("line_no", 0)) if isinstance(kv[1].get("line_no"), int) else 0,
            kv[0],
        ),
    )
    for label_key, info in sorted_labels:
        modal = normalize_label_type(str(info.get("label_type", "")))
        if modal not in MODAL_TYPES:
            continue
        mm_type = "formula" if modal == "equation" else modal
        out[label_key] = counters[mm_type]
        counters[mm_type] += 1
    return out


def _is_enriched(elem: Dict[str, Any]) -> bool:
    return bool(_clean_text(elem.get("enriched_title")) or _clean_text(elem.get("enriched_content")))


def _edge_lookup_key(source_label: str, target_label: str, ref_text: str = "") -> Tuple[str, str, str]:
    return source_label, target_label, ref_text


def _build_edge_lookup(doc: Dict[str, Any]) -> Dict[Tuple[str, str, str], Dict[str, Any]]:
    lookup: Dict[Tuple[str, str, str], Dict[str, Any]] = {}
    for edge in doc.get("edges", []) or []:
        src = str(edge.get("source_label", ""))
        tgt = str(edge.get("target_label", ""))
        ref_text = str(edge.get("ref_text", "") or "")
        if not src or not tgt:
            continue
        lookup.setdefault(_edge_lookup_key(src, tgt, ref_text), edge)
        lookup.setdefault(_edge_lookup_key(tgt, src, ref_text), edge)
        lookup.setdefault(_edge_lookup_key(src, tgt, ""), edge)
        lookup.setdefault(_edge_lookup_key(tgt, src, ""), edge)
    return lookup


def _resolve_edge_context(
    cand_edge: Dict[str, Any],
    edge_lookup: Dict[Tuple[str, str, str], Dict[str, Any]],
) -> Dict[str, Any]:
    src = str(cand_edge.get("source_label", ""))
    tgt = str(cand_edge.get("target_label", ""))
    ref_text = str(cand_edge.get("ref_text", "") or "")
    matched = (
        edge_lookup.get(_edge_lookup_key(src, tgt, ref_text))
        or edge_lookup.get(_edge_lookup_key(src, tgt, ""))
        or {}
    )
    return {
        **cand_edge,
        "context": matched.get("context", ""),
        "is_containment": bool(
            cand_edge.get("is_containment")
            or matched.get("is_containment")
            or matched.get("ref_text") == "[containment]"
        ),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--long-chain-candidates", default="data/01_graphs/long_chain_candidates_v2.json")
    ap.add_argument("--reference-graph", default="data/01_graphs/latex_reference_graph_v2.json")
    ap.add_argument(
        "--elements",
        default="data/02_enriched/method_c_long_chain_elements_enriched_snapshot_2026-04-12T0316.json",
        help="Partially or fully enriched multimodal elements JSON",
    )
    ap.add_argument("--mineru-output-dir", default="data/00_raw/mineru_output")
    ap.add_argument("--min-hop", type=int, default=4)
    ap.add_argument("--max-hop", type=int, default=5)
    ap.add_argument("--max-candidates", type=int, default=0, help="0 = all eligible")
    ap.add_argument("--require-enriched-endpoints", action="store_true", default=True)
    ap.add_argument(
        "--allow-unmapped-internal-modal",
        action="store_true",
        help="If set, internal unmapped modal labels are kept as bridge-like placeholders",
    )
    ap.add_argument(
        "--output",
        default="data/03_queries/method_c_feasibility_candidates_partial_enrich.json",
    )
    args = ap.parse_args()

    lc_path = PROJECT_ROOT / args.long_chain_candidates
    ref_path = PROJECT_ROOT / args.reference_graph
    elements_path = PROJECT_ROOT / args.elements
    mineru_dir = PROJECT_ROOT / args.mineru_output_dir
    out_path = PROJECT_ROOT / args.output

    print("Loading inputs...")
    lc_data = _load_json(lc_path)
    ref_data = _load_json(ref_path)
    mm_data = _load_json(elements_path)

    print("Building multimodal index...")
    mm_index = build_multimodal_index(mm_data, mineru_output_dir=str(mineru_dir))

    ref_docs = ref_data.get("documents", {}) or {}
    label_ordinals_by_doc = {
        doc_id: _build_label_ordinals(doc)
        for doc_id, doc in ref_docs.items()
    }
    edge_lookup_by_doc = {
        doc_id: _build_edge_lookup(doc)
        for doc_id, doc in ref_docs.items()
    }

    summary = Counter()
    pair_type_counts = Counter()
    results: List[Dict[str, Any]] = []

    for cand in lc_data.get("candidates", []) or []:
        hop_distance = int(cand.get("hop_distance", 0))
        if hop_distance < args.min_hop or hop_distance > args.max_hop:
            summary["skip_hop"] += 1
            continue

        doc_id = str(cand.get("doc_id", ""))
        doc = ref_docs.get(doc_id, {})
        labels = doc.get("labels", {}) or {}
        ordinals = label_ordinals_by_doc.get(doc_id, {})
        path_label_keys = list(cand.get("path_label_keys", []) or [])
        path_types = list(cand.get("path_types", []) or [])
        if len(path_label_keys) < 2:
            summary["skip_short_path"] += 1
            continue

        endpoint_labels = (path_label_keys[0], path_label_keys[-1])
        endpoint_infos = (
            labels.get(endpoint_labels[0], {}),
            labels.get(endpoint_labels[1], {}),
        )
        a_eid = map_label_to_element(
            doc_id=doc_id,
            label_key=endpoint_labels[0],
            label_info=endpoint_infos[0],
            mm_index=mm_index,
            label_ordinal=ordinals.get(endpoint_labels[0]),
        )
        b_eid = map_label_to_element(
            doc_id=doc_id,
            label_key=endpoint_labels[1],
            label_info=endpoint_infos[1],
            mm_index=mm_index,
            label_ordinal=ordinals.get(endpoint_labels[1]),
        )
        if not a_eid or not b_eid:
            summary["skip_unmapped_endpoint"] += 1
            continue

        a_elem = (
            mm_data.get("documents", {})
            .get(doc_id, {})
            .get("elements", {})
            .get(a_eid)
        )
        b_elem = (
            mm_data.get("documents", {})
            .get(doc_id, {})
            .get("elements", {})
            .get(b_eid)
        )
        if not a_elem or not b_elem:
            summary["skip_missing_endpoint_element"] += 1
            continue

        if args.require_enriched_endpoints and (not _is_enriched(a_elem) or not _is_enriched(b_elem)):
            summary["skip_unenriched_endpoint"] += 1
            continue

        path_ids: List[str] = []
        node_group: List[Dict[str, Any]] = []
        seen_node_group: set[str] = set()
        skip_candidate = False

        for idx, (label_key, path_type) in enumerate(zip(path_label_keys, path_types)):
            label_info = labels.get(label_key, {})
            modal = normalize_label_type(str(label_info.get("label_type", path_type)))
            if modal in MODAL_TYPES:
                mapped_eid = map_label_to_element(
                    doc_id=doc_id,
                    label_key=label_key,
                    label_info=label_info,
                    mm_index=mm_index,
                    label_ordinal=ordinals.get(label_key),
                )
                if not mapped_eid:
                    if args.allow_unmapped_internal_modal and idx not in (0, len(path_label_keys) - 1):
                        path_ids.append(f"{doc_id}::{label_key}")
                        summary["internal_modal_placeholder"] += 1
                        continue
                    skip_candidate = True
                    summary["skip_unmapped_modal_in_path"] += 1
                    break

                path_ids.append(mapped_eid)
                if idx not in (0, len(path_label_keys) - 1) and mapped_eid not in seen_node_group:
                    elem = (
                        mm_data.get("documents", {})
                        .get(doc_id, {})
                        .get("elements", {})
                        .get(mapped_eid)
                    )
                    if elem:
                        node_group.append(dict(elem))
                        seen_node_group.add(mapped_eid)
            else:
                path_ids.append(f"{doc_id}::{label_key}")

        if skip_candidate:
            continue

        resolved_edges = [
            _resolve_edge_context(edge, edge_lookup_by_doc.get(doc_id, {}))
            for edge in (cand.get("edge_contexts", []) or [])
        ]
        non_empty_contexts = [
            _clean_text(edge.get("context"))
            for edge in resolved_edges
            if _clean_text(edge.get("context"))
        ]
        short_query_seed = ""
        if non_empty_contexts:
            short_query_seed = non_empty_contexts[0][:180]
        else:
            mids = [n.get("label_key", "") for n in (cand.get("intermediate_nodes", []) or [])[:3]]
            short_query_seed = " -> ".join([m for m in mids if m])[:180]

        pair = {
            "pair_id": str(cand.get("candidate_id")),
            "doc_id": doc_id,
            "hop_distance": hop_distance,
            "pair_type": str(cand.get("pair_type", "")),
            "element_a_id": a_eid,
            "element_b_id": b_eid,
            "element_a_type": str(a_elem.get("element_type", "")),
            "element_b_type": str(b_elem.get("element_type", "")),
            "element_a": dict(a_elem),
            "element_b": dict(b_elem),
            "path": path_ids,
            "node_group": node_group,
            "edge_contexts": resolved_edges,
            "hub_semantic_summary": " ".join(non_empty_contexts[:2])[:500],
            "hub_metadata": {
                "source": "method_c_feasibility_partial_enrich",
                "short_query_seed": short_query_seed,
                "path_modalities": cand.get("modalities_in_path", []),
                "line_no_span": cand.get("line_no_span"),
            },
        }
        results.append(pair)
        pair_type_counts[pair["pair_type"]] += 1
        summary["kept"] += 1

        if args.max_candidates > 0 and len(results) >= args.max_candidates:
            break

    out_path.parent.mkdir(parents=True, exist_ok=True)
    output = {
        "metadata": {
            "source_long_chain_candidates": str(lc_path),
            "source_reference_graph": str(ref_path),
            "source_elements": str(elements_path),
            "selection": "method_c_feasibility_partial_enrich",
            "min_hop": args.min_hop,
            "max_hop": args.max_hop,
            "require_enriched_endpoints": args.require_enriched_endpoints,
            "allow_unmapped_internal_modal": args.allow_unmapped_internal_modal,
        },
        "summary": {
            "pairs": len(results),
            "pair_type_counts": dict(pair_type_counts),
            "build_stats": dict(summary),
        },
        "pairs": results,
    }
    out_path.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")

    print("\n=== Method C feasibility candidate summary ===")
    print(f"  Pairs written: {len(results)}")
    for key, value in sorted(summary.items()):
        print(f"  {key}: {value}")
    if pair_type_counts:
        print(f"  Pair types: {dict(pair_type_counts)}")
    print(f"  Output: {out_path}")


if __name__ == "__main__":
    main()
