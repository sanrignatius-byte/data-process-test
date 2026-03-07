#!/usr/bin/env python3
"""Enrich hub multi-hop candidates with MinerU element details.

Converts latex_hub_multihop_candidates.json (topology output) into the same
format as multihop_l1_candidates.json so it can be fed directly to
generate_multihop_l1_queries.py.

The key task: map LaTeX node IDs (e.g. "1306.5204::el::fig:histograms")
back to MinerU element IDs (e.g. "1306.5204_figure_2") and attach full
element details (caption, image_path, context_before/after).

Usage:
    python scripts/enrich_hub_candidates.py \
        --hub-candidates data/latex_hub_multihop_candidates.json \
        --elements data/multimodal_elements.json \
        --latex-graph data/latex_reference_graph.json \
        --output data/hub_candidates_enriched.json \
        [--limit 50]
"""

import argparse
import json
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

# ─── Label type normalization (from analyze_latex_graph_topology.py) ──────────

_LABEL_TYPE_MAP = {
    "figure": "figure",
    "fig": "figure",
    "subfigure": "figure",
    "table": "table",
    "tab": "table",
    "equation": "equation",
    "eq": "equation",
    "align": "equation",
    "eqnarray": "equation",
    "formula": "equation",
}

ELEMENT_MODALITIES = {"figure", "table", "equation"}
MINERU_MODAL_MAP = {"figure": "figure", "table": "table", "equation": "formula"}


def normalize_label_type(raw: str) -> str:
    return _LABEL_TYPE_MAP.get(raw.lower().strip(), raw.lower().strip())


def tokenize(text: str) -> Set[str]:
    return set(re.findall(r"\w+", text.lower()))


def jaccard(a: Set[str], b: Set[str]) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def parse_number(s: str) -> Optional[int]:
    m = re.search(r"(\d+)", s)
    return int(m.group(1)) if m else None


# ─── Build MinerU index ──────────────────────────────────────────────────────

def build_mm_index(mm_data: Dict[str, Any]) -> Dict[str, Any]:
    """Build lookup indices from multimodal_elements.json."""
    by_doc: Dict[str, Dict] = {}
    all_elements: Dict[str, Dict] = {}  # element_id → element dict

    for doc_key, doc in mm_data["documents"].items():
        doc_id = doc["doc_id"]
        by_number: Dict[str, Dict[int, str]] = defaultdict(dict)  # type → {number: eid}
        by_caption: Dict[str, List[Tuple[str, Set[str]]]] = defaultdict(list)

        elements = doc.get("elements", {})
        if isinstance(elements, dict):
            elem_list = elements.values()
        else:
            elem_list = elements

        for el in elem_list:
            eid = el["element_id"]
            etype = el["element_type"]
            all_elements[eid] = el

            num = el.get("number")
            if num is not None:
                by_number[etype][int(num)] = eid

            cap = el.get("caption", "") or ""
            cap_tokens = tokenize(cap)
            if cap_tokens:
                by_caption[etype].append((eid, cap_tokens))

        by_doc[doc_id] = {
            "by_number": dict(by_number),
            "by_caption": dict(by_caption),
        }

    return {"by_doc": by_doc, "all_elements": all_elements}


def map_label_to_element(
    doc_id: str,
    label_key: str,
    label_info: Dict[str, Any],
    mm_index: Dict[str, Any],
) -> Optional[str]:
    """Map a LaTeX label to a MinerU element_id (same logic as topology script)."""
    doc_idx = mm_index["by_doc"].get(doc_id)
    if not doc_idx:
        return None

    ltype = normalize_label_type(str(label_info.get("label_type", "")))
    if ltype not in ELEMENT_MODALITIES:
        return None
    mm_type = MINERU_MODAL_MAP.get(ltype, ltype)

    # 1. number from label key
    number = parse_number(label_key)
    if number is not None:
        eid = doc_idx["by_number"].get(mm_type, {}).get(number)
        if eid:
            return eid
        for part in re.split(r"[_\-:]+", label_key):
            n = parse_number(part)
            if n is not None:
                eid = doc_idx["by_number"].get(mm_type, {}).get(n)
                if eid:
                    return eid

    # 2. caption Jaccard
    caption_tokens = tokenize(str(label_info.get("caption", "")))
    if caption_tokens:
        best_id = None
        best_score = 0.0
        for eid, cap_tokens in doc_idx["by_caption"].get(mm_type, []):
            s = jaccard(caption_tokens, cap_tokens)
            if s > best_score:
                best_score = s
                best_id = eid
        if best_score >= 0.25:
            return best_id

    return None


# ─── Build node→element mapping ─────────────────────────────────────────────

def build_sequential_mapping(
    doc_id: str,
    labels: Dict[str, Any],
    mm_elements: Dict[str, Any],
) -> Dict[str, str]:
    """Sequential order matching: sort labels by line_no, elements by number/position.

    For each modality type, if we have N LaTeX labels and M MinerU elements,
    match them 1:1 in order (up to min(N,M)).
    """
    result: Dict[str, str] = {}

    for latex_type, mm_type in [("figure", "figure"), ("table", "table"),
                                 ("equation", "formula")]:
        # Collect LaTeX labels of this type, sorted by line_no
        type_labels = []
        for lk, info in labels.items():
            lt = normalize_label_type(str(info.get("label_type", "")))
            if lt == latex_type:
                line = info.get("line_no", 99999)
                type_labels.append((line, lk))
        type_labels.sort()

        # Collect MinerU elements of this type, sorted by number then position
        type_elements = []
        for eid, el in mm_elements.items():
            if el["element_type"] == mm_type:
                num = el.get("number") or 99999
                pos = el.get("position_idx") or 99999
                type_elements.append((num, pos, eid))
        type_elements.sort()

        # Match 1:1
        for i in range(min(len(type_labels), len(type_elements))):
            _, label_key = type_labels[i]
            _, _, eid = type_elements[i]
            node_id = f"{doc_id}::el::{label_key}"
            result[node_id] = eid

    return result


def build_node_element_map(
    latex_data: Dict[str, Any],
    mm_index: Dict[str, Any],
    mm_data: Dict[str, Any],
) -> Dict[str, str]:
    """Map all LaTeX element node IDs to MinerU element IDs.

    Strategy (in priority order):
    1. Number match (from label key)
    2. Caption Jaccard >= 0.25
    3. Sequential order match (sort by line_no vs number/position)
    """
    node_to_element: Dict[str, str] = {}

    for doc_key, doc in latex_data.items():
        if not isinstance(doc, dict):
            continue
        doc_id = doc.get("doc_id", doc_key)
        labels = doc.get("labels", {})

        # Phase 1+2: standard mapping (number + caption)
        for label_key, info in labels.items():
            modal = normalize_label_type(str(info.get("label_type", "")))
            if modal not in ELEMENT_MODALITIES:
                continue
            node_id = f"{doc_id}::el::{label_key}"
            mapped_eid = map_label_to_element(doc_id, label_key, info, mm_index)
            if mapped_eid:
                node_to_element[node_id] = mapped_eid

        # Phase 3: sequential matching for remaining unmapped labels
        mm_doc = mm_data["documents"].get(doc_id)
        if mm_doc:
            mm_elements = mm_doc.get("elements", {})
            if isinstance(mm_elements, dict):
                seq_map = build_sequential_mapping(doc_id, labels, mm_elements)
                # Only add mappings that don't conflict with existing ones
                used_eids = set(node_to_element.values())
                for node_id, eid in seq_map.items():
                    if node_id not in node_to_element and eid not in used_eids:
                        node_to_element[node_id] = eid
                        used_eids.add(eid)

    return node_to_element


# ─── Extract edge contexts from LaTeX graph ──────────────────────────────────

def build_edge_context_index(
    latex_data: Dict[str, Any],
) -> Dict[Tuple[str, str], List[Dict[str, str]]]:
    """Build index of edge contexts from LaTeX reference graph."""
    edge_ctx: Dict[Tuple[str, str], List[Dict[str, str]]] = defaultdict(list)

    for doc_key, doc in latex_data.items():
        if not isinstance(doc, dict):
            continue
        doc_id = doc.get("doc_id", doc_key)
        for edge in doc.get("edges", []):
            src_label = edge.get("source_label", edge.get("source", ""))
            tgt_label = edge.get("target_label", edge.get("target", ""))
            src_nid = f"{doc_id}::el::{src_label}"
            tgt_nid = f"{doc_id}::el::{tgt_label}"
            ctx = {
                "source": src_nid,
                "target": tgt_nid,
                "ref_text": edge.get("ref_text", ""),
                "context_snippet": (edge.get("context", "") or "")[:300],
            }
            edge_ctx[(src_nid, tgt_nid)].append(ctx)
            edge_ctx[(tgt_nid, src_nid)].append(ctx)

    return dict(edge_ctx)


# ─── Main enrichment ────────────────────────────────────────────────────────

def enrich_candidates(
    hub_data: Dict[str, Any],
    mm_index: Dict[str, Any],
    node_to_element: Dict[str, str],
    edge_ctx_index: Dict[Tuple[str, str], List[Dict[str, str]]],
    limit: int = 0,
) -> Dict[str, Any]:
    """Convert hub candidates to generation-ready format."""
    candidates = hub_data["candidates"]
    if limit > 0:
        candidates = candidates[:limit]

    pairs: List[Dict[str, Any]] = []
    skipped_no_mapping = 0
    skipped_no_element = 0
    pair_counter: Dict[str, int] = defaultdict(int)

    for cand in candidates:
        # Find the two element endpoints (non-paragraph nodes)
        endpoints = []
        for nid, ntype in zip(cand["path_node_ids"], cand["path_node_types"]):
            if ntype in {"figure", "table", "equation", "formula"}:
                endpoints.append((nid, ntype))

        if len(endpoints) < 2:
            skipped_no_mapping += 1
            continue

        # Take first and last element in path as A and B
        nid_a, ntype_a = endpoints[0]
        nid_b, ntype_b = endpoints[-1]

        # Map to MinerU element IDs
        eid_a = node_to_element.get(nid_a)
        eid_b = node_to_element.get(nid_b)
        if not eid_a or not eid_b:
            skipped_no_mapping += 1
            continue

        # Get element details
        el_a = mm_index["all_elements"].get(eid_a)
        el_b = mm_index["all_elements"].get(eid_b)
        if not el_a or not el_b:
            skipped_no_element += 1
            continue

        # Normalize modality types (equation→formula for MinerU compat)
        mm_type_a = "formula" if ntype_a in ("equation", "formula") else ntype_a
        mm_type_b = "formula" if ntype_b in ("equation", "formula") else ntype_b
        pair_type = "+".join(sorted([mm_type_a, mm_type_b]))

        # Generate pair_id
        doc_id = cand["doc_id"]
        pair_counter[doc_id] += 1
        pair_id = f"{doc_id}_hub_pair_{pair_counter[doc_id]}"

        # Build element dicts (same format as multihop_l1_candidates)
        def make_element_dict(el: Dict) -> Dict[str, Any]:
            return {
                "element_id": el["element_id"],
                "element_type": el["element_type"],
                "caption": el.get("caption", "") or "",
                "content": el.get("content", "") or "",
                "image_path": el.get("image_path", "") or "",
                "context_before": (el.get("context_before", "") or "")[:300],
                "context_after": (el.get("context_after", "") or "")[:300],
            }

        # Collect edge contexts along the path
        path_nids = cand["path_node_ids"]
        edge_contexts = []
        for i in range(len(path_nids) - 1):
            key = (path_nids[i], path_nids[i + 1])
            if key in edge_ctx_index:
                for ctx in edge_ctx_index[key]:
                    # Replace node IDs with element IDs in context
                    edge_contexts.append({
                        "source": node_to_element.get(ctx["source"], ctx["source"]),
                        "target": node_to_element.get(ctx["target"], ctx["target"]),
                        "ref_text": ctx["ref_text"],
                        "context_snippet": ctx["context_snippet"],
                    })

        # Map path node IDs to element IDs where possible
        mapped_path = []
        for nid in path_nids:
            mapped = node_to_element.get(nid)
            mapped_path.append(mapped if mapped else nid)

        pair_dict = {
            "pair_id": pair_id,
            "doc_id": doc_id,
            "element_a_id": eid_a,
            "element_b_id": eid_b,
            "element_a_type": mm_type_a,
            "element_b_type": mm_type_b,
            "pair_type": pair_type,
            "hop_distance": cand["hop_distance"],
            "path": mapped_path,
            "quality_score": 0.8,  # default for topology-validated candidates
            "overlap_with_existing_l1": False,
            "element_a": make_element_dict(el_a),
            "element_b": make_element_dict(el_b),
            "edge_contexts": edge_contexts,
            # Preserve hub metadata for analysis
            "hub_metadata": {
                "hub_node_id": cand["hub_node_id"],
                "hub_label": cand["hub_label"],
                "is_cross_doc": cand["is_cross_doc"],
                "page_span": cand.get("page_span"),
                "line_no_span": cand.get("line_no_span"),
                "short_query_seed": cand.get("short_query_seed", ""),
                "long_query_seed": cand.get("long_query_seed", ""),
            },
        }
        pairs.append(pair_dict)

    # Summary statistics
    by_type: Dict[str, int] = defaultdict(int)
    by_hop: Dict[str, int] = defaultdict(int)
    docs_covered: Set[str] = set()
    cross_doc_count = 0

    for p in pairs:
        by_type[p["pair_type"]] += 1
        by_hop[str(p["hop_distance"])] += 1
        docs_covered.add(p["doc_id"])
        if p["hub_metadata"]["is_cross_doc"]:
            cross_doc_count += 1

    result = {
        "metadata": {
            "source": "latex_hub_multihop_candidates (enriched)",
            "hub_candidates_file": str(hub_data.get("metadata", {}).get("generated_at", "")),
            "algorithm_version": hub_data.get("metadata", {}).get("algorithm_version", ""),
            "max_pairs": len(pairs),
            "enrichment_stats": {
                "input_candidates": len(candidates),
                "output_pairs": len(pairs),
                "skipped_no_mapping": skipped_no_mapping,
                "skipped_no_element": skipped_no_element,
                "mapping_rate": round(len(pairs) / len(candidates), 4) if candidates else 0,
            },
        },
        "summary": {
            "total_selected": len(pairs),
            "by_type": dict(by_type),
            "by_hop": dict(by_hop),
            "docs_covered": len(docs_covered),
            "cross_doc": cross_doc_count,
            "intra_doc": len(pairs) - cross_doc_count,
        },
        "pairs": pairs,
    }

    return result


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--hub-candidates",
        default="data/latex_hub_multihop_candidates.json",
        help="Input hub candidates from topology analysis",
    )
    ap.add_argument(
        "--elements",
        default="data/multimodal_elements.json",
        help="MinerU multimodal elements",
    )
    ap.add_argument(
        "--latex-graph",
        default="data/latex_reference_graph.json",
        help="LaTeX reference graph (for label info and edge contexts)",
    )
    ap.add_argument(
        "--output",
        default="data/hub_candidates_enriched.json",
        help="Output enriched candidates",
    )
    ap.add_argument("--limit", type=int, default=0, help="Limit candidates (0=all)")
    args = ap.parse_args()

    print("Loading data...")
    with open(args.hub_candidates) as f:
        hub_data = json.load(f)
    print(f"  Hub candidates: {len(hub_data['candidates'])}")

    with open(args.elements) as f:
        mm_data = json.load(f)
    print(f"  Documents: {len(mm_data['documents'])}")

    with open(args.latex_graph) as f:
        latex_raw = json.load(f)
    # Handle both flat dict and nested {"documents": {...}} format
    if "documents" in latex_raw and isinstance(latex_raw["documents"], dict):
        latex_data = latex_raw["documents"]
    else:
        latex_data = latex_raw
    latex_docs = {k: v for k, v in latex_data.items() if isinstance(v, dict) and "labels" in v}
    print(f"  LaTeX docs with labels: {len(latex_docs)}")

    print("\nBuilding indices...")
    mm_index = build_mm_index(mm_data)
    print(f"  Total elements indexed: {len(mm_index['all_elements'])}")

    node_to_element = build_node_element_map(latex_data, mm_index, mm_data)
    print(f"  Node→element mappings: {len(node_to_element)}")

    edge_ctx_index = build_edge_context_index(latex_data)
    print(f"  Edge context pairs: {len(edge_ctx_index)}")

    print("\nEnriching candidates...")
    result = enrich_candidates(hub_data, mm_index, node_to_element, edge_ctx_index, args.limit)

    summary = result["summary"]
    stats = result["metadata"]["enrichment_stats"]
    print(f"\n=== Enrichment Results ===")
    print(f"  Input:  {stats['input_candidates']} candidates")
    print(f"  Output: {stats['output_pairs']} pairs ({stats['mapping_rate']*100:.1f}% mapping rate)")
    print(f"  Skipped (no mapping): {stats['skipped_no_mapping']}")
    print(f"  Skipped (no element): {stats['skipped_no_element']}")
    print(f"\n  By type: {dict(summary['by_type'])}")
    print(f"  By hop:  {dict(summary['by_hop'])}")
    print(f"  Docs covered: {summary['docs_covered']}")
    print(f"  Cross-doc: {summary['cross_doc']} / Intra-doc: {summary['intra_doc']}")

    with open(args.output, "w") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    print(f"\nWritten to {args.output}")


if __name__ == "__main__":
    main()
