#!/usr/bin/env python3
"""
Build L1 candidate pairs from Huawei/realworld graph topology.

Strategy (no LaTeX sources):
  1. Identify anchor hubs (high-degree sections) from hub scores
  2. Pair each anchor hub with its most important child/neighbor elements
  3. Also pair adjacent sections for cross-section reasoning queries

Output format compatible with generate_multihop_l1_queries.py:
  { "pairs": [ { pair_id, doc_id, element_a, element_b, ... } ] }

Usage:
  python scripts/build_huawei_candidates.py \
    --graph data/01_graphs/huawei_multimodal_elements.json \
    --hub-scores data/01_graphs/huawei_hub_scores.json \
    --output data/03_queries/huawei_l1_candidates.json \
    --max-pairs-per-doc 5 \
    --limit-docs 30
"""

import argparse, json, sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def build_adjacency(graph: dict) -> tuple[dict, dict]:
    """Build out_adj and in_adj from graph edges."""
    out_adj = defaultdict(set)
    in_adj = defaultdict(set)
    for e in graph.get("edges", []):
        s, t = e["source_id"], e["target_id"]
        out_adj[s].add(t)
        in_adj[t].add(s)
    return out_adj, in_adj


def build_element_index(graph: dict) -> dict[str, dict]:
    """Build element_id → element dict from graph documents."""
    idx = {}
    for did, doc in graph.get("documents", {}).items():
        for eid, el in doc.get("elements", {}).items():
            idx[eid] = el
    return idx


def get_anchor_hubs(hub_scores: dict, min_score: int = 10, max_per_doc: int = 5) -> dict[str, list[dict]]:
    """Get anchor hubs per doc from hub scores."""
    anchors = hub_scores.get("anchor_hubs", [])
    by_doc = defaultdict(list)
    for a in anchors:
        if a.get("score", 0) >= min_score:
            by_doc[a["doc_id"]].append(a)
    # Keep top-N per doc by score
    result = {}
    for did, lst in by_doc.items():
        lst.sort(key=lambda x: -x["score"])
        result[did] = lst[:max_per_doc]
    return result


def build_candidate_pairs(
    graph: dict,
    hub_scores: dict,
    max_pairs_per_doc: int = 5,
    limit_docs: int = 0,
) -> list[dict]:
    """Build candidate pairs from graph topology."""
    out_adj, in_adj = build_adjacency(graph)
    elem_idx = build_element_index(graph)
    anchor_by_doc = get_anchor_hubs(hub_scores, min_score=10, max_per_doc=10)

    doc_ids = sorted(anchor_by_doc.keys())
    if limit_docs:
        doc_ids = doc_ids[:limit_docs]

    pairs = []
    pair_counter = 0

    for did in doc_ids:
        anchors = anchor_by_doc[did]
        doc_pairs = []

        # Get everything this doc contains
        doc_edges = [e for e in graph.get("edges", []) if e.get("doc_id") == did]
        section_contains = defaultdict(list)
        next_of = {}
        prev_of = {}
        for e in doc_edges:
            s, t, et = e["source_id"], e["target_id"], e["edge_type"]
            if et == "section_contains":
                section_contains[s].append(t)
            elif et == "next_element":
                next_of[s] = t
                prev_of[t] = s

        for anchor in anchors:
            if len(doc_pairs) >= max_pairs_per_doc:
                break

            aid = anchor["element_id"]
            anchor_el = elem_idx.get(aid)
            if not anchor_el:
                continue

            # Strategy 1: section → child table/element
            children = section_contains.get(aid, [])
            # Prioritize tables (most valuable for queries)
            table_children = [c for c in children
                              if elem_idx.get(c, {}).get("element_type") == "table"]
            text_children = [c for c in children
                             if elem_idx.get(c, {}).get("element_type") == "text"
                             and len(elem_idx.get(c, {}).get("content", "")) > 200]

            for child_id in (table_children[:2] + text_children[:2]):
                if len(doc_pairs) >= max_pairs_per_doc:
                    break
                child_el = elem_idx.get(child_id)
                if not child_el:
                    continue
                pair_counter += 1
                doc_pairs.append({
                    "pair_id": f"hw_{did[:30]}_pair_{pair_counter}",
                    "doc_id": did,
                    "element_a_id": aid,
                    "element_b_id": child_id,
                    "element_a_type": anchor_el.get("element_type"),
                    "element_b_type": child_el.get("element_type"),
                    "pair_type": "intra_doc_section_child",
                    "hop_distance": 1,
                    "element_a": _strip_element(anchor_el),
                    "element_b": _strip_element(child_el),
                    "node_group": [_strip_element(anchor_el), _strip_element(child_el)],
                })

            # Strategy 2: consecutive sections
            next_eid = next_of.get(aid)
            if next_eid and len(doc_pairs) < max_pairs_per_doc:
                next_el = elem_idx.get(next_eid)
                if next_el and next_el.get("element_type") == "section":
                    pair_counter += 1
                    doc_pairs.append({
                        "pair_id": f"hw_{did[:30]}_pair_{pair_counter}",
                        "doc_id": did,
                        "element_a_id": aid,
                        "element_b_id": next_eid,
                        "element_a_type": "section",
                        "element_b_type": "section",
                        "pair_type": "intra_doc_adjacent_sections",
                        "hop_distance": 1,
                        "element_a": _strip_element(anchor_el),
                        "element_b": _strip_element(next_el),
                        "node_group": [_strip_element(anchor_el), _strip_element(next_el)],
                    })

        pairs.extend(doc_pairs)

    return pairs


def _strip_element(el: dict) -> dict:
    """Return minimal element dict with essential fields."""
    out = {
        "element_id": el.get("element_id", ""),
        "element_type": el.get("element_type", ""),
    }
    # Add type-specific fields
    if el.get("element_type") == "section":
        out["label"] = el.get("label", "")
        out["content"] = (el.get("content", "") or "")[:800]
    elif el.get("element_type") == "text":
        out["content"] = (el.get("content", "") or "")[:800]
    elif el.get("element_type") == "table":
        out["caption"] = el.get("caption", "")
        out["content"] = (el.get("content", "") or "")[:800]
    elif el.get("element_type") == "figure":
        out["caption"] = el.get("caption", "")
        out["image_path"] = el.get("image_path", "")
    elif el.get("element_type") == "formula":
        out["content"] = (el.get("content", "") or "")[:400]

    # Carry enriched fields if present
    for k in ("enriched_title", "enriched_content", "enriched_metadata"):
        if k in el:
            out[k] = el[k]

    return out


def main():
    ap = argparse.ArgumentParser(description="Build L1 candidate pairs from Huawei graph")
    ap.add_argument("--graph", default="data/01_graphs/huawei_multimodal_elements.json")
    ap.add_argument("--hub-scores", default="data/01_graphs/huawei_hub_scores.json")
    ap.add_argument("--output", default="data/03_queries/huawei_l1_candidates.json")
    ap.add_argument("--max-pairs-per-doc", type=int, default=5)
    ap.add_argument("--limit-docs", type=int, default=30, help="Limit to first N docs")
    args = ap.parse_args()

    graph = json.loads(Path(args.graph).read_text(encoding="utf-8"))
    hub_scores = json.loads(Path(args.hub_scores).read_text(encoding="utf-8"))

    pairs = build_candidate_pairs(
        graph, hub_scores,
        max_pairs_per_doc=args.max_pairs_per_doc,
        limit_docs=args.limit_docs,
    )

    out = {
        "generated_at": __import__("datetime").datetime.now(__import__("datetime").timezone.utc).isoformat(),
        "source_graph": str(args.graph),
        "source_hub_scores": str(args.hub_scores),
        "total_pairs": len(pairs),
        "pairs": pairs,
    }

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, ensure_ascii=False, indent=2))

    print(f"Generated {len(pairs)} candidate pairs from {args.limit_docs} docs")
    print(f"Output: {out_path}")

    # Show distribution
    pair_types = defaultdict(int)
    elem_pairs = defaultdict(int)
    for p in pairs:
        pair_types[p["pair_type"]] += 1
        ep = f"{p['element_a_type']}+{p['element_b_type']}"
        elem_pairs[ep] += 1
    print(f"Pair types: {dict(pair_types)}")
    print(f"Element combos: {dict(elem_pairs)}")


if __name__ == "__main__":
    main()
