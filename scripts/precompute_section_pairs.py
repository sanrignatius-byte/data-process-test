#!/usr/bin/env python3
"""
Pre-compute multimodal pairs from graph topology for section-based pairing.

Algorithm:
  For each section node, collect figure/table descendants within a limited
  traversal depth (section_contains → children, then next_element → 2 steps,
  stopping at sub-section boundaries). Pair each figure with each table
  within the same section scope.

Output: injects 'multimodal_pairs' into each document in the graph,
making it compatible with IntraDocPairSelector._section_pairs().

Usage:
  python scripts/precompute_section_pairs.py \
    --graph data/01_graphs/huawei_multimodal_elements.json \
    --output data/01_graphs/huawei_multimodal_elements_with_pairs.json \
    --max-depth 3 \
    --max-pairs-per-section 10
"""

import argparse, json
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Set

MODAL_TYPES = frozenset({"figure", "table", "formula"})


def collect_section_multimodal(
    section_id: str,
    elements: Dict[str, Dict],
    out_adj: Dict[str, List[tuple]],
    max_depth: int = 4,
) -> Dict[str, List[str]]:
    """Collect figure/table elements within traversal from section.

    Traversal:
    - section_contains edges: depth unchanged (structural scope)
    - next_element edges: depth+1 (linear traversal)
    - Allow entering ONE level of sub-section (to aggregate siblings)
    - Stop at deeper sub-section boundaries
    """
    result = {"figure": [], "table": [], "formula": []}
    visited: Set[str] = set()
    stack = [(section_id, 0, 0)]  # (node, depth, sub_section_depth)

    while stack:
        cur, depth, ss_depth = stack.pop()
        if cur in visited or depth > max_depth:
            continue

        ct = elements.get(cur, {}).get("element_type", "")
        if ct in MODAL_TYPES:
            result[ct].append(cur)
            visited.add(cur)
            continue  # don't traverse further from modal elements

        if ct == "section" and cur != section_id:
            if ss_depth >= 1:
                continue  # stop: already entered one sub-section level
            ss_depth += 1  # enter first sub-section level

        visited.add(cur)

        for nb, et in out_adj.get(cur, []):
            if et == "section_contains":
                stack.append((nb, depth, ss_depth))
            elif et == "next_element":
                stack.append((nb, depth + 1, ss_depth))

    return result


def build_multimodal_pairs(
    graph: dict,
    max_depth: int = 3,
    max_pairs_per_section: int = 10,
) -> dict:
    """Inject multimodal_pairs into each document based on section topology."""
    total_pairs = 0
    enriched_docs = 0

    for doc_id, doc in graph["documents"].items():
        elements = doc.get("elements", {})
        edges = doc.get("edges", [])

        # Build outgoing adjacency
        out_adj: Dict[str, List[tuple]] = defaultdict(list)
        for e in edges:
            out_adj[e["source_id"]].append((e["target_id"], e["edge_type"]))

        # Find all section nodes
        sections = [
            eid for eid, el in elements.items()
            if el.get("element_type") == "section"
        ]

        all_pairs = []
        seen_pairs: Set[frozenset] = set()

        for sec_id in sections:
            mm = collect_section_multimodal(sec_id, elements, out_adj, max_depth)

            figs = mm.get("figure", [])
            tbls = mm.get("table", [])
            fmls = mm.get("formula", [])

            # Generate cross-modal pairs
            pairs_in_section = 0
            # figure+table
            for fid in figs:
                for tid in tbls:
                    key = frozenset([fid, tid])
                    if key in seen_pairs:
                        continue
                    seen_pairs.add(key)
                    all_pairs.append({
                        "element_a_id": fid,
                        "element_b_id": tid,
                        "hop_distance": 1,  # conservative
                        "relationship": "co_section",
                        "path": [sec_id, fid, tid],
                    })
                    pairs_in_section += 1
                    if pairs_in_section >= max_pairs_per_section:
                        break
                if pairs_in_section >= max_pairs_per_section:
                    break

            # figure+formula, table+formula similarly...

        if all_pairs:
            doc["multimodal_pairs"] = all_pairs
            enriched_docs += 1
            total_pairs += len(all_pairs)

    print(f"  Enriched {enriched_docs} docs with {total_pairs} multimodal pairs")
    return graph


def main():
    ap = argparse.ArgumentParser(
        description="Pre-compute section-based multimodal pairs from graph topology"
    )
    ap.add_argument("--graph", default="data/01_graphs/huawei_multimodal_elements.json")
    ap.add_argument("--output", default="data/01_graphs/huawei_multimodal_elements_with_pairs.json")
    ap.add_argument("--max-depth", type=int, default=3)
    ap.add_argument("--max-pairs-per-section", type=int, default=10)
    args = ap.parse_args()

    graph = json.loads(Path(args.graph).read_text(encoding="utf-8"))
    graph = build_multimodal_pairs(
        graph,
        max_depth=args.max_depth,
        max_pairs_per_section=args.max_pairs_per_section,
    )
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(graph, indent=2, ensure_ascii=False))
    print(f"  Output: {args.output}")


if __name__ == "__main__":
    main()
