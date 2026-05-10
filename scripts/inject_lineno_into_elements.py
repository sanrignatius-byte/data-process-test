#!/usr/bin/env python3
"""inject_lineno_into_elements.py
=================================
Translate line_no-based chunk membership into position_idx-based membership
so that the existing eval pipeline (eval_graph_topk_rerank.py) reads it.

Why: eval pipeline uses
  element.position_idx -> chunk via pos_to_chunk[position_idx]
which is computed from chunk.paragraph_indices. To make graph rerank reflect
line_no-based chunk-element edges, we re-assign each matched element's
position_idx to a paragraph_idx that belongs to the correct chunk.

Output: data/03_queries/M4query_v1/graphs/multimodal_elements_lineno.json
        (drop-in replacement for the original multimodal_elements.json)
"""
from __future__ import annotations
import argparse
import json
from copy import deepcopy
from pathlib import Path


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--elements", default="data/03_queries/M4query_v1/graphs/multimodal_elements.json")
    p.add_argument("--chunks-lineno", default="data/01_graphs/paragraph_chunks_n400_v2_lineno.json")
    p.add_argument("--element-lineno", default="data/01_graphs/element_latex_lineno_map.json")
    p.add_argument("--output", default="data/03_queries/M4query_v1/graphs/multimodal_elements_lineno.json")
    args = p.parse_args()

    elem_data = json.load(open(args.elements))
    chunks_data = json.load(open(args.chunks_lineno))
    elem_lineno = json.load(open(args.element_lineno))

    new_data = deepcopy(elem_data)

    stats = {
        "elements_total": 0,
        "had_lineno": 0,
        "remapped": 0,
        "no_chunk_match": 0,
        "kept_old_position_idx": 0,
    }

    for doc_id, doc in new_data.get("documents", {}).items():
        chunk_doc = chunks_data.get("documents", {}).get(doc_id)
        if not chunk_doc:
            continue
        # Collect chunks with line_range
        chunks_with_range = []
        for nid, node in chunk_doc.get("nodes", {}).items():
            if "chunk_id" not in node:
                continue
            ls = node.get("chunk_line_start")
            le = node.get("chunk_line_end")
            pidxs = node.get("paragraph_indices", [])
            if ls is not None and le is not None and pidxs:
                chunks_with_range.append((ls, le, pidxs[0]))  # first para_idx of chunk

        for eid, e in doc.get("elements", {}).items():
            stats["elements_total"] += 1
            info = elem_lineno.get(eid)
            if not info or info.get("latex_line_no") is None:
                stats["kept_old_position_idx"] += 1
                continue
            stats["had_lineno"] += 1
            ln = info["latex_line_no"]
            # Find the chunk whose line_range contains ln (with same padding)
            best_para = None
            for ls, le, p0 in chunks_with_range:
                if ls - 5 <= ln <= le + 30:
                    best_para = p0
                    break
            if best_para is not None:
                e["position_idx_old"] = e.get("position_idx")
                e["position_idx"] = best_para
                e["latex_line_no"] = ln
                e["match_method"] = info.get("match_method")
                e["match_confidence"] = info.get("confidence")
                stats["remapped"] += 1
            else:
                stats["no_chunk_match"] += 1

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(new_data, ensure_ascii=False))
    print(f"Wrote {args.output}")
    print(f"  elements_total: {stats['elements_total']}")
    print(f"  had_lineno: {stats['had_lineno']}")
    print(f"  remapped (position_idx -> chunk para0): {stats['remapped']}")
    print(f"  no_chunk_match (lineno but no chunk range): {stats['no_chunk_match']}")
    print(f"  kept_old_position_idx (no lineno): {stats['kept_old_position_idx']}")


if __name__ == "__main__":
    main()
