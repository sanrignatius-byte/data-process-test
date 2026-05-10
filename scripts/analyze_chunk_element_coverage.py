#!/usr/bin/env python3
"""
Analyze chunk→element mapping and coverage statistics.

Answers mentor's questions:
1. How many elements does each chunk contain on average?
2. For M4query_v1 queries, how many real elements does top-k chunk recall cover?
3. Distribution by element type (figure/table/formula).
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Set, Tuple

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def load_element_index(elements_path: Path) -> Dict[str, Dict[int, List[dict]]]:
    """Return {doc_id: {position_idx: [element_info, ...]}}"""
    data = json.loads(elements_path.read_bytes())
    index: Dict[str, Dict[int, List[dict]]] = defaultdict(lambda: defaultdict(list))
    for doc_id, doc in data.get("documents", {}).items():
        for eid, elem in doc.get("elements", {}).items():
            pidx = elem.get("position_idx")
            if pidx is not None:
                index[doc_id][pidx].append({
                    "element_id": eid,
                    "element_type": elem.get("element_type", "unknown"),
                })
    return index


def load_chunk_para_ranges(chunk_file: Path) -> Dict[str, Dict[str, Tuple[int, int, str]]]:
    """Return {doc_id: {chunk_id: (min_para, max_para, section)}}"""
    data = json.loads(chunk_file.read_bytes())
    result: Dict[str, Dict[str, Tuple[int, int, str]]] = {}
    for doc_id, doc in data.get("documents", {}).items():
        chunks: Dict[str, Tuple[int, int, str]] = {}
        # Collect chunk→paragraph edges
        chunk_paras: Dict[str, List[int]] = defaultdict(list)
        for edge in doc.get("edges", []):
            if edge.get("relation") == "chunk_contains_paragraph":
                chunk_paras[edge["source"]].append(
                    doc["paragraph_nodes"][edge["target"]]["para_idx"]
                )
        for chunk_id, para_indices in chunk_paras.items():
            node = doc["nodes"].get(chunk_id, {})
            section = node.get("section_title", "")
            chunks[chunk_id] = (min(para_indices), max(para_indices), section)
        result[doc_id] = chunks
    return result


def map_elements_to_chunks(
    doc_chunks: Dict[str, Tuple[int, int, str]],
    doc_elements: Dict[int, List[dict]],
) -> Dict[str, List[dict]]:
    """Map each chunk to its contained elements. {chunk_id: [element_info, ...]}"""
    mapping: Dict[str, List[dict]] = defaultdict(list)
    for chunk_id, (min_p, max_p, _section) in doc_chunks.items():
        for pidx in range(min_p, max_p + 1):
            for elem in doc_elements.get(pidx, []):
                mapping[chunk_id].append(elem)
    return mapping


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--chunk-file", default="data/01_graphs/chunk_virtual_nodes_v2.json")
    ap.add_argument("--elements-file", default="data/03_queries/M4query_v1/graphs/multimodal_elements.json")
    ap.add_argument("--qrels", default="data/03_queries/M4query_v1/qrels.jsonl")
    args = ap.parse_args()

    chunk_file = PROJECT_ROOT / args.chunk_file
    elements_file = PROJECT_ROOT / args.elements_file
    qrels_file = PROJECT_ROOT / args.qrels

    # Load data
    elem_index = load_element_index(elements_file)
    chunk_ranges = load_chunk_para_ranges(chunk_file)

    # Only analyze docs present in the element file
    doc_ids = set(elem_index.keys()) & set(chunk_ranges.keys())
    print(f"Docs in elements: {len(elem_index)}")
    print(f"Docs in chunks:   {len(chunk_ranges)}")
    print(f"Docs in both:     {len(doc_ids)}")

    # ── Global stats ──
    total_chunks = 0
    total_elements = 0
    chunks_with_elements = 0
    elements_per_chunk: List[int] = []
    type_counts: Dict[str, int] = defaultdict(int)

    for doc_id in sorted(doc_ids):
        mapping = map_elements_to_chunks(
            chunk_ranges[doc_id], elem_index[doc_id]
        )
        for chunk_id, elems in mapping.items():
            total_chunks += 1
            n = len(elems)
            total_elements += n
            elements_per_chunk.append(n)
            if n > 0:
                chunks_with_elements += 1
            for e in elems:
                type_counts[e["element_type"]] += 1

    if total_chunks == 0:
        print("No chunks found!")
        return

    # ── Print report ──
    print(f"\n{'='*60}")
    print(f"CHUNK → ELEMENT COVERAGE REPORT")
    print(f"{'='*60}")
    print(f"Total chunks:            {total_chunks}")
    print(f"Chunks with ≥1 element:  {chunks_with_elements} ({100*chunks_with_elements/total_chunks:.1f}%)")
    print(f"Chunks with 0 elements:  {total_chunks - chunks_with_elements} ({100*(total_chunks-chunks_with_elements)/total_chunks:.1f}%)")
    print(f"Total elements assigned: {total_elements}")
    print(f"Avg elements per chunk:  {total_elements/total_chunks:.2f}")
    print(f"Avg elements per chunk (non-empty only): {total_elements/chunks_with_elements:.2f}")

    # Distribution
    from collections import Counter
    dist = Counter(elements_per_chunk)
    print(f"\nElements-per-chunk distribution:")
    for k in sorted(dist)[:15]:
        bar = "█" * dist[k]
        print(f"  {k:3d} elements: {dist[k]:5d} chunks {bar}")

    # By type
    print(f"\nElement type distribution:")
    for t, c in sorted(type_counts.items()):
        print(f"  {t:12s}: {c:5d} ({100*c/total_elements:.1f}%)")

    # ── Query-level analysis ──
    print(f"\n{'='*60}")
    print(f"QUERY-LEVEL CHUNK COVERAGE (M4query_v1)")
    print(f"{'='*60}")

    # Load qrels: query_id → set of relevant element_ids
    qrels = []
    with open(qrels_file) as f:
        for line in f:
            qrels.append(json.loads(line))

    query_elements: Dict[str, Set[str]] = defaultdict(set)
    for r in qrels:
        query_elements[r["query_id"]].add(r["passage_id"])

    # For each query, find which chunks contain those elements
    # Build reverse: element_id → chunk_id
    elem_to_chunk: Dict[str, str] = {}
    for doc_id in sorted(doc_ids):
        mapping = map_elements_to_chunks(chunk_ranges[doc_id], elem_index[doc_id])
        for chunk_id, elems in mapping.items():
            for e in elems:
                elem_to_chunk[e["element_id"]] = chunk_id

    print(f"Elements with chunk mapping: {len(elem_to_chunk)} / {sum(len(v) for v in query_elements.values())} total qrels entries")

    # Count how many queries have both elements in the SAME chunk vs DIFFERENT chunks
    same_chunk = 0
    diff_chunk = 0
    only_one_mapped = 0

    for qid, eids in query_elements.items():
        chunk_ids = set()
        for eid in eids:
            cid = elem_to_chunk.get(eid)
            if cid:
                chunk_ids.add(cid)
        if len(eids) < 2:
            continue
        n_mapped = sum(1 for eid in eids if eid in elem_to_chunk)
        if n_mapped < 2:
            only_one_mapped += 1
        elif len(chunk_ids) == 1:
            same_chunk += 1
        else:
            diff_chunk += 1

    n_queries = len(query_elements)
    print(f"\nPer-query element co-location:")
    print(f"  Both elements in SAME chunk:     {same_chunk} / {n_queries}")
    print(f"  Elements in DIFFERENT chunks:    {diff_chunk} / {n_queries}")
    print(f"  <2 elements mapped to any chunk: {only_one_mapped} / {n_queries}")


if __name__ == "__main__":
    main()
