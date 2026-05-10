#!/usr/bin/env python3
"""Calculate per-chunk element statistics and estimate true element recall under chunk retrieval.

Answers the mentor's question: "一个chunk平均有几个element？你用R@10的chunk实际recall了几个element？"

Usage:
  python scripts/compute_chunk_element_stats.py \
    --chunks data/01_graphs/chunk_virtual_nodes_v2.json \
    --elements data/02_enriched/multimodal_elements_v2_production_full.json
"""

from __future__ import annotations

import argparse, json, sys
from collections import defaultdict
from pathlib import Path


def load_json(path: str) -> dict:
    with open(path) as f:
        return json.load(f)


def main():
    parser = argparse.ArgumentParser(description="Chunk-element mapping statistics")
    parser.add_argument("--chunks", required=True)
    parser.add_argument("--elements", required=True)
    parser.add_argument("--qrels", help="Optional qrels file for per-query element recall")
    args = parser.parse_args()

    chunk_data = load_json(args.chunks)
    elem_data = load_json(args.elements)

    # Parse chunks
    docs = chunk_data.get("documents", {})
    all_chunks = []
    elem_to_chunks: dict[str, list[str]] = defaultdict(list)
    chunk_elem_counts: list[int] = []

    for doc_id, doc_chunks in docs.items():
        if isinstance(doc_chunks, list):
            clist = doc_chunks
        elif isinstance(doc_chunks, dict):
            clist = [v for v in doc_chunks.values() if isinstance(v, dict) and "chunk_id" in v]
        else:
            continue

        for c in clist:
            chunk_id = c.get("chunk_id", "?")
            eids = c.get("element_ids", []) or []
            chunk_elem_counts.append(len(eids))
            all_chunks.append({"chunk_id": chunk_id, "doc_id": doc_id, "n_elements": len(eids)})
            for eid in eids:
                elem_to_chunks[str(eid)].append(chunk_id)

    # Parse elements
    if isinstance(elem_data, list):
        elements = elem_data
    elif isinstance(elem_data, dict):
        elements = elem_data.get("elements", list(elem_data.values()))
    else:
        elements = []

    total_elements = len(elements)
    elements_with_chunk = sum(1 for e in elements
                              if str(e.get("element_id", "")) in elem_to_chunks)

    # Stats
    n_chunks = len(chunk_elem_counts)
    avg_elems = sum(chunk_elem_counts) / n_chunks if n_chunks else 0
    median_elems = sorted(chunk_elem_counts)[n_chunks // 2] if n_chunks else 0
    chunks_with_zero = sum(1 for c in chunk_elem_counts if c == 0)
    chunks_with_one = sum(1 for c in chunk_elem_counts if c == 1)
    chunks_with_many = sum(1 for c in chunk_elem_counts if c >= 5)

    print("=" * 60)
    print("Chunk → Element Mapping Statistics")
    print("=" * 60)
    print(f"  Total chunks:          {n_chunks}")
    print(f"  Total elements:        {total_elements}")
    print(f"  Elements in any chunk: {elements_with_chunk}/{total_elements} ({elements_with_chunk/total_elements:.1%})")
    print()
    print(f"  Elements per chunk:")
    print(f"    Mean:   {avg_elems:.2f}")
    print(f"    Median: {median_elems}")
    print(f"    Min:    {min(chunk_elem_counts) if chunk_elem_counts else 0}")
    print(f"    Max:    {max(chunk_elem_counts) if chunk_elem_counts else 0}")
    print()
    print(f"  Distribution:")
    print(f"    0 elements:  {chunks_with_zero:>6d} ({chunks_with_zero/n_chunks*100:5.1f}%)")
    print(f"    1 element:   {chunks_with_one:>6d} ({chunks_with_one/n_chunks*100:5.1f}%)")
    print(f"    2-4 elements:{n_chunks-chunks_with_zero-chunks_with_one-chunks_with_many:>6d}")
    print(f"    5+ elements: {chunks_with_many:>6d} ({chunks_with_many/n_chunks*100:5.1f}%)")
    print()

    # Estimate: if we retrieve top-K chunks, how many elements do we actually get?
    print("  Estimated element recall from chunk retrieval (assuming uniform):")
    print(f"  {'K':>5s}  {'Avg Elements':>13s}  {'Element Recall':>15s}")
    print("  " + "-" * 37)
    for k in [1, 5, 10, 15, 20, 50]:
        # Conservative: assume chunks are random wrt element content
        prob_elem_in_k = min(1.0, k / n_chunks) if n_chunks else 0
        expected_elems = prob_elem_in_k * elements_with_chunk
        print(f"  {k:>5d}  {expected_elems:>13.1f}  {prob_elem_in_k:>14.1%}")

    print()
    print("  Interpretation:")
    print(f"  - On average, each chunk contains {avg_elems:.1f} elements")
    print(f"  - {chunks_with_zero/n_chunks*100:.1f}% of chunks contain zero elements → pure text passages")
    print(f"  - R@10 chunks ≈ covers at most {min(1.0, 10/n_chunks)*100:.1f}% of all elements")
    print(f"    (actual coverage depends on query→chunk relevance distribution)")

    # Per-doc summary
    doc_stats = defaultdict(lambda: {"chunks": 0, "elements": 0})
    for doc_id, doc_chunks in docs.items():
        if isinstance(doc_chunks, list):
            clist = doc_chunks
        elif isinstance(doc_chunks, dict):
            clist = [v for v in doc_chunks.values() if isinstance(v, dict) and "chunk_id" in v]
        else:
            continue
        doc_stats[doc_id]["chunks"] = len(clist)
        doc_stats[doc_id]["elements"] = sum(len(c.get("element_ids", [])) for c in clist)

    print()
    print(f"  Per-document (top-5 largest):")
    ranked = sorted(doc_stats.items(), key=lambda x: -x[1]["elements"])
    for doc_id, ds in ranked[:5]:
        avg = ds["elements"] / max(ds["chunks"], 1)
        print(f"  {doc_id}: {ds['chunks']} chunks, {ds['elements']} elements, avg {avg:.1f} elem/chunk")


if __name__ == "__main__":
    main()
