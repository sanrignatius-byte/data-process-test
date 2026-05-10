#!/usr/bin/env python3
"""Compute element discard rate: how many elements are lost during element↔chunk matching.

Verifies the ~50% discard rate concern raised in the 2026-05-02 meeting.
Compares: total elements in multimodal_elements vs elements successfully matched to chunks.

Usage:
  python scripts/compute_element_discard_rate.py \
    --elements data/02_enriched/multimodal_elements_v2_production_full.json \
    --chunks data/01_graphs/chunk_virtual_nodes_v2.json
"""

from __future__ import annotations

import argparse, json, sys
from collections import defaultdict, Counter
from pathlib import Path


def load_elements(path: str) -> list[dict]:
    with open(path) as f:
        data = json.load(f)
    if isinstance(data, list):
        return data
    if isinstance(data, dict):
        # Try common keys
        for key in ["elements", "data", "items"]:
            if key in data:
                return data[key]
        # Maybe keyed by doc_id
        vals = list(data.values())
        if vals and isinstance(vals[0], list):
            result = []
            for v in vals:
                if isinstance(v, list):
                    result.extend(v)
            return result
    return []


def load_chunks(path: str) -> dict[str, list[dict]]:
    """Return {doc_id: [chunk_dict, ...]}."""
    with open(path) as f:
        data = json.load(f)
    docs = data.get("documents", {})
    result = {}
    for doc_id, doc_data in docs.items():
        if isinstance(doc_data, list):
            result[doc_id] = doc_data
        elif isinstance(doc_data, dict) and "chunk_id" in doc_data:
            result[doc_id] = [doc_data]
        elif isinstance(doc_data, dict):
            # doc_data has metadata + chunk_id-keyed chunks
            chunks = [v for k, v in doc_data.items() if isinstance(v, dict) and "chunk_id" in v]
            result[doc_id] = chunks
    return result


def main():
    parser = argparse.ArgumentParser(description="Compute element discard rate")
    parser.add_argument("--elements", required=True)
    parser.add_argument("--chunks", required=True)
    args = parser.parse_args()

    elements = load_elements(args.elements)
    chunks_by_doc = load_chunks(args.chunks)

    if not elements:
        print("ERROR: No elements loaded", file=sys.stderr)
        sys.exit(1)
    if not chunks_by_doc:
        print("ERROR: No chunks loaded", file=sys.stderr)
        sys.exit(1)

    # Index elements by doc_id
    elems_by_doc: dict[str, list[dict]] = defaultdict(list)
    for e in elements:
        doc_id = e.get("doc_id", e.get("document_id", ""))
        elems_by_doc[str(doc_id)].append(e)

    # Index chunk element_ids
    chunk_elem_ids: dict[str, set[str]] = defaultdict(set)
    total_chunks = 0
    for doc_id, clist in chunks_by_doc.items():
        for c in clist:
            total_chunks += 1
            eids = c.get("element_ids", []) or []
            for eid in eids:
                chunk_elem_ids[doc_id].add(str(eid))

    # Compute per-element-type statistics
    type_total = Counter()
    type_matched = Counter()
    type_enriched = Counter()

    all_elem_ids = set()
    matched_elem_ids = set()

    for doc_id, elist in elems_by_doc.items():
        for e in elist:
            eid = str(e.get("element_id", e.get("id", "")))
            if not eid:
                continue
            all_elem_ids.add(eid)
            etype = e.get("element_type", e.get("type", "unknown"))
            type_total[etype] += 1

            if eid in chunk_elem_ids.get(doc_id, set()):
                matched_elem_ids.add(eid)
                type_matched[etype] += 1

            # Check enrichment
            enriched = e.get("enriched_content", e.get("enriched", ""))
            if enriched and enriched.strip():
                type_enriched[etype] += 1

    # Print report
    n_total = len(all_elem_ids)
    n_matched = len(matched_elem_ids)
    n_discarded = n_total - n_matched
    discard_rate = n_discarded / n_total if n_total else 0

    print("=" * 60)
    print("Element Discard Rate Report")
    print("=" * 60)
    print(f"  Elements file: {args.elements}")
    print(f"  Chunks file:    {args.chunks}")
    print(f"  Total docs with chunks: {len(chunks_by_doc)}")
    print(f"  Total chunks: {total_chunks}")
    print()
    print(f"  Total elements:    {n_total}")
    print(f"  Matched to chunks: {n_matched}")
    print(f"  Discarded:         {n_discarded}")
    print(f"  Discard rate:      {discard_rate:.2%}")
    print()
    print("  By element type:")
    print(f"  {'Type':<12s} {'Total':>8s} {'Matched':>8s} {'Discard%':>10s} {'Enriched%':>10s}")
    print("  " + "-" * 50)
    for etype in sorted(type_total.keys()):
        t = type_total[etype]
        m = type_matched.get(etype, 0)
        d = t - m
        dr = d / t if t else 0
        er = type_enriched.get(etype, 0) / t if t else 0
        print(f"  {etype:<12s} {t:>8d} {m:>8d} {dr:>9.1%} {er:>9.1%}")

    print()
    print("  By doc_id (top-10 worst discard):")
    doc_discard = []
    for doc_id in elems_by_doc:
        total = len(elems_by_doc[doc_id])
        matched = sum(1 for e in elems_by_doc[doc_id]
                      if str(e.get("element_id", "")) in chunk_elem_ids.get(doc_id, set()))
        discarded = total - matched
        if total > 0:
            doc_discard.append((doc_id, total, matched, discarded, discarded / total))
    doc_discard.sort(key=lambda x: -x[3])  # worst discard first
    for doc_id, total, matched, discarded, rate in doc_discard[:10]:
        print(f"  {doc_id}: {discarded}/{total} discarded ({rate:.1%})")

    print()
    verdict = "PASS" if discard_rate < 0.15 else ("WARN" if discard_rate < 0.30 else "FAIL")
    print(f"  Verdict: {verdict} (threshold: <15% pass, <30% warn, >=30% fail)")
    print(f"  Target: verify the ~50% claim from meeting → actual is {discard_rate:.1%}")


if __name__ == "__main__":
    main()
