#!/usr/bin/env python3
"""
Inspect citation_graph.json schema.

Prints top-level keys, edge structure, and adjacency structure so you can
verify which fields hold title / context before running export_review_csv.py.

Usage:
    python scripts/inspect_graph.py
    python scripts/inspect_graph.py --input data/citation_graph.json
"""

import argparse
import json
from pathlib import Path


def _abbrev(v, max_len=120):
    s = repr(v)
    return s[:max_len] + "..." if len(s) > max_len else s


def inspect(path: str) -> None:
    p = Path(path)
    if not p.exists():
        print(f"[ERROR] File not found: {p.resolve()}")
        return

    with open(p, encoding="utf-8") as f:
        data = json.load(f)

    print("=" * 60)
    print(f"File: {p.resolve()}")
    print("=" * 60)

    # --- top-level ---
    top_keys = list(data.keys())
    print(f"\nTop-level keys ({len(top_keys)}): {top_keys}")

    for k in top_keys:
        v = data[k]
        if isinstance(v, list):
            print(f"  {k!r}: list, len={len(v)}")
        elif isinstance(v, dict):
            print(f"  {k!r}: dict, len={len(v)}")
        else:
            print(f"  {k!r}: {type(v).__name__} = {_abbrev(v)}")

    # --- edges ---
    edges = data.get("edges", [])
    print(f"\n{'='*60}")
    print(f"edges: {len(edges)} total")
    if edges:
        e = edges[0]
        print(f"\nFirst edge keys: {list(e.keys())}")
        for k, v in e.items():
            print(f"  edge[{k!r}] = {_abbrev(v)}")

        # Show a few more for context variety
        if len(edges) > 1:
            e2 = edges[min(10, len(edges) - 1)]
            print(f"\nEdge #{min(10, len(edges)-1)} keys: {list(e2.keys())}")
            for k, v in e2.items():
                print(f"  edge[{k!r}] = {_abbrev(v)}")

    # --- adjacency ---
    adj = data.get("adjacency", {})
    print(f"\n{'='*60}")
    if isinstance(adj, dict):
        print(f"adjacency: dict, {len(adj)} entries")
        if adj:
            k0 = next(iter(adj))
            print(f"  sample key: {k0!r}")
            print(f"  sample value: {_abbrev(adj[k0])}")
    elif isinstance(adj, list):
        print(f"adjacency: list, len={len(adj)}")
        if adj:
            print(f"  sample[0]: {_abbrev(adj[0])}")

    # --- statistics summary ---
    stats = data.get("statistics", {})
    if stats:
        print(f"\n{'='*60}")
        print("statistics:")
        for k, v in stats.items():
            print(f"  {k}: {v}")

    # --- field path verdict ---
    print(f"\n{'='*60}")
    print("FIELD PATH VERDICT")
    print("=" * 60)
    if edges:
        e = edges[0]
        has_bib_title  = "bib_title"    in e
        has_contexts   = "contexts"     in e
        has_match_meth = "match_method" in e
        has_confidence = "confidence"   in e
        has_meta       = "metadata"     in e

        print(f"  edge['bib_title']    present: {has_bib_title}")
        print(f"  edge['contexts']     present: {has_contexts}")
        print(f"  edge['match_method'] present: {has_match_meth}")
        print(f"  edge['confidence']   present: {has_confidence}")
        print(f"  edge['metadata']     present: {has_meta}")

        if has_meta:
            meta = e.get("metadata", {})
            if isinstance(meta, dict):
                print(f"  edge['metadata'] keys: {list(meta.keys())}")


def main():
    ap = argparse.ArgumentParser(description="Inspect citation_graph.json schema")
    ap.add_argument(
        "--input",
        default="data/citation_graph.json",
        help="Path to citation_graph.json",
    )
    args = ap.parse_args()
    inspect(args.input)


if __name__ == "__main__":
    main()
