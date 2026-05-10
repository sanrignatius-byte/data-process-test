#!/usr/bin/env python3
"""DPT project status check — answers questions from 2026-05-02 meeting.

Checks:
  1. Element enrichment coverage (what % of 27209 elements have LLM enrich?)
  2. Element type distribution (paragraph vs figure vs table vs formula)
  3. Hub pair coverage (how many docs/pairs from hub candidates?)
  4. Chunk→element mapping status (how many chunks have element_ids?)
  5. Discard risk estimation

Usage:
  python scripts/dpt_status_check.py
"""

from __future__ import annotations

import json, sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"


def load_json(path: str) -> dict:
    with open(path) as f:
        return json.load(f)


def iter_elements(data) -> list[dict]:
    """Flatten known element JSON layouts into a list of element dicts."""
    if isinstance(data, list):
        return [e for e in data if isinstance(e, dict)]
    if not isinstance(data, dict):
        return []

    if isinstance(data.get("documents"), dict):
        elems: list[dict] = []
        for doc in data["documents"].values():
            if not isinstance(doc, dict):
                continue
            doc_elems = doc.get("elements", {})
            if isinstance(doc_elems, dict):
                elems.extend(e for e in doc_elems.values() if isinstance(e, dict))
            elif isinstance(doc_elems, list):
                elems.extend(e for e in doc_elems if isinstance(e, dict))
        return elems

    elems = data.get("elements")
    if isinstance(elems, dict):
        return [e for e in elems.values() if isinstance(e, dict)]
    if isinstance(elems, list):
        return [e for e in elems if isinstance(e, dict)]

    return [
        v for k, v in data.items()
        if k not in {"metadata", "stats", "summary"} and isinstance(v, dict)
    ]


def check_enrichment():
    """Check element enrichment coverage."""
    print("=" * 65)
    print("1. Element Enrichment Coverage")
    print("=" * 65)

    elem_files = [
        ("production_full", DATA / "02_enriched/multimodal_elements_v2_production_full.json"),
        ("production_partial", DATA / "02_enriched/multimodal_elements_v2_production_partial.json"),
        ("gap227", DATA / "02_enriched/multimodal_elements_v2_gap227.json"),
        ("trial57", DATA / "02_enriched/multimodal_elements_trial57_enriched_full.json"),
    ]

    for label, path in elem_files:
        if not path.exists():
            print(f"  [{label}] MISSING: {path}")
            continue
        data = load_json(str(path))
        elems = iter_elements(data)

        if not elems:
            print(f"  [{label}] Empty or unparseable")
            continue

        enriched = sum(1 for e in elems
                       if (e.get("enriched_content") or e.get("enriched") or "").strip())
        total = len(elems)
        print(f"  [{label}] {enriched}/{total} enriched ({enriched/total*100:.1f}%)")

    # Total enrichment target
    print()
    print(f"  Target: 27209 total elements across 1040 docs")
    print(f"  Strategy: enrich only hub pair elements (~1200) + gap227 (~4301) ≈ ~5500")


def check_element_types():
    """Check element type distribution."""
    print()
    print("=" * 65)
    print("2. Element Type Distribution")
    print("=" * 65)

    path = DATA / "02_enriched/multimodal_elements_v2_production_full.json"
    if not path.exists():
        print("  production_full not found")
        return

    data = load_json(str(path))
    elems = iter_elements(data)

    type_count = Counter()
    for e in elems:
        etype = e.get("element_type", e.get("type", "unknown"))
        type_count[etype] += 1

    total = sum(type_count.values())
    print(f"  Total elements: {total}")
    for etype, count in type_count.most_common():
        print(f"    {etype:<15s}: {count:>6d} ({count/total*100:5.1f}%)")

    # Note: paragraph IS an element type per mentor's instruction
    print()
    print("  Per 2026-05-02 meeting: paragraph IS an element (text element).")
    print("  Element types: paragraph | figure | table | formula — all equal first-class nodes.")


def check_hub_pairs():
    """Check hub pair coverage."""
    print()
    print("=" * 65)
    print("3. Hub Pair Coverage")
    print("=" * 65)

    path = DATA / "02_enriched/hub_candidates_v2_combined.json"
    if not path.exists():
        print("  hub_candidates_v2_combined not found")
        return

    data = load_json(str(path))
    summary = data.get("summary", {})
    print(f"  Total pairs: {data.get('metadata', {}).get('total_pairs', '?')}")
    print(f"  By type: {summary.get('by_type', {})}")
    print(f"  By hop:  {summary.get('by_hop', {})}")
    print(f"  Docs covered: {summary.get('docs_covered', '?')}")

    pairs = data.get("pairs", [])
    if pairs:
        enriched_count = 0
        for p in pairs:
            ea = p.get("element_a", {})
            eb = p.get("element_b", {})
            if ((ea.get("enriched_content") or "").strip() or
                (eb.get("enriched_content") or "").strip()):
                enriched_count += 1
        print(f"  Pairs with enriched elements: {enriched_count}/{len(pairs)} ({enriched_count/len(pairs)*100:.1f}%)")


def check_chunk_graph():
    """Check chunk graph construction status."""
    print()
    print("=" * 65)
    print("4. Chunk Graph Construction Status")
    print("=" * 65)

    for label, path in [
        ("v2 (original)", DATA / "01_graphs/chunk_virtual_nodes_v2.json"),
        ("n400 v2", DATA / "01_graphs/paragraph_chunks_n400_v2.json"),
        ("n500", DATA / "01_graphs/paragraph_chunks_n500.json"),
    ]:
        if not path.exists():
            print(f"  [{label}] MISSING")
            continue
        data = load_json(str(path))
        stats = data.get("stats", {})
        print(f"  [{label}]")
        print(f"    Docs: {stats.get('total_docs_scanned', stats.get('docs_with_chunks', '?'))}")
        print(f"    Chunks: {stats.get('total_chunks', '?')}")
        print(f"    Avg words/chunk: {stats.get('avg_words_per_chunk', '?')}")

        # Check how many chunks have element_ids populated
        docs = data.get("documents", {})
        total_chunks = 0
        chunks_with_elements = 0
        for doc_id, doc in docs.items():
            nodes = doc.get("nodes", {}) if isinstance(doc, dict) else {}
            for nid, node in nodes.items():
                if node.get("chunk_id"):
                    total_chunks += 1
                    eids = node.get("element_ids", []) or []
                    if eids:
                        chunks_with_elements += 1
        if total_chunks > 0:
            print(f"    Chunks with element_ids: {chunks_with_elements}/{total_chunks} ({chunks_with_elements/total_chunks*100:.1f}%)")


def check_graph_augmented_corpus():
    """Check if graph-augmented corpus has been built."""
    print()
    print("=" * 65)
    print("5. Graph-Augmented Corpus Status")
    print("=" * 65)

    corpus_dir = DATA / "05_eval" / "dense_retrieval"
    if not corpus_dir.exists():
        print("  No eval data directory")
        return

    subdirs = sorted(corpus_dir.glob("*"))
    recent = [d for d in subdirs if d.is_dir()][-5:]
    for d in recent:
        corpus_file = d / "corpus.jsonl"
        ranking_file = d / "ranking.jsonl"
        print(f"  {d.name}:")
        print(f"    corpus: {'✅' if corpus_file.exists() else '❌'}")
        print(f"    ranking: {'✅' if ranking_file.exists() else '❌'}")


def main():
    print("DPT Project Status Check — 2026-05-10")
    print("=" * 65)
    print()

    check_enrichment()
    check_element_types()
    check_hub_pairs()
    check_chunk_graph()
    check_graph_augmented_corpus()

    print()
    print("=" * 65)
    print("Key Action Items from 2026-05-02 Meeting:")
    print("=" * 65)
    print("  1. [MANUAL] Rewrite GRAPH_ARCHITECTURE.md based on old agreed-upon doc")
    print("     - Remove AI jargon: scaffold, fallback, 下游通道, 算子, etc.")
    print("     - paragraph IS element (text element) — no paragraph-vs-element split")
    print("     - Use plain language for edge descriptions")
    print("  2. [SCRIPT] Verify element discard rate: python scripts/compute_element_discard_rate.py")
    print("  3. [SCRIPT] Chunk→element stats: python scripts/compute_chunk_element_stats.py")
    print("  4. [EXPERIMENT] Separate modality retrieval: text→text, figure→figure, formula→formula")
    print("  5. [DONE] Smoke50 set built from available M4query modalities (v1 has no text qrels)")
    print("  6. [FIX] Bring back API key for enrich work — root blocker for all enrichment")


if __name__ == "__main__":
    main()
