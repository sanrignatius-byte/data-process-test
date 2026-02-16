#!/usr/bin/env python3
"""
Build Cross-Document Citation Graph

Reads per-document LaTeX reference graphs and constructs a cross-document
citation network by matching bibliography entries (.bbl) to papers in the
corpus (our 73+ arXiv papers).

Matching strategies (in priority order):
  1. arXiv ID found in .bbl raw text         → confidence 1.0
  2. Bare arXiv ID pattern in .bbl text       → confidence 0.9
  3. Exact normalized title match             → confidence 0.95
  4. Fuzzy title match (Jaccard ≥ 0.55)       → confidence = similarity

Usage:
    # From pre-built graph JSON (recommended: run build_latex_reference_graph.py first)
    python scripts/build_citation_graph.py

    # From LaTeX sources directly
    python scripts/build_citation_graph.py --from-sources data/latex_sources/extracted

    # Custom paths
    python scripts/build_citation_graph.py \
        --input data/latex_reference_graph.json \
        --output data/citation_graph.json
"""

import argparse
import json
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


# ---------------------------------------------------------------------------
# arXiv ID regex for matching bib entries to corpus
# ---------------------------------------------------------------------------

# Explicit: arXiv:YYMM.NNNNN or arxiv.org/abs/YYMM.NNNNN
RE_ARXIV_EXPLICIT = re.compile(
    r'(?:arXiv[:\s]+|arxiv\.org/(?:abs|pdf)/)(\d{4}\.\d{4,5}(?:v\d+)?)',
    re.IGNORECASE,
)

# Old-style arXiv: e.g. hep-ph/0301234 (less common in our corpus)
RE_ARXIV_OLD = re.compile(
    r'(?:arXiv[:\s]+|arxiv\.org/(?:abs|pdf)/)([a-z-]+/\d{7})',
    re.IGNORECASE,
)

# Bare numeric IDs: YYMM.NNNNN surrounded by word boundaries
RE_ARXIV_BARE = re.compile(r'\b(\d{4}\.\d{4,5})\b')


# ---------------------------------------------------------------------------
# Title normalization & similarity
# ---------------------------------------------------------------------------

def normalize_title(title: str) -> str:
    """Normalize a title for comparison: lowercase, strip LaTeX, strip punct."""
    if not title:
        return ""
    t = title.lower()
    t = re.sub(r'\\[a-zA-Z]+\{([^}]*)\}', r'\1', t)
    t = re.sub(r'\\[a-zA-Z]+', '', t)
    t = re.sub(r'[{}$\\~^_]', '', t)
    t = re.sub(r"[^\w\s'-]", '', t)
    t = re.sub(r'\s+', ' ', t).strip()
    return t


def title_jaccard(t1: str, t2: str) -> float:
    """Word-level Jaccard similarity between two normalized titles."""
    w1 = set(t1.split())
    w2 = set(t2.split())
    if not w1 or not w2:
        return 0.0
    return len(w1 & w2) / len(w1 | w2)


# ---------------------------------------------------------------------------
# Single bib entry → corpus matching
# ---------------------------------------------------------------------------

def match_bib_entry(
    raw: str,
    bib_title: Optional[str],
    corpus_ids: Set[str],
    corpus_titles: Dict[str, str],
    title_to_id: Dict[str, str],
) -> Tuple[Optional[str], str, float]:
    """
    Match a single bibliography entry to a corpus paper.

    Returns:
        (matched_arxiv_id, match_method, confidence)
    """
    # Strategy 1: explicit arXiv ID in raw bbl text
    for m in RE_ARXIV_EXPLICIT.finditer(raw):
        aid = re.sub(r'v\d+$', '', m.group(1))
        if aid in corpus_ids:
            return (aid, "arxiv_id_explicit", 1.0)

    # Strategy 2: bare arXiv ID (validate year/month range)
    for m in RE_ARXIV_BARE.finditer(raw):
        aid = m.group(1)
        try:
            yy, mm = int(aid[:2]), int(aid[2:4])
            if 10 <= yy <= 26 and 1 <= mm <= 12 and aid in corpus_ids:
                return (aid, "arxiv_id_bare", 0.9)
        except ValueError:
            continue

    # Strategy 3: title matching
    if bib_title:
        norm = normalize_title(bib_title)
        if norm and len(norm) > 10:
            # Exact normalized match
            if norm in title_to_id:
                return (title_to_id[norm], "title_exact", 0.95)

            # Fuzzy match
            best_id = None
            best_sim = 0.0
            for aid, ct in corpus_titles.items():
                sim = title_jaccard(norm, ct)
                if sim > best_sim:
                    best_sim = sim
                    best_id = aid

            if best_sim >= 0.55 and best_id:
                return (best_id, "title_fuzzy", round(best_sim, 3))

    return (None, "", 0.0)


# ---------------------------------------------------------------------------
# Citation graph builder
# ---------------------------------------------------------------------------

def build_citation_graph(graph_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Build cross-document citation graph.

    Input:  latex_reference_graph.json (with per-doc bib entries and refs)
    Output: citation edges, adjacency, statistics
    """
    documents = graph_data.get("documents", {})
    corpus_ids = set(documents.keys())

    # --- Build corpus title index ---
    corpus_titles: Dict[str, str] = {}
    title_to_id: Dict[str, str] = {}

    for doc_id, doc in documents.items():
        meta = doc.get("metadata", {})
        title = meta.get("title")
        if title:
            norm = normalize_title(title)
            if norm and len(norm) > 5:
                corpus_titles[doc_id] = norm
                # Handle collisions: keep first
                if norm not in title_to_id:
                    title_to_id[norm] = doc_id

    print(f"Corpus: {len(corpus_ids)} papers, {len(corpus_titles)} with extracted titles")
    if len(corpus_titles) < len(corpus_ids) * 0.5:
        print(f"  WARNING: <50% papers have titles. Title matching will be limited.")
        print(f"  Make sure build_latex_reference_graph.py was run with latest extractor")
        print(f"  (which includes title extraction from \\title{{}}).")
    print()

    # --- Match bib entries → corpus ---
    edges: List[Dict[str, Any]] = []
    edge_index: Dict[Tuple[str, str], int] = {}  # (src, tgt) → edge list index
    match_stats: Dict[str, int] = defaultdict(int)
    unmatched_sample: List[Dict[str, Any]] = []
    total_bib = 0

    for doc_id, doc in documents.items():
        bib = doc.get("bib", {})
        refs = doc.get("refs", [])

        # Index cite-type refs by target_key for context lookup
        cite_contexts: Dict[str, List[str]] = defaultdict(list)
        for ref in refs:
            if ref.get("ref_type") == "cite":
                ctx = ref.get("context", "")[:200]
                if ctx:
                    cite_contexts[ref["target_key"]].append(ctx)

        for cite_key, entry in bib.items():
            total_bib += 1
            raw = entry.get("raw", "") or ""
            bib_title = entry.get("title")

            matched_id, method, confidence = match_bib_entry(
                raw, bib_title, corpus_ids, corpus_titles, title_to_id,
            )

            if matched_id and matched_id != doc_id:
                match_stats[method] += 1
                pair = (doc_id, matched_id)

                if pair in edge_index:
                    # Augment existing edge
                    idx = edge_index[pair]
                    if cite_key not in edges[idx]["cite_keys"]:
                        edges[idx]["cite_keys"].append(cite_key)
                    edges[idx]["cite_count"] += len(cite_contexts.get(cite_key, []))
                else:
                    # New edge
                    edge_index[pair] = len(edges)
                    edges.append({
                        "source": doc_id,
                        "target": matched_id,
                        "cite_keys": [cite_key],
                        "match_method": method,
                        "confidence": confidence,
                        "bib_title": bib_title,
                        "cite_count": len(cite_contexts.get(cite_key, [])),
                        "contexts": cite_contexts.get(cite_key, [])[:5],
                    })
            else:
                match_stats["unmatched"] += 1
                if len(unmatched_sample) < 100 and bib_title:
                    unmatched_sample.append({
                        "doc_id": doc_id,
                        "cite_key": cite_key,
                        "title": bib_title,
                        "year": entry.get("year"),
                    })

    # --- Build adjacency ---
    adjacency: Dict[str, Dict[str, List[str]]] = {}
    for doc_id in corpus_ids:
        adjacency[doc_id] = {"cites": [], "cited_by": []}

    for edge in edges:
        s, t = edge["source"], edge["target"]
        if t not in adjacency[s]["cites"]:
            adjacency[s]["cites"].append(t)
        if s not in adjacency[t]["cited_by"]:
            adjacency[t]["cited_by"].append(s)

    # --- Compute statistics ---
    out_degrees = [len(v["cites"]) for v in adjacency.values()]
    in_degrees = [len(v["cited_by"]) for v in adjacency.values()]

    papers_citing = sum(1 for d in out_degrees if d > 0)
    papers_cited = sum(1 for d in in_degrees if d > 0)
    papers_isolated = sum(
        1 for doc_id in corpus_ids
        if not adjacency[doc_id]["cites"] and not adjacency[doc_id]["cited_by"]
    )

    # Connected components (undirected)
    neighbors: Dict[str, Set[str]] = defaultdict(set)
    for edge in edges:
        neighbors[edge["source"]].add(edge["target"])
        neighbors[edge["target"]].add(edge["source"])

    visited: Set[str] = set()
    components: List[Set[str]] = []
    for node in corpus_ids:
        if node not in visited:
            comp: Set[str] = set()
            stack = [node]
            while stack:
                n = stack.pop()
                if n in visited:
                    continue
                visited.add(n)
                comp.add(n)
                for nb in neighbors.get(n, set()):
                    if nb not in visited:
                        stack.append(nb)
            components.append(comp)
    components.sort(key=len, reverse=True)

    # Degree distributions
    def _percentiles(values: List[int]) -> Dict[str, float]:
        if not values:
            return {}
        s = sorted(values)
        n = len(s)
        return {
            "min": s[0],
            "p25": s[max(0, n * 25 // 100)],
            "p50": s[max(0, n * 50 // 100)],
            "p75": s[max(0, n * 75 // 100)],
            "p90": s[max(0, n * 90 // 100)],
            "max": s[-1],
            "mean": round(sum(s) / n, 2),
        }

    # Top cited papers
    top_cited = sorted(
        [(doc_id, len(adjacency[doc_id]["cited_by"])) for doc_id in corpus_ids],
        key=lambda x: -x[1],
    )[:15]

    # Top citers (papers that cite the most others in corpus)
    top_citers = sorted(
        [(doc_id, len(adjacency[doc_id]["cites"])) for doc_id in corpus_ids],
        key=lambda x: -x[1],
    )[:15]

    statistics = {
        "total_bib_entries": total_bib,
        "total_citation_edges": len(edges),
        "total_cite_key_links": sum(len(e["cite_keys"]) for e in edges),
        "papers_citing_corpus": papers_citing,
        "papers_cited_by_corpus": papers_cited,
        "papers_isolated": papers_isolated,
        "match_method_dist": dict(match_stats),
        "match_rate": round(
            (total_bib - match_stats.get("unmatched", 0)) / max(1, total_bib), 4
        ),
        "out_degree_dist": _percentiles(out_degrees),
        "in_degree_dist": _percentiles(in_degrees),
        "connected_components": len(components),
        "largest_component": len(components[0]) if components else 0,
        "component_sizes": [len(c) for c in components[:10]],
    }

    # Only include non-empty adjacency entries
    compact_adj = {
        k: v for k, v in adjacency.items()
        if v["cites"] or v["cited_by"]
    }

    return {
        "metadata": {
            "corpus_size": len(corpus_ids),
            "papers_with_titles": len(corpus_titles),
            "total_bib_entries_scanned": total_bib,
        },
        "edges": edges,
        "adjacency": compact_adj,
        "statistics": statistics,
        "top_cited": [
            {"arxiv_id": aid, "cited_by_count": cnt, "title": corpus_titles.get(aid, "")}
            for aid, cnt in top_cited if cnt > 0
        ],
        "top_citers": [
            {"arxiv_id": aid, "cites_count": cnt, "title": corpus_titles.get(aid, "")}
            for aid, cnt in top_citers if cnt > 0
        ],
        "unmatched_sample": unmatched_sample[:50],
    }


# ---------------------------------------------------------------------------
# Print summary
# ---------------------------------------------------------------------------

def print_summary(result: Dict[str, Any]) -> None:
    stats = result["statistics"]

    print(f"\n{'=' * 60}")
    print(f"CROSS-DOCUMENT CITATION GRAPH")
    print(f"{'=' * 60}")
    print(f"Corpus size:           {result['metadata']['corpus_size']} papers")
    print(f"Papers with titles:    {result['metadata']['papers_with_titles']}")
    print(f"Bib entries scanned:   {stats['total_bib_entries']}")
    print(f"Matched to corpus:     {stats['total_citation_edges']} unique pairs "
          f"({stats['total_cite_key_links']} cite-key links)")
    print(f"Match rate:            {stats['match_rate']:.1%}")
    print(f"Papers citing corpus:  {stats['papers_citing_corpus']}")
    print(f"Papers cited by corpus:{stats['papers_cited_by_corpus']}")
    print(f"Isolated papers:       {stats['papers_isolated']}")
    print()

    print(f"Match method distribution:")
    for method, count in sorted(
        stats["match_method_dist"].items(), key=lambda x: -x[1]
    ):
        print(f"  {method:25s}  {count:5d}")
    print()

    od = stats.get("out_degree_dist", {})
    id_ = stats.get("in_degree_dist", {})
    if od:
        print(f"Out-degree (cites):     "
              f"mean={od['mean']:.1f}  p50={od['p50']}  p90={od['p90']}  max={od['max']}")
    if id_:
        print(f"In-degree (cited-by):   "
              f"mean={id_['mean']:.1f}  p50={id_['p50']}  p90={id_['p90']}  max={id_['max']}")
    print()

    print(f"Connected components:  {stats['connected_components']}")
    print(f"Largest component:     {stats['largest_component']} papers")
    if stats["component_sizes"]:
        print(f"Component sizes:       {stats['component_sizes']}")
    print()

    if result["top_cited"]:
        print(f"Top cited papers (in-degree):")
        for tc in result["top_cited"][:8]:
            t = tc.get("title", "")
            t_short = (t[:50] + "...") if len(t) > 50 else t
            print(f"  {tc['arxiv_id']:15s}  cited by {tc['cited_by_count']:2d}  {t_short}")
        print()

    if result["top_citers"]:
        print(f"Top citers (out-degree):")
        for tc in result["top_citers"][:8]:
            t = tc.get("title", "")
            t_short = (t[:50] + "...") if len(t) > 50 else t
            print(f"  {tc['arxiv_id']:15s}  cites {tc['cites_count']:2d}  {t_short}")
        print()

    # Show edge examples
    print(f"--- Citation Edge Examples (first 10) ---")
    for edge in result["edges"][:10]:
        ctx = edge["contexts"][0][:80] if edge["contexts"] else "(no context)"
        print(f"  {edge['source']} → {edge['target']}  "
              f"[{edge['match_method']}, conf={edge['confidence']:.2f}]")
        print(f"    keys: {edge['cite_keys']}")
        if edge["bib_title"]:
            print(f"    title: {edge['bib_title'][:80]}")
        print(f"    ctx:   {ctx}")
        print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Build cross-document citation graph from LaTeX reference data"
    )
    parser.add_argument(
        "--input",
        default="data/latex_reference_graph.json",
        help="Input: LaTeX reference graph JSON (from build_latex_reference_graph.py)",
    )
    parser.add_argument(
        "--output",
        default="data/citation_graph.json",
        help="Output: citation graph JSON",
    )
    parser.add_argument(
        "--from-sources",
        default=None,
        help="Process directly from LaTeX source dir (instead of pre-built JSON)",
    )
    args = parser.parse_args()

    # Load or build graph data
    if args.from_sources:
        source_dir = Path(args.from_sources)
        if not source_dir.exists():
            print(f"ERROR: Source directory not found: {source_dir}")
            sys.exit(1)

        from src.parsers.latex_reference_extractor import LaTeXReferenceExtractor

        print(f"Processing LaTeX sources from {source_dir} ...")
        extractor = LaTeXReferenceExtractor()
        documents: Dict[str, Any] = {}

        for d in sorted(source_dir.iterdir()):
            if d.is_dir() and not d.name.startswith("."):
                doc_id = d.name
                graph = extractor.extract(doc_id=doc_id, extract_dir=d)
                if "error" not in graph.metadata:
                    documents[doc_id] = graph.to_dict()
                    print(f"  [{doc_id}] labels={len(graph.labels)} bib={len(graph.bib)}"
                          f"  title={graph.metadata.get('title', '(none)')[:50]}")

        graph_data: Dict[str, Any] = {"documents": documents}
        print(f"\nProcessed {len(documents)} papers")
    else:
        input_path = Path(args.input)
        if not input_path.exists():
            print(f"ERROR: Input not found: {input_path}")
            print(f"  Run build_latex_reference_graph.py first, or use --from-sources.")
            sys.exit(1)

        print(f"Loading {input_path} ...")
        with open(input_path, "r", encoding="utf-8") as f:
            graph_data = json.load(f)
        print(f"Loaded {len(graph_data.get('documents', {}))} documents")

    # Build citation graph
    result = build_citation_graph(graph_data)

    # Print summary
    print_summary(result)

    # Save
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    print(f"\nSaved to: {output_path}")


if __name__ == "__main__":
    main()
