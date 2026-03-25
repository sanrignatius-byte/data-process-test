#!/usr/bin/env python3
"""
Build Unified Two-Layer Graph

Combines:
  1. Citation edges (Layer 1, doc-level) from citation_graph.json
  2. Intra-doc reference edges (Layer 2, element-level) from latex_reference_graph.json
  3. MinerU multimodal elements (Layer 2, enrichment) from multimodal_elements.json

Then finds cross-document, cross-modal paths for L2/L3 query generation.

Usage:
    # Basic: build graph and find paths
    python scripts/build_unified_graph.py

    # With hub suppression
    python scripts/build_unified_graph.py --suppress-hubs --max-degree 20

    # Custom paths
    python scripts/build_unified_graph.py \
        --citation data/citation_graph.json \
        --latex-ref data/latex_reference_graph.json \
        --elements data/multimodal_elements.json \
        --output data/unified_graph.json

    # Only export paths (skip graph export)
    python scripts/build_unified_graph.py --paths-only --max-hops 4 --min-score 0.35
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.linkers.unified_graph import UnifiedGraph, NodeType


# 中文注释：加载 JSON 文件，不存在时返回空字典。
def load_json(path: str) -> Dict[str, Any]:
    """Load a JSON file, return empty dict if not found."""
    p = Path(path)
    if not p.exists():
        print(f"  WARNING: {path} not found, skipping")
        return {}
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)


# 中文注释：主流程入口，负责解析参数并串联整体执行。
def main():
    parser = argparse.ArgumentParser(
        description="Build unified two-layer graph from citation + LaTeX + MinerU data"
    )
    parser.add_argument(
        "--citation", default="data/citation_graph.json",
        help="Citation graph JSON (from build_citation_graph.py)",
    )
    parser.add_argument(
        "--latex-ref", default="data/latex_reference_graph.json",
        help="LaTeX reference graph JSON (from build_latex_reference_graph.py)",
    )
    parser.add_argument(
        "--elements", default="data/multimodal_elements.json",
        help="Multimodal elements JSON (from build_multimodal_relationships.py).",
    )
    parser.add_argument(
        "--output", default="data/unified_graph.json",
        help="Output path for the unified graph",
    )
    parser.add_argument(
        "--paths-output", default="data/unified_graph_paths.json",
        help="Output path for scored cross-doc paths",
    )
    parser.add_argument(
        "--hubs-output", default="data/unified_graph_hubs.json",
        help="Output path for ranked intra-doc and cross-doc traffic hubs",
    )
    parser.add_argument(
        "--max-hops", type=int, default=5,
        help="Maximum hops in cross-doc paths",
    )
    parser.add_argument(
        "--min-score", type=float, default=0.02,
        help="Minimum path score to keep",
    )
    parser.add_argument(
        "--max-paths", type=int, default=500,
        help="Maximum number of paths to find",
    )
    parser.add_argument(
        "--max-start-nodes", type=int, default=0,
        help="Cap number of element start nodes for path search (0 = all)",
    )
    parser.add_argument(
        "--neighbor-limit", type=int, default=0,
        help="Cap neighbor expansions per DFS step (0 = no cap)",
    )
    parser.add_argument(
        "--hub-top-docs", type=int, default=20,
        help="How many cross-document hubs to keep",
    )
    parser.add_argument(
        "--hub-top-elements", type=int, default=8,
        help="How many top elements to keep per document",
    )
    parser.add_argument(
        "--hub-min-edge-confidence", type=float, default=0.0,
        help="Only count edges above this confidence when ranking hubs",
    )
    parser.add_argument(
        "--suppress-hubs", action="store_true",
        help="Enable hub node suppression (B4)",
    )
    parser.add_argument(
        "--max-degree", type=int, default=20,
        help="Degree threshold for hub suppression",
    )
    parser.add_argument(
        "--disable-alignment", action="store_true",
        help="Disable LaTeX↔MinerU element alignment edges",
    )
    parser.add_argument(
        "--alignment-threshold", type=float, default=0.55,
        help="Caption similarity threshold for LaTeX↔MinerU alignment",
    )
    parser.add_argument(
        "--paths-only", action="store_true",
        help="Only export paths, skip full graph export",
    )
    parser.add_argument(
        "--stats-only", action="store_true",
        help="Only print statistics, no path finding",
    )
    args = parser.parse_args()

    # ---------------------------------------------------------------
    # 1. Load data sources
    # ---------------------------------------------------------------
    print("=" * 60)
    print("BUILDING UNIFIED TWO-LAYER GRAPH")
    print("=" * 60)
    print()

    print("Loading data sources...")
    citation_data = load_json(args.citation)
    latex_ref_data = load_json(args.latex_ref)
    element_data = load_json(args.elements) if args.elements else None
    if (
        isinstance(element_data, dict)
        and "documents" in element_data
        and isinstance(element_data.get("documents"), dict)
    ):
        wrapped_count = len(element_data["documents"])
        print(f"  Detected wrapped MinerU format: using documents map ({wrapped_count} docs)")
        element_data = element_data["documents"]

    if not citation_data and not latex_ref_data:
        print("ERROR: Need at least one of citation_graph.json or latex_reference_graph.json")
        sys.exit(1)

    # ---------------------------------------------------------------
    # 2. Build unified graph
    # ---------------------------------------------------------------
    graph = UnifiedGraph()

    print("\nLayer 1: Loading citation edges (doc-level)...")
    n_citation = graph.load_citation_edges(citation_data)
    print(f"  Added {n_citation} citation edges")

    print("\nLayer 2: Loading intra-doc edges (element-level)...")
    n_intra = graph.load_intra_doc_edges(
        latex_ref_data,
        element_data,
        enable_alignment=not args.disable_alignment,
        alignment_threshold=args.alignment_threshold,
    )
    print(f"  Added {n_intra} intra-doc edges")

    # ---------------------------------------------------------------
    # 3. Hub suppression (optional)
    # ---------------------------------------------------------------
    if args.suppress_hubs:
        print(f"\nHub suppression (max_degree={args.max_degree})...")
        hubs = graph.identify_hubs(max_degree=args.max_degree)
        if hubs:
            print(f"  Found {len(hubs)} hub nodes:")
            for hub_id in hubs[:10]:
                node = graph.nodes.get(hub_id)
                degree = graph.get_node_degrees().get(hub_id, {})
                print(f"    {hub_id} ({node.node_type.value if node else '?'}) "
                      f"degree={degree.get('total', 0)}")
        else:
            print("  No hubs found above threshold")

    # ---------------------------------------------------------------
    # 4. Print statistics
    # ---------------------------------------------------------------
    stats = graph.get_statistics()
    print(f"\n{'=' * 60}")
    print("GRAPH STATISTICS")
    print(f"{'=' * 60}")
    print(f"Total nodes:            {stats['total_nodes']}")
    print(f"Total edges:            {stats['total_edges']}")
    print(f"Documents:              {stats['num_documents']}")
    print(f"Docs with elements:     {stats['docs_with_elements']}")
    print(f"Avg elements/doc:       {stats['avg_elements_per_doc']:.1f}")
    print()

    print("Node type distribution:")
    for ntype, count in sorted(stats["node_type_distribution"].items(), key=lambda x: -x[1]):
        print(f"  {ntype:20s}  {count:5d}")
    print()

    print("Edge type distribution:")
    for etype, count in sorted(stats["edge_type_distribution"].items(), key=lambda x: -x[1]):
        print(f"  {etype:20s}  {count:5d}")
    print()

    print("Attribution distribution:")
    for attr, count in sorted(stats["attribution_distribution"].items(), key=lambda x: -x[1]):
        print(f"  {attr:25s}  {count:5d}")
    print()

    if args.stats_only:
        print("\nRanking traffic hubs (--stats-only still exports hub ranking)...")
        hub_report = graph.rank_traffic_hubs(
            top_k_docs=args.hub_top_docs,
            top_k_elements_per_doc=args.hub_top_elements,
            min_edge_confidence=args.hub_min_edge_confidence,
        )
        Path(args.hubs_output).parent.mkdir(parents=True, exist_ok=True)
        with open(args.hubs_output, "w", encoding="utf-8") as f:
            json.dump(hub_report, f, indent=2, ensure_ascii=False)
        print(f"  Hubs written to {args.hubs_output}")
        print("--stats-only mode, skipping path finding and other exports.")
        return

    # ---------------------------------------------------------------
    # 5. Find cross-document paths
    # ---------------------------------------------------------------
    print(f"\nFinding cross-doc element paths (max_hops={args.max_hops}, "
          f"min_score={args.min_score})...")

    paths = graph.find_cross_doc_element_paths(
        max_hops=args.max_hops,
        min_score=args.min_score,
        require_cross_doc=True,
        require_cross_modal=True,
        max_paths=args.max_paths,
        max_start_nodes=(args.max_start_nodes if args.max_start_nodes > 0 else None),
        neighbor_limit=(args.neighbor_limit if args.neighbor_limit > 0 else None),
    )

    print(f"  Found {len(paths)} valid paths")

    if paths:
        # Path statistics
        scores = [p.path_score for p in paths]
        hop_counts = [p.num_hops for p in paths]
        print(f"  Score range: {min(scores):.3f} – {max(scores):.3f}")
        print(f"  Hop range:   {min(hop_counts)} – {max(hop_counts)}")

        # Modality distribution in paths
        modality_pairs: Dict[str, int] = {}
        for p in paths:
            key = "↔".join(sorted(p.modalities_involved))
            modality_pairs[key] = modality_pairs.get(key, 0) + 1
        print(f"\n  Modality pairs in paths:")
        for pair, count in sorted(modality_pairs.items(), key=lambda x: -x[1])[:10]:
            print(f"    {pair:30s}  {count:4d}")

        # Show top 5 paths
        print(f"\n  Top 5 paths by score:")
        for i, p in enumerate(paths[:5]):
            print(f"    [{i+1}] score={p.path_score:.3f} hops={p.num_hops} "
                  f"docs={sorted(p.docs_involved)} "
                  f"modalities={sorted(p.modalities_involved)}")
            print(f"        {' → '.join(p.node_ids)}")

    # ---------------------------------------------------------------
    # 6. Export
    # ---------------------------------------------------------------
    if not args.paths_only:
        print(f"\nExporting full graph to {args.output}...")
        graph_dict = graph.to_dict()
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(graph_dict, f, indent=2, ensure_ascii=False)
        print(f"  Written ({Path(args.output).stat().st_size / 1024:.0f} KB)")

    if paths:
        print(f"\nExporting {len(paths)} paths to {args.paths_output}...")
        path_candidates = graph.export_paths_for_query_generation(paths)
        Path(args.paths_output).parent.mkdir(parents=True, exist_ok=True)
        with open(args.paths_output, "w", encoding="utf-8") as f:
            json.dump(path_candidates, f, indent=2, ensure_ascii=False)
        print(f"  Written ({Path(args.paths_output).stat().st_size / 1024:.0f} KB)")

    print("\nRanking traffic hubs for intra-doc and cross-doc retrieval...")
    hub_report = graph.rank_traffic_hubs(
        top_k_docs=args.hub_top_docs,
        top_k_elements_per_doc=args.hub_top_elements,
        min_edge_confidence=args.hub_min_edge_confidence,
    )
    Path(args.hubs_output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.hubs_output, "w", encoding="utf-8") as f:
        json.dump(hub_report, f, indent=2, ensure_ascii=False)
    print(f"  Hubs written ({Path(args.hubs_output).stat().st_size / 1024:.0f} KB) -> {args.hubs_output}")

    print(f"\nDone!")


if __name__ == "__main__":
    main()
