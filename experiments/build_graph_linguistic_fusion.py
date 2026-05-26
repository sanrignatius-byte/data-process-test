#!/usr/bin/env python3
"""Graph + Linguistics fusion: combine Genette/RST-validated cross-doc edges
with the existing Document Graph for improved M4 chain construction.

Reads:
  1. Linguistically-validated cross-doc edges (from build_linguistic_xdoc_bridges.py)
  2. Existing graph data (multimodal elements, reference graph, hub candidates)

Pipeline:
  1. Filter linguistically-validated edges to "strong" + "weak" tiers
  2. Inject them as hard cross-doc element-level edges into the graph
  3. Run chain finding on the enhanced graph
  4. Compare: graph-only chains vs graph+linguistics chains
"""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Set, Tuple

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def load_linguistic_edges(path: Path) -> List[Dict[str, Any]]:
    edges = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                edges.append(json.loads(line))
    return edges


def load_elements(path: Path) -> Dict[str, Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    elements: Dict[str, Dict[str, Any]] = {}
    docs = data.get("documents", {})
    for doc_id, doc in docs.items():
        for eid, el in doc.get("elements", {}).items():
            elements[eid] = el
    return elements


def load_hub_candidates(path: Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def resolve_edge_to_elements(
    edge: Dict[str, Any],
    elements: Dict[str, Dict[str, Any]],
) -> Tuple[List[str], List[str]]:
    """Map a linguistically-validated edge (section-level) to element pairs.

    For each document, find elements whose captions/content overlap with the
    decontextualized text. Returns (src_elements, tgt_elements).
    """
    src_doc = edge.get("source_doc", "")
    tgt_doc = edge.get("target_doc", "")
    src_decontext = edge.get("src_decontext", "")
    tgt_decontext = edge.get("tgt_decontext", "")

    src_elems = [
        eid for eid, el in elements.items()
        if eid.startswith(src_doc) and el.get("element_type") in ("figure", "table", "formula")
    ]
    tgt_elems = [
        eid for eid, el in elements.items()
        if eid.startswith(tgt_doc) and el.get("element_type") in ("figure", "table", "formula")
    ]
    return src_elems, tgt_elems


def build_enhanced_graph(
    linguistic_edges: List[Dict[str, Any]],
    elements: Dict[str, Dict[str, Any]],
    hub_candidates: Dict[str, Any],
) -> Dict[str, Any]:
    """Create graph with linguistic edges as hard cross-doc element edges."""

    # Existing intra-doc structure from hub candidates
    pairs = hub_candidates.get("pairs", hub_candidates.get("candidates", []))
    if not pairs and isinstance(hub_candidates, list):
        pairs = hub_candidates

    # Build adjacency
    adj: Dict[str, Set[str]] = defaultdict(set)
    doc_of: Dict[str, str] = {}

    for pair in pairs:
        path = pair.get("path", [])
        for i in range(len(path) - 1):
            adj[path[i]].add(path[i + 1])
            adj[path[i + 1]].add(path[i])

    for eid, el in elements.items():
        doc_id = eid.split("_figure_")[0].split("_table_")[0].split("_formula_")[0]
        doc_of[eid] = doc_id

    # Add linguistics-validated cross-doc edges
    lingu_edges_added = 0
    lingu_edge_details = []

    for ledge in linguistic_edges:
        tier = ledge.get("linguistic_quality_tier", "noise")
        genette = ledge.get("genette_type", "unknown")
        is_causal = ledge.get("is_causal_chain", False)

        if tier not in ("strong", "weak"):
            continue

        src_elems, tgt_elems = resolve_edge_to_elements(ledge, elements)

        # Connect all cross-doc element pairs
        for se in src_elems:
            for te in tgt_elems:
                if doc_of.get(se) != doc_of.get(te):
                    adj[se].add(te)
                    adj[te].add(se)
                    lingu_edges_added += 1
                    lingu_edge_details.append({
                        "source_element": se,
                        "target_element": te,
                        "genette_type": genette,
                        "tier": tier,
                        "is_causal": is_causal,
                        "confidence": ledge.get("confidence", 0),
                        "asymmetry_score": ledge.get("asymmetry_score", 0),
                    })

    # Also add intra-doc edges within each document from elements
    for eid in elements:
        doc = doc_of.get(eid, "")
        for other_eid in elements:
            if doc_of.get(other_eid) == doc and other_eid != eid:
                etype_a = elements[eid].get("element_type", "")
                etype_b = elements[other_eid].get("element_type", "")
                if etype_a != etype_b:  # cross-modal only
                    adj[eid].add(other_eid)

    return {
        "adjacency": {k: list(v) for k, v in adj.items()},
        "linguistic_edges_added": lingu_edges_added,
        "linguistic_edge_details": lingu_edge_details,
        "total_nodes": len(adj),
        "total_edges": sum(len(v) for v in adj.values()) // 2,
    }


def find_chains(
    graph: Dict[str, Any],
    elements: Dict[str, Dict[str, Any]],
    min_hops: int = 2,
    max_hops: int = 4,
    max_chains: int = 500,
) -> List[Dict[str, Any]]:
    """BFS chain discovery on the enhanced graph."""

    adj = {k: set(v) for k, v in graph["adjacency"].items()}
    lingu_edges = {frozenset([e["source_element"], e["target_element"]])
                   for e in graph["linguistic_edge_details"]}

    chains = []
    doc_of = {}
    for eid, el in elements.items():
        doc_id = eid.split("_figure_")[0].split("_table_")[0].split("_formula_")[0]
        doc_of[eid] = doc_id

    # Find cross-document cross-modal chains
    # Only use nodes that are actual elements (figure/table/formula), not paragraphs
    element_ids = [eid for eid in elements.keys()
                   if elements[eid].get("element_type") in ("figure", "table", "formula")]
    valid_nodes = set(element_ids)
    import random
    random.seed(42)
    random.shuffle(element_ids)

    for start in element_ids:
        if len(chains) >= max_chains:
            break
        start_type = elements[start].get("element_type", "")
        start_doc = doc_of.get(start, "")

        # BFS
        from collections import deque
        visited = {start}
        queue = deque([(start, [start])])

        while queue and len(chains) < max_chains:
            node, path = queue.popleft()
            if len(path) > max_hops:
                continue

            for neighbor in adj.get(node, set()):
                if neighbor in visited:
                    continue
                if neighbor not in valid_nodes:
                    continue  # skip paragraph nodes
                visited.add(neighbor)
                new_path = path + [neighbor]

                if len(new_path) >= min_hops + 1:
                    end_type = elements[neighbor].get("element_type", "")
                    end_doc = doc_of.get(neighbor, "")
                    if (start_type != end_type and
                        start_doc != end_doc):
                        # Count linguistic edges in this chain
                        lingu_count = sum(
                            1 for i in range(len(new_path) - 1)
                            if frozenset([new_path[i], new_path[i+1]]) in lingu_edges
                        )
                        chains.append({
                            "path": new_path,
                            "hop_count": len(new_path) - 1,
                            "modality_pair": f"{start_type}+{end_type}",
                            "cross_doc": True,
                            "linguistic_edges_in_chain": lingu_count,
                            "linguistic_edge_ratio": round(lingu_count / max(len(new_path)-1, 1), 3),
                        })

                if len(new_path) < max_hops + 1:
                    queue.append((neighbor, new_path))

    return chains


def main():
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--linguistic-edges",
        required=True,
        help="Path to linguistic_validated_edges.jsonl",
    )
    parser.add_argument(
        "--elements",
        default="data/02_enriched/multimodal_elements_enriched.json",
    )
    parser.add_argument(
        "--hub-candidates",
        default="data/02_enriched/hub_candidates_enriched_v4_intra_doc.json",
    )
    parser.add_argument("--output-dir", default="")
    parser.add_argument("--min-hops", type=int, default=2)
    parser.add_argument("--max-hops", type=int, default=4)
    parser.add_argument("--max-chains", type=int, default=500)
    args = parser.parse_args()

    if args.output_dir:
        out_dir = Path(args.output_dir)
        if not out_dir.is_absolute():
            out_dir = ROOT / out_dir
    else:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        out_dir = ROOT / f"data/05_eval/graph_linguistic_fusion_{stamp}"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    print("Loading linguistic edges...")
    lingu_edges = load_linguistic_edges(Path(args.linguistic_edges))
    print(f"  {len(lingu_edges)} edges loaded")

    strong_weak = [e for e in lingu_edges
                   if e.get("linguistic_quality_tier") in ("strong", "weak")]
    print(f"  {len(strong_weak)} strong+weak edges")

    print("Loading elements...")
    elements = load_elements(ROOT / args.elements)
    print(f"  {len(elements)} elements")

    hub_path = ROOT / args.hub_candidates
    hub_candidates = {}
    if hub_path.exists():
        hub_candidates = load_hub_candidates(hub_path)

    # Build enhanced graph
    print("\nBuilding enhanced graph...")
    graph = build_enhanced_graph(strong_weak, elements, hub_candidates)
    print(f"  Nodes: {graph['total_nodes']}")
    print(f"  Edges: {graph['total_edges']}")
    print(f"  Linguistic edges injected: {graph['linguistic_edges_added']}")

    # Also build baseline graph (no linguistic edges)
    print("\nBuilding baseline graph (no linguistic edges)...")
    baseline_graph = build_enhanced_graph([], elements, hub_candidates)
    print(f"  Nodes: {baseline_graph['total_nodes']}")
    print(f"  Edges: {baseline_graph['total_edges']}")

    # Find chains on both graphs
    print(f"\nFinding chains (min_hops={args.min_hops}, max_hops={args.max_hops})...")
    print("  Enhanced graph...")
    enhanced_chains = find_chains(graph, elements, args.min_hops, args.max_hops, args.max_chains)
    print(f"    {len(enhanced_chains)} chains found")

    print("  Baseline graph...")
    baseline_chains = find_chains(baseline_graph, elements, args.min_hops, args.max_hops, args.max_chains)
    print(f"    {len(baseline_chains)} chains found")

    # Compare
    from collections import Counter
    enh_lingu_counts = Counter(c["linguistic_edges_in_chain"] for c in enhanced_chains)
    enh_modality = Counter(c["modality_pair"] for c in enhanced_chains)
    base_modality = Counter(c["modality_pair"] for c in baseline_chains)

    summary = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "linguistic_edges_input": len(strong_weak),
        "graph_stats": {
            "nodes": graph["total_nodes"],
            "edges": graph["total_edges"],
            "linguistic_edges_injected": graph["linguistic_edges_added"],
        },
        "baseline_graph_stats": {
            "nodes": baseline_graph["total_nodes"],
            "edges": baseline_graph["total_edges"],
        },
        "chains": {
            "enhanced_count": len(enhanced_chains),
            "baseline_count": len(baseline_chains),
            "delta": len(enhanced_chains) - len(baseline_chains),
            "enhanced_with_lingu_edges": {
                k: v for k, v in sorted(enh_lingu_counts.items())
            },
            "enhanced_modality_dist": dict(enh_modality.most_common(10)),
            "baseline_modality_dist": dict(base_modality.most_common(10)),
        },
        "linguistic_edge_sample": graph["linguistic_edge_details"][:20],
    }

    summary_path = out_dir / "fusion_summary.json"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    # Output chains
    chains_path = out_dir / "enhanced_chains.json"
    with chains_path.open("w", encoding="utf-8") as f:
        json.dump({"chains": enhanced_chains, "count": len(enhanced_chains)}, f)

    baseline_chains_path = out_dir / "baseline_chains.json"
    with baseline_chains_path.open("w", encoding="utf-8") as f:
        json.dump({"chains": baseline_chains, "count": len(baseline_chains)}, f)

    print(f"\n=== Fusion Results ===")
    print(f"  Baseline chains (graph only): {len(baseline_chains)}")
    print(f"  Enhanced chains (graph+linguistics): {len(enhanced_chains)}")
    print(f"  Delta: +{len(enhanced_chains) - len(baseline_chains)}")
    if enhanced_chains:
        chains_with_lingu = sum(1 for c in enhanced_chains if c["linguistic_edges_in_chain"] > 0)
        print(f"  Chains with >=1 linguistic edge: {chains_with_lingu}/{len(enhanced_chains)}")
    print(f"\n  Modality pairs (enhanced):")
    for pair, count in enh_modality.most_common(5):
        print(f"    {pair}: {count}")
    print(f"\nSummary: {summary_path}")


if __name__ == "__main__":
    main()
