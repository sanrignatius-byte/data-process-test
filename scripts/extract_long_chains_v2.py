#!/usr/bin/env python3
"""Extract long-chain candidates from latex_reference_graph_v2.json.

Method C core: find paths of ≥3 hops in the LaTeX \ref{} graph where
endpoints are different modalities (figure/table/equation).  Intermediate
nodes are the reasoning bridge.

Input:
  data/01_graphs/latex_reference_graph_v2.json   (1425 docs, 46K labels, 68K edges)
  data/01_graphs/multimodal_elements_v2.json     (1145 docs, 30K elements)

Output:
  data/01_graphs/long_chain_candidates_v2.json   (method-C ready candidates)
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, FrozenSet, List, Optional, Set, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.parsers.latex_reference_extractor import (
    LaTeXReferenceExtractor,
    LatexDocumentGraph,
    LabelInfo,
    LabelType,
)

# Modalities we care about for method-C endpoints
CROSS_MODAL_TYPES = {LabelType.FIGURE, LabelType.TABLE, LabelType.EQUATION}


def canonical_path(path: List[str]) -> Tuple[str, ...]:
    p = tuple(path)
    r = tuple(reversed(p))
    return p if p <= r else r


def find_long_chains(
    graph: LatexDocumentGraph,
    min_hops: int = 3,
    max_hops: int = 5,
    max_chains_per_doc: int = 200,
) -> List[List[str]]:
    """Find paths of min_hops..max_hops hops where endpoints are different
    modalities and at least one edge is a real \\ref{} (not containment)."""

    adj: Dict[str, Set[str]] = {}
    ref_edge_pairs: Set[Tuple[str, str]] = set()

    for edge in graph.edges:
        adj.setdefault(edge.source_label, set()).add(edge.target_label)
        adj.setdefault(edge.target_label, set()).add(edge.source_label)
        if edge.ref_text != "[containment]":
            ref_edge_pairs.add((edge.source_label, edge.target_label))
            ref_edge_pairs.add((edge.target_label, edge.source_label))

    paths: List[List[str]] = []
    seen: Set[Tuple[str, ...]] = set()

    def _dfs(current: str, path: List[str], has_ref: bool) -> None:
        if len(paths) >= max_chains_per_doc:
            return
        if len(path) > max_hops + 1:
            return

        hops = len(path) - 1
        if hops >= min_hops:
            start_label = graph.labels.get(path[0])
            end_label = graph.labels.get(path[-1])
            if (start_label and end_label
                    and start_label.label_type in CROSS_MODAL_TYPES
                    and end_label.label_type in CROSS_MODAL_TYPES
                    and start_label.label_type != end_label.label_type
                    and has_ref):
                cp = canonical_path(path)
                if cp not in seen:
                    seen.add(cp)
                    paths.append(list(path))

        if hops == max_hops:
            return

        for neighbor in adj.get(current, set()):
            if neighbor not in path and neighbor in graph.labels:
                if len(paths) >= max_chains_per_doc:
                    return
                edge_is_ref = (current, neighbor) in ref_edge_pairs
                path.append(neighbor)
                _dfs(neighbor, path, has_ref or edge_is_ref)
                path.pop()

    # Start only from cross-modal element labels (not section/subsection)
    start_labels = [
        k for k, info in graph.labels.items()
        if info.label_type in CROSS_MODAL_TYPES
    ]

    for start in sorted(start_labels):
        if len(paths) >= max_chains_per_doc:
            break
        _dfs(start, [start], False)

    return paths


def build_chain_record(
    doc_id: str,
    path: List[str],
    graph: LatexDocumentGraph,
    chain_idx: int,
) -> Dict[str, Any]:
    """Convert a path into a method-C candidate record."""
    start_info = graph.labels[path[0]]
    end_info = graph.labels[path[-1]]

    # Collect intermediate node info
    intermediates = []
    for mid_key in path[1:-1]:
        mid_info = graph.labels.get(mid_key)
        if mid_info:
            intermediates.append({
                "label_key": mid_key,
                "label_type": mid_info.label_type.value,
                "environment": mid_info.environment,
                "line_no": mid_info.line_no,
            })

    # Collect edge contexts along the path
    edge_contexts = []
    for i in range(len(path) - 1):
        src, tgt = path[i], path[i + 1]
        # Find matching edges
        for edge in graph.edges:
            if ((edge.source_label == src and edge.target_label == tgt) or
                    (edge.source_label == tgt and edge.target_label == src)):
                edge_contexts.append({
                    "segment_index": i,
                    "source_label": src,
                    "target_label": tgt,
                    "ref_text": edge.ref_text,
                    "source_type": edge.source_type,
                    "target_type": edge.target_type,
                    "is_containment": edge.ref_text == "[containment]",
                })
                break  # one edge per segment is enough

    # Build path type sequence
    path_types = []
    for key in path:
        info = graph.labels.get(key)
        if info:
            path_types.append(info.label_type.value)
        else:
            path_types.append("unknown")

    # Line number span
    line_nos = []
    for key in path:
        info = graph.labels.get(key)
        if info and info.line_no is not None:
            line_nos.append(info.line_no)
    line_no_span = (max(line_nos) - min(line_nos)) if len(line_nos) >= 2 else None

    a_type = start_info.label_type.value
    b_type = end_info.label_type.value
    pair_type = "+".join(sorted([a_type, b_type]))

    return {
        "candidate_id": f"{doc_id}_lc_{chain_idx:04d}",
        "doc_id": doc_id,
        "hop_distance": len(path) - 1,
        "path_label_keys": path,
        "path_types": path_types,
        "pair_type": pair_type,
        "element_a_label": path[0],
        "element_a_type": a_type,
        "element_a_env": start_info.environment,
        "element_a_line_no": start_info.line_no,
        "element_b_label": path[-1],
        "element_b_type": b_type,
        "element_b_env": end_info.environment,
        "element_b_line_no": end_info.line_no,
        "intermediate_nodes": intermediates,
        "edge_contexts": edge_contexts,
        "line_no_span": line_no_span,
        "modalities_in_path": sorted(set(
            t for t in path_types
            if t in {"figure", "table", "equation"}
        )),
    }


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Extract long-chain candidates from LaTeX reference graph (method C)"
    )
    ap.add_argument(
        "--latex-graph",
        default="data/01_graphs/latex_reference_graph_v2.json",
        help="Input LaTeX reference graph",
    )
    ap.add_argument(
        "--output",
        default="data/01_graphs/long_chain_candidates_v2.json",
        help="Output long-chain candidates",
    )
    ap.add_argument("--min-hops", type=int, default=3)
    ap.add_argument("--max-hops", type=int, default=5)
    ap.add_argument("--max-chains-per-doc", type=int, default=200)
    ap.add_argument("--max-total", type=int, default=10000,
                    help="Global cap on total candidates")
    args = ap.parse_args()

    latex_path = Path(args.latex_graph)
    if not latex_path.exists():
        print(f"ERROR: {latex_path} not found")
        sys.exit(1)

    print(f"Loading {latex_path} ...")
    latex_data = json.loads(latex_path.read_text(encoding="utf-8"))
    docs = latex_data.get("documents", {})
    print(f"  {len(docs)} documents loaded")

    all_candidates: List[Dict[str, Any]] = []
    doc_chain_counts: Counter = Counter()
    global_idx = 0

    for doc_id in sorted(docs.keys()):
        doc_data = docs[doc_id]

        # Reconstruct LatexDocumentGraph
        graph = LatexDocumentGraph(doc_id=doc_id)
        graph.metadata = doc_data.get("metadata", {})

        # Rebuild labels
        for key, linfo in doc_data.get("labels", {}).items():
            graph.labels[key] = LabelInfo(
                key=key,
                label_type=LabelType(linfo["label_type"]),
                line_no=linfo.get("line_no", 0),
                environment=linfo.get("environment"),
                caption=linfo.get("caption"),
                file_path=linfo.get("file_path"),
            )

        # Rebuild edges
        from dataclasses import dataclass

        @dataclass
        class SimpleEdge:
            source_label: str
            target_label: str
            source_type: str
            target_type: str
            ref_text: str

        for edata in doc_data.get("edges", []):
            graph.edges.append(SimpleEdge(
                source_label=edata["source_label"],
                target_label=edata["target_label"],
                source_type=edata.get("source_type", ""),
                target_type=edata.get("target_type", ""),
                ref_text=edata.get("ref_text", ""),
            ))

        # Find long chains
        chains = find_long_chains(
            graph,
            min_hops=args.min_hops,
            max_hops=args.max_hops,
            max_chains_per_doc=args.max_chains_per_doc,
        )

        if chains:
            doc_chain_counts[doc_id] = len(chains)
            for path in chains:
                global_idx += 1
                rec = build_chain_record(doc_id, path, graph, global_idx)
                all_candidates.append(rec)

        if len(all_candidates) >= args.max_total:
            print(f"  [CAP] Reached {args.max_total} candidates at doc {doc_id}")
            all_candidates = all_candidates[:args.max_total]
            break

    # Sort: prefer longer chains, then by doc coverage
    all_candidates.sort(key=lambda c: (-c["hop_distance"], c["doc_id"], c["candidate_id"]))

    # Stats
    hop_dist = Counter(c["hop_distance"] for c in all_candidates)
    pair_types = Counter(c["pair_type"] for c in all_candidates)
    modality_sets = Counter(tuple(c["modalities_in_path"]) for c in all_candidates)
    unique_docs = len(set(c["doc_id"] for c in all_candidates))

    output = {
        "metadata": {
            "source": str(latex_path),
            "min_hops": args.min_hops,
            "max_hops": args.max_hops,
            "max_chains_per_doc": args.max_chains_per_doc,
            "total_candidates": len(all_candidates),
            "unique_docs": unique_docs,
            "docs_with_chains": len(doc_chain_counts),
        },
        "summary": {
            "hop_distribution": dict(sorted(hop_dist.items())),
            "pair_type_distribution": dict(pair_types.most_common()),
            "modality_coverage": {
                str(k): v for k, v in modality_sets.most_common()
            },
            "top_docs": dict(doc_chain_counts.most_common(20)),
        },
        "candidates": all_candidates,
    }

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"\n{'='*60}")
    print(f"LONG CHAIN EXTRACTION COMPLETE")
    print(f"{'='*60}")
    print(f"Documents with chains: {len(doc_chain_counts)}/{len(docs)}")
    print(f"Total candidates:      {len(all_candidates)}")
    print(f"Unique docs:           {unique_docs}")
    print(f"\nHop distribution:")
    for h in sorted(hop_dist):
        print(f"  {h}-hop: {hop_dist[h]} ({100*hop_dist[h]/len(all_candidates):.1f}%)")
    print(f"\nPair types:")
    for pt, cnt in pair_types.most_common():
        print(f"  {pt}: {cnt}")
    print(f"\nModality coverage in path:")
    for ms, cnt in modality_sets.most_common():
        print(f"  {ms}: {cnt}")
    print(f"\nOutput: {out_path}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
