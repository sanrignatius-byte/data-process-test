#!/usr/bin/env python3
"""
Build LaTeX Reference Graph (Step 0 v3)

Parses extracted LaTeX sources (.tex/.bbl) to build per-document reference
DAGs based on \\label{} / \\ref{} / \\cite{} commands.

This produces a *structural* reference graph that complements the
MinerU-based multimodal DAG (data/multimodal_elements.json).

Usage:
    # Process all extracted sources (auto-detect from latex_sources/extracted/)
    python scripts/build_latex_reference_graph.py

    # With custom paths
    python scripts/build_latex_reference_graph.py \
        --source-dir data/latex_sources/extracted \
        --output data/latex_reference_graph.json \
        --report data/latex_reference_report.json

    # Process specific papers
    python scripts/build_latex_reference_graph.py --doc-ids 1104.3913 1306.5204

    # Merge with existing multimodal_elements.json
    python scripts/build_latex_reference_graph.py \
        --merge-with data/multimodal_elements.json \
        --merged-output data/multimodal_elements_v2.json
"""

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.parsers.latex_reference_extractor import (
    LaTeXReferenceExtractor,
    LatexDocumentGraph,
    LabelType,
)


# ---------------------------------------------------------------------------
# Merge logic: LaTeX graph + MinerU multimodal_elements.json
# ---------------------------------------------------------------------------

def merge_with_multimodal(
    latex_graphs: dict[str, LatexDocumentGraph],
    multimodal_path: Path,
) -> dict:
    """
    Merge LaTeX reference edges into the existing multimodal_elements.json.

    Strategy:
      - For each document in both sources, add LaTeX edges as new edges
        with source="latex" provenance.
      - Map LaTeX labels to existing element_ids where possible
        (e.g. label "fig:1" → element "1306.5204_figure_1").
      - Add new multi-hop paths discovered through LaTeX references.
    """
    with open(multimodal_path, "r", encoding="utf-8") as f:
        mm_data = json.load(f)

    docs = mm_data.get("documents", {})
    merge_stats = {"docs_matched": 0, "edges_added": 0, "labels_matched": 0}

    for doc_id, lg in latex_graphs.items():
        if doc_id not in docs:
            continue

        merge_stats["docs_matched"] += 1
        doc = docs[doc_id]
        existing_elements = doc.get("elements", {})

        # Build mapping: latex label → multimodal element_id
        label_to_elem = _match_labels_to_elements(lg, existing_elements, doc_id)
        merge_stats["labels_matched"] += len(label_to_elem)

        # Add latex_labels field to matched elements
        for label_key, elem_id in label_to_elem.items():
            if elem_id in existing_elements:
                md = existing_elements[elem_id].get("metadata", {})
                md.setdefault("latex_labels", [])
                if label_key not in md["latex_labels"]:
                    md["latex_labels"].append(label_key)
                existing_elements[elem_id]["metadata"] = md

        # Add LaTeX edges that map to known elements
        existing_edges = doc.get("edges", [])
        existing_edge_set = {
            (e["source_id"], e["target_id"]) for e in existing_edges
        }

        for edge in lg.edges:
            src_elem = label_to_elem.get(edge.source_label)
            tgt_elem = label_to_elem.get(edge.target_label)
            if src_elem and tgt_elem and (src_elem, tgt_elem) not in existing_edge_set:
                existing_edges.append({
                    "source_id": src_elem,
                    "target_id": tgt_elem,
                    "source_type": edge.source_type,
                    "target_type": edge.target_type,
                    "ref_text": edge.ref_text,
                    "context_snippet": edge.context[:300],
                    "provenance": "latex",
                })
                existing_edge_set.add((src_elem, tgt_elem))
                merge_stats["edges_added"] += 1

        doc["edges"] = existing_edges
        doc["num_edges"] = len(existing_edges)

        # Store LaTeX metadata
        doc.setdefault("latex_metadata", {})
        doc["latex_metadata"] = {
            "num_labels": len(lg.labels),
            "num_refs": len(lg.refs),
            "num_bib_entries": len(lg.bib),
            "label_types": _count_label_types(lg),
        }

    mm_data["merge_stats"] = merge_stats
    return mm_data


def _match_labels_to_elements(
    lg: LatexDocumentGraph,
    elements: dict,
    doc_id: str,
) -> dict[str, str]:
    """
    Match LaTeX labels to multimodal element IDs.
    e.g. "fig:model" (label type=figure) → "doc_id_figure_1"
    """
    mapping: dict[str, str] = {}

    # Group existing elements by type
    elements_by_type: dict[str, list] = defaultdict(list)
    for elem_id, elem in elements.items():
        etype = elem.get("element_type", "")
        num = elem.get("number")
        elements_by_type[etype].append((elem_id, num, elem))

    # Map label types to element types
    type_map = {
        "figure": "figure",
        "table": "table",
        "equation": "formula",
        "section": "section",
    }

    for label_key, label_info in lg.labels.items():
        ltype = label_info.label_type.value if isinstance(label_info.label_type, LabelType) else label_info.label_type
        elem_type = type_map.get(ltype)
        if not elem_type:
            continue

        candidates = elements_by_type.get(elem_type, [])
        if not candidates:
            continue

        # Try to match by number extracted from label key
        num = _extract_number_from_label(label_key)
        if num is not None:
            for elem_id, elem_num, _ in candidates:
                if elem_num == num:
                    mapping[label_key] = elem_id
                    break

        # If no number match, try caption similarity
        if label_key not in mapping and label_info.caption:
            best_id = None
            best_overlap = 0
            cap_tokens = set(label_info.caption.lower().split())
            for elem_id, _, elem in candidates:
                elem_cap = (elem.get("caption", "") or "").lower()
                elem_tokens = set(elem_cap.split())
                if not elem_tokens:
                    continue
                overlap = len(cap_tokens & elem_tokens) / max(1, len(cap_tokens | elem_tokens))
                if overlap > best_overlap and overlap > 0.3:
                    best_overlap = overlap
                    best_id = elem_id
            if best_id:
                mapping[label_key] = best_id

    return mapping


def _extract_number_from_label(key: str) -> int | None:
    """Extract trailing number from label key: 'fig:3' → 3, 'table1' → 1."""
    import re
    m = re.search(r'(\d+)\s*$', key)
    return int(m.group(1)) if m else None


def _count_label_types(lg: LatexDocumentGraph) -> dict[str, int]:
    counts: dict[str, int] = defaultdict(int)
    for info in lg.labels.values():
        t = info.label_type.value if isinstance(info.label_type, LabelType) else str(info.label_type)
        counts[t] += 1
    return dict(counts)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Build LaTeX reference graph from extracted sources"
    )
    parser.add_argument(
        "--source-dir",
        default="data/latex_sources/extracted",
        help="Directory with extracted LaTeX sources (one subdir per paper)",
    )
    parser.add_argument(
        "--output",
        default="data/latex_reference_graph.json",
        help="Output path for LaTeX reference graph",
    )
    parser.add_argument(
        "--report",
        default="data/latex_reference_report.json",
        help="Output path for statistics report",
    )
    parser.add_argument(
        "--doc-ids",
        nargs="*",
        default=None,
        help="Process only specific document IDs",
    )
    parser.add_argument(
        "--max-hops",
        type=int,
        default=3,
        help="Maximum hop distance for multi-hop path discovery",
    )
    parser.add_argument(
        "--merge-with",
        default=None,
        help="Path to existing multimodal_elements.json to merge with",
    )
    parser.add_argument(
        "--merged-output",
        default="data/multimodal_elements_v2.json",
        help="Output path for merged graph (only used with --merge-with)",
    )
    args = parser.parse_args()

    source_dir = Path(args.source_dir)
    output_path = Path(args.output)
    report_path = Path(args.report)

    if not source_dir.exists():
        print(f"ERROR: Source directory not found: {source_dir}")
        print("  Run download_latex_sources.py first.")
        sys.exit(1)

    # Discover paper directories
    if args.doc_ids:
        paper_dirs = []
        for did in args.doc_ids:
            safe = did.replace("/", "_")
            d = source_dir / safe
            if d.exists():
                paper_dirs.append((did, d))
            else:
                print(f"  [WARN] Not found: {d}")
    else:
        paper_dirs = []
        for d in sorted(source_dir.iterdir()):
            if d.is_dir() and not d.name.startswith("."):
                # Convert dirname back to arxiv ID
                doc_id = d.name.replace("_", ".", 1) if "_" in d.name else d.name
                paper_dirs.append((doc_id, d))

    print(f"Processing {len(paper_dirs)} papers from {source_dir}")
    print(f"Max hops: {args.max_hops}")
    print()

    extractor = LaTeXReferenceExtractor()
    graphs: dict[str, LatexDocumentGraph] = {}

    # --- Per-document stats ---
    stats = {
        "total_papers": len(paper_dirs),
        "has_tex": 0,
        "has_bbl": 0,
        "has_labels": 0,
        "has_edges": 0,
        "total_labels": 0,
        "total_refs": 0,
        "total_edges": 0,
        "total_ref_edges": 0,
        "total_containment_edges": 0,
        "total_bib_entries": 0,
        "total_multihop_paths": 0,
        "label_type_dist": defaultdict(int),
        "edge_type_dist": defaultdict(int),
        "containment_edge_dist": defaultdict(int),
        "errors": [],
    }

    for doc_id, paper_dir in paper_dirs:
        graph = extractor.extract(doc_id=doc_id, extract_dir=paper_dir)

        if "error" in graph.metadata:
            err = graph.metadata["error"]
            stats["errors"].append({"doc_id": doc_id, "error": err})
            print(f"  [{doc_id}] {err}")
            continue

        stats["has_tex"] += 1

        if graph.bib:
            stats["has_bbl"] += 1
        if graph.labels:
            stats["has_labels"] += 1
        if graph.edges:
            stats["has_edges"] += 1

        stats["total_labels"] += len(graph.labels)
        stats["total_refs"] += len(graph.refs)
        stats["total_edges"] += len(graph.edges)
        stats["total_bib_entries"] += len(graph.bib)

        for info in graph.labels.values():
            t = info.label_type.value
            stats["label_type_dist"][t] += 1

        for edge in graph.edges:
            key = f"{edge.source_type}→{edge.target_type}"
            if edge.ref_text == "[containment]":
                stats["containment_edge_dist"][key] += 1
                stats["total_containment_edges"] += 1
            else:
                stats["edge_type_dist"][key] += 1
                stats["total_ref_edges"] += 1

        # Multi-hop paths
        paths = LaTeXReferenceExtractor.find_multihop_paths(graph, max_hops=args.max_hops)
        graph.metadata["multihop_paths"] = len(paths)
        graph.metadata["multihop_examples"] = [
            {
                "path": p,
                "types": [graph.labels[k].label_type.value for k in p if k in graph.labels],
            }
            for p in paths[:10]  # store first 10 as examples
        ]
        stats["total_multihop_paths"] += len(paths)

        graphs[doc_id] = graph

        label_summary = ", ".join(
            f"{t}:{c}" for t, c in sorted(
                _count_label_types(graph).items(), key=lambda x: -x[1]
            )
        )
        print(
            f"  [{doc_id}] labels={len(graph.labels)} "
            f"refs={len(graph.refs)} edges={len(graph.edges)} "
            f"bib={len(graph.bib)} paths={len(paths)}  ({label_summary})"
        )

    # --- Summary ---
    print(f"\n{'='*60}")
    print(f"LATEX REFERENCE GRAPH COMPLETE")
    print(f"{'='*60}")
    print(f"Papers processed:     {stats['total_papers']}")
    print(f"Papers with .tex:     {stats['has_tex']}")
    print(f"Papers with .bbl:     {stats['has_bbl']}")
    print(f"Papers with labels:   {stats['has_labels']}")
    print(f"Papers with edges:    {stats['has_edges']}")
    print(f"Total labels:         {stats['total_labels']}")
    print(f"Total refs:           {stats['total_refs']}")
    print(f"Total edges:          {stats['total_edges']}")
    print(f"  Reference edges:    {stats['total_ref_edges']}")
    print(f"  Containment edges:  {stats['total_containment_edges']}")
    print(f"Total bib entries:    {stats['total_bib_entries']}")
    print(f"Total multi-hop paths:{stats['total_multihop_paths']}")
    ref_rate = 100 * stats['total_ref_edges'] / max(1, stats['total_refs'])
    print(f"Ref→Edge conversion:  {ref_rate:.1f}%")

    if stats["label_type_dist"]:
        print(f"\nLabel Type Distribution:")
        n_unknown = stats["label_type_dist"].get("unknown", 0)
        for ltype, count in sorted(stats["label_type_dist"].items(), key=lambda x: -x[1]):
            pct = 100 * count / max(1, stats["total_labels"])
            print(f"  {ltype:12s}  {count:5d}  ({pct:.1f}%)")
        if stats["total_labels"] > 0:
            print(f"  → Unknown rate: {100*n_unknown/stats['total_labels']:.1f}%")

    if stats["edge_type_dist"]:
        print(f"\nReference Edge Distribution:")
        for etype, count in sorted(stats["edge_type_dist"].items(), key=lambda x: -x[1]):
            print(f"  {etype:30s}  {count:5d}")

    if stats["containment_edge_dist"]:
        print(f"\nContainment Edge Distribution:")
        for etype, count in sorted(stats["containment_edge_dist"].items(), key=lambda x: -x[1]):
            print(f"  {etype:30s}  {count:5d}")

    # Show multi-hop examples
    print(f"\n--- Multi-Hop Path Examples ---")
    example_count = 0
    for doc_id, graph in graphs.items():
        for ex in graph.metadata.get("multihop_examples", []):
            if example_count >= 8:
                break
            path_str = " → ".join(
                f"{t}:{k}" for k, t in zip(ex["path"], ex["types"])
            )
            print(f"  [{doc_id}] {path_str}")
            example_count += 1
        if example_count >= 8:
            break

    # --- Save output ---
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_data = {
        "metadata": {
            "source_dir": str(source_dir),
            "max_hops": args.max_hops,
            "papers_processed": stats["total_papers"],
            "papers_with_tex": stats["has_tex"],
        },
        "documents": {
            doc_id: g.to_dict() for doc_id, g in graphs.items()
        },
    }
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)
    print(f"\nOutput saved to: {output_path}")

    # --- Save report ---
    stats["label_type_dist"] = dict(stats["label_type_dist"])
    stats["edge_type_dist"] = dict(stats["edge_type_dist"])
    stats["containment_edge_dist"] = dict(stats["containment_edge_dist"])
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)
    print(f"Report saved to: {report_path}")

    # --- Optional merge ---
    if args.merge_with:
        merge_path = Path(args.merge_with)
        if not merge_path.exists():
            print(f"\n[WARN] Merge target not found: {merge_path}")
        else:
            print(f"\nMerging with {merge_path} ...")
            merged = merge_with_multimodal(graphs, merge_path)
            merged_out = Path(args.merged_output)
            merged_out.parent.mkdir(parents=True, exist_ok=True)
            with open(merged_out, "w", encoding="utf-8") as f:
                json.dump(merged, f, indent=2, ensure_ascii=False)
            ms = merged.get("merge_stats", {})
            print(f"  Docs matched:   {ms.get('docs_matched', 0)}")
            print(f"  Labels matched: {ms.get('labels_matched', 0)}")
            print(f"  Edges added:    {ms.get('edges_added', 0)}")
            print(f"  Saved to: {merged_out}")


if __name__ == "__main__":
    main()
