#!/usr/bin/env python3
"""
Build Multimodal Relationships (Step 0 v2)

Extracts all modality elements (figure, table, formula, section) from MinerU
output and builds per-document cross-reference DAGs + multimodal pairs.

Usage:
    # Full run
    python scripts/build_multimodal_relationships.py

    # With custom paths
    python scripts/build_multimodal_relationships.py \
        --mineru-dir data/mineru_output \
        --output data/multimodal_elements.json \
        --report data/multimodal_report.json

    # Process specific documents
    python scripts/build_multimodal_relationships.py --doc-ids 1104.3913 1306.5204

    # Reuse existing figure_text_pairs.json as fallback input
    python scripts/build_multimodal_relationships.py \
        --fallback-pairs data/figure_text_pairs.json
"""

import argparse
import json
import sys
from pathlib import Path
from collections import defaultdict

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.linkers.multimodal_relationship_builder import (
    MultimodalRelationshipBuilder,
    DocumentDAG,
    ElementType,
)


def build_from_existing_pairs(
    pairs_path: Path,
    builder: MultimodalRelationshipBuilder,
) -> dict:
    """
    Build multimodal relationships from existing figure_text_pairs.json.

    This is useful when MinerU output dir is not available locally but we
    have the previously extracted figure_text_pairs.json.
    """
    with open(pairs_path, "r", encoding="utf-8") as f:
        pairs_data = json.load(f)

    dags = {}
    all_pairs = {}

    for doc_id, doc_pairs in pairs_data.items():
        # Convert figure_text_pairs into flat records that the builder can process
        records = []
        for pair in doc_pairs:
            # Context before
            if pair.get("context_before"):
                records.append({
                    "kind": "text",
                    "text": pair["context_before"],
                    "page_idx": 0,
                    "source_type": "text",
                })

            # The figure/table itself
            source_type = pair.get("metadata", {}).get("source_type", "image")
            records.append({
                "kind": "image",
                "source_type": source_type,
                "image_path": pair.get("image_path", ""),
                "image_filename": pair.get("image_filename", ""),
                "caption": pair.get("caption", ""),
                "page_idx": 0,
            })

            # Context after
            if pair.get("context_after"):
                records.append({
                    "kind": "text",
                    "text": pair["context_after"],
                    "page_idx": 0,
                    "source_type": "text",
                })

            # Referring paragraphs as additional text
            for ref in pair.get("referring_paragraphs", []):
                records.append({
                    "kind": "text",
                    "text": ref,
                    "page_idx": 0,
                    "source_type": "text",
                })

        # Now extract elements and build DAG from these records
        dag = DocumentDAG(doc_id=doc_id)
        elements = builder._extract_elements(records, doc_id)
        for elem in elements:
            dag.elements[elem.element_id] = elem

        if dag.elements:
            builder._build_edges(dag, records)
            dags[doc_id] = dag
            doc_multimodal_pairs = builder.generate_multimodal_pairs(dag)
            if doc_multimodal_pairs:
                all_pairs[doc_id] = doc_multimodal_pairs

    return {"dags": dags, "pairs": all_pairs}


def main():
    parser = argparse.ArgumentParser(
        description="Build multimodal relationships from MinerU output"
    )
    parser.add_argument(
        "--mineru-dir",
        default="data/mineru_output",
        help="Path to MinerU output directory",
    )
    parser.add_argument(
        "--output",
        default="data/multimodal_elements.json",
        help="Output path for elements + DAG + pairs",
    )
    parser.add_argument(
        "--report",
        default="data/multimodal_report.json",
        help="Output path for statistics report",
    )
    parser.add_argument(
        "--fallback-pairs",
        default=None,
        help="Path to existing figure_text_pairs.json (used when mineru_output is empty)",
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
        help="Maximum hop distance for multi-hop paths",
    )
    parser.add_argument(
        "--context-window",
        type=int,
        default=3,
        help="Number of paragraphs for context",
    )
    args = parser.parse_args()

    mineru_dir = Path(args.mineru_dir)
    output_path = Path(args.output)
    report_path = Path(args.report)

    builder = MultimodalRelationshipBuilder(
        mineru_output_dir=str(mineru_dir),
        context_window=args.context_window,
        max_hops=args.max_hops,
    )

    # Decide source: MinerU dir or fallback pairs
    has_mineru = mineru_dir.exists() and any(
        d.is_dir() and not d.name.startswith('.')
        for d in mineru_dir.iterdir()
    ) if mineru_dir.exists() else False

    fallback_path = Path(args.fallback_pairs) if args.fallback_pairs else None
    if not fallback_path and not has_mineru:
        # Auto-detect fallback
        default_fallback = Path("data/figure_text_pairs.json")
        if default_fallback.exists():
            fallback_path = default_fallback
            print(f"MinerU output dir empty, using fallback: {fallback_path}")

    if has_mineru:
        print(f"Processing MinerU output from: {mineru_dir}")
        if args.doc_ids:
            dags = {}
            for doc_id in args.doc_ids:
                dag = builder.process_document(doc_id)
                if dag and dag.elements:
                    dags[doc_id] = dag
        else:
            dags = builder.process_all_documents()

        all_pairs = {}
        for doc_id, dag in dags.items():
            doc_pairs = builder.generate_multimodal_pairs(dag)
            if doc_pairs:
                all_pairs[doc_id] = doc_pairs
    elif fallback_path and fallback_path.exists():
        print(f"Processing from fallback pairs: {fallback_path}")
        result = build_from_existing_pairs(fallback_path, builder)
        dags = result["dags"]
        all_pairs = result["pairs"]
    else:
        print("ERROR: No MinerU output and no fallback pairs found.")
        print("  Specify --mineru-dir or --fallback-pairs")
        sys.exit(1)

    # --- Print summary ---
    print(f"\n{'='*60}")
    print(f"MULTIMODAL RELATIONSHIP BUILDING COMPLETE")
    print(f"{'='*60}")
    print(f"Documents processed:  {len(dags)}")

    total_elements = sum(len(d.elements) for d in dags.values())
    total_edges = sum(len(d.edges) for d in dags.values())
    total_pairs = sum(len(p) for p in all_pairs.values())
    print(f"Total elements:       {total_elements}")
    print(f"Total edges:          {total_edges}")
    print(f"Total multimodal pairs: {total_pairs}")

    # Element type breakdown
    type_counts = defaultdict(int)
    for dag in dags.values():
        for elem in dag.elements.values():
            type_counts[elem.element_type.value] += 1

    print(f"\nElement Type Distribution:")
    for etype, count in sorted(type_counts.items(), key=lambda x: -x[1]):
        pct = 100 * count / max(1, total_elements)
        print(f"  {etype:12s}  {count:5d}  ({pct:.1f}%)")

    # Edge type breakdown
    edge_type_counts = defaultdict(int)
    for dag in dags.values():
        for edge in dag.edges:
            key = f"{edge.source_type}→{edge.target_type}"
            edge_type_counts[key] += 1

    if edge_type_counts:
        print(f"\nEdge Type Distribution:")
        for etype, count in sorted(edge_type_counts.items(), key=lambda x: -x[1]):
            print(f"  {etype:25s}  {count:5d}")

    # Pair type breakdown
    pair_type_counts = defaultdict(int)
    hop_counts = defaultdict(int)
    for doc_pairs in all_pairs.values():
        for p in doc_pairs:
            key = "↔".join(sorted([p.element_a_type, p.element_b_type]))
            pair_type_counts[key] += 1
            hop_counts[p.hop_distance] += 1

    if pair_type_counts:
        print(f"\nMultimodal Pair Types:")
        for ptype, count in sorted(pair_type_counts.items(), key=lambda x: -x[1]):
            print(f"  {ptype:25s}  {count:5d}")

        print(f"\nHop Distance Distribution:")
        for hop, count in sorted(hop_counts.items()):
            print(f"  {hop}-hop:  {count}")

    # Multi-hop path examples
    print(f"\n--- Example Multi-Hop Paths ---")
    example_count = 0
    for doc_id, doc_pairs in all_pairs.items():
        for p in doc_pairs:
            if p.hop_distance >= 2 and example_count < 5:
                path_labels = []
                dag = dags[doc_id]
                for elem_id in p.path:
                    if elem_id in dag.elements:
                        e = dag.elements[elem_id]
                        path_labels.append(f"{e.label or e.element_type.value}")
                print(f"  [{doc_id}] {' → '.join(path_labels)}  (quality={p.quality_score:.2f})")
                example_count += 1

    # --- Save output ---
    output_data = {
        "metadata": {
            "source": str(mineru_dir) if has_mineru else str(fallback_path),
            "max_hops": args.max_hops,
            "context_window": args.context_window,
        },
        "documents": {},
    }

    for doc_id, dag in dags.items():
        doc_data = dag.to_dict()
        doc_data["multimodal_pairs"] = [
            p.to_dict() for p in all_pairs.get(doc_id, [])
        ]
        output_data["documents"][doc_id] = doc_data

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)
    print(f"\nOutput saved to: {output_path}")

    # --- Save report ---
    stats = builder.compute_stats(dags, all_pairs)
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)
    print(f"Report saved to: {report_path}")


if __name__ == "__main__":
    main()
