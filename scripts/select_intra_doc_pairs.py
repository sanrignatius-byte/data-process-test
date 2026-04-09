#!/usr/bin/env python3
"""Select intra-document multimodal element pairs for query generation.

Reads multimodal_elements.json and produces enriched pairs in the same
format as hub_candidates_enriched_v3.json — ready to feed directly into
generate_multihop_l1_queries.py.

All pairs are strictly intra-document (no cross-doc leaks).

Usage::

    python scripts/select_intra_doc_pairs.py \\
        --elements data/01_graphs/multimodal_elements.json \\
        --output data/02_enriched/intra_doc_pairs_v1.json \\
        --strategy all \\
        --max-per-doc 15

    # Only direct-reference figure+table pairs:
    python scripts/select_intra_doc_pairs.py \\
        --elements data/01_graphs/multimodal_elements.json \\
        --output data/02_enriched/intra_doc_pairs_direct_ft.json \\
        --strategy direct \\
        --pair-type figure+table
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.pairing import IntraDocPairSelector  # noqa: E402


def build_output(
    pairs: list,
    selector_stats: Dict[str, Any],
    args: argparse.Namespace,
) -> Dict[str, Any]:
    """Build the output JSON in hub_candidates_enriched_v3-compatible format."""
    # Summary stats
    by_type: Dict[str, int] = Counter()
    by_hop: Dict[str, int] = Counter()
    by_strategy: Dict[str, int] = Counter()
    docs_covered: Set[str] = set()

    pair_dicts = []
    for p in pairs:
        d = p.to_dict()
        pair_dicts.append(d)
        by_type[p.pair_type] += 1
        by_hop[str(p.hop_distance)] += 1
        by_strategy[p.strategy] += 1
        docs_covered.add(p.doc_id)

    return {
        "metadata": {
            "source": "select_intra_doc_pairs.py",
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "strategy": args.strategy,
            "max_per_doc": args.max_per_doc,
            "pair_type_filter": args.pair_type or "none",
            "min_quality": args.min_quality,
            "selector_stats": selector_stats,
        },
        "summary": {
            "total_selected": len(pair_dicts),
            "by_type": dict(by_type),
            "by_hop": dict(by_hop),
            "by_strategy": dict(by_strategy),
            "docs_covered": len(docs_covered),
            "cross_doc": 0,  # guaranteed by design
            "intra_doc": len(pair_dicts),
        },
        "pairs": pair_dicts,
        # Empty for compat; not needed for intra-doc selection.
        "adjacent_bridge_elements": {},
        "adjacent_bridge_adjacency": [],
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Select intra-document multimodal element pairs.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--elements",
        default="data/01_graphs/multimodal_elements.json",
        help="Path to multimodal_elements.json (default: %(default)s)",
    )
    parser.add_argument(
        "--output", "-o",
        required=True,
        help="Output JSON path (hub_candidates_enriched format)",
    )
    parser.add_argument(
        "--strategy",
        choices=["direct", "2hop", "section", "all"],
        default="all",
        help="Pair selection strategy (default: all)",
    )
    parser.add_argument(
        "--max-per-doc",
        type=int,
        default=15,
        help="Maximum pairs per document (default: 15)",
    )
    parser.add_argument(
        "--pair-type",
        type=str,
        default=None,
        help="Filter by pair type, e.g. 'figure+table'. "
             "Comma-separated for multiple: 'figure+table,figure+formula'",
    )
    parser.add_argument(
        "--min-quality",
        type=float,
        default=0.0,
        help="Minimum quality score threshold (default: 0.0)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Global limit on total pairs (0 = no limit)",
    )
    args = parser.parse_args()

    # Load and select
    print(f"Loading elements from {args.elements} ...")
    selector = IntraDocPairSelector.from_file(args.elements)
    stats = selector.stats()
    print(f"  {stats['documents']} documents, {stats['elements']} elements, "
          f"{stats['edges']} edges")
    print(f"  Type distribution: {stats['type_distribution']}")
    print(f"  Docs with ≥2 modalities: {stats['docs_with_multi_modal']}")

    # Parse pair type filter
    pair_types: Optional[Set[str]] = None
    if args.pair_type:
        pair_types = set(args.pair_type.split(","))

    print(f"\nSelecting pairs: strategy={args.strategy}, "
          f"max_per_doc={args.max_per_doc}, "
          f"pair_types={pair_types or 'all'}, "
          f"min_quality={args.min_quality}")

    pairs = selector.select(
        strategy=args.strategy,
        max_per_doc=args.max_per_doc,
        pair_types=pair_types,
        min_quality=args.min_quality,
    )

    if args.limit > 0:
        pairs = pairs[:args.limit]

    # Build output
    result = build_output(pairs, stats, args)

    # Write
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2, ensure_ascii=False),
                        encoding="utf-8")

    print(f"\n✅ Wrote {len(pairs)} pairs to {args.output}")
    print(f"   By type: {dict(result['summary']['by_type'])}")
    print(f"   By hop:  {dict(result['summary']['by_hop'])}")
    print(f"   By strategy: {dict(result['summary']['by_strategy'])}")
    print(f"   Docs covered: {result['summary']['docs_covered']}")


if __name__ == "__main__":
    main()
