#!/usr/bin/env python3
"""Select multi-hop, cross-modal candidates from the DAG for L1 query generation.

Reads multimodal_elements.json, filters for cross-modal pairs (figure↔table,
figure↔formula, formula↔table), deduplicates against existing L1 queries, and
outputs prioritized candidates.

Zero API cost — pure filtering and sorting.
"""

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Set, Tuple


# Priority order for modality combinations (lower = higher priority)
MODALITY_PRIORITY = {
    ("figure", "table"): 0,
    ("table", "figure"): 0,
    ("figure", "formula"): 1,
    ("formula", "figure"): 1,
    ("formula", "table"): 2,
    ("table", "formula"): 2,
}


def normalize_pair_type(a_type: str, b_type: str) -> str:
    """Canonical label for modality pair (alphabetical order)."""
    types = sorted([a_type, b_type])
    return f"{types[0]}+{types[1]}"


def is_cross_modal(a_type: str, b_type: str) -> bool:
    """True if the pair crosses modality boundaries (excludes same-type)."""
    return (a_type, b_type) in MODALITY_PRIORITY


def load_existing_element_ids(l1_path: str) -> Set[str]:
    """Extract figure_ids from existing L1 queries to detect overlap."""
    ids = set()
    p = Path(l1_path)
    if not p.exists():
        return ids
    with open(p, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            # figure_id like "1306.5204_fig_1" — map to element_id format
            fid = obj.get("figure_id", "")
            doc_id = obj.get("doc_id", "")
            if fid:
                ids.add(fid)
                # Also store doc_id+figure_id combo for dedup
                ids.add(f"{doc_id}:{fid}")
    return ids


def element_has_content(elem: Dict[str, Any]) -> bool:
    """Check if an element has usable visual or textual content."""
    if elem.get("image_path") and Path(elem["image_path"]).exists():
        return True
    content = elem.get("content", "") or ""
    if len(content) > 50:
        return True
    caption = elem.get("caption", "") or ""
    if len(caption) > 30:
        return True
    return False


def collect_edge_context(
    edges: List[Dict], path: List[str]
) -> List[Dict[str, str]]:
    """Collect context_snippets for edges along the path."""
    # Build edge lookup: (source, target) -> edge
    edge_map: Dict[Tuple[str, str], Dict] = {}
    for e in edges:
        key = (e["source_id"], e["target_id"])
        edge_map[key] = e
        # Also reverse direction
        edge_map[(e["target_id"], e["source_id"])] = e

    contexts = []
    for i in range(len(path) - 1):
        key = (path[i], path[i + 1])
        edge = edge_map.get(key)
        if edge:
            contexts.append({
                "source": edge["source_id"],
                "target": edge["target_id"],
                "ref_text": edge.get("ref_text", ""),
                "context_snippet": edge.get("context_snippet", ""),
            })
    return contexts


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Select multi-hop candidates from DAG for L1 query generation"
    )
    ap.add_argument(
        "--elements",
        default="data/multimodal_elements.json",
        help="Path to multimodal_elements.json",
    )
    ap.add_argument(
        "--existing-l1",
        default="data/l1_cross_modal_queries_v3.jsonl",
        help="Existing L1 queries for deduplication",
    )
    ap.add_argument(
        "--output",
        default="data/multihop_l1_candidates.json",
        help="Output path for candidate pairs",
    )
    ap.add_argument("--max-pairs", type=int, default=150, help="Total pair limit")
    ap.add_argument("--max-per-doc", type=int, default=5, help="Max pairs per document")
    ap.add_argument(
        "--max-hops", type=int, default=2, help="Max hop distance (default 2)"
    )
    ap.add_argument(
        "--min-quality", type=float, default=0.6, help="Min quality_score for pairs"
    )
    args = ap.parse_args()

    # Load DAG
    elements_path = Path(args.elements)
    if not elements_path.exists():
        print(f"ERROR: {elements_path} not found")
        sys.exit(1)
    with open(elements_path, encoding="utf-8") as f:
        dag = json.load(f)

    # Load existing L1 element IDs for overlap detection
    existing_ids = load_existing_element_ids(args.existing_l1)
    print(f"Existing L1 element IDs: {len(existing_ids)}")

    # Collect all cross-modal candidates
    all_candidates: List[Dict[str, Any]] = []
    stats = defaultdict(int)

    for doc_id, doc in dag.get("documents", {}).items():
        elements = doc.get("elements", {})
        edges = doc.get("edges", [])
        pairs = doc.get("multimodal_pairs", [])

        for pair in pairs:
            a_type = pair["element_a_type"]
            b_type = pair["element_b_type"]
            hop = pair["hop_distance"]
            qscore = pair.get("quality_score", 0)

            stats["total_pairs"] += 1

            # Filter: cross-modal only
            if not is_cross_modal(a_type, b_type):
                stats["skip_same_modal"] += 1
                continue

            # Filter: hop distance
            if hop > args.max_hops:
                stats["skip_too_many_hops"] += 1
                continue

            # Filter: quality
            if qscore < args.min_quality:
                stats["skip_low_quality"] += 1
                continue

            # Get elements
            elem_a = elements.get(pair["element_a_id"])
            elem_b = elements.get(pair["element_b_id"])
            if not elem_a or not elem_b:
                stats["skip_missing_element"] += 1
                continue

            # Filter: at least one element has usable content
            if not element_has_content(elem_a) and not element_has_content(elem_b):
                stats["skip_no_content"] += 1
                continue

            # Check overlap with existing L1
            a_id = pair["element_a_id"]
            b_id = pair["element_b_id"]
            # Existing L1 uses figure_id format like "doc_fig_N" not "doc_figure_N"
            overlap = (
                a_id in existing_ids
                or b_id in existing_ids
                or f"{doc_id}:{a_id}" in existing_ids
                or f"{doc_id}:{b_id}" in existing_ids
            )

            # Collect edge contexts along the path
            path = pair.get("path", [a_id, b_id])
            edge_contexts = collect_edge_context(edges, path)

            pair_type = normalize_pair_type(a_type, b_type)
            modality_prio = MODALITY_PRIORITY.get((a_type, b_type), 99)

            candidate = {
                "pair_id": pair["pair_id"],
                "doc_id": doc_id,
                "element_a_id": a_id,
                "element_b_id": b_id,
                "element_a_type": a_type,
                "element_b_type": b_type,
                "pair_type": pair_type,
                "hop_distance": hop,
                "path": path,
                "quality_score": qscore,
                "overlap_with_existing_l1": overlap,
                # Element details for prompt building
                "element_a": {
                    "element_id": elem_a["element_id"],
                    "element_type": elem_a["element_type"],
                    "caption": elem_a.get("caption", ""),
                    "content": (elem_a.get("content", "") or "")[:1000],
                    "image_path": elem_a.get("image_path"),
                    "context_before": (elem_a.get("context_before", "") or "")[:500],
                    "context_after": (elem_a.get("context_after", "") or "")[:500],
                },
                "element_b": {
                    "element_id": elem_b["element_id"],
                    "element_type": elem_b["element_type"],
                    "caption": elem_b.get("caption", ""),
                    "content": (elem_b.get("content", "") or "")[:1000],
                    "image_path": elem_b.get("image_path"),
                    "context_before": (elem_b.get("context_before", "") or "")[:500],
                    "context_after": (elem_b.get("context_after", "") or "")[:500],
                },
                "edge_contexts": edge_contexts,
                # Sort keys
                "_modality_prio": modality_prio,
                "_hop_prio": hop,
                "_overlap_prio": 1 if overlap else 0,
            }
            all_candidates.append(candidate)

    print(f"\nFiltering stats:")
    for k, v in sorted(stats.items()):
        print(f"  {k}: {v}")
    print(f"  candidates_after_filter: {len(all_candidates)}")

    # Sort within each type: no overlap first → fewer hops → higher quality
    all_candidates.sort(
        key=lambda c: (
            c["_overlap_prio"],  # 0 = no overlap = higher priority
            c["_hop_prio"],
            -c["quality_score"],
        )
    )

    # Group by pair_type
    by_type: Dict[str, List[Dict]] = defaultdict(list)
    for c in all_candidates:
        by_type[c["pair_type"]].append(c)

    # Quota-based selection: ensure each type gets representation
    # figure+table gets 60%, figure+formula 30%, formula+table 10%
    type_quotas = {
        "figure+table": int(args.max_pairs * 0.60),
        "figure+formula": int(args.max_pairs * 0.30),
        "formula+table": args.max_pairs,  # remainder
    }
    print(f"\n  Quotas: {type_quotas}")

    doc_counts: Dict[str, int] = defaultdict(int)
    selected: List[Dict[str, Any]] = []
    type_selected: Dict[str, int] = defaultdict(int)

    # First pass: fill each type up to its quota
    for ptype in ["figure+table", "figure+formula", "formula+table"]:
        quota = type_quotas.get(ptype, 0)
        for c in by_type.get(ptype, []):
            if type_selected[ptype] >= quota:
                break
            if len(selected) >= args.max_pairs:
                break
            doc_id = c["doc_id"]
            if doc_counts[doc_id] >= args.max_per_doc:
                continue
            doc_counts[doc_id] += 1
            type_selected[ptype] += 1
            selected.append(c)

    # Second pass: fill remaining slots from any type (prefer underrepresented)
    if len(selected) < args.max_pairs:
        remaining = [c for c in all_candidates if c not in selected]
        for c in remaining:
            if len(selected) >= args.max_pairs:
                break
            doc_id = c["doc_id"]
            if doc_counts[doc_id] >= args.max_per_doc:
                continue
            doc_counts[doc_id] += 1
            selected.append(c)

    # Remove internal sort keys
    for c in selected:
        del c["_modality_prio"]
        del c["_hop_prio"]
        del c["_overlap_prio"]

    # Summary stats
    type_counts = defaultdict(int)
    hop_counts = defaultdict(int)
    overlap_count = 0
    docs_covered = set()
    for c in selected:
        type_counts[c["pair_type"]] += 1
        hop_counts[c["hop_distance"]] += 1
        if c["overlap_with_existing_l1"]:
            overlap_count += 1
        docs_covered.add(c["doc_id"])

    # Save output
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    output = {
        "metadata": {
            "source": str(elements_path),
            "existing_l1": args.existing_l1,
            "max_pairs": args.max_pairs,
            "max_per_doc": args.max_per_doc,
            "max_hops": args.max_hops,
            "min_quality": args.min_quality,
        },
        "summary": {
            "total_selected": len(selected),
            "by_type": dict(type_counts),
            "by_hop": {str(k): v for k, v in sorted(hop_counts.items())},
            "overlap_with_existing_l1": overlap_count,
            "docs_covered": len(docs_covered),
        },
        "pairs": selected,
    }
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    print(f"\n{'='*60}")
    print(f"Multi-hop Candidate Selection Summary")
    print(f"{'='*60}")
    print(f"  Total selected:         {len(selected)}")
    print(f"  By type:")
    for t, cnt in sorted(type_counts.items()):
        print(f"    {t}: {cnt}")
    print(f"  By hop distance:")
    for h, cnt in sorted(hop_counts.items()):
        print(f"    {h}-hop: {cnt}")
    print(f"  Overlap with existing:  {overlap_count}")
    print(f"  Documents covered:      {len(docs_covered)}")
    print(f"  Output: {out_path}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
