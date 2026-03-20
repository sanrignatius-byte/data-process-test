#!/usr/bin/env python3
"""Filter 3-hop hub candidates for Level 3 reasoning chain generation.

Takes enriched hub candidates (hub_candidates_enriched_v3.json) which already
have MinerU-mapped element IDs and full element details, and filters to
candidates suitable for 3-step reasoning chains:
  - hop_distance >= 3
  - Both endpoints have non-empty context
  - Cross-modal (different element types)
  - Adds reasoning_chain_target=True flag for the generator

Outputs enriched candidates compatible with generate_multihop_l1_queries.py.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"


def has_context(elem: Dict[str, Any]) -> bool:
    """Check element has usable context text."""
    for key in ("context_before", "context_after", "caption", "content", "enriched_content"):
        val = (elem.get(key) or "").strip()
        if len(val) >= 20:
            return True
    return False


def main() -> None:
    ap = argparse.ArgumentParser(description="Filter candidates for Level 3 generation")
    ap.add_argument(
        "--candidates",
        default="",
        help="Enriched hub candidates JSON (auto-detected from data111/ or data/)",
    )
    ap.add_argument(
        "--output",
        default=str(DATA_DIR / "m2" / "l3_candidates_filtered.json"),
    )
    ap.add_argument("--min-hops", type=int, default=3)
    ap.add_argument("--max-candidates", type=int, default=200)
    args = ap.parse_args()

    # Auto-detect enriched candidates file
    if args.candidates:
        cand_path = Path(args.candidates)
    else:
        candidates_search = [
            DATA_DIR / "hub_candidates_enriched.json",
            PROJECT_ROOT / "data111" / "hub_candidates_enriched_v3.json",
            PROJECT_ROOT / "data111" / "hub_candidates_enriched_v2.json",
        ]
        cand_path = None
        for p in candidates_search:
            if p.exists():
                cand_path = p
                break
        if not cand_path:
            print("ERROR: No enriched hub candidates found. Searched:")
            for p in candidates_search:
                print(f"  {p}")
            return

    print(f"Loading candidates from {cand_path}...")
    cand_data = json.loads(cand_path.read_text(encoding="utf-8"))
    all_pairs = cand_data.get("pairs", [])
    print(f"Total pairs: {len(all_pairs)}")

    # Filter
    filtered: List[Dict[str, Any]] = []
    skip_reasons: Dict[str, int] = {}

    for pair in all_pairs:
        hop = pair.get("hop_distance", 0)
        if hop < args.min_hops:
            skip_reasons["hop_too_short"] = skip_reasons.get("hop_too_short", 0) + 1
            continue

        elem_a = pair.get("element_a", {})
        elem_b = pair.get("element_b", {})

        if not elem_a or not elem_b:
            skip_reasons["missing_element_data"] = skip_reasons.get("missing_element_data", 0) + 1
            continue

        a_type = pair.get("element_a_type", elem_a.get("element_type", ""))
        b_type = pair.get("element_b_type", elem_b.get("element_type", ""))

        if a_type == b_type:
            skip_reasons["same_type"] = skip_reasons.get("same_type", 0) + 1
            continue

        if not has_context(elem_a):
            skip_reasons["no_context_a"] = skip_reasons.get("no_context_a", 0) + 1
            continue

        if not has_context(elem_b):
            skip_reasons["no_context_b"] = skip_reasons.get("no_context_b", 0) + 1
            continue

        # Mark as Level 3 reasoning chain target
        pair_copy = dict(pair)
        pair_copy["reasoning_chain_target"] = True
        pair_copy["min_reasoning_steps"] = 3
        # Ensure pair_id has l3 prefix for clarity
        if not pair_copy.get("pair_id", "").startswith("l3_"):
            pair_copy["pair_id"] = f"l3_{pair_copy.get('pair_id', '')}"

        filtered.append(pair_copy)

        if len(filtered) >= args.max_candidates:
            break

    # Save
    output = {
        "metadata": {
            "source": str(cand_path),
            "total_pairs": len(all_pairs),
            "filtered": len(filtered),
            "min_hops": args.min_hops,
        },
        "pairs": filtered,
    }
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(output, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"\nFiltered: {len(filtered)} / {len(all_pairs)}")
    print(f"Output: {out_path}")

    if skip_reasons:
        print(f"\nSkip reasons:")
        for reason, count in sorted(skip_reasons.items(), key=lambda x: -x[1]):
            print(f"  {reason}: {count}")

    # Distribution
    type_dist: Dict[str, int] = {}
    cross_doc = 0
    for p in filtered:
        type_dist[p["pair_type"]] = type_dist.get(p["pair_type"], 0) + 1
        if p.get("is_cross_doc"):
            cross_doc += 1
    print(f"\nPair type distribution:")
    for t, c in sorted(type_dist.items(), key=lambda x: -x[1]):
        print(f"  {t}: {c}")
    print(f"\nCross-doc: {cross_doc} / {len(filtered)}")


if __name__ == "__main__":
    main()
