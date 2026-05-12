#!/usr/bin/env python3
"""Filter pair candidates to strict intra-doc pairs with enriched endpoints."""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, Set, Tuple

import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.pair_filters import filter_intra_doc_pairs, pair_doc_id  # noqa: E402


ENRICH_KEYS = ("enriched_title", "enriched_content", "enriched_metadata")
MODAL_TYPES = {"figure", "table", "formula", "equation"}


def has_enrichment(elem: Dict[str, Any]) -> bool:
    return any(bool(elem.get(key)) for key in ENRICH_KEYS)


def pair_endpoints(pair: Dict[str, Any]) -> Iterable[Tuple[str, Dict[str, Any]]]:
    for key in ("element_a", "element_b"):
        elem = pair.get(key)
        if isinstance(elem, dict):
            yield key, elem


def parse_int_set(raw: str) -> Set[int]:
    if not raw:
        return set()
    return {int(x.strip()) for x in raw.split(",") if x.strip()}


def multimodal_path_elements(pair: Dict[str, Any]) -> list[Dict[str, Any]]:
    """Return unique multimodal element blobs participating in the path.

    In the production discussion, "2-hop/3-hop" means two or three multimodal
    elements in the reasoning path, not the graph-theoretic hop_distance field.
    `node_group` is the authoritative bundle when present; otherwise fall back
    to endpoint elements.
    """
    elems: list[Dict[str, Any]] = []
    seen: Set[str] = set()

    candidates = pair.get("node_group") or []
    if not isinstance(candidates, list) or not candidates:
        candidates = [pair.get("element_a"), pair.get("element_b")]

    for elem in candidates:
        if not isinstance(elem, dict):
            continue
        etype = str(elem.get("element_type", "") or elem.get("type", "")).lower()
        if etype not in MODAL_TYPES:
            continue
        eid = str(elem.get("element_id", "") or "").strip()
        if not eid or eid in seen:
            continue
        seen.add(eid)
        elems.append(elem)
    return elems


def bridge_text_available(pair: Dict[str, Any]) -> bool:
    lb = pair.get("latex_bridge") or {}
    if isinstance(lb, dict) and str(lb.get("bridge_text", "") or "").strip():
        return True
    for ec in pair.get("edge_contexts") or []:
        if isinstance(ec, dict) and str(ec.get("text", "") or "").strip():
            return True
    if str(pair.get("hub_semantic_summary", "") or "").strip():
        return True
    # The generator may still resolve bridge text from the reference graph.
    return False


def load_used_pair_ids(paths: Iterable[str]) -> Set[str]:
    used: Set[str] = set()
    for raw in paths:
        path = Path(raw)
        if not path.is_absolute():
            path = PROJECT_ROOT / path
        if not path.exists():
            continue
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    continue
                pair_id = str(obj.get("pair_id", "") or "").strip()
                if pair_id:
                    used.add(pair_id)
    return used


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input", required=True, help="Input pair JSON with pairs[]")
    ap.add_argument("--output", required=True, help="Filtered output pair JSON")
    ap.add_argument(
        "--hops",
        default="",
        help="Legacy graph hop_distance filter, e.g. '2,3'. Empty keeps all.",
    )
    ap.add_argument(
        "--multimodal-counts",
        default="",
        help=(
            "Comma-separated counts of multimodal elements in node_group to keep, "
            "e.g. '2,3'. This is the production meaning of 2-hop/3-hop."
        ),
    )
    ap.add_argument(
        "--require-both-endpoints",
        action="store_true",
        default=True,
        help="Require both element_a and element_b to have enrichment (default).",
    )
    ap.add_argument(
        "--force-reasoning-chain-target",
        action="store_true",
        help="Mark every kept pair as reasoning_chain_target=True for the 3-step prompt.",
    )
    ap.add_argument(
        "--require-candidate-bridge-text",
        action="store_true",
        help="Require bridge text already present in candidate fields before keeping a pair.",
    )
    ap.add_argument(
        "--require-all-multimodal-elements",
        action="store_true",
        help="Require every multimodal element in node_group, not just endpoints, to be enriched.",
    )
    ap.add_argument(
        "--exclude-query-jsonl",
        action="append",
        default=[],
        help="Existing query JSONL whose pair_id values should be excluded. Repeatable.",
    )
    ap.add_argument("--limit", type=int, default=0, help="Cap output pairs after filtering.")
    ap.add_argument("--shuffle", action="store_true", help="Shuffle kept pairs deterministically.")
    ap.add_argument("--seed", type=int, default=42, help="Shuffle seed.")
    args = ap.parse_args()

    in_path = Path(args.input)
    if not in_path.is_absolute():
        in_path = PROJECT_ROOT / in_path
    out_path = Path(args.output)
    if not out_path.is_absolute():
        out_path = PROJECT_ROOT / out_path

    data = json.loads(in_path.read_text(encoding="utf-8"))
    raw_pairs = data.get("pairs", []) or []
    intra_pairs, intra_stats = filter_intra_doc_pairs(raw_pairs)
    hop_filter = parse_int_set(args.hops)
    multimodal_count_filter = parse_int_set(args.multimodal_counts)
    used_pair_ids = load_used_pair_ids(args.exclude_query_jsonl)

    kept = []
    stats: Counter[str] = Counter()
    by_type: Counter[str] = Counter()
    by_hop: Counter[str] = Counter()
    docs = set()

    for pair in intra_pairs:
        if hop_filter:
            try:
                hop = int(pair.get("hop_distance", 0) or 0)
            except (TypeError, ValueError):
                hop = 0
            if hop not in hop_filter:
                stats["drop_wrong_hop"] += 1
                continue

        path_elements = multimodal_path_elements(pair)
        if multimodal_count_filter and len(path_elements) not in multimodal_count_filter:
            stats["drop_wrong_multimodal_count"] += 1
            continue

        if args.require_candidate_bridge_text and not bridge_text_available(pair):
            stats["drop_missing_bridge_text"] += 1
            continue

        if args.require_all_multimodal_elements and not all(has_enrichment(elem) for elem in path_elements):
            stats["drop_unenriched_path_element"] += 1
            continue

        pair_id = str(pair.get("pair_id", "") or "").strip()
        if pair_id and pair_id in used_pair_ids:
            stats["drop_used_pair_id"] += 1
            continue

        endpoint_items = list(pair_endpoints(pair))
        if len(endpoint_items) < 2:
            stats["drop_missing_endpoint_blob"] += 1
            continue
        enriched = {key: has_enrichment(elem) for key, elem in endpoint_items}
        if args.require_both_endpoints and not all(enriched.values()):
            stats["drop_unenriched_endpoint"] += 1
            continue

        doc_id = pair_doc_id(pair)
        if not doc_id:
            stats["drop_unresolved_doc"] += 1
            continue

        out_pair = dict(pair)
        if args.force_reasoning_chain_target:
            out_pair["reasoning_chain_target"] = True
            out_pair["production_profile"] = "reasoning_path_2hop_3hop_enriched_intra_doc"
            out_pair["multimodal_path_element_count"] = len(path_elements)
        kept.append(out_pair)
        docs.add(doc_id)
        by_type[str(pair.get("pair_type", ""))] += 1
        by_hop[str(int(pair.get("hop_distance", 0) or 0))] += 1

    if args.shuffle:
        random.Random(args.seed).shuffle(kept)
    if args.limit > 0:
        kept = kept[: args.limit]

    result = dict(data)
    result["pairs"] = kept
    result.setdefault("metadata", {})
    result["metadata"]["enriched_pair_filter"] = {
        "source": "filter_enriched_pair_candidates.py",
        "input_pairs": len(raw_pairs),
        "strict_intra_doc": intra_stats,
        "hop_filter": sorted(hop_filter),
        "multimodal_count_filter": sorted(multimodal_count_filter),
        "require_both_endpoints": args.require_both_endpoints,
        "force_reasoning_chain_target": args.force_reasoning_chain_target,
        "require_candidate_bridge_text": args.require_candidate_bridge_text,
        "require_all_multimodal_elements": args.require_all_multimodal_elements,
        "excluded_pair_ids": len(used_pair_ids),
        "drop_wrong_hop": stats["drop_wrong_hop"],
        "drop_wrong_multimodal_count": stats["drop_wrong_multimodal_count"],
        "drop_missing_bridge_text": stats["drop_missing_bridge_text"],
        "drop_unenriched_path_element": stats["drop_unenriched_path_element"],
        "drop_used_pair_id": stats["drop_used_pair_id"],
        "drop_unenriched_endpoint": stats["drop_unenriched_endpoint"],
        "drop_missing_endpoint_blob": stats["drop_missing_endpoint_blob"],
        "drop_unresolved_doc": stats["drop_unresolved_doc"],
        "output_pairs": len(kept),
    }
    result["summary"] = {
        "total_selected": len(kept),
        "by_type": dict(by_type),
        "by_hop": dict(by_hop),
        "docs_covered": len(docs),
        "strict_intra_doc_pairs": len(kept),
        "cross_doc": 0,
        "intra_doc": len(kept),
        "enriched_endpoint_policy": "both_endpoints",
        "reasoning_chain_target": bool(args.force_reasoning_chain_target),
    }

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")

    print("[done] enriched intra-doc pair filtering complete")
    print(f"  input pairs       : {len(raw_pairs)}")
    print(f"  intra kept        : {intra_stats.get('kept', 0)}")
    print(f"  cross-doc dropped : {intra_stats.get('dropped', 0)}")
    print(f"  wrong-hop dropped : {stats['drop_wrong_hop']}")
    print(f"  wrong-mm-count    : {stats['drop_wrong_multimodal_count']}")
    print(f"  no-bridge dropped : {stats['drop_missing_bridge_text']}")
    print(f"  path unenriched   : {stats['drop_unenriched_path_element']}")
    print(f"  used-pair dropped : {stats['drop_used_pair_id']}")
    print(f"  unenriched dropped: {stats['drop_unenriched_endpoint']}")
    print(f"  output pairs      : {len(kept)}")
    print(f"  docs covered      : {len(docs)}")
    print(f"  output file       : {out_path}")


if __name__ == "__main__":
    main()
