#!/usr/bin/env python3
"""Build a stratified judge pack from v1 resolver pairs.

Primary strata are target anchor/reason buckets, with pair_type balanced secondarily.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PAIRS = ROOT / "data/05_eval/xdoc_element_resolver_v1_latest/cross_doc_pairs_v1.json"
DEFAULT_OUT_DIR = ROOT / "data/05_eval/xdoc_element_resolver_v1_latest"

STRATA = [
    ("A_hard_title_window", "title_words_in_window"),
    ("B_edge_title_match", "title_match_ge_0.2"),
    ("C_soft_fanout_or_single_ref", "low_fanout/single_ref_high_prob"),
    ("D_unanchored_explicit", "unanchored"),
    ("E_overlap_high", "target_caption_overlap score >= 0.20"),
    ("F_overlap_low", "target_caption_overlap score < 0.20"),
]


def load_pairs(path: Path) -> list[dict[str, Any]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    return data.get("pairs", [])


def anchor_reason(pair: dict[str, Any]) -> str:
    detail = pair["hub_metadata"].get("target_resolution_detail", {}) or {}
    return str(detail.get("anchor_reason") or pair["hub_metadata"]["target_resolution_method"])


def bucket_pair(pair: dict[str, Any]) -> str:
    m = pair["hub_metadata"]
    reason = anchor_reason(pair)
    score = float(m["target_resolution_score"])
    if reason == "title_words_in_window":
        return "A_hard_title_window"
    if reason == "title_match_ge_0.2":
        return "B_edge_title_match"
    if reason in {"low_fanout", "single_ref_high_prob"}:
        return "C_soft_fanout_or_single_ref"
    if reason == "unanchored":
        return "D_unanchored_explicit"
    if score >= 0.20:
        return "E_overlap_high"
    return "F_overlap_low"


def build_judge_item(pair: dict[str, Any], idx: int, stratum: str) -> dict[str, Any]:
    m = pair["hub_metadata"]
    return {
        "candidate_id": pair["pair_id"],
        "judge_index": idx,
        "target_stratum": stratum,
        "target_anchor_reason": anchor_reason(pair),
        "source_doc": pair["source_doc"],
        "target_doc": pair["target_doc"],
        "source_element_id": pair["element_a_id"],
        "target_element_id": pair["element_b_id"],
        "source_element_type": pair["element_a_type"],
        "target_element_type": pair["element_b_type"],
        "pair_type": pair["pair_type"],
        "source_caption_or_content": (
            pair["element_a"].get("caption") or pair["element_a"].get("content", "")
        )[:400],
        "target_caption_or_content": (
            pair["element_b"].get("caption") or pair["element_b"].get("content", "")
        )[:400],
        "citation_bridge_text": pair["edge_contexts"][0]["context_snippet"],
        "section_title": pair["edge_contexts"][0].get("section_title", ""),
        "citation_probability": m["citation_probability"],
        "source_resolution_method": m["source_resolution_method"],
        "target_resolution_method": m["target_resolution_method"],
        "target_resolution_score": m["target_resolution_score"],
        "target_resolution_detail": m.get("target_resolution_detail", {}),
        "citation_fanout": m["citation_fanout"],
        "source_fanout_penalty": m.get("source_fanout_penalty", 1.0),
        "quality_score": pair["quality_score"],
        "element_a_image_path": pair["element_a"].get("image_path", ""),
        "element_b_image_path": pair["element_b"].get("image_path", ""),
        "question_for_judge": (
            f"Does the citation bridge in [{pair['source_doc']}] specifically discuss "
            f"the target element {pair['element_b_id']} ({pair['element_b_type']}) "
            f"from [{pair['target_doc']}]? Is the source element {pair['element_a_id']} "
            f"({pair['element_a_type']}) a plausible local anchor for this citation?"
        ),
    }


def sample_balanced_by_pair_type(pool: list[dict[str, Any]], n: int) -> list[dict[str, Any]]:
    """Deterministically sample up to n pairs, balancing pair_type when possible."""
    sorted_pool = sorted(pool, key=lambda p: (-p["quality_score"], p["pair_id"]))
    by_type: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for p in sorted_pool:
        by_type[p["pair_type"]].append(p)
    types = sorted(by_type.keys())
    selected: list[dict[str, Any]] = []
    used: set[str] = set()
    while len(selected) < n:
        progressed = False
        for pair_type in types:
            bucket = by_type[pair_type]
            while bucket and bucket[0]["pair_id"] in used:
                bucket.pop(0)
            if not bucket:
                continue
            pair = bucket.pop(0)
            selected.append(pair)
            used.add(pair["pair_id"])
            progressed = True
            if len(selected) >= n:
                break
        if not progressed:
            break
    return selected


def main() -> None:
    parser = argparse.ArgumentParser(description="Build stratified judge pack from v1 pairs.")
    parser.add_argument("--pairs", default=str(DEFAULT_PAIRS))
    parser.add_argument("--output", default=str(DEFAULT_OUT_DIR / "judge_pack_120.jsonl"))
    parser.add_argument("--n", type=int, default=120)
    args = parser.parse_args()

    pairs = load_pairs(Path(args.pairs))
    print(f"Loaded {len(pairs)} v1 pairs")

    # Stratify by target anchor reason first.
    strata: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for p in pairs:
        strata[bucket_pair(p)].append(p)

    print("Strata sizes:")
    stratum_order = [name for name, _ in STRATA]
    for bucket in stratum_order:
        print(f"  {bucket}: {len(strata[bucket])}")

    # Target: equal budget per stratum, then redistribute real shortfalls.
    target_per = args.n // len(STRATA)
    budget = {bucket: target_per for bucket in stratum_order}
    remainder = args.n - sum(budget.values())
    for bucket in stratum_order[:remainder]:
        budget[bucket] += 1

    # Redistribute short strata
    shortfall = 0
    for bucket, target in list(budget.items()):
        available = len(strata[bucket])
        if available < target:
            short = target - available
            shortfall += short
            budget[bucket] = available
            print(f"  {bucket}: short by {short}, adjusted budget to {available}")

    # Distribute shortfall to strata with excess
    if shortfall > 0:
        for bucket in ["B_edge_title_match", "E_overlap_high", "F_overlap_low",
                       "C_soft_fanout_or_single_ref", "D_unanchored_explicit",
                       "A_hard_title_window"]:
            available = len(strata[bucket])
            extra = available - budget[bucket]
            if extra > 0 and shortfall > 0:
                add = min(extra, shortfall)
                budget[bucket] += add
                shortfall -= add
                print(f"  {bucket}: +{add} from redistribution, budget now {budget[bucket]}")

    # Sample: within each stratum, balance by pair_type as much as possible.
    sampled_with_strata: list[tuple[str, dict[str, Any]]] = []
    for bucket in stratum_order:
        pool = strata[bucket]
        n = budget.get(bucket, 0)
        if n <= 0:
            continue
        sampled_with_strata.extend((bucket, p) for p in sample_balanced_by_pair_type(pool, n))

    # Trim to exact N while preserving stratum labels.
    sampled_with_strata = sampled_with_strata[:args.n]

    # Build judge items
    items = [
        build_judge_item(pair, i + 1, bucket)
        for i, (bucket, pair) in enumerate(sampled_with_strata)
    ]

    # Summary
    method_counts = Counter(item["target_resolution_method"] for item in items)
    stratum_counts = Counter(item["target_stratum"] for item in items)
    anchor_reason_counts = Counter(item["target_anchor_reason"] for item in items)
    type_counts = Counter(item["pair_type"] for item in items)
    stratum_pair_type_counts: dict[str, dict[str, int]] = {}
    for bucket in stratum_order:
        stratum_pair_type_counts[bucket] = dict(Counter(
            item["pair_type"] for item in items if item["target_stratum"] == bucket
        ))
    fanout_buckets = Counter()
    for item in items:
        f = item["citation_fanout"]
        if f <= 2:
            fanout_buckets["1-2"] += 1
        elif f <= 5:
            fanout_buckets["3-5"] += 1
        elif f <= 10:
            fanout_buckets["6-10"] += 1
        else:
            fanout_buckets[">10"] += 1

    summary = {
        "total_items": len(items),
        "target_strata": {b: budget.get(b, 0) for b in stratum_order},
        "actual_stratum_counts": dict(stratum_counts),
        "stratum_descriptions": dict(STRATA),
        "actual_method_counts": dict(method_counts),
        "anchor_reason_counts": dict(anchor_reason_counts),
        "pair_type_counts": dict(type_counts),
        "stratum_pair_type_counts": stratum_pair_type_counts,
        "fanout_buckets": dict(fanout_buckets),
        "precision_claim_scope": {
            "G3_hard_explicit_precision": "Only stratum A_hard_title_window supports a hard explicit precision claim.",
            "exploratory_strata": ["B_edge_title_match", "C_soft_fanout_or_single_ref"],
            "not_judged": True,
        },
        "judge_rubric": {
            "valid_target_element": "target element is specifically discussed or strongly implied by the citation bridge",
            "valid_source_anchor": "source element is a plausible local anchor for the citing chunk",
            "valid_chain": "source element -> citation bridge -> target element forms a coherent M4 chain",
            "verdicts": ["strong_chain", "weak_but_related", "topic_only",
                         "wrong_target", "wrong_source", "insufficient_context"],
            "primary_precision_metric": "Fraction of verdicts == strong_chain",
        },
    }

    # Write
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for item in items:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    summary_path = out_path.parent / "judge_pack_summary.json"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    # Preview markdown
    preview_lines = [
        "# Judge Pack Preview",
        "",
        f"Total items: {len(items)}",
        "",
        "## Stratum distribution",
        "",
        "| Stratum | Target | Actual |",
        "|---|---|---|",
    ]
    for b in stratum_order:
        actual = sum(1 for item in items if item["target_stratum"] == b)
        preview_lines.append(f"| {b} | {budget.get(b, 0)} | {actual} |")

    preview_lines.extend([
        "",
        "## Anchor reason distribution",
        "",
        f"`{dict(anchor_reason_counts)}`",
        "",
        "## Pair type distribution",
        "",
        f"`{dict(type_counts)}`",
        "",
        "## Fanout distribution",
        "",
        f"`{dict(fanout_buckets)}`",
        "",
        "## Sample items (first 3)",
        "",
    ])
    for bucket in stratum_order:
        bucket_items = [item for item in items if item["target_stratum"] == bucket]
        if not bucket_items:
            continue
        item = bucket_items[0]
        preview_lines.append(f"### {item['candidate_id']}")
        preview_lines.append(f"- stratum: {item['target_stratum']}")
        preview_lines.append(f"- anchor_reason: {item['target_anchor_reason']}")
        preview_lines.append(f"- source: `{item['source_element_id']}` ({item['source_element_type']})")
        preview_lines.append(f"- target: `{item['target_element_id']}` ({item['target_element_type']})")
        preview_lines.append(f"- method: {item['target_resolution_method']}")
        preview_lines.append(f"- fanout: {item['citation_fanout']}")
        preview_lines.append(f"- bridge: {item['citation_bridge_text'][:200]}")
        preview_lines.append("")

    preview_path = out_path.parent / "judge_pack_preview.md"
    with preview_path.open("w", encoding="utf-8") as f:
        f.write("\n".join(preview_lines))

    print(f"\nJudge pack: {out_path} ({len(items)} items)")
    print(f"Summary:   {summary_path}")
    print(f"Preview:   {preview_path}")
    print(f"Strata:    {dict(stratum_counts)}")
    print(f"Methods:   {dict(method_counts)}")
    print(f"Types:     {dict(type_counts)}")


if __name__ == "__main__":
    main()
