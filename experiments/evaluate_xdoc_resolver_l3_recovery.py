#!/usr/bin/env python3
"""Evaluate cross-doc element resolver against L3 pass cross-doc gold rows.

Compares v0 vs v1 on recovery of source/target doc pairs and endpoint element pairs.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_V1_PAIRS = ROOT / "data/05_eval/xdoc_element_resolver_v1_latest/cross_doc_pairs_v1.json"
DEFAULT_V0_PAIRS = ROOT / "data/05_eval/xdoc_element_resolver_v0_latest/cross_doc_pairs_v0.json"
DEFAULT_OUT_DIR = ROOT / "data/05_eval/xdoc_element_resolver_v1_latest/l3_recovery"

# L3 pass files with reasoning chains (the design's expected gold source)
DEFAULT_L3_GLOB = [
    "data/03_queries/l3_enriched_v3_rerun2_pass.jsonl",
    "data/03_queries/l3_enriched_v3_new82_rerun2_pass.jsonl",
    "archive/data/batch_phase2a/l3_reasoning_chain_queries_pass.jsonl",
]


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def load_pairs_index(pairs_path: Path) -> dict[str, Any]:
    """Load resolver pairs and build lookup indexes."""
    data = read_json(pairs_path)
    pairs = data.get("pairs", [])
    version = data.get("metadata", {}).get("source", "unknown")
    # Build indexes:
    # 1. (source_doc, target_doc) -> list of pairs
    # 2. (element_a_id, element_b_id) -> pair
    doc_pair_idx: dict[tuple[str, str], list[dict]] = defaultdict(list)
    endpoint_pair_idx: dict[tuple[str, str], dict] = {}
    source_endpoint_idx: dict[tuple[str, str], list[dict]] = defaultdict(list)
    target_endpoint_idx: dict[tuple[str, str], list[dict]] = defaultdict(list)

    for p in pairs:
        sd, td = p["source_doc"], p["target_doc"]
        doc_pair_idx[(sd, td)].append(p)
        # Also store (td, sd) for unordered matching
        doc_pair_idx[(td, sd)].append(p)

        ea = p["element_a_id"]
        eb = p["element_b_id"]
        key_ab = (ea, eb)
        key_ba = (eb, ea)
        if key_ab not in endpoint_pair_idx:
            endpoint_pair_idx[key_ab] = p
        if key_ba not in endpoint_pair_idx:
            endpoint_pair_idx[key_ba] = p

        source_endpoint_idx[(ea, td)].append(p)
        target_endpoint_idx[(sd, eb)].append(p)

    return {
        "version": version,
        "n_pairs": len(pairs),
        "doc_pair_idx": doc_pair_idx,
        "endpoint_pair_idx": endpoint_pair_idx,
        "source_endpoint_idx": source_endpoint_idx,
        "target_endpoint_idx": target_endpoint_idx,
        "pairs": pairs,
    }


def pair_anchor_reason(pair: dict[str, Any]) -> str:
    meta = pair.get("hub_metadata") or {}
    detail = meta.get("target_resolution_detail") or {}
    return str(detail.get("anchor_reason") or meta.get("target_resolution_method") or "unknown")


def pair_target_route(pair: dict[str, Any]) -> str:
    method = str((pair.get("hub_metadata") or {}).get("target_resolution_method") or "")
    if "explicit_number" in method:
        return "explicit_number"
    if method == "target_caption_overlap":
        return "caption_overlap"
    return method or "unknown"


def discover_l3_gold(l3_paths: list[Path]) -> list[dict[str, Any]]:
    """Extract cross-doc L3 pass rows from known pass files."""
    gold = []
    rejected = Counter()
    for fpath in l3_paths:
        if not fpath.exists():
            print(f"  WARNING: {fpath} not found, skipping")
            continue
        with fpath.open(encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                row = json.loads(line)
                eids = row.get("element_ids") or []
                if isinstance(eids, str):
                    eids = [eids]
                if len(eids) < 2:
                    rejected["too_few_elements"] += 1
                    continue
                # Determine doc_ids from element IDs
                endpoint_eids = eids[-2:]
                doc_ids = set()
                for eid in endpoint_eids:
                    parts = str(eid).split("_", 1)
                    if parts:
                        doc_ids.add(parts[0])
                if len(doc_ids) < 2:
                    rejected["not_cross_doc"] += 1
                    continue
                gold.append({
                    "query_id": row.get("query_id", ""),
                    "source_file": str(fpath),
                    "element_ids": endpoint_eids,
                    "source_doc": sorted(doc_ids)[0],
                    "target_doc": sorted(doc_ids)[1],
                    "element_a_type": row.get("element_a_type", ""),
                    "element_b_type": row.get("element_b_type", ""),
                })
    print(f"  L3 cross-doc gold rows: {len(gold)}")
    print(f"  Rejected: {dict(rejected)}")
    return gold


def evaluate_recovery(
    gold: list[dict[str, Any]],
    index: dict[str, Any],
    label: str,
    K_values: list[int] = [100, 500, 1000, 5000],
) -> dict[str, Any]:
    """Compute recovery metrics for a resolver's output."""
    pairs = index["pairs"]
    max_k = min(max(K_values), len(pairs))
    top_pairs = pairs[:max_k]

    doc_pair_set: set[tuple[str, str]] = set()
    endpoint_pair_set: set[tuple[str, str]] = set()
    for p in top_pairs:
        doc_pair_set.add((p["source_doc"], p["target_doc"]))
        doc_pair_set.add((p["target_doc"], p["source_doc"]))
        endpoint_pair_set.add((p["element_a_id"], p["element_b_id"]))
        endpoint_pair_set.add((p["element_b_id"], p["element_a_id"]))

    results: dict[str, Any] = {
        "label": label,
        "n_pairs": index["n_pairs"],
        "n_gold": len(gold),
    }

    # Per-K metrics
    for k in K_values:
        if k > len(pairs):
            continue
        tk_pairs = pairs[:k]
        tk_doc_set: set[tuple[str, str]] = set()
        tk_ep_set: set[tuple[str, str]] = set()
        for p in tk_pairs:
            tk_doc_set.add((p["source_doc"], p["target_doc"]))
            tk_doc_set.add((p["target_doc"], p["source_doc"]))
            tk_ep_set.add((p["element_a_id"], p["element_b_id"]))
            tk_ep_set.add((p["element_b_id"], p["element_a_id"]))

        doc_pair_hits = 0
        endpoint_hits = 0
        miss_reasons = Counter()
        method_hits = Counter()
        anchor_reason_hits = Counter()
        route_hits = Counter()
        hit_examples = []
        target_method_inventory = Counter(
            p["hub_metadata"]["target_resolution_method"] for p in tk_pairs
        )
        anchor_reason_inventory = Counter(pair_anchor_reason(p) for p in tk_pairs)
        route_inventory = Counter(pair_target_route(p) for p in tk_pairs)
        for g in gold:
            g_dp = (g["source_doc"], g["target_doc"])
            if g_dp in tk_doc_set:
                doc_pair_hits += 1
            g_ep = (g["element_ids"][0], g["element_ids"][1])
            if g_ep in tk_ep_set:
                endpoint_hits += 1
                # Find which method resolved it
                for p in tk_pairs:
                    pe_ab = (p["element_a_id"], p["element_b_id"])
                    pe_ba = (p["element_b_id"], p["element_a_id"])
                    if g_ep == pe_ab or g_ep == pe_ba:
                        method = p["hub_metadata"]["target_resolution_method"]
                        method_hits[method] += 1
                        anchor_reason_hits[pair_anchor_reason(p)] += 1
                        route_hits[pair_target_route(p)] += 1
                        if len(hit_examples) < 10:
                            hit_examples.append({
                                "query_id": g["query_id"],
                                "gold_element_ids": list(g["element_ids"]),
                                "pair_id": p["pair_id"],
                                "target_resolution_method": method,
                                "target_anchor_reason": pair_anchor_reason(p),
                                "target_route": pair_target_route(p),
                            })
                        break
            else:
                # Classify miss
                if g_dp in tk_doc_set:
                    miss_reasons["doc_pair_found_endpoint_missed"] += 1
                else:
                    miss_reasons["doc_pair_not_found"] += 1

        results[f"K={k}"] = {
            "doc_pair_recall": round(doc_pair_hits / len(gold), 4) if gold else 0,
            "endpoint_pair_recall": round(endpoint_hits / len(gold), 4) if gold else 0,
            "doc_pair_hits": doc_pair_hits,
            "endpoint_hits": endpoint_hits,
            "miss_reasons": dict(miss_reasons),
            "method_breakdown": dict(method_hits),
            "anchor_reason_breakdown": dict(anchor_reason_hits),
            "target_route_breakdown": dict(route_hits),
            "explicit_endpoint_hits": route_hits.get("explicit_number", 0),
            "caption_overlap_endpoint_hits": route_hits.get("caption_overlap", 0),
            "target_method_inventory": dict(target_method_inventory),
            "anchor_reason_inventory": dict(anchor_reason_inventory),
            "target_route_inventory": dict(route_inventory),
            "hit_examples": hit_examples,
        }

    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate xdoc resolver L3 recovery.")
    parser.add_argument("--pairs", default=str(DEFAULT_V1_PAIRS))
    parser.add_argument("--baseline-pairs", default=str(DEFAULT_V0_PAIRS))
    parser.add_argument("--l3-input", action="append", default=[])
    parser.add_argument("--output-dir", default=str(DEFAULT_OUT_DIR))
    parser.add_argument("--K", type=int, nargs="+", default=[100, 500, 1000, 5000])
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Discover gold
    print("=== Gold Discovery ===")
    l3_paths = [Path(p) for p in args.l3_input] if args.l3_input else [ROOT / p for p in DEFAULT_L3_GLOB]
    gold = discover_l3_gold(l3_paths)

    if not gold:
        print("ERROR: No cross-doc L3 gold rows found. Check input paths.")
        return

    # Check element ID intersection
    elems = read_json(ROOT / "data/01_graphs/multimodal_elements_v2.json")
    all_elem_ids = set()
    for doc_id, doc in (elems.get("documents") or {}).items():
        all_elem_ids.update(doc.get("elements", {}).keys())

    gold_elem_ids = set()
    for g in gold:
        gold_elem_ids.update(g["element_ids"])
    intersection_rate = len(gold_elem_ids & all_elem_ids) / max(1, len(gold_elem_ids))
    print(f"  Element ID intersection: {intersection_rate:.1%}")

    # Save gold
    with open(out_dir / "l3_crossdoc_gold.jsonl", "w") as f:
        for g in gold:
            f.write(json.dumps(g, ensure_ascii=False) + "\n")

    gold_report = {
        "total_gold_rows": len(gold),
        "gold_element_ids_total": len(gold_elem_ids),
        "gold_element_ids_in_index": len(gold_elem_ids & all_elem_ids),
        "intersection_rate": round(intersection_rate, 4),
        "source_files": [str(p) for p in l3_paths],
    }
    with open(out_dir / "gold_discovery_report.json", "w") as f:
        json.dump(gold_report, f, indent=2)

    # Evaluate v1
    print("\n=== v1 Evaluation ===")
    v1_index = load_pairs_index(Path(args.pairs))
    v1_results = evaluate_recovery(gold, v1_index, "v1", args.K)

    # Evaluate v0
    v0_path = Path(args.baseline_pairs)
    v0_results = None
    if v0_path.exists():
        print("\n=== v0 Evaluation ===")
        v0_index = load_pairs_index(v0_path)
        v0_results = evaluate_recovery(gold, v0_index, "v0", args.K)
    else:
        print(f"\n  v0 pairs not found at {v0_path}, skipping baseline comparison")

    # Comparison table
    report_lines = [
        "# L3 Cross-Doc Recovery Report",
        "",
        f"Gold rows: {len(gold)}",
        f"Element ID intersection rate: {intersection_rate:.1%}",
        "",
        "## v0 vs v1 Comparison",
        "",
        "| metric | v0 | v1 | delta |",
        "|---|---|---|---|",
    ]

    for k in args.K:
        r_v1 = v1_results.get(f"K={k}", {})
        r_v0 = v0_results.get(f"K={k}", {}) if v0_results else {}

        for metric in ["doc_pair_recall", "endpoint_pair_recall"]:
            v0_val = r_v0.get(metric, 0)
            v1_val = r_v1.get(metric, 0)
            delta = v1_val - v0_val
            delta_str = f"+{delta:.4f}" if delta > 0 else f"{delta:.4f}"
            report_lines.append(
                f"| {metric} @K={k} | {v0_val:.4f} | {v1_val:.4f} | {delta_str} |")

    report_lines.extend([
        "",
        "## v1 Per-K Detail",
        "",
    ])
    for k in args.K:
        r = v1_results.get(f"K={k}", {})
        report_lines.append(f"### K={k}")
        report_lines.append(f"- doc_pair_recall: {r.get('doc_pair_recall', 0):.4f} ({r.get('doc_pair_hits', 0)}/{len(gold)})")
        report_lines.append(f"- endpoint_pair_recall: {r.get('endpoint_pair_recall', 0):.4f} ({r.get('endpoint_hits', 0)}/{len(gold)})")
        report_lines.append(f"- miss_reasons: `{r.get('miss_reasons', {})}`")
        report_lines.append(f"- method_breakdown: `{r.get('method_breakdown', {})}`")
        report_lines.append(f"- anchor_reason_breakdown: `{r.get('anchor_reason_breakdown', {})}`")
        report_lines.append(f"- target_route_breakdown: `{r.get('target_route_breakdown', {})}`")
        report_lines.append(f"- explicit_endpoint_hits: **{r.get('explicit_endpoint_hits', 0)}**")
        report_lines.append(f"- caption_overlap_endpoint_hits: **{r.get('caption_overlap_endpoint_hits', 0)}**")
        report_lines.append(f"- target_method_inventory: `{r.get('target_method_inventory', {})}`")
        report_lines.append("")

    report = "\n".join(report_lines)
    print("\n" + report)

    with open(out_dir / "l3_recovery_report.md", "w") as f:
        f.write(report)
    with open(out_dir / "l3_recovery_report.json", "w") as f:
        json.dump({"v1": v1_results, "v0": v0_results, "gold_report": gold_report}, f, indent=2)

    print(f"\nReports saved to {out_dir}")


if __name__ == "__main__":
    main()
