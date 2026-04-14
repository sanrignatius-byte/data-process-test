#!/usr/bin/env python3
"""
Package all pass queries into a single deduplicated delivery JSONL.

Usage:
    python scripts/package_delivery.py

Output:
    data/03_queries/delivery_v1_2026-04-13.jsonl   – full delivery file
    data/03_queries/delivery_v1_2026-04-13_stats.json – summary stats
"""

import json
import os
import collections
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
QUERY_DIR = ROOT / "data" / "03_queries"
OUT_STEM = f"delivery_v1_{datetime.now().strftime('%Y-%m-%d')}"

# ── All known pass files ──────────────────────────────────────────────
PASS_FILES = [
    # Old pass files (pre-sweep)
    ("data/03_queries/l3_enriched_v3_rerun_fixed_pass.jsonl",   "old_l3_v3"),
    ("data/03_queries/l3_enriched_v3_new82_pass.jsonl",         "old_l3_v3"),
    ("data/03_queries/l3_enriched_v3_new82_rerun2_pass.jsonl",  "old_l3_v3"),
    ("data/03_queries/l3_enriched_v3_rerun2_pass.jsonl",        "old_l3_v3"),
    ("data/03_queries/long_chain_iterative_pass.jsonl",         "old_long_chain"),
    ("data/03_queries/m2_diverse_v1_hub_kb_pass.jsonl",         "old_m2"),
    # Sweep 2026-04-12
    ("data/03_queries/sweep_2026-04-12/l3_academic_pass.jsonl",         "sweep_l3_academic"),
    ("data/03_queries/sweep_2026-04-12/l3_academic_persona_pass.jsonl", "sweep_l3_academic_persona"),
    ("data/03_queries/sweep_2026-04-12/l3_mixed_pass.jsonl",            "sweep_l3_mixed"),
    ("data/03_queries/sweep_2026-04-12/l3_mixed_persona_pass.jsonl",    "sweep_l3_mixed_persona"),
    ("data/03_queries/sweep_2026-04-12/m2_academic_pass.jsonl",         "sweep_m2_academic"),
    ("data/03_queries/sweep_2026-04-12/m2_mixed_persona_pass.jsonl",    "sweep_m2_mixed_persona"),
]

# ── Delivery schema: fields to keep (order matters) ──────────────────
DELIVERY_FIELDS = [
    "query_id",
    "query",
    "answer",
    "doc_id",
    "hop_distance",
    "query_style",
    "query_type",
    "cross_modal",
    "element_ids",
    "element_a_type",
    "element_b_type",
    "pair_type",
    "text_evidence",
    "visual_anchors",
    "image_paths",
    "reasoning_chain",
    "reasoning_steps",
    "reasoning_depth",
    "reasoning_structure",
    "required_evidence_spans",
    "dual_evidence",
    "persona",
    "quality_tier",
    "qc_pass",
    "qc_metrics",
    "qc_issues",
    # Provenance (added by this script)
    "_source_batch",
]


def load_all():
    """Load all pass queries, dedup by query_id, skip method-C."""
    all_queries = []
    seen = set()
    stats = {"total_raw": 0, "dupes": 0, "method_c_skipped": 0}

    for rel_path, batch_name in PASS_FILES:
        fpath = ROOT / rel_path
        if not fpath.exists():
            continue
        with open(fpath) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                stats["total_raw"] += 1
                d = json.loads(line)
                qid = d.get("query_id", "")
                # Skip method-C queries
                if "method_c" in qid or "methodc" in qid.lower():
                    stats["method_c_skipped"] += 1
                    continue
                if qid in seen:
                    stats["dupes"] += 1
                    continue
                seen.add(qid)
                d["_source_batch"] = batch_name
                all_queries.append(d)

    return all_queries, stats


def build_delivery_record(raw):
    """Normalise a raw query dict into the delivery schema."""
    out = {}
    for field in DELIVERY_FIELDS:
        if field in raw:
            out[field] = raw[field]
    # Ensure hop_distance is int
    if "hop_distance" in out:
        try:
            out["hop_distance"] = int(out["hop_distance"])
        except (ValueError, TypeError):
            pass
    return out


def compute_stats(records):
    """Compute summary statistics."""
    hop = collections.Counter()
    doc = collections.Counter()
    style = collections.Counter()
    batch = collections.Counter()
    cross_modal = collections.Counter()
    pair_type = collections.Counter()

    for r in records:
        hop[r.get("hop_distance", "?")] += 1
        doc[r.get("doc_id", "unknown")] += 1
        style[r.get("query_style", "unknown")] += 1
        batch[r.get("_source_batch", "unknown")] += 1
        cross_modal[str(r.get("cross_modal", False))] += 1
        pair_type[r.get("pair_type", "unknown")] += 1

    return {
        "total": len(records),
        "unique_docs": len(doc),
        "hop_distribution": dict(sorted(hop.items(), key=lambda x: str(x[0]))),
        "style_distribution": dict(sorted(style.items())),
        "batch_distribution": dict(sorted(batch.items())),
        "cross_modal": dict(sorted(cross_modal.items())),
        "pair_type": dict(sorted(pair_type.items())),
        "top_10_docs": dict(doc.most_common(10)),
    }


def main():
    print("Loading all pass queries...")
    raw_queries, load_stats = load_all()
    print(f"  Raw lines: {load_stats['total_raw']}")
    print(f"  Dupes removed: {load_stats['dupes']}")
    print(f"  Method-C skipped: {load_stats['method_c_skipped']}")
    print(f"  Unique queries: {len(raw_queries)}")

    # Build delivery records
    records = [build_delivery_record(q) for q in raw_queries]

    # Sort by doc_id, then query_id for deterministic order
    records.sort(key=lambda r: (r.get("doc_id", ""), r.get("query_id", "")))

    # Write delivery JSONL
    out_jsonl = QUERY_DIR / f"{OUT_STEM}.jsonl"
    with open(out_jsonl, "w") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"\n✅ Delivery file: {out_jsonl}")
    print(f"   {len(records)} queries")

    # Write stats
    summary = compute_stats(records)
    summary["load_stats"] = load_stats
    summary["generated_at"] = datetime.now().isoformat()
    summary["source_files"] = [p for p, _ in PASS_FILES]

    out_stats = QUERY_DIR / f"{OUT_STEM}_stats.json"
    with open(out_stats, "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"✅ Stats file:    {out_stats}")

    # Print summary table
    print(f"\n{'='*50}")
    print(f"DELIVERY SUMMARY")
    print(f"{'='*50}")
    print(f"Total queries:    {summary['total']}")
    print(f"Unique documents: {summary['unique_docs']}")
    print(f"\nHop distribution:")
    for k, v in sorted(summary["hop_distribution"].items(), key=lambda x: str(x[0])):
        print(f"  hop={k}: {v}")
    print(f"\nStyle distribution:")
    for k, v in summary["style_distribution"].items():
        print(f"  {k}: {v}")
    print(f"\nBatch distribution:")
    for k, v in summary["batch_distribution"].items():
        print(f"  {k}: {v}")
    print(f"\nCross-modal:")
    for k, v in summary["cross_modal"].items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
