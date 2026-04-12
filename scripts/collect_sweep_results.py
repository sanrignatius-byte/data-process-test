#!/usr/bin/env python3
"""Collect and merge results from all production sweep outputs.

Usage:
    python scripts/collect_sweep_results.py
    python scripts/collect_sweep_results.py --sweep-dir data/03_queries/sweep_2026-04-12
    python scripts/collect_sweep_results.py --merge-existing   # also include older pass files
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "03_queries"


def load_jsonl(path: Path) -> list[dict]:
    rows = []
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sweep-dir", default="data/03_queries/sweep_2026-04-12")
    ap.add_argument(
        "--merge-existing",
        action="store_true",
        help="Also include pre-existing pass files (l3_rerun2, m2, long_chain)",
    )
    ap.add_argument(
        "--output",
        default="data/03_queries/delivery_pool_merged.jsonl",
        help="Merged pass-only output",
    )
    args = ap.parse_args()

    sweep_dir = Path(args.sweep_dir)
    if not sweep_dir.is_absolute():
        sweep_dir = PROJECT_ROOT / sweep_dir

    # ── Collect sweep pass files ──────────────────────────────────────────
    print("=" * 60)
    print("Production Sweep Results")
    print("=" * 60)

    all_pass: list[dict] = []
    seen_query_texts: set[str] = set()
    seen_pair_ids: set[str] = set()

    # Sweep outputs
    sweep_files = sorted(sweep_dir.glob("*_pass.jsonl")) if sweep_dir.exists() else []
    for pf in sweep_files:
        rows = load_jsonl(pf)
        print(f"  {pf.name}: {len(rows)} pass")
        for r in rows:
            qt = r.get("query", "")
            pid = r.get("pair_id", "")
            if qt not in seen_query_texts:
                all_pass.append(r)
                seen_query_texts.add(qt)
                seen_pair_ids.add(pid)

    print(f"\nSweep total (deduped): {len(all_pass)}")

    # ── Merge existing pass files ─────────────────────────────────────────
    if args.merge_existing:
        existing_files = [
            DATA_DIR / "l3_enriched_v3_rerun2_pass.jsonl",
            DATA_DIR / "l3_enriched_v3_new82_rerun2_pass.jsonl",
            DATA_DIR / "m2_diverse_v1_hub_kb_pass.jsonl",
            DATA_DIR / "long_chain_iterative_pass.jsonl",
        ]
        for ef in existing_files:
            rows = load_jsonl(ef)
            added = 0
            for r in rows:
                qt = r.get("query", "")
                if qt not in seen_query_texts:
                    all_pass.append(r)
                    seen_query_texts.add(qt)
                    seen_pair_ids.add(r.get("pair_id", ""))
                    added += 1
            print(f"  {ef.name}: {len(rows)} total, {added} new after dedup")

    print(f"\nMerged total: {len(all_pass)} unique pass queries")
    print(f"Unique pair_ids: {len(seen_pair_ids)}")

    # ── Stats breakdown ───────────────────────────────────────────────────
    styles = Counter(r.get("query_style", "?") for r in all_pass)
    persona = Counter(
        "yes" if r.get("persona_used") or r.get("has_persona") else "no"
        for r in all_pass
    )
    hops = Counter(r.get("hop_distance", r.get("hops", "?")) for r in all_pass)

    print(f"\nStyles:  {dict(styles)}")
    print(f"Persona: {dict(persona)}")
    print(f"Hops:    {dict(sorted(hops.items()))}")

    # ── Write merged output ───────────────────────────────────────────────
    out_path = Path(args.output)
    if not out_path.is_absolute():
        out_path = PROJECT_ROOT / out_path
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for r in all_pass:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"\nWritten to: {out_path}")
    print(f"Total deliverable queries: {len(all_pass)}")


if __name__ == "__main__":
    main()
