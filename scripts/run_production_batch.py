#!/usr/bin/env python3
"""Production batch: generate queries from unused enriched candidates.

Workflow:
  1. Load existing L2/L3 queries → get used pair_ids
  2. Filter enriched hub candidates to unused pairs only
  3. Write filtered candidates to temp file
  4. Call generate_multihop_l1_queries.py on filtered candidates
  5. Merge output with existing data
  6. Run package_m2_levels.py to rebuild level files

Usage:
    python scripts/run_production_batch.py \
        --enriched-candidates data/05_eval/m2/hub_candidates_enriched_full.json \
        --batch-name production_v1 \
        --provider company --model gpt-5.4

    # Enrich remaining candidates first, then generate
    python scripts/run_production_batch.py \
        --enrich-first \
        --raw-candidates data/01_graphs/latex_hub_multihop_candidates.json \
        --batch-name production_v2 \
        --provider company --model gpt-5.4
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Set

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
M2_DIR = DATA_DIR / "05_eval" / "m2"


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows = []
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def get_used_pair_ids() -> Set[str]:
    """Collect all pair_ids already used in L2 + L3 queries."""
    used = set()

    # All existing query files
    query_files = [
        DATA_DIR / "03_queries" / "l1_dual_evidence_queries_v3_pass.jsonl",
        DATA_DIR / "03_queries" / "l1_dual_evidence_queries_v4_4_run1_pass.jsonl",
        M2_DIR / "level2_dual_evidence.jsonl",
        M2_DIR / "level3_reasoning_chain.jsonl",
        M2_DIR / "l2_new_batch.jsonl",
        M2_DIR / "l3_new_batch.jsonl",
        M2_DIR / "l3_reasoning_chain_queries.jsonl",
    ]

    for qf in query_files:
        for q in load_jsonl(qf):
            pid = q.get("pair_id", "")
            if pid:
                used.add(pid)

    return used


def filter_candidates(
    candidates_path: Path,
    used_pair_ids: Set[str],
    output_path: Path,
    hop_filter: int | None = None,
) -> int:
    """Filter enriched candidates to unused pairs only.

    Returns: number of remaining candidates.
    """
    obj = json.loads(candidates_path.read_text(encoding="utf-8"))
    pairs = obj.get("pairs", [])

    filtered = []
    for p in pairs:
        pid = p.get("pair_id", "")
        if pid in used_pair_ids:
            continue
        # Check mapping: both elements must have element_a_id and element_b_id
        if not p.get("element_a_id") or not p.get("element_b_id"):
            continue
        # Hop filter
        if hop_filter is not None and p.get("hop_distance") != hop_filter:
            continue
        filtered.append(p)

    # Preserve metadata
    output_obj = dict(obj)
    output_obj["pairs"] = filtered

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(output_obj, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"  Filtered: {len(pairs)} → {len(filtered)} unused candidates → {output_path}")
    return len(filtered)


def run_enrichment(
    raw_candidates: Path,
    existing_enriched: Path,
    output: Path,
    elements: Path,
    latex_graph: Path,
    enriched_elements: Path | None = None,
) -> None:
    """Run enrich_hub_candidates.py to enrich raw candidates."""
    cmd = [
        sys.executable, str(PROJECT_ROOT / "scripts" / "enrich_hub_candidates.py"),
        "--hub-candidates", str(raw_candidates),
        "--elements", str(elements),
        "--latex-graph", str(latex_graph),
        "--output", str(output),
    ]
    if enriched_elements and enriched_elements.exists():
        cmd.extend(["--enriched-elements", str(enriched_elements)])
    hubs_path = DATA_DIR / "01_graphs" / "latex_graph_hubs.json"
    if hubs_path.exists():
        cmd.extend(["--hubs", str(hubs_path)])

    print(f"\n  Running enrichment: {' '.join(cmd[-6:])}")
    result = subprocess.run(cmd, cwd=str(PROJECT_ROOT))
    if result.returncode != 0:
        print(f"  ERROR: Enrichment failed with code {result.returncode}")
        sys.exit(1)


def run_generation(
    candidates_path: Path,
    output_path: Path,
    provider: str,
    model: str,
    limit: int = 0,
    delay: float = 0.5,
    query_style: str = "mixed",
    use_persona: bool = False,
) -> None:
    """Run generate_multihop_l1_queries.py on candidates."""
    cmd = [
        sys.executable, str(PROJECT_ROOT / "scripts" / "generate_multihop_l1_queries.py"),
        "--candidates", str(candidates_path),
        "--output", str(output_path),
        "--pass-only",
        "--provider", provider,
        "--model", model,
        "--query-style", query_style,
        "--delay", str(delay),
    ]
    if use_persona:
        cmd.append("--use-persona")
    if limit > 0:
        cmd.extend(["--limit", str(limit)])

    print(f"\n  Running generation: provider={provider}, model={model}")
    print(f"  Output → {output_path}")
    result = subprocess.run(cmd, cwd=str(PROJECT_ROOT))
    if result.returncode != 0:
        print(f"  ERROR: Generation failed with code {result.returncode}")
        sys.exit(1)


def merge_pass_files(
    new_batch: Path,
    existing_level_file: Path,
    level: int,
    level_label: str,
) -> int:
    """Merge new pass queries into existing level file (dedup by pair_id)."""
    new_pass_path = new_batch.with_name(new_batch.stem + "_pass.jsonl")
    if not new_pass_path.exists():
        print(f"  WARN: {new_pass_path} not found")
        return 0

    new_queries = load_jsonl(new_pass_path)
    existing = load_jsonl(existing_level_file)

    # Dedup by pair_id + query_id
    seen = set()
    for q in existing:
        qid = q.get("query_id", q.get("pair_id", ""))
        if qid:
            seen.add(qid)

    added = 0
    for q in new_queries:
        qid = q.get("query_id", q.get("pair_id", ""))
        if qid and qid in seen:
            continue
        seen.add(qid)
        q["difficulty_level"] = level
        q["difficulty_label"] = level_label
        existing.append(q)
        added += 1

    # Write back
    existing_level_file.parent.mkdir(parents=True, exist_ok=True)
    with existing_level_file.open("w", encoding="utf-8") as f:
        for q in existing:
            f.write(json.dumps(q, ensure_ascii=False) + "\n")

    print(f"  Merged {added} new queries → {existing_level_file} (total: {len(existing)})")
    return added


def main() -> None:
    ap = argparse.ArgumentParser(description="Production batch query generation")
    ap.add_argument(
        "--enriched-candidates", type=Path,
        default=M2_DIR / "hub_candidates_enriched_full.json",
        help="Enriched hub candidates JSON",
    )
    ap.add_argument(
        "--raw-candidates", type=Path,
        default=DATA_DIR / "01_graphs" / "latex_hub_multihop_candidates.json",
        help="Raw hub candidates (used with --enrich-first)",
    )
    ap.add_argument(
        "--enrich-first", action="store_true",
        help="Run enrichment on raw candidates before generation",
    )
    ap.add_argument(
        "--batch-name", type=str, default="production_v1",
        help="Batch name for output files",
    )
    ap.add_argument(
        "--provider", type=str, default="company",
        help="LLM provider: company / anthropic / openai",
    )
    ap.add_argument(
        "--model", type=str, default="gpt-5.4",
        help="Model name",
    )
    ap.add_argument(
        "--limit", type=int, default=0,
        help="Limit candidates per level (0=all)",
    )
    ap.add_argument(
        "--delay", type=float, default=0.5,
        help="Delay between API calls (seconds)",
    )
    ap.add_argument(
        "--query-style", type=str, default="mixed",
        choices=["academic", "real_user", "mixed"],
        help="Query style",
    )
    ap.add_argument(
        "--use-persona", action="store_true", default=False,
        help=(
            "Inject PersonaHub persona prefix into prompts (76 diverse reader "
            "personas, deterministic by pair_id hash). Strongly recommended "
            "for production runs to improve query diversity."
        ),
    )
    ap.add_argument(
        "--dry-run", action="store_true",
        help="Show plan without executing",
    )
    ap.add_argument(
        "--skip-package", action="store_true",
        help="Skip package_m2_levels.py at the end",
    )
    ap.add_argument(
        "--enriched-elements", type=Path,
        default=DATA_DIR / "02_enriched" / "multimodal_elements_enriched.json",
        help="Path to MoDora-enriched elements JSON (default: data/02_enriched/multimodal_elements_enriched.json)",
    )
    ap.add_argument(
        "--use-pairing-module", action="store_true",
        help=(
            "Use new src/pairing module (select_intra_doc_pairs.py) instead of "
            "legacy hub_candidates_enriched JSON. Generates fresh pairs directly "
            "from multimodal_elements.json with strict doc-boundary enforcement."
        ),
    )
    ap.add_argument(
        "--pairing-elements", type=Path,
        default=DATA_DIR / "01_graphs" / "multimodal_elements.json",
        help="Path to multimodal_elements.json for --use-pairing-module",
    )
    ap.add_argument(
        "--pairing-strategy", type=str, default="all",
        choices=["direct", "2hop", "section", "chain", "all"],
        help="Pairing strategy when using --use-pairing-module (default: all)",
    )
    ap.add_argument(
        "--pairing-max-per-doc", type=int, default=15,
        help="Max pairs per document when using --use-pairing-module (default: 15)",
    )
    args = ap.parse_args()

    print("=" * 60)
    print(f"Production Batch: {args.batch_name}")
    print("=" * 60)

    # Step 1: Get used pair_ids
    print("\n[1/5] Collecting used pair_ids...")
    used_pair_ids = get_used_pair_ids()
    print(f"  Found {len(used_pair_ids)} already-used pair_ids")

    # Step 2: Optionally enrich raw candidates, or use new pairing module
    enriched_path = args.enriched_candidates
    if args.use_pairing_module:
        print("\n[2/5] Generating pairs via src/pairing module...")
        enriched_path = M2_DIR / f"pairing_pairs_{args.batch_name}.json"
        cmd = [
            sys.executable, str(PROJECT_ROOT / "scripts" / "select_intra_doc_pairs.py"),
            "--elements", str(args.pairing_elements),
            "--output", str(enriched_path),
            "--strategy", args.pairing_strategy,
            "--max-per-doc", str(args.pairing_max_per_doc),
        ]
        print(f"  Running: {' '.join(cmd[-8:])}")
        result = subprocess.run(cmd, cwd=str(PROJECT_ROOT))
        if result.returncode != 0:
            print(f"  ERROR: Pairing module failed with code {result.returncode}")
            sys.exit(1)
    elif args.enrich_first:
        print("\n[2/5] Enriching raw candidates...")
        enriched_path = M2_DIR / f"hub_candidates_enriched_{args.batch_name}.json"
        run_enrichment(
            raw_candidates=args.raw_candidates,
            existing_enriched=args.enriched_candidates,
            output=enriched_path,
            elements=DATA_DIR / "01_graphs" / "multimodal_elements.json",
            latex_graph=DATA_DIR / "01_graphs" / "latex_reference_graph.json",
            enriched_elements=args.enriched_elements,
        )
    else:
        print("\n[2/5] Skipping enrichment (using existing enriched candidates)")

    # Step 3: Filter to unused candidates
    print("\n[3/5] Filtering candidates...")
    filtered_path = M2_DIR / f"filtered_{args.batch_name}.json"
    n_remaining = filter_candidates(
        candidates_path=enriched_path,
        used_pair_ids=used_pair_ids,
        output_path=filtered_path,
    )

    if n_remaining == 0:
        print("\n  No unused candidates remaining. Nothing to generate.")
        return

    # Show plan
    print(f"\n  Plan: generate queries from {n_remaining} unused candidates")
    print(f"  Provider: {args.provider}, Model: {args.model}")
    print(f"  Style: {args.query_style}")

    if args.dry_run:
        print("\n  [DRY RUN] Would generate queries. Exiting.")
        return

    # Step 4: Generate queries
    print(f"\n[4/5] Generating queries...")
    output_path = M2_DIR / f"{args.batch_name}.jsonl"
    run_generation(
        candidates_path=filtered_path,
        output_path=output_path,
        provider=args.provider,
        model=args.model,
        limit=args.limit,
        delay=args.delay,
        query_style=args.query_style,
        use_persona=args.use_persona,
    )

    # Step 5: Merge into level files
    print(f"\n[5/5] Merging results...")
    pass_path = output_path.with_name(output_path.stem + "_pass.jsonl")
    if pass_path.exists():
        pass_queries = load_jsonl(pass_path)
        # Split by difficulty level
        l2_queries = [q for q in pass_queries if q.get("difficulty_level", 2) == 2]
        l3_queries = [q for q in pass_queries if q.get("difficulty_level", 3) == 3]

        # Write batch-specific pass files for traceability
        l2_batch = M2_DIR / f"l2_{args.batch_name}_pass.jsonl"
        l3_batch = M2_DIR / f"l3_{args.batch_name}_pass.jsonl"
        for queries, path in [(l2_queries, l2_batch), (l3_queries, l3_batch)]:
            if queries:
                with path.open("w", encoding="utf-8") as f:
                    for q in queries:
                        f.write(json.dumps(q, ensure_ascii=False) + "\n")
                print(f"  Wrote {len(queries)} → {path.name}")

        print(f"\n  Total pass: {len(pass_queries)} (L2={len(l2_queries)}, L3={len(l3_queries)})")
    else:
        print(f"  WARN: {pass_path} not found — generation may have failed")

    # Repackage
    if not args.skip_package:
        print(f"\n  Running package_m2_levels.py...")
        subprocess.run(
            [sys.executable, str(PROJECT_ROOT / "scripts" / "package_m2_levels.py")],
            cwd=str(PROJECT_ROOT),
        )

    # Summary
    print(f"\n{'='*60}")
    print(f"Production Batch Complete: {args.batch_name}")
    print(f"{'='*60}")
    for level_file in [
        M2_DIR / "level1_single_element.jsonl",
        M2_DIR / "level2_dual_evidence.jsonl",
        M2_DIR / "level3_reasoning_chain.jsonl",
        M2_DIR / "all_levels_combined.jsonl",
    ]:
        if level_file.exists():
            count = sum(1 for _ in level_file.open())
            print(f"  {level_file.name}: {count}")


if __name__ == "__main__":
    main()
