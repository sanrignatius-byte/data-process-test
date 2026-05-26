#!/usr/bin/env python3
"""Apply G11 filters to C18 MinerU cross-document citation predictions."""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import sys
from typing import Any, Dict, Iterator

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.pairing.xdoc_citation_filter import (  # noqa: E402
    CitationFilterConfig,
    filter_edges,
    section_bucket,
)


def read_jsonl(path: Path) -> Iterator[Dict[str, Any]]:
    with path.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def write_jsonl(path: Path, rows: Iterator[Dict[str, Any]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            n += 1
    return n


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Filter inferred cross-doc citation edges into a stable M4 citation backbone.",
    )
    parser.add_argument(
        "--input",
        default="data/04_xdoc_citation/predicted_xdoc_edges_chunks.jsonl",
        help="Input JSONL from infer_xdoc_citation_chunks.py",
    )
    parser.add_argument(
        "--output",
        default="data/04_xdoc_citation/predicted_xdoc_edges_chunks_filtered.jsonl",
        help="Filtered JSONL output path",
    )
    parser.add_argument(
        "--stats",
        default="data/04_xdoc_citation/predicted_xdoc_edges_chunks_filtered_stats.json",
        help="Filtering statistics JSON path",
    )
    parser.add_argument("--min-probability", type=float, default=0.5)
    parser.add_argument("--min-title-match", type=float, default=0.2)
    parser.add_argument("--min-semantic-text-sim", type=float, default=0.75)
    parser.add_argument("--min-semantic-probability", type=float, default=0.95)
    parser.add_argument(
        "--keep-references",
        action="store_true",
        help="Keep references/bibliography chunks. Default drops them for M4 paragraph bridges.",
    )
    parser.add_argument(
        "--keep-semantic-high-conf",
        action="store_true",
        help="Keep edges with no citation/title evidence only if they are very high confidence.",
    )
    args = parser.parse_args()

    input_path = PROJECT_ROOT / args.input
    output_path = PROJECT_ROOT / args.output
    stats_path = PROJECT_ROOT / args.stats

    config = CitationFilterConfig(
        min_probability=args.min_probability,
        min_title_match=args.min_title_match,
        min_semantic_text_sim=args.min_semantic_text_sim,
        min_semantic_probability=args.min_semantic_probability,
        body_only=not args.keep_references,
        keep_semantic_high_conf=args.keep_semantic_high_conf,
    )

    total = 0
    raw_section_buckets: Counter = Counter()

    def counted_rows() -> Iterator[Dict[str, Any]]:
        nonlocal total
        for edge in read_jsonl(input_path):
            total += 1
            raw_section_buckets[section_bucket(edge)] += 1
            yield edge

    kept_rows, kept_by_tier, dropped_by_reason = filter_edges(counted_rows(), config)
    kept = write_jsonl(output_path, kept_rows)

    stats = {
        "input": str(input_path.relative_to(PROJECT_ROOT)),
        "output": str(output_path.relative_to(PROJECT_ROOT)),
        "filter_config": config.__dict__,
        "total_edges": total,
        "kept_edges": kept,
        "dropped_edges": total - kept,
        "kept_ratio": round(kept / total, 6) if total else 0.0,
        "raw_section_buckets": dict(raw_section_buckets),
        "kept_by_tier": dict(kept_by_tier),
        "dropped_by_reason": dict(dropped_by_reason),
    }
    stats_path.parent.mkdir(parents=True, exist_ok=True)
    stats_path.write_text(json.dumps(stats, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"Input edges: {total}")
    print(f"Kept edges: {kept}")
    print(f"Dropped: {total - kept}")
    print(f"Kept by tier: {dict(kept_by_tier)}")
    print(f"Dropped by reason: {dict(dropped_by_reason)}")
    print(f"Wrote {output_path}")
    print(f"Wrote {stats_path}")


if __name__ == "__main__":
    main()
