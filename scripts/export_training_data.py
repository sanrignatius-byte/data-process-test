#!/usr/bin/env python3
"""Export normalised queries into model-ready training data.

End-to-end pipeline::

    raw L1/L2/L3 JSONL
        → normalize_queries.py  (if not already done)
        → export_training_data.py
        → data/contrastive_data/v<N>/
              train.jsonl
              val.jsonl
              test.jsonl
              manifest.json

Usage
-----
::

    # Quick run with defaults
    python scripts/export_training_data.py

    # Custom paths / config
    python scripts/export_training_data.py \\
        --queries  data/queries_normalized.jsonl \\
        --elements data/multimodal_elements.json \\
        --output   data/contrastive_data/v2 \\
        --train-ratio 0.85 \\
        --num-negatives 5 \\
        --negative-strategy in_doc_swap
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

# ── path bootstrap ────────────────────────────────────────────────────────────
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

from src.export.dataset_builder import DatasetBuilder, DatasetConfig  # noqa: E402

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Export normalised queries → train/val/test triplets",
    )
    ap.add_argument(
        "--queries", type=Path,
        default=_PROJECT_ROOT / "data" / "queries_normalized.jsonl",
        help="Path to normalised queries (output of normalize_queries.py)",
    )
    ap.add_argument(
        "--elements", type=Path,
        default=_PROJECT_ROOT / "data" / "multimodal_elements.json",
        help="Path to multimodal_elements.json (element text/metadata)",
    )
    ap.add_argument(
        "--output", "-o", type=Path,
        default=_PROJECT_ROOT / "data" / "contrastive_data" / "v1",
        help="Output directory for train/val/test JSONL + manifest",
    )
    ap.add_argument("--train-ratio", type=float, default=0.9)
    ap.add_argument("--val-ratio", type=float, default=0.05)
    ap.add_argument("--num-negatives", type=int, default=3)
    ap.add_argument(
        "--negative-strategy", type=str, default="random",
        choices=["random", "in_doc_swap", "same_type_hard",
                 "modal_mixed", "graph_aware"],
    )
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument(
        "--config", type=Path, default=None,
        help="Optional YAML config (reads negative_sampling + output sections)",
    )
    args = ap.parse_args()

    # Build config — CLI args take precedence over YAML
    neg_config: dict = {}
    if args.config and args.config.exists():
        import yaml

        with open(args.config, encoding="utf-8") as fh:
            yaml_cfg = yaml.safe_load(fh) or {}
        neg_section = yaml_cfg.get("negative_sampling", {})
        neg_config = dict(neg_section)
        output_section = yaml_cfg.get("output", {})
        if "train_ratio" in output_section:
            args.train_ratio = output_section["train_ratio"]

    cfg = DatasetConfig(
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        num_negatives=args.num_negatives,
        negative_strategy=args.negative_strategy,
        negative_config=neg_config,
        seed=args.seed,
    )

    builder = DatasetBuilder(cfg)
    stats = builder.build(
        queries_path=args.queries,
        elements_path=args.elements,
        output_dir=args.output,
    )

    logger.info("Export complete.  Stats: %s", json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
