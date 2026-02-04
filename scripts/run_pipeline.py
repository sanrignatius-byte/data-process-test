#!/usr/bin/env python3
"""
Entry script for running the multimodal contrastive learning data pipeline.

Usage:
    # Full pipeline (download + parse + generate)
    python scripts/run_pipeline.py --target-docs 200

    # Skip download, use existing PDFs
    python scripts/run_pipeline.py --skip-download --target-docs 200

    # Skip download and parse, regenerate queries only
    python scripts/run_pipeline.py --skip-download --skip-parse

    # Custom config
    python scripts/run_pipeline.py --config configs/custom_config.yaml
"""

import argparse
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.pipeline import ContrastiveDataPipeline, run_pipeline


def parse_args():
    parser = argparse.ArgumentParser(
        description="Multimodal Contrastive Learning Data Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Full pipeline for 200 documents
    python scripts/run_pipeline.py --target-docs 200

    # Use existing PDFs (skip download)
    python scripts/run_pipeline.py --skip-download

    # Only regenerate queries (skip download and parse)
    python scripts/run_pipeline.py --skip-download --skip-parse

    # Use custom configuration
    python scripts/run_pipeline.py --config configs/my_config.yaml
        """
    )

    parser.add_argument(
        "--config",
        type=str,
        default="configs/config.yaml",
        help="Path to configuration file"
    )

    parser.add_argument(
        "--target-docs",
        type=int,
        default=200,
        help="Target number of documents to process"
    )

    parser.add_argument(
        "--skip-download",
        action="store_true",
        help="Skip PDF download stage (use existing PDFs)"
    )

    parser.add_argument(
        "--skip-parse",
        action="store_true",
        help="Skip document parsing stage (use existing parsed data)"
    )

    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="Disable checkpoint resume (start fresh)"
    )

    parser.add_argument(
        "--output-name",
        type=str,
        default="contrastive_dataset",
        help="Output dataset name"
    )

    return parser.parse_args()


def main():
    args = parse_args()

    print("=" * 60)
    print("Multimodal Contrastive Learning Data Pipeline")
    print("=" * 60)
    print(f"Config: {args.config}")
    print(f"Target documents: {args.target_docs}")
    print(f"Skip download: {args.skip_download}")
    print(f"Skip parse: {args.skip_parse}")
    print(f"Resume enabled: {not args.no_resume}")
    print("=" * 60)

    # Run pipeline
    stats = run_pipeline(
        config_path=args.config,
        target_docs=args.target_docs,
        skip_download=args.skip_download,
        skip_parse=args.skip_parse,
        resume=not args.no_resume,
        output_name=args.output_name
    )

    # Print summary
    print("\n" + "=" * 60)
    print("Pipeline Completed!")
    print("=" * 60)
    print(f"Total PDFs processed: {stats.parsed_pdfs}/{stats.total_pdfs}")
    print(f"Total passages extracted: {stats.total_passages}")
    print(f"Total queries generated: {stats.total_queries}")
    print(f"Total triplets created: {stats.total_triplets}")

    if stats.modal_distribution:
        print("\nModal Distribution:")
        for modal, count in stats.modal_distribution.items():
            print(f"  {modal}: {count}")

    if stats.end_time and stats.start_time:
        duration = (stats.end_time - stats.start_time).total_seconds()
        print(f"\nTotal time: {duration:.1f} seconds")

    print("=" * 60)


if __name__ == "__main__":
    main()
