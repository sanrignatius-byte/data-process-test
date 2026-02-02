#!/usr/bin/env python3
"""
Script to parse PDFs only using MinerU.

Usage:
    python scripts/parse_only.py --input ./data/raw_pdfs --output ./data/mineru_output
"""

import argparse
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.parsers.mineru_parser import MinerUParser
from src.utils.file_utils import get_pdf_files
from tqdm import tqdm


def main():
    parser = argparse.ArgumentParser(description="Parse PDFs using MinerU")

    parser.add_argument(
        "--input",
        type=str,
        default="./data/raw_pdfs",
        help="Input directory containing PDFs"
    )

    parser.add_argument(
        "--output",
        type=str,
        default="./data/mineru_output",
        help="Output directory for parsed results"
    )

    parser.add_argument(
        "--workers",
        type=int,
        default=4,
        help="Number of parallel workers (should match GPU count)"
    )

    parser.add_argument(
        "--backend",
        type=str,
        default="auto",
        choices=["auto", "pipeline", "hybrid", "vlm"],
        help="MinerU parsing backend"
    )

    parser.add_argument(
        "--devices",
        nargs="+",
        default=["cuda:0", "cuda:1", "cuda:2", "cuda:3"],
        help="CUDA devices to use"
    )

    parser.add_argument(
        "--timeout",
        type=int,
        default=300,
        help="Timeout per document in seconds"
    )

    args = parser.parse_args()

    # Get PDF files
    pdf_paths = get_pdf_files(args.input)
    print(f"Found {len(pdf_paths)} PDFs to parse")

    if not pdf_paths:
        print("No PDFs found!")
        return

    # Initialize parser
    mineru_parser = MinerUParser(
        output_dir=args.output,
        backend=args.backend,
        devices=args.devices,
        num_workers=args.workers,
        timeout=args.timeout
    )

    # Progress tracking
    success_count = 0
    fail_count = 0

    def progress_callback(completed, total, result):
        nonlocal success_count, fail_count
        if result.success:
            success_count += 1
        else:
            fail_count += 1
            print(f"Failed: {result.doc_id} - {result.error_message}")

    # Parse
    results = mineru_parser.parse_batch(
        [str(p) for p in pdf_paths],
        progress_callback=progress_callback
    )

    print(f"\n{'='*60}")
    print(f"Parsing complete!")
    print(f"Success: {success_count}/{len(pdf_paths)}")
    print(f"Failed: {fail_count}/{len(pdf_paths)}")
    print(f"Output: {args.output}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
