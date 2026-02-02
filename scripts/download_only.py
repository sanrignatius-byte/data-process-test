#!/usr/bin/env python3
"""
Script to download PDFs only from arXiv.

Usage:
    python scripts/download_only.py --count 200 --categories cs.CL cs.CV
"""

import argparse
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.parsers.pdf_downloader import download_papers_for_training


def main():
    parser = argparse.ArgumentParser(description="Download PDFs from arXiv")

    parser.add_argument(
        "--count",
        type=int,
        default=200,
        help="Number of papers to download"
    )

    parser.add_argument(
        "--output",
        type=str,
        default="./data/raw_pdfs",
        help="Output directory for PDFs"
    )

    parser.add_argument(
        "--categories",
        nargs="+",
        default=["cs.CL", "cs.CV", "cs.LG", "cs.AI"],
        help="arXiv categories to search"
    )

    args = parser.parse_args()

    print(f"Downloading {args.count} papers from categories: {args.categories}")
    print(f"Output directory: {args.output}")

    papers = download_papers_for_training(
        output_dir=args.output,
        target_count=args.count,
        categories=args.categories
    )

    print(f"\nSuccessfully downloaded {len(papers)} papers")


if __name__ == "__main__":
    main()
