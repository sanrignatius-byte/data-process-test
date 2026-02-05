#!/usr/bin/env python3
"""Download referenced papers' PDFs directly from an input arXiv ID."""

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.parsers.reference_pdf_collector import ReferencePDFCollector
from src.utils.file_utils import ensure_dir


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download all obtainable reference PDFs from an arXiv paper"
    )
    parser.add_argument("--arxiv-id", required=True, help="Seed arXiv ID, e.g., 2401.12345")
    parser.add_argument(
        "--output",
        "-o",
        type=Path,
        default=Path("data/referenced_pdfs"),
        help="Output directory for downloaded PDFs",
    )
    parser.add_argument(
        "--max-references",
        type=int,
        default=None,
        help="Optional cap on number of references to process",
    )
    parser.add_argument(
        "--min-citations",
        type=int,
        default=0,
        help="Filter out low-citation references",
    )
    parser.add_argument(
        "--api-key",
        default=None,
        help="Semantic Scholar API key (optional)",
    )

    args = parser.parse_args()

    ensure_dir(args.output)

    collector = ReferencePDFCollector(output_dir=args.output, api_key=args.api_key)
    try:
        records = collector.collect_from_arxiv(
            arxiv_id=args.arxiv_id,
            max_references=args.max_references,
            min_citations=args.min_citations,
        )
    except ValueError as exc:
        print(f"Error: {exc}")
        print("Tip: check your network/API access, or pass --api-key for higher Semantic Scholar quota.")
        sys.exit(1)

    report_path = args.output / "reference_download_report.json"
    collector.save_report(report_path=report_path, source_arxiv_id=args.arxiv_id, records=records)

    downloaded = sum(1 for r in records if r.download_success)
    print("=" * 60)
    print(f"Done. Downloaded {downloaded}/{len(records)} references")
    print(f"PDF directory: {args.output}")
    print(f"Report: {report_path}")


if __name__ == "__main__":
    main()
