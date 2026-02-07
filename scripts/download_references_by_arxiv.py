#!/usr/bin/env python3
"""Download referenced papers' PDFs directly from an input arXiv ID."""

import argparse
from pathlib import Path
import sys
import os

# 核心修正：将项目根目录加入 sys.path，确保能 import src 下的模块
# 假设脚本位于 scripts/ 下，根目录就是两级父目录
current_file = Path(__file__).resolve()
project_root = current_file.parent.parent
sys.path.insert(0, str(project_root))

from src.parsers.reference_pdf_collector import ReferencePDFCollector
from src.utils.file_utils import ensure_dir

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download all obtainable reference PDFs from an arXiv paper"
    )
    parser.add_argument("--arxiv-id", required=True, help="Seed arXiv ID, e.g., 2401.12345")
    
    # 修改：默认路径适配你的 data/raw_pdfs 结构
    parser.add_argument(
        "--output",
        "-o",
        type=Path,
        default=project_root / "data" / "raw_pdfs", 
        help="Output directory for downloaded PDFs (default: data/raw_pdfs)",
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
    parser.add_argument(
        "--arxiv-only",
        action="store_true",
        help="Only download papers with arXiv IDs, use arXiv ID as filename",
    )

    args = parser.parse_args()

    print(f"Target Output Directory: {args.output}")
    ensure_dir(args.output)

    collector = ReferencePDFCollector(output_dir=args.output, api_key=args.api_key)
    try:
        records = collector.collect_from_arxiv(
            arxiv_id=args.arxiv_id,
            max_references=args.max_references,
            min_citations=args.min_citations,
            arxiv_only=args.arxiv_only,
        )
    except ValueError as exc:
        print(f"Error: {exc}")
        sys.exit(1)

    # 报告保存到 output 目录下
    report_path = args.output / f"{args.arxiv_id}_references_report.json"
    collector.save_report(report_path=report_path, source_arxiv_id=args.arxiv_id, records=records)

    downloaded = sum(1 for r in records if r.download_success)
    print("=" * 60)
    print(f"Done. Downloaded {downloaded}/{len(records)} references")
    print(f"PDF directory: {args.output}")
    print(f"Report: {report_path}")

if __name__ == "__main__":
    main()