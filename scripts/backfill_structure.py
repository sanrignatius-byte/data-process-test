#!/usr/bin/env python3
"""
Backfill structure.json and formula files for existing MinerU outputs.

Usage:
    python scripts/backfill_structure.py --input ./data/mineru_output
    python scripts/backfill_structure.py --input ./data/mineru_output --pdf-dir ./data/raw_pdfs --remaining-dir ./data/raw_pdfs_remaining
"""

import argparse
import sys
from pathlib import Path
from typing import Optional

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.parsers.mineru_parser import MinerUParser, ParsedDocument


def find_pdf_path(doc_dir: Path, doc_id: str) -> Optional[str]:
    """Try to locate the original PDF within a MinerU output directory."""
    candidates = list(doc_dir.rglob(f"{doc_id}_origin.pdf"))
    if candidates:
        return str(candidates[0])
    # Fallback: any PDF in the subtree (excluding layout if possible)
    for p in doc_dir.rglob("*.pdf"):
        if p.name.endswith("_layout.pdf"):
            continue
        return str(p)
    return None


def backfill_structures(
    input_dir: Path,
    force: bool = False,
    limit: Optional[int] = None
) -> int:
    """Backfill structure.json for docs under input_dir."""
    parser = MinerUParser(
        output_dir=str(input_dir),
        backend="auto",
        devices=["cuda:0"],
        num_workers=1,
        timeout=1,
        verify_installation=False
    )

    updated = 0
    processed = 0

    for doc_dir in sorted(input_dir.iterdir()):
        if not doc_dir.is_dir():
            continue

        structure_path = doc_dir / "structure.json"
        if structure_path.exists() and not force:
            continue

        doc_id = doc_dir.name
        elements = parser._extract_elements_from_output(doc_dir, doc_id)
        total_pages = parser._count_pages(doc_dir)

        if not elements:
            print(f"[skip] {doc_id}: no elements extracted")
            continue

        parsed = ParsedDocument(
            doc_id=doc_id,
            pdf_path=find_pdf_path(doc_dir, doc_id) or "",
            output_path=str(doc_dir),
            total_pages=total_pages,
            elements=elements,
            parse_time=0,
            success=True
        )

        parser.save_structure(parsed)
        updated += 1
        processed += 1

        if limit and processed >= limit:
            break

    return updated


def create_remaining_symlinks(
    pdf_dir: Path,
    output_dir: Path,
    remaining_dir: Path
) -> int:
    """Create symlinks for PDFs that have no structure.json output."""
    remaining_dir.mkdir(parents=True, exist_ok=True)

    done = {d.name for d in output_dir.iterdir() if (d / "structure.json").exists()}
    count = 0
    for pdf in pdf_dir.glob("*.pdf"):
        if pdf.stem in done:
            continue
        target = remaining_dir / pdf.name
        if not target.exists():
            target.symlink_to(pdf.resolve())
        count += 1

    return count


def main() -> None:
    parser = argparse.ArgumentParser(description="Backfill MinerU structure outputs")
    parser.add_argument(
        "--input",
        type=str,
        default="./data/mineru_output",
        help="MinerU output directory"
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing structure.json"
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit number of documents to process"
    )
    parser.add_argument(
        "--pdf-dir",
        type=str,
        default=None,
        help="Raw PDF directory for creating remaining symlinks"
    )
    parser.add_argument(
        "--remaining-dir",
        type=str,
        default="./data/raw_pdfs_remaining",
        help="Directory to place remaining PDF symlinks"
    )

    args = parser.parse_args()

    input_dir = Path(args.input)
    updated = backfill_structures(
        input_dir=input_dir,
        force=args.force,
        limit=args.limit
    )
    print(f"Backfilled structure.json for {updated} documents.")

    if args.pdf_dir:
        remaining = create_remaining_symlinks(
            pdf_dir=Path(args.pdf_dir),
            output_dir=input_dir,
            remaining_dir=Path(args.remaining_dir)
        )
        print(f"Created {remaining} remaining PDF symlinks in {args.remaining_dir}.")


if __name__ == "__main__":
    main()
