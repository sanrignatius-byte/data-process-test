#!/usr/bin/env python3
"""Batch download referenced papers from seed surveys' LaTeX sources.

Extracts arXiv IDs from .bbl/.bib/.tex files of seed papers, then downloads
both PDFs and LaTeX sources in parallel for all unique referenced papers.

Usage:
    # Dry run (just count what would be downloaded):
    python scripts/batch_download_references.py --dry-run

    # Full download (PDF + LaTeX):
    python scripts/batch_download_references.py

    # Limit to N papers:
    python scripts/batch_download_references.py --limit 500

    # Resume after interruption (skips already downloaded):
    python scripts/batch_download_references.py
"""

from __future__ import annotations

import argparse
import gzip
import io
import json
import os
import re
import sys
import tarfile
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import requests

# ---------------------------------------------------------------------------
# Project paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

LATEX_SOURCES_DIR = PROJECT_ROOT / "data" / "00_raw" / "latex_sources" / "extracted"
PDF_OUTPUT_DIR = PROJECT_ROOT / "data" / "00_raw" / "pdfs_batch2"
LATEX_OUTPUT_DIR = PROJECT_ROOT / "data" / "00_raw" / "latex_sources_batch2"
EXISTING_MINERU_DIR = PROJECT_ROOT / "data" / "00_raw" / "mineru_output"
MANIFEST_PATH = PROJECT_ROOT / "data" / "00_raw" / "reference_manifest_batch2.json"
ID_LIST_PATH = PROJECT_ROOT / "data" / "00_raw" / "arxiv_ids_batch2.txt"

ARXIV_PDF_URL = "https://arxiv.org/pdf/{arxiv_id}.pdf"
ARXIV_EPRINT_URL = "https://arxiv.org/e-print/{arxiv_id}"
ARXIV_DELAY = 3.5  # polite delay between arXiv requests (seconds)
USER_AGENT = "m4-ref-collector/1.0 (research)"


# ---------------------------------------------------------------------------
# Step 1: Extract arXiv IDs from seed papers' LaTeX
# ---------------------------------------------------------------------------
ARXIV_ID_PATTERNS = [
    re.compile(r'arXiv[:\s]+(\d{4}\.\d{4,5})', re.I),
    re.compile(r'arxiv\.org/abs/(\d{4}\.\d{4,5})', re.I),
    re.compile(r'arxiv\.org/pdf/(\d{4}\.\d{4,5})', re.I),
    re.compile(r'\b(\d{4}\.\d{5})\b'),  # bare 5-digit arXiv IDs (2023+ format)
]


def extract_arxiv_ids_from_latex(latex_dir: Path) -> Dict[str, Set[str]]:
    """Extract arXiv IDs from .bbl/.bib/.tex files in each subdirectory.
    
    Returns: {seed_arxiv_id: {referenced_arxiv_id, ...}}
    """
    results: Dict[str, Set[str]] = {}
    
    if not latex_dir.is_dir():
        print(f"  Warning: LaTeX directory not found: {latex_dir}")
        return results

    for seed_dir in sorted(latex_dir.iterdir()):
        if not seed_dir.is_dir():
            continue

        ids_found: Set[str] = set()
        for ext in ["*.bbl", "*.bib", "*.tex"]:
            for f in seed_dir.rglob(ext):
                try:
                    text = f.read_text(errors="ignore")
                    for pat in ARXIV_ID_PATTERNS:
                        for m in pat.finditer(text):
                            ids_found.add(m.group(1))
                except Exception:
                    pass

        if ids_found:
            results[seed_dir.name] = ids_found

    return results


def collect_unique_ids(
    seed_refs: Dict[str, Set[str]],
    exclude_existing: bool = True,
) -> List[str]:
    """Collect unique arXiv IDs, excluding seeds and already-processed papers."""
    all_ids: Set[str] = set()
    for refs in seed_refs.values():
        all_ids |= refs

    # Remove seed IDs themselves
    seed_ids = set(seed_refs.keys())
    all_ids -= seed_ids

    # Remove already-processed papers (batch 1 mineru_output)
    if exclude_existing and EXISTING_MINERU_DIR.is_dir():
        existing = {d.name for d in EXISTING_MINERU_DIR.iterdir() if d.is_dir()}
        all_ids -= existing

    return sorted(all_ids)


# ---------------------------------------------------------------------------
# Step 2: Download PDF
# ---------------------------------------------------------------------------

def download_pdf(
    arxiv_id: str,
    output_dir: Path,
    session: requests.Session,
) -> Tuple[bool, str]:
    """Download PDF from arXiv. Returns (success, message)."""
    safe_name = arxiv_id.replace("/", "_")
    pdf_path = output_dir / f"{safe_name}.pdf"

    if pdf_path.exists() and pdf_path.stat().st_size > 1000:
        return True, "already exists"

    url = ARXIV_PDF_URL.format(arxiv_id=arxiv_id)
    
    for attempt in range(3):
        try:
            resp = session.get(url, timeout=60, stream=True)
            if resp.status_code == 429:
                wait = 15 * (attempt + 1)
                time.sleep(wait)
                continue
            if resp.status_code == 404:
                return False, "404 not found"
            resp.raise_for_status()

            # Verify it's a PDF
            first_chunk = b""
            chunks = []
            for chunk in resp.iter_content(chunk_size=8192):
                chunks.append(chunk)
                if not first_chunk:
                    first_chunk = chunk
                if len(b"".join(chunks)) > 50 * 1024 * 1024:  # 50MB limit
                    return False, "file too large"

            content = b"".join(chunks)
            if b"%PDF" not in content[:1024]:
                return False, "not a PDF"

            pdf_path.parent.mkdir(parents=True, exist_ok=True)
            with open(pdf_path, "wb") as f:
                f.write(content)

            size_mb = len(content) / (1024 * 1024)
            return True, f"ok ({size_mb:.1f}MB)"

        except requests.exceptions.RequestException as e:
            if attempt < 2:
                time.sleep(5)
            else:
                return False, f"error: {e}"

    return False, "max retries exceeded"


# ---------------------------------------------------------------------------
# Step 3: Download LaTeX source
# ---------------------------------------------------------------------------

def download_latex(
    arxiv_id: str,
    output_dir: Path,
    session: requests.Session,
) -> Tuple[bool, str]:
    """Download and extract LaTeX source from arXiv e-print endpoint."""
    safe_name = arxiv_id.replace("/", "_")
    extract_dir = output_dir / "extracted" / safe_name

    if extract_dir.is_dir() and any(extract_dir.rglob("*.tex")):
        return True, "already extracted"

    url = ARXIV_EPRINT_URL.format(arxiv_id=arxiv_id)
    archive_dir = output_dir / "archives"
    archive_dir.mkdir(parents=True, exist_ok=True)
    archive_path = archive_dir / f"{safe_name}.tar.gz"

    for attempt in range(3):
        try:
            resp = session.get(url, timeout=60)
            if resp.status_code == 429:
                wait = 15 * (attempt + 1)
                time.sleep(wait)
                continue
            if resp.status_code == 404:
                return False, "404 not found"
            resp.raise_for_status()

            content = resp.content
            if not content or len(content) < 100:
                return False, "empty response"

            # Try to extract as tar.gz
            extract_dir.mkdir(parents=True, exist_ok=True)

            try:
                # Try tar.gz first
                with tarfile.open(fileobj=io.BytesIO(content), mode="r:gz") as tar:
                    # Security: filter paths
                    members = [m for m in tar.getmembers()
                               if not m.name.startswith("/") and ".." not in m.name]
                    tar.extractall(path=extract_dir, members=members)
                return True, f"tar.gz ({len(members)} files)"
            except tarfile.TarError:
                pass

            try:
                # Try single gzipped file
                decompressed = gzip.decompress(content)
                tex_path = extract_dir / f"{safe_name}.tex"
                tex_path.write_bytes(decompressed)
                return True, "single gzip"
            except gzip.BadGzipFile:
                pass

            # Maybe it's a raw PDF (no source available)
            if content[:4] == b"%PDF":
                # Clean up
                if extract_dir.exists():
                    import shutil
                    shutil.rmtree(extract_dir)
                return False, "source is PDF (no LaTeX)"

            # Unknown format
            raw_path = extract_dir / f"{safe_name}.raw"
            raw_path.write_bytes(content)
            return False, "unknown format"

        except requests.exceptions.RequestException as e:
            if attempt < 2:
                time.sleep(5)
            else:
                return False, f"error: {e}"

    return False, "max retries exceeded"


# ---------------------------------------------------------------------------
# Step 4: Parallel download orchestrator
# ---------------------------------------------------------------------------

def download_one_paper(
    arxiv_id: str,
    pdf_dir: Path,
    latex_dir: Path,
    download_latex_flag: bool,
) -> Dict[str, Any]:
    """Download PDF + optionally LaTeX for one paper. Called from thread pool."""
    session = requests.Session()
    session.headers.update({"User-Agent": USER_AGENT})

    result: Dict[str, Any] = {"arxiv_id": arxiv_id}

    # Download PDF
    time.sleep(ARXIV_DELAY)
    pdf_ok, pdf_msg = download_pdf(arxiv_id, pdf_dir, session)
    result["pdf_ok"] = pdf_ok
    result["pdf_msg"] = pdf_msg

    # Download LaTeX
    if download_latex_flag:
        time.sleep(ARXIV_DELAY)
        latex_ok, latex_msg = download_latex(arxiv_id, latex_dir, session)
        result["latex_ok"] = latex_ok
        result["latex_msg"] = latex_msg
    else:
        result["latex_ok"] = None
        result["latex_msg"] = "skipped"

    return result


def batch_download(
    arxiv_ids: List[str],
    pdf_dir: Path,
    latex_dir: Path,
    download_latex_flag: bool = True,
    workers: int = 2,
    progress_interval: int = 20,
) -> List[Dict[str, Any]]:
    """Download PDFs + LaTeX for a list of arXiv IDs.
    
    Uses sequential download with polite delays (arXiv rate limiting).
    Workers > 1 can be used but arXiv may block aggressive parallelism.
    """
    pdf_dir.mkdir(parents=True, exist_ok=True)
    if download_latex_flag:
        latex_dir.mkdir(parents=True, exist_ok=True)
        (latex_dir / "extracted").mkdir(parents=True, exist_ok=True)
        (latex_dir / "archives").mkdir(parents=True, exist_ok=True)

    results: List[Dict[str, Any]] = []
    session = requests.Session()
    session.headers.update({"User-Agent": USER_AGENT})

    pdf_ok_count = 0
    latex_ok_count = 0
    start_time = time.time()

    for i, arxiv_id in enumerate(arxiv_ids):
        # Progress report
        if i > 0 and i % progress_interval == 0:
            elapsed = time.time() - start_time
            rate = i / elapsed * 3600 if elapsed > 0 else 0
            print(f"  [{i}/{len(arxiv_ids)}] PDF: {pdf_ok_count} ok | "
                  f"LaTeX: {latex_ok_count} ok | "
                  f"rate: {rate:.0f}/hr | "
                  f"elapsed: {elapsed/60:.1f}min")

        # Download PDF
        time.sleep(ARXIV_DELAY)
        pdf_ok, pdf_msg = download_pdf(arxiv_id, pdf_dir, session)
        
        # Download LaTeX
        latex_ok, latex_msg = False, "skipped"
        if download_latex_flag:
            time.sleep(ARXIV_DELAY)
            latex_ok, latex_msg = download_latex(arxiv_id, latex_dir, session)

        result = {
            "arxiv_id": arxiv_id,
            "pdf_ok": pdf_ok,
            "pdf_msg": pdf_msg,
            "latex_ok": latex_ok,
            "latex_msg": latex_msg,
        }
        results.append(result)

        if pdf_ok:
            pdf_ok_count += 1
        if latex_ok:
            latex_ok_count += 1

        # Verbose for first 5
        if i < 5:
            print(f"  {arxiv_id}: pdf={pdf_msg}, latex={latex_msg}")

    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Batch download referenced papers from seed surveys' LaTeX"
    )
    parser.add_argument(
        "--latex-dir", type=Path, default=LATEX_SOURCES_DIR,
        help="Directory with extracted seed LaTeX sources"
    )
    parser.add_argument(
        "--pdf-dir", type=Path, default=PDF_OUTPUT_DIR,
        help="Output directory for PDFs"
    )
    parser.add_argument(
        "--latex-output", type=Path, default=LATEX_OUTPUT_DIR,
        help="Output directory for LaTeX sources"
    )
    parser.add_argument(
        "--limit", type=int, default=None,
        help="Limit number of papers to download"
    )
    parser.add_argument(
        "--skip-latex", action="store_true",
        help="Skip LaTeX source download (PDF only)"
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Just extract IDs and print stats, no download"
    )
    parser.add_argument(
        "--save-ids", action="store_true",
        help="Save extracted arXiv IDs to text file"
    )
    parser.add_argument(
        "--offset", type=int, default=0,
        help="Start from this index (for resume/splitting)"
    )
    parser.add_argument(
        "--id-list-output", type=Path, default=ID_LIST_PATH,
        help="Where to save extracted arXiv IDs when --save-ids or --dry-run is set",
    )
    parser.add_argument(
        "--manifest-output", type=Path, default=MANIFEST_PATH,
        help="Where to write the download manifest",
    )

    args = parser.parse_args()

    # ── Step 1: Extract arXiv IDs from seed LaTeX ──
    print("=" * 60)
    print("Step 1: Extracting arXiv IDs from seed papers' LaTeX")
    print("=" * 60)
    
    seed_refs = extract_arxiv_ids_from_latex(args.latex_dir)
    print(f"\nSeed papers with references: {len(seed_refs)}")
    for seed_id, refs in sorted(seed_refs.items()):
        print(f"  {seed_id}: {len(refs)} arXiv refs")

    all_ids = collect_unique_ids(seed_refs)
    print(f"\nTotal unique new arXiv IDs: {len(all_ids)}")

    # Apply offset + limit
    if args.offset > 0:
        all_ids = all_ids[args.offset:]
        print(f"After offset {args.offset}: {len(all_ids)} remaining")

    if args.limit:
        all_ids = all_ids[:args.limit]
        print(f"After limit: {len(all_ids)} to download")

    # Save ID list
    if args.save_ids or args.dry_run:
        args.id_list_output.parent.mkdir(parents=True, exist_ok=True)
        with open(args.id_list_output, "w") as f:
            for aid in all_ids:
                f.write(f"{aid}\n")
        print(f"ID list saved: {args.id_list_output} ({len(all_ids)} IDs)")

    if args.dry_run:
        # Show sample
        print(f"\nFirst 20 IDs:")
        for aid in all_ids[:20]:
            print(f"  {aid}")
        print(f"\n[DRY RUN] No downloads performed.")
        return

    # ── Step 2: Batch download ──
    print(f"\n{'='*60}")
    print(f"Step 2: Downloading {len(all_ids)} papers (PDF + LaTeX)")
    print(f"  PDF dir:   {args.pdf_dir}")
    print(f"  LaTeX dir: {args.latex_output}")
    print(f"  Estimated time: {len(all_ids) * ARXIV_DELAY * 2 / 60:.0f} min")
    print(f"{'='*60}\n")

    results = batch_download(
        arxiv_ids=all_ids,
        pdf_dir=args.pdf_dir,
        latex_dir=args.latex_output,
        download_latex_flag=not args.skip_latex,
    )

    # ── Step 3: Save manifest + report ──
    pdf_ok = sum(1 for r in results if r["pdf_ok"])
    latex_ok = sum(1 for r in results if r.get("latex_ok"))

    manifest = {
        "total": len(results),
        "pdf_downloaded": pdf_ok,
        "latex_downloaded": latex_ok,
        "seed_papers": list(seed_refs.keys()),
        "results": results,
    }

    args.manifest_output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.manifest_output, "w") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)

    print(f"\n{'='*60}")
    print(f"Done!")
    print(f"  PDFs:  {pdf_ok}/{len(results)} downloaded")
    print(f"  LaTeX: {latex_ok}/{len(results)} downloaded")
    print(f"  Manifest: {args.manifest_output}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
