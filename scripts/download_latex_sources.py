#!/usr/bin/env python3
"""Download and extract LaTeX source packages from arXiv.

arXiv provides source files at https://arxiv.org/e-print/{arxiv_id}.
The response can be:
  1) A .tar.gz archive (most common) — contains .tex, .bbl, images, etc.
  2) A single gzipped file (just one .tex compressed)
  3) A PDF (no source available)

This script handles all three cases, batch-extracts them, and locates
the main .tex file for each paper.

Usage:
    # From a list of known arXiv IDs (one per line in a text file):
    python scripts/download_latex_sources.py --id-file data/arxiv_ids.txt

    # From the existing figure_text_pairs.json (auto-extract unique doc IDs):
    python scripts/download_latex_sources.py --from-pairs data/figure_text_pairs.json

    # Just a few IDs on the command line:
    python scripts/download_latex_sources.py --ids 1908.09635 1412.3756 2005.07293

    # Skip download, only re-extract / locate main .tex:
    python scripts/download_latex_sources.py --from-pairs data/figure_text_pairs.json --extract-only
"""

from __future__ import annotations

import argparse
import gzip
import io
import json
import sys
import tarfile
import time
from pathlib import Path
from typing import Dict, List, Optional, Set

import urllib3
import requests

# ---------------------------------------------------------------------------
# Project imports
# ---------------------------------------------------------------------------
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

from src.utils.file_utils import ensure_dir, safe_json_dump  # noqa: E402

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
ARXIV_EPRINT_URL = "https://arxiv.org/e-print/{arxiv_id}"
DEFAULT_OUTPUT_DIR = _PROJECT_ROOT / "data" / "latex_sources"
DEFAULT_DELAY = 3.0  # arXiv requests polite delay (seconds)
USER_AGENT = "m4-latex-collector/1.0 (research; mailto:research@example.com)"


# ---------------------------------------------------------------------------
# Helpers: gathering arXiv IDs from various sources
# ---------------------------------------------------------------------------

def ids_from_text_file(path: Path) -> List[str]:
    """Read one arXiv ID per line."""
    ids: List[str] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#"):
                ids.append(line)
    return ids


def ids_from_figure_text_pairs(path: Path) -> List[str]:
    """Extract unique doc_ids from figure_text_pairs.json."""
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    # Support both formats:
    # - dict keyed by doc_id: {doc_id: [pairs...], ...}
    # - flat list of pair dicts: [{doc_id: ..., ...}, ...]
    if isinstance(data, dict):
        return sorted(k for k in data if k)
    seen: Set[str] = set()
    ids: List[str] = []
    for pair in data:
        doc_id = pair.get("doc_id", "")
        if doc_id and doc_id not in seen:
            seen.add(doc_id)
            ids.append(doc_id)
    return sorted(ids)


def ids_from_mineru_output(path: Path) -> List[str]:
    """Extract doc IDs from mineru_output directory names."""
    if not path.is_dir():
        return []
    return sorted(d.name for d in path.iterdir() if d.is_dir())


# ---------------------------------------------------------------------------
# Download
# ---------------------------------------------------------------------------

def download_source(
    arxiv_id: str,
    output_dir: Path,
    session: requests.Session,
    delay: float = DEFAULT_DELAY,
) -> Optional[Path]:
    """Download e-print archive for one paper. Returns path to saved file or None."""
    safe_name = arxiv_id.replace("/", "_")
    archive_path = output_dir / f"{safe_name}.tar.gz"

    # Already downloaded?
    if archive_path.exists() and archive_path.stat().st_size > 0:
        return archive_path

    url = ARXIV_EPRINT_URL.format(arxiv_id=arxiv_id)

    for attempt in range(4):
        try:
            resp = session.get(url, timeout=60, stream=True)

            if resp.status_code == 200:
                content = resp.content
                # Check if it's actually a PDF (no source available)
                if content[:5] == b"%PDF-":
                    print(f"  [{arxiv_id}] No source — PDF only")
                    return None

                with open(archive_path, "wb") as f:
                    f.write(content)
                return archive_path

            if resp.status_code == 429:
                wait = (attempt + 1) * 5
                print(f"  [{arxiv_id}] Rate-limited (429), waiting {wait}s …")
                time.sleep(wait)
                continue

            print(f"  [{arxiv_id}] HTTP {resp.status_code}")
            return None

        except requests.RequestException as exc:
            if attempt < 3:
                wait = 2 ** (attempt + 1)
                print(f"  [{arxiv_id}] Network error: {exc}, retry in {wait}s …")
                time.sleep(wait)
            else:
                print(f"  [{arxiv_id}] Failed after 4 attempts: {exc}")
                return None

    return None


# ---------------------------------------------------------------------------
# Extraction
# ---------------------------------------------------------------------------

def extract_source(archive_path: Path, extract_dir: Path) -> bool:
    """Extract a downloaded archive into extract_dir.

    Handles:
      - tar.gz archives (most common)
      - plain gzipped single files (.tex)
    Returns True on success.
    """
    if extract_dir.exists() and any(extract_dir.iterdir()):
        return True  # already extracted

    ensure_dir(extract_dir)

    data = archive_path.read_bytes()

    # Try as tar.gz first
    try:
        with tarfile.open(fileobj=io.BytesIO(data), mode="r:gz") as tar:
            # Security: skip absolute paths and path traversal
            safe_members = []
            for member in tar.getmembers():
                if member.name.startswith("/") or ".." in member.name:
                    continue
                safe_members.append(member)
            tar.extractall(path=extract_dir, members=safe_members)
        return True
    except (tarfile.TarError, EOFError):
        pass

    # Try as plain gzip (single .tex file)
    try:
        decompressed = gzip.decompress(data)
        # Heuristic: if it looks like LaTeX, save as main.tex
        if b"\\documentclass" in decompressed or b"\\begin{document}" in decompressed:
            (extract_dir / "main.tex").write_bytes(decompressed)
            return True
        # Might be a plain .tex without standard preamble
        if b"\\" in decompressed[:500]:
            (extract_dir / "main.tex").write_bytes(decompressed)
            return True
    except (gzip.BadGzipFile, OSError):
        pass

    # Try as raw tar (no gzip)
    try:
        with tarfile.open(fileobj=io.BytesIO(data), mode="r:") as tar:
            safe_members = [
                m for m in tar.getmembers()
                if not m.name.startswith("/") and ".." not in m.name
            ]
            tar.extractall(path=extract_dir, members=safe_members)
        return True
    except tarfile.TarError:
        pass

    print(f"  [WARN] Could not extract: {archive_path.name}")
    return False


# ---------------------------------------------------------------------------
# Locate main .tex file
# ---------------------------------------------------------------------------

def find_main_tex(extract_dir: Path) -> Optional[Path]:
    """Heuristic to find the main .tex file in an extracted source directory.

    Priority:
      1. File containing \\documentclass
      2. File named main.tex / paper.tex / ms.tex
      3. Largest .tex file
    """
    tex_files = list(extract_dir.rglob("*.tex"))
    if not tex_files:
        return None

    # 1) Look for \documentclass
    for f in tex_files:
        try:
            content = f.read_text(errors="replace")
            if "\\documentclass" in content:
                return f
        except Exception:
            continue

    # 2) Common names
    for name in ("main.tex", "paper.tex", "ms.tex", "article.tex"):
        candidates = [f for f in tex_files if f.name.lower() == name]
        if candidates:
            return candidates[0]

    # 3) Largest file
    return max(tex_files, key=lambda f: f.stat().st_size)


def find_bbl_file(extract_dir: Path) -> Optional[Path]:
    """Find .bbl bibliography file."""
    bbl_files = list(extract_dir.rglob("*.bbl"))
    if bbl_files:
        return bbl_files[0]
    return None


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def process_papers(
    arxiv_ids: List[str],
    output_dir: Path,
    delay: float = DEFAULT_DELAY,
    extract_only: bool = False,
    verify_ssl: bool = True,
) -> Dict:
    """Download, extract, and index LaTeX sources for a list of papers."""
    archive_dir = output_dir / "archives"
    source_dir = output_dir / "extracted"
    ensure_dir(archive_dir)
    ensure_dir(source_dir)

    session = requests.Session()
    session.headers.update({"User-Agent": USER_AGENT})

    if not verify_ssl:
        session.verify = False
        urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
        print("[SSL] Certificate verification disabled by --no-verify flag.")
    else:
        # Auto-detect SSL issues with a probe request
        try:
            session.get("https://arxiv.org/", timeout=10)
        except requests.exceptions.SSLError:
            print("[SSL] Certificate verification failed (proxy/firewall detected).")
            print("[SSL] Falling back to verify=False automatically.")
            session.verify = False
            urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
        except requests.RequestException:
            pass  # Non-SSL network issues handled later per-paper

    results = []
    stats = {
        "total": len(arxiv_ids),
        "downloaded": 0,
        "extracted": 0,
        "has_tex": 0,
        "has_bbl": 0,
        "pdf_only": 0,
        "failed": 0,
    }

    for i, arxiv_id in enumerate(arxiv_ids):
        safe_name = arxiv_id.replace("/", "_")
        paper_extract_dir = source_dir / safe_name
        entry = {
            "arxiv_id": arxiv_id,
            "archive_path": None,
            "extract_dir": None,
            "main_tex": None,
            "bbl_file": None,
            "tex_count": 0,
            "status": "pending",
        }

        print(f"[{i + 1}/{len(arxiv_ids)}] {arxiv_id}", end=" … ")

        # Download
        if not extract_only:
            archive_path = download_source(
                arxiv_id, archive_dir, session, delay=delay
            )
            if archive_path:
                entry["archive_path"] = str(archive_path)
                stats["downloaded"] += 1
            else:
                entry["status"] = "no_source"
                stats["pdf_only"] += 1
                print("skip (no source)")
                results.append(entry)
                if not extract_only:
                    time.sleep(delay)
                continue
        else:
            archive_path = archive_dir / f"{safe_name}.tar.gz"
            if not archive_path.exists():
                entry["status"] = "no_archive"
                stats["failed"] += 1
                print("skip (no archive)")
                results.append(entry)
                continue
            entry["archive_path"] = str(archive_path)

        # Extract
        ok = extract_source(archive_path, paper_extract_dir)
        if ok:
            entry["extract_dir"] = str(paper_extract_dir)
            stats["extracted"] += 1

            # Find main .tex
            main_tex = find_main_tex(paper_extract_dir)
            if main_tex:
                entry["main_tex"] = str(main_tex)
                stats["has_tex"] += 1

            # Find .bbl
            bbl = find_bbl_file(paper_extract_dir)
            if bbl:
                entry["bbl_file"] = str(bbl)
                stats["has_bbl"] += 1

            # Count .tex files
            entry["tex_count"] = len(list(paper_extract_dir.rglob("*.tex")))
            entry["status"] = "ok"
            print(f"ok ({entry['tex_count']} .tex" +
                  (", has .bbl" if bbl else "") + ")")
        else:
            entry["status"] = "extract_failed"
            stats["failed"] += 1
            print("extract failed")

        results.append(entry)

        # Polite delay between downloads
        if not extract_only:
            time.sleep(delay)

    # Save report
    report = {"stats": stats, "papers": results}
    report_path = output_dir / "latex_source_report.json"
    safe_json_dump(report, report_path)

    print("\n" + "=" * 60)
    print(f"Total:     {stats['total']}")
    print(f"Downloaded:{stats['downloaded']}")
    print(f"Extracted: {stats['extracted']}")
    print(f"Has .tex:  {stats['has_tex']}")
    print(f"Has .bbl:  {stats['has_bbl']}")
    print(f"PDF-only:  {stats['pdf_only']}")
    print(f"Failed:    {stats['failed']}")
    print(f"Report:    {report_path}")

    return report


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download and extract LaTeX source from arXiv"
    )

    # Input sources (mutually exclusive)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--ids", nargs="+",
        help="arXiv IDs to download, e.g. 1908.09635 1412.3756"
    )
    group.add_argument(
        "--id-file", type=Path,
        help="Text file with one arXiv ID per line"
    )
    group.add_argument(
        "--from-pairs", type=Path,
        help="Extract IDs from figure_text_pairs.json"
    )
    group.add_argument(
        "--from-mineru", type=Path,
        help="Extract IDs from mineru_output directory"
    )

    parser.add_argument(
        "--output", "-o", type=Path, default=DEFAULT_OUTPUT_DIR,
        help=f"Output directory (default: {DEFAULT_OUTPUT_DIR})"
    )
    parser.add_argument(
        "--delay", type=float, default=DEFAULT_DELAY,
        help=f"Delay between arXiv requests in seconds (default: {DEFAULT_DELAY})"
    )
    parser.add_argument(
        "--extract-only", action="store_true",
        help="Skip download, only extract already-downloaded archives"
    )
    parser.add_argument(
        "--no-verify", action="store_true",
        help="Disable SSL certificate verification (use behind corporate/university proxies)"
    )

    args = parser.parse_args()

    # Gather IDs
    if args.ids:
        arxiv_ids = args.ids
    elif args.id_file:
        arxiv_ids = ids_from_text_file(args.id_file)
    elif args.from_pairs:
        arxiv_ids = ids_from_figure_text_pairs(args.from_pairs)
    elif args.from_mineru:
        arxiv_ids = ids_from_mineru_output(args.from_mineru)
    else:
        parser.error("No input source specified")
        return

    if not arxiv_ids:
        print("No arXiv IDs found. Nothing to do.")
        return

    print(f"Processing {len(arxiv_ids)} papers …")
    print(f"Output: {args.output}")
    print(f"Delay: {args.delay}s between requests")
    print()

    process_papers(
        arxiv_ids=arxiv_ids,
        output_dir=args.output,
        delay=args.delay,
        extract_only=args.extract_only,
        verify_ssl=not args.no_verify,
    )


if __name__ == "__main__":
    main()
