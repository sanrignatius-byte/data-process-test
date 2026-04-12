#!/usr/bin/env python3
"""Search arXiv for survey papers and download PDFs.

Workflow:
  1. Query arXiv API for survey papers on specified topics
  2. Filter to papers with "survey" indicators
  3. Download PDFs from arXiv
  4. Optionally download LaTeX sources (reuses download_latex_sources.py)
  5. Save metadata manifest for downstream MinerU parsing

Usage:
    # Search and download (dry-run first):
    python scripts/search_and_download_papers.py --dry-run

    # Full run:
    python scripts/search_and_download_papers.py

    # Custom topics:
    python scripts/search_and_download_papers.py --topics "vision transformer survey" "multimodal embedding"

    # Just search, no download:
    python scripts/search_and_download_papers.py --search-only
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any, Dict, List, Optional, Set
from urllib.parse import quote

import requests

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
# arXiv API (free, no key needed)
ARXIV_API_BASE = "https://export.arxiv.org/api/query"
ARXIV_DELAY = 3.0  # polite delay for arXiv API

ARXIV_PDF_URL = "https://arxiv.org/pdf/{arxiv_id}.pdf"

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_PDF_DIR = PROJECT_ROOT / "data" / "00_raw" / "pdfs_batch2"
DEFAULT_MANIFEST = PROJECT_ROOT / "data" / "00_raw" / "paper_manifest_batch2.json"
EXISTING_MINERU_DIR = PROJECT_ROOT / "data" / "00_raw" / "mineru_output"

# Default search topics - arXiv API uses different query syntax
DEFAULT_TOPICS = [
    "multimodal large language model survey",
    "vision language model survey",
    "multimodal embedding survey",
    "multimodal encoder survey",
    "visual question answering survey",
    "image text retrieval survey",
    "multimodal representation learning survey",
    "vision transformer survey",
    "CLIP contrastive learning survey",
    "document understanding multimodal survey",
]

# How many results per topic query
RESULTS_PER_TOPIC = 30

# Target total papers
TARGET_PAPERS = 25

# arXiv API XML namespace
ARXIV_NS = {
    "atom": "http://www.w3.org/2005/Atom",
    "arxiv": "http://arxiv.org/schemas/atom",
}


# ---------------------------------------------------------------------------
# arXiv API search
# ---------------------------------------------------------------------------

def _extract_arxiv_id_from_url(url: str) -> str:
    """Extract arXiv ID from entry URL like http://arxiv.org/abs/2306.13549v2"""
    # e.g. http://arxiv.org/abs/2306.13549v2  ->  2306.13549
    m = re.search(r"/abs/(.+?)(?:v\d+)?$", url)
    return m.group(1) if m else url


def search_arxiv(
    query: str,
    max_results: int = RESULTS_PER_TOPIC,
    session: Optional[requests.Session] = None,
) -> List[Dict[str, Any]]:
    """Search arXiv API for papers matching query.

    Uses the arXiv search API: https://arxiv.org/help/api/user-manual
    Returns list of dicts with keys: arxiv_id, title, abstract, authors, year, categories
    """
    s = session or requests.Session()

    # Build query: split words, search in title+abstract
    # arXiv query: (ti:word1 AND ti:word2) OR (abs:word1 AND abs:word2)
    words = query.strip().split()
    ti_parts = " AND ".join(f"ti:{w}" for w in words)
    abs_parts = " AND ".join(f"abs:{w}" for w in words)
    search_query = f"({ti_parts}) OR ({abs_parts})"

    params = {
        "search_query": search_query,
        "start": 0,
        "max_results": max_results,
        "sortBy": "relevance",
        "sortOrder": "descending",
    }

    for attempt in range(3):
        try:
            resp = s.get(ARXIV_API_BASE, params=params, timeout=30)
            resp.raise_for_status()
            break
        except requests.exceptions.RequestException as e:
            print(f"  Error querying arXiv: {e}")
            if attempt < 2:
                time.sleep(5 * (attempt + 1))
            else:
                return []

    # Parse XML response
    root = ET.fromstring(resp.text)
    entries = root.findall("atom:entry", ARXIV_NS)

    papers = []
    for entry in entries:
        title_el = entry.find("atom:title", ARXIV_NS)
        abstract_el = entry.find("atom:summary", ARXIV_NS)
        published_el = entry.find("atom:published", ARXIV_NS)
        id_el = entry.find("atom:id", ARXIV_NS)

        if id_el is None:
            continue

        arxiv_id = _extract_arxiv_id_from_url(id_el.text.strip())
        title = title_el.text.strip().replace("\n", " ") if title_el is not None else ""
        abstract = abstract_el.text.strip().replace("\n", " ") if abstract_el is not None else ""

        # Get year from published date
        year = None
        if published_el is not None:
            m = re.match(r"(\d{4})", published_el.text)
            if m:
                year = int(m.group(1))

        # Get categories
        categories = []
        for cat in entry.findall("atom:category", ARXIV_NS):
            term = cat.get("term", "")
            if term:
                categories.append(term)

        # Get authors
        authors = []
        for author_el in entry.findall("atom:author", ARXIV_NS):
            name_el = author_el.find("atom:name", ARXIV_NS)
            if name_el is not None:
                authors.append(name_el.text.strip())

        papers.append({
            "arxiv_id": arxiv_id,
            "title": title,
            "abstract": abstract,
            "year": year,
            "categories": categories,
            "authors": authors,
            "_topic_query": query,
        })

    return papers


def is_survey_paper(paper: Dict[str, Any]) -> bool:
    """Heuristic: is this paper a survey/review?"""
    title = (paper.get("title") or "").lower()
    abstract = (paper.get("abstract") or "").lower()

    # Check title keywords
    survey_keywords = [
        "survey", "review", "overview", "tutorial", "comprehensive",
        "systematic", "state of the art", "state-of-the-art",
        "benchmark", "comparison", "landscape", "progress",
    ]
    for kw in survey_keywords:
        if kw in title:
            return True

    # Check abstract (weaker signal)
    abstract_keywords = [
        "we survey", "this survey", "this review",
        "comprehensive overview", "systematic review",
        "we review", "extensive survey", "this paper surveys",
    ]
    for kw in abstract_keywords:
        if kw in abstract:
            return True

    return False


def get_existing_arxiv_ids() -> Set[str]:
    """Get arXiv IDs already in mineru_output (batch 1)."""
    if not EXISTING_MINERU_DIR.is_dir():
        return set()
    return {d.name for d in EXISTING_MINERU_DIR.iterdir() if d.is_dir()}


# ---------------------------------------------------------------------------
# Search pipeline
# ---------------------------------------------------------------------------

def search_all_topics(
    topics: List[str],
    target: int = TARGET_PAPERS,
) -> List[Dict[str, Any]]:
    """Search across all topics via arXiv API, deduplicate, filter, rank."""

    existing_ids = get_existing_arxiv_ids()
    print(f"Existing papers in batch 1: {len(existing_ids)}")

    session = requests.Session()
    all_papers: Dict[str, Dict[str, Any]] = {}  # arxiv_id -> paper

    for i, topic in enumerate(topics):
        print(f"\n[{i+1}/{len(topics)}] Searching: {topic}")
        time.sleep(ARXIV_DELAY)  # polite delay between queries

        results = search_arxiv(topic, session=session)
        print(f"  Got {len(results)} results")

        new_count = 0
        for paper in results:
            arxiv_id = paper["arxiv_id"]
            if arxiv_id in existing_ids:
                continue
            if arxiv_id in all_papers:
                continue

            all_papers[arxiv_id] = paper
            new_count += 1

        print(f"  {new_count} new unique candidates")

    print(f"\n{'='*60}")
    print(f"Total unique candidates: {len(all_papers)}")

    # Rank: prefer surveys, then by recency (year)
    candidates = list(all_papers.values())
    for p in candidates:
        p["_is_survey"] = is_survey_paper(p)

    # Filter: require CS category (cs.CV, cs.CL, cs.AI, cs.LG, cs.MM, cs.IR)
    cs_prefixes = {"cs.CV", "cs.CL", "cs.AI", "cs.LG", "cs.MM", "cs.IR", "cs.HC"}
    filtered = []
    for p in candidates:
        cats = set(p.get("categories", []))
        if cats & cs_prefixes:
            filtered.append(p)
    print(f"After CS-category filter: {len(filtered)} (from {len(candidates)})")
    candidates = filtered

    # Sort: surveys first, then by year (recent first)
    candidates.sort(
        key=lambda p: (p["_is_survey"], p.get("year", 0) or 0),
        reverse=True,
    )

    # Take top N
    selected = candidates[:target]

    survey_count = sum(1 for p in selected if p["_is_survey"])
    print(f"Selected {len(selected)} papers ({survey_count} surveys)")

    return selected


# ---------------------------------------------------------------------------
# Download PDFs
# ---------------------------------------------------------------------------

def download_pdf(
    arxiv_id: str,
    output_dir: Path,
    session: requests.Session,
    delay: float = ARXIV_DELAY,
) -> Optional[Path]:
    """Download PDF from arXiv. Returns path or None."""
    safe_name = arxiv_id.replace("/", "_")
    pdf_path = output_dir / f"{safe_name}.pdf"

    if pdf_path.exists() and pdf_path.stat().st_size > 1000:
        print(f"  [skip] {arxiv_id} already downloaded")
        return pdf_path

    url = ARXIV_PDF_URL.format(arxiv_id=arxiv_id)

    for attempt in range(3):
        try:
            time.sleep(delay)
            resp = session.get(
                url,
                timeout=60,
                headers={"User-Agent": "m4-paper-collector/1.0 (research)"},
                stream=True,
            )
            if resp.status_code == 429:
                wait = 10 * (attempt + 1)
                print(f"  Rate limited, waiting {wait}s ...")
                time.sleep(wait)
                continue
            resp.raise_for_status()

            # Write PDF
            pdf_path.parent.mkdir(parents=True, exist_ok=True)
            with open(pdf_path, "wb") as f:
                for chunk in resp.iter_content(chunk_size=8192):
                    f.write(chunk)

            size_mb = pdf_path.stat().st_size / (1024 * 1024)
            print(f"  [ok] {arxiv_id} ({size_mb:.1f} MB)")
            return pdf_path

        except requests.exceptions.RequestException as e:
            print(f"  [err] {arxiv_id} attempt {attempt+1}: {e}")
            if attempt < 2:
                time.sleep(5)

    return None


# ---------------------------------------------------------------------------
# Manifest
# ---------------------------------------------------------------------------

def save_manifest(
    papers: List[Dict[str, Any]],
    pdf_dir: Path,
    manifest_path: Path,
) -> None:
    """Save paper manifest JSON for downstream processing."""
    entries = []
    for p in papers:
        arxiv_id = p["arxiv_id"]
        safe_name = arxiv_id.replace("/", "_")
        pdf_path = pdf_dir / f"{safe_name}.pdf"

        entries.append({
            "arxiv_id": arxiv_id,
            "title": p.get("title", ""),
            "year": p.get("year"),
            "is_survey": p.get("_is_survey", False),
            "topic_query": p.get("_topic_query", ""),
            "abstract": (p.get("abstract") or "")[:500],
            "categories": p.get("categories", []),
            "authors": p.get("authors", [])[:5],  # first 5 authors
            "pdf_downloaded": pdf_path.exists(),
            "pdf_path": str(pdf_path) if pdf_path.exists() else None,
        })

    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(entries, f, indent=2, ensure_ascii=False)
    print(f"\nManifest saved: {manifest_path} ({len(entries)} entries)")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Search S2 for survey papers and download PDFs from arXiv"
    )
    parser.add_argument(
        "--topics", nargs="+", default=DEFAULT_TOPICS,
        help="Search topic queries (default: multimodal LLM + embedding topics)"
    )
    parser.add_argument(
        "--target", type=int, default=TARGET_PAPERS,
        help=f"Target number of papers to select (default: {TARGET_PAPERS})"
    )
    parser.add_argument(
        "--pdf-dir", type=Path, default=DEFAULT_PDF_DIR,
        help=f"Directory to save PDFs (default: {DEFAULT_PDF_DIR})"
    )
    parser.add_argument(
        "--manifest", type=Path, default=DEFAULT_MANIFEST,
        help=f"Output manifest path (default: {DEFAULT_MANIFEST})"
    )
    parser.add_argument(
        "--search-only", action="store_true",
        help="Only search, don't download PDFs"
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Search and print results, no download or save"
    )
    parser.add_argument(
        "--no-verify", action="store_true",
        help="Disable SSL verification"
    )

    args = parser.parse_args()

    # ── Step 1: Search ──
    print("=" * 60)
    print("Step 1: Searching arXiv")
    print("=" * 60)
    papers = search_all_topics(
        topics=args.topics,
        target=args.target,
    )

    if not papers:
        print("No papers found. Exiting.")
        return

    # Print summary
    print(f"\n{'='*60}")
    print("Selected papers:")
    print(f"{'='*60}")
    for i, p in enumerate(papers):
        arxiv_id = p["arxiv_id"]
        title = p.get("title", "???")[:80]
        year = p.get("year", "?")
        survey_tag = " [SURVEY]" if p.get("_is_survey") else ""
        print(f"  {i+1:2d}. [{arxiv_id}] ({year}){survey_tag}")
        print(f"      {title}")

    if args.dry_run:
        print("\n[DRY RUN] No files written.")
        return

    # ── Step 2: Download PDFs ──
    if not args.search_only:
        print(f"\n{'='*60}")
        print("Step 2: Downloading PDFs from arXiv")
        print(f"{'='*60}")

        args.pdf_dir.mkdir(parents=True, exist_ok=True)
        session = requests.Session()
        if args.no_verify:
            session.verify = False

        success = 0
        for p in papers:
            arxiv_id = p["arxiv_id"]
            result = download_pdf(arxiv_id, args.pdf_dir, session)
            if result:
                success += 1

        print(f"\nDownloaded {success}/{len(papers)} PDFs")

    # ── Step 3: Save manifest ──
    save_manifest(papers, args.pdf_dir, args.manifest)

    # ── Next steps ──
    print(f"\n{'='*60}")
    print("Next steps:")
    print(f"{'='*60}")
    print(f"  1. Parse PDFs with MinerU:")
    print(f"     sbatch slurm_scripts/03_parse_mineru_batch2.sh")
    print(f"  2. Download LaTeX sources:")
    print(f"     python scripts/download_latex_sources.py \\")
    print(f"       --id-file <(jq -r '.[].arxiv_id' {args.manifest})")
    print(f"  3. Build graphs + generate queries")


if __name__ == "__main__":
    main()
