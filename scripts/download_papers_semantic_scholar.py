#!/usr/bin/env python3
"""Download arXiv papers via Semantic Scholar API, expanding from seed papers.

Crawls the citation/reference network of seed papers, filters for arXiv papers,
then downloads their LaTeX sources using the existing download_latex_sources pipeline.

Semantic Scholar API docs: https://api.semanticscholar.org/api-docs/

Usage:
    # Expand from one seed paper (2-hop citation network):
    python scripts/download_papers_semantic_scholar.py \
        --seeds 1908.09635 \
        --output-ids data/ss_arxiv_ids.txt \
        --max-papers 500 \
        --hops 2

    # Use API key for higher rate limits:
    python scripts/download_papers_semantic_scholar.py \
        --seeds 1908.09635 2003.05991 \
        --api-key $SEMANTIC_SCHOLAR_API_KEY \
        --max-papers 1000 \
        --hops 2 \
        --download  # also trigger LaTeX source download
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import deque
from pathlib import Path
from typing import Dict, List, Optional, Set

import requests

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

SS_BASE = "https://api.semanticscholar.org/graph/v1"
DEFAULT_DELAY = 1.0   # seconds between requests (no key)
KEY_DELAY = 0.2       # seconds between requests (with key)
MAX_RETRIES = 4

PAPER_FIELDS = "externalIds,title,year,citationCount,referenceCount"
NEIGHBOR_FIELDS = "externalIds,title,year,citationCount"


def _get(url: str, params: dict, api_key: Optional[str], delay: float) -> Optional[dict]:
    headers = {}
    if api_key:
        headers["x-api-key"] = api_key
    for attempt in range(MAX_RETRIES):
        try:
            r = requests.get(url, params=params, headers=headers, timeout=30)
            if r.status_code == 200:
                time.sleep(delay)
                return r.json()
            if r.status_code == 429:
                wait = 2 ** (attempt + 2)
                print(f"  Rate limited, waiting {wait}s...", flush=True)
                time.sleep(wait)
                continue
            print(f"  HTTP {r.status_code} for {url}", flush=True)
            return None
        except Exception as e:
            wait = 2 ** (attempt + 1)
            print(f"  Request error ({e}), retrying in {wait}s...", flush=True)
            time.sleep(wait)
    return None


def arxiv_id_to_ss_id(arxiv_id: str, api_key: Optional[str], delay: float) -> Optional[str]:
    """Convert arXiv ID to Semantic Scholar paper ID."""
    url = f"{SS_BASE}/paper/arXiv:{arxiv_id}"
    data = _get(url, {"fields": "paperId,externalIds"}, api_key, delay)
    if data and "paperId" in data:
        return data["paperId"]
    return None


def fetch_neighbors(ss_id: str, direction: str, api_key: Optional[str], delay: float, limit: int = 500) -> List[dict]:
    """Fetch citations or references for a paper. direction: 'citations' or 'references'."""
    url = f"{SS_BASE}/paper/{ss_id}/{direction}"
    results = []
    offset = 0
    while True:
        params = {"fields": NEIGHBOR_FIELDS, "limit": min(limit, 1000), "offset": offset}
        data = _get(url, params, api_key, delay)
        if not data:
            break
        batch = data.get("data", [])
        results.extend(batch)
        if not data.get("next") or len(results) >= limit:
            break
        offset = data["next"]
    return results


def extract_arxiv_id(paper: dict) -> Optional[str]:
    """Extract arXiv ID from a Semantic Scholar paper object."""
    ext = (paper.get("externalIds") or {})
    arxiv = ext.get("ArXiv") or ext.get("arxiv")
    if arxiv:
        # Normalize: strip version suffix if present
        return arxiv.split("v")[0] if "v" in arxiv and arxiv.split("v")[-1].isdigit() else arxiv
    return None


def crawl_network(
    seed_arxiv_ids: List[str],
    api_key: Optional[str],
    max_papers: int,
    hops: int,
    delay: float,
    min_citations: int = 5,
    directions: List[str] = ("citations", "references"),
) -> Set[str]:
    """BFS expansion of citation/reference network from seed papers."""
    collected: Set[str] = set(seed_arxiv_ids)
    frontier: deque[tuple[str, int]] = deque()  # (arxiv_id, depth)

    print(f"Resolving {len(seed_arxiv_ids)} seed(s) to SS IDs...", flush=True)
    seed_ss_ids: Dict[str, str] = {}  # arxiv_id → ss_id
    for arxiv_id in seed_arxiv_ids:
        ss_id = arxiv_id_to_ss_id(arxiv_id, api_key, delay)
        if ss_id:
            seed_ss_ids[arxiv_id] = ss_id
            frontier.append((arxiv_id, 0))
            print(f"  {arxiv_id} → {ss_id}", flush=True)
        else:
            print(f"  WARNING: could not resolve {arxiv_id}", flush=True)

    # Cache arxiv_id → ss_id to avoid redundant lookups
    ss_id_cache: Dict[str, str] = dict(seed_ss_ids)
    visited_ss: Set[str] = set(seed_ss_ids.values())

    while frontier and len(collected) < max_papers:
        arxiv_id, depth = frontier.popleft()
        if depth >= hops:
            continue

        ss_id = ss_id_cache.get(arxiv_id)
        if not ss_id:
            continue

        for direction in directions:
            print(f"  [{depth+1}/{hops}] Fetching {direction} of {arxiv_id}...", flush=True)
            neighbors = fetch_neighbors(ss_id, direction, api_key, delay)
            added = 0
            for item in neighbors:
                paper = item.get("citedPaper") or item.get("citingPaper") or item
                neighbor_arxiv = extract_arxiv_id(paper)
                if not neighbor_arxiv:
                    continue
                if paper.get("citationCount", 0) < min_citations and depth == hops - 1:
                    continue  # Skip low-citation papers at final hop
                if neighbor_arxiv not in collected:
                    collected.add(neighbor_arxiv)
                    added += 1
                    # Queue for next hop if not at max depth
                    if depth + 1 < hops:
                        neighbor_ss = (paper.get("paperId") or
                                       ss_id_cache.get(neighbor_arxiv))
                        if neighbor_ss and neighbor_ss not in visited_ss:
                            visited_ss.add(neighbor_ss)
                            ss_id_cache[neighbor_arxiv] = neighbor_ss
                            frontier.append((neighbor_arxiv, depth + 1))
                if len(collected) >= max_papers:
                    break
            print(f"    Added {added} new papers (total: {len(collected)})", flush=True)

    return collected


def topic_search(
    query: str,
    api_key: Optional[str],
    delay: float,
    max_papers: int = 200,
    year_range: Optional[str] = None,
) -> Set[str]:
    """Search SS by topic and return arXiv IDs."""
    url = f"{SS_BASE}/paper/search"
    results: Set[str] = set()
    offset = 0
    while len(results) < max_papers:
        params: dict = {
            "query": query,
            "fields": NEIGHBOR_FIELDS,
            "limit": min(100, max_papers - len(results)),
            "offset": offset,
        }
        if year_range:
            params["year"] = year_range
        data = _get(url, params, api_key, delay)
        if not data:
            break
        batch = data.get("data", [])
        for paper in batch:
            arxiv_id = extract_arxiv_id(paper)
            if arxiv_id:
                results.add(arxiv_id)
        if not data.get("next") or not batch:
            break
        offset = data["next"]
    return results


def load_existing_ids(paths: List[Path]) -> Set[str]:
    """Load already-downloaded arXiv IDs from various sources."""
    ids: Set[str] = set()
    for path in paths:
        if not path.exists():
            continue
        if path.suffix == ".json":
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
                if isinstance(data, dict):
                    ids.update(data.keys())
                elif isinstance(data, list):
                    for item in data:
                        if isinstance(item, str):
                            ids.add(item)
                        elif isinstance(item, dict):
                            doc_id = item.get("doc_id") or item.get("arxiv_id")
                            if doc_id:
                                ids.add(str(doc_id))
            except Exception:
                pass
        else:
            for line in path.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if line and not line.startswith("#"):
                    ids.add(line)
    return ids


def main():
    parser = argparse.ArgumentParser(description="Download arXiv papers via Semantic Scholar API")
    parser.add_argument("--seeds", nargs="+", default=["1908.09635"],
                        help="Seed arXiv IDs to expand from (default: 1908.09635)")
    parser.add_argument("--api-key", default=None,
                        help="Semantic Scholar API key (or set SS_API_KEY env var)")
    parser.add_argument("--max-papers", type=int, default=500,
                        help="Maximum number of arXiv papers to collect (default: 500)")
    parser.add_argument("--hops", type=int, default=2,
                        help="Citation network expansion depth (default: 2)")
    parser.add_argument("--directions", nargs="+", default=["citations", "references"],
                        choices=["citations", "references"],
                        help="Which edges to follow (default: both)")
    parser.add_argument("--min-citations", type=int, default=5,
                        help="Min citation count for papers at final hop (default: 5)")
    parser.add_argument("--topic-query", default=None,
                        help="Optional: also search by topic (e.g. 'fairness machine learning')")
    parser.add_argument("--topic-max", type=int, default=200,
                        help="Max papers from topic search (default: 200)")
    parser.add_argument("--year-range", default="2018-2026",
                        help="Year range for topic search (default: 2018-2026)")
    parser.add_argument("--output-ids", type=Path,
                        default=_PROJECT_ROOT / "data" / "ss_arxiv_ids.txt",
                        help="Output file for collected arXiv IDs (one per line)")
    parser.add_argument("--existing-ids", nargs="*", type=Path, default=None,
                        help="Files with already-downloaded IDs to exclude")
    parser.add_argument("--download", action="store_true",
                        help="Also invoke download_latex_sources.py to fetch LaTeX sources")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print what would be done without making API calls")
    args = parser.parse_args()

    # Resolve API key from env if not provided
    import os
    api_key = args.api_key or os.environ.get("SS_API_KEY") or os.environ.get("SEMANTIC_SCHOLAR_API_KEY")
    delay = KEY_DELAY if api_key else DEFAULT_DELAY

    if api_key:
        print(f"Using Semantic Scholar API key (rate limit: {delay}s)", flush=True)
    else:
        print(f"No API key (rate limit: {delay}s/req). Set SS_API_KEY for higher limits.", flush=True)

    if args.dry_run:
        print("DRY RUN: would crawl citation network for seeds:", args.seeds)
        print(f"  hops={args.hops}, max_papers={args.max_papers}, directions={args.directions}")
        return

    # Load existing IDs to skip
    existing_ids: Set[str] = set()
    if args.existing_ids:
        existing_ids = load_existing_ids(args.existing_ids)
        print(f"Excluding {len(existing_ids)} already-downloaded IDs", flush=True)

    # Also auto-detect existing LaTeX sources
    latex_dir = _PROJECT_ROOT / "data" / "00_raw" / "latex_sources" / "extracted"
    if latex_dir.exists():
        for d in latex_dir.iterdir():
            if d.is_dir():
                existing_ids.add(d.name)
        print(f"Auto-detected {len(existing_ids)} already-extracted papers", flush=True)

    # Crawl citation network
    print(f"\n=== Crawling citation network (hops={args.hops}) ===", flush=True)
    collected = crawl_network(
        seed_arxiv_ids=args.seeds,
        api_key=api_key,
        max_papers=args.max_papers,
        hops=args.hops,
        delay=delay,
        min_citations=args.min_citations,
        directions=args.directions,
    )

    # Optional topic search
    if args.topic_query:
        print(f"\n=== Topic search: '{args.topic_query}' ===", flush=True)
        topic_ids = topic_search(args.topic_query, api_key, delay, args.topic_max, args.year_range)
        print(f"Topic search added {len(topic_ids)} candidate papers", flush=True)
        collected.update(topic_ids)

    # Remove already-downloaded
    new_ids = sorted(collected - existing_ids)
    print(f"\nTotal collected: {len(collected)}, already have: {len(existing_ids)}, new: {len(new_ids)}", flush=True)

    # Write output
    args.output_ids.parent.mkdir(parents=True, exist_ok=True)
    args.output_ids.write_text("\n".join(new_ids) + "\n", encoding="utf-8")
    print(f"Wrote {len(new_ids)} new arXiv IDs to {args.output_ids}", flush=True)

    # Optionally trigger LaTeX source download
    if args.download and new_ids:
        print(f"\n=== Triggering LaTeX source download for {len(new_ids)} papers ===", flush=True)
        import subprocess
        cmd = [
            sys.executable,
            str(_PROJECT_ROOT / "scripts" / "download_latex_sources.py"),
            "--id-file", str(args.output_ids),
        ]
        print("Running:", " ".join(cmd), flush=True)
        subprocess.run(cmd, check=False)
    elif args.download:
        print("No new papers to download.", flush=True)
    else:
        print("\nTo download LaTeX sources, run:", flush=True)
        print(f"  python scripts/download_latex_sources.py --id-file {args.output_ids}", flush=True)


if __name__ == "__main__":
    main()
