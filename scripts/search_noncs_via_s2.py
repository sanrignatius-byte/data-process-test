#!/usr/bin/env python3
"""Discover non-CS arXiv paper IDs via Semantic Scholar API (S2), then hand
the resulting ID list off to `search_noncs_papers.py --phase download_refs`
for the actual arXiv PDF + LaTeX download.

Why S2 instead of arXiv API for discovery?
  * S2 returns `fieldsOfStudy` directly — no need for per-paper category lookup
  * S2 returns reference lists directly — no need to parse .bbl / .bib
  * S2's per-key limit (1 req/s) is much more forgiving than arXiv's burst limit
  * Bypasses the arXiv API 429 wall we keep hitting

Pipeline (compatible with search_noncs_papers.py):
  S1 s2_seed         Pick survey-like seeds by fieldsOfStudy via paper/search
                     → data/00_raw/noncs_s2_seeds.json (with arxivId + fields)
  S2 s2_expand_refs  Expand each seed's references via paper/{id}/references
                     Filter to: has arxivId + non-CS fieldsOfStudy
                     → data/00_raw/noncs_filtered_ids.txt   (drop-in replacement
                                                             for arXiv pipeline)

Then run the existing download phase to actually grab PDF + LaTeX:
  python scripts/search_noncs_papers.py --phase download_refs --target 5000

Usage:
  export SEMANTIC_SCHOLAR_API_KEY=...
  python scripts/search_noncs_via_s2.py --phase all --seeds-per-field 100 --target-candidates 12000
  python scripts/search_noncs_via_s2.py --phase s2_seed
  python scripts/search_noncs_via_s2.py --phase s2_expand_refs --target-candidates 12000
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Set, Tuple

import requests

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# ---------------------------------------------------------------------------
# S2 API config
# ---------------------------------------------------------------------------

S2_BASE = "https://api.semanticscholar.org/graph/v1"
USER_AGENT = "data-process-test/0.1 (non-cs corpus builder)"

# S2 documented: 1 req/s per key; we add headroom.
S2_DELAY_WITH_KEY = 1.1
S2_DELAY_ANON = 4.0  # anon limit is much harsher in practice

# Non-CS fields used for S2 discovery. Order matters: roughly priority for
# how much budget each field gets in seed-search.
S2_NON_CS_FIELDS = [
    "Mathematics",
    "Physics",
    "Biology",
    "Medicine",          # has overlap with q-bio, but lots of LaTeX papers
    "Economics",
    "Materials Science",
    "Chemistry",
    "Engineering",
    "Environmental Science",
    "Geology",
]
# Whitelist of S2 fields treated as "non-CS" when filtering references.
S2_NON_CS_FIELDS_SET: Set[str] = set(S2_NON_CS_FIELDS)
# Hard reject if a paper's fields include "Computer Science" as PRIMARY
# (we still accept cross-listed CS + non-CS where non-CS is primary).

# Seed search queries — broad-but-survey-biased, one per field
SEED_QUERIES_PER_FIELD: Dict[str, List[str]] = {
    "Mathematics": ["survey", "review", "introduction"],
    "Physics": ["review", "survey", "introduction"],
    "Biology": ["review", "perspective"],
    "Medicine": ["review", "meta-analysis"],
    "Economics": ["survey", "review"],
    "Materials Science": ["review", "perspective"],
    "Chemistry": ["review", "perspective"],
    "Engineering": ["review", "survey"],
    "Environmental Science": ["review"],
    "Geology": ["review"],
}

DATA_ROOT = PROJECT_ROOT / "data" / "00_raw"
SEEDS_FILE = DATA_ROOT / "noncs_s2_seeds.json"
FILTERED_IDS_FILE = DATA_ROOT / "noncs_filtered_ids.txt"
META_CACHE_FILE = DATA_ROOT / "noncs_s2_meta_cache.json"


# ---------------------------------------------------------------------------
# S2 client
# ---------------------------------------------------------------------------

class S2Client:
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.environ.get("SEMANTIC_SCHOLAR_API_KEY")
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": USER_AGENT})
        if self.api_key:
            self.session.headers["x-api-key"] = self.api_key
        self.delay = S2_DELAY_WITH_KEY if self.api_key else S2_DELAY_ANON
        self._last = 0.0

    def _sleep(self) -> None:
        wait = self.delay - (time.time() - self._last)
        if wait > 0:
            time.sleep(wait)
        self._last = time.time()

    def _get(self, path: str, params: Dict[str, str], retries: int = 4) -> Optional[Dict]:
        for attempt in range(retries):
            self._sleep()
            try:
                r = self.session.get(f"{S2_BASE}{path}", params=params, timeout=45)
            except requests.exceptions.RequestException as e:
                print(f"  [s2] req error: {e}; sleep {30 * (attempt + 1)}s", flush=True)
                time.sleep(30 * (attempt + 1))
                continue
            if r.status_code == 200:
                try:
                    return r.json()
                except ValueError:
                    print(f"  [s2] non-JSON 200 response (head={r.text[:200]!r})", flush=True)
                    return None
            if r.status_code == 429:
                wait = 30 * (attempt + 1)
                print(f"  [s2] 429 rate-exceeded; sleep {wait}s", flush=True)
                time.sleep(wait)
                continue
            if r.status_code == 403:
                # Often a per-key cooldown after burst; back off long.
                wait = 120 * (attempt + 1)
                print(f"  [s2] 403 Forbidden (key cooldown?); sleep {wait}s", flush=True)
                time.sleep(wait)
                continue
            print(f"  [s2] HTTP {r.status_code} body={r.text[:200]!r}; sleep 15s", flush=True)
            time.sleep(15)
        return None

    def search_papers(
        self,
        query: str,
        field_of_study: str,
        year_from: int = 2010,
        limit: int = 100,
        max_total: int = 500,
    ) -> List[Dict]:
        """Use paper/search to find candidate seed papers. Paginates via offset.
        Returns list of paper dicts with externalIds + fieldsOfStudy."""
        out: List[Dict] = []
        offset = 0
        while len(out) < max_total:
            params = {
                "query": query,
                "fieldsOfStudy": field_of_study,
                "year": f"{year_from}-",
                "fields": "title,externalIds,year,fieldsOfStudy,citationCount,s2FieldsOfStudy",
                "limit": str(min(limit, max_total - len(out))),
                "offset": str(offset),
            }
            data = self._get("/paper/search", params)
            if not data:
                break
            items = data.get("data") or []
            if not items:
                break
            out.extend(items)
            next_offset = data.get("next")
            if next_offset is None:
                break
            offset = next_offset
        return out

    def get_references(self, arxiv_id: str, limit: int = 100, max_total: int = 200) -> List[Dict]:
        """Pull a paper's references via /paper/{id}/references."""
        out: List[Dict] = []
        offset = 0
        while len(out) < max_total:
            params = {
                "fields": "title,externalIds,year,fieldsOfStudy,citationCount",
                "limit": str(min(limit, max_total - len(out))),
                "offset": str(offset),
            }
            data = self._get(f"/paper/arXiv:{arxiv_id}/references", params)
            if not data:
                break
            items = data.get("data") or []
            if not items:
                break
            out.extend(items)
            next_offset = data.get("next")
            if next_offset is None:
                break
            offset = next_offset
        return out


# ---------------------------------------------------------------------------
# Filtering helpers
# ---------------------------------------------------------------------------

def has_arxiv_id(paper_or_ref: Dict) -> Optional[str]:
    """Extract arXiv ID from S2's externalIds; handle both /paper and
    /references shapes (the latter wraps the cited paper in `citedPaper`)."""
    p = paper_or_ref.get("citedPaper") or paper_or_ref
    eids = p.get("externalIds") or {}
    aid = eids.get("ArXiv") or eids.get("arxiv")
    if not aid:
        return None
    # Strip vN suffix
    return re.sub(r"v\d+$", "", aid)


def is_non_cs(paper_or_ref: Dict) -> bool:
    p = paper_or_ref.get("citedPaper") or paper_or_ref
    fos = p.get("fieldsOfStudy") or []
    s2fos_raw = p.get("s2FieldsOfStudy") or []
    s2fos = [x.get("category") for x in s2fos_raw if isinstance(x, dict)]
    all_fields = set(fos) | set(s2fos)
    if not all_fields:
        return False
    # Reject if ALL fields are Computer Science (cross-listed CS+math is fine)
    non_cs_present = bool(all_fields & S2_NON_CS_FIELDS_SET)
    if not non_cs_present:
        return False
    # Bias against papers that are CS-primary: simple heuristic — if CS appears
    # but no non-CS, reject (already handled). If CS appears together with
    # non-CS, accept (cross-listed).
    return True


# ---------------------------------------------------------------------------
# Phase: S1 seed
# ---------------------------------------------------------------------------

def phase_s2_seed(seeds_per_field: int, output: Path) -> None:
    print(f"=== S1: discover seeds via S2 paper/search ({len(S2_NON_CS_FIELDS)} fields × ~{seeds_per_field}) ===")
    DATA_ROOT.mkdir(parents=True, exist_ok=True)
    existing: Dict[str, Dict] = {}
    if output.exists():
        try:
            existing = json.loads(output.read_text())
            print(f"  resume: {len(existing)} seeds already saved")
        except json.JSONDecodeError:
            pass

    client = S2Client()
    if not client.api_key:
        print("  WARNING: no SEMANTIC_SCHOLAR_API_KEY env; running anonymously (slower + harsher limits)")

    found: Dict[str, Dict] = dict(existing)
    for field in S2_NON_CS_FIELDS:
        queries = SEED_QUERIES_PER_FIELD.get(field, ["review"])
        per_query = max(50, seeds_per_field // len(queries))
        for q in queries:
            print(f"  [{field}/{q}] up to {per_query}", flush=True)
            results = client.search_papers(q, field, max_total=per_query)
            kept = 0
            for r in results:
                aid = has_arxiv_id(r)
                if not aid:
                    continue
                if aid in found:
                    continue
                if not is_non_cs(r):
                    continue
                found[aid] = {
                    "arxivId": aid,
                    "title": (r.get("title") or "")[:160],
                    "year": r.get("year"),
                    "fieldsOfStudy": r.get("fieldsOfStudy") or [],
                    "primaryField": field,
                    "citationCount": r.get("citationCount", 0),
                }
                kept += 1
            print(f"    +{kept} kept (total {len(found)})", flush=True)
            output.write_text(json.dumps(found, indent=1))  # flush after each query
    print(f"  -> {output} ({len(found)} seeds)")


# ---------------------------------------------------------------------------
# Phase: S2 expand_refs
# ---------------------------------------------------------------------------

def phase_s2_expand_refs(
    seeds_file: Path,
    target_candidates: int,
    output: Path,
    cache_file: Path,
    max_per_seed: int = 100,
) -> None:
    print(f"=== S2: expand seed refs via S2 paper/references (target {target_candidates}) ===")
    if not seeds_file.exists():
        sys.exit(f"missing {seeds_file} — run --phase s2_seed first")
    seeds = json.loads(seeds_file.read_text())
    print(f"  {len(seeds)} seeds")

    # Resume from cache: arxivId -> {fieldsOfStudy, ...}
    cache: Dict[str, Dict] = {}
    if cache_file.exists():
        cache = json.loads(cache_file.read_text())
        print(f"  cache hit: {len(cache)} ref papers already known")

    client = S2Client()
    candidates: Dict[str, Dict] = dict(cache)
    processed_seeds: Set[str] = set()

    # Round-robin through seeds so we get balanced coverage across fields
    seed_ids = list(seeds.keys())
    for i, seed_id in enumerate(seed_ids):
        if len(candidates) >= target_candidates:
            print(f"  reached target ({len(candidates)} ≥ {target_candidates}); stopping", flush=True)
            break
        print(f"  [{i + 1}/{len(seed_ids)}] {seed_id}  ({len(candidates)} candidates so far)", flush=True)
        refs = client.get_references(seed_id, max_total=max_per_seed)
        kept = 0
        for ref in refs:
            aid = has_arxiv_id(ref)
            if not aid or aid in candidates:
                continue
            if not is_non_cs(ref):
                continue
            cp = ref.get("citedPaper") or ref
            candidates[aid] = {
                "arxivId": aid,
                "title": (cp.get("title") or "")[:160],
                "fieldsOfStudy": cp.get("fieldsOfStudy") or [],
                "year": cp.get("year"),
                "from_seed": seed_id,
            }
            kept += 1
        processed_seeds.add(seed_id)
        print(f"    +{kept} kept (total {len(candidates)})", flush=True)
        # flush cache periodically
        if (i + 1) % 10 == 0:
            cache_file.write_text(json.dumps(candidates, indent=1))

    cache_file.write_text(json.dumps(candidates, indent=1))

    # Write final IDs file (drop-in for search_noncs_papers.py P5)
    output.write_text("\n".join(candidates.keys()) + "\n")
    print(f"  -> {output} ({len(candidates)} candidate arxiv IDs)")
    print(f"  -> {cache_file} (full metadata cached)")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--phase", choices=["s2_seed", "s2_expand_refs", "all"], default="all")
    p.add_argument("--seeds-per-field", type=int, default=100,
                   help="how many seeds to pull per S2 field (default 100; with 10 fields ≈ 1000 seeds)")
    p.add_argument("--target-candidates", type=int, default=12000,
                   help="stop expanding refs once this many unique non-CS arxiv IDs are collected")
    p.add_argument("--max-refs-per-seed", type=int, default=100,
                   help="max references to pull per seed paper (S2 paginates)")
    args = p.parse_args()

    if args.phase in ("s2_seed", "all"):
        phase_s2_seed(args.seeds_per_field, SEEDS_FILE)
    if args.phase in ("s2_expand_refs", "all"):
        phase_s2_expand_refs(
            seeds_file=SEEDS_FILE,
            target_candidates=args.target_candidates,
            output=FILTERED_IDS_FILE,
            cache_file=META_CACHE_FILE,
            max_per_seed=args.max_refs_per_seed,
        )

    print("\nNext step (when ready):")
    print("  python scripts/search_noncs_papers.py --phase download_refs --target 5000")


if __name__ == "__main__":
    main()
