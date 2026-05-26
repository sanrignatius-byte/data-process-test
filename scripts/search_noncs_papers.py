#!/usr/bin/env python3
"""Search arXiv for non-CS papers (math/physics/bio/econ/stat/eess) with both
PDF and LaTeX source, using a 'survey-seed → follow references' strategy.

Pipeline (each phase is resumable and stand-alone):
  P1 search_surveys     arXiv API category-search for survey/review papers
                        across non-CS category groups
  P2 download_seeds     download PDF + LaTeX for the survey seeds
  P3 expand_refs        extract arXiv ref IDs from seeds' LaTeX (.bbl/.bib/.tex)
  P4 filter_to_noncs    query arXiv metadata in batches; keep only those whose
                        primary category is in the non-CS whitelist
  P5 download_refs      download PDF + LaTeX for the filtered candidates,
                        until --target is reached or candidates exhausted

Usage (smoke run, ~30 surveys, no ref expansion):
    python scripts/search_noncs_papers.py --phase smoke

Full pipeline (~5000 papers total):
    python scripts/search_noncs_papers.py --phase all --target 5000

Per-phase (resume after interruption):
    python scripts/search_noncs_papers.py --phase search_surveys
    python scripts/search_noncs_papers.py --phase download_seeds
    python scripts/search_noncs_papers.py --phase expand_refs
    python scripts/search_noncs_papers.py --phase filter_to_noncs
    python scripts/search_noncs_papers.py --phase download_refs --target 5000
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple
from urllib.parse import quote

import requests

# Re-use the workhorses from the existing batch download script.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
from batch_download_references import (  # noqa: E402
    batch_download,
    download_latex,
    download_pdf,
    extract_arxiv_ids_from_latex,
)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

ARXIV_API = "https://export.arxiv.org/api/query"
USER_AGENT = "data-process-test/0.1 (m2query non-cs survey crawler; +https://arxiv.org)"
NS = {"a": "http://www.w3.org/2005/Atom", "os": "http://a9.com/-/spec/opensearch/1.1/"}

# arXiv asks for ≥3s between requests; we use 4s + jitter + 429-backoff.
BASE_DELAY = 4.0

# Non-CS primary-category whitelist (regex match against primary cat string).
# A paper's primary cat must match ONE of these AND must NOT start with "cs.".
NONCS_PRIMARY_RE = re.compile(
    r"^("
    r"math\.|stat\.|"
    r"physics\.|astro-ph(?:\.|$)|hep-|cond-mat(?:\.|$)|gr-qc|nucl-|nlin\.|"
    r"q-bio\.|q-fin\.|econ\.|"
    r"eess\."
    r")"
)

# Survey-search seeds: one (cat_pattern, topic_keyword) tuple per group.
# We rotate through these to spread coverage; each is independently rate-limited.
SURVEY_QUERIES: List[Tuple[str, str]] = [
    # Group 1: math + stat
    ("math.AG", "survey"), ("math.AP", "survey"), ("math.AT", "survey"),
    ("math.CO", "survey"), ("math.DG", "survey"), ("math.DS", "survey"),
    ("math.NA", "review"), ("math.OC", "review"), ("math.PR", "survey"),
    ("math.ST", "review"), ("stat.ME", "review"), ("stat.AP", "review"),
    # Group 2: physics / astro / hep / cond-mat / gr-qc
    ("physics.app-ph", "review"), ("physics.bio-ph", "review"),
    ("physics.chem-ph", "review"), ("physics.flu-dyn", "review"),
    ("astro-ph.CO", "review"), ("astro-ph.GA", "review"), ("astro-ph.SR", "review"),
    ("hep-ph", "review"), ("hep-th", "review"), ("hep-ex", "review"),
    ("cond-mat.mes-hall", "review"), ("cond-mat.soft", "review"),
    ("cond-mat.stat-mech", "review"), ("gr-qc", "review"),
    ("nucl-th", "review"), ("nlin.CD", "review"),
    # Group 3: q-bio / q-fin / econ
    ("q-bio.BM", "review"), ("q-bio.GN", "review"), ("q-bio.NC", "review"),
    ("q-bio.PE", "review"), ("q-bio.QM", "review"), ("q-bio.TO", "review"),
    ("q-fin.PM", "review"), ("q-fin.RM", "review"), ("q-fin.ST", "review"),
    ("econ.EM", "review"), ("econ.TH", "review"), ("econ.GN", "review"),
    # Group 4: eess (signal/audio/image processing — not strictly CS)
    ("eess.AS", "review"), ("eess.IV", "review"), ("eess.SP", "review"),
    ("eess.SY", "review"),
]

# Output paths (all under data-process-test/data/00_raw/)
DATA_ROOT = PROJECT_ROOT / "data" / "00_raw"
SURVEY_IDS_FILE = DATA_ROOT / "noncs_survey_ids.txt"
CANDIDATE_IDS_FILE = DATA_ROOT / "noncs_candidate_ids.txt"
FILTERED_IDS_FILE = DATA_ROOT / "noncs_filtered_ids.txt"
META_CACHE_FILE = DATA_ROOT / "noncs_arxiv_meta_cache.json"
PDF_DIR = DATA_ROOT / "pdfs_noncs"
LATEX_DIR = DATA_ROOT / "latex_sources_noncs"

# ---------------------------------------------------------------------------
# arXiv API client with rate-limit + backoff
# ---------------------------------------------------------------------------

class ArxivClient:
    def __init__(self, base_delay: float = BASE_DELAY):
        self.base_delay = base_delay
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": USER_AGENT})
        self._last_call = 0.0

    def _sleep(self) -> None:
        elapsed = time.time() - self._last_call
        if elapsed < self.base_delay:
            time.sleep(self.base_delay - elapsed)
        self._last_call = time.time()

    def get_xml(self, params: Dict[str, str], retries: int = 4) -> Optional[ET.Element]:
        for attempt in range(retries):
            self._sleep()
            try:
                resp = self.session.get(ARXIV_API, params=params, timeout=60)
            except requests.exceptions.RequestException as e:
                print(f"  [arxiv] req error: {e}; backoff {30 * (attempt + 1)}s")
                time.sleep(30 * (attempt + 1))
                continue
            if resp.status_code == 429:
                wait = 60 * (attempt + 1)
                print(f"  [arxiv] 429 Rate Exceeded; sleep {wait}s")
                time.sleep(wait)
                continue
            if resp.status_code != 200:
                print(f"  [arxiv] HTTP {resp.status_code}; trying again")
                time.sleep(10 * (attempt + 1))
                continue
            try:
                return ET.fromstring(resp.text)
            except ET.ParseError as e:
                print(f"  [arxiv] XML parse error: {e}; head={resp.text[:200]!r}")
                time.sleep(10 * (attempt + 1))
        return None

    def search(self, query: str, start: int = 0, max_results: int = 30) -> List[Dict]:
        params = {
            "search_query": query,
            "start": str(start),
            "max_results": str(max_results),
            "sortBy": "relevance",
            "sortOrder": "descending",
        }
        root = self.get_xml(params)
        if root is None:
            return []
        out: List[Dict] = []
        for e in root.findall("a:entry", NS):
            id_el = e.find("a:id", NS)
            if id_el is None or not id_el.text:
                continue
            aid = id_el.text.rsplit("/", 1)[-1]
            # strip vN suffix to canonicalize
            aid = re.sub(r"v\d+$", "", aid)
            title_el = e.find("a:title", NS)
            cats = [c.get("term") for c in e.findall("a:category", NS) if c.get("term")]
            primary = cats[0] if cats else ""
            out.append({
                "arxiv_id": aid,
                "title": (title_el.text or "").strip() if title_el is not None else "",
                "primary_cat": primary,
                "all_cats": cats,
            })
        return out

    def fetch_meta_batch(self, ids: List[str], batch: int = 50) -> Dict[str, Dict]:
        """Fetch primary category for many arxiv IDs in batches (id_list= query)."""
        out: Dict[str, Dict] = {}
        for i in range(0, len(ids), batch):
            chunk = ids[i:i + batch]
            params = {
                "id_list": ",".join(chunk),
                "max_results": str(len(chunk)),
            }
            root = self.get_xml(params)
            if root is None:
                continue
            for e in root.findall("a:entry", NS):
                id_el = e.find("a:id", NS)
                if id_el is None or not id_el.text:
                    continue
                aid = re.sub(r"v\d+$", "", id_el.text.rsplit("/", 1)[-1])
                cats = [c.get("term") for c in e.findall("a:category", NS) if c.get("term")]
                title_el = e.find("a:title", NS)
                out[aid] = {
                    "primary_cat": cats[0] if cats else "",
                    "all_cats": cats,
                    "title": ((title_el.text or "").strip() if title_el is not None else ""),
                }
            print(f"  [meta] {i + len(chunk)}/{len(ids)} fetched")
        return out


# ---------------------------------------------------------------------------
# Phase implementations
# ---------------------------------------------------------------------------

def phase_search_surveys(per_query: int, output: Path) -> None:
    print(f"=== P1: search surveys (per_query={per_query}, queries={len(SURVEY_QUERIES)}) ===")
    DATA_ROOT.mkdir(parents=True, exist_ok=True)
    existing: Set[str] = set()
    if output.exists():
        existing = {line.strip() for line in output.read_text().splitlines() if line.strip()}
        print(f"  resume: {len(existing)} surveys already in file")

    client = ArxivClient()
    found: Set[str] = set(existing)
    for cat, topic in SURVEY_QUERIES:
        q = f"cat:{cat}+AND+(abs:{topic}+OR+ti:{topic})"
        # arxiv accepts space or '+' in query — use quote for safety
        q_enc = quote(q, safe="+:")
        print(f"  [{cat}/{topic}]")
        results = client.search(q_enc, start=0, max_results=per_query)
        new_in_cat = 0
        for r in results:
            aid = r["arxiv_id"]
            primary = r.get("primary_cat", "")
            if primary.startswith("cs."):
                continue  # exclude CS even if it appears in search results
            if not NONCS_PRIMARY_RE.match(primary):
                continue
            if aid not in found:
                found.add(aid)
                new_in_cat += 1
        print(f"    +{new_in_cat} new (total {len(found)})")
        # flush after every query so we never lose progress
        output.write_text("\n".join(sorted(found)) + "\n")
    print(f"  -> {output} ({len(found)} surveys)")


def phase_download_seeds(ids_file: Path, workers: int) -> None:
    print(f"=== P2: download survey seeds (workers={workers}) ===")
    if not ids_file.exists():
        sys.exit(f"missing {ids_file} — run --phase search_surveys first")
    ids = [line.strip() for line in ids_file.read_text().splitlines() if line.strip()]
    print(f"  downloading PDF + LaTeX for {len(ids)} seeds")
    results = batch_download(
        arxiv_ids=ids,
        pdf_dir=PDF_DIR,
        latex_dir=LATEX_DIR,
        download_latex_flag=True,
        workers=workers,
    )
    pdf_ok = sum(1 for r in results if r.get("pdf_ok"))
    tex_ok = sum(1 for r in results if r.get("latex_ok"))
    print(f"  PDF ok: {pdf_ok}/{len(ids)} ; LaTeX ok: {tex_ok}/{len(ids)}")


def phase_expand_refs(latex_dir: Path, output: Path) -> None:
    print(f"=== P3: expand refs from {latex_dir / 'extracted'} ===")
    refs_per_doc = extract_arxiv_ids_from_latex(latex_dir / "extracted")
    all_refs: Set[str] = set()
    for ids in refs_per_doc.values():
        all_refs.update(ids)
    output.write_text("\n".join(sorted(all_refs)) + "\n")
    print(f"  {len(refs_per_doc)} surveys → {len(all_refs)} unique candidate ref IDs")
    print(f"  -> {output}")


def phase_filter_to_noncs(
    candidate_file: Path,
    cache_file: Path,
    output: Path,
    batch: int = 50,
) -> None:
    print(f"=== P4: filter candidates to non-CS via arxiv metadata ===")
    candidates = [line.strip() for line in candidate_file.read_text().splitlines() if line.strip()]
    print(f"  {len(candidates)} candidates from {candidate_file.name}")
    cache: Dict[str, Dict] = {}
    if cache_file.exists():
        cache = json.loads(cache_file.read_text())
        print(f"  meta cache hit: {len(cache)} ids already known")
    missing = [c for c in candidates if c not in cache]
    print(f"  fetching meta for {len(missing)} new ids (batch={batch})")
    client = ArxivClient()
    new_meta = client.fetch_meta_batch(missing, batch=batch)
    cache.update(new_meta)
    cache_file.write_text(json.dumps(cache, indent=1))
    print(f"  cache updated → {cache_file}")
    kept = [c for c in candidates
            if c in cache
            and not cache[c]["primary_cat"].startswith("cs.")
            and NONCS_PRIMARY_RE.match(cache[c]["primary_cat"])]
    output.write_text("\n".join(kept) + "\n")
    print(f"  kept {len(kept)}/{len(candidates)} non-CS candidates → {output}")


def phase_download_refs(
    filtered_file: Path,
    target: int,
    workers: int,
) -> None:
    print(f"=== P5: download refs (target={target}, workers={workers}) ===")
    ids = [line.strip() for line in filtered_file.read_text().splitlines() if line.strip()]
    # Count existing successful downloads (LaTeX-source bearing) so we resume.
    done: Set[str] = set()
    extracted = LATEX_DIR / "extracted"
    if extracted.is_dir():
        for d in extracted.iterdir():
            if d.is_dir() and any(d.rglob("*.tex")):
                done.add(d.name.replace("_", "/", 1) if "_" in d.name and not d.name[:4].isdigit() else d.name)
    print(f"  already have LaTeX for {len(done)} papers (resuming)")
    todo = [i for i in ids if i.replace("/", "_") not in {d.replace("/", "_") for d in done}]
    quota = max(0, target - len(done))
    todo = todo[:quota]
    print(f"  downloading {len(todo)} more papers (quota = target {target} − have {len(done)})")
    if not todo:
        print("  nothing to do; quota satisfied")
        return
    results = batch_download(
        arxiv_ids=todo,
        pdf_dir=PDF_DIR,
        latex_dir=LATEX_DIR,
        download_latex_flag=True,
        workers=workers,
    )
    pdf_ok = sum(1 for r in results if r.get("pdf_ok"))
    tex_ok = sum(1 for r in results if r.get("latex_ok"))
    print(f"  PDF ok: {pdf_ok}/{len(todo)} ; LaTeX ok: {tex_ok}/{len(todo)}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--phase",
        choices=["smoke", "search_surveys", "download_seeds", "expand_refs",
                 "filter_to_noncs", "download_refs", "all"],
        default="search_surveys",
    )
    parser.add_argument("--per-query", type=int, default=20,
                        help="surveys to fetch per (category, topic) query in P1 (default 20)")
    parser.add_argument("--target", type=int, default=5000,
                        help="target total non-CS papers in final corpus (default 5000)")
    parser.add_argument("--workers", type=int, default=2,
                        help="parallel download workers (arxiv-polite ≤3)")
    parser.add_argument("--meta-batch", type=int, default=50,
                        help="ids per arxiv metadata query in P4 (default 50)")
    args = parser.parse_args()

    if args.phase == "smoke":
        phase_search_surveys(per_query=3, output=SURVEY_IDS_FILE)
        phase_download_seeds(ids_file=SURVEY_IDS_FILE, workers=args.workers)
        return

    if args.phase in ("search_surveys", "all"):
        phase_search_surveys(per_query=args.per_query, output=SURVEY_IDS_FILE)
    if args.phase in ("download_seeds", "all"):
        phase_download_seeds(ids_file=SURVEY_IDS_FILE, workers=args.workers)
    if args.phase in ("expand_refs", "all"):
        phase_expand_refs(latex_dir=LATEX_DIR, output=CANDIDATE_IDS_FILE)
    if args.phase in ("filter_to_noncs", "all"):
        phase_filter_to_noncs(
            candidate_file=CANDIDATE_IDS_FILE,
            cache_file=META_CACHE_FILE,
            output=FILTERED_IDS_FILE,
            batch=args.meta_batch,
        )
    if args.phase in ("download_refs", "all"):
        phase_download_refs(
            filtered_file=FILTERED_IDS_FILE,
            target=args.target,
            workers=args.workers,
        )


if __name__ == "__main__":
    main()
