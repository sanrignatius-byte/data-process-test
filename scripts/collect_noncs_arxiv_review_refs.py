#!/usr/bin/env python3
"""Collect a non-CS arXiv corpus from review-paper references.

The pipeline is intentionally isolated from the existing batch2 production data:

1. Search arXiv for non-CS review/survey seed papers.
2. Download and extract the seed papers' LaTeX sources.
3. Extract arXiv IDs from their bibliography/source files.
4. Keep at most N references per seed, preferring references whose arXiv
   metadata has no ``cs.*`` category.
5. Download target PDFs and LaTeX sources until TARGET papers have both.

Outputs are resume-friendly and suitable for a downstream MinerU + graph Slurm
job.  The downloaded review seeds are not counted in the target corpus.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
import xml.etree.ElementTree as ET
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set

import requests

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from batch_download_references import (  # noqa: E402
    download_latex,
    download_pdf,
    extract_arxiv_ids_from_latex,
)
from download_latex_sources import process_papers as download_seed_sources  # noqa: E402

ARXIV_API = "https://export.arxiv.org/api/query"
ARXIV_NS = {"atom": "http://www.w3.org/2005/Atom"}

DEFAULT_BASE = PROJECT_ROOT / "data" / "00_raw"
DEFAULT_SEED_SOURCE_DIR = DEFAULT_BASE / "noncs400_seed_latex_sources"
DEFAULT_PDF_DIR = DEFAULT_BASE / "noncs400_pdfs"
DEFAULT_LATEX_DIR = DEFAULT_BASE / "noncs400_latex_sources"
DEFAULT_SEED_MANIFEST = DEFAULT_BASE / "noncs400_seed_manifest.json"
DEFAULT_TARGET_IDS = DEFAULT_BASE / "noncs400_arxiv_ids.txt"
DEFAULT_SUCCESS_IDS = DEFAULT_BASE / "noncs400_success_ids.txt"
DEFAULT_MANIFEST = DEFAULT_BASE / "noncs400_manifest.json"

REVIEW_KEYWORDS = ("review", "survey", "overview", "tutorial", "primer")
NONCS_TOPIC_QUERIES = [
    ("astro_cosmology", "all:review AND all:cosmology AND (cat:astro-ph.CO OR cat:gr-qc)"),
    ("astro_exoplanets", "all:review AND all:exoplanet AND (cat:astro-ph.EP OR cat:astro-ph.SR)"),
    ("astro_galaxies", "all:review AND all:galaxy AND (cat:astro-ph.GA OR cat:astro-ph.CO)"),
    ("gravitational_waves", "all:review AND all:gravitational AND all:waves AND (cat:gr-qc OR cat:astro-ph.HE)"),
    ("dark_matter", "all:review AND all:dark AND all:matter AND (cat:hep-ph OR cat:astro-ph.CO OR cat:gr-qc)"),
    ("neutrino_physics", "all:review AND all:neutrino AND (cat:hep-ph OR cat:nucl-th OR cat:astro-ph.HE)"),
    ("topological_materials", "all:review AND all:topological AND (cat:cond-mat.mes-hall OR cat:cond-mat.mtrl-sci OR cat:cond-mat.str-el)"),
    ("quantum_materials", "all:review AND all:quantum AND all:materials AND (cat:cond-mat.mtrl-sci OR cat:cond-mat.str-el)"),
    ("soft_matter", "all:review AND all:soft AND all:matter AND (cat:cond-mat.soft OR cat:physics.bio-ph)"),
    ("plasma_physics", "all:review AND all:plasma AND (cat:physics.plasm-ph OR cat:astro-ph.SR)"),
    ("quantum_information", "all:review AND all:quantum AND all:information AND cat:quant-ph"),
    ("nuclear_physics", "all:review AND all:nuclear AND (cat:nucl-th OR cat:nucl-ex)"),
    ("number_theory", "all:survey AND all:number AND all:theory AND cat:math.NT"),
    ("algebraic_geometry", "all:survey AND all:algebraic AND all:geometry AND cat:math.AG"),
    ("dynamical_systems", "all:review AND all:dynamical AND all:systems AND (cat:math.DS OR cat:nlin.CD)"),
    ("genomics", "all:review AND all:genomics AND (cat:q-bio.GN OR cat:q-bio.QM)"),
    ("protein_biophysics", "all:review AND all:protein AND (cat:q-bio.BM OR cat:physics.bio-ph)"),
    ("epidemiology", "all:review AND all:epidemiology AND (cat:q-bio.PE OR cat:stat.AP)"),
    ("finance", "all:review AND all:finance AND (cat:q-fin.GN OR cat:q-fin.ST OR cat:econ.GN)"),
    ("earth_planetary", "all:review AND all:planetary AND (cat:physics.geo-ph OR cat:astro-ph.EP)"),
]


def strip_version(arxiv_id: str) -> str:
    return re.sub(r"v\d+$", "", arxiv_id.strip())


def safe_name(arxiv_id: str) -> str:
    return arxiv_id.replace("/", "_")


def is_noncs_categories(categories: Sequence[str]) -> bool:
    cats = [c for c in categories if c]
    return bool(cats) and not any(c.startswith("cs.") for c in cats)


def parse_arxiv_entries(xml_text: str) -> List[Dict[str, Any]]:
    root = ET.fromstring(xml_text)
    rows: List[Dict[str, Any]] = []
    for entry in root.findall("atom:entry", ARXIV_NS):
        id_text = entry.findtext("atom:id", "", ARXIV_NS)
        if "/abs/" not in id_text:
            continue
        arxiv_id = strip_version(id_text.rsplit("/abs/", 1)[-1])
        title = " ".join((entry.findtext("atom:title", "", ARXIV_NS) or "").split())
        abstract = " ".join((entry.findtext("atom:summary", "", ARXIV_NS) or "").split())
        published = (entry.findtext("atom:published", "", ARXIV_NS) or "")[:10]
        categories = [c.get("term", "") for c in entry.findall("atom:category", ARXIV_NS)]
        authors = [
            a.findtext("atom:name", "", ARXIV_NS)
            for a in entry.findall("atom:author", ARXIV_NS)
        ]
        rows.append(
            {
                "arxiv_id": arxiv_id,
                "title": title,
                "abstract": abstract,
                "published": published,
                "year": int(published[:4]) if published[:4].isdigit() else None,
                "categories": categories,
                "authors": [a for a in authors if a],
            }
        )
    return rows


def request_arxiv(
    session: requests.Session,
    params: Dict[str, Any],
    delay: float,
    max_retries: int,
) -> Optional[str]:
    for attempt in range(max_retries):
        if delay > 0:
            time.sleep(delay)
        try:
            resp = session.get(ARXIV_API, params=params, timeout=60)
        except requests.RequestException as exc:
            wait = min(300, 20 * (attempt + 1))
            print(f"  [arxiv-api] network error: {exc}; wait {wait}s", flush=True)
            time.sleep(wait)
            continue
        if resp.status_code == 200:
            return resp.text
        if resp.status_code == 429:
            wait = min(600, 60 * (attempt + 1))
            print(f"  [arxiv-api] 429 rate limit; wait {wait}s", flush=True)
            time.sleep(wait)
            continue
        print(f"  [arxiv-api] HTTP {resp.status_code}: {resp.text[:200]}", flush=True)
        if resp.status_code >= 500:
            time.sleep(min(300, 20 * (attempt + 1)))
            continue
        return None
    return None


def search_review_seeds(args: argparse.Namespace) -> List[Dict[str, Any]]:
    session = requests.Session()
    session.headers.update({"User-Agent": "m4-noncs400-seed-search/1.0 (research)"})
    seen: Set[str] = set()
    candidates: List[Dict[str, Any]] = []

    for label, query in NONCS_TOPIC_QUERIES:
        print(f"[seed-search] {label}: {query}", flush=True)
        xml_text = request_arxiv(
            session,
            {
                "search_query": query,
                "start": 0,
                "max_results": args.search_results_per_topic,
                "sortBy": "relevance",
                "sortOrder": "descending",
            },
            delay=args.api_delay,
            max_retries=args.api_retries,
        )
        if not xml_text:
            print("  no response", flush=True)
            continue
        rows = parse_arxiv_entries(xml_text)
        print(f"  got {len(rows)} rows", flush=True)
        for row in rows:
            aid = row["arxiv_id"]
            if aid in seen:
                continue
            if not is_noncs_categories(row.get("categories", [])):
                continue
            hay = f"{row.get('title','')} {row.get('abstract','')}".lower()
            if not any(k in hay for k in REVIEW_KEYWORDS):
                continue
            row["seed_topic"] = label
            row["_review_score"] = score_seed(row)
            seen.add(aid)
            candidates.append(row)

    candidates.sort(key=lambda r: (r.get("_review_score", 0), r.get("year") or 0), reverse=True)
    return candidates


def score_seed(row: Dict[str, Any]) -> float:
    title = (row.get("title") or "").lower()
    abstract = (row.get("abstract") or "").lower()
    score = 0.0
    if any(k in title for k in REVIEW_KEYWORDS):
        score += 10.0
    if "review" in title:
        score += 5.0
    if "survey" in title:
        score += 4.0
    if any(phrase in abstract for phrase in ("we review", "we survey", "this review", "this survey")):
        score += 3.0
    year = row.get("year") or 0
    if year:
        score += min(3.0, max(0.0, (year - 2015) / 4.0))
    return score


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def write_ids(path: Path, ids: Iterable[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(f"{aid}\n" for aid in ids), encoding="utf-8")


def load_seed_ids(args: argparse.Namespace) -> List[str]:
    ids: List[str] = []
    if args.seed_ids:
        ids.extend(args.seed_ids)
    if args.seed_id_file:
        for line in args.seed_id_file.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line and not line.startswith("#"):
                ids.append(line)
    return [strip_version(aid) for aid in ids]


def fetch_metadata_for_ids(
    ids: Sequence[str],
    args: argparse.Namespace,
) -> Dict[str, Dict[str, Any]]:
    session = requests.Session()
    session.headers.update({"User-Agent": "m4-noncs400-metadata/1.0 (research)"})
    metadata: Dict[str, Dict[str, Any]] = {}
    chunk_size = 50
    for start in range(0, len(ids), chunk_size):
        chunk = list(ids[start:start + chunk_size])
        print(f"[metadata] {start + 1}-{start + len(chunk)} / {len(ids)}", flush=True)
        xml_text = request_arxiv(
            session,
            {"id_list": ",".join(chunk), "max_results": len(chunk)},
            delay=args.api_delay,
            max_retries=args.api_retries,
        )
        if not xml_text:
            continue
        for row in parse_arxiv_entries(xml_text):
            metadata[row["arxiv_id"]] = row
    return metadata


def collect_refs_from_seed_sources(
    seed_ids: Sequence[str],
    args: argparse.Namespace,
) -> Dict[str, Set[str]]:
    print(f"[seed-source] downloading/extracting {len(seed_ids)} seed review sources", flush=True)
    download_seed_sources(
        arxiv_ids=list(seed_ids),
        output_dir=args.seed_source_dir,
        delay=args.download_delay,
        extract_only=False,
        verify_ssl=not args.no_verify,
    )
    extracted = args.seed_source_dir / "extracted"
    seed_refs = extract_arxiv_ids_from_latex(extracted)
    # Keep only selected seed IDs.  The extractor reads all subdirs in the source
    # directory, and this run may be resuming from a larger previous search.
    seed_set = set(seed_ids)
    return {seed: refs for seed, refs in seed_refs.items() if seed in seed_set}


def choose_target_candidates(
    seed_refs: Dict[str, Set[str]],
    seed_rows: Dict[str, Dict[str, Any]],
    args: argparse.Namespace,
) -> List[Dict[str, Any]]:
    all_ref_ids: List[str] = []
    seen: Set[str] = set(seed_refs.keys())
    for refs in seed_refs.values():
        for aid in sorted(strip_version(r) for r in refs):
            if aid not in seen:
                seen.add(aid)
                all_ref_ids.append(aid)

    print(f"[refs] unique refs before metadata/category filter: {len(all_ref_ids)}", flush=True)
    metadata = fetch_metadata_for_ids(all_ref_ids, args)
    print(f"[refs] metadata rows fetched: {len(metadata)}", flush=True)

    existing_ids: Set[str] = set()
    for d in (PROJECT_ROOT / "data" / "00_raw" / "mineru_output").glob("*"):
        if d.is_dir():
            existing_ids.add(d.name)
    for d in args.mineru_output_dir.glob("*"):
        if d.is_dir():
            existing_ids.add(d.name)

    rows: List[Dict[str, Any]] = []
    per_seed_kept: Dict[str, int] = defaultdict(int)
    row_seen: Set[str] = set()
    for seed in seed_refs:
        seed_meta = seed_rows.get(seed, {})
        refs = sorted(strip_version(r) for r in seed_refs[seed])
        ranked_refs = sorted(refs, key=lambda aid: (aid not in metadata, aid))
        for aid in ranked_refs:
            if per_seed_kept[seed] >= args.refs_per_seed:
                break
            if aid in row_seen:
                continue
            if aid in existing_ids or aid in seed_refs:
                continue
            meta = metadata.get(aid)
            if meta:
                if not is_noncs_categories(meta.get("categories", [])):
                    continue
                categories = meta.get("categories", [])
                title = meta.get("title", "")
                year = meta.get("year")
            elif not args.allow_unverified_refs:
                continue
            else:
                categories = []
                title = ""
                year = None
            rows.append(
                {
                    "arxiv_id": aid,
                    "seed_arxiv_id": seed,
                    "seed_title": seed_meta.get("title", ""),
                    "seed_topic": seed_meta.get("seed_topic", ""),
                    "title": title,
                    "year": year,
                    "categories": categories,
                    "metadata_verified": bool(meta),
                }
            )
            row_seen.add(aid)
            per_seed_kept[seed] += 1

    # Preserve at most refs_per_seed per review, then use seed diversity first.
    selected: List[Dict[str, Any]] = []
    seed_order = sorted(seed_refs.keys(), key=lambda s: len(seed_refs.get(s, ())), reverse=True)
    by_seed: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_seed[row["seed_arxiv_id"]].append(row)
    for seed in by_seed:
        by_seed[seed].sort(key=lambda r: (r.get("metadata_verified", False), r.get("year") or 0), reverse=True)

    round_idx = 0
    while len(selected) < args.candidate_limit:
        added = False
        for seed in seed_order:
            bucket = by_seed.get(seed, [])
            if round_idx < len(bucket):
                selected.append(bucket[round_idx])
                added = True
                if len(selected) >= args.candidate_limit:
                    break
        if not added:
            break
        round_idx += 1

    print(f"[refs] selected candidates: {len(selected)}", flush=True)
    print("[refs] per-seed selected counts:", flush=True)
    for seed in seed_order[: args.max_seeds]:
        cnt = sum(1 for r in selected if r["seed_arxiv_id"] == seed)
        if cnt:
            print(f"  {seed}: {cnt}", flush=True)
    return selected


def download_targets(rows: List[Dict[str, Any]], args: argparse.Namespace) -> List[Dict[str, Any]]:
    session = requests.Session()
    session.headers.update({"User-Agent": "m4-noncs400-target-download/1.0 (research)"})
    results: List[Dict[str, Any]] = []
    success_both = 0
    args.pdf_dir.mkdir(parents=True, exist_ok=True)
    args.latex_dir.mkdir(parents=True, exist_ok=True)
    start_time = time.time()

    for idx, row in enumerate(rows):
        aid = row["arxiv_id"]
        if success_both >= args.target:
            break
        if idx > 0 and idx % args.progress_interval == 0:
            elapsed = max(1.0, time.time() - start_time)
            print(
                f"[download] {idx}/{len(rows)} tried | success_both={success_both} | "
                f"rate={idx / elapsed * 3600:.1f}/hr | elapsed={elapsed / 60:.1f}min",
                flush=True,
            )

        time.sleep(args.download_delay)
        pdf_ok, pdf_msg = download_pdf(aid, args.pdf_dir, session)
        time.sleep(args.download_delay)
        latex_ok, latex_msg = download_latex(aid, args.latex_dir, session)
        out = {
            **row,
            "pdf_ok": pdf_ok,
            "pdf_msg": pdf_msg,
            "latex_ok": latex_ok,
            "latex_msg": latex_msg,
            "counted_success": bool(pdf_ok and latex_ok),
            "pdf_path": str(args.pdf_dir / f"{safe_name(aid)}.pdf"),
            "latex_extract_dir": str(args.latex_dir / "extracted" / safe_name(aid)),
        }
        results.append(out)
        if pdf_ok and latex_ok:
            success_both += 1
        if idx < 10 or (idx + 1) % args.progress_interval == 0:
            print(f"  {aid}: pdf={pdf_msg}; latex={latex_msg}; success={success_both}", flush=True)

    return results


def summarize(seed_rows: List[Dict[str, Any]], candidates: List[Dict[str, Any]], results: List[Dict[str, Any]]) -> Dict[str, Any]:
    success = [r for r in results if r.get("counted_success")]
    by_seed: Dict[str, int] = defaultdict(int)
    by_category: Dict[str, int] = defaultdict(int)
    for row in success:
        by_seed[row.get("seed_arxiv_id", "")] += 1
        for cat in row.get("categories", []):
            by_category[cat] += 1
    return {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "seed_count": len(seed_rows),
        "candidate_count": len(candidates),
        "download_attempts": len(results),
        "success_both_pdf_latex": len(success),
        "pdf_ok": sum(1 for r in results if r.get("pdf_ok")),
        "latex_ok": sum(1 for r in results if r.get("latex_ok")),
        "by_seed_success": dict(sorted(by_seed.items(), key=lambda kv: (-kv[1], kv[0]))),
        "top_categories_success": dict(sorted(by_category.items(), key=lambda kv: (-kv[1], kv[0]))[:40]),
        "seed_rows": seed_rows,
        "candidates": candidates,
        "results": results,
    }


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--target", type=int, default=400)
    ap.add_argument("--refs-per-seed", type=int, default=20)
    ap.add_argument("--max-seeds", type=int, default=35)
    ap.add_argument("--candidate-limit", type=int, default=900)
    ap.add_argument("--search-results-per-topic", type=int, default=40)
    ap.add_argument("--api-delay", type=float, default=4.0)
    ap.add_argument("--api-retries", type=int, default=5)
    ap.add_argument("--download-delay", type=float, default=3.0)
    ap.add_argument("--progress-interval", type=int, default=20)
    ap.add_argument("--seed-source-dir", type=Path, default=DEFAULT_SEED_SOURCE_DIR)
    ap.add_argument("--pdf-dir", type=Path, default=DEFAULT_PDF_DIR)
    ap.add_argument("--latex-dir", type=Path, default=DEFAULT_LATEX_DIR)
    ap.add_argument("--mineru-output-dir", type=Path, default=DEFAULT_BASE / "noncs400_mineru_output")
    ap.add_argument("--seed-manifest", type=Path, default=DEFAULT_SEED_MANIFEST)
    ap.add_argument("--ids-output", type=Path, default=DEFAULT_TARGET_IDS)
    ap.add_argument("--success-ids-output", type=Path, default=DEFAULT_SUCCESS_IDS)
    ap.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    ap.add_argument("--seed-id-file", type=Path, default=None)
    ap.add_argument("--seed-ids", nargs="*", default=None)
    ap.add_argument("--allow-unverified-refs", action="store_true")
    ap.add_argument("--search-only", action="store_true")
    ap.add_argument("--collect-only", action="store_true")
    ap.add_argument("--download-only", action="store_true")
    ap.add_argument("--no-verify", action="store_true")
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    explicit_seed_ids = load_seed_ids(args)

    if args.download_only:
        if not args.ids_output.exists():
            raise FileNotFoundError(args.ids_output)
        candidates = [{"arxiv_id": line.strip()} for line in args.ids_output.read_text(encoding="utf-8").splitlines() if line.strip()]
        seed_rows: List[Dict[str, Any]] = []
    else:
        if explicit_seed_ids:
            seed_rows = [{"arxiv_id": aid, "title": "", "categories": [], "seed_topic": "manual"} for aid in explicit_seed_ids]
        else:
            seed_rows = search_review_seeds(args)
        seed_rows = seed_rows[: args.max_seeds]
        write_json(args.seed_manifest, {"seeds": seed_rows})
        write_ids(args.seed_manifest.with_suffix(".txt"), [r["arxiv_id"] for r in seed_rows])
        print(f"[seed] selected {len(seed_rows)} review seeds", flush=True)
        for i, row in enumerate(seed_rows[:20], 1):
            print(f"  {i:02d}. {row['arxiv_id']} {row.get('categories', [])} {row.get('title', '')[:90]}", flush=True)

        if args.search_only:
            return

        seed_ids = [r["arxiv_id"] for r in seed_rows]
        seed_refs = collect_refs_from_seed_sources(seed_ids, args)
        seed_row_map = {r["arxiv_id"]: r for r in seed_rows}
        candidates = choose_target_candidates(seed_refs, seed_row_map, args)
        write_json(args.manifest.with_name(args.manifest.stem + "_candidates.json"), {"candidates": candidates})
        write_ids(args.ids_output, [r["arxiv_id"] for r in candidates])
        if args.collect_only:
            return

    results = download_targets(candidates, args)
    success_ids = [r["arxiv_id"] for r in results if r.get("counted_success")]
    write_ids(args.success_ids_output, success_ids)
    manifest = summarize(seed_rows, candidates, results)
    write_json(args.manifest, manifest)
    print("=" * 72, flush=True)
    print(f"Success with PDF+LaTeX: {len(success_ids)} / target {args.target}", flush=True)
    print(f"IDs:      {args.success_ids_output}", flush=True)
    print(f"Manifest: {args.manifest}", flush=True)
    print(f"PDF dir:  {args.pdf_dir}", flush=True)
    print(f"LaTeX:    {args.latex_dir}", flush=True)
    print("=" * 72, flush=True)
    if len(success_ids) < args.target:
        raise SystemExit(f"Only {len(success_ids)} papers with both PDF and LaTeX; target={args.target}")


if __name__ == "__main__":
    main()
