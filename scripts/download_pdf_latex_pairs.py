#!/usr/bin/env python3
"""Download arXiv PDF+LaTeX pairs only.

Goal:
- collect up to N papers
- keep only papers that have BOTH PDF and LaTeX source
- if source is missing, skip that PDF

Usage:
  python scripts/download_pdf_latex_pairs.py --count 5000 --categories cs.CL cs.LG
"""

from __future__ import annotations

import argparse
import re
import time
from pathlib import Path
from typing import Iterable, List

import feedparser
import requests

ARXIV_API = "https://export.arxiv.org/api/query"
PDF_URL = "https://arxiv.org/pdf/{arxiv_id}.pdf"
EPRINT_URL = "https://arxiv.org/e-print/{arxiv_id}"
UA = "pdf-latex-pair-downloader/1.0"


_ID_RE = re.compile(r"arxiv\.org/abs/([^v]+)(v\d+)?$")


def parse_arxiv_id(entry_id: str) -> str | None:
    m = _ID_RE.search(entry_id)
    if not m:
        return None
    return m.group(1)


def search_ids(categories: Iterable[str], max_results: int, start: int = 0) -> List[str]:
    query = " OR ".join(f"cat:{c}" for c in categories)
    params = {
        "search_query": query,
        "start": start,
        "max_results": max_results,
        "sortBy": "submittedDate",
        "sortOrder": "descending",
    }
    r = requests.get(ARXIV_API, params=params, timeout=60, headers={"User-Agent": UA})
    r.raise_for_status()
    feed = feedparser.parse(r.text)
    ids: List[str] = []
    for entry in feed.entries:
        arxiv_id = parse_arxiv_id(entry.id)
        if arxiv_id:
            ids.append(arxiv_id)
    return ids


def has_latex_source(session: requests.Session, arxiv_id: str) -> bool:
    r = session.get(EPRINT_URL.format(arxiv_id=arxiv_id), timeout=60, headers={"User-Agent": UA})
    if r.status_code != 200:
        return False
    return not r.content.startswith(b"%PDF-")


def download_file(session: requests.Session, url: str, out_path: Path) -> bool:
    r = session.get(url, timeout=120, stream=True, headers={"User-Agent": UA})
    if r.status_code != 200:
        return False
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "wb") as f:
        for chunk in r.iter_content(chunk_size=1 << 20):
            if chunk:
                f.write(chunk)
    return out_path.stat().st_size > 0


def main() -> None:
    p = argparse.ArgumentParser(description="Download only PDF+LaTeX pairs from arXiv")
    p.add_argument("--count", type=int, default=5000, help="Target number of valid pairs")
    p.add_argument("--categories", nargs="+", default=["cs.CL", "cs.LG", "cs.AI", "cs.CV"])
    p.add_argument("--batch-size", type=int, default=200, help="arXiv API page size")
    p.add_argument("--delay", type=float, default=1.5, help="Delay seconds between items")
    p.add_argument("--output", type=Path, default=Path("data/paper_pairs"))
    args = p.parse_args()

    pdf_dir = args.output / "pdfs"
    src_dir = args.output / "latex_sources"
    report = args.output / "pair_ids.txt"
    args.output.mkdir(parents=True, exist_ok=True)

    accepted: List[str] = []
    seen = set()
    start = 0

    with requests.Session() as s:
        while len(accepted) < args.count:
            ids = search_ids(args.categories, args.batch_size, start=start)
            if not ids:
                break
            start += len(ids)

            for arxiv_id in ids:
                if arxiv_id in seen:
                    continue
                seen.add(arxiv_id)
                if len(accepted) >= args.count:
                    break

                print(f"[{len(accepted)+1}/{args.count}] checking {arxiv_id}")

                if not has_latex_source(s, arxiv_id):
                    print("  -> skip: no LaTeX source")
                    time.sleep(args.delay)
                    continue

                pdf_ok = download_file(s, PDF_URL.format(arxiv_id=arxiv_id), pdf_dir / f"{arxiv_id}.pdf")
                src_ok = download_file(s, EPRINT_URL.format(arxiv_id=arxiv_id), src_dir / f"{arxiv_id}.tar.gz")

                if pdf_ok and src_ok:
                    accepted.append(arxiv_id)
                    print("  -> accepted: PDF+LaTeX saved")
                else:
                    print("  -> failed: incomplete pair, dropped")
                    (pdf_dir / f"{arxiv_id}.pdf").unlink(missing_ok=True)
                    (src_dir / f"{arxiv_id}.tar.gz").unlink(missing_ok=True)

                time.sleep(args.delay)

    report.write_text("\n".join(accepted) + ("\n" if accepted else ""), encoding="utf-8")
    print("=" * 60)
    print(f"Target pairs : {args.count}")
    print(f"Collected    : {len(accepted)}")
    print(f"PDF dir      : {pdf_dir}")
    print(f"LaTeX dir    : {src_dir}")
    print(f"IDs report   : {report}")


if __name__ == "__main__":
    main()
