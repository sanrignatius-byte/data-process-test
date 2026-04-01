#!/usr/bin/env python3
"""Snowball downloader for strict arXiv PDF+LaTeX pairs.

Features:
- Starts from survey arXiv IDs and expands on *references* (BFS).
- Global dedup by normalized arXiv ID (version stripped).
- Strict pair policy: keep a paper only if BOTH PDF and LaTeX source are valid.
- Resume-friendly checkpoint/state files.
- Metadata manifest for one-to-one pairing and provenance.

Example:
  python scripts/download_pdf_latex_pairs_snowball.py \
      --seeds 2204.09140 2401.00963 \
      --target-count 5000 \
      --output data/arxiv_pairs_snowball \
      --s2-api-key $SEMANTIC_SCHOLAR_API_KEY
"""

from __future__ import annotations

import argparse
import json
import re
import time
from collections import deque
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Deque, Dict, Iterable, List, Optional, Set, Tuple

import requests

S2_API = "https://api.semanticscholar.org/graph/v1"
ARXIV_PDF = "https://arxiv.org/pdf/{arxiv_id}.pdf"
ARXIV_EPRINT = "https://arxiv.org/e-print/{arxiv_id}"
UA = "m4-arxiv-pair-snowball/1.0"

ARXIV_ID_PATTERNS = [
    re.compile(r"^(\d{4}\.\d{4,5})(?:v\d+)?$", re.IGNORECASE),
    re.compile(r"^([a-z\-]+/\d{7})(?:v\d+)?$", re.IGNORECASE),
]


@dataclass
class QueueItem:
    arxiv_id: str
    depth: int
    parent: Optional[str] = None
    discovered_from: Optional[str] = None


@dataclass
class PairRecord:
    arxiv_id: str
    depth: int
    parent: Optional[str]
    discovered_from: Optional[str]
    pdf_path: str
    source_path: str
    collected_at: str


class SnowballPairDownloader:
    def __init__(
        self,
        output_dir: Path,
        target_count: int,
        s2_api_key: Optional[str],
        max_depth: int,
        arxiv_delay_s: float,
        s2_delay_s: float,
        max_refs_per_paper: int,
        checkpoint_every: int,
    ):
        self.output_dir = output_dir
        self.target_count = target_count
        self.max_depth = max_depth
        self.arxiv_delay_s = arxiv_delay_s
        self.s2_delay_s = s2_delay_s
        self.max_refs_per_paper = max_refs_per_paper
        self.checkpoint_every = checkpoint_every

        self.pdf_dir = output_dir / "pdfs"
        self.src_dir = output_dir / "latex_sources"
        self.state_path = output_dir / "state.json"
        self.pairs_path = output_dir / "pairs.jsonl"
        self.failed_path = output_dir / "failed_ids.txt"
        self.queue_path = output_dir / "pending_queue.jsonl"

        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.pdf_dir.mkdir(parents=True, exist_ok=True)
        self.src_dir.mkdir(parents=True, exist_ok=True)

        self.session = requests.Session()
        self.session.headers.update({"User-Agent": UA})
        if s2_api_key:
            self.session.headers.update({"x-api-key": s2_api_key})

        self._last_arxiv_call = 0.0
        self._last_s2_call = 0.0

        self.seen: Set[str] = set()
        self.enqueued: Set[str] = set()
        self.success: List[PairRecord] = []
        self.failed: Set[str] = set()

    @staticmethod
    def normalize_arxiv_id(raw: Optional[str]) -> Optional[str]:
        if not raw:
            return None
        text = str(raw).strip()
        text = text.replace("arXiv:", "").replace("ARXIV:", "")
        text = text.replace("https://arxiv.org/abs/", "").replace("http://arxiv.org/abs/", "")
        text = text.replace("https://arxiv.org/pdf/", "").replace("http://arxiv.org/pdf/", "")
        text = text.removesuffix(".pdf")
        text = re.sub(r"v\d+$", "", text)
        text = text.strip()
        for p in ARXIV_ID_PATTERNS:
            m = p.match(text)
            if m:
                return m.group(1)
        return None

    def _rate_limit(self, domain: str) -> None:
        now = time.time()
        if domain == "arxiv":
            elapsed = now - self._last_arxiv_call
            if elapsed < self.arxiv_delay_s:
                time.sleep(self.arxiv_delay_s - elapsed)
            self._last_arxiv_call = time.time()
        elif domain == "s2":
            elapsed = now - self._last_s2_call
            if elapsed < self.s2_delay_s:
                time.sleep(self.s2_delay_s - elapsed)
            self._last_s2_call = time.time()

    def _request_with_retry(self, url: str, *, params: Optional[Dict] = None, stream: bool = False, domain: str = "s2", timeout: int = 60, retries: int = 4) -> Optional[requests.Response]:
        for i in range(retries):
            self._rate_limit(domain)
            try:
                resp = self.session.get(url, params=params, timeout=timeout, stream=stream)
                if resp.status_code == 429:
                    wait = min(60, 10 * (i + 1))
                    time.sleep(wait)
                    continue
                if resp.status_code >= 500:
                    time.sleep(2 * (i + 1))
                    continue
                return resp
            except requests.RequestException:
                time.sleep(2 * (i + 1))
        return None

    def _resolve_s2_paper_id(self, arxiv_id: str) -> Optional[str]:
        url = f"{S2_API}/paper/arXiv:{arxiv_id}"
        resp = self._request_with_retry(url, params={"fields": "paperId"}, domain="s2", timeout=30)
        if not resp or resp.status_code != 200:
            return None
        try:
            return (resp.json() or {}).get("paperId")
        except ValueError:
            return None

    def fetch_reference_arxiv_ids(self, arxiv_id: str) -> List[str]:
        paper_id = self._resolve_s2_paper_id(arxiv_id)
        if not paper_id:
            return []

        out: List[str] = []
        offset = 0
        limit = min(1000, max(100, self.max_refs_per_paper))
        while len(out) < self.max_refs_per_paper:
            url = f"{S2_API}/paper/{paper_id}/references"
            params = {
                "fields": "citedPaper.externalIds",
                "limit": limit,
                "offset": offset,
            }
            resp = self._request_with_retry(url, params=params, domain="s2", timeout=60)
            if not resp or resp.status_code != 200:
                break
            try:
                data = (resp.json() or {}).get("data", [])
            except ValueError:
                break
            if not data:
                break

            for item in data:
                cited = (item or {}).get("citedPaper") or {}
                ext_ids = cited.get("externalIds") or {}
                cid = self.normalize_arxiv_id(ext_ids.get("ArXiv"))
                if cid:
                    out.append(cid)
                    if len(out) >= self.max_refs_per_paper:
                        break

            if len(data) < limit:
                break
            offset += limit

        # keep order + dedup
        uniq: List[str] = []
        seen_local: Set[str] = set()
        for x in out:
            if x not in seen_local:
                seen_local.add(x)
                uniq.append(x)
        return uniq

    def _is_probably_html(self, chunk: bytes) -> bool:
        head = chunk[:256].lower()
        return b"<!doctype html" in head or b"<html" in head

    def _download_source(self, arxiv_id: str, out_path: Path) -> bool:
        url = ARXIV_EPRINT.format(arxiv_id=arxiv_id)
        resp = self._request_with_retry(url, stream=True, domain="arxiv", timeout=120)
        if not resp or resp.status_code != 200:
            return False

        first = next(resp.iter_content(chunk_size=2048), b"")
        if not first:
            return False
        # e-print fallback to pdf/html means source unavailable
        if first.startswith(b"%PDF-") or self._is_probably_html(first):
            return False

        try:
            with out_path.open("wb") as f:
                f.write(first)
                for chunk in resp.iter_content(chunk_size=1 << 20):
                    if chunk:
                        f.write(chunk)
        except OSError:
            out_path.unlink(missing_ok=True)
            return False

        return out_path.exists() and out_path.stat().st_size > 1024

    def _download_pdf(self, arxiv_id: str, out_path: Path) -> bool:
        url = ARXIV_PDF.format(arxiv_id=arxiv_id)
        resp = self._request_with_retry(url, stream=True, domain="arxiv", timeout=120)
        if not resp or resp.status_code != 200:
            return False

        first = next(resp.iter_content(chunk_size=2048), b"")
        if not first or not first.startswith(b"%PDF-"):
            return False

        try:
            with out_path.open("wb") as f:
                f.write(first)
                for chunk in resp.iter_content(chunk_size=1 << 20):
                    if chunk:
                        f.write(chunk)
        except OSError:
            out_path.unlink(missing_ok=True)
            return False

        return out_path.exists() and out_path.stat().st_size > 1024

    def download_pair(self, arxiv_id: str) -> Tuple[bool, str]:
        pdf_path = self.pdf_dir / f"{arxiv_id}.pdf"
        src_path = self.src_dir / f"{arxiv_id}.tar.gz"

        if pdf_path.exists() and src_path.exists() and pdf_path.stat().st_size > 1024 and src_path.stat().st_size > 1024:
            return True, "already_exists"

        src_ok = self._download_source(arxiv_id, src_path)
        if not src_ok:
            src_path.unlink(missing_ok=True)
            return False, "source_missing_or_invalid"

        pdf_ok = self._download_pdf(arxiv_id, pdf_path)
        if not pdf_ok:
            pdf_path.unlink(missing_ok=True)
            src_path.unlink(missing_ok=True)
            return False, "pdf_missing_or_invalid"

        return True, "ok"

    def _save_state(self, queue: Deque[QueueItem]) -> None:
        payload = {
            "saved_at": datetime.now(timezone.utc).isoformat(),
            "target_count": self.target_count,
            "seen": sorted(self.seen),
            "failed": sorted(self.failed),
            "success_ids": [r.arxiv_id for r in self.success],
            "pending_queue_size": len(queue),
        }
        self.state_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

        with self.queue_path.open("w", encoding="utf-8") as f:
            for item in queue:
                f.write(json.dumps(asdict(item), ensure_ascii=False) + "\n")

        with self.failed_path.open("w", encoding="utf-8") as f:
            for fid in sorted(self.failed):
                f.write(fid + "\n")

    def _append_pair(self, rec: PairRecord) -> None:
        with self.pairs_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(asdict(rec), ensure_ascii=False) + "\n")

    def run(self, seeds: Iterable[str]) -> None:
        queue: Deque[QueueItem] = deque()
        for s in seeds:
            sid = self.normalize_arxiv_id(s)
            if not sid:
                continue
            if sid in self.enqueued:
                continue
            self.enqueued.add(sid)
            queue.append(QueueItem(arxiv_id=sid, depth=0, parent=None, discovered_from="seed"))

        print(f"[*] Seeds queued: {len(queue)}")
        print(f"[*] Target pairs: {self.target_count}")

        processed = 0
        while queue and len(self.success) < self.target_count:
            item = queue.popleft()
            aid = item.arxiv_id
            if aid in self.seen:
                continue
            self.seen.add(aid)
            processed += 1

            print(f"\n[->] {aid} depth={item.depth} success={len(self.success)}/{self.target_count} queue={len(queue)}")

            ok, reason = self.download_pair(aid)
            if ok:
                rec = PairRecord(
                    arxiv_id=aid,
                    depth=item.depth,
                    parent=item.parent,
                    discovered_from=item.discovered_from,
                    pdf_path=str((self.pdf_dir / f"{aid}.pdf").relative_to(self.output_dir)),
                    source_path=str((self.src_dir / f"{aid}.tar.gz").relative_to(self.output_dir)),
                    collected_at=datetime.now(timezone.utc).isoformat(),
                )
                self.success.append(rec)
                self._append_pair(rec)
                print("     [+] pair ok")
            else:
                self.failed.add(aid)
                print(f"     [-] dropped ({reason})")

            if item.depth < self.max_depth:
                refs = self.fetch_reference_arxiv_ids(aid)
                new_count = 0
                for rid in refs:
                    if rid in self.seen or rid in self.enqueued:
                        continue
                    queue.append(QueueItem(arxiv_id=rid, depth=item.depth + 1, parent=aid, discovered_from="reference"))
                    self.enqueued.add(rid)
                    new_count += 1
                print(f"     [*] expanded refs: {new_count}/{len(refs)} enqueued")

            if processed % self.checkpoint_every == 0:
                self._save_state(queue)
                print("     [*] checkpoint saved")

        self._save_state(queue)
        print("\n" + "=" * 70)
        print(f"Done. success_pairs={len(self.success)} target={self.target_count}")
        print(f"Output: {self.output_dir}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Snowball downloader for strict arXiv PDF+LaTeX pairs")
    ap.add_argument("--seeds", nargs="+", required=True, help="Seed survey arXiv IDs")
    ap.add_argument("--target-count", type=int, default=5000, help="Target number of strict pairs")
    ap.add_argument("--max-depth", type=int, default=2, help="BFS depth on reference graph")
    ap.add_argument("--max-refs-per-paper", type=int, default=500, help="Cap of references expanded per paper")
    ap.add_argument("--arxiv-delay-s", type=float, default=4.0, help="Delay between arXiv requests")
    ap.add_argument("--s2-delay-s", type=float, default=1.0, help="Delay between Semantic Scholar requests")
    ap.add_argument("--checkpoint-every", type=int, default=50, help="Save state every N processed papers")
    ap.add_argument("--s2-api-key", type=str, default=None, help="Semantic Scholar API key")
    ap.add_argument("--output", type=Path, default=Path("data/large_multimodal_dataset"))
    args = ap.parse_args()

    d = SnowballPairDownloader(
        output_dir=args.output,
        target_count=args.target_count,
        s2_api_key=args.s2_api_key,
        max_depth=args.max_depth,
        arxiv_delay_s=args.arxiv_delay_s,
        s2_delay_s=args.s2_delay_s,
        max_refs_per_paper=args.max_refs_per_paper,
        checkpoint_every=args.checkpoint_every,
    )
    d.run(args.seeds)


if __name__ == "__main__":
    main()
