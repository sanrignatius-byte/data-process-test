#!/usr/bin/env python3
"""Snowball downloader for strict arXiv PDF+LaTeX pairs.

Features:
- BFS expansion on *references* starting from seed survey arXiv IDs.
- Global dedup by normalized arXiv ID (version stripped).
- Strict pair policy: keep only if BOTH PDF and LaTeX source are valid.
- Crash-safe resume: saves seen/enqueued/queue/success count atomically.
- Metadata manifest (pairs.jsonl) is append-only; no duplicate entries.
- Magic-byte validation for both PDF (%PDF-) and source (not PDF/HTML).

Example:
  python scripts/download_pdf_latex_pairs_snowball.py \\
      --seeds 2204.09140 2401.00963 \\
      --target-count 5000 \\
      --output data/arxiv_pairs_snowball \\
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

# ---------------------------------------------------------------------------
S2_API = "https://api.semanticscholar.org/graph/v1"
ARXIV_PDF = "https://arxiv.org/pdf/{arxiv_id}.pdf"
ARXIV_EPRINT = "https://arxiv.org/e-print/{arxiv_id}"
UA = "m4-arxiv-pair-snowball/1.1"

ARXIV_ID_RE = [
    re.compile(r"^(\d{4}\.\d{4,5})(?:v\d+)?$", re.IGNORECASE),
    re.compile(r"^([a-z\-]+/\d{7})(?:v\d+)?$", re.IGNORECASE),
]


# ---------------------------------------------------------------------------
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


# ---------------------------------------------------------------------------
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
        expand_failed: bool,
    ):
        self.output_dir = output_dir
        self.target_count = target_count
        self.max_depth = max_depth
        self.arxiv_delay_s = arxiv_delay_s
        self.s2_delay_s = s2_delay_s
        self.max_refs_per_paper = max_refs_per_paper
        self.checkpoint_every = checkpoint_every
        self.expand_failed = expand_failed

        self.pdf_dir = output_dir / "pdfs"
        self.src_dir = output_dir / "latex_sources"
        self.state_path = output_dir / "state.json"
        self.pairs_path = output_dir / "pairs.jsonl"
        self.queue_path = output_dir / "pending_queue.jsonl"
        self.failed_path = output_dir / "failed_ids.txt"

        for d in (output_dir, self.pdf_dir, self.src_dir):
            d.mkdir(parents=True, exist_ok=True)

        self.session = requests.Session()
        self.session.headers.update({"User-Agent": UA})
        if s2_api_key:
            self.session.headers.update({"x-api-key": s2_api_key})

        self._last_arxiv_call = 0.0
        self._last_s2_call = 0.0

        # Runtime state — populated by _load_state() before run()
        self.seen: Set[str] = set()
        self.enqueued: Set[str] = set()
        self.success_ids: Set[str] = set()   # IDs already in pairs.jsonl
        self.failed: Set[str] = set()
        self.success_count: int = 0

    # ------------------------------------------------------------------
    # ID normalisation
    # ------------------------------------------------------------------

    @staticmethod
    def normalize(raw: Optional[str]) -> Optional[str]:
        if not raw:
            return None
        t = str(raw).strip()
        for prefix in ("arXiv:", "ARXIV:", "https://arxiv.org/abs/",
                        "http://arxiv.org/abs/", "https://arxiv.org/pdf/",
                        "http://arxiv.org/pdf/"):
            if t.startswith(prefix):
                t = t[len(prefix):]
        t = t.removesuffix(".pdf")
        t = re.sub(r"v\d+$", "", t).strip()
        for pat in ARXIV_ID_RE:
            m = pat.match(t)
            if m:
                return m.group(1)
        return None

    # ------------------------------------------------------------------
    # Checkpoint  (save + load)
    # ------------------------------------------------------------------

    def _save_state(self, queue: Deque[QueueItem]) -> None:
        """Atomically persist all mutable state."""
        payload = {
            "saved_at": datetime.now(timezone.utc).isoformat(),
            "target_count": self.target_count,
            "success_count": self.success_count,
            "seen": sorted(self.seen),
            "enqueued": sorted(self.enqueued),
            "success_ids": sorted(self.success_ids),
            "failed": sorted(self.failed),
        }
        # Atomic write: tmp → replace
        tmp = self.state_path.with_suffix(".tmp")
        tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        tmp.replace(self.state_path)

        # Queue snapshot (not critical to be atomic — state.json is the source of truth)
        with self.queue_path.open("w", encoding="utf-8") as f:
            for item in queue:
                f.write(json.dumps(asdict(item), ensure_ascii=False) + "\n")

        with self.failed_path.open("w", encoding="utf-8") as f:
            for fid in sorted(self.failed):
                f.write(fid + "\n")

    def _load_state(self) -> Deque[QueueItem]:
        """Restore runtime state from disk. Returns the pending queue."""
        queue: Deque[QueueItem] = deque()

        if self.state_path.exists():
            try:
                payload = json.loads(self.state_path.read_text(encoding="utf-8"))
                self.seen = set(payload.get("seen", []))
                self.enqueued = set(payload.get("enqueued", []))
                self.success_ids = set(payload.get("success_ids", []))
                self.failed = set(payload.get("failed", []))
                self.success_count = int(payload.get("success_count", 0))
                print(f"[resume] state loaded: seen={len(self.seen)} "
                      f"success={self.success_count} failed={len(self.failed)}")
            except Exception as exc:
                print(f"[warn] could not parse state.json ({exc}), starting fresh")

        if self.queue_path.exists():
            try:
                for line in self.queue_path.read_text(encoding="utf-8").splitlines():
                    line = line.strip()
                    if not line:
                        continue
                    d = json.loads(line)
                    item = QueueItem(**d)
                    if item.arxiv_id not in self.seen:
                        queue.append(item)
                print(f"[resume] queue restored: {len(queue)} items")
            except Exception as exc:
                print(f"[warn] could not restore queue ({exc})")

        return queue

    def _append_pair(self, rec: PairRecord) -> None:
        with self.pairs_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(asdict(rec), ensure_ascii=False) + "\n")

    # ------------------------------------------------------------------
    # Rate limiting
    # ------------------------------------------------------------------

    def _wait(self, domain: str) -> None:
        delay = self.arxiv_delay_s if domain == "arxiv" else self.s2_delay_s
        attr = f"_last_{domain}_call"
        elapsed = time.time() - getattr(self, attr)
        if elapsed < delay:
            time.sleep(delay - elapsed)
        setattr(self, attr, time.time())

    # ------------------------------------------------------------------
    # HTTP with retry
    # ------------------------------------------------------------------

    def _get(
        self,
        url: str,
        *,
        params: Optional[Dict] = None,
        stream: bool = False,
        domain: str = "s2",
        timeout: int = 60,
        retries: int = 4,
    ) -> Optional[requests.Response]:
        for attempt in range(retries):
            self._wait(domain)
            try:
                resp = self.session.get(url, params=params, timeout=timeout, stream=stream)
                if resp.status_code == 429:
                    time.sleep(min(60, 10 * (attempt + 1)))
                    continue
                if resp.status_code >= 500:
                    time.sleep(2 ** attempt)
                    continue
                return resp
            except requests.RequestException:
                time.sleep(2 ** attempt)
        return None

    # ------------------------------------------------------------------
    # Semantic Scholar reference expansion
    # ------------------------------------------------------------------

    def _s2_paper_id(self, arxiv_id: str) -> Optional[str]:
        resp = self._get(
            f"{S2_API}/paper/arXiv:{arxiv_id}",
            params={"fields": "paperId"},
            domain="s2",
            timeout=30,
        )
        if not resp or resp.status_code != 200:
            return None
        try:
            return (resp.json() or {}).get("paperId")
        except ValueError:
            return None

    def fetch_reference_arxiv_ids(self, arxiv_id: str) -> List[str]:
        paper_id = self._s2_paper_id(arxiv_id)
        if not paper_id:
            return []

        out: List[str] = []
        seen_local: Set[str] = set()
        offset, limit = 0, min(1000, max(100, self.max_refs_per_paper))

        while len(out) < self.max_refs_per_paper:
            resp = self._get(
                f"{S2_API}/paper/{paper_id}/references",
                params={"fields": "citedPaper.externalIds", "limit": limit, "offset": offset},
                domain="s2",
                timeout=60,
            )
            if not resp or resp.status_code != 200:
                break
            try:
                data = (resp.json() or {}).get("data", [])
            except ValueError:
                break
            if not data:
                break

            for item in data:
                ext = ((item or {}).get("citedPaper") or {}).get("externalIds") or {}
                cid = self.normalize(ext.get("ArXiv"))
                if cid and cid not in seen_local:
                    seen_local.add(cid)
                    out.append(cid)
                    if len(out) >= self.max_refs_per_paper:
                        break

            if len(data) < limit:
                break
            offset += limit

        return out

    # ------------------------------------------------------------------
    # File downloads
    # ------------------------------------------------------------------

    @staticmethod
    def _is_html(chunk: bytes) -> bool:
        head = chunk[:256].lower()
        return b"<!doctype html" in head or b"<html" in head

    def _stream_to_file(
        self,
        resp: requests.Response,
        path: Path,
        first_chunk: bytes,
    ) -> bool:
        try:
            with path.open("wb") as f:
                f.write(first_chunk)
                for chunk in resp.iter_content(chunk_size=1 << 20):
                    if chunk:
                        f.write(chunk)
        except OSError:
            path.unlink(missing_ok=True)
            return False
        return path.exists() and path.stat().st_size > 1024

    def _download_source(self, arxiv_id: str, out_path: Path) -> bool:
        """Download LaTeX e-print. Rejects if it's a PDF or HTML page."""
        resp = self._get(
            ARXIV_EPRINT.format(arxiv_id=arxiv_id),
            stream=True,
            domain="arxiv",
            timeout=120,
        )
        if not resp or resp.status_code != 200:
            return False
        first = next(resp.iter_content(chunk_size=4096), b"")
        if not first:
            return False
        if first.startswith(b"%PDF-") or self._is_html(first):
            return False   # no LaTeX source available
        return self._stream_to_file(resp, out_path, first)

    def _download_pdf(self, arxiv_id: str, out_path: Path) -> bool:
        resp = self._get(
            ARXIV_PDF.format(arxiv_id=arxiv_id),
            stream=True,
            domain="arxiv",
            timeout=120,
        )
        if not resp or resp.status_code != 200:
            return False
        first = next(resp.iter_content(chunk_size=4096), b"")
        if not first or not first.startswith(b"%PDF-"):
            return False
        return self._stream_to_file(resp, out_path, first)

    def download_pair(self, arxiv_id: str) -> Tuple[bool, str]:
        """
        Download source first (fail-fast), then PDF.
        Returns (success, reason_string).
        Both files are deleted if either fails (atomic pair guarantee).
        """
        pdf_path = self.pdf_dir / f"{arxiv_id}.pdf"
        src_path = self.src_dir / f"{arxiv_id}.tar.gz"

        # Files already present and valid — skip download, DON'T re-record.
        if (pdf_path.exists() and src_path.exists()
                and pdf_path.stat().st_size > 1024
                and src_path.stat().st_size > 1024):
            return True, "already_exists"

        # Source first — most papers lack LaTeX; fail fast.
        if not self._download_source(arxiv_id, src_path):
            src_path.unlink(missing_ok=True)
            return False, "source_missing_or_invalid"

        if not self._download_pdf(arxiv_id, pdf_path):
            pdf_path.unlink(missing_ok=True)
            src_path.unlink(missing_ok=True)
            return False, "pdf_missing_or_invalid"

        return True, "ok"

    # ------------------------------------------------------------------
    # Main BFS loop
    # ------------------------------------------------------------------

    def run(self, seeds: Iterable[str]) -> None:
        # Restore previous run if any
        queue = self._load_state()

        # Enqueue seeds only if this is a fresh run (queue empty after load)
        if not queue and not self.seen:
            for s in seeds:
                sid = self.normalize(s)
                if sid and sid not in self.enqueued:
                    self.enqueued.add(sid)
                    queue.append(QueueItem(arxiv_id=sid, depth=0, discovered_from="seed"))
            print(f"[*] Fresh run — seeds queued: {len(queue)}")
        else:
            print(f"[*] Resuming — queue size: {len(queue)}")

        print(f"[*] Target pairs: {self.target_count}  already collected: {self.success_count}")

        processed = 0
        while queue and self.success_count < self.target_count:
            item = queue.popleft()
            aid = item.arxiv_id

            if aid in self.seen:
                continue
            self.seen.add(aid)

            processed += 1
            print(
                f"\n[->] {aid}  depth={item.depth}  "
                f"ok={self.success_count}/{self.target_count}  queue={len(queue)}"
            )

            ok, reason = self.download_pair(aid)

            if ok:
                if aid not in self.success_ids:
                    # New pair — record it
                    rel_pdf = str((self.pdf_dir / f"{aid}.pdf").relative_to(self.output_dir))
                    rel_src = str((self.src_dir / f"{aid}.tar.gz").relative_to(self.output_dir))
                    rec = PairRecord(
                        arxiv_id=aid,
                        depth=item.depth,
                        parent=item.parent,
                        discovered_from=item.discovered_from,
                        pdf_path=rel_pdf,
                        source_path=rel_src,
                        collected_at=datetime.now(timezone.utc).isoformat(),
                    )
                    self._append_pair(rec)
                    self.success_ids.add(aid)
                    self.success_count += 1
                    print(f"     [+] pair recorded ({self.success_count})")
                else:
                    print(f"     [=] already recorded, skipping append")
            else:
                self.failed.add(aid)
                print(f"     [-] dropped ({reason})")

            # Reference expansion
            should_expand = (
                item.depth < self.max_depth
                and (ok or self.expand_failed)
            )
            if should_expand:
                refs = self.fetch_reference_arxiv_ids(aid)
                new_count = 0
                for rid in refs:
                    if rid not in self.seen and rid not in self.enqueued:
                        queue.append(QueueItem(
                            arxiv_id=rid,
                            depth=item.depth + 1,
                            parent=aid,
                            discovered_from="reference",
                        ))
                        self.enqueued.add(rid)
                        new_count += 1
                print(f"     [*] refs expanded: {new_count}/{len(refs)}")

            if processed % self.checkpoint_every == 0:
                self._save_state(queue)
                print("     [checkpoint saved]")

        # Final save
        self._save_state(queue)
        print("\n" + "=" * 70)
        print(f"Done.  success_pairs={self.success_count}  target={self.target_count}")
        print(f"Output: {self.output_dir}")


# ---------------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser(
        description="Snowball downloader for strict arXiv PDF+LaTeX pairs"
    )
    ap.add_argument("--seeds", nargs="+", required=True,
                    help="Seed survey arXiv IDs")
    ap.add_argument("--target-count", type=int, default=5000,
                    help="Target number of strict pairs (default: 5000)")
    ap.add_argument("--max-depth", type=int, default=2,
                    help="BFS depth on reference graph (default: 2)")
    ap.add_argument("--max-refs-per-paper", type=int, default=500,
                    help="Cap of references expanded per paper (default: 500)")
    ap.add_argument("--arxiv-delay-s", type=float, default=4.0,
                    help="Min seconds between arXiv requests (default: 4.0)")
    ap.add_argument("--s2-delay-s", type=float, default=6.0,
                    help="Min seconds between S2 requests (default: 6.0, ~10/min without key)")
    ap.add_argument("--checkpoint-every", type=int, default=50,
                    help="Save state every N processed papers (default: 50)")
    ap.add_argument("--expand-failed", action="store_true",
                    help="Also expand references of papers with no LaTeX source")
    ap.add_argument("--s2-api-key", type=str, default=None,
                    help="Semantic Scholar API key (raises rate limit)")
    ap.add_argument("--output", type=Path, default=Path("data/arxiv_pairs_snowball"),
                    help="Output directory")
    args = ap.parse_args()

    downloader = SnowballPairDownloader(
        output_dir=args.output,
        target_count=args.target_count,
        s2_api_key=args.s2_api_key,
        max_depth=args.max_depth,
        arxiv_delay_s=args.arxiv_delay_s,
        s2_delay_s=args.s2_delay_s,
        max_refs_per_paper=args.max_refs_per_paper,
        checkpoint_every=args.checkpoint_every,
        expand_failed=args.expand_failed,
    )
    downloader.run(args.seeds)


if __name__ == "__main__":
    main()
