#!/usr/bin/env python3
"""
Targeted re-download of LaTeX sources for PDFs that are missing them.
Reads logs/missing_latex.txt and downloads only the LaTeX source for each ID.
"""
import time
import tarfile
import re
import io
from pathlib import Path
import urllib.request
import urllib.error

PROJECT_DIR = Path(__file__).resolve().parent.parent
MISSING_IDS_FILE = PROJECT_DIR / "logs" / "missing_latex.txt"
LATEX_EXTRACTED_DIR = PROJECT_DIR / "data" / "00_raw" / "latex_sources_batch2" / "extracted"
DELAY = 4.0  # seconds between requests (arXiv rate limit)

HEADERS = {"User-Agent": "Mozilla/5.0 (compatible; research-bot/1.0; mailto:research@example.com)"}


def download_latex_source(arxiv_id: str, out_dir: Path) -> tuple[bool, str]:
    """Download and extract LaTeX source for a single arXiv ID."""
    if out_dir.exists() and any(out_dir.iterdir()):
        return True, "already extracted"

    url = f"https://arxiv.org/src/{arxiv_id}"
    try:
        req = urllib.request.Request(url, headers=HEADERS)
        with urllib.request.urlopen(req, timeout=60) as resp:
            content_type = resp.headers.get("Content-Type", "")
            data = resp.read()
    except urllib.error.HTTPError as e:
        return False, f"HTTP {e.code}"
    except Exception as e:
        return False, str(e)

    # Detect if it's a tar archive
    if not (data[:2] == b'\x1f\x8b' or data[:5] == b'ustar' or len(data) > 1000):
        return False, f"unexpected content ({len(data)} bytes, type={content_type})"

    out_dir.mkdir(parents=True, exist_ok=True)
    try:
        with tarfile.open(fileobj=io.BytesIO(data), mode="r:*") as tar:
            # Filter to safe members only
            members = []
            for m in tar.getmembers():
                # Skip absolute paths and path traversal
                if m.name.startswith("/") or ".." in m.name:
                    continue
                members.append(m)
            tar.extractall(path=out_dir, members=members)
        n_files = sum(1 for _ in out_dir.rglob("*") if _.is_file())
        return True, f"extracted {n_files} files"
    except Exception as e:
        # Might be a single .tex file (non-archive), save raw
        try:
            (out_dir / f"{arxiv_id}.tex").write_bytes(data)
            return True, "saved as single .tex"
        except Exception as e2:
            return False, f"extract failed: {e} / save failed: {e2}"


def main():
    ids = [line.strip() for line in MISSING_IDS_FILE.read_text().splitlines() if line.strip()]
    print(f"Missing LaTeX: {len(ids)} IDs")
    print(f"Output dir: {LATEX_EXTRACTED_DIR}")

    ok = fail = skip = 0
    for i, arxiv_id in enumerate(ids, 1):
        out_dir = LATEX_EXTRACTED_DIR / arxiv_id
        success, msg = download_latex_source(arxiv_id, out_dir)
        status = "ok" if success else "FAIL"
        if msg == "already extracted":
            skip += 1
            status = "skip"
        elif success:
            ok += 1
        else:
            fail += 1
        print(f"  [{i}/{len(ids)}] {arxiv_id}: {status} — {msg}")
        if msg != "already extracted":
            time.sleep(DELAY)

    print(f"\nDone: {ok} ok, {skip} skipped, {fail} failed")


if __name__ == "__main__":
    main()
