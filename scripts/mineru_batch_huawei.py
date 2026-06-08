#!/usr/bin/env python3
"""Batch submit Huawei PDFs to MinerU API (port 8001) and save results.

Resume-friendly: skips papers already in the output directory.

Usage:
    python scripts/mineru_batch_huawei.py --dry-run
    python scripts/mineru_batch_huawei.py -w 4   # 4 concurrent workers
"""

import argparse
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parent.parent
PDF_DIR = ROOT / "data" / "00_raw" / "huawei_pdfs"
OUTPUT_DIR = ROOT / "data" / "00_raw" / "huawei_mineru_output"
API_URL = "http://localhost:8001/file_parse"


def parse_one(pdf_path: str, output_dir: str) -> dict:
    """Submit one PDF to MinerU API, wait for result, save to disk."""
    arxiv_id = Path(pdf_path).stem
    out_dir = Path(output_dir) / arxiv_id

    # Skip if already done
    if out_dir.exists() and any(out_dir.rglob("*.md")):
        return {"arxiv_id": arxiv_id, "status": "skipped", "error": None}

    try:
        with open(pdf_path, "rb") as f:
            resp = requests.post(
                API_URL,
                files={"files": (f"{arxiv_id}.pdf", f, "application/pdf")},
                data={"parameters": json.dumps({
                    "parse_method": "auto",
                    "lang": "en",
                    "output_dir": str(out_dir),
                })},
                timeout=300,
            )
        resp.raise_for_status()
        result = resp.json()

        # Save raw result
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "mineru_result.json").write_text(
            json.dumps(result, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

        # Extract and save markdown content
        results = result.get("results", {})
        for fname, content in results.items():
            if isinstance(content, dict) and "md_content" in content:
                md_path = out_dir / f"{arxiv_id}.md"
                md_path.write_text(content["md_content"], encoding="utf-8")

        status = result.get("status", "unknown")
        error = result.get("error")
        return {"arxiv_id": arxiv_id, "status": status, "error": error}

    except Exception as e:
        return {"arxiv_id": arxiv_id, "status": "error", "error": str(e)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-w", "--workers", type=int, default=2,
                        help="Concurrent workers (default: 2)")
    parser.add_argument("--dry-run", action="store_true",
                        help="List PDFs without submitting")
    parser.add_argument("--limit", type=int, default=0,
                        help="Limit number of PDFs (0 = all)")
    args = parser.parse_args()

    pdf_files = sorted(PDF_DIR.glob("*.pdf"))
    existing = set()
    if OUTPUT_DIR.exists():
        for d in OUTPUT_DIR.iterdir():
            if d.is_dir() and any(d.rglob("*.md")):
                existing.add(d.name)

    todo = [str(p) for p in pdf_files if p.stem not in existing]
    if args.limit:
        todo = todo[: args.limit]

    print(f"Total PDFs: {len(pdf_files)}")
    print(f"Already parsed: {len(existing)}")
    print(f"To parse: {len(todo)}")

    if args.dry_run:
        for p in todo[:10]:
            print(f"  {Path(p).stem}")
        if len(todo) > 10:
            print(f"  ... and {len(todo) - 10} more")
        return

    if not todo:
        print("All done!")
        return

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    start = time.time()
    ok = 0
    fail = 0
    skip = 0

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {
            pool.submit(parse_one, pdf, str(OUTPUT_DIR)): Path(pdf).stem
            for pdf in todo
        }
        for i, future in enumerate(as_completed(futures)):
            r = future.result()
            if r["status"] == "completed":
                ok += 1
            elif r["status"] == "skipped":
                skip += 1
            else:
                fail += 1
                print(f"  FAIL {r['arxiv_id']}: {r.get('error','?')}")

            if (i + 1) % 20 == 0:
                elapsed = time.time() - start
                rate = (i + 1) / elapsed * 3600 if elapsed > 0 else 0
                print(f"  [{i+1}/{len(todo)}] ok={ok} fail={fail} skip={skip} "
                      f"rate={rate:.0f}/hr elapsed={elapsed/60:.1f}min")

    elapsed = time.time() - start
    print(f"\nDone: ok={ok} fail={fail} skip={skip} "
          f"in {elapsed/60:.1f}min ({len(todo)/elapsed*3600:.0f}/hr)")


if __name__ == "__main__":
    main()
