#!/usr/bin/env python3
"""Batch submit remaining realworld PDFs to MinerU API (port 8001).

Resume-friendly: skips papers already parsed.
"""

import json, os, sys, time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import requests

ROOT = Path(__file__).resolve().parent.parent
PDF_DIR = ROOT / "data" / "00_raw" / "realworld_pdfs"
OUTPUT_DIR = ROOT / "data" / "00_raw" / "realworld_mineru_output"
API_URL = "http://localhost:8001/file_parse"


def parse_one(pdf_path: str, output_dir: str) -> dict:
    """Submit one PDF to MinerU API and save results."""
    pdf_name = Path(pdf_path).stem
    out_dir = Path(output_dir) / pdf_name

    # Skip if already done (markdown file exists)
    if out_dir.exists() and (out_dir / f"{pdf_name}.md").exists():
        return {"name": pdf_name, "status": "skipped", "error": None}

    try:
        file_size_kb = Path(pdf_path).stat().st_size / 1024
        with open(pdf_path, "rb") as f:
            resp = requests.post(
                API_URL,
                files={"files": (f"{pdf_name}.pdf", f, "application/pdf")},
                data={"parameters": json.dumps({
                    "parse_method": "auto",
                    "lang": "en",
                })},
                timeout=600,  # longer timeout for large files
            )
        resp.raise_for_status()
        result = resp.json()

        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "mineru_result.json").write_text(
            json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")

        # Save markdown
        results = result.get("results", {})
        for fname, content in results.items():
            if isinstance(content, dict) and "md_content" in content:
                (out_dir / f"{pdf_name}.md").write_text(
                    content["md_content"], encoding="utf-8")

        return {"name": pdf_name, "status": "ok", "error": None, "size_kb": file_size_kb}

    except Exception as e:
        return {"name": pdf_name, "status": "error", "error": str(e)[:200], "size_kb": 0}


def main():
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-w", "--workers", type=int, default=3,
                        help="Concurrent workers (default: 3)")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--retry-failed", action="store_true",
                        help="Re-parse failed entries (no .md file)")
    args = parser.parse_args()

    pdf_files = sorted(PDF_DIR.glob("*.pdf"))
    existing = set()
    if OUTPUT_DIR.exists():
        for d in OUTPUT_DIR.iterdir():
            if d.is_dir() and (d / f"{d.name}.md").exists():
                existing.add(d.name)
    if not args.retry_failed:
        # Also count dirs with mineru_result.json but no .md as failed
        for d in OUTPUT_DIR.iterdir():
            if d.is_dir() and not (d / f"{d.name}.md").exists():
                pass  # will be retried

    todo = [str(p) for p in pdf_files if p.stem not in existing]
    
    total_size_mb = sum(Path(p).stat().st_size for p in pdf_files if str(p) in todo) / 1024 / 1024

    print(f"Total PDFs: {len(pdf_files)}")
    print(f"Already parsed: {len(existing)}")
    print(f"To parse: {len(todo)} ({total_size_mb:.1f} MB)")
    print(f"Concurrency: {args.workers}")
    
    if args.dry_run:
        for p in todo[:20]:
            sz = Path(p).stat().st_size / 1024
            print(f"  {Path(p).stem} ({sz:.0f}KB)")
        return

    if not todo:
        print("All done!")
        return

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    start = time.time()
    ok = fail = skip = 0

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {
            pool.submit(parse_one, pdf, str(OUTPUT_DIR)): Path(pdf).stem
            for pdf in todo
        }
        for i, future in enumerate(as_completed(futures)):
            r = future.result()
            name = r["name"]
            if r["status"] == "ok":
                ok += 1
                print(f"  ✅ {name} ({r.get('size_kb',0):.0f}KB)")
            elif r["status"] == "skipped":
                skip += 1
            else:
                fail += 1
                print(f"  ❌ {name}: {r.get('error', '?')}")

            if (i + 1) % 5 == 0:
                elapsed = time.time() - start
                rate = (i + 1) / elapsed * 3600 if elapsed > 0 else 0
                print(f"  [{i+1}/{len(todo)}] ok={ok} fail={fail} rate={rate:.0f}/hr elapsed={elapsed/60:.1f}min")

    elapsed = time.time() - start
    print(f"\nDone: ok={ok} fail={fail} skip={skip} in {elapsed/60:.1f}min")


if __name__ == "__main__":
    main()
