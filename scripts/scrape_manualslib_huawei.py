#!/usr/bin/env python3
"""
Scrape additional Huawei manuals from manualslib.com brand index.

Strategy:
  1. Parse the brand index page (already downloaded) for all manual links
  2. Download manual HTML pages with rate limiting
  3. Convert to markdown for downstream processing

Usage:
  python scripts/scrape_manualslib_huawei.py           # scrape all
  python scripts/scrape_manualslib_huawei.py --limit 50 # limit count
  python scripts/scrape_manualslib_huawei.py --dry-run  # list URLs only
"""

import argparse
import json
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parents[1]
BRAND_INDEX = ROOT / "data/00_raw/huawei_corpus_v2/manuals/huawei_manual_huawei_brand_index.html"
OUT_DIR = ROOT / "data/00_raw/huawei_manuals_expanded"
OUT_DIR.mkdir(parents=True, exist_ok=True)

SESSION = requests.Session()
SESSION.headers.update({
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/131.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,*/*",
    "Accept-Language": "en-US,en;q=0.9",
    "Referer": "https://www.manualslib.com/brand/huawei/",
})


def extract_manual_links(html: str) -> list[dict]:
    """Extract all manual links and product names from brand index page."""
    results = []
    # Pattern: /manual/{id}/{product-name}.html
    link_pattern = re.compile(
        r'href="(/manual/(\d+)/([^"#]+)\.html)"',
        re.IGNORECASE,
    )
    seen = set()
    for m in link_pattern.finditer(html):
        url_path = m.group(1)
        manual_id = m.group(2)
        product = m.group(3)
        full_url = f"https://www.manualslib.com{url_path}"
        if full_url not in seen:
            seen.add(full_url)
            results.append({
                "url": full_url,
                "manual_id": manual_id,
                "product_slug": product,
                "product_name": product.replace("-", " ").title(),
            })
    return results


def download_manual_page(meta: dict, out_dir: Path, timeout: int = 60) -> dict:
    """Download single manual HTML page."""
    slug = meta["product_slug"][:80]
    fname = f"huawei_manual_{meta['manual_id']}_{slug}.html"
    dest = out_dir / fname

    result = {
        "url": meta["url"],
        "filename": fname,
        "product": meta["product_name"],
        "manual_id": meta["manual_id"],
        "success": False,
        "size_bytes": 0,
        "error": None,
    }

    if dest.exists() and dest.stat().st_size > 500:
        result["success"] = True
        result["size_bytes"] = dest.stat().st_size
        result["status"] = "skipped"
        return result

    try:
        r = SESSION.get(meta["url"], timeout=timeout)
        r.raise_for_status()

        content = r.text
        if len(content) < 500:
            raise ValueError(f"Too small: {len(content)} bytes")
        if "captcha" in content.lower() or "robot" in content.lower():
            raise ValueError("Bot detection triggered")

        dest.write_text(content, encoding="utf-8")
        result["success"] = True
        result["size_bytes"] = len(content)
    except Exception as e:
        result["error"] = str(e)[:150]

    return result


def main():
    parser = argparse.ArgumentParser(description="Scrape Huawei manuals from manualslib")
    parser.add_argument("--limit", type=int, default=0, help="Max manuals to download (0=all)")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--max-workers", type=int, default=3,
                        help="Parallel workers (keep low for rate limiting)")
    parser.add_argument("--delay", type=float, default=2.0,
                        help="Delay between requests in seconds")
    args = parser.parse_args()

    # ── Extract links ──
    if not BRAND_INDEX.exists():
        print(f"ERROR: Brand index not found: {BRAND_INDEX}")
        sys.exit(1)

    html = BRAND_INDEX.read_text(encoding="utf-8")
    manuals = extract_manual_links(html)
    print(f"Found {len(manuals)} unique manual links on brand index page.")

    if args.limit > 0:
        manuals = manuals[:args.limit]
        print(f"Limited to {args.limit}.")

    if args.dry_run:
        print("\nSample manuals:")
        for m in manuals[:20]:
            print(f"  {m['product_name'][:50]:50s} → {m['url']}")
        if len(manuals) > 20:
            print(f"  ... and {len(manuals) - 20} more")
        return

    # ── Download ──
    print(f"\nDownloading {len(manuals)} manual pages...")
    print(f"Workers: {args.max_workers}, Delay: {args.delay}s\n")

    results = []
    ok = fail = skip = 0
    batch_size = 10  # process in batches to avoid overwhelming the server

    for batch_start in range(0, len(manuals), batch_size):
        batch = manuals[batch_start:batch_start + batch_size]
        with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
            future_map = {
                executor.submit(download_manual_page, m, OUT_DIR): m
                for m in batch
            }
            for future in as_completed(future_map):
                m = future_map[future]
                try:
                    r = future.result()
                    results.append(r)
                    if r.get("status") == "skipped":
                        skip += 1
                    elif r["success"]:
                        ok += 1
                        print(f"  [ok] {r['product'][:40]:40s} ({r['size_bytes']//1024}KB)")
                    else:
                        fail += 1
                        print(f"  [fail] {r['product'][:40]:40s} {r['error'][:50]}")
                except Exception as e:
                    fail += 1
                    print(f"  [err] {m['product_name'][:40]:40s} {e}")

        # Rate limiting between batches
        if batch_start + batch_size < len(manuals):
            time.sleep(args.delay * 2)

        # Progress
        total_done = ok + fail + skip
        if total_done % 50 == 0:
            print(f"  --- Progress: {total_done}/{len(manuals)} ({ok} ok, {fail} fail, {skip} skip) ---")

    # ── Summary ──
    print(f"\n{'='*50}")
    print(f"  Scrape Complete: {ok} ok, {fail} fail, {skip} skip")
    print(f"  Output: {OUT_DIR}/")
    print(f"  Files: {len(list(OUT_DIR.glob('*.html')))}")

    # Save manifest
    manifest = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "source": "manualslib.com/brand/huawei/",
        "total_found": len(manuals),
        "downloaded": ok,
        "failed": fail,
        "skipped": skip,
        "files": [
            {"filename": r["filename"], "product": r["product"],
             "manual_id": r["manual_id"], "success": r["success"],
             "size_bytes": r.get("size_bytes", 0)}
            for r in results if r["success"]
        ],
    }
    manifest_path = OUT_DIR / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False))
    print(f"  Manifest: {manifest_path}")


if __name__ == "__main__":
    main()
