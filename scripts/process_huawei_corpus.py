#!/usr/bin/env python3
"""
Huawei Corpus Pipeline: HTML → Markdown (pandoc) → Realworld Graph

Fast path that skips PDF entirely:
  1. pandoc HTML→MD (preserves structure, tables, headers)
  2. Organize in MinerU-compatible dir layout
  3. Run build_realworld_graph.py on the markdown

Usage:
  python scripts/process_huawei_corpus.py                    # full pipeline
  python scripts/process_huawei_corpus.py --skip-convert     # skip pandoc step
  python scripts/process_huawei_corpus.py --skip-graph       # skip graph build
  python scripts/process_huawei_corpus.py --category patents # single category
"""

import argparse
import json
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CORPUS_DIR = ROOT / "data" / "00_raw" / "huawei_corpus_v2"

# Output dirs — mimic MinerU structure: {category}/{doc_stem}/{doc_stem}.md
MD_OUT_BASE = ROOT / "data" / "00_raw" / "huawei_corpus_md"
GRAPH_OUT = ROOT / "data" / "01_graphs" / "huawei_multimodal_elements.json"
MANIFEST_FILE = MD_OUT_BASE / "pipeline_manifest.json"

# ──────────────────────────────────────────────────────────────────────
# Step 1: HTML → Markdown via pandoc
# ──────────────────────────────────────────────────────────────────────

def html_to_md(html_path: Path, md_dir: Path) -> dict:
    """Convert single HTML to Markdown using pandoc.

    Output layout (MinerU-compatible):
      md_dir/{doc_stem}/
        {doc_stem}.md
        pipeline_meta.json
    """
    doc_stem = html_path.stem
    out_subdir = md_dir / doc_stem
    out_subdir.mkdir(parents=True, exist_ok=True)
    md_path = out_subdir / f"{doc_stem}.md"

    result = {
        "html_file": html_path.name,
        "md_dir": str(out_subdir),
        "success": False,
        "size_bytes": 0,
        "error": None,
    }

    # Skip already converted
    if md_path.exists() and md_path.stat().st_size > 100:
        result["success"] = True
        result["size_bytes"] = md_path.stat().st_size
        result["status"] = "skipped"
        return result

    try:
        proc = subprocess.run(
            [
                "pandoc", str(html_path),
                "-f", "html",
                "-t", "gfm-raw_html",
                "--wrap=none",
                "-o", str(md_path),
            ],
            capture_output=True, text=True, timeout=60,
        )
        if proc.returncode != 0:
            raise RuntimeError(proc.stderr[:200] if proc.stderr else f"exit {proc.returncode}")

        size = md_path.stat().st_size
        if size < 100:
            raise RuntimeError(f"Output too small: {size} bytes")

        result["success"] = True
        result["size_bytes"] = size

        # Save metadata for traceability
        meta = {
            "source_html": str(html_path),
            "converted_at": datetime.now(timezone.utc).isoformat(),
            "pandoc_args": ["-f", "html", "-t", "gfm-raw_html"],
            "html_size": html_path.stat().st_size,
            "md_size": size,
        }
        (out_subdir / "pipeline_meta.json").write_text(
            json.dumps(meta, indent=2, ensure_ascii=False))

    except Exception as e:
        result["error"] = str(e)[:200]
        # Create empty marker so we don't retry
        if not md_path.exists():
            md_path.write_text(f"# Conversion Failed\n\nError: {e}\n")

    return result


def convert_all_html(
    corpus_dir: Path = CORPUS_DIR,
    md_base: Path = MD_OUT_BASE,
    max_workers: int = 4,
    dry_run: bool = False,
    categories: list[str] = None,
) -> dict:
    """Convert all HTML to Markdown, organized by category."""
    md_base.mkdir(parents=True, exist_ok=True)

    html_files = {}
    for cat_dir in sorted(corpus_dir.iterdir()):
        if not cat_dir.is_dir():
            continue
        if categories and cat_dir.name not in categories:
            continue
        htmls = sorted(cat_dir.glob("*.html"))
        if htmls:
            html_files[cat_dir.name] = htmls

    total = sum(len(v) for v in html_files.values())
    if total == 0:
        print("  No HTML files found.")
        return {}

    if dry_run:
        print(f"\n  [DRY RUN] Would convert {total} HTML → Markdown:")
        for cat, files in html_files.items():
            print(f"    {cat}: {len(files)} files")
        return {}

    print(f"\n  Converting {total} HTML → Markdown (pandoc)...")
    all_results = {}
    total_ok = total_fail = total_skip = 0

    for cat_name, html_list in html_files.items():
        cat_md_dir = md_base / cat_name
        print(f"\n  [{cat_name}] {len(html_list)} files...")
        results = []
        ok = fail = skip = 0

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_map = {
                executor.submit(html_to_md, hf, cat_md_dir): hf.name
                for hf in html_list
            }
            for future in as_completed(future_map):
                name = future_map[future]
                try:
                    r = future.result()
                    results.append(r)
                    if r.get("status") == "skipped":
                        skip += 1
                    elif r["success"]:
                        ok += 1
                        print(f"    [ok] {name} → {r['size_bytes']//1024}KB MD")
                    else:
                        fail += 1
                        print(f"    [fail] {name}: {r['error'][:60]}")
                except Exception as e:
                    fail += 1
                    print(f"    [err] {name}: {e}")

        all_results[cat_name] = results
        total_ok += ok
        total_fail += fail
        total_skip += skip
        print(f"    {cat_name}: {ok} ok, {skip} skip, {fail} fail")

    print(f"\n  HTML→MD Total: {total_ok} ok, {total_skip} skip, {total_fail} fail")
    return all_results


# ──────────────────────────────────────────────────────────────────────
# Step 2: Build realworld graph from generated Markdown
# ──────────────────────────────────────────────────────────────────────

def build_graph(
    md_dir: Path = MD_OUT_BASE,
    graph_out: Path = GRAPH_OUT,
    dry_run: bool = False,
) -> dict:
    """Run build_realworld_graph.py on the converted Markdown files."""
    script = ROOT / "scripts" / "build_realworld_graph.py"

    if not script.exists():
        print(f"  ERROR: Graph builder not found: {script}")
        return {"success": False, "error": "script not found"}

    if dry_run:
        md_count = sum(1 for _ in md_dir.rglob("*.md"))
        print(f"\n  [DRY RUN] Would build graph from {md_count} markdown files")
        print(f"    Input:  {md_dir}")
        print(f"    Output: {graph_out}")
        return {}

    # Count MD files
    md_count = sum(1 for _ in md_dir.rglob("*.md"))
    print(f"\n  Building realworld graph from {md_count} markdown files...")
    print(f"  This runs the same pipeline as realworld documents...")

    try:
        proc = subprocess.run(
            [
                sys.executable, str(script),
                "--mineru-dir", str(md_dir),
                "--output", str(graph_out),
            ],
            capture_output=True, text=True, timeout=300,
            cwd=str(ROOT),
        )
        if proc.returncode != 0:
            raise RuntimeError(proc.stderr[:500] if proc.stderr else f"exit {proc.returncode}")

        # Check output
        if graph_out.exists():
            graph_size = graph_out.stat().st_size
            graph_data = json.loads(graph_out.read_text())
            nodes = len(graph_data.get("nodes", []))
            edges = len(graph_data.get("edges", []))
            print(f"    Graph built: {nodes} nodes, {edges} edges ({graph_size//1024}KB)")
            return {"success": True, "nodes": nodes, "edges": edges, "size": graph_size}
        else:
            print(f"    WARNING: Graph output not found at {graph_out}")
            print(f"    stdout: {proc.stdout[:500]}")
            return {"success": False, "error": "output not found"}

    except Exception as e:
        print(f"    ERROR: {e}")
        return {"success": False, "error": str(e)}


# ──────────────────────────────────────────────────────────────────────
# Pipeline manifest
# ──────────────────────────────────────────────────────────────────────

def save_manifest(convert_results: dict, graph_result: dict):
    manifest = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "source": str(CORPUS_DIR),
        "markdown_output": str(MD_OUT_BASE),
        "graph_output": str(GRAPH_OUT),
        "stage1_html_to_md": {},
        "stage2_graph": {
            "success": graph_result.get("success", False),
            "nodes": graph_result.get("nodes", 0),
            "edges": graph_result.get("edges", 0),
        },
    }

    for cat, results in convert_results.items():
        ok = sum(1 for r in results if r.get("success"))
        fail = sum(1 for r in results if not r.get("success"))
        size = sum(r.get("size_bytes", 0) for r in results if r.get("success"))
        manifest["stage1_html_to_md"][cat] = {
            "total": len(results), "success": ok, "failed": fail,
            "total_size_mb": round(size / 1024 / 1024, 2),
        }

    MANIFEST_FILE.write_text(json.dumps(manifest, indent=2, ensure_ascii=False))
    print(f"\n  Pipeline manifest: {MANIFEST_FILE}")


# ──────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Huawei corpus: HTML→Markdown→Graph pipeline"
    )
    parser.add_argument("--skip-convert", action="store_true",
                        help="Skip HTML→MD conversion")
    parser.add_argument("--skip-graph", action="store_true",
                        help="Skip graph building")
    parser.add_argument("--category", nargs="*",
                        help="Specific categories to process")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--max-workers", type=int, default=8,
                        help="Parallel pandoc workers (default: 8)")
    args = parser.parse_args()

    categories = args.category if args.category else None
    convert_results = {}
    graph_result = {}

    # ── Stage 1: HTML → Markdown ──
    if not args.skip_convert:
        print("=" * 60)
        print("  STAGE 1: HTML → Markdown (pandoc)")
        print("=" * 60)
        convert_results = convert_all_html(
            corpus_dir=CORPUS_DIR,
            md_base=MD_OUT_BASE,
            max_workers=args.max_workers,
            dry_run=args.dry_run,
            categories=categories,
        )

    # ── Stage 2: Build Graph ──
    if not args.skip_graph and not args.dry_run:
        print("\n" + "=" * 60)
        print("  STAGE 2: Build Realworld Graph")
        print("=" * 60)
        graph_result = build_graph(
            md_dir=MD_OUT_BASE,
            graph_out=GRAPH_OUT,
            dry_run=args.dry_run,
        )

    # ── Manifest ──
    if not args.dry_run and convert_results:
        save_manifest(convert_results, graph_result)

    # ── Summary ──
    if not args.dry_run:
        print("\n" + "=" * 60)
        print("  PIPELINE COMPLETE")
        print("=" * 60)
        if convert_results:
            total_ok = sum(
                sum(1 for r in recs if r.get("success"))
                for recs in convert_results.values()
            )
            total_all = sum(len(recs) for recs in convert_results.values())
            print(f"  HTML→MD:  {total_ok}/{total_all} files converted")
        if graph_result:
            print(f"  Graph:    {graph_result.get('nodes', 0)} nodes, "
                  f"{graph_result.get('edges', 0)} edges")
        print(f"\n  Markdown:  {MD_OUT_BASE}/")
        print(f"  Graph:     {GRAPH_OUT}")


if __name__ == "__main__":
    main()
