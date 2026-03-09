#!/usr/bin/env python3
"""Diagnose alignment between MinerU output and LaTeX sources.

When MinerU results come from a cluster and LaTeX sources are downloaded
separately on a company machine, the two sets may not fully overlap.
This script reports the intersection, differences, and per-doc health.

Usage:
    python scripts/diagnose_data_alignment.py
    python scripts/diagnose_data_alignment.py \
        --mineru-dir data/mineru_output \
        --latex-dir data/latex_sources/extracted \
        --output data/data_alignment_report.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Set


PROJECT_ROOT = Path(__file__).resolve().parent.parent


def normalize_doc_id(name: str) -> str:
    """Normalize directory name to arxiv-style doc_id (dots)."""
    # LaTeX dirs often use underscores: 1908_09635 -> 1908.09635
    # MinerU dirs may use dots or underscores depending on source
    return name.replace("_", ".", 1)


def scan_mineru(mineru_dir: Path) -> Dict[str, Dict[str, Any]]:
    """Scan MinerU output directory, return {doc_id: info}."""
    results = {}
    if not mineru_dir.exists():
        return results
    for d in sorted(mineru_dir.iterdir()):
        if not d.is_dir():
            continue
        doc_id = d.name
        # Check for content_list.json (needed for page_idx mapping)
        content_list = None
        for candidate in [
            d / doc_id / "hybrid_auto" / "content_list.json",
            d / doc_id / "content_list.json",
            d / "content_list.json",
        ]:
            if candidate.exists():
                content_list = str(candidate.relative_to(mineru_dir))
                break
        # Check for images
        images_dir = None
        image_count = 0
        for candidate in [
            d / doc_id / "hybrid_auto" / "images",
            d / doc_id / "images",
            d / "images",
        ]:
            if candidate.is_dir():
                images_dir = str(candidate.relative_to(mineru_dir))
                image_count = len(list(candidate.glob("*.jpg")) + list(candidate.glob("*.png")))
                break
        # Check for multimodal_elements.json
        mm_elements = None
        for candidate in [
            d / doc_id / "hybrid_auto" / "multimodal_elements.json",
            d / doc_id / "multimodal_elements.json",
            d / "multimodal_elements.json",
        ]:
            if candidate.exists():
                mm_elements = str(candidate.relative_to(mineru_dir))
                break

        results[doc_id] = {
            "path": str(d.relative_to(mineru_dir)),
            "has_content_list": content_list is not None,
            "content_list_path": content_list,
            "has_images": images_dir is not None,
            "image_count": image_count,
            "has_multimodal_elements": mm_elements is not None,
        }
    return results


def scan_latex(latex_dir: Path) -> Dict[str, Dict[str, Any]]:
    """Scan LaTeX sources directory, return {normalized_doc_id: info}."""
    results = {}
    if not latex_dir.exists():
        return results
    for d in sorted(latex_dir.iterdir()):
        if not d.is_dir():
            continue
        raw_name = d.name
        doc_id = normalize_doc_id(raw_name)
        # Check for .tex files
        tex_files = list(d.rglob("*.tex"))
        # Check for .bbl files (bibliography)
        bbl_files = list(d.rglob("*.bbl"))
        results[doc_id] = {
            "path": str(d.relative_to(latex_dir)),
            "raw_dirname": raw_name,
            "tex_count": len(tex_files),
            "has_bbl": len(bbl_files) > 0,
        }
    return results


def main():
    ap = argparse.ArgumentParser(description="Diagnose MinerU-LaTeX data alignment")
    ap.add_argument("--mineru-dir", type=Path,
                    default=PROJECT_ROOT / "data" / "mineru_output")
    ap.add_argument("--latex-dir", type=Path,
                    default=PROJECT_ROOT / "data" / "latex_sources" / "extracted")
    ap.add_argument("--output", type=Path,
                    default=PROJECT_ROOT / "data" / "data_alignment_report.json")
    args = ap.parse_args()

    print(f"MinerU dir: {args.mineru_dir}")
    print(f"LaTeX dir:  {args.latex_dir}")

    mineru = scan_mineru(args.mineru_dir)
    latex = scan_latex(args.latex_dir)

    mineru_ids: Set[str] = set(mineru.keys())
    latex_ids: Set[str] = set(latex.keys())

    both = sorted(mineru_ids & latex_ids)
    mineru_only = sorted(mineru_ids - latex_ids)
    latex_only = sorted(latex_ids - mineru_ids)

    # Per-doc health for matched docs
    matched_health = []
    for doc_id in both:
        m = mineru[doc_id]
        l = latex[doc_id]
        matched_health.append({
            "doc_id": doc_id,
            "mineru_has_content_list": m["has_content_list"],
            "mineru_has_images": m["has_images"],
            "mineru_image_count": m["image_count"],
            "mineru_has_multimodal_elements": m["has_multimodal_elements"],
            "latex_tex_count": l["tex_count"],
            "latex_has_bbl": l["has_bbl"],
        })

    report = {
        "summary": {
            "mineru_total": len(mineru_ids),
            "latex_total": len(latex_ids),
            "matched": len(both),
            "mineru_only": len(mineru_only),
            "latex_only": len(latex_only),
            "match_rate_mineru": f"{len(both)/max(len(mineru_ids),1)*100:.1f}%",
            "match_rate_latex": f"{len(both)/max(len(latex_ids),1)*100:.1f}%",
        },
        "matched_docs": both,
        "mineru_only_docs": mineru_only,
        "latex_only_docs": latex_only,
        "matched_health": matched_health,
    }

    # Console output
    print(f"\n{'='*60}")
    print("Data Alignment Report")
    print(f"{'='*60}")
    print(f"  MinerU docs:   {len(mineru_ids)}")
    print(f"  LaTeX docs:    {len(latex_ids)}")
    print(f"  Matched:       {len(both)}")
    print(f"  MinerU-only:   {len(mineru_only)}")
    print(f"  LaTeX-only:    {len(latex_only)}")
    print(f"  Match rate (MinerU): {report['summary']['match_rate_mineru']}")
    print(f"  Match rate (LaTeX):  {report['summary']['match_rate_latex']}")

    if mineru_only:
        print(f"\n  MinerU-only (first 10): {mineru_only[:10]}")
    if latex_only:
        print(f"  LaTeX-only (first 10):  {latex_only[:10]}")

    # Health check
    no_content_list = sum(1 for h in matched_health if not h["mineru_has_content_list"])
    no_images = sum(1 for h in matched_health if not h["mineru_has_images"])
    no_bbl = sum(1 for h in matched_health if not h["latex_has_bbl"])
    print(f"\n  Matched docs missing content_list: {no_content_list}")
    print(f"  Matched docs missing images:       {no_images}")
    print(f"  Matched docs missing .bbl:         {no_bbl}")

    # Write report
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"\nReport saved to: {args.output}")


if __name__ == "__main__":
    main()
