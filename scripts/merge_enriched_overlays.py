#!/usr/bin/env python3
"""
Merge one or more enriched overlays onto a base multimodal_elements graph.

Typical usage for the trial retrieval lane:
  base graph elements
  + old trial enrich overlay
  + trial57 backfill enrich overlay
  -> trial57 full enriched overlay
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple


ENRICH_FIELDS = [
    "enriched_title",
    "enriched_metadata",
    "enriched_content",
    "enrichment_issues",
]


def load_json(path: Path) -> dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--base",
        required=True,
        help="Base multimodal_elements.json to copy and overlay enrichments onto",
    )
    ap.add_argument(
        "--overlay",
        action="append",
        default=[],
        help="Overlay file containing enriched fields; later overlays win",
    )
    ap.add_argument("--output", required=True, help="Merged output path")
    args = ap.parse_args()

    base_path = Path(args.base)
    overlay_paths = [Path(p) for p in args.overlay]
    output_path = Path(args.output)

    merged = load_json(base_path)
    merged_elements: Dict[Tuple[str, str], dict] = {}
    for doc_id, doc in merged.get("documents", {}).items():
        for element_id, elem in doc.get("elements", {}).items():
            merged_elements[(doc_id, element_id)] = elem

    applied = 0
    touched: Dict[Tuple[str, str], List[str]] = {}
    for overlay_path in overlay_paths:
        if not overlay_path.exists():
            print(f"[warn] overlay missing, skipping: {overlay_path}")
            continue
        overlay_data = load_json(overlay_path)
        for doc_id, doc in overlay_data.get("documents", {}).items():
            for element_id, overlay_elem in doc.get("elements", {}).items():
                target = merged_elements.get((doc_id, element_id))
                if target is None:
                    continue
                wrote_any = False
                for field in ENRICH_FIELDS:
                    if field not in overlay_elem:
                        continue
                    value = overlay_elem[field]
                    if field == "enriched_content" and not (value or "").strip():
                        continue
                    target[field] = value
                    wrote_any = True
                if wrote_any:
                    applied += 1
                    touched.setdefault((doc_id, element_id), []).append(str(overlay_path))

    total_enriched = 0
    for doc in merged.get("documents", {}).values():
        for elem in doc.get("elements", {}).values():
            if (elem.get("enriched_content") or "").strip():
                total_enriched += 1

    merged.setdefault("metadata", {})
    merged["metadata"]["merged_enrichment"] = {
        "base": str(base_path),
        "overlays": [str(p) for p in overlay_paths],
        "applied_updates": applied,
        "total_enriched_elements": total_enriched,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(merged, f, indent=2, ensure_ascii=False)

    print(f"[ok] overlays merged: {len(overlay_paths)}")
    print(f"[ok] applied element updates: {applied}")
    print(f"[ok] total elements with enriched_content: {total_enriched}")
    print(f"[ok] output: {output_path}")


if __name__ == "__main__":
    main()
