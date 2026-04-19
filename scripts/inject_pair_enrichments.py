#!/usr/bin/env python3
"""Inject element enrichments into selected pair files.

The intra-doc selector emits raw element bodies with empty enriched fields.
This script fills those fields from a merged enriched-elements asset so the
old query generator can use both raw element bodies and any available partial
enrichments on the 1040-doc production corpus.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterable, Tuple


ENRICH_KEYS = (
    "enriched_title",
    "enriched_metadata",
    "enriched_content",
    "enrichment_issues",
)


def load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def iter_elements(doc: Dict[str, Any]) -> Iterable[Tuple[str, Dict[str, Any]]]:
    elements = doc.get("elements", {})
    if isinstance(elements, dict):
        yield from elements.items()
        return
    if isinstance(elements, list):
        for elem in elements:
            if not isinstance(elem, dict):
                continue
            eid = str(elem.get("element_id", "") or "").strip()
            if eid:
                yield eid, elem


def build_enrich_index(elements_data: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    idx: Dict[str, Dict[str, Any]] = {}
    for doc in (elements_data.get("documents") or {}).values():
        for eid, elem in iter_elements(doc):
            if any(elem.get(key) for key in ENRICH_KEYS[:3]):
                idx[eid] = elem
    return idx


def inject_into_element(elem: Dict[str, Any], idx: Dict[str, Dict[str, Any]]) -> bool:
    eid = str(elem.get("element_id", "") or "").strip()
    if not eid or eid not in idx:
        return False
    src = idx[eid]
    for key in ENRICH_KEYS:
        if key in src:
            elem[key] = src[key]
    return True


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--pairs", required=True, help="Input pair JSON")
    ap.add_argument("--elements", required=True, help="Merged enriched elements JSON")
    ap.add_argument("--output", required=True, help="Output pair JSON")
    args = ap.parse_args()

    pairs_data = load_json(Path(args.pairs))
    elements_data = load_json(Path(args.elements))
    enrich_idx = build_enrich_index(elements_data)

    pair_count = 0
    matched_pairs = 0
    matched_endpoints = 0
    total_endpoints = 0

    for pair in pairs_data.get("pairs", []):
        pair_count += 1
        pair_hit = False
        for key in ("element_a", "element_b"):
            elem = pair.get(key)
            if not isinstance(elem, dict):
                continue
            total_endpoints += 1
            if inject_into_element(elem, enrich_idx):
                matched_endpoints += 1
                pair_hit = True
        node_group = pair.get("node_group") or []
        if isinstance(node_group, list):
            for elem in node_group:
                if isinstance(elem, dict):
                    inject_into_element(elem, enrich_idx)
        if pair_hit:
            matched_pairs += 1

    metadata = pairs_data.setdefault("metadata", {})
    metadata["element_enrichment_injection"] = {
        "source": "inject_pair_enrichments.py",
        "elements_file": args.elements,
        "pairs_total": pair_count,
        "pairs_with_any_enriched_endpoint": matched_pairs,
        "matched_endpoints": matched_endpoints,
        "total_endpoints": total_endpoints,
    }

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(pairs_data, ensure_ascii=False, indent=2), encoding="utf-8")

    print("Injected enrichments into pair file")
    print(f"  Output: {out_path}")
    print(f"  Pairs touched: {matched_pairs} / {pair_count}")
    print(f"  Endpoints matched: {matched_endpoints} / {total_endpoints}")


if __name__ == "__main__":
    main()
