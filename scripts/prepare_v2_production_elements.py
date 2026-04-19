#!/usr/bin/env python3
"""Prepare production-ready v2 multimodal elements for the 1040-doc corpus.

This script takes the raw v2 multimodal elements, filters them to the pruned
1040 production documents, and merges any existing partial enrichments from the
new-document pipeline (currently Method C long-chain element enrichment).

It also emits a compact coverage report so we can audit how much section /
element enrichment is actually available on the production corpus.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterable, Tuple


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


def count_enriched_elements(docs: Dict[str, Any]) -> Tuple[int, int, int]:
    total_docs = len(docs)
    docs_with_enriched = 0
    enriched_elements = 0
    for doc in docs.values():
        doc_has_enriched = False
        for _, elem in iter_elements(doc):
            if (
                elem.get("enriched_title")
                or elem.get("enriched_content")
                or elem.get("enriched_metadata")
            ):
                enriched_elements += 1
                doc_has_enriched = True
        if doc_has_enriched:
            docs_with_enriched += 1
    return total_docs, docs_with_enriched, enriched_elements


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--base-elements",
        default="data/01_graphs/multimodal_elements_v2.json",
        help="Raw v2 multimodal elements JSON",
    )
    ap.add_argument(
        "--pruned-graph",
        default="data/01_graphs/pruned_graph_v2.json",
        help="Pruned production graph used to define the 1040 production docs",
    )
    ap.add_argument(
        "--partial-element-enrich",
        default="data/02_enriched/method_c_long_chain_elements_enriched_v1.json",
        help="Partial new-doc element enrichment JSON to merge into production elements",
    )
    ap.add_argument(
        "--legacy-element-enrich",
        default="data/02_enriched/multimodal_elements_enriched.json",
        help="Legacy element enrich JSON used only for overlap auditing",
    )
    ap.add_argument(
        "--legacy-section-enrich",
        default="data/05_eval/m2/section_nodes_enriched_2026-03-26.json",
        help="Legacy section enrich JSON used only for overlap auditing",
    )
    ap.add_argument(
        "--legacy-summary-graph",
        default="data/01_graphs/llm_summary_virtual_nodes.json",
        help="Legacy summary graph used only for overlap auditing",
    )
    ap.add_argument(
        "--output",
        default="data/02_enriched/multimodal_elements_v2_production_partial.json",
        help="Merged production elements output path",
    )
    ap.add_argument(
        "--report",
        default="data/02_enriched/multimodal_elements_v2_production_partial_report.json",
        help="Coverage report output path",
    )
    args = ap.parse_args()

    base_elements = load_json(Path(args.base_elements))
    pruned_graph = load_json(Path(args.pruned_graph))
    partial_enrich = load_json(Path(args.partial_element_enrich))

    pruned_ids = set((pruned_graph.get("documents") or {}).keys())
    base_docs = base_elements.get("documents") or {}

    production_docs = {
        did: doc
        for did, doc in base_docs.items()
        if did in pruned_ids
    }

    merged = {
        "metadata": {
            **(base_elements.get("metadata") or {}),
            "production_filter": {
                "source": "prepare_v2_production_elements.py",
                "target_docs": len(pruned_ids),
                "kept_docs": len(production_docs),
            },
        },
        "documents": json.loads(json.dumps(production_docs)),
    }

    partial_docs = partial_enrich.get("documents") or {}
    merged_count = 0
    merged_docs = set()
    for did, partial_doc in partial_docs.items():
        if did not in merged["documents"]:
            continue
        base_doc = merged["documents"][did]
        base_elements_map = base_doc.get("elements") or {}
        if not isinstance(base_elements_map, dict):
            continue
        partial_elements = partial_doc.get("elements") or {}
        if not isinstance(partial_elements, dict):
            continue

        for eid, partial_elem in partial_elements.items():
            base_elem = base_elements_map.get(eid)
            if not isinstance(base_elem, dict):
                continue
            for key in (
                "enriched_title",
                "enriched_metadata",
                "enriched_content",
                "enrichment_issues",
            ):
                if key in partial_elem:
                    base_elem[key] = partial_elem[key]
            merged_count += 1
            merged_docs.add(did)

    merged["metadata"]["enrichment"] = {
        "method": "partial_method_c_long_chain_merge",
        "source": args.partial_element_enrich,
        "merged_docs": len(merged_docs),
        "merged_elements": merged_count,
    }

    report: Dict[str, Any] = {
        "production_docs_target": len(pruned_ids),
        "base_v2_docs": len(base_docs),
        "production_docs_available": len(production_docs),
        "partial_element_enrich_docs": len(partial_docs),
        "partial_element_enrich_overlap_docs": len(set(partial_docs) & pruned_ids),
        "merged_docs": len(merged_docs),
        "merged_elements": merged_count,
    }

    prod_total_docs, prod_docs_with_enriched, prod_enriched_elements = count_enriched_elements(
        merged["documents"]
    )
    report["production_elements"] = {
        "docs": prod_total_docs,
        "docs_with_any_enriched_element": prod_docs_with_enriched,
        "docs_without_any_enriched_element": prod_total_docs - prod_docs_with_enriched,
        "enriched_elements": prod_enriched_elements,
    }

    legacy_element_path = Path(args.legacy_element_enrich)
    if legacy_element_path.exists():
        legacy_element_docs = set((load_json(legacy_element_path).get("documents") or {}).keys())
        report["legacy_element_enrich_overlap_docs"] = len(legacy_element_docs & pruned_ids)

    legacy_section_path = Path(args.legacy_section_enrich)
    if legacy_section_path.exists():
        sec_rows = load_json(legacy_section_path).get("sections") or []
        sec_docs = set()
        for row in sec_rows:
            sid = str((row or {}).get("section_id", "") or "")
            if "::" in sid:
                sec_docs.add(sid.split("::")[0])
        report["legacy_section_enrich"] = {
            "rows": len(sec_rows),
            "docs": len(sec_docs),
            "overlap_production_docs": len(sec_docs & pruned_ids),
        }

    legacy_summary_path = Path(args.legacy_summary_graph)
    if legacy_summary_path.exists():
        summary_docs = set((load_json(legacy_summary_path).get("documents") or {}).keys())
        report["legacy_summary_graph"] = {
            "docs": len(summary_docs),
            "overlap_production_docs": len(summary_docs & pruned_ids),
        }

    output_path = Path(args.output)
    report_path = Path(args.report)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(merged, ensure_ascii=False, indent=2), encoding="utf-8")
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print("Prepared production v2 elements")
    print(f"  Output: {output_path}")
    print(f"  Report: {report_path}")
    print(f"  Production docs: {report['production_docs_available']} / {report['production_docs_target']}")
    print(
        "  Element enrich coverage: "
        f"{report['production_elements']['docs_with_any_enriched_element']} docs, "
        f"{report['production_elements']['enriched_elements']} elements"
    )


if __name__ == "__main__":
    main()
