#!/usr/bin/env python3
"""
Build a dedicated enrich-backfill subset for the 57-doc trial retrieval lane.

This script keeps the old trial retrieval-eval lane separate from the 1040-doc
production lane:
  - gold docs are inferred from qrels
  - only enrichable element types are considered (figure/table/formula)
  - elements already carrying non-empty enriched_content are skipped
  - outputs a compact multimodal_elements-style subset JSON plus a coverage report
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, List, Set, Tuple


ENRICHABLE_TYPES = {"figure", "table", "formula"}


def load_json(path: Path) -> dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def load_jsonl(path: Path) -> List[dict]:
    rows: List[dict] = []
    with open(path, encoding="utf-8-sig") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def get_gold_targets(qrels_rows: Iterable[dict]) -> Tuple[Set[str], Set[str]]:
    doc_ids: Set[str] = set()
    element_ids: Set[str] = set()
    for row in qrels_rows:
        passage_id = row["passage_id"]
        element_ids.add(passage_id)
        doc_ids.add(passage_id.split("_", 1)[0])
    return doc_ids, element_ids


def build_enriched_lookup(enriched_data: dict) -> Set[Tuple[str, str]]:
    covered: Set[Tuple[str, str]] = set()
    for doc_id, doc in enriched_data.get("documents", {}).items():
        for element_id, elem in doc.get("elements", {}).items():
            if (elem.get("enriched_content") or "").strip():
                covered.add((doc_id, element_id))
    return covered


def summarize_doc(
    doc_id: str,
    elements: Dict[str, dict],
    enriched_lookup: Set[Tuple[str, str]],
    gold_element_ids: Set[str],
) -> Tuple[dict, Dict[str, dict]]:
    total_enrichable = 0
    already_enriched = 0
    missing = 0
    gold_enrichable = 0
    gold_missing = 0
    missing_elements: Dict[str, dict] = {}
    by_type_total: Counter[str] = Counter()
    by_type_missing: Counter[str] = Counter()

    for element_id, elem in elements.items():
        etype = elem.get("element_type")
        if etype not in ENRICHABLE_TYPES:
            continue
        total_enrichable += 1
        by_type_total[etype] += 1
        is_gold = element_id in gold_element_ids
        if is_gold:
            gold_enrichable += 1

        if (doc_id, element_id) in enriched_lookup:
            already_enriched += 1
            continue

        missing += 1
        by_type_missing[etype] += 1
        if is_gold:
            gold_missing += 1
        missing_elements[element_id] = elem

    coverage = already_enriched / total_enrichable if total_enrichable else 1.0
    return (
        {
            "doc_id": doc_id,
            "total_enrichable": total_enrichable,
            "already_enriched": already_enriched,
            "missing": missing,
            "coverage": coverage,
            "gold_enrichable": gold_enrichable,
            "gold_missing": gold_missing,
            "by_type_total": dict(sorted(by_type_total.items())),
            "by_type_missing": dict(sorted(by_type_missing.items())),
        },
        missing_elements,
    )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--graph-elements",
        default="data/03_queries/M4query_v1/graphs/multimodal_elements.json",
        help="Full trial multimodal_elements graph",
    )
    ap.add_argument(
        "--existing-enriched",
        default="data/02_enriched/multimodal_elements_enriched.json",
        help="Existing old-trial enriched overlay",
    )
    ap.add_argument(
        "--qrels",
        default="data/03_queries/M4query_v1/qrels.jsonl",
        help="Trial qrels used to define the 57 gold docs",
    )
    ap.add_argument(
        "--output-subset",
        required=True,
        help="Output JSON containing only missing enrichable elements from the 57 gold docs",
    )
    ap.add_argument(
        "--output-report",
        required=True,
        help="Output JSON report with coverage stats",
    )
    ap.add_argument(
        "--top-missing-docs",
        type=int,
        default=20,
        help="How many highest-gap docs to retain in the report summary",
    )
    args = ap.parse_args()

    graph_path = Path(args.graph_elements)
    enriched_path = Path(args.existing_enriched)
    qrels_path = Path(args.qrels)
    subset_path = Path(args.output_subset)
    report_path = Path(args.output_report)

    graph_data = load_json(graph_path)
    enriched_data = load_json(enriched_path) if enriched_path.exists() else {"documents": {}}
    qrels_rows = load_jsonl(qrels_path)

    gold_docs, gold_element_ids = get_gold_targets(qrels_rows)
    enriched_lookup = build_enriched_lookup(enriched_data)

    subset_documents: Dict[str, dict] = {}
    per_doc: List[dict] = []
    totals = Counter()
    missing_type_totals: Counter[str] = Counter()
    covered_gold_qrel = 0
    total_gold_qrel_enrichable = 0

    for doc_id in sorted(gold_docs):
        doc = graph_data.get("documents", {}).get(doc_id)
        if not doc:
            continue
        elements = doc.get("elements", {})
        doc_report, missing_elements = summarize_doc(
            doc_id=doc_id,
            elements=elements,
            enriched_lookup=enriched_lookup,
            gold_element_ids=gold_element_ids,
        )
        per_doc.append(doc_report)

        totals["gold_docs"] += 1
        totals["total_enrichable"] += doc_report["total_enrichable"]
        totals["already_enriched"] += doc_report["already_enriched"]
        totals["missing"] += doc_report["missing"]
        totals["gold_enrichable"] += doc_report["gold_enrichable"]
        totals["gold_missing"] += doc_report["gold_missing"]

        if doc_report["missing"] > 0:
            subset_doc = dict(doc)
            subset_doc["elements"] = missing_elements
            subset_doc["num_elements"] = len(missing_elements)
            subset_doc["num_edges"] = 0
            subset_doc["edges"] = []
            subset_doc["multimodal_pairs"] = []
            subset_documents[doc_id] = subset_doc

        for etype, count in doc_report["by_type_missing"].items():
            missing_type_totals[etype] += count

    total_gold_qrel_enrichable = totals["gold_enrichable"]
    covered_gold_qrel = total_gold_qrel_enrichable - totals["gold_missing"]

    top_missing_docs = sorted(
        (d for d in per_doc if d["missing"] > 0),
        key=lambda x: (-x["missing"], x["doc_id"]),
    )[: args.top_missing_docs]

    subset_payload = {
        "metadata": {
            "source_graph": str(graph_path),
            "source_enriched_overlay": str(enriched_path),
            "source_qrels": str(qrels_path),
            "subset_scope": "trial57_missing_enrichable_only",
            "enrichable_types": sorted(ENRICHABLE_TYPES),
            "num_documents": len(subset_documents),
            "num_elements": totals["missing"],
        },
        "documents": subset_documents,
    }
    subset_path.parent.mkdir(parents=True, exist_ok=True)
    with open(subset_path, "w", encoding="utf-8") as f:
        json.dump(subset_payload, f, indent=2, ensure_ascii=False)

    report = {
        "summary": {
            "gold_docs": totals["gold_docs"],
            "total_enrichable": totals["total_enrichable"],
            "already_enriched": totals["already_enriched"],
            "missing": totals["missing"],
            "coverage": (
                totals["already_enriched"] / totals["total_enrichable"]
                if totals["total_enrichable"]
                else 1.0
            ),
            "gold_qrel_enrichable": total_gold_qrel_enrichable,
            "gold_qrel_enrichable_covered": covered_gold_qrel,
            "gold_qrel_enrichable_missing": totals["gold_missing"],
            "gold_qrel_enrichable_coverage": (
                covered_gold_qrel / total_gold_qrel_enrichable
                if total_gold_qrel_enrichable
                else 1.0
            ),
            "missing_by_type": dict(sorted(missing_type_totals.items())),
        },
        "top_missing_docs": top_missing_docs,
        "per_doc": per_doc,
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    print(f"[ok] gold docs: {totals['gold_docs']}")
    print(
        f"[ok] enrichable elements: {totals['already_enriched']}/{totals['total_enrichable']} "
        f"({report['summary']['coverage']:.1%})"
    )
    print(
        f"[ok] gold-qrel enrichable elements: {covered_gold_qrel}/{total_gold_qrel_enrichable} "
        f"({report['summary']['gold_qrel_enrichable_coverage']:.1%})"
    )
    print(f"[ok] missing subset docs: {len(subset_documents)}")
    print(f"[ok] missing subset elements: {totals['missing']}")
    print(f"[ok] subset written to: {subset_path}")
    print(f"[ok] report written to: {report_path}")


if __name__ == "__main__":
    main()
