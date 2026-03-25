#!/usr/bin/env python3
"""Build proxy JSONL queries for C-Pool universal query evaluation.

The current corpus only contains figure/table/formula elements, while C-Pool
queries often expect section-level evidence. This script creates a proxy
ground-truth JSONL that keeps only supported figure/table evidence types and
maps them to candidate element IDs via lightweight caption/content heuristics.

The output is compatible with scripts/run_phase0_eval_ab.py.
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Set


PATTERNS: Dict[str, List[str]] = {
    "figure:architecture": [
        r"architecture",
        r"framework",
        r"pipeline",
        r"system overview",
        r"model overview",
    ],
    "figure:overview": [
        r"overview",
        r"framework",
        r"pipeline",
    ],
    "figure:comparison": [
        r"comparison",
        r"\bcompare\b",
    ],
    "figure:results": [
        r"\bresults?\b",
        r"performance",
        r"accuracy",
    ],
    "figure:error_analysis": [
        r"error analysis",
        r"failure case",
        r"error cases",
    ],
    "figure:visualization": [
        r"visualization",
        r"visualisation",
        r"heatmap",
        r"t-sne",
        r"tsne",
        r"umap",
    ],
    "table:main_results": [
        r"main results?",
        r"\bresults?\b",
        r"performance comparison",
        r"comparison with",
        r"sota",
        r"benchmark",
    ],
    "table:ablation": [
        r"ablation",
    ],
    "table:dataset": [
        r"dataset",
        r"data statistics",
        r"statistics",
    ],
    "table:efficiency": [
        r"efficiency",
        r"speed",
        r"latency",
        r"runtime",
        r"inference time",
        r"throughput",
    ],
}


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def iter_elements(elements_obj: Any) -> Iterable[Dict[str, Any]]:
    docs = elements_obj.get("documents", elements_obj) if isinstance(elements_obj, dict) else elements_obj
    if isinstance(docs, dict):
        for doc_id, doc in docs.items():
            if not isinstance(doc, dict):
                continue
            elements = doc.get("elements", {})
            values = elements.values() if isinstance(elements, dict) else elements
            for elem in values or []:
                if isinstance(elem, dict):
                    elem = {**elem, "doc_id": doc_id}
                    yield elem
    elif isinstance(docs, list):
        for elem in docs:
            if isinstance(elem, dict):
                yield elem


def element_text(elem: Dict[str, Any]) -> str:
    fields = [
        elem.get("caption", ""),
        elem.get("content", ""),
        elem.get("context_before", ""),
        elem.get("context_after", ""),
    ]
    return " ".join(x for x in fields if isinstance(x, str)).strip()


def classification_text(elem: Dict[str, Any]) -> str:
    fields = [
        elem.get("caption", ""),
        elem.get("content", ""),
    ]
    return " ".join(x for x in fields if isinstance(x, str)).strip()


def span_text(elem: Dict[str, Any], max_chars: int = 220) -> str:
    for key in ("caption", "content", "context_before", "context_after"):
        value = (elem.get(key) or "").strip()
        if value:
            return value[:max_chars]
    return (elem.get("element_id") or "")[:max_chars]


def infer_proxy_labels(elem: Dict[str, Any]) -> Set[str]:
    elem_type = str(elem.get("element_type", "")).strip().lower()
    if not elem_type:
        return set()

    # Keep proxy typing conservative: captions/content are usually the clearest
    # place to infer "architecture", "results", or "dataset" table roles.
    text = classification_text(elem).lower()
    labels: Set[str] = set()
    for label, patterns in PATTERNS.items():
        if not label.startswith(f"{elem_type}:"):
            continue
        if any(re.search(pattern, text) for pattern in patterns):
            labels.add(label)
    return labels


def build_label_index(elements_path: Path) -> Dict[str, List[Dict[str, str]]]:
    elements_obj = load_json(elements_path)
    label_index: Dict[str, List[Dict[str, str]]] = defaultdict(list)
    seen_ids: Dict[str, Set[str]] = defaultdict(set)

    for elem in iter_elements(elements_obj):
        eid = str(elem.get("element_id", "")).strip()
        if not eid:
            continue
        for label in infer_proxy_labels(elem):
            if eid in seen_ids[label]:
                continue
            seen_ids[label].add(eid)
            label_index[label].append(
                {
                    "element_id": eid,
                    "span": span_text(elem),
                }
            )
    return dict(label_index)


def build_proxy_rows(
    cpool_queries: List[Dict[str, Any]],
    label_index: Dict[str, List[Dict[str, str]]],
) -> tuple[List[Dict[str, Any]], Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    supported_queries = 0
    unsupported_queries = 0
    label_usage = Counter()
    query_summaries: List[Dict[str, Any]] = []

    for q in cpool_queries:
        expected = [str(x) for x in (q.get("expected_evidence_types") or []) if x]
        supported = [label for label in expected if label_index.get(label)]
        unsupported = [label for label in expected if label not in supported]

        if not supported:
            unsupported_queries += 1
            query_summaries.append(
                {
                    "query_id": q.get("id", ""),
                    "status": "unsupported",
                    "expected_evidence_types": expected,
                }
            )
            continue

        supported_queries += 1
        required_evidence_spans: List[Dict[str, Any]] = []
        seen_ids: Set[str] = set()
        matched_counts: Dict[str, int] = {}

        for label in supported:
            matches = label_index.get(label, [])
            matched_counts[label] = len(matches)
            label_usage[label] += 1
            for match in matches:
                eid = match["element_id"]
                if eid in seen_ids:
                    continue
                required_evidence_spans.append(
                    {
                        "element_id": eid,
                        "span": match["span"],
                        "proxy_type": label,
                    }
                )
                seen_ids.add(eid)

        rows.append(
            {
                "query_id": q.get("id", ""),
                "query": q.get("query", ""),
                "query_intent": q.get("query_intent"),
                "category": q.get("category"),
                "paraphrase_group": q.get("paraphrase_group"),
                "expected_evidence_types": expected,
                "supported_expected_evidence_types": supported,
                "unsupported_expected_evidence_types": unsupported,
                "proxy_required_evidence_count": len(required_evidence_spans),
                "required_evidence_spans": required_evidence_spans,
                "note": (
                    "Proxy GT from figure/table evidence-type heuristics. "
                    "Section-level evidence types are excluded from this file."
                ),
            }
        )
        query_summaries.append(
            {
                "query_id": q.get("id", ""),
                "status": "supported",
                "supported_expected_evidence_types": supported,
                "unsupported_expected_evidence_types": unsupported,
                "proxy_required_evidence_count": len(required_evidence_spans),
                "matched_counts": matched_counts,
            }
        )

    summary = {
        "total_queries": len(cpool_queries),
        "supported_queries": supported_queries,
        "unsupported_queries": unsupported_queries,
        "supported_rate": round(supported_queries / max(len(cpool_queries), 1), 4),
        "label_match_counts": {label: len(matches) for label, matches in sorted(label_index.items())},
        "label_query_usage": dict(label_usage),
        "query_summaries": query_summaries,
    }
    return rows, summary


def main() -> None:
    ap = argparse.ArgumentParser(description="Build proxy JSONL for C-Pool evaluation")
    ap.add_argument("--input", type=Path, default=Path("data/c_pool_universal_queries.json"))
    ap.add_argument("--elements", type=Path, default=Path("data111/multimodal_elements_enriched.json"))
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--summary", type=Path, default=None)
    args = ap.parse_args()

    cpool_obj = load_json(args.input)
    cpool_queries = cpool_obj.get("queries", []) if isinstance(cpool_obj, dict) else cpool_obj
    label_index = build_label_index(args.elements)
    rows, summary = build_proxy_rows(cpool_queries, label_index)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    if args.summary:
        args.summary.parent.mkdir(parents=True, exist_ok=True)
        args.summary.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"Loaded C-Pool queries: {len(cpool_queries)}")
    print(f"Proxy-supported queries: {summary['supported_queries']}/{summary['total_queries']}")
    print(f"Wrote proxy JSONL: {args.output}")
    if args.summary:
        print(f"Wrote summary JSON: {args.summary}")


if __name__ == "__main__":
    main()
