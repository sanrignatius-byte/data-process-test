#!/usr/bin/env python3
"""Audit MinerU cross-document embedding matches.

Outputs:
1) JSON summary report with distribution/hubness/quality signals.
2) CSV with suspicious top-1 matches for manual review.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from statistics import mean
from typing import Any, Dict, List, Optional, Tuple


TOKEN_RE = re.compile(r"\b[a-zA-Z][a-zA-Z0-9_-]{2,}\b")
STOPWORDS = {
    "the", "and", "for", "that", "with", "from", "this", "were", "are", "was",
    "using", "used", "into", "between", "than", "have", "has", "had", "their",
    "our", "your", "you", "its", "can", "may", "might", "should", "would",
    "could", "not", "but", "also", "such", "these", "those", "over", "under",
    "after", "before", "through", "show", "shows", "shown", "table", "figure",
    "formula", "equation", "results", "method", "methods", "dataset", "datasets",
}


@dataclass
class ElementInfo:
    doc_id: str
    element_id: str
    element_type: str
    caption: str
    text: str
    image_path: Optional[str]


def clean(s: str) -> str:
    return " ".join((s or "").split())


def truncate(s: str, n: int = 240) -> str:
    if len(s) <= n:
        return s
    return s[: n - 3].rstrip() + "..."


def quantiles(vals: List[float], ps: Tuple[float, ...]) -> Dict[str, float]:
    if not vals:
        return {f"p{int(p * 100):02d}": 0.0 for p in ps}
    x = sorted(vals)
    out: Dict[str, float] = {}
    for p in ps:
        if len(x) == 1:
            out[f"p{int(p * 100):02d}"] = round(float(x[0]), 6)
            continue
        idx = p * (len(x) - 1)
        lo = math.floor(idx)
        hi = math.ceil(idx)
        if lo == hi:
            v = x[lo]
        else:
            w = idx - lo
            v = x[lo] * (1.0 - w) + x[hi] * w
        out[f"p{int(p * 100):02d}"] = round(float(v), 6)
    return out


def token_set(s: str) -> set:
    toks = {t.lower() for t in TOKEN_RE.findall(s or "")}
    return {t for t in toks if t not in STOPWORDS}


def jaccard(a: set, b: set) -> float:
    if not a and not b:
        return 0.0
    u = a | b
    if not u:
        return 0.0
    return len(a & b) / len(u)


def build_element_text(elem: Dict[str, Any], max_chars: int = 900) -> str:
    etype = clean(str(elem.get("element_type", "")))
    caption = clean(str(elem.get("caption", "")))
    content = clean(str(elem.get("content", "")))
    ctx_before = clean(str(elem.get("context_before", "")))
    ctx_after = clean(str(elem.get("context_after", "")))
    parts = [f"Type: {etype}."]
    if caption:
        parts.append(f"Caption: {caption}")
    if content:
        parts.append(f"Content: {content}")
    if ctx_before:
        parts.append(f"Context before: {ctx_before}")
    if ctx_after:
        parts.append(f"Context after: {ctx_after}")
    txt = "\n".join(parts)
    if len(txt) <= max_chars:
        return txt
    return txt[: max_chars - 3].rstrip() + "..."


def load_elements(multimodal_path: Path) -> Dict[str, ElementInfo]:
    data = json.loads(multimodal_path.read_text(encoding="utf-8"))
    docs = data.get("documents", {})
    out: Dict[str, ElementInfo] = {}
    for doc_id, doc in docs.items():
        for eid, elem in (doc.get("elements", {}) or {}).items():
            out[eid] = ElementInfo(
                doc_id=str(doc_id),
                element_id=str(eid),
                element_type=str(elem.get("element_type", "unknown")).lower(),
                caption=clean(str(elem.get("caption", ""))),
                text=build_element_text(elem, max_chars=900),
                image_path=elem.get("image_path"),
            )
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="Audit embedding-based cross-doc matches")
    ap.add_argument("--matches", required=True, help="JSONL from match_mineru_crossdoc_with_embeddings.py")
    ap.add_argument("--elements", default="data/multimodal_elements.json")
    ap.add_argument("--report", required=True, help="Output JSON report")
    ap.add_argument("--suspicious-csv", required=True, help="Output CSV for manual review")
    ap.add_argument("--sample-size", type=int, default=60)
    args = ap.parse_args()

    matches_path = Path(args.matches)
    elements_path = Path(args.elements)
    report_path = Path(args.report)
    suspicious_path = Path(args.suspicious_csv)

    records: List[Dict[str, Any]] = []
    with matches_path.open("r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if s:
                records.append(json.loads(s))

    if not records:
        raise SystemExit("No records in matches file.")

    elements = load_elements(elements_path)

    source_type_counter = Counter()
    topk_len_counter = Counter()
    total_matches = 0
    top1_scores: List[float] = []
    all_scores: List[float] = []
    rank_scores: Dict[int, List[float]] = defaultdict(list)
    same_doc_violations = 0
    type_mismatches = 0
    missing_source_image = 0
    missing_target_image = 0
    source_docs = set()
    target_docs = set()
    model_names = Counter()

    # graph signals
    top1_target_counter = Counter()
    all_target_counter = Counter()
    adjacency: Dict[str, set] = {}

    # quality features on top-1
    top1_rows: List[Dict[str, Any]] = []

    for rec in records:
        src_id = rec.get("source_element_id", "")
        src_doc = rec.get("source_doc_id", "")
        src_type = rec.get("source_type", "unknown")
        source_type_counter[src_type] += 1
        source_docs.add(src_doc)
        model_names[str(rec.get("model", ""))] += 1

        if not rec.get("source_image_path"):
            missing_source_image += 1

        matches = rec.get("matches", []) or []
        topk_len_counter[len(matches)] += 1
        total_matches += len(matches)

        nbrs = set()
        for rank, m in enumerate(matches, start=1):
            tgt_id = m.get("target_element_id", "")
            tgt_doc = m.get("target_doc_id", "")
            tgt_type = m.get("target_type", "unknown")
            score = float(m.get("score", 0.0))
            all_scores.append(score)
            rank_scores[rank].append(score)
            all_target_counter[tgt_id] += 1
            target_docs.add(tgt_doc)
            if not m.get("target_image_path"):
                missing_target_image += 1
            if src_doc == tgt_doc:
                same_doc_violations += 1
            if src_type != tgt_type:
                type_mismatches += 1
            nbrs.add(tgt_id)

        adjacency[src_id] = nbrs

        if matches:
            m1 = matches[0]
            top1_target_counter[m1.get("target_element_id", "")] += 1
            top1_score = float(m1.get("score", 0.0))
            top1_scores.append(top1_score)
            score2 = float(matches[1].get("score", 0.0)) if len(matches) > 1 else 0.0
            margin12 = top1_score - score2

            src_elem = elements.get(src_id)
            tgt_elem = elements.get(m1.get("target_element_id", ""))
            src_text = src_elem.text if src_elem else ""
            tgt_text = tgt_elem.text if tgt_elem else ""
            src_cap = src_elem.caption if src_elem else ""
            tgt_cap = tgt_elem.caption if tgt_elem else ""

            text_j = jaccard(token_set(src_text), token_set(tgt_text))
            cap_j = jaccard(token_set(src_cap), token_set(tgt_cap))

            top1_rows.append(
                {
                    "source_element_id": src_id,
                    "source_doc_id": src_doc,
                    "source_type": src_type,
                    "target_element_id": m1.get("target_element_id", ""),
                    "target_doc_id": m1.get("target_doc_id", ""),
                    "target_type": m1.get("target_type", ""),
                    "score": round(top1_score, 6),
                    "score2": round(score2, 6),
                    "margin12": round(margin12, 6),
                    "text_jaccard": round(text_j, 6),
                    "caption_jaccard": round(cap_j, 6),
                    "target_top1_indegree": 0,  # fill later
                    "source_caption": truncate(src_cap, 220),
                    "target_caption": truncate(tgt_cap, 220),
                }
            )

    # fill in target top1 indegree
    for row in top1_rows:
        row["target_top1_indegree"] = int(top1_target_counter[row["target_element_id"]])

    # reciprocity
    reciprocal_top1 = 0
    for row in top1_rows:
        src = row["source_element_id"]
        tgt = row["target_element_id"]
        if src in adjacency.get(tgt, set()):
            reciprocal_top1 += 1
    reciprocal_top1_rate = reciprocal_top1 / len(top1_rows) if top1_rows else 0.0

    # suspicious sampling thresholds
    q_score = quantiles([r["score"] for r in top1_rows], (0.1, 0.5, 0.9, 0.95))
    q_jacc = quantiles([r["text_jaccard"] for r in top1_rows], (0.1, 0.5, 0.9))
    q_margin = quantiles([r["margin12"] for r in top1_rows], (0.1, 0.5, 0.9))
    indegrees = list(top1_target_counter.values())
    indegree_q = quantiles([float(x) for x in indegrees], (0.9, 0.95, 0.99)) if indegrees else {"p95": 0.0}

    score_hi = q_score.get("p90", 0.0)
    jacc_lo = q_jacc.get("p10", 0.0)
    margin_lo = q_margin.get("p10", 0.0)
    indegree_hi = max(6, int(indegree_q.get("p95", 0.0)))

    suspicious: List[Dict[str, Any]] = []
    for row in top1_rows:
        flags: List[str] = []
        if row["score"] >= score_hi and row["text_jaccard"] <= jacc_lo:
            flags.append("high_score_low_overlap")
        if row["margin12"] <= margin_lo:
            flags.append("low_top1_margin")
        if row["target_top1_indegree"] >= indegree_hi:
            flags.append("hub_target")
        if row["target_type"] != row["source_type"]:
            flags.append("type_mismatch")

        if flags:
            row2 = dict(row)
            row2["flags"] = "|".join(flags)
            row2["n_flags"] = len(flags)
            suspicious.append(row2)

    suspicious.sort(
        key=lambda r: (
            -int(r["n_flags"]),
            -float(r["score"]),
            float(r["text_jaccard"]),
            float(r["margin12"]),
        )
    )
    suspicious_sample = suspicious[: max(0, int(args.sample_size))]

    # write suspicious CSV
    suspicious_path.parent.mkdir(parents=True, exist_ok=True)
    with suspicious_path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "flags",
                "source_doc_id",
                "source_element_id",
                "source_type",
                "target_doc_id",
                "target_element_id",
                "target_type",
                "score",
                "score2",
                "margin12",
                "text_jaccard",
                "caption_jaccard",
                "target_top1_indegree",
                "source_caption",
                "target_caption",
            ]
        )
        for r in suspicious_sample:
            writer.writerow(
                [
                    r.get("flags", ""),
                    r.get("source_doc_id", ""),
                    r.get("source_element_id", ""),
                    r.get("source_type", ""),
                    r.get("target_doc_id", ""),
                    r.get("target_element_id", ""),
                    r.get("target_type", ""),
                    r.get("score", 0.0),
                    r.get("score2", 0.0),
                    r.get("margin12", 0.0),
                    r.get("text_jaccard", 0.0),
                    r.get("caption_jaccard", 0.0),
                    r.get("target_top1_indegree", 0),
                    r.get("source_caption", ""),
                    r.get("target_caption", ""),
                ]
            )

    # summary report
    report: Dict[str, Any] = {
        "input": str(matches_path),
        "elements": str(elements_path),
        "records": len(records),
        "source_docs": len(source_docs),
        "target_docs_observed": len(target_docs),
        "models": dict(model_names),
        "source_type_distribution": dict(source_type_counter),
        "top_k_length_distribution": {str(k): v for k, v in sorted(topk_len_counter.items())},
        "total_matches": total_matches,
        "score_overall": {
            "mean": round(mean(all_scores), 6) if all_scores else 0.0,
            "min": round(min(all_scores), 6) if all_scores else 0.0,
            "max": round(max(all_scores), 6) if all_scores else 0.0,
            **quantiles(all_scores, (0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99)),
        },
        "score_top1": {
            "mean": round(mean(top1_scores), 6) if top1_scores else 0.0,
            "min": round(min(top1_scores), 6) if top1_scores else 0.0,
            "max": round(max(top1_scores), 6) if top1_scores else 0.0,
            **quantiles(top1_scores, (0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99)),
        },
        "score_by_rank_mean": {
            str(rank): round(mean(vals), 6) for rank, vals in sorted(rank_scores.items())
        },
        "constraint_checks": {
            "same_doc_violations": same_doc_violations,
            "type_mismatches": type_mismatches,
        },
        "image_path_coverage": {
            "source_missing_rate": round(missing_source_image / len(records), 6) if records else 0.0,
            "target_missing_rate": round(missing_target_image / total_matches, 6) if total_matches else 0.0,
        },
        "hubness": {
            "unique_targets_top1": len(top1_target_counter),
            "unique_targets_all": len(all_target_counter),
            "top1_target_top10": top1_target_counter.most_common(10),
            "all_target_top10": all_target_counter.most_common(10),
            "top1_concentration_top10_rate": round(
                (sum(c for _, c in top1_target_counter.most_common(10)) / len(top1_rows))
                if top1_rows
                else 0.0,
                6,
            ),
            "all_concentration_top10_rate": round(
                (sum(c for _, c in all_target_counter.most_common(10)) / total_matches)
                if total_matches
                else 0.0,
                6,
            ),
        },
        "reciprocity": {
            "top1_reciprocal_count": reciprocal_top1,
            "top1_reciprocal_rate": round(reciprocal_top1_rate, 6),
        },
        "top1_quality_signals": {
            "text_jaccard": {
                "mean": round(mean([r["text_jaccard"] for r in top1_rows]), 6) if top1_rows else 0.0,
                **quantiles([r["text_jaccard"] for r in top1_rows], (0.1, 0.25, 0.5, 0.75, 0.9)),
            },
            "caption_jaccard": {
                "mean": round(mean([r["caption_jaccard"] for r in top1_rows]), 6) if top1_rows else 0.0,
                **quantiles([r["caption_jaccard"] for r in top1_rows], (0.1, 0.25, 0.5, 0.75, 0.9)),
            },
            "margin12": {
                "mean": round(mean([r["margin12"] for r in top1_rows]), 6) if top1_rows else 0.0,
                **quantiles([r["margin12"] for r in top1_rows], (0.1, 0.25, 0.5, 0.75, 0.9)),
            },
        },
        "suspicious_sampling": {
            "thresholds": {
                "score_hi_p90": score_hi,
                "text_jaccard_lo_p10": jacc_lo,
                "margin_lo_p10": margin_lo,
                "target_top1_indegree_hi": indegree_hi,
            },
            "num_suspicious_candidates": len(suspicious),
            "num_suspicious_sampled": len(suspicious_sample),
            "suspicious_csv": str(suspicious_path),
        },
    }

    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print("=" * 60)
    print("Cross-Doc Embedding Match Audit")
    print("=" * 60)
    print(f"Input records:     {len(records)}")
    print(f"Total matches:     {total_matches}")
    print(f"Source docs:       {len(source_docs)}")
    print(f"Types:             {dict(source_type_counter)}")
    print(f"Top-1 mean score:  {report['score_top1']['mean']:.6f}")
    print(f"Top-1 p90 score:   {report['score_top1']['p90']:.6f}")
    print(f"Top-1 reciprocity: {report['reciprocity']['top1_reciprocal_rate']:.4f}")
    print(f"Suspicious sample: {len(suspicious_sample)} -> {suspicious_path}")
    print(f"Report:            {report_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()
