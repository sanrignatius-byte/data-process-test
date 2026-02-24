#!/usr/bin/env python3
"""
Build contrastive triplets from dual-evidence L1 query JSONL.

Input:
  - L1 dual-evidence queries JSONL (query + element_ids + evidence metadata)
  - multimodal_elements.json (real extracted element text/image data)

Output:
  - Training triplets JSONL (query / positive / negatives)
  - Hard-negative audit CSV
  - Report JSON with distribution and quality diagnostics
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


TOKEN_RE = re.compile(r"[A-Za-z][A-Za-z0-9_-]{1,}")
REPO_ROOTS = (
    "/projects/_hdd/myyyx1/data-process-test/",
    "/projects/myyyx1/data-process-test/",
)


def _clean_text(text: str) -> str:
    text = text or ""
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _truncate(text: str, max_chars: int) -> str:
    if max_chars <= 0:
        return ""
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 3].rstrip() + "..."


def _tokens(text: str) -> set:
    return {t.lower() for t in TOKEN_RE.findall(text)}


def _jaccard(a: set, b: set) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def _overlap_ratio(a: set, b: set) -> float:
    """Asymmetric hardness-friendly overlap ratio."""
    if not a or not b:
        return 0.0
    return len(a & b) / max(1, min(len(a), len(b)))


def _first_image_path(paths: List[str]) -> Optional[str]:
    for p in paths or []:
        if isinstance(p, str) and p.strip():
            return p
    return None


def _normalize_path(path: Optional[str]) -> Optional[str]:
    if not path:
        return None
    p = str(path).strip()
    if not p:
        return None
    for root in REPO_ROOTS:
        if p.startswith(root):
            return p[len(root):]
    return p


def _dedupe_paths(paths: List[Optional[str]]) -> List[str]:
    out: List[str] = []
    seen = set()
    for p in paths:
        if not p:
            continue
        if p in seen:
            continue
        seen.add(p)
        out.append(p)
    return out


def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def _build_element_text(
    elem: Dict[str, Any],
    max_caption_chars: int,
    max_content_chars: int,
    max_context_chars: int,
    include_context: bool,
) -> str:
    etype = _clean_text(str(elem.get("element_type", "unknown")))
    caption = _truncate(_clean_text(str(elem.get("caption", ""))), max_caption_chars)
    content = _truncate(_clean_text(str(elem.get("content", ""))), max_content_chars)
    context_before = _truncate(_clean_text(str(elem.get("context_before", ""))), max_context_chars)
    context_after = _truncate(_clean_text(str(elem.get("context_after", ""))), max_context_chars)

    parts = [f"Evidence modality: {etype}."]
    if caption:
        parts.append(f"Caption text: {caption}")
    if content:
        parts.append(f"Extracted content: {content}")
    if include_context and context_before:
        parts.append(f"Preceding context: {context_before}")
    if include_context and context_after:
        parts.append(f"Following context: {context_after}")
    return "\n".join(parts)


def _index_elements(multimodal_path: Path) -> Dict[str, Dict[str, Dict[str, Any]]]:
    data = json.loads(multimodal_path.read_text(encoding="utf-8"))
    docs = data.get("documents", {})
    out: Dict[str, Dict[str, Dict[str, Any]]] = {}
    for doc_id, doc in docs.items():
        out[doc_id] = doc.get("elements", {})
    return out


def _build_query_span_index(rows: List[Dict[str, Any]]) -> Dict[Tuple[str, str], List[str]]:
    idx: Dict[Tuple[str, str], List[str]] = defaultdict(list)
    for row in rows:
        doc_id = str(row.get("doc_id", ""))
        for span_obj in row.get("required_evidence_spans", []) or []:
            if not isinstance(span_obj, dict):
                continue
            eid = str(span_obj.get("element_id", "")).strip()
            span = _clean_text(str(span_obj.get("span", "")))
            if doc_id and eid and span:
                idx[(doc_id, eid)].append(span)
    return idx


def _build_text_evidence_index(rows: List[Dict[str, Any]]) -> Dict[Tuple[str, str], List[str]]:
    idx: Dict[Tuple[str, str], List[str]] = defaultdict(list)
    for row in rows:
        doc_id = str(row.get("doc_id", ""))
        text_evidence = _clean_text(str(row.get("text_evidence", "")))
        if not text_evidence:
            continue
        for eid in row.get("element_ids", []) or []:
            if doc_id and eid:
                idx[(doc_id, str(eid))].append(text_evidence)
    return idx


def _best_span_for_element(
    doc_id: str,
    eid: str,
    span_index: Dict[Tuple[str, str], List[str]],
    fallback_element_text: str,
    max_chars: int = 180,
) -> str:
    spans = span_index.get((doc_id, eid), [])
    if spans:
        return _truncate(spans[0], max_chars)
    return _truncate(fallback_element_text, max_chars)


def _element_type_for_id(doc_elements: Dict[str, Dict[str, Any]], eid: str) -> str:
    elem = doc_elements.get(eid, {})
    return str(elem.get("element_type", "unknown"))


def _build_row_image_map(row: Dict[str, Any], doc_elements: Dict[str, Dict[str, Any]]) -> Dict[str, str]:
    out: Dict[str, str] = {}
    eids = [str(x) for x in (row.get("element_ids", []) or [])]
    img_paths = list(row.get("image_paths", []) or [])

    for idx, eid in enumerate(eids):
        if idx < len(img_paths):
            p = _normalize_path(str(img_paths[idx]))
            if p:
                out[eid] = p

    for eid in eids:
        if eid in out:
            continue
        elem = doc_elements.get(eid, {})
        p = _normalize_path(elem.get("image_path"))
        if p:
            out[eid] = p

    return out


def _make_bundle_text(element_a_text: str, element_b_text: str, bridge_text: str = "") -> str:
    parts = [
        "Evidence unit 1:",
        element_a_text,
        "",
        "Evidence unit 2:",
        element_b_text,
    ]
    if bridge_text:
        parts.extend(["", f"Shared paper context: {bridge_text}"])
    return "\n".join(parts).strip()


def _percentile(values: List[float], q: float) -> float:
    if not values:
        return 0.0
    arr = sorted(values)
    idx = int(round((len(arr) - 1) * q))
    idx = max(0, min(len(arr) - 1, idx))
    return float(arr[idx])


def _score_stats(scores: List[float]) -> Dict[str, float]:
    if not scores:
        return {"count": 0, "min": 0.0, "mean": 0.0, "p50": 0.0, "p90": 0.0, "max": 0.0}
    return {
        "count": len(scores),
        "min": round(min(scores), 4),
        "mean": round(sum(scores) / len(scores), 4),
        "p50": round(_percentile(scores, 0.5), 4),
        "p90": round(_percentile(scores, 0.9), 4),
        "max": round(max(scores), 4),
    }


def _pick_in_doc_swap_negative(
    row: Dict[str, Any],
    doc_elements: Dict[str, Dict[str, Any]],
    span_index: Dict[Tuple[str, str], List[str]],
    text_evidence_index: Dict[Tuple[str, str], List[str]],
    element_text_cache_full: Dict[Tuple[str, str], str],
    element_text_cache_short: Dict[Tuple[str, str], str],
    row_image_map: Dict[str, str],
    rng: random.Random,
) -> Optional[Dict[str, Any]]:
    doc_id = str(row.get("doc_id", ""))
    pair_type = str(row.get("pair_type", "unknown"))
    element_ids = list(row.get("element_ids", []) or [])
    if len(element_ids) != 2:
        return None
    a_id, b_id = element_ids[0], element_ids[1]
    if a_id not in doc_elements or b_id not in doc_elements:
        return None

    a_type = _element_type_for_id(doc_elements, a_id)
    b_type = _element_type_for_id(doc_elements, b_id)
    all_ids_by_type: Dict[str, List[str]] = defaultdict(list)
    for eid, elem in doc_elements.items():
        all_ids_by_type[str(elem.get("element_type", "unknown"))].append(eid)

    def _pick_distractor(anchor_id: str, swap_type: str, original_swap_id: str) -> Optional[Tuple[str, float]]:
        cands = [eid for eid in all_ids_by_type.get(swap_type, []) if eid not in {anchor_id, original_swap_id}]
        if not cands:
            return None
        orig_text = element_text_cache_short[(doc_id, original_swap_id)]
        orig_tokens = _tokens(orig_text)
        scored = []
        for eid in cands:
            sim = _jaccard(orig_tokens, _tokens(element_text_cache_short[(doc_id, eid)]))
            scored.append((eid, sim))
        scored.sort(key=lambda x: x[1], reverse=True)
        top_k = scored[: min(3, len(scored))]
        return rng.choice(top_k) if top_k else None

    swap_plan = [
        (a_id, a_type, b_id, b_type),
        (b_id, b_type, a_id, a_type),
    ]

    for anchor_id, anchor_type, orig_other_id, other_type in swap_plan:
        picked = _pick_distractor(anchor_id, other_type, orig_other_id)
        if not picked:
            continue
        distractor_id, swap_sim = picked

        anchor_text_full = element_text_cache_full[(doc_id, anchor_id)]
        anchor_text_short = element_text_cache_short[(doc_id, anchor_id)]
        distractor_text_full = element_text_cache_full[(doc_id, distractor_id)]
        distractor_text_short = element_text_cache_short[(doc_id, distractor_id)]

        anchor_span = _best_span_for_element(doc_id, anchor_id, span_index, anchor_text_short)
        distractor_span = _best_span_for_element(doc_id, distractor_id, span_index, distractor_text_short)

        evidence_candidates = text_evidence_index.get((doc_id, distractor_id), [])
        distractor_evidence = evidence_candidates[0] if evidence_candidates else ""

        bridge = _truncate(distractor_evidence, 320) if distractor_evidence else ""
        negative_text = _make_bundle_text(anchor_text_full, distractor_text_full, bridge)
        negative_text_short = _make_bundle_text(
            anchor_text_short,
            distractor_text_short,
            _truncate(bridge, 180) if bridge else "",
        )

        anchor_img = row_image_map.get(anchor_id) or _normalize_path(doc_elements.get(anchor_id, {}).get("image_path"))
        distractor_img = _normalize_path(doc_elements.get(distractor_id, {}).get("image_path"))
        image_paths = _dedupe_paths([anchor_img, distractor_img])

        return {
            "text": negative_text,
            "text_short": negative_text_short,
            "modal_type": "dual_evidence_bundle",
            "negative_type": "in_doc_swap",
            "image_paths": image_paths,
            "image_path": _first_image_path(image_paths),
            "metadata": {
                "doc_id": doc_id,
                "pair_type": pair_type,
                "anchor_element_id": anchor_id,
                "distractor_element_id": distractor_id,
                "anchor_element_type": anchor_type,
                "distractor_element_type": other_type,
                "anchor_span": anchor_span,
                "distractor_span": distractor_span,
                "distractor_context_evidence": _truncate(distractor_evidence, 320),
                "source_positive_pair_id": row.get("pair_id", ""),
                "swap_side_type": other_type,
            },
            "score": round(float(swap_sim), 4),
        }

    return None


def _pick_same_type_hard_negative(
    row: Dict[str, Any],
    candidate_rows: List[Dict[str, Any]],
    row_to_positive_payload: Dict[str, Dict[str, Any]],
    row_query_tokens: Dict[str, set],
    row_bundle_tokens: Dict[str, set],
    row_span_tokens: Dict[str, set],
    row_bridge_tokens: Dict[str, set],
) -> Optional[Dict[str, Any]]:
    pair_type = str(row.get("pair_type", ""))
    query_type = str(row.get("query_type", ""))
    quality_tier = str(row.get("quality_tier", "unknown"))
    doc_id = str(row.get("doc_id", ""))
    qid = str(row.get("query_id", ""))

    q_tokens = row_query_tokens.get(qid, set())
    p_tokens = row_bundle_tokens.get(qid, set())
    s_tokens = row_span_tokens.get(qid, set())
    b_tokens = row_bridge_tokens.get(qid, set())
    if not q_tokens and not p_tokens:
        return None

    ranked: List[Tuple[Dict[str, Any], float, Dict[str, float]]] = []
    for cand in candidate_rows:
        cand_qid = str(cand.get("query_id", ""))
        if cand_qid == qid:
            continue
        if str(cand.get("doc_id", "")) == doc_id:
            continue
        if str(cand.get("pair_type", "")) != pair_type:
            continue
        if cand_qid not in row_to_positive_payload:
            continue

        cq = row_query_tokens.get(cand_qid, set())
        cp = row_bundle_tokens.get(cand_qid, set())
        cs = row_span_tokens.get(cand_qid, set())
        cb = row_bridge_tokens.get(cand_qid, set())

        sim_q = _jaccard(q_tokens, cq)
        sim_p = _jaccard(p_tokens, cp)
        sim_s = _jaccard(s_tokens, cs)
        sim_b = _jaccard(b_tokens, cb)
        over_q = _overlap_ratio(q_tokens, cq)

        score = (
            0.45 * sim_q
            + 0.20 * sim_p
            + 0.20 * sim_s
            + 0.10 * sim_b
            + 0.05 * over_q
        )
        if str(cand.get("query_type", "")) == query_type:
            score += 0.05
        if str(cand.get("quality_tier", "unknown")) == quality_tier:
            score += 0.02

        ranked.append(
            (
                cand,
                float(min(score, 1.0)),
                {
                    "sim_query": round(sim_q, 4),
                    "sim_bundle": round(sim_p, 4),
                    "sim_span": round(sim_s, 4),
                    "sim_bridge": round(sim_b, 4),
                    "overlap_query": round(over_q, 4),
                },
            )
        )

    if not ranked:
        return None

    ranked.sort(key=lambda x: x[1], reverse=True)
    cand, score, score_parts = ranked[0]

    cand_qid = str(cand.get("query_id", ""))
    payload = row_to_positive_payload.get(cand_qid, {})
    neg_text = str(payload.get("text", ""))
    neg_text_short = str(payload.get("text_short", ""))
    image_paths = list(payload.get("image_paths", []) or [])
    if not neg_text:
        return None

    return {
        "text": neg_text,
        "text_short": neg_text_short,
        "modal_type": "dual_evidence_bundle",
        "negative_type": "same_type_hard_plus",
        "image_paths": image_paths,
        "image_path": _first_image_path(image_paths),
        "metadata": {
            "source_query_id": cand_qid,
            "source_doc_id": cand.get("doc_id", ""),
            "source_pair_id": cand.get("pair_id", ""),
            "pair_type": pair_type,
            "query_type": cand.get("query_type", ""),
            "source_quality_tier": cand.get("quality_tier", "unknown"),
            "score_components": score_parts,
        },
        "score": round(float(score), 4),
    }


def _make_positive_bundle_payload(
    row: Dict[str, Any],
    doc_elements: Dict[str, Dict[str, Any]],
    span_index: Dict[Tuple[str, str], List[str]],
    element_text_cache_full: Dict[Tuple[str, str], str],
    element_text_cache_short: Dict[Tuple[str, str], str],
    row_image_map: Dict[str, str],
    max_bridge_chars: int,
) -> Dict[str, Any]:
    doc_id = str(row.get("doc_id", ""))
    element_ids = list(row.get("element_ids", []) or [])
    if len(element_ids) != 2:
        return {}
    a_id, b_id = element_ids[0], element_ids[1]

    if a_id not in doc_elements or b_id not in doc_elements:
        return {}

    a_text_full = element_text_cache_full[(doc_id, a_id)]
    b_text_full = element_text_cache_full[(doc_id, b_id)]
    a_text_short = element_text_cache_short[(doc_id, a_id)]
    b_text_short = element_text_cache_short[(doc_id, b_id)]

    bridge = _truncate(_clean_text(str(row.get("text_evidence", ""))), max_bridge_chars)
    bridge_short = _truncate(bridge, 180) if bridge else ""

    text = _make_bundle_text(a_text_full, b_text_full, bridge)
    text_short = _make_bundle_text(a_text_short, b_text_short, bridge_short)

    a_span = _best_span_for_element(doc_id, a_id, span_index, a_text_short)
    b_span = _best_span_for_element(doc_id, b_id, span_index, b_text_short)

    img_a = row_image_map.get(a_id) or _normalize_path(doc_elements.get(a_id, {}).get("image_path"))
    img_b = row_image_map.get(b_id) or _normalize_path(doc_elements.get(b_id, {}).get("image_path"))
    image_paths = _dedupe_paths([img_a, img_b])

    return {
        "text": text,
        "text_short": text_short,
        "image_paths": image_paths,
        "image_path": _first_image_path(image_paths),
        "span_by_element": {a_id: a_span, b_id: b_span},
        "bridge_text": bridge,
    }


def build_triplets(args: argparse.Namespace) -> None:
    raw_query_rows = _read_jsonl(Path(args.queries))
    query_rows = raw_query_rows
    if args.pass_only:
        query_rows = [r for r in query_rows if bool(r.get("qc_pass", False))]

    elements_by_doc = _index_elements(Path(args.elements))
    span_index = _build_query_span_index(query_rows)
    text_evidence_index = _build_text_evidence_index(query_rows)
    rng = random.Random(args.seed)

    element_text_cache_full: Dict[Tuple[str, str], str] = {}
    element_text_cache_short: Dict[Tuple[str, str], str] = {}
    for doc_id, elems in elements_by_doc.items():
        for eid, elem in elems.items():
            element_text_cache_full[(doc_id, eid)] = _build_element_text(
                elem=elem,
                max_caption_chars=args.max_caption_chars,
                max_content_chars=args.max_content_chars,
                max_context_chars=args.max_context_chars,
                include_context=True,
            )
            element_text_cache_short[(doc_id, eid)] = _build_element_text(
                elem=elem,
                max_caption_chars=args.max_short_caption_chars,
                max_content_chars=args.max_short_content_chars,
                max_context_chars=args.max_short_context_chars,
                include_context=False,
            )

    usable_rows: List[Dict[str, Any]] = []
    missing_element_rows = 0
    for row in query_rows:
        doc_id = str(row.get("doc_id", ""))
        eids = list(row.get("element_ids", []) or [])
        if len(eids) != 2:
            continue
        if doc_id not in elements_by_doc:
            missing_element_rows += 1
            continue
        if eids[0] not in elements_by_doc[doc_id] or eids[1] not in elements_by_doc[doc_id]:
            missing_element_rows += 1
            continue
        usable_rows.append(row)

    row_to_positive_payload: Dict[str, Dict[str, Any]] = {}
    row_query_tokens: Dict[str, set] = {}
    row_bundle_tokens: Dict[str, set] = {}
    row_span_tokens: Dict[str, set] = {}
    row_bridge_tokens: Dict[str, set] = {}

    for row in usable_rows:
        qid = str(row.get("query_id", ""))
        doc_id = str(row.get("doc_id", ""))
        row_image_map = _build_row_image_map(row, elements_by_doc[doc_id])
        payload = _make_positive_bundle_payload(
            row=row,
            doc_elements=elements_by_doc[doc_id],
            span_index=span_index,
            element_text_cache_full=element_text_cache_full,
            element_text_cache_short=element_text_cache_short,
            row_image_map=row_image_map,
            max_bridge_chars=args.max_bridge_chars,
        )
        if not payload:
            continue

        row_to_positive_payload[qid] = payload
        row_query_tokens[qid] = _tokens(str(row.get("query", "")))
        row_bundle_tokens[qid] = _tokens(str(payload.get("text_short", payload.get("text", ""))))

        span_tokens = set()
        for s in (row.get("required_evidence_spans", []) or []):
            if isinstance(s, dict):
                span_tokens |= _tokens(str(s.get("span", "")))
        row_span_tokens[qid] = span_tokens
        row_bridge_tokens[qid] = _tokens(str(row.get("text_evidence", "")))

    triplets: List[Dict[str, Any]] = []
    neg_stats = Counter()
    neg_score_by_type: Dict[str, List[float]] = defaultdict(list)
    skipped_no_negative = 0

    for row in usable_rows:
        qid = str(row.get("query_id", ""))
        doc_id = str(row.get("doc_id", ""))
        positive_payload = row_to_positive_payload.get(qid, {})
        if not positive_payload:
            continue

        row_image_map = _build_row_image_map(row, elements_by_doc[doc_id])
        in_doc_swap = _pick_in_doc_swap_negative(
            row=row,
            doc_elements=elements_by_doc[doc_id],
            span_index=span_index,
            text_evidence_index=text_evidence_index,
            element_text_cache_full=element_text_cache_full,
            element_text_cache_short=element_text_cache_short,
            row_image_map=row_image_map,
            rng=rng,
        )
        same_type_hard = _pick_same_type_hard_negative(
            row=row,
            candidate_rows=usable_rows,
            row_to_positive_payload=row_to_positive_payload,
            row_query_tokens=row_query_tokens,
            row_bundle_tokens=row_bundle_tokens,
            row_span_tokens=row_span_tokens,
            row_bridge_tokens=row_bridge_tokens,
        )

        negatives = [n for n in [in_doc_swap, same_type_hard] if n is not None]
        if not negatives:
            skipped_no_negative += 1
            continue

        for n in negatives:
            neg_stats[n["negative_type"]] += 1
            neg_score_by_type[n["negative_type"]].append(float(n.get("score", 0.0)))

        hard_scores = [float(n.get("score", 0.0)) for n in negatives]
        difficulty = 0.35 if not hard_scores else min(1.0, 0.35 + max(hard_scores))

        triplets.append(
            {
                "triplet_id": f"{qid}_triplet",
                "source_query_id": qid,
                "query": str(row.get("query", "")),
                "query_type": str(row.get("query_type", "unknown")),
                "positive": {
                    "text": str(positive_payload.get("text", "")),
                    "text_short": str(positive_payload.get("text_short", "")),
                    "modal_type": "dual_evidence_bundle",
                    "image_paths": list(positive_payload.get("image_paths", []) or []),
                    "image_path": positive_payload.get("image_path"),
                    "metadata": {
                        "doc_id": doc_id,
                        "pair_id": row.get("pair_id", ""),
                        "pair_type": row.get("pair_type", ""),
                        "element_ids": row.get("element_ids", []),
                        "image_paths": list(positive_payload.get("image_paths", []) or []),
                        "quality_tier": row.get("quality_tier", "unknown"),
                        "required_evidence_spans": row.get("required_evidence_spans", []),
                        "span_by_element": positive_payload.get("span_by_element", {}),
                        "bridge_evidence": positive_payload.get("bridge_text", ""),
                    },
                },
                "negatives": negatives,
                "difficulty_score": round(difficulty, 4),
                "metadata": {
                    "doc_id": doc_id,
                    "pair_type": row.get("pair_type", ""),
                    "qc_pass": bool(row.get("qc_pass", False)),
                    "negative_types": [n["negative_type"] for n in negatives],
                },
            }
        )

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for t in triplets:
            f.write(json.dumps(t, ensure_ascii=False) + "\n")

    audit_path = Path(args.audit_csv)
    with audit_path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "triplet_id",
                "source_query_id",
                "doc_id",
                "pair_type",
                "query_type",
                "negative_type",
                "negative_score",
                "negative_source_doc_id",
                "negative_source_pair_id",
                "score_sim_query",
                "score_sim_bundle",
                "score_sim_span",
                "score_sim_bridge",
            ]
        )
        for t in triplets:
            for neg in t.get("negatives", []):
                md = neg.get("metadata", {}) or {}
                score_parts = md.get("score_components", {}) or {}
                writer.writerow(
                    [
                        t.get("triplet_id", ""),
                        t.get("source_query_id", ""),
                        t.get("metadata", {}).get("doc_id", ""),
                        t.get("metadata", {}).get("pair_type", ""),
                        t.get("query_type", ""),
                        neg.get("negative_type", ""),
                        neg.get("score", 0.0),
                        md.get("source_doc_id", md.get("doc_id", "")),
                        md.get("source_pair_id", ""),
                        score_parts.get("sim_query", ""),
                        score_parts.get("sim_bundle", ""),
                        score_parts.get("sim_span", ""),
                        score_parts.get("sim_bridge", ""),
                    ]
                )

    pair_type_dist = Counter(t.get("metadata", {}).get("pair_type", "unknown") for t in triplets)
    query_type_dist = Counter(t.get("query_type", "unknown") for t in triplets)
    neg_per_triplet = [len(t.get("negatives", [])) for t in triplets]
    difficulty_vals = [float(t.get("difficulty_score", 0.0)) for t in triplets]

    pos_text_chars = [len(str(t.get("positive", {}).get("text", ""))) for t in triplets]
    pos_short_chars = [len(str(t.get("positive", {}).get("text_short", ""))) for t in triplets]
    neg_text_chars = [
        len(str(n.get("text", "")))
        for t in triplets
        for n in (t.get("negatives", []) or [])
    ]
    neg_short_chars = [
        len(str(n.get("text_short", "")))
        for t in triplets
        for n in (t.get("negatives", []) or [])
    ]

    pos_has_images = sum(1 for t in triplets if (t.get("positive", {}).get("image_paths") or []))
    neg_has_images = sum(
        1 for t in triplets for n in (t.get("negatives", []) or []) if (n.get("image_paths") or [])
    )
    total_negs = sum(len(t.get("negatives", []) or []) for t in triplets)

    report = {
        "input_queries": str(args.queries),
        "input_elements": str(args.elements),
        "output_triplets": str(args.output),
        "pass_only": bool(args.pass_only),
        "seed": args.seed,
        "raw_total_queries": len(raw_query_rows),
        "total_queries_loaded": len(query_rows),
        "pass_filter_drop": len(raw_query_rows) - len(query_rows),
        "usable_queries": len(usable_rows),
        "missing_element_rows": missing_element_rows,
        "triplets_written": len(triplets),
        "skipped_no_negative": skipped_no_negative,
        "pair_type_distribution": dict(pair_type_dist),
        "query_type_distribution": dict(query_type_dist),
        "negative_type_distribution": dict(neg_stats),
        "avg_negatives_per_triplet": round(sum(neg_per_triplet) / max(1, len(neg_per_triplet)), 4),
        "avg_difficulty_score": round(sum(difficulty_vals) / max(1, len(difficulty_vals)), 4),
        "positive_text_chars_avg": round(sum(pos_text_chars) / max(1, len(pos_text_chars)), 2),
        "positive_text_short_chars_avg": round(sum(pos_short_chars) / max(1, len(pos_short_chars)), 2),
        "negative_text_chars_avg": round(sum(neg_text_chars) / max(1, len(neg_text_chars)), 2),
        "negative_text_short_chars_avg": round(sum(neg_short_chars) / max(1, len(neg_short_chars)), 2),
        "image_coverage": {
            "positive_with_images": pos_has_images,
            "positive_total": len(triplets),
            "negative_with_images": neg_has_images,
            "negative_total": total_negs,
            "positive_ratio": round(pos_has_images / max(1, len(triplets)), 4),
            "negative_ratio": round(neg_has_images / max(1, total_negs), 4),
        },
        "negative_score_stats": {
            ntype: _score_stats(scores)
            for ntype, scores in neg_score_by_type.items()
        },
    }

    report_path = Path(args.report_json)
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print("=" * 60)
    print("Dual-Evidence Triplet Build Summary")
    print("=" * 60)
    print(f"Loaded queries:         {len(query_rows)}")
    print(f"Usable queries:         {len(usable_rows)}")
    print(f"Triplets written:       {len(triplets)}")
    print(f"Skipped no negatives:   {skipped_no_negative}")
    print(f"Missing element rows:   {missing_element_rows}")
    print(f"Negative types:         {dict(neg_stats)}")
    print(f"Image coverage:         pos={pos_has_images}/{len(triplets)} neg={neg_has_images}/{total_negs}")
    print(f"Output:                 {out_path}")
    print(f"Audit CSV:              {audit_path}")
    print(f"Report:                 {report_path}")
    print("=" * 60)


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Build triplets from dual-evidence query JSONL")
    ap.add_argument(
        "--queries",
        default="data/l1_dual_evidence_queries_slurm_img150_tuned_v4_official.jsonl",
        help="Input dual-evidence query JSONL",
    )
    ap.add_argument(
        "--elements",
        default="data/multimodal_elements.json",
        help="Input multimodal elements JSON",
    )
    ap.add_argument(
        "--output",
        default="data/l1_dual_evidence_triplets_v2.jsonl",
        help="Output triplet JSONL",
    )
    ap.add_argument(
        "--audit-csv",
        default="data/l1_dual_evidence_triplets_v2_hardneg_audit.csv",
        help="Output hard-negative audit CSV",
    )
    ap.add_argument(
        "--report-json",
        default="data/l1_dual_evidence_triplets_v2_report.json",
        help="Output summary report JSON",
    )
    ap.add_argument(
        "--pass-only",
        action="store_true",
        help="Use only qc_pass=true rows from query file",
    )
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max-caption-chars", type=int, default=260)
    ap.add_argument("--max-content-chars", type=int, default=420)
    ap.add_argument("--max-context-chars", type=int, default=220)
    ap.add_argument("--max-short-caption-chars", type=int, default=180)
    ap.add_argument("--max-short-content-chars", type=int, default=220)
    ap.add_argument("--max-short-context-chars", type=int, default=0)
    ap.add_argument("--max-bridge-chars", type=int, default=320)
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    build_triplets(args)


if __name__ == "__main__":
    main()
