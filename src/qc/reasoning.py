"""Reasoning structure classification and depth analysis.

Extracted from generate_multihop_l1_queries.py — M4 Schema 1 support.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Tuple

from src.qc.constants import (
    EVIDENCE_TYPE_PATTERNS,
    GENERIC_ANCHOR_PATTERNS,
    OBJECTIVE_QUERY_PATTERNS,
    PARALLEL_EVIDENCE_MARKERS,
    SERIAL_REASONING_MARKERS,
    SPECIFIC_ANCHOR_MARKERS,
    SUBJECTIVE_QUERY_PATTERNS,
)


def classify_query_intent(query: str) -> str:
    """Classify query as ``'objective'`` or ``'subjective'``.

    Heuristic rule-based; errs toward ``'objective'`` when uncertain so that QC
    checks remain strict by default.
    """
    q = (query or "").strip()
    subj_hits = sum(1 for p in SUBJECTIVE_QUERY_PATTERNS if p.search(q))
    obj_hits = sum(1 for p in OBJECTIVE_QUERY_PATTERNS if p.search(q))
    if subj_hits > obj_hits:
        return "subjective"
    return "objective"


def classify_reasoning_structure(
    reasoning_chain: str,
    answer: str,
    evidence_spans: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """Classify reasoning as parallel vs serial.

    Returns a dict with reasoning_depth_estimate, structure, marker lists,
    evidence_types, distinct_evidence_elements, and is_true_multihop.
    """
    combined_text = (reasoning_chain or "") + " " + (answer or "")

    serial_hits = SERIAL_REASONING_MARKERS.findall(combined_text)
    parallel_hits = PARALLEL_EVIDENCE_MARKERS.findall(combined_text)

    evidence_types = []
    for span_obj in (evidence_spans or []):
        span_text = str(span_obj.get("span", "")) if isinstance(span_obj, dict) else ""
        best_type = "observation"
        best_count = 0
        for etype, pat in EVIDENCE_TYPE_PATTERNS.items():
            hits = len(pat.findall(span_text))
            if hits > best_count:
                best_count = hits
                best_type = etype
        evidence_types.append(best_type)

    distinct_elements = len(set(
        str(s.get("element_id", ""))
        for s in (evidence_spans or [])
        if isinstance(s, dict) and s.get("element_id")
    ))

    n_serial = len(serial_hits)
    n_parallel = len(parallel_hits)

    if n_serial >= 3 and distinct_elements >= 3:
        depth = max(3, min(distinct_elements, n_serial))
        structure = "serial"
    elif n_serial >= 2 and distinct_elements >= 2:
        depth = max(2, distinct_elements)
        structure = "serial" if n_serial > n_parallel else "mixed"
    elif distinct_elements >= 2:
        depth = 2
        structure = "parallel"
    else:
        depth = 1
        structure = "parallel"

    flat_and_pattern = re.compile(r"\b(both .{5,60} and )\b", re.I)
    if flat_and_pattern.search(answer) and structure == "serial":
        structure = "mixed"

    is_true_multihop = structure == "serial" and depth >= 3

    return {
        "reasoning_depth_estimate": depth,
        "structure": structure,
        "serial_markers_found": serial_hits[:5],
        "parallel_markers_found": parallel_hits[:5],
        "evidence_types": evidence_types,
        "distinct_evidence_elements": distinct_elements,
        "is_true_multihop": is_true_multihop,
    }


def qc_reasoning_depth(
    obj: Dict[str, Any],
    pair: Dict[str, Any],
    min_depth: int = 3,
) -> Tuple[List[str], Dict[str, Any]]:
    """Reasoning depth analysis and auto-tagging (M4 Schema 1 support).

    Two modes:
      A) If obj contains explicit ``reasoning_steps[]``: structural validation
         with hard-fail issues.
      B) Otherwise: heuristic analysis with advisory metrics only.
    """
    issues: List[str] = []
    metrics: Dict[str, Any] = {}

    reasoning_chain = obj.get("reasoning_chain", "")
    answer = obj.get("answer", "")
    evidence_spans = obj.get("required_evidence_spans", []) or []
    reasoning_steps = obj.get("reasoning_steps", []) or []

    # ── Mode A: explicit reasoning_steps ──
    if reasoning_steps and len(reasoning_steps) >= 2:
        metrics["has_explicit_reasoning_steps"] = True
        metrics["num_reasoning_steps"] = len(reasoning_steps)

        has_dependency_chain = False
        for step in reasoning_steps[1:]:
            deps = step.get("depends_on_steps", [])
            if deps:
                has_dependency_chain = True
                break
        metrics["has_dependency_chain"] = has_dependency_chain
        if not has_dependency_chain:
            issues.append("reasoning_steps_no_dependencies")

        step_elements = [s.get("evidence_element_id", "") for s in reasoning_steps]
        unique_step_elements = set(e for e in step_elements if e)
        metrics["unique_step_elements"] = len(unique_step_elements)
        if len(unique_step_elements) < len(reasoning_steps):
            issues.append("reasoning_steps_duplicate_evidence")

        step_types = [s.get("evidence_type", "observation") for s in reasoning_steps]
        unique_types = set(step_types)
        metrics["reasoning_step_types"] = step_types
        if len(unique_types) < 2:
            issues.append("reasoning_steps_uniform_type")

        roles = [s.get("reasoning_role", "") for s in reasoning_steps]
        if roles:
            has_premise = "premise" in roles
            has_conclusion = "conclusion" in roles
            metrics["has_premise_conclusion_arc"] = has_premise and has_conclusion
            if not (has_premise and has_conclusion):
                issues.append("reasoning_steps_no_arc")

        depth = len(reasoning_steps)
        metrics["reasoning_depth"] = depth
        if depth < min_depth:
            issues.append(f"reasoning_depth_insufficient_{depth}_of_{min_depth}")

    else:
        # ── Mode B: heuristic analysis ──
        metrics["has_explicit_reasoning_steps"] = False

        analysis = classify_reasoning_structure(reasoning_chain, answer, evidence_spans)
        metrics.update({
            "reasoning_depth": analysis["reasoning_depth_estimate"],
            "reasoning_structure": analysis["structure"],
            "serial_markers": analysis["serial_markers_found"],
            "parallel_markers": analysis["parallel_markers_found"],
            "evidence_types_detected": analysis["evidence_types"],
            "distinct_evidence_elements": analysis["distinct_evidence_elements"],
            "is_true_multihop": analysis["is_true_multihop"],
        })

        depth = analysis["reasoning_depth_estimate"]
        if depth < min_depth:
            metrics["reasoning_depth_gap"] = min_depth - depth

    # ── Step-deletion heuristic ──
    causal_link_pattern = re.compile(
        r"(?:because|since|due to|as a result of|which (?:causes|leads|explains|means)|"
        r"therefore|thus|hence|consequently|this (?:causes|leads|explains|means))",
        re.IGNORECASE,
    )
    causal_links = causal_link_pattern.findall(answer)
    metrics["causal_link_count"] = len(causal_links)
    metrics["step_deletion_proxy"] = len(causal_links) >= (min_depth - 1)

    # ── P3: Hub-aware QC — bridge grounding check ──
    if reasoning_steps and len(reasoning_steps) >= 3:
        bridge_step = next(
            (s for s in reasoning_steps if s.get("reasoning_role") == "intermediate"),
            None,
        )
        if bridge_step:
            bridge_span = (bridge_step.get("evidence_span", "") or "").strip()
            bridge_claim = (bridge_step.get("produces_claim", "") or "").strip()
            if len(bridge_span) < 15:
                issues.append("bridge_span_too_short")
            if not bridge_claim or len(bridge_claim) < 10:
                issues.append("bridge_claim_empty")
            metrics["bridge_span_length"] = len(bridge_span)
            metrics["bridge_claim_length"] = len(bridge_claim)

    if pair.get("reasoning_chain_target") and metrics.get("reasoning_structure") == "parallel":
        if metrics.get("has_explicit_reasoning_steps"):
            if not metrics.get("has_dependency_chain", False):
                issues.append("l3_parallel_not_serial")

    # ── P4: Anchor specificity check ──
    visual_anchors = obj.get("visual_anchors", []) or []
    generic_anchor_count = 0
    specific_anchor_count = 0
    for va in visual_anchors:
        anchor_text = (va.get("anchor", "") or "").strip()
        if not anchor_text or len(anchor_text) < 5:
            generic_anchor_count += 1
        elif GENERIC_ANCHOR_PATTERNS.search(anchor_text):
            generic_anchor_count += 1
        elif SPECIFIC_ANCHOR_MARKERS.search(anchor_text):
            specific_anchor_count += 1
        else:
            if len(anchor_text.split()) >= 3:
                specific_anchor_count += 1
            else:
                generic_anchor_count += 1

    metrics["anchor_specificity"] = {
        "generic": generic_anchor_count,
        "specific": specific_anchor_count,
        "total": len(visual_anchors),
    }
    if visual_anchors and generic_anchor_count == len(visual_anchors):
        issues.append("all_anchors_generic")

    return issues, metrics
