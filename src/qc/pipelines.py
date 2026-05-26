"""QC composition pipelines — compose atomic checks into full quality gates.

Two pipelines:
  - qc_multihop_query: strict (academic style), 15+ checks, anchor amnesty mechanism
  - qc_real_user_query: relaxed (real-user style), no template checks, yes/no is soft

Each function returns (issues_list, metrics_dict); caller decides pass/fail.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Set, Tuple

from src.qc.constants import (
    ANCHOR_LEAK_THRESHOLD,
    BAD_META_PATTERNS,
    ENTITY_AMNESTY_TERMS,
    SHORT_QUERY_MAX_WORDS,
    TEXT_EVIDENCE_OVERLAP_WARN_THRESHOLD,
)
from src.qc.checks import (
    anchor_leak_jaccard,
    anchor_overlap_tokens,
    anchor_token_copy_count,
    answer_text_evidence_overlap,
    check_causal_chain_direction,
    check_evidence_spans,
    check_fact_distribution,
    check_no_shortcut,
    check_single_element_answer,
    formula_symbol_hit,
    has_architecture_intent,
    has_bare_deictic,
    has_bridge_overclaim_signal,
    has_conditional_hedge_overload,
    has_min_reasoning_chain,
    has_no_cross_modal_operator,
    has_numeric_leakage,
    has_parallel_dual_ask,
    has_premise_answer_contradiction,
    has_relationship_connector,
    has_semantic_category_mismatch,
    has_shortcut_template,
    has_template_collapse,
    has_templated_opening,
    is_architecture_pair,
    is_yes_no_answer,
    is_yes_no_question,
    query_length_bucket,
    query_word_count,
)
from src.qc.reasoning import classify_query_intent
from src.utils.text_utils import content_tokens, extract_formula_symbol_terms, number_tokens


def qc_multihop_query(
    obj: Dict[str, Any],
    pair: Dict[str, Any],
) -> Tuple[List[str], Dict[str, Any]]:
    """Strict academic-style QC — 15+ checks in sequence, no bad query escapes.

    Check order: meta-language → yes/no → numeric leakage → template → logic
    → anchor leakage → single-element answer → evidence length → formula grounding
    → reasoning connector → architecture intent → answer well-posedness.

    Anchor leakage has an amnesty mechanism: if overlapping tokens are all
    domain-essential terms (accuracy/f1 etc.), the issue is pardoned.

    Returns (issues_list, metrics_dict).
    """
    issues: List[str] = []
    metrics: Dict[str, Any] = {}
    q = obj.get("query", "")
    q_lower = q.lower().strip()
    a = obj.get("answer", "")
    anchors = obj.get("visual_anchors", [])
    q_words = query_word_count(q)
    q_bucket = query_length_bucket(q)
    metrics["query_word_count"] = q_words
    metrics["query_length_bucket"] = q_bucket

    # 1. Meta-language
    if any(re.search(p, q_lower) for p in BAD_META_PATTERNS):
        issues.append("meta_language")

    # 2. Yes/no question
    if is_yes_no_question(q):
        issues.append("yes_no_question")

    # 2b. Yes/no answer
    if is_yes_no_answer(a):
        issues.append("yes_no_answer")

    # 2c. Numeric leakage in query
    if has_numeric_leakage(q):
        issues.append("numeric_leakage")

    # 2c2. Bare deictic references (advisory, not hard fail)
    if has_bare_deictic(q):
        issues.append("bare_deictic")
        metrics["bare_deictic_warn"] = True

    # 2d. Weak shortcut templates
    if has_shortcut_template(q):
        issues.append("template_shortcut")

    # 2e. Templated opening collapse
    if has_templated_opening(q):
        issues.append("templated_opening")
        metrics["templated_opening_warn"] = True

    # 2ea. High-frequency shell patterns
    if has_template_collapse(q):
        issues.append("template_collapse")
        metrics["template_collapse_warn"] = True

    # 2f. Parallel dual-ask shortcut
    if has_parallel_dual_ask(q):
        issues.append("pseudo_multihop_parallel")
        metrics["parallel_dual_ask_warn"] = True

    # 2f2. HopWeaver: Fact Distribution — each hop must use different documents
    if not check_fact_distribution(obj, pair):
        issues.append("fact_distribution_violation")
        metrics["fact_distribution_pass"] = False

    # 2f3. HopWeaver: No-Shortcut — no single doc can have all evidence
    if not check_no_shortcut(obj, pair):
        issues.append("no_shortcut_violation")
        metrics["no_shortcut_pass"] = False

    # 2f4. Causal chain direction — premise→intermediate→conclusion
    causal_pass, causal_metrics = check_causal_chain_direction(obj)
    metrics.update({f"causal_{k}": v for k, v in causal_metrics.items()})
    if not causal_pass:
        issues.append("non_causal_chain")
        metrics["causal_chain_pass"] = False

    # 2g. Cross-category mismatch
    if has_semantic_category_mismatch(q):
        issues.append("semantic_category_mismatch")
        metrics["semantic_category_mismatch_warn"] = True

    # 2h. Reasoning chain requirement
    if not has_min_reasoning_chain(obj):
        issues.append("missing_reasoning_chain")
        metrics["reasoning_chain_warn"] = True

    # 3. Short answer
    if len(a) < 20:
        issues.append("short_answer")

    # 4. Empty query
    if not q or len(q) < 10:
        issues.append("empty_query")
    if q_bucket == "too_short":
        issues.append("query_too_short")
    if q_bucket == "too_long":
        issues.append("query_too_long")

    # 4b. Premise-answer contradiction
    if has_premise_answer_contradiction(q, a):
        issues.append("premise_answer_contradiction")
        metrics["premise_contradiction_warn"] = True

    # 5. Anchor leakage
    leak = anchor_leak_jaccard(q, anchors)
    metrics["anchor_leak_jaccard"] = round(leak, 4)
    ov_tokens = anchor_overlap_tokens(q, anchors)
    if leak > ANCHOR_LEAK_THRESHOLD:
        span_tokens: Set[str] = set()
        for s in obj.get("required_evidence_spans", []) or []:
            if isinstance(s, dict):
                span_tokens |= content_tokens(str(s.get("span", "")))
        allowed_tokens = ENTITY_AMNESTY_TERMS | span_tokens
        if ov_tokens and ov_tokens <= allowed_tokens:
            metrics["anchor_leakage_amnestied"] = True
        else:
            issues.append("anchor_leakage")
    anchor_copy = anchor_token_copy_count(q, anchors)
    metrics["anchor_token_copy_count"] = anchor_copy
    if anchor_copy >= 4:
        metrics["bridge_entity_leakage_warn"] = True

    # 5b. Cross-modal operator (advisory)
    metrics["has_cross_modal_operator"] = not has_no_cross_modal_operator(q)

    # 5c. Evidence spans
    if not check_evidence_spans(obj, pair):
        issues.append("evidence_spans_incomplete")

    # 6. Missing dual anchor
    elem_a_id = pair.get("element_a_id", "")
    elem_b_id = pair.get("element_b_id", "")
    anchor_element_ids = {
        a_item.get("element_id", "") for a_item in anchors if isinstance(a_item, dict)
    }
    if elem_a_id not in anchor_element_ids or elem_b_id not in anchor_element_ids:
        issues.append("missing_dual_anchor")
    metrics["anchor_count"] = len(anchors)

    # 7. Single-element answer
    a_toks = content_tokens(a)
    a_num_toks = number_tokens(a)
    if a_toks or a_num_toks:
        sea_fail, sea_metrics = check_single_element_answer(obj, pair, a_toks, a_num_toks)
        metrics.update(sea_metrics)
        if sea_fail:
            issues.append("single_element_answer")

    # 8. Text evidence length
    evidence = obj.get("text_evidence", "")
    if len(evidence) < 40:
        issues.append("short_evidence")

    # 8b. Evidence overlap
    text_ev_ov = answer_text_evidence_overlap(a, evidence)
    metrics["text_evidence_overlap"] = round(text_ev_ov, 4)
    if text_ev_ov > TEXT_EVIDENCE_OVERLAP_WARN_THRESHOLD:
        issues.append("text_evidence_over_reliance")

    # 8c. Figure+formula symbolic grounding
    pair_type = str(pair.get("pair_type", ""))
    if pair_type == "figure+formula":
        formula_elem = (
            pair.get("element_a", {})
            if pair.get("element_a_type") == "formula"
            else pair.get("element_b", {})
        )
        formula_text = (
            (formula_elem.get("caption", "") or "")
            + " "
            + (formula_elem.get("content", "") or "")
        )
        formula_terms = extract_formula_symbol_terms(formula_text)
        metrics["formula_symbol_term_count"] = len(formula_terms)
        metrics["formula_symbol_grounded"] = formula_symbol_hit(a, formula_terms)
        if formula_terms and not metrics["formula_symbol_grounded"]:
            issues.append("formula_symbol_grounding_missing")

    # 9. Weak reasoning connector
    qtype = str(obj.get("query_type", "")).lower()
    explanatory_types = {
        "trend_explanation", "anomaly_investigation", "bridge_reasoning",
        "theory_vs_experiment", "data_formula_consistency", "causal_explanation",
        "discrepancy_analysis", "hypothesis_verification",
    }
    if qtype in explanatory_types and not has_relationship_connector(a):
        issues.append("weak_reasoning_connector")

    # 10. Architecture check
    is_arch_case = is_architecture_pair(pair)
    metrics["is_architecture_case"] = bool(is_arch_case)
    if is_arch_case and not has_architecture_intent(q, a):
        issues.append("architecture_intent_missing")

    # 11. Conditional hedge overload (underdetermined query)
    if has_conditional_hedge_overload(a):
        issues.append("underdetermined_query")
        metrics["conditional_hedge_overload"] = True

    # 12. Bridge overclaim (query implies strong causation, answer hedges)
    if has_bridge_overclaim_signal(q, a):
        issues.append("bridge_overclaim")
        metrics["bridge_overclaim_warn"] = True

    return issues, metrics


def qc_real_user_query(
    obj: Dict[str, Any],
    pair: Dict[str, Any],
    persona: str = "",
) -> Tuple[List[str], Dict[str, Any]]:
    """Relaxed QC for real-user style queries — lower bar, broader tolerance.

    Compared to academic: no template shortcut checks, yes/no is soft issue
    only, adds retrievability_score (answer length + evidence + spans) and
    numeric_unsupported (answer numbers not found in evidence).
    """
    issues: List[str] = []
    metrics: Dict[str, Any] = {}
    q = obj.get("query", "")
    query_intent = classify_query_intent(q)
    metrics["query_intent"] = query_intent
    q_lower = q.lower().strip()
    a = obj.get("answer", "")
    q_words = query_word_count(q)
    metrics["query_word_count"] = q_words
    metrics["query_length_bucket"] = query_length_bucket(q)

    # 1. Meta-language
    if any(re.search(p, q_lower) for p in BAD_META_PATTERNS):
        issues.append("meta_language")

    # 2. Empty / length
    if not q.strip():
        issues.append("empty_query")
    elif q_words < 4:
        issues.append("query_too_short")
    elif q_words > 35:
        issues.append("query_too_long")

    # 2b. Skim-reader persona hard length constraint
    _skim_keywords = ("skim", "lazy", "busy", "conference_attendee")
    if persona and any(kw in persona for kw in _skim_keywords) and q_words > SHORT_QUERY_MAX_WORDS:
        issues.append("lazy_query_too_long")
    metrics["persona_applied"] = persona or "none"

    # 3. Yes/no question
    if is_yes_no_question(q):
        issues.append("yes_no_question")

    # 4. Single-element answer (objective only)
    a_toks = content_tokens(a)
    a_num_toks = number_tokens(a)
    if (a_toks or a_num_toks) and query_intent == "objective":
        sea_fail, sea_metrics = check_single_element_answer(obj, pair, a_toks, a_num_toks)
        metrics.update(sea_metrics)
        if sea_fail:
            issues.append("single_element_answer")

    # 5. Retrievability score
    retv = 0
    if len(a.strip()) > 30:
        retv += 1
    if obj.get("required_evidence_spans"):
        retv += 1
    evidence = obj.get("text_evidence", "")
    if len(evidence) >= 30:
        retv += 1
    metrics["retrievability_score"] = retv

    # 6. Evidence overlap
    evidence = obj.get("text_evidence", "")
    text_ev_ov = answer_text_evidence_overlap(a, evidence)
    metrics["text_evidence_overlap"] = round(text_ev_ov, 4)
    if text_ev_ov > TEXT_EVIDENCE_OVERLAP_WARN_THRESHOLD:
        issues.append("text_evidence_over_reliance")

    # 7. Numeric consistency
    answer_nums = number_tokens(a)
    source_parts: List[str] = []
    for key in ("element_a", "element_b"):
        elem = pair.get(key, {})
        for field in ("caption", "content", "enriched_content", "context_before", "context_after"):
            source_parts.append(str(elem.get(field, "") or ""))
    source_parts.append(str(evidence or ""))
    for s in obj.get("required_evidence_spans", []) or []:
        if isinstance(s, dict):
            source_parts.append(str(s.get("span", "") or ""))
    source_nums = number_tokens(" ".join(source_parts))
    unsupported_nums = sorted(answer_nums - source_nums)
    metrics["unsupported_answer_numbers"] = unsupported_nums
    if unsupported_nums:
        issues.append("numeric_unsupported")

    # 8. Figure+formula symbolic grounding
    pair_type = str(pair.get("pair_type", ""))
    if pair_type == "figure+formula":
        formula_elem = (
            pair.get("element_a", {})
            if pair.get("element_a_type") == "formula"
            else pair.get("element_b", {})
        )
        formula_text = (
            (formula_elem.get("caption", "") or "")
            + " "
            + (formula_elem.get("content", "") or "")
        )
        formula_terms = extract_formula_symbol_terms(formula_text)
        metrics["formula_symbol_term_count"] = len(formula_terms)
        metrics["formula_symbol_grounded"] = formula_symbol_hit(a, formula_terms)
        if formula_terms and not metrics["formula_symbol_grounded"]:
            issues.append("formula_symbol_grounding_missing")

    # 9. Conditional hedge overload (underdetermined query)
    if has_conditional_hedge_overload(a):
        issues.append("underdetermined_query")
        metrics["conditional_hedge_overload"] = True

    # 10. Bridge overclaim
    if has_bridge_overclaim_signal(q, a):
        issues.append("bridge_overclaim")
        metrics["bridge_overclaim_warn"] = True

    return issues, metrics
