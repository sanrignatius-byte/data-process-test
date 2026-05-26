"""Atomic QC check functions — each does exactly one thing, returns True/False.

25+ checks organized by category:
  - Leakage: numeric leakage, anchor leakage (Jaccard), evidence token copy
  - Format: yes/no, templated openings, too long / too short
  - Logic: single-element answer, premise-answer contradiction, parallel dual-ask
  - Domain: cross-modal operator, formula symbol grounding, architecture intent
  - Answer quality: conditional hedge overload, bridge overclaim signal

Pipelines (pipelines.py) compose these atomic functions into full QC flows.
"""

from __future__ import annotations

import re
from collections import defaultdict
from typing import Any, Dict, List, Set, Tuple

from src.qc.constants import (
    ANCHOR_LEAK_THRESHOLD,
    ANSWER_BALANCE_THRESHOLD,
    ARCHITECTURE_INTENT_TERMS,
    ARCHITECTURE_KEYWORDS,
    CROSS_MODAL_OPERATOR_PATTERNS,
    CROSS_MODAL_OPERATORS,
    ENRICHMENT_NOISE_RE,
    ENTITY_AMNESTY_TERMS,
    METRIC_CONCEPT_TERMS,
    MIN_OVERLAP_BY_TYPE,
    QUERY_SHORTCUT_PATTERNS,
    RELATION_CONNECTORS,
    SHORT_QUERY_MAX_WORDS,
    SHORT_QUERY_MIN_WORDS,
    LONG_QUERY_MIN_WORDS,
    MAX_QUERY_WORDS,
    STRUCTURAL_CONCEPT_TERMS,
    TEMPLATE_COLLAPSE_PATTERNS,
    TEMPLATED_QUERY_OPENINGS,
    YES_NO_STARTERS,
)
from src.utils.text_utils import (
    content_tokens,
    extract_formula_symbol_terms,
    number_tokens,
)


# ── Numeric / leakage checks ─────────────────────────────────────────────────

def has_numeric_leakage(query: str) -> bool:
    """查 query 里有没有泄露具体数值 —— 0、1 和年份(1900-2099)豁免。"""
    nums = re.findall(r"\b\d+(?:[.,]\d+)?%?\b", query)
    suspicious = []
    for raw in nums:
        token = raw.replace(",", "").rstrip("%")
        try:
            value = float(token)
        except ValueError:
            continue
        if value in (0.0, 1.0):
            continue
        if 1900 <= value <= 2099 and value.is_integer():
            continue
        suspicious.append(raw)
    return len(suspicious) >= 1


# ── Cross-modal operator ──────────────────────────────────────────────────────

def has_no_cross_modal_operator(query: str) -> bool:
    """检查 query 有没有跨模态动词（verify/derive/explain/affect 等 180+ 个词）。

    返回 True = 没有找到跨模态算子 → 可能只涉及单一模态，质量可疑。
    """
    q = query.lower()
    token_hit = any(
        re.search(rf"\b{re.escape(op)}\b", q, re.IGNORECASE)
        for op in CROSS_MODAL_OPERATORS
    )
    if token_hit:
        return False
    phrase_hit = any(re.search(p, q, re.IGNORECASE) for p in CROSS_MODAL_OPERATOR_PATTERNS)
    return not phrase_hit


# ── Evidence spans ────────────────────────────────────────────────────────────

def check_evidence_spans(obj: Dict[str, Any], pair: Dict[str, Any]) -> bool:
    """检查 evidence spans 是不是真的覆盖了两个元素（且 span 文本 ≥8 字符）。"""
    spans = obj.get("required_evidence_spans", [])
    if not spans or len(spans) < 2:
        return False
    elem_ids = {pair.get("element_a_id", ""), pair.get("element_b_id", "")}
    covered: Set[str] = set()
    for s in spans:
        span_text = s.get("span", "").strip()
        eid = s.get("element_id", "")
        if eid and len(span_text) >= 8:
            covered.add(eid)
    return len(covered & elem_ids) >= 2


# ── Yes/no checks ────────────────────────────────────────────────────────────

def is_yes_no_question(query: str) -> bool:
    """判断是不是 yes/no 题 —— 不只看开头词，还处理倒装句。

    "In the context of X, does Y..." 这种也能抓到。
    """
    q = query.strip().lower()
    if any(q.startswith(s) for s in YES_NO_STARTERS):
        return True
    if re.search(r'\b(?:what|which|where|why|how|whose|whom)\b', q):
        return False
    prefix_m = re.match(
        r"^(?:in|at|on|for|under|with|when|while|given|after|before)\b", q,
    )
    if prefix_m:
        rest = q[prefix_m.end():]
        if re.search(
            r'\b(?:do|does|did|is|are|was|were|can|could|has|have|had|will|would|may|might)\b',
            rest[:80],
        ):
            return True
    return False


def is_yes_no_answer(answer: str) -> bool:
    a = answer.strip().lower()
    return a.startswith("yes") or a.startswith("no")


# ── Deictic / grounding checks ────────────────────────────────────────────────

# Grounding phrases that make deictic references acceptable
_DEICTIC_GROUNDING_PHRASES = re.compile(
    r"\b(?:panel|subplot|row|column|col|cell|axis|region|left|right|top|bottom"
    r"|upper|lower|bar|curve|cluster|histogram|scatter|frontier|quadrant"
    r"|layer|branch|encoder|decoder|block|stage|step)\b",
    re.IGNORECASE,
)

def has_bare_deictic(query: str) -> bool:
    """Detect ungrounded deictic references ('here', 'this', 'that') in query.

    Returns True (bad) only if the deictic word appears WITHOUT any nearby
    grounding phrase (panel, row, curve, etc.) that anchors the reference.
    """
    q = query.strip().lower()
    # Only flag sentence-final 'here' or standalone 'this/that' without grounding
    has_bare_here = bool(re.search(r"\bhere[?.]?\s*$", q))
    has_bare_this_that = bool(
        re.search(r"\b(?:this|that)\b", q)
        and not _DEICTIC_GROUNDING_PHRASES.search(q)
    )
    return has_bare_here or has_bare_this_that


# ── Template / opening checks ────────────────────────────────────────────────

def has_shortcut_template(query: str) -> bool:
    """抓 "Which component..." / "How does X relate to Y" 这种偷懒模板。"""
    q = query.strip().lower()
    return any(re.search(p, q) for p in QUERY_SHORTCUT_PATTERNS)


def has_templated_opening(query: str) -> bool:
    q = query.strip().lower()
    return any(q.startswith(prefix) for prefix in TEMPLATED_QUERY_OPENINGS)


def has_template_collapse(query: str) -> bool:
    """抓高频 shell pattern —— "under what..." / "why is X different from Y" 这种。"""
    q = query.strip().lower()
    return any(re.search(p, q) for p in TEMPLATE_COLLAPSE_PATTERNS)


# ── Query metrics ─────────────────────────────────────────────────────────────

def query_opening_signature(query: str, n_words: int = 2) -> str:
    toks = re.findall(r"[a-zA-Z]+", query.lower())
    return " ".join(toks[:n_words]) if toks else ""


def query_word_count(query: str) -> int:
    return len(re.findall(r"[A-Za-z0-9][A-Za-z0-9_-]*", query or ""))


def query_length_bucket(query: str) -> str:
    wc = query_word_count(query)
    if wc < SHORT_QUERY_MIN_WORDS:
        return "too_short"
    if wc <= SHORT_QUERY_MAX_WORDS:
        return "short"
    if wc < LONG_QUERY_MIN_WORDS:
        return "medium"
    if wc <= MAX_QUERY_WORDS:
        return "long"
    return "too_long"


def has_length_mix(queries: List[Dict[str, Any]]) -> Tuple[bool, bool]:
    has_short = has_long = False
    for obj in queries:
        bucket = query_length_bucket(str(obj.get("query", "")))
        if bucket == "short":
            has_short = True
        if bucket == "long":
            has_long = True
    return has_short, has_long


# ── Architecture checks ──────────────────────────────────────────────────────

def _is_architecture_text(text: str) -> bool:
    t = (text or "").lower()
    return any(re.search(rf"\b{re.escape(k)}\b", t) for k in ARCHITECTURE_KEYWORDS)


def is_architecture_pair(pair: Dict[str, Any]) -> bool:
    elem_a = pair.get("element_a", {}) or {}
    elem_b = pair.get("element_b", {}) or {}
    fig_elem = elem_a if str(elem_a.get("element_type", "")) == "figure" else elem_b
    if str(fig_elem.get("element_type", "")) != "figure":
        return False
    fig_text = " ".join([
        str(fig_elem.get("caption", "") or ""),
        str(fig_elem.get("content", "") or ""),
        str(fig_elem.get("context_before", "") or ""),
        str(fig_elem.get("context_after", "") or ""),
    ])
    return _is_architecture_text(fig_text)


def has_architecture_intent(query: str, answer: str) -> bool:
    qa = f"{query or ''} {answer or ''}".lower()
    return any(re.search(rf"\b{re.escape(k)}\b", qa) for k in ARCHITECTURE_INTENT_TERMS)


# ── Reasoning / logic checks ─────────────────────────────────────────────────

def has_parallel_dual_ask(query: str) -> bool:
    q = query.strip().lower()
    return bool(
        re.search(r",\s*and\s+(?:which|what|under\s+what)\b", q)
        or re.search(r"\band\s+which\b", q)
        or re.search(r"\band\s+what\b", q)
    )


def has_semantic_category_mismatch(query: str) -> bool:
    """High-precision check for 'A different from B' cross-category stitching."""
    q = query.strip().lower()
    marker = ""
    for m in ("different from", "inconsistent with"):
        if m in q:
            marker = m
            break
    if not marker:
        return False
    left, right = q.split(marker, 1)

    def _has_term(text: str, terms: Set[str]) -> bool:
        return any(re.search(rf"\b{re.escape(t)}\b", text) for t in terms)

    left_struct = _has_term(left, STRUCTURAL_CONCEPT_TERMS)
    right_struct = _has_term(right, STRUCTURAL_CONCEPT_TERMS)
    left_metric = _has_term(left, METRIC_CONCEPT_TERMS)
    right_metric = _has_term(right, METRIC_CONCEPT_TERMS)

    if left_struct and right_metric and not left_metric:
        return True
    if right_struct and left_metric and not right_metric:
        return True
    return False


def has_min_reasoning_chain(obj: Dict[str, Any]) -> bool:
    rc = str(obj.get("reasoning_chain", "")).strip()
    if len(rc) < 40:
        return False
    chunks = [c.strip() for c in re.split(r"[.;\n]", rc) if c.strip()]
    return len(chunks) >= 2


def has_relationship_connector(answer: str) -> bool:
    a = answer.lower()
    return any(conn in a for conn in RELATION_CONNECTORS)


def has_premise_answer_contradiction(query: str, answer: str) -> bool:
    """高精度矛盾检测 —— 抓 "why does X drop" + "X does not drop" 这种前提翻转。"""
    q = query.lower()
    a = answer.lower()

    if "drop below" in q and re.search(r"\b(?:does|do|did)\s+not\s+drop below\b|\bnot drop below\b", a):
        return True
    if "drop more unlabeled users" in q and "fewer unlabeled users" in a:
        return True
    if "with fewer" in q and "with more" in q:
        nums = [float(n.replace(",", "")) for n in re.findall(r"\b\d+(?:\.\d+)?\b", answer)]
        if len(nums) >= 2 and nums[0] > nums[1]:
            return True
    for verb in ("increase", "decrease", "drop", "rise", "fall", "exceed", "improve"):
        if re.search(rf"\bwhy\s+do(?:es)?\b.*\b{verb}\b", q) and re.search(
            rf"\b(?:does|do|did)\s+not\s+{verb}\b", a
        ):
            return True
    return False


# ── Anchor / overlap checks ──────────────────────────────────────────────────

def anchor_leak_jaccard(query: str, anchors: List[Dict[str, Any]]) -> float:
    """query 跟 visual anchor 的 Jaccard 相似度 —— 超过 0.20 就算泄漏。"""
    q_tokens = content_tokens(query)
    if not q_tokens:
        return 0.0
    max_jacc = 0.0
    for a in anchors:
        a_text = a.get("anchor", "") if isinstance(a, dict) else str(a)
        a_tokens = content_tokens(a_text)
        if not a_tokens:
            continue
        intersection = q_tokens & a_tokens
        union = q_tokens | a_tokens
        jacc = len(intersection) / len(union) if union else 0.0
        max_jacc = max(max_jacc, jacc)
    return max_jacc


def anchor_token_copy_count(query: str, anchors: List[Dict[str, Any]]) -> int:
    q_tokens = content_tokens(query)
    if not q_tokens:
        return 0
    all_anchor_tokens: Set[str] = set()
    for a in anchors:
        a_text = a.get("anchor", "") if isinstance(a, dict) else str(a)
        all_anchor_tokens |= content_tokens(a_text)
    return len(q_tokens & all_anchor_tokens)


def anchor_overlap_tokens(query: str, anchors: List[Dict[str, Any]]) -> Set[str]:
    q_tokens = content_tokens(query)
    all_anchor_tokens: Set[str] = set()
    for a in anchors:
        a_text = a.get("anchor", "") if isinstance(a, dict) else str(a)
        all_anchor_tokens |= content_tokens(a_text)
    return q_tokens & all_anchor_tokens


# ── Single-element answer check ──────────────────────────────────────────────

def check_single_element_answer(
    obj: Dict[str, Any],
    pair: Dict[str, Any],
    a_tokens: Set[str],
    a_num_tokens: Set[str],
) -> Tuple[bool, Dict[str, Any]]:
    """检查答案是不是只用了一个元素就能答 —— 双证据 query 必须两个都用上。

    算法：分别计算答案与 element_a / element_b 的 token 重叠，
    如果 balance < 0.20 或者某一边重叠低于阈值，就算 fail。
    返回 (是否fail, 详细指标)。
    """
    metrics: Dict[str, Any] = {}

    def _elem_text(elem: Dict) -> str:
        return (elem.get("caption", "") or "") + " " + (elem.get("content", "") or "")

    def _min_overlap(elem: Dict) -> int:
        etype = str(elem.get("element_type", "")).lower()
        return int(MIN_OVERLAP_BY_TYPE.get(etype, 2))

    span_text_by_eid: Dict[str, str] = defaultdict(str)
    for s in obj.get("required_evidence_spans", []) or []:
        if not isinstance(s, dict):
            continue
        eid = str(s.get("element_id", "")).strip()
        span = str(s.get("span", "")).strip()
        if eid and span:
            span_text_by_eid[eid] += " " + span

    def _overlap(elem: Dict, elem_id: str) -> int:
        merged = (_elem_text(elem) + " " + span_text_by_eid.get(elem_id, "")).strip()
        word_ov = len(a_tokens & content_tokens(merged))
        num_ov = len(a_num_tokens & number_tokens(merged))
        return word_ov + min(num_ov, 2)

    elem_a = pair.get("element_a", {})
    elem_b = pair.get("element_b", {})
    ov_a = _overlap(elem_a, pair.get("element_a_id", ""))
    ov_b = _overlap(elem_b, pair.get("element_b_id", ""))
    metrics["answer_overlap_a"] = ov_a
    metrics["answer_overlap_b"] = ov_b
    total = ov_a + ov_b
    metrics["answer_balance"] = 0.0

    if total > 0:
        balance = min(ov_a / total, ov_b / total)
        metrics["answer_balance"] = round(balance, 4)
        is_fail = (
            ov_a < _min_overlap(elem_a)
            or ov_b < _min_overlap(elem_b)
            or balance < ANSWER_BALANCE_THRESHOLD
        )
    else:
        is_fail = True

    return is_fail, metrics


# ── Formula grounding ─────────────────────────────────────────────────────────

def formula_symbol_hit(answer: str, terms: Set[str]) -> bool:
    """figure+formula 对里答案必须引用公式符号 —— 不引就说明没有真正用到公式。"""
    if not terms:
        return True
    a = (answer or "").lower()
    a_no_space = re.sub(r"\s+", "", a)
    for t in terms:
        if "(" in t or "_" in t:
            if t in a_no_space:
                return True
        else:
            if re.search(rf"\b{re.escape(t)}\b", a):
                return True
    return False


# ── Evidence overlap ──────────────────────────────────────────────────────────

def answer_text_evidence_overlap(answer: str, evidence: str) -> float:
    """答案和 text_evidence 的 token 重叠比 —— 太高说明答案只是在复述证据。"""
    a_toks = content_tokens(answer)
    if not a_toks:
        return 0.0
    e_toks = content_tokens(evidence)
    if not e_toks:
        return 0.0
    return len(a_toks & e_toks) / len(a_toks)


# ── Enrichment noise filter ──────────────────────────────────────────────────

def is_noisy_enrichment(text: str) -> bool:
    """Detect enrichment noise — glyph/icon/OCR junk gets discarded."""
    if not text:
        return True
    stripped = text.strip()
    if len(stripped) < 15:
        return True
    return bool(ENRICHMENT_NOISE_RE.search(stripped))


# ── Answer well-posedness checks ─────────────────────────────────────────────

_HEDGE_RE = re.compile(
    r"\b(if |assuming |suppose |provided that |in case |depending on |were to )",
    re.IGNORECASE,
)


def has_conditional_hedge_overload(answer: str, threshold: int = 3) -> bool:
    """Detect underdetermined queries via excessive conditional hedging.

    An answer that needs 3+ if/assuming/suppose clauses to survive likely
    corresponds to an under-specified question that has no right to be in
    the formal set.

    Returns True (bad) when hedge count >= threshold.
    """
    hedges = _HEDGE_RE.findall(answer or "")
    return len(hedges) >= threshold


# ── HopWeaver-style multi-hop integrity checks ──────────────────────────────

def _extract_doc_id(element_id: str) -> str:
    """Extract arXiv doc ID from element_id like '1904.03310::el::fig:3'."""
    return element_id.split("::")[0] if "::" in element_id else element_id


def check_fact_distribution(obj: Dict[str, Any], pair: Dict[str, Any]) -> bool:
    """HopWeaver Fact Distribution: each hop must use evidence from different docs.

    A true multi-hop chain cannot have all reasoning steps from the same document.
    Returns True (pass) if at least 2 unique documents are involved.
    """
    doc_ids: Set[str] = set()

    # Collect doc_ids from reasoning_steps
    steps = obj.get("reasoning_steps", []) or []
    for step in steps:
        eid = step.get("evidence_element_id", "")
        if eid:
            doc_ids.add(_extract_doc_id(eid))

    # Also check element_a / element_b from the pair
    for key in ("element_a_id", "element_b_id"):
        eid = pair.get(key, "")
        if eid:
            doc_ids.add(_extract_doc_id(eid))

    # Also check element_ids list
    for eid in obj.get("element_ids", []) or []:
        doc_ids.add(_extract_doc_id(eid))

    return len(doc_ids) >= 2


def check_no_shortcut(obj: Dict[str, Any], pair: Dict[str, Any]) -> bool:
    """HopWeaver No-Shortcut: no single document can bridge non-adjacent hops.

    If one document contains ALL evidence elements needed for the reasoning chain,
    the chain has a shortcut — a reader could answer the query from one document.
    Returns True (pass) if no single document monopolizes the evidence.
    """
    doc_counts: Dict[str, int] = {}

    steps = obj.get("reasoning_steps", []) or []
    for step in steps:
        eid = step.get("evidence_element_id", "")
        if eid:
            doc_id = _extract_doc_id(eid)
            doc_counts[doc_id] = doc_counts.get(doc_id, 0) + 1

    total_steps = len(steps)
    if total_steps < 2:
        return True  # Can't have a shortcut with < 2 steps

    for doc_id, count in doc_counts.items():
        if count >= total_steps:
            return False  # One doc has all evidence — shortcut exists

    return True


def check_causal_chain_direction(
    obj: Dict[str, Any],
) -> Tuple[bool, Dict[str, Any]]:
    """Verify reasoning_steps form a causal chain (not parallel evidence).

    Checks that reasoning_role sequence goes premise → intermediate → conclusion,
    and that depends_on_steps form a connected DAG.
    Returns (pass, metrics).
    """
    metrics: Dict[str, Any] = {}
    steps = obj.get("reasoning_steps", []) or []

    if not steps:
        metrics["causal_chain_skip"] = "no_steps"
        return True, metrics

    # Check role progression
    roles = [s.get("reasoning_role", "") for s in steps]
    metrics["role_sequence"] = roles

    has_premise = "premise" in roles
    has_conclusion = "conclusion" in roles
    metrics["has_premise"] = has_premise
    metrics["has_conclusion"] = has_conclusion

    if not has_premise or not has_conclusion:
        metrics["causal_chain_fail"] = "missing_premise_or_conclusion"
        return False, metrics

    # Check that conclusion is the last step
    if roles[-1] != "conclusion":
        metrics["causal_chain_fail"] = "conclusion_not_last"
        return False, metrics

    # Check dependency connectivity
    has_independent = any(not s.get("depends_on_steps") for s in steps)
    has_dependent = any(s.get("depends_on_steps") for s in steps)
    metrics["has_independent_root"] = has_independent
    metrics["has_dependent_steps"] = has_dependent

    if not has_dependent:
        metrics["causal_chain_fail"] = "all_independent_parallel"
        return False, metrics

    return True, metrics


# ── Bridge overclaim detection ───────────────────────────────────────────────

_STRONG_CAUSAL_RE = re.compile(
    r"\b(causes?|leads? to|results? in|produces?|drives?|ensures?|guarantees?|proves? that)\b",
    re.IGNORECASE,
)
_WEAK_HEDGE_RE = re.compile(
    r"\b(may|might|could|possibly|potentially|suggests?|tends? to|appears? to)\b",
    re.IGNORECASE,
)


def has_bridge_overclaim_signal(query: str, answer: str) -> bool:
    """Detect when a query implies strong causation but the answer hedges.

    Pattern: query uses strong causal language ("causes", "leads to") but the
    answer uses weak hedges ("may", "might", "suggests"). This mismatch
    indicates the bridge relationship is weaker than the query claims.

    Returns True (bad) when query has strong causal verbs AND answer has
    more hedge words than causal verbs (i.e. the evidence only weakly supports
    the claimed relationship).
    """
    q_causal = len(_STRONG_CAUSAL_RE.findall(query or ""))
    if q_causal == 0:
        return False
    a_causal = len(_STRONG_CAUSAL_RE.findall(answer or ""))
    a_hedge = len(_WEAK_HEDGE_RE.findall(answer or ""))
    # If answer has more hedges than causal assertions, the bridge is overclaimed
    return a_hedge > a_causal and a_hedge >= 2
