#!/usr/bin/env python3
"""Generate cross-modal dual-evidence L1 queries from DAG candidates.

Reads multihop_l1_candidates.json (from select_multihop_candidates.py),
sends element pairs to Claude Vision API with modality-specific prompts,
and outputs QC-filtered queries to l1_multihop_queries_v2.jsonl.
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import re
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

SYSTEM_PROMPT = (
    "You are a data annotator creating cross-modal retrieval training data "
    "for multimodal academic documents. "
    "Output valid JSON only, no other text, no markdown fences."
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# ──────────────────────────────────────────────────────────────
# Prompt templates per modality combination
# ──────────────────────────────────────────────────────────────

PROMPT_FIGURE_TABLE_1HOP = """Generate 2 cross-modal retrieval queries requiring BOTH a figure and a table from the same document.

## Figure ({fig_id})
Caption: {fig_caption}
Context: {fig_context}
[Figure image is provided above]

## Table ({tbl_id})
Caption: {tbl_caption}
Column/row headers: {tbl_headers}
Context: {tbl_context}
[Table image is provided above; read specific values from the table image]

## Textual connection
{edge_context}

## YOUR TASK
Create queries that REQUIRE BOTH elements. The query should create an INFORMATION GAP:
describe context from one element, ask a question answerable only by the other.

## STRICT RULES
1. Query MUST be UNANSWERABLE if either the figure or the table is removed.
2. NEVER start with Do/Does/Did/Is/Are/Can/Has/Will/Would; NO yes/no questions.
3. NEVER put specific numbers, percentages, or exact values in the query.
   Use descriptive references instead.
4. NEVER use meta-language: "figure", "table", "the text", "according to", "as shown in".
5. Max 30 words per query. Answer max 3 sentences with specific values from BOTH elements.
6. visual_anchors must reference BOTH elements with specific visual details.
7. The two queries must use DIFFERENT aspects of the data.
8. DE-NAME bridge entities in the query: do NOT copy named terms from anchors
   (e.g., country names, component labels, exact row names). Use visual descriptions
   like "the dominant term", "the leftmost peak", "the most imbalanced group".
9. Avoid weak templates such as "Which component..." and "How does X relate to Y...".
10. Answer should include a causal connector (e.g., because, due to, leads to, explains, matches).

## BAD vs GOOD examples
BAD: "Did the red line peak at 90,000 and match the listed keyword set?"
GOOD: "What collection setup explains the order-of-magnitude gap between the two API curves during the late-December surge?"

BAD: "Is 0.85 in the third row consistent with the upward trend?"
GOOD: "Which setting shows the largest gain over baseline, and how does its curve shape differ from the others?"

## Output format (JSON only):
{{
  "queries": [
    {{
      "query": "open-ended question, max 30 words, NO specific values",
      "answer": "factual answer citing specific values from BOTH elements, max 3 sentences",
      "query_type": "trend_explanation|parameter_outcome|cross_reading|anomaly_investigation",
      "visual_anchors": [
        {{"element_id": "{fig_id}", "anchor": "specific visual element (color, position, value)"}},
        {{"element_id": "{tbl_id}", "anchor": "specific cell, row, or column"}}
      ],
      "text_evidence": "direct quote from context, min 40 chars"
    }}
  ]
}}"""

PROMPT_FIGURE_TABLE_2HOP = """Generate 2 cross-modal retrieval queries requiring BOTH a figure and a table from the same document.

## Figure ({fig_id})
Caption: {fig_caption}
Context: {fig_context}
[Figure image is provided above]

## Table ({tbl_id})
Caption: {tbl_caption}
Column/row headers: {tbl_headers}
Context: {tbl_context}
[Table image is provided above; read specific values from the table image]

## Connection chain (2-hop)
{edge_context}
Intermediate element(s): {intermediate_info}

## YOUR TASK
Create queries that REQUIRE BOTH elements with CHAIN REASONING.
The intermediate element is the bridge, not optional context.

## STRICT RULES
1. Query MUST be UNANSWERABLE if either endpoint or bridge is removed.
2. NEVER start with Do/Does/Did/Is/Are/Can/Has/Will/Would; NO yes/no questions.
3. NEVER put specific numbers or exact values in the query.
4. NEVER use meta-language: "figure", "table", "the text", "according to", "as shown in".
5. Max 30 words per query; answer max 3 sentences with values from BOTH endpoints.
6. visual_anchors must reference BOTH endpoints.
7. DE-NAME bridge entities in the query: do NOT copy named terms from anchors.
8. Avoid weak templates such as "Which component..." and "How does X relate to Y...".
9. Answer should include a causal connector (e.g., because, due to, leads to, explains, matches).

## Output format (JSON only):
{{
  "queries": [
    {{
      "query": "chain-reasoning question, max 30 words, NO specific values",
      "answer": "factual answer citing values from both elements, max 3 sentences",
      "query_type": "chain_verification|parameter_outcome|condition_result|bridge_reasoning",
      "visual_anchors": [
        {{"element_id": "{fig_id}", "anchor": "specific visual element"}},
        {{"element_id": "{tbl_id}", "anchor": "specific cell, row, or column"}}
      ],
      "text_evidence": "direct quote from context, min 40 chars"
    }}
  ]
}}"""

PROMPT_FIGURE_FORMULA = """Generate 2 cross-modal retrieval queries connecting a figure with a mathematical formula.

## Figure ({fig_id})
Caption: {fig_caption}
Context: {fig_context}
[Figure image is provided above]

## Formula ({formula_id})
Key variables: {formula_variables}
Context: {formula_context}

## Textual connection
{edge_context}

## YOUR TASK
Create queries where the figure gives empirical evidence and the formula gives
the theoretical framework.

## STRICT RULES
1. Query MUST be UNANSWERABLE without both the figure and the formula.
2. NEVER start with Do/Does/Did/Is/Are/Can/Has/Will/Would; NO yes/no questions.
3. NEVER copy LaTeX symbols, raw variable strings, or specific values into the query.
4. NEVER use meta-language: "equation", "formula", "figure", "as shown in".
5. Max 30 words per query.
6. visual_anchors: one figure detail + one formula term/variable described naturally.
7. DE-NAME bridge entities in the query: avoid direct component/variable labels from anchors.
8. Avoid weak templates such as "Which component..." and "How does X relate to Y...".
9. Answer should include a causal connector (e.g., because, due to, leads to, explains, matches).

## BAD vs GOOD examples
BAD: "Does gamma*L_dis achieve 0.24?"
GOOD: "How does stronger disentanglement regularization affect the attribute-removal metric across model variants?"

## Output format (JSON only):
{{
  "queries": [
    {{
      "query": "theory-experiment question, max 30 words, NO LaTeX/values",
      "answer": "factual answer referencing formula terms AND figure data, max 3 sentences",
      "query_type": "theory_vs_experiment|parameter_identification|boundary_behavior|sensitivity",
      "visual_anchors": [
        {{"element_id": "{fig_id}", "anchor": "specific visual element"}},
        {{"element_id": "{formula_id}", "anchor": "specific term or variable in the formula"}}
      ],
      "text_evidence": "direct quote from context, min 40 chars"
    }}
  ]
}}"""

PROMPT_FORMULA_TABLE = """Generate 2 cross-modal retrieval queries connecting a formula with a table from the same document.

## Formula ({formula_id})
Key variables: {formula_variables}
Context: {formula_context}

## Table ({tbl_id})
Caption: {tbl_caption}
Column/row headers: {tbl_headers}
Context: {tbl_context}
[Table image is provided above; read specific values from the table image]

## Textual connection
{edge_context}

## YOUR TASK
Create queries that connect mathematical relationships with tabular evidence.

## STRICT RULES
1. Query MUST be UNANSWERABLE without both the formula and the table.
2. NEVER start with Do/Does/Did/Is/Are/Can/Has/Will/Would; NO yes/no questions.
3. NEVER copy LaTeX strings or exact table values into the query.
4. NEVER use meta-language: "table", "equation", "formula", "the text".
5. Max 30 words per query; answer max 3 sentences with specific values from the table.
6. visual_anchors must include one formula anchor and one table anchor.
7. DE-NAME bridge entities in the query: avoid direct row/variable/component names from anchors.
8. Avoid weak templates such as "Which component..." and "How does X relate to Y...".
9. Answer should include a causal connector (e.g., because, due to, leads to, explains, matches).

## BAD vs GOOD examples
BAD: "Does beta=0.3 in row 4 satisfy Eq. 2?"
GOOD: "Which parameter setting best fits the model relation, and what residual pattern remains across rows?"

## Output format (JSON only):
{{
  "queries": [
    {{
      "query": "formula-data question, max 30 words, NO LaTeX/values",
      "answer": "factual answer with specific values from both, max 3 sentences",
      "query_type": "formula_instantiation|data_formula_consistency|sensitivity_analysis|unit_verification",
      "visual_anchors": [
        {{"element_id": "{formula_id}", "anchor": "specific term or variable"}},
        {{"element_id": "{tbl_id}", "anchor": "specific cell, row, or column"}}
      ],
      "text_evidence": "direct quote from context, min 40 chars"
    }}
  ]
}}"""


# ──────────────────────────────────────────────────────────────
# QC infrastructure (reused + extended from L2 script)
# ──────────────────────────────────────────────────────────────

BAD_META_PATTERNS = [
    r"\bfigure\b",
    r"\btable\b",
    r"\bequation\b",
    r"\bformula\b",
    r"according to",
    r"as (?:shown|mentioned|stated|described|depicted|illustrated) in",
    r"the (?:text|caption|paper|section|paragraph)",
    r"(?:this|the) (?:figure|table|chart|plot|graph|diagram)",
]

YES_NO_STARTERS = [
    "do ", "does ", "did ", "can ", "could ", "is ", "are ",
    "would ", "has ", "have ", "will ", "was ", "were ",
    "had ", "should ", "may ", "might ",
]

LEAK_STOPWORDS = {
    "the", "a", "an", "of", "in", "to", "for", "on", "at", "by", "and", "or",
    "is", "are", "was", "were", "be", "been", "with", "from", "as", "that",
    "this", "it", "its", "how", "what", "which", "when", "where", "does", "do",
    "between", "across", "than", "both", "each", "all", "into", "over",
}

ANCHOR_LEAK_THRESHOLD = 0.15
ANSWER_BALANCE_THRESHOLD = 0.15   # v2.1: relaxed from 0.25 — token overlap is noisy proxy
MIN_OVERLAP_PER_ELEMENT = 1       # v2.1: relaxed from 2 — visual captions can be short

QUERY_SHORTCUT_PATTERNS = [
    r"^which\s+component\b",
    r"^which\s+method\b",
    r"^which\s+approach\b",
    r"^which\s+variable\b",
    r"^which\s+pair\b",
    r"^how\s+does\s+.+\s+relate\s+to\s+.+",
    r"^how\s+do\s+.+\s+relate\s+to\s+.+",
    r"^what\s+relationship\s+exists\b",
]

CAUSAL_CONNECTORS = {
    "because", "due to", "therefore", "thus", "hence",
    "leads to", "results in", "explains", "matches", "corresponds to",
    "driven by", "caused by", "consistent with",
}


def has_numeric_leakage(query: str) -> bool:
    """Flag queries that leak too many specific numeric values."""
    nums = re.findall(r"\b\d+(?:[.,]\d+)?%?\b", query)
    suspicious = []
    for raw in nums:
        token = raw.replace(",", "").rstrip("%")
        try:
            value = float(token)
        except ValueError:
            continue
        # allow small ordinals/indices and likely years
        if 1 <= value <= 10 and value.is_integer():
            continue
        if 1900 <= value <= 2099 and value.is_integer():
            continue
        suspicious.append(raw)
    return len(suspicious) >= 2


def is_yes_no_question(query: str) -> bool:
    q = query.strip().lower()
    if any(q.startswith(s) for s in YES_NO_STARTERS):
        return True
    # Catch prefixed forms: "At n=1000, does ...?"
    if re.search(
        r"^(?:in|at|on|for|under|with|when|while|given|after|before)\b[^?]{0,80}\b"
        r"(?:do|does|did|is|are|was|were|can|could|has|have|had|will|would|may|might)\b",
        q,
    ):
        return True
    return False


def is_yes_no_answer(answer: str) -> bool:
    a = answer.strip().lower()
    return a.startswith("yes") or a.startswith("no")


def has_shortcut_template(query: str) -> bool:
    q = query.strip().lower()
    return any(re.search(p, q) for p in QUERY_SHORTCUT_PATTERNS)


def has_causal_connector(answer: str) -> bool:
    a = answer.lower()
    return any(conn in a for conn in CAUSAL_CONNECTORS)


def extract_formula_variables(content: str) -> str:
    """Extract lightweight variable/function hints from formula text."""
    if not content:
        return "(none)"

    text = content

    # Prefer explicit math regions to avoid pulling narrative words.
    regions: List[str] = []
    regions += re.findall(r"\$(.+?)\$", text, flags=re.DOTALL)
    regions += re.findall(r"\\\((.+?)\\\)", text, flags=re.DOTALL)
    regions += re.findall(r"\\\[(.+?)\\\]", text, flags=re.DOTALL)
    math_text = " ".join(regions) if regions else text

    # capture latex commands likely representing functions/terms
    funcs = re.findall(r"\\([A-Za-z]{2,})", math_text)
    funcs = [f for f in funcs if f not in {
        "begin", "end", "left", "right", "frac", "cdot", "times",
        "sum", "prod", "int", "mid", "tag",
    }]

    # capture symbolic variable names (prefer math-like tokens)
    vars_sub = re.findall(r"\b([A-Za-z]+_[A-Za-z0-9]+)\b", math_text)
    vars_single = re.findall(
        r"(?:^|[=+\-*/(,\s{])([A-Za-z])(?:$|[=+\-*/),\s}_^])",
        math_text,
    )
    greek = re.findall(
        r"\\(alpha|beta|gamma|delta|epsilon|lambda|mu|sigma|tau|theta|phi|psi|omega)",
        math_text,
    )
    tokens = list(dict.fromkeys(vars_sub + vars_single + greek))

    # keep compact
    functions = ", ".join(list(dict.fromkeys(funcs))[:8]) if funcs else ""
    variables = ", ".join(tokens[:12]) if tokens else ""
    parts = []
    if variables:
        parts.append(f"Variables: {variables}")
    if functions:
        parts.append(f"Functions/terms: {functions}")
    return "; ".join(parts) if parts else "(no clear variables found)"


def extract_table_headers(content: str, max_chars: int = 150) -> str:
    """Extract table headers/labels and avoid leaking dense numeric values."""
    if not content:
        return "(none)"

    headers: List[str] = []

    # HTML table case
    if "<td" in content.lower() or "<th" in content.lower():
        cells = re.findall(r"<t[dh][^>]*>(.*?)</t[dh]>", content, flags=re.IGNORECASE | re.DOTALL)
        for c in cells[:24]:
            txt = re.sub(r"<[^>]+>", " ", c)
            txt = re.sub(r"\s+", " ", txt).strip()
            if not txt:
                continue
            # skip mostly numeric cells
            if re.search(r"\d", txt) and not re.search(r"[A-Za-z]", txt):
                continue
            headers.append(txt)
            if len(" ; ".join(headers)) >= max_chars:
                break
    else:
        lines = [ln.strip() for ln in content.splitlines() if ln.strip()]
        for ln in lines[:20]:
            # prioritize markdown header-like or first token labels
            if "|" in ln:
                cells = [c.strip() for c in ln.split("|") if c.strip()]
                for c in cells:
                    # skip mostly-numeric cells
                    if re.search(r"\d", c) and not re.search(r"[A-Za-z]", c):
                        continue
                    headers.append(c)
            else:
                left = ln.split(":")[0].strip()
                if left and re.search(r"[A-Za-z]", left):
                    headers.append(left)
            if len(" ; ".join(headers)) >= max_chars:
                break

    deduped: List[str] = []
    seen: Set[str] = set()
    for h in headers:
        norm = re.sub(r"\s+", " ", h.lower()).strip()
        if not norm or norm in seen:
            continue
        seen.add(norm)
        cleaned = re.sub(r"\b\d+(?:[.,]\d+)?%?\b", "", h).strip()
        cleaned = re.sub(r"\s+", " ", cleaned)
        if cleaned:
            deduped.append(cleaned)

    out = " ; ".join(deduped)
    return out[:max_chars] if out else "(headers unavailable)"


def _content_tokens(text: str) -> Set[str]:
    words = set(re.findall(r"\b[a-zA-Z]{3,}\b", text.lower()))
    return words - LEAK_STOPWORDS


def anchor_leak_jaccard(query: str, anchors: List[Dict[str, Any]]) -> float:
    q_tokens = _content_tokens(query)
    if not q_tokens:
        return 0.0
    max_jacc = 0.0
    for a in anchors:
        a_text = a.get("anchor", "") if isinstance(a, dict) else str(a)
        a_tokens = _content_tokens(a_text)
        if not a_tokens:
            continue
        intersection = q_tokens & a_tokens
        union = q_tokens | a_tokens
        jacc = len(intersection) / len(union) if union else 0.0
        max_jacc = max(max_jacc, jacc)
    return max_jacc


def anchor_token_copy_count(query: str, anchors: List[Dict[str, Any]]) -> int:
    """Count copied content tokens between query and all anchor texts."""
    q_tokens = _content_tokens(query)
    if not q_tokens:
        return 0
    all_anchor_tokens: Set[str] = set()
    for a in anchors:
        a_text = a.get("anchor", "") if isinstance(a, dict) else str(a)
        all_anchor_tokens |= _content_tokens(a_text)
    return len(q_tokens & all_anchor_tokens)


def qc_multihop_query(
    obj: Dict[str, Any],
    pair: Dict[str, Any],
) -> Tuple[List[str], Dict[str, float]]:
    """Run QC checks on a multi-hop L1 query. Returns (issues, metrics)."""
    issues: List[str] = []
    metrics: Dict[str, float] = {}
    q = obj.get("query", "")
    q_lower = q.lower().strip()
    a = obj.get("answer", "")
    anchors = obj.get("visual_anchors", [])

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

    # 2d. Weak shortcut templates
    if has_shortcut_template(q):
        issues.append("template_shortcut")

    # 3. Short answer
    if len(a) < 20:
        issues.append("short_answer")

    # 4. Empty query
    if not q or len(q) < 10:
        issues.append("empty_query")

    # 5. Anchor leakage
    leak = anchor_leak_jaccard(q, anchors)
    metrics["anchor_leak_jaccard"] = round(leak, 4)
    if leak > ANCHOR_LEAK_THRESHOLD:
        issues.append("anchor_leakage")
    anchor_copy = anchor_token_copy_count(q, anchors)
    metrics["anchor_token_copy_count"] = anchor_copy
    if anchor_copy >= 4:   # v2.1: relaxed from 3 — 3 shared tokens is a weak signal
        issues.append("bridge_entity_leakage")

    # 6. Missing dual anchor — both elements must have an anchor
    elem_a_id = pair.get("element_a_id", "")
    elem_b_id = pair.get("element_b_id", "")
    anchor_element_ids = {
        a.get("element_id", "") for a in anchors if isinstance(a, dict)
    }
    has_a = elem_a_id in anchor_element_ids
    has_b = elem_b_id in anchor_element_ids
    if not has_a or not has_b:
        issues.append("missing_dual_anchor")
    metrics["anchor_count"] = len(anchors)

    # 7. Single-element answer — answer should reference content from both
    # Heuristic: check if answer tokens overlap with both element contexts
    a_tokens = _content_tokens(a)
    if a_tokens:
        ctx_a = (pair.get("element_a", {}).get("caption", "") + " " +
                 pair.get("element_a", {}).get("content", ""))
        ctx_b = (pair.get("element_b", {}).get("caption", "") + " " +
                 pair.get("element_b", {}).get("content", ""))
        overlap_a = len(a_tokens & _content_tokens(ctx_a))
        overlap_b = len(a_tokens & _content_tokens(ctx_b))
        metrics["answer_overlap_a"] = overlap_a
        metrics["answer_overlap_b"] = overlap_b
        total = overlap_a + overlap_b
        if total > 0:
            contrib_a = overlap_a / total
            contrib_b = overlap_b / total
            balance = min(contrib_a, contrib_b)
            metrics["answer_balance"] = round(balance, 4)
            # Require non-trivial overlap from BOTH elements.
            if (
                overlap_a < MIN_OVERLAP_PER_ELEMENT
                or overlap_b < MIN_OVERLAP_PER_ELEMENT
                or balance < ANSWER_BALANCE_THRESHOLD
            ):
                issues.append("single_element_answer")
        else:
            issues.append("single_element_answer")

    # 8. Text evidence length
    evidence = obj.get("text_evidence", "")
    if len(evidence) < 40:
        issues.append("short_evidence")

    # 9. Encourage explanatory cross-modal answers instead of pure lookup
    # v2.1: cross_reading is a lookup/referencing type, not causal — exempt from WRC check
    qtype = str(obj.get("query_type", "")).lower()
    explanatory_types = {
        "trend_explanation",
        "anomaly_investigation",
        "bridge_reasoning",
        "theory_vs_experiment",
        "data_formula_consistency",
    }
    if qtype in explanatory_types and not has_causal_connector(a):
        issues.append("weak_reasoning_connector")

    return issues, metrics


# ──────────────────────────────────────────────────────────────
# Image encoding
# ──────────────────────────────────────────────────────────────

def encode_image(path: Optional[str]) -> Optional[Tuple[str, str]]:
    """Return (base64_data, mime_type) or None if file missing."""
    if not path:
        return None
    p = Path(path)
    if not p.is_absolute():
        p = PROJECT_ROOT / path
    if not p.exists() or p.stat().st_size < 500:
        return None
    ext = p.suffix.lower().lstrip(".")
    mime = {"jpg": "image/jpeg", "jpeg": "image/jpeg", "png": "image/png"}.get(
        ext, "image/jpeg"
    )
    with open(p, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8"), mime


# ──────────────────────────────────────────────────────────────
# Prompt building
# ──────────────────────────────────────────────────────────────

def build_edge_context_text(edge_contexts: List[Dict]) -> str:
    """Format edge context snippets into readable text."""
    if not edge_contexts:
        return "(no direct textual connection found)"
    parts = []
    for ec in edge_contexts:
        snippet = ec.get("context_snippet", "")
        if snippet:
            parts.append(f"Reference '{ec.get('ref_text', '')}': {snippet[:300]}")
    return "\n".join(parts) if parts else "(no context snippets)"


def build_intermediate_info(pair: Dict, all_elements: Optional[Dict] = None) -> str:
    """Describe intermediate elements in a multi-hop path."""
    path = pair.get("path", [])
    if len(path) <= 2:
        return "(direct connection)"
    intermediate_ids = path[1:-1]
    parts = []
    for mid_id in intermediate_ids:
        parts.append(mid_id)
    return ", ".join(parts)


def select_template(pair: Dict) -> str:
    """Choose the right prompt template based on modality combo and hop distance."""
    a_type = pair["element_a_type"]
    b_type = pair["element_b_type"]
    hop = pair["hop_distance"]

    types = {a_type, b_type}

    if types == {"figure", "table"}:
        if hop <= 1:
            return "figure_table_1hop"
        else:
            return "figure_table_2hop"
    elif types == {"figure", "formula"}:
        return "figure_formula"
    elif types == {"formula", "table"}:
        return "formula_table"
    else:
        return "figure_table_1hop"  # fallback


def build_prompt(pair: Dict) -> str:
    """Build the prompt text for a candidate pair."""
    template_name = select_template(pair)
    elem_a = pair["element_a"]
    elem_b = pair["element_b"]
    edge_text = build_edge_context_text(pair.get("edge_contexts", []))

    # Identify which element is figure/table/formula
    fig_elem = table_elem = formula_elem = None
    fig_key = table_key = formula_key = "a"

    for key, elem in [("a", elem_a), ("b", elem_b)]:
        if elem["element_type"] == "figure":
            fig_elem = elem
            fig_key = key
        elif elem["element_type"] == "table":
            table_elem = elem
            table_key = key
        elif elem["element_type"] == "formula":
            formula_elem = elem
            formula_key = key

    def _context(elem: Dict) -> str:
        before = (elem.get("context_before", "") or "")[:300]
        after = (elem.get("context_after", "") or "")[:300]
        parts = []
        if before:
            parts.append(before)
        if after:
            parts.append(after)
        return " ... ".join(parts) if parts else "(no context)"

    if template_name == "figure_table_1hop":
        return PROMPT_FIGURE_TABLE_1HOP.format(
            fig_id=fig_elem["element_id"],
            fig_caption=(fig_elem.get("caption", "") or "")[:400],
            fig_context=_context(fig_elem),
            tbl_id=table_elem["element_id"],
            tbl_caption=(table_elem.get("caption", "") or "")[:400],
            tbl_headers=extract_table_headers((table_elem.get("content", "") or ""), max_chars=150),
            tbl_context=_context(table_elem),
            edge_context=edge_text,
        )
    elif template_name == "figure_table_2hop":
        return PROMPT_FIGURE_TABLE_2HOP.format(
            fig_id=fig_elem["element_id"],
            fig_caption=(fig_elem.get("caption", "") or "")[:400],
            fig_context=_context(fig_elem),
            tbl_id=table_elem["element_id"],
            tbl_caption=(table_elem.get("caption", "") or "")[:400],
            tbl_headers=extract_table_headers((table_elem.get("content", "") or ""), max_chars=150),
            tbl_context=_context(table_elem),
            edge_context=edge_text,
            intermediate_info=build_intermediate_info(pair),
        )
    elif template_name == "figure_formula":
        return PROMPT_FIGURE_FORMULA.format(
            fig_id=fig_elem["element_id"],
            fig_caption=(fig_elem.get("caption", "") or "")[:400],
            fig_context=_context(fig_elem),
            formula_id=formula_elem["element_id"],
            formula_variables=extract_formula_variables((formula_elem.get("content", "") or "")[:1200]),
            formula_context=_context(formula_elem),
            edge_context=edge_text,
        )
    elif template_name == "formula_table":
        return PROMPT_FORMULA_TABLE.format(
            formula_id=formula_elem["element_id"],
            formula_variables=extract_formula_variables((formula_elem.get("content", "") or "")[:1200]),
            formula_context=_context(formula_elem),
            tbl_id=table_elem["element_id"],
            tbl_caption=(table_elem.get("caption", "") or "")[:400],
            tbl_headers=extract_table_headers((table_elem.get("content", "") or ""), max_chars=150),
            tbl_context=_context(table_elem),
            edge_context=edge_text,
        )

    return ""


# ──────────────────────────────────────────────────────────────
# API call
# ──────────────────────────────────────────────────────────────

def call_api(
    client: Any,
    model: str,
    prompt: str,
    images: List[Optional[Tuple[str, str]]],
) -> Tuple[Optional[str], int, int]:
    """Call Anthropic API. Returns (text, input_tokens, output_tokens)."""
    content: List[Dict[str, Any]] = []
    for img in images:
        if img is not None:
            b64, mime = img
            content.append({
                "type": "image",
                "source": {"type": "base64", "media_type": mime, "data": b64},
            })
    content.append({"type": "text", "text": prompt})

    r = client.messages.create(
        model=model,
        system=SYSTEM_PROMPT,
        max_tokens=1536,
        temperature=0.4,
        messages=[{"role": "user", "content": content}],
    )
    return (
        r.content[0].text,
        r.usage.input_tokens,
        r.usage.output_tokens,
    )


def parse_json(txt: Optional[str]) -> Optional[Dict[str, Any]]:
    if not txt:
        return None
    t = txt.strip()
    if t.startswith("```"):
        t = re.sub(r"^```(?:json)?\s*", "", t).strip()
        t = re.sub(r"\s*```$", "", t).strip()
    try:
        return json.loads(t)
    except Exception:
        m = re.search(r"\{.*\}", t, re.DOTALL)
        if m:
            try:
                return json.loads(m.group())
            except Exception:
                pass
    return None


# ──────────────────────────────────────────────────────────────
# Path normalization
# ──────────────────────────────────────────────────────────────

REPO_ROOTS = [
    "/projects/_hdd/myyyx1/data-process-test/",
    "/projects/myyyx1/data-process-test/",
]


def normalize_path(img_path: str) -> str:
    for root in REPO_ROOTS:
        if img_path.startswith(root):
            return img_path[len(root):]
    return img_path


# ──────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser(
        description="Generate multi-hop L1 queries from DAG candidates"
    )
    ap.add_argument(
        "--candidates",
        default="data/multihop_l1_candidates.json",
        help="Input from select_multihop_candidates.py",
    )
    ap.add_argument(
        "--output",
        default="data/l1_multihop_queries_v2.jsonl",
        help="Output JSONL path",
    )
    ap.add_argument("--model", default="claude-sonnet-4-5-20250929")
    ap.add_argument("--limit", type=int, default=0, help="Limit pairs (0=all)")
    ap.add_argument("--delay", type=float, default=0.5, help="Seconds between API calls")
    ap.add_argument("--dry-run", action="store_true", help="Print prompts without calling API")
    ap.add_argument("--no-images", action="store_true", help="Skip sending images")
    args = ap.parse_args()

    # Load candidates
    cand_path = Path(args.candidates)
    if not cand_path.exists():
        print(f"ERROR: {cand_path} not found. Run select_multihop_candidates.py first.")
        sys.exit(1)
    cand_data = json.loads(cand_path.read_text(encoding="utf-8"))
    pairs = cand_data.get("pairs", [])
    if args.limit > 0:
        pairs = pairs[:args.limit]

    print(f"Multi-hop L1 Query Generation")
    print(f"  Candidates: {len(pairs)}")
    print(f"  Model: {args.model}")
    print(f"  Images: {'disabled' if args.no_images else 'enabled'}")
    print(f"  Output: {args.output}")
    print()

    # Initialize client
    client = None
    if not args.dry_run:
        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            print("ERROR: ANTHROPIC_API_KEY not set. Run: export $(grep -v '^#' .env | xargs)")
            sys.exit(1)
        import anthropic
        client = anthropic.Anthropic(api_key=api_key)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    total_input_tokens = 0
    total_output_tokens = 0
    kept = 0
    qc_failed_count = 0
    parse_failed = 0
    query_idx = 0

    # Stats
    type_stats = defaultdict(int)
    qc_issue_stats = defaultdict(int)

    # Dry-run should never mutate output files.
    out_stream = open(os.devnull, "w", encoding="utf-8") if args.dry_run else out_path.open("w", encoding="utf-8")

    with out_stream as f:
        for i, pair in enumerate(pairs):
            doc_id = pair["doc_id"]
            pair_type = pair["pair_type"]
            hop = pair["hop_distance"]
            template_name = select_template(pair)

            # Build prompt
            prompt = build_prompt(pair)
            if not prompt:
                print(f"  [{i+1}/{len(pairs)}] SKIP (no prompt template for {pair_type})")
                continue

            # Prepare images
            images: List[Optional[Tuple[str, str]]] = []
            if not args.no_images:
                img_a = encode_image(pair["element_a"].get("image_path"))
                img_b = encode_image(pair["element_b"].get("image_path"))
                images = [img_a, img_b]
                img_count = sum(1 for x in images if x is not None)
            else:
                img_count = 0

            if args.dry_run:
                print(f"\n--- pair {i+1}/{len(pairs)}: {pair['pair_id']} ({pair_type}, {hop}-hop) ---")
                print(f"  doc: {doc_id}")
                print(f"  A: {pair['element_a_id']} ({pair['element_a_type']})")
                print(f"  B: {pair['element_b_id']} ({pair['element_b_type']})")
                print(f"  template: {template_name}")
                print(f"  images: A={'OK' if (not args.no_images and encode_image(pair['element_a'].get('image_path'))) else 'NONE'}, "
                      f"B={'OK' if (not args.no_images and encode_image(pair['element_b'].get('image_path'))) else 'NONE'}")
                print(f"  prompt preview:\n{prompt[:500]}\n...")
                continue

            print(f"  [{i+1}/{len(pairs)}] {pair['pair_id']} ({pair_type}, {hop}-hop, {img_count} imgs)...",
                  end=" ", flush=True)

            # API call
            try:
                raw, in_tok, out_tok = call_api(client, args.model, prompt, images)
                total_input_tokens += in_tok
                total_output_tokens += out_tok
            except Exception as e:
                print(f"API ERROR: {e}")
                if "rate" in str(e).lower() or "429" in str(e):
                    print("  Rate limited, waiting 30s...")
                    time.sleep(30)
                continue

            obj = parse_json(raw)
            if not obj:
                print("PARSE FAIL")
                parse_failed += 1
                continue

            queries = obj.get("queries", [])
            if not queries:
                print("NO QUERIES")
                parse_failed += 1
                continue

            pair_kept = 0
            pair_failed = 0

            for q_obj in queries:
                issues, metrics = qc_multihop_query(q_obj, pair)

                # Normalize image paths
                img_a_path = normalize_path(pair["element_a"].get("image_path", "") or "")
                img_b_path = normalize_path(pair["element_b"].get("image_path", "") or "")

                entry = {
                    "query_id": f"l1_mh_{doc_id}_{query_idx:04d}",
                    "query": q_obj.get("query", ""),
                    "answer": q_obj.get("answer", ""),
                    "doc_id": doc_id,
                    "pair_id": pair["pair_id"],
                    "element_ids": [pair["element_a_id"], pair["element_b_id"]],
                    "element_a_type": pair["element_a_type"],
                    "element_b_type": pair["element_b_type"],
                    "pair_type": pair_type,
                    "hop_distance": hop,
                    "path": pair.get("path", []),
                    "multi_hop": len(pair.get("path", [])) >= 3,
                    "cross_modal": True,
                    "image_paths": [p for p in [img_a_path, img_b_path] if p],
                    "query_type": q_obj.get("query_type", "unknown"),
                    "visual_anchors": q_obj.get("visual_anchors", []),
                    "text_evidence": q_obj.get("text_evidence", ""),
                    "qc_issues": issues,
                    "qc_pass": len(issues) == 0,
                    "qc_metrics": metrics,
                }
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")
                query_idx += 1

                if entry["qc_pass"]:
                    pair_kept += 1
                    kept += 1
                    type_stats[pair_type] += 1
                else:
                    pair_failed += 1
                    qc_failed_count += 1
                    for iss in issues:
                        qc_issue_stats[iss] += 1

            status = f"{pair_kept} OK" + (f", {pair_failed} QC fail" if pair_failed else "")
            print(status)

            if args.delay > 0 and i < len(pairs) - 1:
                time.sleep(args.delay)

    if args.dry_run:
        print(f"\nDry-run complete for {len(pairs)} pairs")
        return

    # Cost: Sonnet 4.5 = $3/M input, $15/M output
    est_cost = total_input_tokens * 3 / 1e6 + total_output_tokens * 15 / 1e6

    print(f"\n{'='*60}")
    print(f"Multi-hop L1 Generation Summary")
    print(f"{'='*60}")
    print(f"  Total pairs processed: {len(pairs)}")
    print(f"  Total queries written: {query_idx}")
    print(f"  QC passed:             {kept}")
    print(f"  QC failed:             {qc_failed_count}")
    print(f"  Parse failures:        {parse_failed}")
    print(f"  Input tokens:          {total_input_tokens:,}")
    print(f"  Output tokens:         {total_output_tokens:,}")
    print(f"  Est. cost:             ${est_cost:.2f}")
    print(f"  Output:                {out_path}")
    print(f"\n  QC passed by type:")
    for t, cnt in sorted(type_stats.items()):
        print(f"    {t}: {cnt}")
    if qc_issue_stats:
        print(f"\n  QC issue breakdown:")
        for iss, cnt in sorted(qc_issue_stats.items(), key=lambda x: -x[1]):
            print(f"    {iss}: {cnt}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
