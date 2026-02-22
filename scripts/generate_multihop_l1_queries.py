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

PROMPT_FIGURE_TABLE_1HOP = """You are a PhD student presenting experimental results at a group meeting. Ask direct, empirically grounded questions the way scientists actually discuss data — specific and precise, not convoluted.

Generate 2 cross-modal retrieval queries requiring BOTH a figure and a table from the same document.

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
{latex_bridge}

## YOUR TASK
Create queries that REQUIRE BOTH elements. The query should create an INFORMATION GAP:
describe context from one element, ask a question answerable only by the other.

## STRICT RULES
1. Query MUST be UNANSWERABLE if either the figure or the table is removed.
2. NEVER start with Do/Does/Did/Is/Are/Can/Has/Will/Would; NO yes/no questions.
3. NEVER start with "Given that" or "What causes" (avoid template collapse).
4. NEVER put specific numbers, percentages, or exact values in the query.
5. NEVER use meta-language: "figure", "table", "the text", "according to", "as shown in".
6. Max 30 words per query. Answer max 3 sentences with specific values from BOTH elements.
7. visual_anchors are PHYSICAL COORDINATES for evaluation only — must NEVER appear in the query text.
8. The two queries must use DIFFERENT aspects of the data.
9. CONCEPTUAL MASKING: You MAY use type-level names ("the fairness metric", "the regularization
   weight", "the highest-performing baseline", "the ablation without data augmentation").
   MUST NOT copy exact labels, raw row names, or symbol strings verbatim.
   NEVER use pure visual layout words in the query ("red line", "leftmost bar", "top row",
   "blue curve", "third column") — those belong only in visual_anchors.
10. CROSS-MODAL OPERATOR: pick ONE per query, use DIFFERENT operators for your 2 queries.
   ALLOWED: show, cause, drop, exceed, mismatch, require, predict, contradict, attribute, derive, converge, expose, regulate, bound, separate, reveal
   BANNED (do NOT use in query): validate, quantify, justify, demonstrate, enforce, constrain, decompose, propagate, calibrate, verify, instantiate, map, relate, align, explain
11. Avoid weak templates: "Which component..." "How does X relate to Y..."
12. Answer must include a relationship connector: because / due to / consistent with /
    constrained by / compared with / whereas / despite / under.

## SENTENCE STRUCTURE — your 2 queries MUST use different structures from this list:
- HOW-DISCREPANCY: "How does [difference in A] correspond to [constraint in B]?"
- UNDER-CONDITION: "Under what condition does [pattern in A] appear, given [mechanism in B]?"
- WHY-INCONSISTENT: "Why is [pattern in A] different from [expectation from B]?"
- WHICH-MAPPING:  "Which mapping between [A regime] and [B condition] accounts for [outcome]?"
- WHAT-CONSTRAINT: "What constraint in [B] limits [behavior in A]?"

## BAD vs GOOD examples
BAD: "Did the red line peak at 90,000 and match the keyword set?" — yes/no, visual coords
BAD: "Which configuration best validates the theoretical optimum?" — banned verb "validates"
BAD: "How does the high-recall setting relate to the fairness tradeoff?" — banned "relate", vague

GOOD (HOW-DISCREPANCY): "How does the drop in session frequency after later positions correspond to the retrieval-window setting that yields the highest scores?"
GOOD (WHY-INCONSISTENT): "Why is the ensemble gain larger on the minority subgroup than on the aggregate benchmark under the same evaluation regime?"
GOOD (WHAT-CONSTRAINT): "What constraint in the interaction pattern limits the accuracy gain at low-resource settings?"

## Output format (JSON only):
{{
  "queries": [
    {{
      "query": "open-ended question with operator word, max 30 words, NO specific values",
      "answer": "factual answer citing specific values from BOTH elements, max 3 sentences, with connector",
      "query_type": "trend_explanation|parameter_outcome|cross_reading|anomaly_investigation",
      "required_evidence_spans": [
        {{"element_id": "{fig_id}", "span": "short extractive phrase from figure caption/content (semantic concept)", "evidence_type": "observation"}},
        {{"element_id": "{tbl_id}", "span": "short extractive phrase from table headers/content (semantic concept)", "evidence_type": "result"}}
      ],
      "visual_anchors": [
        {{"element_id": "{fig_id}", "anchor": "physical coords only: color/position/curve (evaluation only — NOT in query)"}},
        {{"element_id": "{tbl_id}", "anchor": "physical coords only: row/column/cell (evaluation only — NOT in query)"}}
      ],
      "text_evidence": "direct quote from context, min 40 chars"
    }}
  ]
}}"""

PROMPT_FIGURE_TABLE_2HOP = """You are a PhD student presenting experimental results at a group meeting. Ask direct, empirically grounded questions the way scientists actually discuss data — specific and precise, not convoluted.

Generate 2 cross-modal retrieval queries requiring BOTH a figure and a table with CHAIN REASONING.

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
{latex_bridge}

## YOUR TASK
Create queries that REQUIRE BOTH elements with CHAIN REASONING.
The intermediate element is the bridge — use it as a cognitive stepping stone.

## STRICT RULES
1. Query MUST be UNANSWERABLE if either endpoint or bridge is removed.
2. NEVER start with Do/Does/Did/Is/Are/Can/Has/Will/Would; NO yes/no questions.
3. NEVER start with "Given that" or "What causes" (avoid template collapse).
4. NEVER put specific numbers or exact values in the query.
5. NEVER use meta-language: "figure", "table", "the text", "according to", "as shown in".
6. Max 30 words per query; answer max 3 sentences with values from BOTH endpoints.
7. visual_anchors are PHYSICAL COORDINATES for evaluation only — must NEVER appear in the query.
8. CONCEPTUAL MASKING: You MAY use type-level names. MUST NOT copy exact labels or visual
   layout words ("red line", "leftmost bar") in the query.
9. CROSS-MODAL OPERATOR: pick ONE per query, use DIFFERENT operators for your 2 queries.
   ALLOWED: show, cause, drop, exceed, mismatch, require, predict, contradict, attribute, derive, converge, expose, regulate, bound, separate, reveal
   BANNED (do NOT use in query): validate, quantify, justify, demonstrate, enforce, constrain, decompose, propagate, calibrate, verify, instantiate, map, relate, align, explain
10. Avoid weak templates: "Which component..." "How does X relate to Y..."
11. Answer must include a relationship connector: because / due to / consistent with /
    constrained by / compared with / whereas / despite / under.

## SENTENCE STRUCTURE — your 2 queries MUST use different structures from this list:
- HOW-DISCREPANCY: "How does [difference in A] correspond to [constraint in B]?"
- UNDER-CONDITION: "Under what condition does [pattern in A] appear, given [mechanism in B]?"
- WHY-INCONSISTENT: "Why is [pattern in A] different from [expectation from B]?"
- WHICH-MAPPING:  "Which mapping between [A regime] and [B condition] accounts for [outcome]?"
- WHAT-CONSTRAINT: "What constraint in [B] limits [behavior in A]?"

## Output format (JSON only):
{{
  "queries": [
    {{
      "query": "chain-reasoning question with operator word, max 30 words, NO specific values",
      "answer": "factual answer citing values from both elements, max 3 sentences, with connector",
      "query_type": "chain_verification|parameter_outcome|condition_result|bridge_reasoning",
      "required_evidence_spans": [
        {{"element_id": "{fig_id}", "span": "short extractive phrase from figure (semantic concept)", "evidence_type": "observation"}},
        {{"element_id": "{tbl_id}", "span": "short extractive phrase from table (semantic concept)", "evidence_type": "result"}}
      ],
      "bridge": {{
        "element_id": "{intermediate_info}",
        "anchor": "key property of the bridge element used in the chain",
        "evidence_span": "extractive phrase connecting bridge to both endpoints"
      }},
      "visual_anchors": [
        {{"element_id": "{fig_id}", "anchor": "physical coords only — evaluation only, NOT in query"}},
        {{"element_id": "{tbl_id}", "anchor": "physical coords only — evaluation only, NOT in query"}}
      ],
      "text_evidence": "direct quote from context, min 40 chars"
    }}
  ]
}}"""

PROMPT_FIGURE_FORMULA = """You are a PhD student presenting experimental results at a group meeting. Ask questions that connect what you see in experiment plots or model diagrams to what the math predicts or requires.

Generate 2 cross-modal retrieval queries connecting a figure with a mathematical formula.

## Figure ({fig_id})
Caption: {fig_caption}
Context: {fig_context}
[Figure image is provided above]

## Formula ({formula_id})
Key variables: {formula_variables}
Context: {formula_context}

## Textual connection
{edge_context}
{latex_bridge}

## CORE PRINCIPLE: DUAL-EVIDENCE INFORMATION GAP

Your query must create an information gap that ONLY closes when evidence from BOTH the figure AND the formula is combined. Think of it as two puzzle halves:

**Half A (Figure):** Something OBSERVABLE in the figure — a trend, a structural choice, a comparison, a spatial arrangement, a quantitative pattern. Identify a SPECIFIC, CONCRETE observation.
**Half B (Formula):** Something DERIVABLE from the formula — a mathematical constraint, an optimization objective, a theoretical bound, a functional relationship. Identify a SPECIFIC mathematical mechanism.

The query asks WHY or HOW Half A is explained / constrained / justified by Half B. The answer MUST cite concrete details from BOTH halves.

## FIGURE TYPE STRATEGY (CRITICAL)

**If the figure shows QUANTITATIVE results** (curves, bar charts, heatmaps, scatter plots):
- Half A = a specific trend, peak, crossing point, gap, plateau, or comparison between conditions.

**If the figure shows STRUCTURAL/ARCHITECTURAL diagrams** (network topology, causal graphs, pipeline flowcharts, encoder-decoder):
- Half A = a specific STRUCTURAL CHOICE: number of branches, where a loss is applied, which components share weights, direction of information flow, separation of pathways.
- Name the structural choice concretely (e.g., "two separate encoder branches feeding a bottleneck", "the adversarial path after the feature extractor").
- Ask: how does this specific structural choice satisfy / enforce / follow from the mathematical constraint in the formula?

## CROSS-MODAL OPERATOR: pick ONE per query, use DIFFERENT operators for your 2 queries.
ALLOWED: show, cause, drop, exceed, mismatch, require, predict, contradict, attribute, derive, converge, expose, regulate, bound, separate, reveal
BANNED (do NOT use in query): validate, quantify, justify, demonstrate, enforce, constrain, decompose, propagate, calibrate, verify, instantiate, map, relate, align, explain

## STRICT RULES
1. Query MUST be UNANSWERABLE without BOTH the figure AND the formula. Test: if you remove the figure, can you still answer from the formula alone? If yes — REJECT and rewrite.
2. NEVER start with Do/Does/Did/Is/Are/Can/Has/Will/Would; NO yes/no questions.
3. NEVER start with "Given that" or "What causes" (avoid template collapse).
4. NEVER copy raw LaTeX strings or raw variable symbols into the query.
5. You MAY use at most ONE anchor value in the query only if needed to preserve directionality.
6. NEVER use meta-language: "equation", "formula", "figure", "as shown in", "diagram", "architecture".
7. Max 30 words per query.
8. visual_anchors are PHYSICAL COORDINATES for evaluation only — must NEVER appear in the query.
9. CONCEPTUAL MASKING: Use type-level names ("the regularization weight", "the penalty term", "the class-conditional probability"). MUST NOT copy raw variable names verbatim.
10. FORMULA MASKING: Describe variables by their MATHEMATICAL/PHYSICAL MEANING, NEVER as standalone letters.
11. Avoid weak templates: "Which component..." "How does X relate to Y..." "What role does..."
12. Answer must include a relationship connector (because / due to / consistent with /
    constrained by / compared with / whereas / despite / under).

## SELF-CHECK before outputting each query

Ask yourself:
1. Does the answer cite a SPECIFIC, CONCRETE observation from the figure (trend/structure/comparison)?
2. Does the answer cite a SPECIFIC mathematical mechanism from the formula (constraint/bound/objective)?
3. If I removed either piece of evidence, would the answer become incomplete or wrong?

If any answer is NO — rewrite.

## SENTENCE STRUCTURE — your 2 queries MUST use different structures from this list:
- HOW-DISCREPANCY: "How does [difference in A] correspond to [constraint in B]?"
- UNDER-CONDITION: "Under what condition does [pattern in A] appear, given [mechanism in B]?"
- WHY-INCONSISTENT: "Why is [pattern in A] different from [expectation from B]?"
- WHICH-MAPPING:  "Which mapping between [A regime] and [B condition] accounts for [outcome]?"
- WHAT-CONSTRAINT: "What constraint in [B] limits [behavior in A]?"

## BAD vs GOOD examples

BAD: "How does the two-branch encoder instantiate the disentanglement penalty?" — banned "instantiate"
BAD: "Why does the adversarial discriminator enforce the mutual exclusivity constraint?" — banned "enforce", banned "constraint"
BAD: "What role does the regularization term play in the convergence pattern?" — vague, no operator

GOOD (HOW-DISCREPANCY, quantitative figure):
"How does the plateau in selection rates correspond to the sharper utility drop in the minority group under the same threshold region?"
→ Figure half: minority vs majority utility curves diverge at a specific selection rate.
→ Formula half: the fairness constraint bounds minority selection independently of the aggregate rate.

GOOD (WHAT-CONSTRAINT, architectural figure):
"What constraint in the penalty term limits correlation between the two encoder-path representations?"
→ Figure half: two parallel encoder branches with no shared weights (specific structural choice).
→ Formula half: the penalty minimizes mutual information between branch outputs.

## Output format (JSON only):
{{
  "queries": [
    {{
      "query": "max 30 words, one operator word, NO LaTeX/values/letters/meta-language",
      "answer": "max 4 sentences with connector. Must cite both halves explicitly.",
      "answer_figure_evidence": "1-2 sentences: what specific observation from the figure is cited (trend/structure/comparison). Start with the concrete observable.",
      "answer_formula_evidence": "1-2 sentences: what specific mathematical mechanism from the formula is cited (constraint/bound/objective). Start with the mechanism.",
      "query_type": "theory_vs_experiment|structural_justification|parameter_sensitivity|boundary_behavior|convergence_analysis",
      "required_evidence_spans": [
        {{"element_id": "{fig_id}", "span": "specific observable feature: name the trend/structure/comparison concretely", "evidence_type": "observation"}},
        {{"element_id": "{formula_id}", "span": "specific mathematical mechanism: name the constraint/term/property (NOT generic 'the formula')", "evidence_type": "mechanism"}}
      ],
      "visual_anchors": [
        {{"element_id": "{fig_id}", "anchor": "physical coords only: color/position/curve/branch/edge — NOT in query"}},
        {{"element_id": "{formula_id}", "anchor": "specific term or variable — NOT standalone letter in query"}}
      ],
      "text_evidence": "direct quote from context, min 40 chars"
    }}
  ]
}}"""

PROMPT_FORMULA_TABLE = """You are a PhD student presenting experimental results at a group meeting. Ask questions that connect what the theory (formula) predicts to what the numbers (table) actually show.

Generate 2 cross-modal retrieval queries connecting a formula with a table from the same document.

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
{latex_bridge}

## YOUR TASK
Create queries that connect mathematical relationships with tabular evidence.
The query must require BOTH the formula's theoretical structure AND the table's data.

## STRICT RULES
1. Query MUST be UNANSWERABLE without both the formula and the table.
2. NEVER start with Do/Does/Did/Is/Are/Can/Has/Will/Would; NO yes/no questions.
3. NEVER start with "Given that" or "What causes" (avoid template collapse).
4. NEVER copy raw LaTeX strings or raw variable symbols into the query.
5. You MAY use at most ONE anchor value in the query only if needed to preserve directionality.
6. NEVER use meta-language: "table", "equation", "formula", "the text".
7. Max 30 words per query; answer max 3 sentences with specific values from the table.
8. visual_anchors are PHYSICAL COORDINATES for evaluation only — must NEVER appear in the query.
9. CONCEPTUAL MASKING: You MAY use type-level names. MUST NOT copy exact row names, column
   headers, or variable strings verbatim.
10. FORMULA MASKING: Describe variables by their MATHEMATICAL/PHYSICAL MEANING in context,
   NEVER as standalone letters (NOT "β" but "the smoothing parameter").
11. CROSS-MODAL OPERATOR: pick ONE per query, use DIFFERENT operators for your 2 queries.
   ALLOWED: show, cause, drop, exceed, mismatch, require, predict, contradict, attribute, derive, converge, expose, regulate, bound, separate, reveal
   BANNED (do NOT use in query): validate, quantify, justify, demonstrate, enforce, constrain, decompose, propagate, calibrate, verify, instantiate, map, relate, align, explain
12. Avoid weak templates: "Which component..." "How does X relate to Y..."
13. Answer must include a relationship connector: because / due to / consistent with /
    constrained by / compared with / whereas / despite / under.

## SENTENCE STRUCTURE — your 2 queries MUST use different structures from this list:
- HOW-DISCREPANCY: "How does [difference in A] correspond to [constraint in B]?"
- UNDER-CONDITION: "Under what condition does [pattern in A] appear, given [mechanism in B]?"
- WHY-INCONSISTENT: "Why is [pattern in A] different from [expectation from B]?"
- WHICH-MAPPING:  "Which mapping between [A regime] and [B condition] accounts for [outcome]?"
- WHAT-CONSTRAINT: "What constraint in [B] limits [behavior in A]?"

## BAD vs GOOD examples
BAD: "Does beta=0.3 in row 4 satisfy Eq. 2?" — yes/no, numbers, meta-language
BAD: "What does N represent?" — standalone letter, vague
BAD: "Which configuration best instantiates the theoretical threshold?" — banned "instantiate"

GOOD (HOW-DISCREPANCY): "How does the smoothing-driven sparsity regime correspond to the steeper accuracy drop under low-resource conditions?"
GOOD (WHY-INCONSISTENT): "Why is the highest-regularization setting better than the unregularized baseline in out-of-domain evaluation despite overfitting risk?"

## Output format (JSON only):
{{
  "queries": [
    {{
      "query": "formula-data question with operator word, max 30 words, NO LaTeX/values/letters",
      "answer": "factual answer with specific values from both, max 3 sentences, with connector",
      "query_type": "formula_instantiation|data_formula_consistency|sensitivity_analysis|unit_verification",
      "required_evidence_spans": [
        {{"element_id": "{formula_id}", "span": "short extractive phrase describing the formula's role or constraint (NOT raw LaTeX)", "evidence_type": "constraint"}},
        {{"element_id": "{tbl_id}", "span": "short extractive phrase from table headers/content (semantic concept)", "evidence_type": "result"}}
      ],
      "visual_anchors": [
        {{"element_id": "{formula_id}", "anchor": "specific term or variable — NOT standalone letter in query"}},
        {{"element_id": "{tbl_id}", "anchor": "physical coords only: row/column/cell — NOT in query"}}
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

TEMPLATED_QUERY_OPENINGS = (
    "given that",
    "what causes",
)

RELATION_CONNECTORS = {
    "because", "due to", "therefore", "thus", "hence",
    "leads to", "results in", "explains", "matches", "corresponds to",
    "driven by", "caused by", "consistent with", "deviates from",
    "constrained by", "compared with", "whereas", "despite", "under",
}

# Cross-modal operators accepted by QC (superset of prompt-allowed operators).
# v4.2: "validate/quantify/justify/demonstrate/enforce/constrain/decompose/propagate/calibrate/verify"
# are banned from prompts (prevent academic jargon), but still accepted here so v4/v4.1 data
# is not retroactively penalized.
CROSS_MODAL_OPERATORS = {
    "verify", "verified", "verifies",
    "derive", "derived", "derives",
    "map", "maps", "mapped",
    "align", "aligns", "aligned",
    "contradict", "contradicts", "contradicted",
    "explain", "explains", "explained",
    "instantiate", "instantiates", "instantiated",   # accepted by QC, banned in prompt
    "calibrate", "calibrates", "calibrated",
    "attribute", "attributes", "attributed",
    "quantify", "quantifies", "quantified",
    "predict", "predicts", "predicted",
    "justify", "justifies", "justified",
    "enforce", "enforces", "enforced",
    "converge", "converges", "converged",
    "reveal", "reveals", "revealed",
    "validate", "validates", "validated",
    "expose", "exposes", "exposed",
    "constrain", "constrains", "constrained",
    "decompose", "decomposes", "decomposed",
    "propagate", "propagates", "propagated",
    "regulate", "regulates", "regulated",
    "bound", "bounds", "bounded",
    "separate", "separates", "separated",
    # v4.2 allowed prompt operators (also accepted by QC)
    "show", "shows", "showed",
    "cause", "causes", "caused",
    "drop", "drops", "dropped",
    "exceed", "exceeds", "exceeded",
    "mismatch", "mismatches", "mismatched",
    "require", "requires", "required",
    # v4.2 natural English verbs added to QC (not in prompt list but too common to penalise)
    "affect", "affects", "affected",
    "differ", "differs", "differed",
    "increase", "increases", "increased",
    "decrease", "decreases", "decreased",
    "change", "changes", "changed",
    "improve", "improves", "improved",
    "reduce", "reduces", "reduced",
    "lead", "leads",
    "result", "results", "resulted",
    "indicate", "indicates", "indicated",
    "suggest", "suggests", "suggested",
    "support", "supports", "supported",
    "reflect", "reflects", "reflected",
    "produce", "produces", "produced",
    "impact", "impacts", "impacted",
    "correlate", "correlates", "correlated",
    "matter", "matters", "mattered",
    "achieve", "achieves", "achieved",
    "occur", "occurs", "occurred",
    "remain", "remains", "remained",
    "shift", "shifts", "shifted",
    "fail", "fails", "failed",
    "vary", "varies", "varied",
    "scale", "scales", "scaled",
}


def has_numeric_leakage(query: str) -> bool:
    """Flag queries that leak specific numeric values (v4: threshold 1, not 2)."""
    nums = re.findall(r"\b\d+(?:[.,]\d+)?%?\b", query)
    suspicious = []
    for raw in nums:
        token = raw.replace(",", "").rstrip("%")
        try:
            value = float(token)
        except ValueError:
            continue
        # allow 0 and 1 (universal constants / boolean-like)
        if value in (0.0, 1.0):
            continue
        # allow years
        if 1900 <= value <= 2099 and value.is_integer():
            continue
        suspicious.append(raw)
    return len(suspicious) >= 1


def has_no_cross_modal_operator(query: str) -> bool:
    """Flag queries missing an explicit cross-modal operator word (v4)."""
    q = query.lower()
    return not any(op in q for op in CROSS_MODAL_OPERATORS)


def check_evidence_spans(obj: Dict[str, Any], pair: Dict[str, Any]) -> bool:
    """Return True if required_evidence_spans covers both elements with non-trivial spans.
    Also checks answer_figure_evidence / answer_formula_evidence for figure+formula (v4.1).
    """
    spans = obj.get("required_evidence_spans", [])
    if not spans or len(spans) < 2:
        return False
    elem_ids = {pair.get("element_a_id", ""), pair.get("element_b_id", "")}
    covered: Set[str] = set()
    for s in spans:
        span_text = s.get("span", "").strip()
        eid = s.get("element_id", "")
        if eid and len(span_text) >= 8:  # non-trivial span
            covered.add(eid)
    return len(covered & elem_ids) >= 2


def is_yes_no_question(query: str) -> bool:
    q = query.strip().lower()
    if any(q.startswith(s) for s in YES_NO_STARTERS):
        return True
    # If the query contains ANY wh-word anywhere, it cannot be a yes/no question.
    # This handles "Given that X are Y, why does Z..." where 'are' from the subordinate
    # clause would otherwise trigger a false positive before 'why' is seen.
    if re.search(r'\b(?:what|which|where|why|how|whose|whom)\b', q):
        return False
    # Catch prefixed forms like "At n=1000, does X hold?" — no wh-word present.
    prefix_m = re.match(
        r"^(?:in|at|on|for|under|with|when|while|given|after|before)\b",
        q,
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


def has_shortcut_template(query: str) -> bool:
    q = query.strip().lower()
    return any(re.search(p, q) for p in QUERY_SHORTCUT_PATTERNS)


def has_templated_opening(query: str) -> bool:
    q = query.strip().lower()
    return any(q.startswith(prefix) for prefix in TEMPLATED_QUERY_OPENINGS)


def has_relationship_connector(answer: str) -> bool:
    a = answer.lower()
    return any(conn in a for conn in RELATION_CONNECTORS)


def has_premise_answer_contradiction(query: str, answer: str) -> bool:
    """High-precision contradiction checks to catch premise reversal artifacts."""
    q = query.lower()
    a = answer.lower()

    # Common generated artifact: premise says "drop below", answer negates it.
    if "drop below" in q and re.search(r"\b(?:does|do|did)\s+not\s+drop below\b|\bnot drop below\b", a):
        return True

    # Common generated artifact: premise says "drop more unlabeled users",
    # answer says "fewer unlabeled users".
    if "drop more unlabeled users" in q and "fewer unlabeled users" in a:
        return True

    # Guard against reversed comparative phrasing with explicit numeric evidence.
    if "with fewer" in q and "with more" in q:
        nums = [float(n.replace(",", "")) for n in re.findall(r"\b\d+(?:\.\d+)?\b", answer)]
        if len(nums) >= 2 and nums[0] > nums[1]:
            return True

    # Generic negation check for directional verbs in "why does/do" questions.
    for verb in ("increase", "decrease", "drop", "rise", "fall", "exceed", "improve"):
        if re.search(rf"\bwhy\s+do(?:es)?\b.*\b{verb}\b", q) and re.search(
            rf"\b(?:does|do|did)\s+not\s+{verb}\b", a
        ):
            return True

    return False


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

    # 2e. Templated opening collapse
    if has_templated_opening(q):
        issues.append("templated_opening")
        metrics["templated_opening_warn"] = True

    # 3. Short answer
    if len(a) < 20:
        issues.append("short_answer")

    # 4. Empty query
    if not q or len(q) < 10:
        issues.append("empty_query")

    # 4b. Premise-answer contradiction (high precision, hard fail)
    if has_premise_answer_contradiction(q, a):
        issues.append("premise_answer_contradiction")
        metrics["premise_contradiction_warn"] = True

    # 5. Anchor leakage
    leak = anchor_leak_jaccard(q, anchors)
    metrics["anchor_leak_jaccard"] = round(leak, 4)
    if leak > ANCHOR_LEAK_THRESHOLD:
        issues.append("anchor_leakage")
    anchor_copy = anchor_token_copy_count(q, anchors)
    metrics["anchor_token_copy_count"] = anchor_copy
    # v4 (Option A): bridge_entity_leakage is a SOFT tracking metric only,
    # not a hard QC fail. For dual-evidence (path_len=2) queries, entity names
    # are legitimate search targets — hiding them makes no sense without multi-hop.
    if anchor_copy >= 4:
        metrics["bridge_entity_leakage_warn"] = True

    # 5b. Cross-modal operator check (v4 new)
    if has_no_cross_modal_operator(q):
        issues.append("no_cross_modal_operator")

    # 5c. Required evidence spans (v4 new)
    if not check_evidence_spans(obj, pair):
        issues.append("evidence_spans_incomplete")

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
    # For formula elements: raw content is LaTeX (low overlap with natural-language answer).
    # Use context_before + context_after instead, which contains the prose explanation.
    a_tokens = _content_tokens(a)
    if a_tokens:
        def _elem_text(elem: Dict) -> str:
            caption = elem.get("caption", "") or ""
            etype   = elem.get("element_type", "")
            if etype == "formula":
                # LaTeX content has poor token overlap; prefer surrounding prose
                return (caption + " " +
                        (elem.get("context_before", "") or "") + " " +
                        (elem.get("context_after", "") or ""))
            return caption + " " + (elem.get("content", "") or "")

        ctx_a = _elem_text(pair.get("element_a", {}))
        ctx_b = _elem_text(pair.get("element_b", {}))
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

    # 9. Encourage explicit relationship grounding instead of generic lookup answers.
    # v2.1: cross_reading is a lookup/referencing type, not explanatory — exempt from check.
    qtype = str(obj.get("query_type", "")).lower()
    explanatory_types = {
        "trend_explanation",
        "anomaly_investigation",
        "bridge_reasoning",
        "theory_vs_experiment",
        "data_formula_consistency",
    }
    if qtype in explanatory_types and not has_relationship_connector(a):
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


def build_latex_bridge_section(pair: Dict) -> str:
    """Extract full latex_bridge.bridge_text as an 'author's own words' section.

    This is the key new input from latex_cross_modal_pairs.json: the exact
    sentence(s) the paper's authors wrote to connect the two elements via \\ref{}.
    Using the full text (not the 300-char truncated context_snippet) gives the
    model semantic grounding without exposing raw content that causes anchor leakage.
    """
    lb = pair.get("latex_bridge", {})
    if not lb:
        return ""
    bridge = lb.get("bridge_text", "").strip()
    if not bridge:
        return ""
    strategy = lb.get("strategy", "")
    label_a = lb.get("label_a", "")
    label_b = lb.get("label_b", "")
    # Strip raw LaTeX commands (e.g. \includegraphics, \caption{...}) that
    # carry no semantic value and may leak visual implementation details.
    bridge = re.sub(r'\\includegraphics[^}]*\}', '', bridge)
    bridge = re.sub(r'\\[a-zA-Z]+\*?\s*(?:\[[^\]]*\])?\{[^}]{0,80}\}', ' ', bridge)
    bridge = re.sub(r'\\[a-zA-Z]+\*?', ' ', bridge)
    bridge = re.sub(r'[${}]', ' ', bridge)
    bridge = re.sub(r'\s+', ' ', bridge).strip()
    if len(bridge) < 20:
        return ""  # nothing useful left after stripping
    header = "## Author's connection (from LaTeX source)\n"
    meta = f"[Labels: {label_a} ↔ {label_b}, strategy: {strategy}]\n" if label_a else ""
    return header + meta + f'"{bridge[:600]}"'


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

    latex_bridge_section = build_latex_bridge_section(pair)

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
            latex_bridge=latex_bridge_section,
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
            latex_bridge=latex_bridge_section,
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
            latex_bridge=latex_bridge_section,
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
            latex_bridge=latex_bridge_section,
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
        default="data/latex_cross_modal_pairs.json",
        help="Input candidate pairs (latex_cross_modal_pairs.json or multihop_l1_candidates.json)",
    )
    ap.add_argument(
        "--output",
        default="data/l1_dual_evidence_queries_v3.jsonl",
        help="Output JSONL path",
    )
    ap.add_argument(
        "--pass-only",
        action="store_true",
        help="Also write a pass-only subset to {output_stem}_pass.jsonl alongside the full output",
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

    print(f"Dual-Evidence L1 Query Generation (v4.2)")
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
    pass_path = out_path.with_name(out_path.stem + "_pass" + out_path.suffix) if args.pass_only else None

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
    pass_stream = open(os.devnull, "w", encoding="utf-8") if (args.dry_run or not pass_path) else pass_path.open("w", encoding="utf-8")

    with out_stream as f, pass_stream as fp:
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
                    "query_id": f"l1_de_{doc_id}_{query_idx:04d}",
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
                    "dual_evidence": True,   # v4: renamed from multi_hop (path_len always 2 for single-doc pairs)
                    "cross_modal": True,
                    "image_paths": [p for p in [img_a_path, img_b_path] if p],
                    "quality_tier": pair.get("quality_tier", "unknown"),
                    "query_type": q_obj.get("query_type", "unknown"),
                    "required_evidence_spans": q_obj.get("required_evidence_spans", []),
                    "visual_anchors": q_obj.get("visual_anchors", []),
                    "text_evidence": q_obj.get("text_evidence", ""),
                    "qc_issues": issues,
                    "qc_pass": len(issues) == 0,
                    "qc_metrics": metrics,
                }
                # Always write all entries to main file
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")
                # Also write to pass-only file if enabled
                if pass_path and entry["qc_pass"]:
                    fp.write(json.dumps(entry, ensure_ascii=False) + "\n")
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
    print(f"Dual-Evidence L1 Generation Summary (v4.2)")
    print(f"{'='*60}")
    print(f"  Total pairs processed: {len(pairs)}")
    print(f"  Total queries written: {query_idx}")
    print(f"  QC passed:             {kept}")
    print(f"  QC failed:             {qc_failed_count}")
    print(f"  Parse failures:        {parse_failed}")
    print(f"  Input tokens:          {total_input_tokens:,}")
    print(f"  Output tokens:         {total_output_tokens:,}")
    print(f"  Est. cost:             ${est_cost:.2f}")
    print(f"  Output (full):         {out_path}")
    if pass_path:
        print(f"  Output (pass-only):    {pass_path}")
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
