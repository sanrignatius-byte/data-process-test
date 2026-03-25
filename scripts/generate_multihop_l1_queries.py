#!/usr/bin/env python3
"""Generate cross-modal dual-evidence L1 queries from DAG candidates.

Reads multihop_l1_candidates.json (from select_multihop_candidates.py),
sends element pairs to Claude Vision API with modality-specific prompts,
and outputs QC-filtered queries to l1_multihop_queries_v2.jsonl.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import os
import random
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
sys.path.insert(0, str(PROJECT_ROOT))
from src.utils.token_logger import log_run

# ──────────────────────────────────────────────────────────────
# P0: Bridge paragraph text resolver — loads raw paragraph
# context from latex_reference_graph.json so L3 prompts can
# inject real bridge text instead of empty placeholders.
# ──────────────────────────────────────────────────────────────

_BRIDGE_TEXT_CACHE: Dict[str, Dict[str, str]] = {}  # {doc_id: {label: text}}
_ELEMENT_TO_LABELS: Dict[str, List[str]] = {}  # {element_id: [latex_labels]}
_SECTION_ENRICH_CACHE: Dict[str, Dict[str, Any]] = {}  # {section_id: enrichment row}


# P0 核心：从 LaTeX 引用图提取边 context 作为 bridge 段落文本
# 链路：element_id → LaTeX label → ref graph edge → context
def load_reference_graph_bridge_texts(
    ref_graph_path: str,
    topology_candidates_path: str = "",
) -> None:
    """Pre-load paragraph contexts from latex_reference_graph.json.

    Also loads topology candidates to build element_id → LaTeX label mapping,
    so we can resolve MinerU element IDs to the LaTeX labels used in edge contexts.

    Bridge text is found by: element_id → LaTeX label → edges referencing that label
    → edge context = the bridge paragraph text.
    """
    if not ref_graph_path or not Path(ref_graph_path).exists():
        return
    data = json.loads(Path(ref_graph_path).read_text(encoding="utf-8"))
    docs = data.get("documents", {})
    for doc_id, doc in docs.items():
        ctx_by_label: Dict[str, List[str]] = defaultdict(list)

        # Index edge contexts by target label (the element being referenced)
        for edge in doc.get("edges", []):
            ctx = (edge.get("context", "") or "").strip()
            if len(ctx) < 20:
                continue
            # Skip containment edges ("fig:X is within sec:Y")
            if " is within " in ctx:
                continue
            ctx_clean = _clean_latex_bridge(ctx)
            if len(ctx_clean) < 20:
                continue
            tgt = edge.get("target_label", "")
            if tgt:
                ctx_by_label[tgt].append(ctx_clean)

        _BRIDGE_TEXT_CACHE[doc_id] = {
            k: " | ".join(dict.fromkeys(vs[:3]))  # dedup while preserving order
            for k, vs in ctx_by_label.items()
        }

    # Build element_id → LaTeX label mapping from topology candidates
    # The topology uses node_ids like "1904.03310::el::tab:lm_cor" which
    # contain the LaTeX label, while enriched candidates use MinerU IDs
    # like "1904.03310_table_1". The enrichment step mapped between them.
    _build_element_label_map_from_topology(topology_candidates_path)
    _build_element_label_map_from_ref_graph(data)


# 加载 section-level 语义摘要，供 section-aware 路径注入 prompt
def load_section_enrichments(section_enrich_path: str) -> None:
    """Load section/subsection enrichment JSON keyed by section_id."""
    _SECTION_ENRICH_CACHE.clear()
    if not section_enrich_path or not Path(section_enrich_path).exists():
        return
    data = json.loads(Path(section_enrich_path).read_text(encoding="utf-8"))
    rows = data.get("sections", []) if isinstance(data, dict) else []
    for row in rows:
        if not isinstance(row, dict):
            continue
        sid = str(row.get("section_id", "") or "").strip()
        if sid:
            _SECTION_ENRICH_CACHE[sid] = row


# 从 topology 候选的 node_id 拆出 LaTeX label（"doc::el::fig:X" → "fig:X"）
def _build_element_label_map_from_topology(topo_path: str) -> None:
    """Build element_id → label mapping from topology candidates."""
    if not topo_path or not Path(topo_path).exists():
        return
    topo = json.loads(Path(topo_path).read_text(encoding="utf-8"))
    for cand in topo.get("candidates", []):
        path_ids = cand.get("path_node_ids", [])
        path_types = cand.get("path_node_types", [])
        for nid, ntype in zip(path_ids, path_types):
            if ntype in ("figure", "table", "formula", "equation"):
                # Extract LaTeX label from node_id: "doc::el::fig:cooking" → "fig:cooking"
                if "::el::" in nid:
                    latex_label = nid.split("::el::")[-1]
                    doc_id = nid.split("::")[0]
                    # We'll match this to element_ids later
                    _ELEMENT_TO_LABELS.setdefault(f"{doc_id}::{ntype}", []).append(latex_label)


# 按 type+line_no 排序分配序号，建立 MinerU 风格 element_id → label 映射
def _build_element_label_map_from_ref_graph(data: Dict) -> None:
    """Build element_id → label mapping from reference graph labels.

    Uses caption matching: MinerU element captions ↔ LaTeX label captions.
    Since we don't have MinerU elements here, we build a label_type:ordinal → label
    index that resolve_bridge_texts_for_path can use.
    """
    for doc_id, doc in data.get("documents", {}).items():
        labels = doc.get("labels", {}) or {}
        # Group labels by type, sorted by line_no
        by_type: Dict[str, List[Tuple[int, str]]] = defaultdict(list)
        for label_key, info in labels.items():
            lt = (info.get("label_type", "") or "").lower()
            # Normalize type
            if "fig" in lt:
                etype = "figure"
            elif "tab" in lt:
                etype = "table"
            elif "eq" in lt or "formula" in lt:
                etype = "formula"
            else:
                continue
            line_no = int(info.get("line_no", 0)) if isinstance(info.get("line_no"), int) else 0
            by_type[etype].append((line_no, label_key))

        # Sort by line_no and assign ordinal (1-based to match MinerU numbering)
        for etype, items in by_type.items():
            items.sort()
            for ordinal, (_, label_key) in enumerate(items, start=1):
                element_id = f"{doc_id}_{etype}_{ordinal}"
                if element_id not in _ELEMENT_TO_LABELS:
                    _ELEMENT_TO_LABELS[element_id] = []
                _ELEMENT_TO_LABELS[element_id].append(label_key)


# 清洗 LaTeX 命令但保留语义：\ref{X}→[X], \cite{Y}→[cite:Y]
def _clean_latex_bridge(text: str) -> str:
    """Strip LaTeX commands from bridge text while preserving semantic content."""
    text = re.sub(r'\\includegraphics[^}]*\}', '', text)
    text = re.sub(r'\\(?:ref|eqref|autoref|cref|Cref)\{([^}]*)\}', r'[\1]', text)
    text = re.sub(r'\\cite\{([^}]*)\}', r'[cite:\1]', text)
    text = re.sub(r'\\[a-zA-Z]+\*?\s*(?:\[[^\]]*\])?\{([^}]{0,120})\}', r'\1', text)
    text = re.sub(r'\\[a-zA-Z]+\*?', ' ', text)
    text = re.sub(r'[${}]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


# element_id → LaTeX label → edge context：沿映射链取回作者写的桥接原文
def resolve_bridge_texts_for_path(pair: Dict) -> List[str]:
    """Given a candidate pair with path, resolve actual bridge paragraph texts.

    Strategy: map MinerU element_ids → LaTeX labels (via ordinal mapping),
    then look up edge contexts referencing those labels in the reference graph.
    The edge context IS the bridge paragraph text — the sentence where the
    author connects two elements via \\ref{}.

    Returns a list of bridge paragraph texts (cleaned, max 3).
    """
    elem_a_id = pair.get("element_a_id", "")
    elem_b_id = pair.get("element_b_id", "")

    bridge_texts: List[str] = []
    seen: Set[str] = set()

    for eid in [elem_a_id, elem_b_id]:
        # Extract doc_id from element_id: "1709.02012_figure_4" → "1709.02012"
        parts = eid.rsplit("_", 2)
        if len(parts) < 3:
            continue
        eid_doc = parts[0]

        cache = _BRIDGE_TEXT_CACHE.get(eid_doc, {})
        if not cache:
            continue

        # Get LaTeX labels for this element_id from the mapping
        latex_labels = _ELEMENT_TO_LABELS.get(eid, [])
        for label in latex_labels:
            if label in cache:
                text = cache[label]
                if text not in seen:
                    seen.add(text)
                    bridge_texts.append(text)

    # Fallback: use bridge_contexts from topology (if stored by P0 enhancement)
    if not bridge_texts:
        for bc in pair.get("bridge_contexts", []):
            text = (bc.get("text", "") or "").strip()
            if text and text not in seen:
                seen.add(text)
                bridge_texts.append(text)

    return bridge_texts[:3]  # Cap at 3 bridge segments


# ──────────────────────────────────────────────────────────────
# P2: Bridge quality scoring — filter out unreadable bridges
# ──────────────────────────────────────────────────────────────

_BRIDGE_QUALITY_VERBS = re.compile(
    r'\b(show|demonstrate|indicate|present|report|achieve|compare|'
    r'observe|suggest|confirm|reveal|illustrate|summarize|highlight|'
    r'describe|measure|evaluate|compute|define|propose|introduce|'
    r'increase|decrease|improve|reduce|degrade|outperform)\w*\b',
    re.IGNORECASE,
)

_BRIDGE_BOILERPLATE = re.compile(
    r'\b(see also|cf\.|e\.g\.|i\.e\.|op\.?\s*cit\.|ibid|et al)\b',
    re.IGNORECASE,
)


# P2：基于动词密度/长度/公式比/引用标记给 bridge 打 0-1 分
def score_bridge_quality(bridge_text: str) -> float:
    """Score bridge paragraph quality for reasoning chain suitability.

    Returns 0.0–1.0 where:
      ≥ 0.5 = usable bridge (descriptive, has verbs, connects ideas)
      < 0.5 = unusable (too short, pure formula, boilerplate, citation-only)
    """
    if not bridge_text or len(bridge_text.strip()) < 30:
        return 0.0

    text = bridge_text.strip()
    score = 0.0

    # Length bonus (longer = more descriptive, capped at 200 chars)
    score += min(len(text) / 200, 0.3)

    # Semantic verb count (indicates explanatory prose)
    verb_matches = _BRIDGE_QUALITY_VERBS.findall(text)
    score += min(len(verb_matches) * 0.15, 0.35)

    # Penalty for boilerplate-heavy text
    boilerplate_hits = len(_BRIDGE_BOILERPLATE.findall(text))
    score -= boilerplate_hits * 0.1

    # Penalty for formula-dominated text (high ratio of special chars)
    alpha_ratio = sum(1 for c in text if c.isalpha()) / max(len(text), 1)
    if alpha_ratio < 0.4:
        score -= 0.2  # too much math notation, not readable

    # Bonus for cross-reference markers (indicates bridge function)
    ref_markers = len(re.findall(r'\[(?:fig|tab|eq|sec|cite)[:\w]*\]', text, re.I))
    score += min(ref_markers * 0.1, 0.2)

    return max(0.0, min(1.0, score))

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
7. OBSERVATION INJECTION (MANDATORY): Each query MUST describe in natural language what you concretely observe in one element (a trend, drop, gap, plateau, cluster, contrast). Use phrases like "the score stays flat while...", "the curve drops sharply above...", "X outperforms Y on minority but not aggregate". FORBIDDEN: meta-language ("figure"/"table"/"as shown in") and verbatim anchor coords. The visual_anchors field contains orientation coordinates for evaluation only — paraphrase the observation, never copy the anchor string.
8. The two queries must use DIFFERENT aspects of the data.
9. ENTITY AMNESTY: you MUST use exact paper terminology (method names, metric names,
   dataset names, variable names like "F1 score" or "p-value") when needed.
   Do NOT replace concrete terms with vague descriptors.
10. CAUSAL TOPOLOGY: each query must test one explicit relationship:
    (a) table value explains a visual trend, or (b) visual anomaly is supported/refuted by
    table values. Do NOT stitch unrelated facts.
11. Avoid weak templates: "Which component..." "How does X relate to Y..."
12. Answer must include a relationship connector: because / due to / consistent with /
    constrained by / compared with / whereas / despite / under.

## STYLE DIVERSITY — MANDATORY
- The 2 queries MUST use different opening bigrams (first two words).
- At least one query MUST NOT start with "Why" or "Under what".
- LENGTH MIX (STRUCTURAL, not just word count): queries[0] = SHORT compressed causal question (8-14 words, e.g. "The X drops after Y — does Z explain this?"); queries[1] = LONG I-notice + why/how + context (18-30 words, e.g. "X stays flat while Y rises across all conditions — given the constraint in the formula, what mechanism prevents X from scaling?"). Count words BEFORE finalizing.
- DO NOT use template shells: "Under what condition does..." or "Why is A different from B...".
- Do NOT create parallel dual asks using "..., and which ...". Each query should ask one causal/comparative question.
- Use natural research wording; explicit terms like "F1 score", "p-value", and "regularization strength" are allowed.

## BAD vs GOOD examples
BAD: "Did the red line peak at 90,000 and match the keyword set?" — yes/no, visual coords
BAD: "Which configuration best validates the theoretical optimum?" — banned verb "validates"
BAD: "How does the high-recall setting relate to the fairness tradeoff?" — banned "relate", vague
BAD: "How does fixing shared background nodes satisfy the identity requirement?" — abstract, no observation

GOOD (I-NOTICE + WHY, short): "The streaming volume drops after mid-December — does the sampling cap explain this?"
GOOD (I-NOTICE + WHY, long): "The minority-group gain stays flat above 1k samples while majority-group accuracy keeps rising — does the regularization term bound this asymmetry?"
GOOD (COUNTERINTUITIVE CONTRAST): "Why does direct discrimination approach zero while indirect pathway scores stay elevated despite the same fairness objective?"
GOOD (HOW-DISCREPANCY): "How does the drop in session frequency after later positions correspond to the retrieval-window setting that yields the highest scores?"
GOOD (WHY-INCONSISTENT): "Why is the ensemble gain larger on the minority subgroup than on the aggregate benchmark under the same evaluation regime?"

## Output format (JSON only):
CRITICAL: Generate EXACTLY 2 queries. queries[0] MUST be SHORT (8-14 words). queries[1] MUST be LONG (18-30 words). Count words carefully before writing.
{{
  "queries": [
    {{
      "query_length_bucket": "short",
      "reasoning_chain": "Max 3 sentences: (1) concrete figure observation, (2) concrete table metric/value pattern, (3) causal link that requires both",
      "query": "SHORT question (8-14 words ONLY), based on reasoning_chain, NO specific values",
      "answer": "factual answer citing specific values from BOTH elements, max 3 sentences, with connector",
      "query_type": "causal_explanation|discrepancy_analysis|hypothesis_verification",
      "required_evidence_spans": [
        {{"element_id": "{fig_id}", "span": "short extractive phrase from figure caption/content (semantic concept)", "evidence_type": "observation"}},
        {{"element_id": "{tbl_id}", "span": "short extractive phrase from table headers/content (semantic concept)", "evidence_type": "result"}}
      ],
      "visual_anchors": [
        {{"element_id": "{fig_id}", "anchor": "physical coords only: color/position/curve (evaluation only — NOT in query)"}},
        {{"element_id": "{tbl_id}", "anchor": "physical coords only: row/column/cell (evaluation only — NOT in query)"}}
      ],
      "text_evidence": "direct quote from context, min 40 chars"
    }},
    {{
      "query_length_bucket": "long",
      "reasoning_chain": "Max 3 sentences: (1) concrete figure observation, (2) concrete table metric/value pattern, (3) causal link that requires both",
      "query": "LONG question (18-30 words), based on reasoning_chain, NO specific values",
      "answer": "factual answer citing specific values from BOTH elements, max 3 sentences, with connector",
      "query_type": "causal_explanation|discrepancy_analysis|hypothesis_verification",
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
7. OBSERVATION INJECTION (MANDATORY): Each query MUST describe in natural language what you concretely observe in one element (a trend, drop, gap, plateau, cluster, contrast). Use phrases like "the score stays flat while...", "the curve drops sharply above...", "X outperforms Y on minority but not aggregate". FORBIDDEN: meta-language ("figure"/"table"/"as shown in") and verbatim anchor coords. The visual_anchors field contains orientation coordinates for evaluation only — paraphrase the observation, never copy the anchor string.
8. ENTITY AMNESTY: you MUST use exact paper terminology when needed.
9. CAUSAL TOPOLOGY: the query must require a chain:
   figure observation -> bridge mechanism -> table metric (or the reverse), not parallel lookup.
10. Avoid weak templates: "Which component..." "How does X relate to Y..."
11. Answer must include a relationship connector: because / due to / consistent with /
    constrained by / compared with / whereas / despite / under.

## STYLE DIVERSITY — MANDATORY
- The 2 queries MUST use different opening bigrams (first two words).
- At least one query MUST NOT start with "Why" or "Under what".
- LENGTH MIX (STRUCTURAL, not just word count): queries[0] = SHORT compressed causal question (8-14 words, e.g. "The X drops after Y — does Z explain this?"); queries[1] = LONG I-notice + why/how + context (18-30 words, e.g. "X stays flat while Y rises across all conditions — given the constraint in the formula, what mechanism prevents X from scaling?"). Count words BEFORE finalizing.
- DO NOT use template shells: "Under what condition does..." or "Why is A different from B...".
- Do NOT create parallel dual asks using "..., and which ...". Each query should ask one causal/comparative question.
- Use natural research wording; explicit terms like "F1 score", "p-value", and "regularization strength" are allowed.

## Output format (JSON only):
CRITICAL: Generate EXACTLY 2 queries. queries[0] MUST be SHORT (8-14 words). queries[1] MUST be LONG (18-30 words). Count words carefully before writing.
{{
  "queries": [
    {{
      "query_length_bucket": "short",
      "reasoning_chain": "Max 3 sentences: endpoint A observation -> bridge property -> endpoint B metric/conclusion",
      "query": "SHORT chain-reasoning question (8-14 words ONLY), NO specific values",
      "answer": "factual answer citing values from both elements, max 3 sentences, with connector",
      "query_type": "causal_explanation|discrepancy_analysis|hypothesis_verification",
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
    }},
    {{
      "query_length_bucket": "long",
      "reasoning_chain": "Max 3 sentences: endpoint A observation -> bridge property -> endpoint B metric/conclusion",
      "query": "LONG chain-reasoning question (18-30 words), NO specific values",
      "answer": "factual answer citing values from both elements, max 3 sentences, with connector",
      "query_type": "causal_explanation|discrepancy_analysis|hypothesis_verification",
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

{architecture_guidance}

## ENTITY AMNESTY
Use exact paper terminology (method names, metric names, variable names) when needed.
Avoid vague substitutions like "the best-performing method" when a concrete name exists.

## STRICT RULES
1. Query MUST be UNANSWERABLE without BOTH the figure AND the formula. Test: if you remove the figure, can you still answer from the formula alone? If yes — REJECT and rewrite.
2. NEVER start with Do/Does/Did/Is/Are/Can/Has/Will/Would; NO yes/no questions.
3. NEVER start with "Given that" or "What causes" (avoid template collapse).
4. NEVER copy raw LaTeX strings or raw variable symbols into the query.
5. You MAY use at most ONE anchor value in the query only if needed to preserve directionality.
6. NEVER use meta-language: "equation", "formula", "figure", "as shown in", "diagram", "architecture".
7. Max 30 words per query.
8. OBSERVATION INJECTION (MANDATORY): Each query MUST describe in natural language what you concretely observe in one element (a trend, drop, gap, plateau, cluster, contrast). Use phrases like "the score stays flat while...", "the curve drops sharply above...", "X outperforms Y on minority but not aggregate". FORBIDDEN: meta-language ("figure"/"table"/"as shown in") and verbatim anchor coords. The visual_anchors field contains orientation coordinates for evaluation only — paraphrase the observation, never copy the anchor string.
9. CAUSAL TOPOLOGY: query must connect one concrete visual phenomenon and one concrete
   mathematical mechanism; no unrelated stitching.
10. Avoid weak templates: "Which component..." "How does X relate to Y..." "What role does..."
11. Answer must include a relationship connector (because / due to / consistent with /
    constrained by / compared with / whereas / despite / under).
12. IRON RULE (symbolic grounding): if Element B is mathematical, the answer MUST explicitly quote at least one specific variable/function/constraint term from Element B (e.g., lambda, theta, f(A,C)) and explain how it maps to the observed visual topology in Element A. Avoid generic phrases like "the mathematical structure".

## SELF-CHECK before outputting each query

Ask yourself:
1. Does the answer cite a SPECIFIC, CONCRETE observation from the figure (trend/structure/comparison)?
2. Does the answer cite a SPECIFIC mathematical mechanism from the formula (constraint/bound/objective)?
3. If I removed either piece of evidence, would the answer become incomplete or wrong?

If any answer is NO — rewrite.

## STYLE DIVERSITY — MANDATORY
- The 2 queries MUST use different opening bigrams (first two words).
- At least one query MUST NOT start with "Why" or "Under what".
- LENGTH MIX (STRUCTURAL, not just word count): queries[0] = SHORT compressed causal question (8-14 words, e.g. "The X drops after Y — does Z explain this?"); queries[1] = LONG I-notice + why/how + context (18-30 words, e.g. "X stays flat while Y rises across all conditions — given the constraint in the formula, what mechanism prevents X from scaling?"). Count words BEFORE finalizing.
- DO NOT use template shells: "Under what condition does..." or "Why is A different from B...".
- Do NOT create parallel dual asks using "..., and which ...". Each query should ask one causal/comparative question.
- Use natural research wording; explicit terms like "F1 score", "p-value", and "regularization strength" are allowed.

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
CRITICAL: Generate EXACTLY 2 queries. queries[0] MUST be SHORT (8-14 words). queries[1] MUST be LONG (18-30 words). Count words carefully before writing.
{{
  "queries": [
    {{
      "query_length_bucket": "short",
      "reasoning_chain": "Max 3 sentences: concrete figure observation -> concrete formula mechanism -> explicit causality between them",
      "query": "SHORT question (8-14 words ONLY), based on reasoning_chain, NO LaTeX/values/letters/meta-language",
      "answer": "max 4 sentences with connector. Must cite both halves explicitly.",
      "answer_figure_evidence": "1-2 sentences: what specific observation from the figure is cited (trend/structure/comparison). Start with the concrete observable.",
      "answer_formula_evidence": "1-2 sentences: what specific mathematical mechanism from the formula is cited (constraint/bound/objective). Start with the mechanism.",
      "query_type": "causal_explanation|discrepancy_analysis|hypothesis_verification",
      "required_evidence_spans": [
        {{"element_id": "{fig_id}", "span": "specific observable feature: name the trend/structure/comparison concretely", "evidence_type": "observation"}},
        {{"element_id": "{formula_id}", "span": "specific mathematical mechanism: name the constraint/term/property (NOT generic 'the formula')", "evidence_type": "mechanism"}}
      ],
      "visual_anchors": [
        {{"element_id": "{fig_id}", "anchor": "physical coords only: color/position/curve/branch/edge — NOT in query"}},
        {{"element_id": "{formula_id}", "anchor": "specific term or variable — NOT standalone letter in query"}}
      ],
      "text_evidence": "direct quote from context, min 40 chars"
    }},
    {{
      "query_length_bucket": "long",
      "reasoning_chain": "Max 3 sentences: concrete figure observation -> concrete formula mechanism -> explicit causality between them",
      "query": "LONG question (18-30 words), based on reasoning_chain, NO LaTeX/values/letters/meta-language",
      "answer": "max 4 sentences with connector. Must cite both halves explicitly.",
      "answer_figure_evidence": "1-2 sentences: what specific observation from the figure is cited (trend/structure/comparison). Start with the concrete observable.",
      "answer_formula_evidence": "1-2 sentences: what specific mathematical mechanism from the formula is cited (constraint/bound/objective). Start with the mechanism.",
      "query_type": "causal_explanation|discrepancy_analysis|hypothesis_verification",
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
8. OBSERVATION INJECTION (MANDATORY): Each query MUST describe in natural language what you concretely observe in one element (a trend, drop, gap, plateau, cluster, contrast). Use phrases like "the score stays flat while...", "the curve drops sharply above...", "X outperforms Y on minority but not aggregate". FORBIDDEN: meta-language ("figure"/"table"/"as shown in") and verbatim anchor coords. The visual_anchors field contains orientation coordinates for evaluation only — paraphrase the observation, never copy the anchor string.
9. ENTITY AMNESTY: you MUST use exact paper terminology (method/metric/variable names)
   when needed; avoid vague substitutions.
10. CAUSAL TOPOLOGY: question must connect one concrete mathematical mechanism and one
    concrete tabular result through a causal/comparative claim.
11. Avoid weak templates: "Which component..." "How does X relate to Y..."
12. Answer must include a relationship connector: because / due to / consistent with /
    constrained by / compared with / whereas / despite / under.

## STYLE DIVERSITY — MANDATORY
- The 2 queries MUST use different opening bigrams (first two words).
- At least one query MUST NOT start with "Why" or "Under what".
- LENGTH MIX (STRUCTURAL, not just word count): queries[0] = SHORT compressed causal question (8-14 words, e.g. "The X drops after Y — does Z explain this?"); queries[1] = LONG I-notice + why/how + context (18-30 words, e.g. "X stays flat while Y rises across all conditions — given the constraint in the formula, what mechanism prevents X from scaling?"). Count words BEFORE finalizing.
- DO NOT use template shells: "Under what condition does..." or "Why is A different from B...".
- Do NOT create parallel dual asks using "..., and which ...". Each query should ask one causal/comparative question.
- Use natural research wording; explicit terms like "F1 score", "p-value", and "regularization strength" are allowed.

## BAD vs GOOD examples
BAD: "Does beta=0.3 in row 4 satisfy Eq. 2?" — yes/no, numbers, meta-language
BAD: "What does N represent?" — standalone letter, vague
BAD: "Which configuration best instantiates the theoretical threshold?" — banned "instantiate"

GOOD (HOW-DISCREPANCY): "How does the smoothing-driven sparsity regime correspond to the steeper accuracy drop under low-resource conditions?"
GOOD (WHY-INCONSISTENT): "Why is the highest-regularization setting better than the unregularized baseline in out-of-domain evaluation despite overfitting risk?"

## Output format (JSON only):
CRITICAL: Generate EXACTLY 2 queries. queries[0] MUST be SHORT (8-14 words). queries[1] MUST be LONG (18-30 words). Count words carefully before writing.
{{
  "queries": [
    {{
      "query_length_bucket": "short",
      "reasoning_chain": "Max 3 sentences: formula mechanism -> table metric pattern -> explicit causality/comparison",
      "query": "SHORT formula-data question (8-14 words ONLY), based on reasoning_chain, NO LaTeX/values/letters",
      "answer": "factual answer with specific values from both, max 3 sentences, with connector",
      "query_type": "causal_explanation|discrepancy_analysis|hypothesis_verification",
      "required_evidence_spans": [
        {{"element_id": "{formula_id}", "span": "short extractive phrase describing the formula's role or constraint (NOT raw LaTeX)", "evidence_type": "constraint"}},
        {{"element_id": "{tbl_id}", "span": "short extractive phrase from table headers/content (semantic concept)", "evidence_type": "result"}}
      ],
      "visual_anchors": [
        {{"element_id": "{formula_id}", "anchor": "specific term or variable — NOT standalone letter in query"}},
        {{"element_id": "{tbl_id}", "anchor": "physical coords only: row/column/cell — NOT in query"}}
      ],
      "text_evidence": "direct quote from context, min 40 chars"
    }},
    {{
      "query_length_bucket": "long",
      "reasoning_chain": "Max 3 sentences: formula mechanism -> table metric pattern -> explicit causality/comparison",
      "query": "LONG formula-data question (18-30 words), based on reasoning_chain, NO LaTeX/values/letters",
      "answer": "factual answer with specific values from both, max 3 sentences, with connector",
      "query_type": "causal_explanation|discrepancy_analysis|hypothesis_verification",
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
# Level 3: 3-step reasoning chain prompt (M4 Schema 1)
# ──────────────────────────────────────────────────────────────

PROMPT_3STEP_REASONING_CHAIN = """You are building a benchmark that tests whether a system can follow a multi-step reasoning chain across different evidence sources. The chain MUST have exactly 3 steps, each grounded in a different piece of evidence.

## Graph-Grounded Evidence Path

The document graph connects these 3 nodes via explicit \\ref{{}} links in the LaTeX source. The bridge paragraph is the ACTUAL text the authors wrote to connect the two endpoint elements.

### Node 1 (Premise): {elem_a_type} ({elem_a_id})
Caption: {elem_a_caption}
Context: {elem_a_context}
{elem_a_image_note}

### Node 2 (Bridge): paragraph — author's own connecting text
{bridge_text}
Bridge quality: {bridge_quality_label}

### Node 3 (Conclusion): {elem_b_type} ({elem_b_id})
Caption: {elem_b_caption}
Context: {elem_b_context}
{elem_b_image_note}

### Graph path: {graph_path_description}

## YOUR TASK

Generate 1 query that REQUIRES all 3 evidence nodes IN SEQUENCE. The bridge paragraph is the key — it explains WHY Node 1's observation leads to Node 3's conclusion.

### SERIAL CHAIN PATTERN (REQUIRED)
Node 1 observation → Bridge paragraph mechanism/explanation → Node 3 confirmation/outcome

Example of VALID serial chain:
  "Node 1 shows accuracy drops for minority groups"
  → Bridge: "As shown in [fig:3], the accuracy drops significantly for minority groups. The exact numbers are in [tab:2]."
  → "Node 3 provides per-group accuracy values confirming the 12% gap"
  Query: "What numerical precision confirms the minority group accuracy drop visible in the performance trend?"

### PARALLEL IS FORBIDDEN
"Node 1 says A, Node 3 says B, therefore A+B" — this is just two independent lookups. The bridge MUST provide a causal/explanatory link, not just co-occurrence.

### STEP-DELETION TEST (self-check before outputting)
For each step, ask: "If I remove THIS step's evidence, can I still derive the answer?"
- If YES for any step → your chain is too weak, rewrite it
- If NO for all 3 steps → your chain is valid

### BRIDGE GROUNDING RULE
Your answer MUST quote or paraphrase specific content from the bridge paragraph text above. If the bridge text says "X leads to Y", your answer must use that causal link. Do NOT invent connections not present in the bridge.

## STRICT RULES
1. Query MUST require ALL 3 evidence nodes. Removing ANY node makes the answer underivable.
2. NEVER start with Do/Does/Did/Is/Are/Can/Has/Will/Would; NO yes/no questions.
3. NEVER use meta-language: "figure", "table", "equation", "the text", "according to", "as shown in".
4. Max 30 words for the query. Answer max 4 sentences.
5. ENTITY AMNESTY: use exact paper terminology (method names, metrics, variables).
6. Each reasoning_step MUST have a DIFFERENT evidence_type from: observation, attribution, explanation, verification, prediction.
7. The role arc MUST follow: premise → intermediate → conclusion.
8. Visual anchors MUST specify physical location: row/column for tables, axis region/color/marker for figures, specific variable/term for formulas. Generic anchors like "the table" or "the figure" will be rejected.

## Output format (JSON only):
{{
  "queries": [
    {{
      "query": "A question requiring serial 3-step reasoning (max 30 words)",
      "answer": "Answer citing specific evidence from all 3 nodes, referencing bridge text explicitly, max 4 sentences",
      "query_type": "causal_chain|mechanism_trace|conditional_prediction",
      "reasoning_steps": [
        {{
          "step_id": 1,
          "evidence_element_id": "{elem_a_id}",
          "evidence_type": "observation",
          "evidence_span": "extractive phrase from Node 1",
          "reasoning_role": "premise",
          "depends_on_steps": [],
          "produces_claim": "What this step establishes (1 sentence)"
        }},
        {{
          "step_id": 2,
          "evidence_element_id": "bridge_paragraph",
          "evidence_type": "attribution",
          "evidence_span": "extractive phrase copied from the bridge paragraph text above",
          "reasoning_role": "intermediate",
          "depends_on_steps": [1],
          "produces_claim": "How the bridge connects step 1's observation to a mechanism (1 sentence)"
        }},
        {{
          "step_id": 3,
          "evidence_element_id": "{elem_b_id}",
          "evidence_type": "explanation",
          "evidence_span": "extractive phrase from Node 3",
          "reasoning_role": "conclusion",
          "depends_on_steps": [1, 2],
          "produces_claim": "What final conclusion requires both prior steps (1 sentence)"
        }}
      ],
      "required_evidence_spans": [
        {{"element_id": "{elem_a_id}", "span": "extractive phrase from Node 1", "evidence_type": "observation"}},
        {{"element_id": "bridge_paragraph", "span": "extractive phrase from bridge paragraph", "evidence_type": "attribution", "content": "verbatim or close-paraphrase from bridge text above, min 40 chars"}},
        {{"element_id": "{elem_b_id}", "span": "extractive phrase from Node 3", "evidence_type": "explanation"}}
      ],
      "visual_anchors": [
        {{"element_id": "{elem_a_id}", "anchor": "specific physical location: row X col Y / axis region / marker color / variable name"}},
        {{"element_id": "{elem_b_id}", "anchor": "specific physical location: row X col Y / axis region / marker color / variable name"}}
      ],
      "text_evidence": "direct quote from bridge paragraph context, min 40 chars"
    }}
  ]
}}"""


# ──────────────────────────────────────────────────────────────
# Real-user query style prompt templates (B1)
# ──────────────────────────────────────────────────────────────
# These templates generate naturally phrased queries as a curious
# reader / practitioner would ask, rather than the structured
# PhD dual-evidence academic style.  Activated via --query-style
# real_user or mixed.  Both elements are still required to answer.

PROMPT_REAL_USER_FACTUAL = """A researcher is skimming an AI/ML paper and wants quick factual clarity.
Generate 2 natural lookup queries that require BOTH the figure and the table/formula below.

## Element A ({elem_a_id}, {elem_a_type})
Caption: {elem_a_caption}
Context: {elem_a_context}

## Element B ({elem_b_id}, {elem_b_type})
Caption: {elem_b_caption}
Context: {elem_b_context}

## Connection
{edge_context}
{latex_bridge}

## YOUR TASK
Write 2 questions a researcher would genuinely ask while reading this paper.
Style: factual_lookup — "What is X?", "What value does Y take when Z?", "What does X show about Y?"

## RULES
1. Both elements must be needed to fully answer the question.
2. Questions must be in natural English — no academic formalism.
3. NEVER use meta-language: "figure", "table", "equation", "as shown in".
4. Max 25 words per question. Answer max 2 sentences.
5. Do NOT copy raw LaTeX, variable names as standalone letters, or numeric values into the question.
6. Each question must have a different opening word.

## Output (JSON only):
{{
  "queries": [
    {{
      "query": "natural factual question requiring both elements (max 25 words)",
      "answer": "direct factual answer (max 2 sentences, cite values from both elements)",
      "query_type": "factual_lookup",
      "required_evidence_spans": [
        {{"element_id": "{elem_a_id}", "span": "key concept or value from element A", "evidence_type": "data"}},
        {{"element_id": "{elem_b_id}", "span": "key concept or value from element B", "evidence_type": "data"}}
      ],
      "visual_anchors": [
        {{"element_id": "{elem_a_id}", "anchor": "specific region or term in element A"}},
        {{"element_id": "{elem_b_id}", "anchor": "specific region or term in element B"}}
      ],
      "text_evidence": "direct quote from context supporting the answer (min 30 chars)"
    }},
    {{
      "query": "second natural factual question, different opening word (max 25 words)",
      "answer": "direct factual answer (max 2 sentences)",
      "query_type": "factual_lookup",
      "required_evidence_spans": [
        {{"element_id": "{elem_a_id}", "span": "key concept from element A", "evidence_type": "data"}},
        {{"element_id": "{elem_b_id}", "span": "key concept from element B", "evidence_type": "data"}}
      ],
      "visual_anchors": [
        {{"element_id": "{elem_a_id}", "anchor": "specific region or term in element A"}},
        {{"element_id": "{elem_b_id}", "anchor": "specific region or term in element B"}}
      ],
      "text_evidence": "direct quote from context (min 30 chars)"
    }}
  ]
}}"""

PROMPT_REAL_USER_SUMMARY = """A reader wants a concise synthesis of what two pieces of evidence mean together.
Generate 2 natural summary questions requiring BOTH elements below.

## Element A ({elem_a_id}, {elem_a_type})
Caption: {elem_a_caption}
Context: {elem_a_context}

## Element B ({elem_b_id}, {elem_b_type})
Caption: {elem_b_caption}
Context: {elem_b_context}

## Connection
{edge_context}
{latex_bridge}

## YOUR TASK
Write 2 questions asking for a combined summary of both elements.
Style: summary — "Summarize how...", "What does the combination of X and Y suggest?", "What can we conclude from X and Y together?"

## RULES
1. Both elements must be needed — a summary of only one element is wrong.
2. Natural English. No academic formalism.
3. NEVER use meta-language: "figure", "table", "equation", "as shown in".
4. Max 25 words per question. Answer max 3 sentences.
5. Each question must have a different opening word.
6. Do NOT use yes/no questions.

## Output (JSON only):
{{
  "queries": [
    {{
      "query": "natural summary question requiring both elements (max 25 words)",
      "answer": "synthesized answer covering both elements (max 3 sentences)",
      "query_type": "summary",
      "required_evidence_spans": [
        {{"element_id": "{elem_a_id}", "span": "key insight from element A", "evidence_type": "observation"}},
        {{"element_id": "{elem_b_id}", "span": "key insight from element B", "evidence_type": "observation"}}
      ],
      "visual_anchors": [
        {{"element_id": "{elem_a_id}", "anchor": "main visual or conceptual anchor in element A"}},
        {{"element_id": "{elem_b_id}", "anchor": "main visual or conceptual anchor in element B"}}
      ],
      "text_evidence": "relevant context quote supporting the synthesis (min 30 chars)"
    }},
    {{
      "query": "second summary question, different opening word (max 25 words)",
      "answer": "synthesized answer (max 3 sentences)",
      "query_type": "summary",
      "required_evidence_spans": [
        {{"element_id": "{elem_a_id}", "span": "key insight from element A", "evidence_type": "observation"}},
        {{"element_id": "{elem_b_id}", "span": "key insight from element B", "evidence_type": "observation"}}
      ],
      "visual_anchors": [
        {{"element_id": "{elem_a_id}", "anchor": "main anchor in element A"}},
        {{"element_id": "{elem_b_id}", "anchor": "main anchor in element B"}}
      ],
      "text_evidence": "context quote (min 30 chars)"
    }}
  ]
}}"""

PROMPT_REAL_USER_COMPARISON = """A reader wants to understand how two results or components compare.
Generate 2 natural comparison questions requiring BOTH elements below.

## Element A ({elem_a_id}, {elem_a_type})
Caption: {elem_a_caption}
Context: {elem_a_context}

## Element B ({elem_b_id}, {elem_b_type})
Caption: {elem_b_caption}
Context: {elem_b_context}

## Connection
{edge_context}
{latex_bridge}

## YOUR TASK
Write 2 questions comparing something across the two elements.
Style: comparison — "How does X in [element A] compare to Y in [element B]?", "Which is better at..., and why?", "What differences exist between X and Y?"

## RULES
1. The comparison must be grounded in both elements — not inferable from one alone.
2. Natural English, conversational tone.
3. NEVER use meta-language: "figure", "table", "equation".
4. Max 25 words per question. Answer max 3 sentences.
5. Do NOT copy raw LaTeX or standalone variable letters.
6. Each question must have a different opening word.
7. Do NOT use yes/no questions.

## Output (JSON only):
{{
  "queries": [
    {{
      "query": "natural comparison question requiring both elements (max 25 words)",
      "answer": "comparative answer with specific values from both (max 3 sentences)",
      "query_type": "comparison",
      "required_evidence_spans": [
        {{"element_id": "{elem_a_id}", "span": "compared aspect from element A", "evidence_type": "observation"}},
        {{"element_id": "{elem_b_id}", "span": "compared aspect from element B", "evidence_type": "observation"}}
      ],
      "visual_anchors": [
        {{"element_id": "{elem_a_id}", "anchor": "compared feature in element A"}},
        {{"element_id": "{elem_b_id}", "anchor": "compared feature in element B"}}
      ],
      "text_evidence": "context quote supporting the comparison (min 30 chars)"
    }},
    {{
      "query": "second comparison question, different opening word (max 25 words)",
      "answer": "comparative answer (max 3 sentences)",
      "query_type": "comparison",
      "required_evidence_spans": [
        {{"element_id": "{elem_a_id}", "span": "compared aspect from element A", "evidence_type": "observation"}},
        {{"element_id": "{elem_b_id}", "span": "compared aspect from element B", "evidence_type": "observation"}}
      ],
      "visual_anchors": [
        {{"element_id": "{elem_a_id}", "anchor": "compared feature in element A"}},
        {{"element_id": "{elem_b_id}", "anchor": "compared feature in element B"}}
      ],
      "text_evidence": "context quote (min 30 chars)"
    }}
  ]
}}"""

PROMPT_REAL_USER_HOW_WORKS = """A practitioner wants to understand the mechanism connecting two results.
Generate 2 natural how/why questions requiring BOTH elements below.

## Element A ({elem_a_id}, {elem_a_type})
Caption: {elem_a_caption}
Context: {elem_a_context}

## Element B ({elem_b_id}, {elem_b_type})
Caption: {elem_b_caption}
Context: {elem_b_context}

## Connection
{edge_context}
{latex_bridge}

## YOUR TASK
Write 2 mechanistic questions about how/why the two pieces of evidence are related.
Style: how_works — "How does X lead to Y?", "Why does X affect Y?", "What mechanism connects X and Y?"

## RULES
1. Both elements must be needed to explain the mechanism.
2. Natural English — not overly academic.
3. NEVER use meta-language: "figure", "table", "equation", "as shown in".
4. Max 25 words per question. Answer max 3 sentences.
5. Do NOT copy raw LaTeX or standalone variable letters.
6. Each question must have a different opening word.
7. Do NOT use yes/no questions.

## Output (JSON only):
{{
  "queries": [
    {{
      "query": "natural how/why question connecting both elements (max 25 words)",
      "answer": "mechanistic answer drawing on both elements (max 3 sentences)",
      "query_type": "how_works",
      "required_evidence_spans": [
        {{"element_id": "{elem_a_id}", "span": "mechanism aspect from element A", "evidence_type": "mechanism"}},
        {{"element_id": "{elem_b_id}", "span": "mechanism aspect from element B", "evidence_type": "mechanism"}}
      ],
      "visual_anchors": [
        {{"element_id": "{elem_a_id}", "anchor": "key mechanism feature in element A"}},
        {{"element_id": "{elem_b_id}", "anchor": "key mechanism feature in element B"}}
      ],
      "text_evidence": "context quote supporting the mechanism (min 30 chars)"
    }},
    {{
      "query": "second how/why question, different opening word (max 25 words)",
      "answer": "mechanistic answer (max 3 sentences)",
      "query_type": "how_works",
      "required_evidence_spans": [
        {{"element_id": "{elem_a_id}", "span": "mechanism aspect from element A", "evidence_type": "mechanism"}},
        {{"element_id": "{elem_b_id}", "span": "mechanism aspect from element B", "evidence_type": "mechanism"}}
      ],
      "visual_anchors": [
        {{"element_id": "{elem_a_id}", "anchor": "key feature in element A"}},
        {{"element_id": "{elem_b_id}", "anchor": "key feature in element B"}}
      ],
      "text_evidence": "context quote (min 30 chars)"
    }}
  ]
}}"""

PROMPT_REAL_USER_WHAT_IF = """A researcher wants to reason about hypothetical changes to a method or result.
Generate 2 natural what-if questions requiring BOTH elements below.

## Element A ({elem_a_id}, {elem_a_type})
Caption: {elem_a_caption}
Context: {elem_a_context}

## Element B ({elem_b_id}, {elem_b_type})
Caption: {elem_b_caption}
Context: {elem_b_context}

## Connection
{edge_context}
{latex_bridge}

## YOUR TASK
Write 2 hypothetical / counterfactual questions that require both elements to reason about.
Style: what_if — "What would happen to X if Y changed?", "If X were different, how would Y be affected?", "What would change in [result] if [condition] were altered?"

## RULES
1. Both elements must be needed to reason about the hypothetical.
2. Natural English — conversational speculation.
3. NEVER use meta-language: "figure", "table", "equation".
4. Max 25 words per question. Answer max 3 sentences.
5. Do NOT copy raw LaTeX or standalone variable letters.
6. Each question must have a different opening word.
7. Do NOT use yes/no questions.

## Output (JSON only):
{{
  "queries": [
    {{
      "query": "natural what-if question connecting both elements (max 25 words)",
      "answer": "reasoned counterfactual answer drawing on both elements (max 3 sentences)",
      "query_type": "what_if",
      "required_evidence_spans": [
        {{"element_id": "{elem_a_id}", "span": "condition or variable from element A", "evidence_type": "condition"}},
        {{"element_id": "{elem_b_id}", "span": "affected result from element B", "evidence_type": "result"}}
      ],
      "visual_anchors": [
        {{"element_id": "{elem_a_id}", "anchor": "hypothetical feature in element A"}},
        {{"element_id": "{elem_b_id}", "anchor": "affected feature in element B"}}
      ],
      "text_evidence": "context quote supporting the reasoning (min 30 chars)"
    }},
    {{
      "query": "second what-if question, different opening word (max 25 words)",
      "answer": "reasoned counterfactual answer (max 3 sentences)",
      "query_type": "what_if",
      "required_evidence_spans": [
        {{"element_id": "{elem_a_id}", "span": "condition from element A", "evidence_type": "condition"}},
        {{"element_id": "{elem_b_id}", "span": "affected result from element B", "evidence_type": "result"}}
      ],
      "visual_anchors": [
        {{"element_id": "{elem_a_id}", "anchor": "feature in element A"}},
        {{"element_id": "{elem_b_id}", "anchor": "feature in element B"}}
      ],
      "text_evidence": "context quote (min 30 chars)"
    }}
  ]
}}"""

# Mapping used by select_template() for real_user style
_REAL_USER_TEMPLATES: Dict[str, str] = {
    "factual_lookup": PROMPT_REAL_USER_FACTUAL,
    "summary": PROMPT_REAL_USER_SUMMARY,
    "comparison": PROMPT_REAL_USER_COMPARISON,
    "how_works": PROMPT_REAL_USER_HOW_WORKS,
    "what_if": PROMPT_REAL_USER_WHAT_IF,
}
_REAL_USER_STYLE_CYCLE = list(_REAL_USER_TEMPLATES.keys())


# ──────────────────────────────────────────────────────────────
# PersonaHub integration: diverse reader personas for query generation
#
# Based on PersonaHub methodology (Ge et al., 2024):
#   "Scaling Synthetic Data Creation with 1,000,000,000 Personas"
#   arXiv:2406.20094 | Dataset: proj-persona/PersonaHub (HuggingFace)
#   GitHub: tencent-ailab/persona-hub
#
# We curate a domain-adapted subset of academic personas following
# PersonaHub's Text-to-Persona format.  Each persona is a short
# natural-language description injected as prompt prefix via {persona}.
# When HuggingFace access is available, these can be augmented with
# personas sampled directly from PersonaHub's persona.jsonl (200K+).
# ──────────────────────────────────────────────────────────────

_PERSONAHUB_PERSONAS: Optional[List[Dict[str, str]]] = None  # lazy-loaded cache


# 懒加载 PersonaHub 人设库（50 类学术读者），缺文件时退回 5 条内置人设
def _load_personahub_personas(path: Optional[str] = None) -> List[Dict[str, str]]:
    """Load PersonaHub-format personas from JSON file.

    Returns a list of dicts with at least 'id' and 'persona' keys.
    Falls back to a minimal built-in set if the file is unavailable.
    """
    global _PERSONAHUB_PERSONAS
    if _PERSONAHUB_PERSONAS is not None:
        return _PERSONAHUB_PERSONAS

    if path is None:
        path = str(PROJECT_ROOT / "data" / "personahub_academic_personas.json")

    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        personas = data.get("personas", [])
        if personas:
            _PERSONAHUB_PERSONAS = personas
            return _PERSONAHUB_PERSONAS
    except (FileNotFoundError, json.JSONDecodeError, KeyError) as e:
        print(f"  WARNING: Failed to load PersonaHub personas from {path}: {e}")
        print("  Falling back to built-in minimal persona set.")

    # Minimal fallback (should not be needed in normal operation)
    _PERSONAHUB_PERSONAS = [
        {"id": "phd_ml_fairness", "persona": "A third-year PhD student researching algorithmic fairness who is preparing for their qualifying exam and needs to deeply understand how different fairness metrics interact with model performance across demographic groups."},
        {"id": "ml_engineer_production", "persona": "A machine learning engineer at a mid-size tech company evaluating whether to adopt a new method in production, primarily concerned with compute cost, data requirements, latency constraints, and whether reported gains hold on real-world benchmarks."},
        {"id": "reviewer_harsh", "persona": "An experienced conference reviewer known for thorough and critical reviews, who systematically checks whether baselines are fair, experiments are reproducible, and limitations are honestly disclosed."},
        {"id": "busy_professor", "persona": "A tenured professor with heavy administrative duties who only has 15 minutes to skim a paper before a committee meeting and needs to quickly grasp the main contribution and its significance."},
        {"id": "undergrad_first_paper", "persona": "An undergraduate computer science student reading their first machine learning research paper, struggling with mathematical notation and needing concrete examples to understand abstract concepts."},
    ]
    return _PERSONAHUB_PERSONAS


# 按 pair_id 的 md5 稳定哈希分配人设，保证跨机器可复现
def resolve_persona(pair_id: str, persona_file: Optional[str] = None) -> str:
    """Deterministically assign a PersonaHub persona to a pair via stable hash.

    Uses the PersonaHub-format persona list (loaded from JSON file).
    Returns the persona description text (not the id).
    """
    personas = _load_personahub_personas(persona_file)
    if not pair_id or not personas:
        return personas[0]["persona"] if personas else ""
    stable = int(hashlib.md5(pair_id.encode("utf-8")).hexdigest()[:8], 16)
    idx = stable % len(personas)
    return personas[idx]["persona"]


# 同上，返回短标签用于日志 / 切片分析
def resolve_persona_id(pair_id: str, persona_file: Optional[str] = None) -> str:
    """Return the persona id (short label) for logging/slicing."""
    personas = _load_personahub_personas(persona_file)
    if not pair_id or not personas:
        return personas[0]["id"] if personas else "unknown"
    stable = int(hashlib.md5(pair_id.encode("utf-8")).hexdigest()[:8], 16)
    idx = stable % len(personas)
    return personas[idx]["id"]


# 用人设描述替换 prompt 首句 "You are a …"，避免双重角色定义
def inject_persona_prefix(prompt: str, persona: str) -> str:
    """Prepend a PersonaHub persona description to an existing prompt template.

    The persona replaces the first "You are a ..." sentence in the prompt
    so we don't double-define the role.  Falls back to simple prepend.
    """
    if not persona:
        return prompt
    # Replace first "You are a ..." sentence with the persona
    first_sentence_pat = re.compile(r"^(You are [^.]+\.)", re.MULTILINE)
    if first_sentence_pat.search(prompt):
        return first_sentence_pat.sub(persona, prompt, count=1)
    return persona + "\n\n" + prompt


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

ANCHOR_LEAK_THRESHOLD = 0.20
ANSWER_BALANCE_THRESHOLD = 0.20
MIN_OVERLAP_BY_TYPE = {
    "figure": 1,
    "table": 2,
    "formula": 2,
}
SHORT_QUERY_MIN_WORDS = 8
SHORT_QUERY_MAX_WORDS = 14
LONG_QUERY_MIN_WORDS = 18
MAX_QUERY_WORDS = 30
TEXT_EVIDENCE_OVERLAP_WARN_THRESHOLD = 0.4

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

TEMPLATE_COLLAPSE_PATTERNS = (
    r"^under\s+what\b",
    r"^why\s+is\b.+\bdifferent from\b",
)

ENTITY_AMNESTY_TERMS = {
    "f1", "auc", "rmse", "accuracy", "precision", "recall", "pvalue", "p-value",
    "lambda", "alpha", "beta", "baseline", "attention", "adversary", "adversarial",
    "fairness", "bias", "parity", "statistical", "significance", "classification",
    "error", "score", "scores", "metric", "metrics", "dataset", "datasets",
}

STRUCTURAL_CONCEPT_TERMS = {
    "separation", "cluster", "topology", "network", "node", "nodes",
    "graph", "curve", "curves", "heatmap", "bar", "bars", "layout",
}

METRIC_CONCEPT_TERMS = {
    "performance", "score", "scores", "accuracy", "error", "errors",
    "precision", "recall", "auc", "f1", "rmse", "metric", "metrics",
    "rate", "rates", "p-value", "pvalue", "statistical",
}

ARCHITECTURE_KEYWORDS = {
    "architecture", "framework", "pipeline", "overview", "module", "modules",
    "component", "components", "encoder", "decoder", "branch", "branches",
    "backbone", "head", "heads", "block", "blocks", "graph", "topology",
    "fusion", "attention", "stage", "stages", "stream", "pathway",
}

ARCHITECTURE_INTENT_TERMS = {
    "summarize", "summary", "design", "structure", "innovation", "novel",
    "component", "module", "branch", "encoder", "decoder", "pipeline",
    "objective", "constraint", "loss", "penalty", "regularization",
    "ablation", "experiment", "effect", "improvement", "performance",
}

RELATION_CONNECTORS = {
    "because", "due to", "therefore", "thus", "hence",
    "leads to", "results in", "explains", "matches", "corresponds to",
    "driven by", "caused by", "consistent with", "deviates from",
    "constrained by", "compared with", "whereas", "despite", "under",
}

# Cross-modal operator cues tracked by QC as advisory features.
# v4.3 does not hard-fail missing operator words; we keep these for diagnostics.
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
    # synonyms frequently used by prompt sentence structures
    "correspond", "corresponds", "corresponded",
    "account", "accounts", "accounted",
    "limit", "limits", "limited",
    "associate", "associates", "associated",
    "depend", "depends", "depended",
    "link", "links", "linked",
    "relate", "relates", "related",
    "appear", "appears", "appeared",
    "decline", "declines", "declined", "declining",
    "inconsistent",
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
    # comparative/linking variants frequently produced by the model
    "difference", "differences",
    "better", "worse",
    "emerge", "emerges", "emerged",
    "outperform", "outperforms", "outperformed",
}

# Phrase-level operator patterns that often express cross-modal linkage
# but are not always captured by single-token operator matching.
CROSS_MODAL_OPERATOR_PATTERNS = (
    r"\bdifferent(?:\s+from|\s+than)?\b",
    r"\bcompar(?:e|es|ed|ing|ison)\b",
    r"\b(higher|lower|greater|less)\b.{0,20}\bthan\b",
    r"\bmaintain(?:s|ed|ing)?\b",
)


# 检测 query 泄露具体数值（排除 0/1 常数和年份）
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


# query 缺乏跨模态动词（affect/differ/produce 等），词级+短语级双检
def has_no_cross_modal_operator(query: str) -> bool:
    """Return True when no explicit cross-modal operator cue is found."""
    q = query.lower()
    token_hit = any(
        re.search(rf"\b{re.escape(op)}\b", q, re.IGNORECASE)
        for op in CROSS_MODAL_OPERATORS
    )
    if token_hit:
        return False
    phrase_hit = any(re.search(p, q, re.IGNORECASE) for p in CROSS_MODAL_OPERATOR_PATTERNS)
    return not phrase_hit


# evidence_spans 必须覆盖双端元素且每条 span ≥ 8 字符
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


# 先检 wh-word 全局否决，再检 aux-verb 开头和介词前缀嵌套
# 解决了 "Given X are Y, why…" 被首词 are 误判的问题
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


def has_template_collapse(query: str) -> bool:
    q = query.strip().lower()
    return any(re.search(p, q) for p in TEMPLATE_COLLAPSE_PATTERNS)


# 取前 2 词作为开头指纹，用于 template collapse 去重
def query_opening_signature(query: str, n_words: int = 2) -> str:
    toks = re.findall(r"[a-zA-Z]+", query.lower())
    return " ".join(toks[:n_words]) if toks else ""


def query_word_count(query: str) -> int:
    return len(re.findall(r"[A-Za-z0-9][A-Za-z0-9_-]*", query or ""))


# 分桶：too_short / short / medium / long / too_long
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


# 检查同一 pair 的两条 query 是否覆盖 short+long 两种长度
def has_length_mix(queries: List[Dict[str, Any]]) -> Tuple[bool, bool]:
    has_short = False
    has_long = False
    for obj in queries:
        bucket = query_length_bucket(str(obj.get("query", "")))
        if bucket == "short":
            has_short = True
        if bucket == "long":
            has_long = True
    return has_short, has_long


def _is_architecture_text(text: str) -> bool:
    t = (text or "").lower()
    return any(re.search(rf"\b{re.escape(k)}\b", t) for k in ARCHITECTURE_KEYWORDS)


# 架构图检测：figure 的 caption+context 命中 encoder/decoder/module 等关键词
def is_architecture_pair(pair: Dict[str, Any]) -> bool:
    elem_a = pair.get("element_a", {}) or {}
    elem_b = pair.get("element_b", {}) or {}
    fig_elem = elem_a if str(elem_a.get("element_type", "")) == "figure" else elem_b
    if str(fig_elem.get("element_type", "")) != "figure":
        return False
    fig_text = " ".join(
        [
            str(fig_elem.get("caption", "") or ""),
            str(fig_elem.get("content", "") or ""),
            str(fig_elem.get("context_before", "") or ""),
            str(fig_elem.get("context_after", "") or ""),
        ]
    )
    return _is_architecture_text(fig_text)


def has_architecture_intent(query: str, answer: str) -> bool:
    qa = f"{query or ''} {answer or ''}".lower()
    return any(re.search(rf"\b{re.escape(k)}\b", qa) for k in ARCHITECTURE_INTENT_TERMS)


# 检测 ", and which/what" 式伪多跳——两个独立子问题拼接而非串行推理
def has_parallel_dual_ask(query: str) -> bool:
    q = query.strip().lower()
    return bool(
        re.search(r",\s*and\s+(?:which|what|under\s+what)\b", q)
        or re.search(r"\band\s+which\b", q)
        or re.search(r"\band\s+what\b", q)
    )


# 高精度检测："A different from B" 中一边是 structural 另一边是 metric
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

    # Mismatch only when one side is clearly structural and the other clearly metric.
    if left_struct and right_metric and not left_metric:
        return True
    if right_struct and left_metric and not right_metric:
        return True
    return False


# reasoning_chain 须 ≥40 字且含 ≥2 个句子片段，确保有因果分解
def has_min_reasoning_chain(obj: Dict[str, Any]) -> bool:
    rc = str(obj.get("reasoning_chain", "")).strip()
    if len(rc) < 40:
        return False
    # Require at least 2 sentence fragments as minimal causal decomposition.
    chunks = [c.strip() for c in re.split(r"[.;\n]", rc) if c.strip()]
    return len(chunks) >= 2


def has_relationship_connector(answer: str) -> bool:
    a = answer.lower()
    return any(conn in a for conn in RELATION_CONNECTORS)


# 高精度：捕获 "drop below" vs "not drop below" 等 LLM 生成的前提-答案矛盾
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


# 从 LaTeX 公式中提取变量/函数符号供 QC grounding 检查
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


# 从 HTML/markdown 表格提取列头，避免泄露密集数值
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


# ≥3 字母的词集合，去停用词——用于 anchor leak 和 answer overlap 计算
def _content_tokens(text: str) -> Set[str]:
    words = set(re.findall(r"\b[a-zA-Z]{3,}\b", text.lower()))
    return words - LEAK_STOPWORDS


def _number_tokens(text: str) -> Set[str]:
    """Extract normalized numeric tokens used for evidence overlap."""
    nums = set()
    for raw in re.findall(r"\b\d+(?:[.,]\d+)?%?\b", text):
        tok = raw.replace(",", "").rstrip("%")
        if tok:
            nums.add(tok)
    return nums


# 提取希腊字母 / 下标变量 / LaTeX 命令等公式专有符号
def _extract_formula_symbol_terms(content: str) -> Set[str]:
    """Extract formula-specific symbolic terms used for grounding checks."""
    if not content:
        return set()
    text = content
    regions: List[str] = []
    regions += re.findall(r"\$(.+?)\$", text, flags=re.DOTALL)
    regions += re.findall(r"\\\((.+?)\\\)", text, flags=re.DOTALL)
    regions += re.findall(r"\\\[(.+?)\\\]", text, flags=re.DOTALL)
    math_text = " ".join(regions) if regions else text

    terms: Set[str] = set()
    for g in re.findall(r"\\(alpha|beta|gamma|delta|epsilon|lambda|mu|sigma|tau|theta|phi|psi|omega)", math_text):
        terms.add(g.lower())
    for t in re.findall(r"\b([A-Za-z]+_[A-Za-z0-9]+)\b", math_text):
        terms.add(t.lower())
    for t in re.findall(r"\b([A-Za-z]+\([A-Za-z0-9,\s]+\))", math_text):
        terms.add(re.sub(r"\s+", "", t.lower()))
    ignore_cmds = {"begin", "end", "left", "right", "frac", "cdot", "times", "sum", "prod", "int", "mid", "tag"}
    for cmd in re.findall(r"\\([A-Za-z]{2,})", math_text):
        c = cmd.lower()
        if c not in ignore_cmds:
            terms.add(c)
    return {t for t in terms if t and len(t) >= 2}


# answer 是否提及了至少一个公式符号（grounding 验证）
def _formula_symbol_hit(answer: str, terms: Set[str]) -> bool:
    """Return True if answer explicitly mentions at least one formula symbol/term."""
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


# answer 与 text_evidence 的词重合率——过高说明 answer 只是在复述
def _answer_text_evidence_overlap(answer: str, evidence: str) -> float:
    """Token overlap ratio between answer and text_evidence (answer-normalized)."""
    a_tokens = _content_tokens(answer)
    if not a_tokens:
        return 0.0
    e_tokens = _content_tokens(evidence)
    if not e_tokens:
        return 0.0
    return len(a_tokens & e_tokens) / len(a_tokens)


# query 与 visual_anchors 的最大 Jaccard——超 0.20 则判定 anchor 泄露
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


def anchor_overlap_tokens(query: str, anchors: List[Dict[str, Any]]) -> Set[str]:
    """Return overlapping content tokens between query and all anchors."""
    q_tokens = _content_tokens(query)
    all_anchor_tokens: Set[str] = set()
    for a in anchors:
        a_text = a.get("anchor", "") if isinstance(a, dict) else str(a)
        all_anchor_tokens |= _content_tokens(a_text)
    return q_tokens & all_anchor_tokens


# QC 主入口：meta_language / anchor_leakage / single_element_answer 等全套检查
def qc_multihop_query(
    obj: Dict[str, Any],
    pair: Dict[str, Any],
) -> Tuple[List[str], Dict[str, Any]]:
    """Run QC checks on a multi-hop L1 query. Returns (issues, metrics)."""
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

    # 2d. Weak shortcut templates
    if has_shortcut_template(q):
        issues.append("template_shortcut")

    # 2e. Templated opening collapse
    if has_templated_opening(q):
        issues.append("templated_opening")
        metrics["templated_opening_warn"] = True

    # 2ea. High-frequency shell patterns that indicate template collapse.
    if has_template_collapse(q):
        issues.append("template_collapse")
        metrics["template_collapse_warn"] = True

    # 2f. Parallel dual-ask shortcut (fake multi-hop by sentence stitching)
    if has_parallel_dual_ask(q):
        issues.append("pseudo_multihop_parallel")
        metrics["parallel_dual_ask_warn"] = True

    # 2g. Cross-category mismatch in "different from"/"inconsistent with" framing
    if has_semantic_category_mismatch(q):
        issues.append("semantic_category_mismatch")
        metrics["semantic_category_mismatch_warn"] = True

    # 2h. Require pre-query reasoning chain so generation is causality-first.
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

    # 4b. Premise-answer contradiction (high precision, hard fail)
    if has_premise_answer_contradiction(q, a):
        issues.append("premise_answer_contradiction")
        metrics["premise_contradiction_warn"] = True

    # 5. Anchor leakage
    leak = anchor_leak_jaccard(q, anchors)
    metrics["anchor_leak_jaccard"] = round(leak, 4)
    overlap_tokens = anchor_overlap_tokens(q, anchors)
    if leak > ANCHOR_LEAK_THRESHOLD:
        span_tokens: Set[str] = set()
        for s in obj.get("required_evidence_spans", []) or []:
            if isinstance(s, dict):
                span_tokens |= _content_tokens(str(s.get("span", "")))
        allowed_tokens = ENTITY_AMNESTY_TERMS | span_tokens
        # Entity amnesty: if overlap is only domain entities/evidence terms, track but do not fail.
        if overlap_tokens and overlap_tokens <= allowed_tokens:
            metrics["anchor_leakage_amnestied"] = True
        else:
            issues.append("anchor_leakage")
    anchor_copy = anchor_token_copy_count(q, anchors)
    metrics["anchor_token_copy_count"] = anchor_copy
    # v4 (Option A): bridge_entity_leakage is a SOFT tracking metric only,
    # not a hard QC fail. For dual-evidence (path_len=2) queries, entity names
    # are legitimate search targets — hiding them makes no sense without multi-hop.
    if anchor_copy >= 4:
        metrics["bridge_entity_leakage_warn"] = True

    # 5b. Cross-modal operator is now advisory (no hard fail).
    metrics["has_cross_modal_operator"] = not has_no_cross_modal_operator(q)

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
    a_tokens = _content_tokens(a)
    a_num_tokens = _number_tokens(a)
    if a_tokens or a_num_tokens:
        # 按元素类型返回最低 overlap 阈值：figure=1, table/formula=2
        def _min_overlap_for_elem(elem: Dict) -> int:
            etype = str(elem.get("element_type", "")).lower()
            return int(MIN_OVERLAP_BY_TYPE.get(etype, 2))

        # 拼接 caption + content + evidence_spans 的合并文本
        def _elem_text(elem: Dict) -> str:
            caption = elem.get("caption", "") or ""
            # Use caption + raw content for all modalities (including formula)
            # to avoid prose inflation in overlap scoring.
            return caption + " " + (elem.get("content", "") or "")

        span_text_by_eid: Dict[str, str] = defaultdict(str)
        for s in obj.get("required_evidence_spans", []) or []:
            if not isinstance(s, dict):
                continue
            eid = str(s.get("element_id", "")).strip()
            span = str(s.get("span", "")).strip()
            if eid and span:
                span_text_by_eid[eid] += " " + span

        # 计算 answer 与单个元素的词+数字重合度
        def _elem_overlap(elem: Dict, elem_id: str) -> int:
            # Merge raw element text with its required evidence span to reduce
            # false negatives from OCR/style variation in table/formula content.
            merged = (_elem_text(elem) + " " + span_text_by_eid.get(elem_id, "")).strip()
            word_overlap = len(a_tokens & _content_tokens(merged))
            num_overlap = len(a_num_tokens & _number_tokens(merged))
            # Cap numeric contribution so lexical grounding still dominates.
            return word_overlap + min(num_overlap, 2)

        elem_a = pair.get("element_a", {})
        elem_b = pair.get("element_b", {})
        elem_a_id = pair.get("element_a_id", "")
        elem_b_id = pair.get("element_b_id", "")
        overlap_a = _elem_overlap(elem_a, elem_a_id)
        overlap_b = _elem_overlap(elem_b, elem_b_id)
        metrics["answer_overlap_a"] = overlap_a
        metrics["answer_overlap_b"] = overlap_b
        total = overlap_a + overlap_b
        metrics["answer_balance"] = 0.0
        if total > 0:
            contrib_a = overlap_a / total
            contrib_b = overlap_b / total
            balance = min(contrib_a, contrib_b)
            metrics["answer_balance"] = round(balance, 4)
            # Require non-trivial overlap from BOTH elements.
            if (
                overlap_a < _min_overlap_for_elem(elem_a)
                or overlap_b < _min_overlap_for_elem(elem_b)
                or balance < ANSWER_BALANCE_THRESHOLD
            ):
                issues.append("single_element_answer")
        else:
            issues.append("single_element_answer")

    # 8. Text evidence length
    evidence = obj.get("text_evidence", "")
    if len(evidence) < 40:
        issues.append("short_evidence")

    # 8b. Guard against answer over-reliance on text_evidence paraphrase.
    text_evidence_overlap = _answer_text_evidence_overlap(a, evidence)
    metrics["text_evidence_overlap"] = round(text_evidence_overlap, 4)
    if text_evidence_overlap > TEXT_EVIDENCE_OVERLAP_WARN_THRESHOLD:
        issues.append("text_evidence_over_reliance")

    # 8c. Figure+formula symbolic grounding hard check.
    pair_type = str(pair.get("pair_type", ""))
    if pair_type == "figure+formula":
        formula_elem = pair.get("element_a", {}) if pair.get("element_a_type") == "formula" else pair.get("element_b", {})
        formula_text = (formula_elem.get("caption", "") or "") + " " + (formula_elem.get("content", "") or "")
        formula_terms = _extract_formula_symbol_terms(formula_text)
        metrics["formula_symbol_term_count"] = len(formula_terms)
        metrics["formula_symbol_grounded"] = _formula_symbol_hit(a, formula_terms)
        if formula_terms and not metrics["formula_symbol_grounded"]:
            issues.append("formula_symbol_grounding_missing")

    # 9. Encourage explicit relationship grounding instead of generic lookup answers.
    # v2.1: cross_reading is a lookup/referencing type, not explanatory — exempt from check.
    qtype = str(obj.get("query_type", "")).lower()
    explanatory_types = {
        "trend_explanation",
        "anomaly_investigation",
        "bridge_reasoning",
        "theory_vs_experiment",
        "data_formula_consistency",
        "causal_explanation",
        "discrepancy_analysis",
        "hypothesis_verification",
    }
    if qtype in explanatory_types and not has_relationship_connector(a):
        issues.append("weak_reasoning_connector")

    # 10. Architecture-specific check: avoid generic non-structural asks.
    is_arch_case = is_architecture_pair(pair)
    metrics["is_architecture_case"] = bool(is_arch_case)
    if is_arch_case and not has_architecture_intent(q, a):
        issues.append("architecture_intent_missing")

    return issues, metrics


# ── Query intent classification (objective vs subjective) ──────────────────
# Objective queries have a specific, verifiable answer (a number, a name,
# a yes/no fact).  Subjective queries are open-ended — the "answer" is a
# summary or analysis whose quality is hard to verify automatically.
# The distinction drives QC strictness: subjective queries skip
# single_element_answer checks (no single correct evidence span exists).

_SUBJECTIVE_QUERY_PATTERNS: List[re.Pattern] = [
    re.compile(r"\b(summarize|summarise|summary|overview|describe|explain|discuss)\b", re.I),
    re.compile(r"\bwhat (is|are|does|do|did)\b.{0,30}\b(about|mean|trying|aim|purpose|motivation|contribution|novel|limit)\b", re.I),
    re.compile(r"\b(briefly|concisely|in (one|a few) (sentence|word|paragraph))\b", re.I),
    re.compile(r"\b(tell me|give me|walk me|help me)\b", re.I),
    re.compile(r"\b(what (distinguishes|makes|sets|defines))\b", re.I),
    re.compile(r"\b(pros and cons|trade.?off|advantage|disadvantage|strength|weakness)\b", re.I),
]

_OBJECTIVE_QUERY_PATTERNS: List[re.Pattern] = [
    re.compile(r"\b(how (many|much|often|long|far|large|small|fast|accurate))\b", re.I),
    re.compile(r"\b(what (percentage|fraction|ratio|number|value|score|result|accuracy|f1|bleu|rouge))\b", re.I),
    re.compile(r"\b(which (model|method|approach|baseline|dataset|benchmark))\b", re.I),
    re.compile(r"\b(is there|does .+? show|does .+? outperform|is .+? better|is .+? worse)\b", re.I),
    re.compile(r"\b(what (specific|exact|precise))\b", re.I),
]


# 启发式判断 objective（有确定答案）vs subjective（开放性）
# 不确定时偏 objective，使 QC 门禁保持严格
def classify_query_intent(query: str) -> str:
    """Classify query as 'objective' or 'subjective'.

    Returns:
        "subjective": open-ended, no single correct answer, evidence-finding task.
        "objective": has a specific verifiable answer (number, name, comparison).

    Heuristic rule-based classification; errs toward 'objective' when uncertain
    so that QC checks remain strict by default.
    """
    q = (query or "").strip()
    subj_hits = sum(1 for p in _SUBJECTIVE_QUERY_PATTERNS if p.search(q))
    obj_hits = sum(1 for p in _OBJECTIVE_QUERY_PATTERNS if p.search(q))
    if subj_hits > obj_hits:
        return "subjective"
    return "objective"


# real-user 风格的宽松 QC：去掉 template/reasoning_chain 硬门禁，
# 新增 retrievability_score 和 query_intent 分轨
def qc_real_user_query(
    obj: Dict[str, Any],
    pair: Dict[str, Any],
    persona: str = "",
) -> Tuple[List[str], Dict[str, Any]]:
    """QC checks for real-user style queries (relaxed relative to academic style).

    Kept checks:
      - meta_language: no "figure"/"table"/"equation" in query
      - yes_no_question: no pure yes/no questions
      - single_element_answer: answer must draw on both elements
        (SKIPPED for subjective queries — open-ended questions have no single
        correct evidence span; retrievability_score is used instead)
      - empty/too_short/too_long: basic length sanity

    Removed / relaxed checks (relative to qc_multihop_query):
      - template_shortcut: "How does X relate to Y" is fine in real-user style
      - templated_opening / template_collapse: natural openers allowed
      - pseudo_multihop_parallel: dual-ask style acceptable in conversational queries
      - weak_reasoning_connector: colloquial answers needn't use "because/due to"
      - missing_reasoning_chain: no explicit reasoning_chain requirement
      - architecture_intent_missing: not applicable
      - length_mix_missing: no SHORT+LONG bucket requirement

    New metrics:
      - query_intent ("objective" | "subjective"): whether the query has a
        specific verifiable answer or is open-ended.
      - retrievability_score (0–3): rough estimate of how retrievable the answer is
        from the two elements without additional context.
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

    # 1. Meta-language (kept)
    if any(re.search(p, q_lower) for p in BAD_META_PATTERNS):
        issues.append("meta_language")

    # 2. Empty / length (kept, relaxed max to 35 words for real-user style)
    if not q.strip():
        issues.append("empty_query")
    elif q_words < 4:
        issues.append("query_too_short")
    elif q_words > 35:
        issues.append("query_too_long")

    # 2b. Skim-reader persona hard length constraint: max SHORT_QUERY_MAX_WORDS (14 words).
    #     Personas whose id contains "skim" or "lazy" or "busy" or "conference_attendee"
    #     are expected to produce terse queries; anything longer defeats the purpose.
    _skim_keywords = ("skim", "lazy", "busy", "conference_attendee")
    if persona and any(kw in persona for kw in _skim_keywords) and q_words > SHORT_QUERY_MAX_WORDS:
        issues.append("lazy_query_too_long")
    metrics["persona_applied"] = persona or "none"

    # 3. Yes/no question (kept)
    if is_yes_no_question(q):
        issues.append("yes_no_question")

    # 4. Single-element answer: enforced only for objective queries.
    #    Subjective / open-ended queries (summarize, explain, describe…) often
    #    don't have a single bounded answer that overlaps with element text, so
    #    applying this check would incorrectly reject valid queries.
    #    For subjective queries we rely on retrievability_score instead.
    a_tokens = _content_tokens(a)
    a_num_tokens = _number_tokens(a)
    if (a_tokens or a_num_tokens) and query_intent == "objective":
        def _min_overlap_ru(elem: Dict) -> int:
            etype = str(elem.get("element_type", "")).lower()
            return int(MIN_OVERLAP_BY_TYPE.get(etype, 2))

        def _elem_text_ru(elem: Dict) -> str:
            return (elem.get("caption", "") or "") + " " + (elem.get("content", "") or "")

        span_text_by_eid: Dict[str, str] = defaultdict(str)
        for s in obj.get("required_evidence_spans", []) or []:
            if not isinstance(s, dict):
                continue
            eid = str(s.get("element_id", "")).strip()
            span = str(s.get("span", "")).strip()
            if eid and span:
                span_text_by_eid[eid] += " " + span

        elem_a = pair.get("element_a", {})
        elem_b = pair.get("element_b", {})
        elem_a_id = pair.get("element_a_id", "")
        elem_b_id = pair.get("element_b_id", "")

        def _overlap_ru(elem: Dict, elem_id: str) -> int:
            merged = (_elem_text_ru(elem) + " " + span_text_by_eid.get(elem_id, "")).strip()
            word_ov = len(a_tokens & _content_tokens(merged))
            num_ov = len(a_num_tokens & _number_tokens(merged))
            return word_ov + min(num_ov, 2)

        ov_a = _overlap_ru(elem_a, elem_a_id)
        ov_b = _overlap_ru(elem_b, elem_b_id)
        metrics["answer_overlap_a"] = ov_a
        metrics["answer_overlap_b"] = ov_b
        total = ov_a + ov_b
        metrics["answer_balance"] = 0.0
        if total > 0:
            balance = min(ov_a / total, ov_b / total)
            metrics["answer_balance"] = round(balance, 4)
            if (
                ov_a < _min_overlap_ru(elem_a)
                or ov_b < _min_overlap_ru(elem_b)
                or balance < ANSWER_BALANCE_THRESHOLD
            ):
                issues.append("single_element_answer")
        else:
            issues.append("single_element_answer")

    # 5. Retrievability score (0–3, higher = more self-contained / easier to retrieve)
    #    Heuristic: +1 if answer length > 30 chars, +1 if required_evidence_spans non-empty,
    #    +1 if text_evidence non-empty
    retv = 0
    if len(a.strip()) > 30:
        retv += 1
    if obj.get("required_evidence_spans"):
        retv += 1
    evidence = obj.get("text_evidence", "")
    if len(evidence) >= 30:
        retv += 1
    metrics["retrievability_score"] = retv

    # 6. Guard against answer over-reliance on text_evidence paraphrase.
    evidence = obj.get("text_evidence", "")
    text_evidence_overlap = _answer_text_evidence_overlap(a, evidence)
    metrics["text_evidence_overlap"] = round(text_evidence_overlap, 4)
    if text_evidence_overlap > TEXT_EVIDENCE_OVERLAP_WARN_THRESHOLD:
        issues.append("text_evidence_over_reliance")

    # 7. Real-user numeric consistency guard: answer numbers must be grounded.
    answer_nums = _number_tokens(a)
    source_text_parts: List[str] = [
        str(pair.get("element_a", {}).get("caption", "") or ""),
        str(pair.get("element_a", {}).get("content", "") or ""),
        str(pair.get("element_a", {}).get("enriched_content", "") or ""),
        str(pair.get("element_a", {}).get("context_before", "") or ""),
        str(pair.get("element_a", {}).get("context_after", "") or ""),
        str(pair.get("element_b", {}).get("caption", "") or ""),
        str(pair.get("element_b", {}).get("content", "") or ""),
        str(pair.get("element_b", {}).get("enriched_content", "") or ""),
        str(pair.get("element_b", {}).get("context_before", "") or ""),
        str(pair.get("element_b", {}).get("context_after", "") or ""),
        str(evidence or ""),
    ]
    for s in obj.get("required_evidence_spans", []) or []:
        if isinstance(s, dict):
            source_text_parts.append(str(s.get("span", "") or ""))
    source_nums = _number_tokens(" ".join(source_text_parts))
    unsupported_nums = sorted(answer_nums - source_nums)
    metrics["unsupported_answer_numbers"] = unsupported_nums
    if unsupported_nums:
        issues.append("numeric_unsupported")

    # 8. Figure+formula symbolic grounding hard check.
    pair_type = str(pair.get("pair_type", ""))
    if pair_type == "figure+formula":
        formula_elem = pair.get("element_a", {}) if pair.get("element_a_type") == "formula" else pair.get("element_b", {})
        formula_text = (formula_elem.get("caption", "") or "") + " " + (formula_elem.get("content", "") or "")
        formula_terms = _extract_formula_symbol_terms(formula_text)
        metrics["formula_symbol_term_count"] = len(formula_terms)
        metrics["formula_symbol_grounded"] = _formula_symbol_hit(a, formula_terms)
        if formula_terms and not metrics["formula_symbol_grounded"]:
            issues.append("formula_symbol_grounding_missing")

    return issues, metrics


# ──────────────────────────────────────────────────────────────
# M4 Multi-hop Reasoning Depth QC (Schema 1 validation)
# ──────────────────────────────────────────────────────────────

# Causal/logical connectors that indicate serial reasoning (step N depends on step N-1)
_SERIAL_REASONING_MARKERS = re.compile(
    r"\b(because|therefore|thus|hence|consequently|as a result|this (?:means|implies|suggests|explains|shows)|"
    r"which (?:means|implies|suggests|explains|leads|causes)|"
    r"due to|owing to|given that|since .{5,30}then|"
    r"if .{5,40}then|from .{5,40}(?:follows|we (?:can|see|know))|"
    r"(?:step|first|second|third|next|finally|building on|combining))\b",
    re.IGNORECASE,
)

# Parallel/additive connectors that indicate evidence combination (not serial reasoning)
_PARALLEL_EVIDENCE_MARKERS = re.compile(
    r"\b((?:both|together|combined|alongside|in addition|additionally|also|moreover|furthermore|"
    r"similarly|likewise|and .{0,10}respectively|while .{5,30}also))\b",
    re.IGNORECASE,
)

# Evidence type indicators for reasoning_steps classification
_EVIDENCE_TYPE_PATTERNS = {
    "observation": re.compile(r"\b(show|display|illustrate|reveal|depict|present|indicate|demonstrate)\b", re.I),
    "attribution": re.compile(r"\b(cause|due to|because|result from|attribute|ablation|removing|without)\b", re.I),
    "explanation": re.compile(r"\b(prove|equation|formula|inequality|theorem|bound|derive|constraint)\b", re.I),
    "verification": re.compile(r"\b(confirm|validate|verify|consistent with|align|match|support)\b", re.I),
    "prediction": re.compile(r"\b(predict|expect|would|should|if .{5,30} then|hypothesize|imply)\b", re.I),
}


# 用连接词模式 (because→serial, both→parallel) 区分推理结构
# 局限：写作风格可欺骗，仅适合 auto-tagging 不适合严格门禁
def classify_reasoning_structure(
    reasoning_chain: str,
    answer: str,
    evidence_spans: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """Classify the reasoning structure of a query as parallel vs serial.

    Returns a dict with:
      - reasoning_depth_estimate: int (1=single, 2=parallel, 3+=serial chain)
      - structure: "parallel" | "serial" | "mixed"
      - serial_markers_found: list of matched serial reasoning phrases
      - parallel_markers_found: list of matched parallel evidence phrases
      - evidence_types: list of detected evidence types per span
      - is_true_multihop: bool (True if serial with depth >= 3)
    """
    combined_text = (reasoning_chain or "") + " " + (answer or "")

    serial_hits = _SERIAL_REASONING_MARKERS.findall(combined_text)
    parallel_hits = _PARALLEL_EVIDENCE_MARKERS.findall(combined_text)

    # Classify evidence types per span
    evidence_types = []
    for span_obj in (evidence_spans or []):
        span_text = str(span_obj.get("span", "")) if isinstance(span_obj, dict) else ""
        best_type = "observation"  # default
        best_count = 0
        for etype, pat in _EVIDENCE_TYPE_PATTERNS.items():
            hits = len(pat.findall(span_text))
            if hits > best_count:
                best_count = hits
                best_type = etype
        evidence_types.append(best_type)

    # Count distinct evidence sources
    distinct_elements = len(set(
        str(s.get("element_id", ""))
        for s in (evidence_spans or [])
        if isinstance(s, dict) and s.get("element_id")
    ))

    # Estimate reasoning depth
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

    # Detect step-deletion vulnerability: if all evidence is used in a flat AND,
    # it's parallel even if serial markers exist
    # Heuristic: if answer contains "both" + "and" patterns, likely parallel
    flat_and_pattern = re.compile(
        r"\b(both .{5,60} and )\b", re.I
    )
    if flat_and_pattern.search(answer) and structure == "serial":
        structure = "mixed"  # downgrade: serial markers but flat combination in answer

    is_true_multihop = structure == "serial" and depth >= 3

    return {
        "reasoning_depth_estimate": depth,
        "structure": structure,
        "serial_markers_found": serial_hits[:5],  # cap for storage
        "parallel_markers_found": parallel_hits[:5],
        "evidence_types": evidence_types,
        "distinct_evidence_elements": distinct_elements,
        "is_true_multihop": is_true_multihop,
    }


# M4 Schema 1 支持：有 reasoning_steps[] 则结构验证（hard fail），
# 否则用启发式连接词分析（advisory only，不做硬门禁）
def qc_reasoning_depth(
    obj: Dict[str, Any],
    pair: Dict[str, Any],
    min_depth: int = 3,
) -> Tuple[List[str], Dict[str, Any]]:
    """Reasoning depth analysis and auto-tagging (M4 Schema 1 support).

    Two modes:
      A) If obj contains explicit `reasoning_steps[]` (new Schema 1 format):
         performs structural validation (dependencies, element diversity, role arc)
         and returns hard-fail issues.
      B) Otherwise: heuristic analysis of free-text reasoning_chain + answer
         using causal/parallel connector patterns. Returns advisory metrics only.

    IMPORTANT — Known limitations of heuristic mode (B):
      - Based on language surface features (connector words), NOT actual reasoning
        structure. Writing style can inflate or deflate depth estimates.
      - Not robust across query styles (academic vs real_user have different
        connector distributions).
      - evidence_type classification depends on span keyword matching.
      - step_deletion_proxy is causal-link word count, NOT a true step-deletion
        test (which would require deleting a step and re-judging answer derivability).

    Best used for: auto-tagging / dataset profiling / analytics.
    NOT suitable for: strict M4 qualification gate (until Phase 1 real validation).

    Args:
        obj: The generated query object
        pair: The candidate pair (currently unused in heuristic mode; reserved for
              future graph-topology-aware validation)
        min_depth: Minimum reasoning depth to consider as "true multi-hop" (default 3)

    Returns:
        (issues, metrics) tuple compatible with existing QC framework
    """
    issues: List[str] = []
    metrics: Dict[str, Any] = {}

    reasoning_chain = obj.get("reasoning_chain", "")
    answer = obj.get("answer", "")
    evidence_spans = obj.get("required_evidence_spans", []) or []
    reasoning_steps = obj.get("reasoning_steps", []) or []

    # ── If explicit reasoning_steps are provided (new Schema 1 format) ──
    if reasoning_steps and len(reasoning_steps) >= 2:
        metrics["has_explicit_reasoning_steps"] = True
        metrics["num_reasoning_steps"] = len(reasoning_steps)

        # Check step dependencies form a chain (not all independent)
        has_dependency_chain = False
        for step in reasoning_steps[1:]:
            deps = step.get("depends_on_steps", [])
            if deps:
                has_dependency_chain = True
                break
        metrics["has_dependency_chain"] = has_dependency_chain
        if not has_dependency_chain:
            issues.append("reasoning_steps_no_dependencies")

        # Check all steps use different evidence elements
        step_elements = [s.get("evidence_element_id", "") for s in reasoning_steps]
        unique_step_elements = set(e for e in step_elements if e)
        metrics["unique_step_elements"] = len(unique_step_elements)
        if len(unique_step_elements) < len(reasoning_steps):
            issues.append("reasoning_steps_duplicate_evidence")

        # Check evidence types are diverse (not all "observation")
        step_types = [s.get("evidence_type", "observation") for s in reasoning_steps]
        unique_types = set(step_types)
        metrics["reasoning_step_types"] = step_types
        if len(unique_types) < 2:
            issues.append("reasoning_steps_uniform_type")

        # Check reasoning roles have progression (premise → intermediate → conclusion)
        roles = [s.get("reasoning_role", "") for s in reasoning_steps]
        if roles:
            has_premise = "premise" in roles
            has_conclusion = "conclusion" in roles
            metrics["has_premise_conclusion_arc"] = has_premise and has_conclusion
            if not (has_premise and has_conclusion):
                issues.append("reasoning_steps_no_arc")

        # Depth from explicit steps
        depth = len(reasoning_steps)
        metrics["reasoning_depth"] = depth
        if depth < min_depth:
            issues.append(f"reasoning_depth_insufficient_{depth}_of_{min_depth}")

    else:
        # ── Heuristic analysis of free-text reasoning chain + answer ──
        metrics["has_explicit_reasoning_steps"] = False

        analysis = classify_reasoning_structure(
            reasoning_chain, answer, evidence_spans,
        )
        metrics.update({
            "reasoning_depth": analysis["reasoning_depth_estimate"],
            "reasoning_structure": analysis["structure"],
            "serial_markers": analysis["serial_markers_found"],
            "parallel_markers": analysis["parallel_markers_found"],
            "evidence_types_detected": analysis["evidence_types"],
            "distinct_evidence_elements": analysis["distinct_evidence_elements"],
            "is_true_multihop": analysis["is_true_multihop"],
        })

        # For existing dual-evidence queries, classify but don't hard-fail
        # (they were designed as parallel, not serial)
        depth = analysis["reasoning_depth_estimate"]
        if depth < min_depth:
            # Soft classification, not hard fail for backward compat
            metrics["reasoning_depth_gap"] = min_depth - depth

    # ── Step-deletion heuristic (works for both formats) ──
    # Count distinct causal links in the answer: "A causes B", "B leads to C"
    causal_link_pattern = re.compile(
        r"(?:because|since|due to|as a result of|which (?:causes|leads|explains|means)|"
        r"therefore|thus|hence|consequently|this (?:causes|leads|explains|means))",
        re.IGNORECASE,
    )
    causal_links = causal_link_pattern.findall(answer)
    metrics["causal_link_count"] = len(causal_links)
    metrics["step_deletion_proxy"] = len(causal_links) >= (min_depth - 1)
    # A true 3-step chain needs at least 2 causal links: A→B and B→C

    # ── P3: Hub-aware QC — bridge grounding check ──────────────
    # For L3 queries with explicit reasoning_steps, verify bridge step
    # actually references content from the bridge paragraph
    if reasoning_steps and len(reasoning_steps) >= 3:
        bridge_step = next(
            (s for s in reasoning_steps if s.get("reasoning_role") == "intermediate"),
            None,
        )
        if bridge_step:
            bridge_span = (bridge_step.get("evidence_span", "") or "").strip()
            bridge_claim = (bridge_step.get("produces_claim", "") or "").strip()
            # Check bridge step has substantive content (not generic filler)
            if len(bridge_span) < 15:
                issues.append("bridge_span_too_short")
            if not bridge_claim or len(bridge_claim) < 10:
                issues.append("bridge_claim_empty")
            metrics["bridge_span_length"] = len(bridge_span)
            metrics["bridge_claim_length"] = len(bridge_claim)

    # P3: For L3, hard-fail if structure is parallel (not serial)
    # This is activated only when the pair has reasoning_chain_target flag
    if pair.get("reasoning_chain_target") and metrics.get("reasoning_structure") == "parallel":
        if metrics.get("has_explicit_reasoning_steps"):
            # Check if reasoning_steps actually form a dependency chain
            if not metrics.get("has_dependency_chain", False):
                issues.append("l3_parallel_not_serial")

    # ── P4: Anchor specificity check ──────────────────────────
    # Verify visual_anchors have specific location info, not generic descriptions
    visual_anchors = obj.get("visual_anchors", []) or []
    _GENERIC_ANCHOR_PATTERNS = re.compile(
        r'^(?:the (?:table|figure|formula|chart|graph|plot|image|equation)|'
        r'(?:table|figure|formula) (?:content|data|information)|'
        r'overall (?:structure|layout|content))$',
        re.IGNORECASE,
    )
    _SPECIFIC_ANCHOR_MARKERS = re.compile(
        r'(?:row|column|col|cell|axis|line|bar|marker|label|legend|'
        r'panel|subplot|left|right|top|bottom|color|dashed|solid|'
        r'\d+(?:st|nd|rd|th)|x-axis|y-axis|variable|term|coefficient|'
        r'numerator|denominator|subscript|superscript)',
        re.IGNORECASE,
    )
    generic_anchor_count = 0
    specific_anchor_count = 0
    for va in visual_anchors:
        anchor_text = (va.get("anchor", "") or "").strip()
        if not anchor_text or len(anchor_text) < 5:
            generic_anchor_count += 1
        elif _GENERIC_ANCHOR_PATTERNS.search(anchor_text):
            generic_anchor_count += 1
        elif _SPECIFIC_ANCHOR_MARKERS.search(anchor_text):
            specific_anchor_count += 1
        else:
            # Heuristic: anchors with 3+ words that mention specific content are OK
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


# ──────────────────────────────────────────────────────────────
# Image encoding
# ──────────────────────────────────────────────────────────────

# 集群路径 → 本地 MinerU 输出目录的 fallback 解析
def _fallback_image_path(original: str) -> Optional[Path]:
    """Try to resolve an image path from a different environment.

    When multimodal_elements.json was generated on cluster, image_path may
    contain absolute cluster paths (e.g. /projects/.../mineru_output/...).
    This tries to find the same file under the local data/mineru_output/.
    """
    parts = Path(original).parts
    for i, part in enumerate(parts):
        if part == "mineru_output" and i + 1 < len(parts):
            doc_id = parts[i + 1]
            fname = Path(original).name
            # Try common MinerU output layouts
            for subpath in [
                Path(doc_id, doc_id, "hybrid_auto", "images", fname),
                Path(doc_id, doc_id, "images", fname),
                Path(doc_id, "images", fname),
            ]:
                candidate = PROJECT_ROOT / "data" / "mineru_output" / subpath
                if candidate.exists():
                    return candidate
            break
    return None


# 图片 → (base64, mime)，含跨环境路径 fallback
def encode_image(path: Optional[str]) -> Optional[Tuple[str, str]]:
    """Return (base64_data, mime_type) or None if file missing."""
    if not path:
        return None
    p = Path(path)
    if not p.is_absolute():
        p = PROJECT_ROOT / path
    # Fallback: resolve cross-environment paths (cluster → local)
    if not p.exists():
        fallback = _fallback_image_path(path)
        if fallback:
            p = fallback
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

# 将 edge_context 列表格式化为 prompt 可用的文本块
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


# 注入作者原话——LaTeX \ref{} 处的桥接句，给模型语义 grounding
# 而非暴露原始 content（避免 anchor leakage）
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


# ──────────────────────────────────────────────────────────────
# C1: Enrichment noise filter
# ──────────────────────────────────────────────────────────────

# Patterns that indicate a low-quality / noisy enriched field.
# If any pattern matches, the enriched field is discarded and we
# fall back to the original raw context so the LLM isn't misled.
_ENRICHMENT_NOISE_PATTERNS = [
    # Unicode decorative / glyph / icon characters
    r"[\u2460-\u2473\u25a0-\u25ff\u2600-\u26ff\u2700-\u27bf]",
    # Explicit noise labels from common OCR / PDF-extraction artefacts
    r"\b(glyph|icon|marker|symbol|bullet|arrow|checkmark|watermark)\b",
    # Circled / enclosed numbers used as decorative list markers
    r"[①②③④⑤⑥⑦⑧⑨⑩⑪⑫⑬⑭⑮]",
    # Self-describing OCR failure strings
    r"\b(ocr error|illegible|unreadable|corrupted text|extraction failed)\b",
    # Repeated non-alphanumeric filler (e.g. "--- --- ---", "· · ·")
    r"^[\W\d\s]{0,20}$",
]

_ENRICHMENT_NOISE_RE = re.compile(
    "|".join(_ENRICHMENT_NOISE_PATTERNS), re.IGNORECASE | re.UNICODE
)


# C1 噪声过滤：glyph/icon/OCR artefact 等低质量 enrichment 直接丢弃
def _is_noisy_enrichment(text: str) -> bool:
    """Return True if the enriched field should be discarded as noise.

    A field is considered noisy if:
    - It is empty or too short to carry semantic meaning (< 15 chars after strip)
    - It matches any of the known glyph / icon / OCR-error patterns
    """
    if not text:
        return True
    stripped = text.strip()
    if len(stripped) < 15:
        return True
    return bool(_ENRICHMENT_NOISE_RE.search(stripped))


# MoDora [T]/[M]/[C] 富化描述 → prompt section（自动跳过噪声字段）
def build_enriched_context_section(pair: Dict) -> str:
    """Build enriched context section from MoDora-style [T]/[M]/[C] fields.

    Provides richer element descriptions when enriched fields are available
    from enrich_elements_modora.py output.  Low-quality / noisy enriched
    fields are silently dropped (C1 noise filter).
    """
    parts: List[str] = []

    for key, label in [("element_a", "Element A"), ("element_b", "Element B")]:
        elem = pair.get(key, {})
        enriched_title = elem.get("enriched_title", "") or ""
        enriched_content = elem.get("enriched_content", "") or ""
        # C1: discard noisy fields before including them in the prompt
        if _is_noisy_enrichment(enriched_title):
            enriched_title = ""
        if _is_noisy_enrichment(enriched_content):
            enriched_content = ""
        if enriched_title or enriched_content:
            section = f"[{label} enriched description]"
            if enriched_title:
                section += f" {enriched_title}."
            if enriched_content:
                section += f" {enriched_content}"
            parts.append(section)

    hub_summary = pair.get("hub_semantic_summary", "") or ""
    if not _is_noisy_enrichment(hub_summary):
        parts.append(f"[Hub bridge summary] {hub_summary}")

    if not parts:
        return ""
    return "## Enriched element descriptions\n" + "\n".join(parts)


# 将路径中经过的 section 节点的语义摘要注入 prompt
def build_section_context_section(pair: Dict) -> str:
    """Attach section/subsection summaries for section-aware paths when available."""
    if not _SECTION_ENRICH_CACHE:
        return ""

    section_ids: List[str] = []
    for nid in pair.get("path", []) or []:
        nid_str = str(nid)
        if "::sec::" in nid_str and nid_str not in section_ids:
            section_ids.append(nid_str)

    hub_meta = pair.get("hub_metadata", {}) or {}
    hub_id = str(hub_meta.get("node_id", "") or "").strip()
    if "::sec::" in hub_id and hub_id not in section_ids:
        section_ids.append(hub_id)

    parts: List[str] = []
    for sid in section_ids[:3]:
        row = _SECTION_ENRICH_CACHE.get(sid)
        if not row:
            continue
        title = (row.get("enriched_title") or row.get("section_title") or sid).strip()
        content = (row.get("enriched_content") or "").strip()
        metadata = row.get("enriched_metadata") or {}
        keywords = metadata.get("keywords") or []
        section_type = metadata.get("section_type") or row.get("node_type") or "section"

        block = f"[{section_type}] {title}"
        if keywords:
            block += f"\nKeywords: {', '.join(str(k) for k in keywords[:8])}"
        if content:
            block += f"\n{content[:500]}"
        parts.append(block)

    if not parts:
        return ""
    return "## Section-level semantic context\n" + "\n\n".join(parts)


# 架构图专项指导：强制问结构组件而非趋势，降低 intent_missing 率
def build_architecture_guidance(pair: Dict) -> str:
    """Inject a failure-case block for architecture diagrams."""
    if not is_architecture_pair(pair):
        return ""
    return """## Failure-case focus: architecture diagram quality
This figure is likely a model architecture/system diagram. Use a real scholar perspective.
- Query A (short, 8-14 words): summarize the core architecture choice or key innovation.
- Query B (long, 18-30 words): explain one concrete component/module/branch and connect it to a specific formula term plus experimental effect.
- Do NOT ask generic trend questions when the figure is structural.
- Prefer concrete wording: encoder/decoder branch, fusion module, loss path, regularization term, ablation effect."""


# 多跳路径的中间节点说明：优先用 bridge 原文，fallback 到 node ID
def build_intermediate_info(pair: Dict, all_elements: Optional[Dict] = None) -> str:
    """Describe intermediate elements in a multi-hop path.

    Enhanced (P0): resolves actual bridge paragraph text from the reference
    graph cache when available, instead of returning opaque node IDs.
    """
    path = pair.get("path", [])
    if len(path) <= 2:
        return "(direct connection)"

    # Try to resolve bridge texts from reference graph
    bridge_texts = resolve_bridge_texts_for_path(pair)
    if bridge_texts:
        return " → ".join(bridge_texts)

    # Fallback: return node IDs (backward-compatible)
    intermediate_ids = path[1:-1]
    parts = []
    for mid_id in intermediate_ids:
        parts.append(mid_id)
    return ", ".join(parts)


# 图路径可视化：figure_4 →[ref]→ paragraph_12 →[backbone]→ table_2
def build_graph_path_description(pair: Dict) -> str:
    """Build a human-readable graph path description for prompt injection.

    Example: "figure_4 →[ref]→ paragraph_12 →[backbone]→ paragraph_13 →[ref]→ table_2"
    """
    path = pair.get("path", [])
    if not path:
        return "(no path)"
    parts = []
    for i, node_id in enumerate(path):
        # Shorten node IDs for readability
        short = node_id.split("::")[-1] if "::" in node_id else node_id
        parts.append(short)
        if i < len(path) - 1:
            parts.append("→")
    return " ".join(parts)


# mixed 模式：按 pair_id md5 稳定 50/50 分配 academic / real_user
def resolve_query_style(query_style: str, pair_id: str) -> str:
    """Resolve effective style per pair.

    For `mixed`, use a deterministic 50/50 split by stable md5 hash so the
    same pair gets the same style across runs/machines.
    """
    if query_style != "mixed":
        return query_style
    stable_hash = int(hashlib.md5(pair_id.encode("utf-8")).hexdigest()[:8], 16) if pair_id else 0
    return "academic" if (stable_hash % 2 == 0) else "real_user"


# 模板路由：先按 style 分轨（academic/real_user），再按模态组合选 prompt
def select_template(pair: Dict, query_style: str = "academic") -> str:
    """Choose the right prompt template based on modality combo, hop distance, and style.

    query_style:
      "academic"  — original dual-evidence PhD persona templates (default, backward-compat)
      "real_user" — natural-language reader templates (5 rotating sub-types)
      "mixed"     — 50 % academic / 50 % real_user, chosen deterministically by pair hash
    """
    if query_style == "mixed":
        query_style = resolve_query_style(query_style, str(pair.get("pair_id", "")))

    if query_style == "real_user":
        # Rotate through 5 real-user sub-types deterministically by stable pair_id hash
        pid = str(pair.get("pair_id", ""))
        stable_hash = int(hashlib.md5(pid.encode("utf-8")).hexdigest()[:8], 16) if pid else 0
        return f"real_user_{_REAL_USER_STYLE_CYCLE[stable_hash % len(_REAL_USER_STYLE_CYCLE)]}"

    # Level 3: 3-step reasoning chain (activated by reasoning_chain_target flag)
    if pair.get("reasoning_chain_target"):
        return "3step_reasoning_chain"

    # academic (default)
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


# prompt 组装总入口——模板 + bridge + enriched + section + persona
# 每层可选注入，缺失时自动降级
def build_prompt(pair: Dict, query_style: str = "academic", use_persona: bool = False) -> str:
    """Build the prompt text for a candidate pair.

    Args:
        pair: Candidate element pair dict.
        query_style: "academic" | "real_user" | "mixed".
        use_persona: If True, replace the "You are …" role line with a
            persona-specific prefix chosen deterministically by pair_id.
            Persona distribution: phd 30%, lazy 25%, careful 20%,
            practitioner 15%, skeptic 10%.
    """
    template_name = select_template(pair, query_style)
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

    # 优先用 enriched_content，fallback 到原始 caption+context
    def _context(elem: Dict) -> str:
        # Prefer enriched content when available (MoDora-style).
        # C1: discard noisy enriched fields before using them.
        enriched = (elem.get("enriched_content", "") or "").strip()
        if _is_noisy_enrichment(enriched):
            enriched = ""
        before = (elem.get("context_before", "") or "")[:300]
        after = (elem.get("context_after", "") or "")[:300]
        parts = []
        if enriched:
            parts.append(f"[Enriched] {enriched}")
        if before:
            parts.append(before)
        if after:
            parts.append(after)
        return " ... ".join(parts) if parts else "(no context)"

    latex_bridge_section = build_latex_bridge_section(pair)
    enriched_section = build_enriched_context_section(pair)
    section_context_section = build_section_context_section(pair)

    # Helper: append enriched section if non-empty, then optionally inject persona
    _persona_text = resolve_persona(str(pair.get("pair_id", ""))) if use_persona else ""

    # 尾部追加所有可选 section（enriched/bridge/architecture/section）
    def _with_enriched(prompt_text: str) -> str:
        if enriched_section:
            prompt_text = prompt_text + "\n\n" + enriched_section
        if section_context_section:
            prompt_text = prompt_text + "\n\n" + section_context_section
        if use_persona and _persona_text:
            prompt_text = inject_persona_prefix(prompt_text, _persona_text)
        return prompt_text

    if template_name == "3step_reasoning_chain":
        # Level 3: 3-step reasoning chain prompt
        # P0: Resolve REAL bridge paragraph text from reference graph
        bridge_parts: List[str] = []

        # Priority 1: reference graph bridge texts (actual LaTeX paragraph context)
        resolved_bridges = resolve_bridge_texts_for_path(pair)
        if resolved_bridges:
            bridge_parts.extend(resolved_bridges)

        # Priority 2: edge_contexts from candidate enrichment
        if not bridge_parts:
            for ec in pair.get("edge_contexts", []):
                t = (ec.get("text", "") or "").strip()
                if t:
                    bridge_parts.append(t)

        # Priority 3: hub_semantic_summary (caption-level fallback)
        if not bridge_parts:
            hub_summary = (pair.get("hub_semantic_summary") or "").strip()
            if hub_summary:
                bridge_parts.append(hub_summary)

        bridge_text = "\n".join(bridge_parts) if bridge_parts else "(bridge paragraph context not available)"

        # P2: Score bridge quality and label it
        bridge_quality = score_bridge_quality(bridge_text)
        if bridge_quality >= 0.6:
            bridge_quality_label = f"HIGH ({bridge_quality:.2f}) — rich descriptive text, suitable for serial reasoning"
        elif bridge_quality >= 0.4:
            bridge_quality_label = f"MEDIUM ({bridge_quality:.2f}) — some descriptive content, ensure causal link is grounded"
        else:
            bridge_quality_label = f"LOW ({bridge_quality:.2f}) — sparse text, you MUST work harder to find a genuine causal link or output empty queries"

        # P1: Build graph path description
        graph_path_desc = build_graph_path_description(pair)

        return _with_enriched(PROMPT_3STEP_REASONING_CHAIN.format(
            elem_a_type=elem_a.get("element_type", "element"),
            elem_a_id=elem_a["element_id"],
            elem_a_caption=(elem_a.get("caption", "") or "")[:400],
            elem_a_context=_context(elem_a),
            elem_a_image_note="[Image provided above]" if elem_a.get("image_path") else "",
            bridge_text=bridge_text[:1000],
            bridge_quality_label=bridge_quality_label,
            elem_b_type=elem_b.get("element_type", "element"),
            elem_b_id=elem_b["element_id"],
            elem_b_caption=(elem_b.get("caption", "") or "")[:400],
            elem_b_context=_context(elem_b),
            elem_b_image_note="[Image provided above]" if elem_b.get("image_path") else "",
            graph_path_description=graph_path_desc,
        ))

    if template_name == "figure_table_1hop":
        return _with_enriched(PROMPT_FIGURE_TABLE_1HOP.format(
            fig_id=fig_elem["element_id"],
            fig_caption=(fig_elem.get("caption", "") or "")[:400],
            fig_context=_context(fig_elem),
            tbl_id=table_elem["element_id"],
            tbl_caption=(table_elem.get("caption", "") or "")[:400],
            tbl_headers=extract_table_headers((table_elem.get("content", "") or ""), max_chars=150),
            tbl_context=_context(table_elem),
            edge_context=edge_text,
            latex_bridge=latex_bridge_section,
        ))
    elif template_name == "figure_table_2hop":
        return _with_enriched(PROMPT_FIGURE_TABLE_2HOP.format(
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
        ))
    elif template_name == "figure_formula":
        return _with_enriched(PROMPT_FIGURE_FORMULA.format(
            fig_id=fig_elem["element_id"],
            fig_caption=(fig_elem.get("caption", "") or "")[:400],
            fig_context=_context(fig_elem),
            formula_id=formula_elem["element_id"],
            formula_variables=extract_formula_variables((formula_elem.get("content", "") or "")[:1200]),
            formula_context=_context(formula_elem),
            edge_context=edge_text,
            latex_bridge=latex_bridge_section,
            architecture_guidance=build_architecture_guidance(pair),
        ))
    elif template_name == "formula_table":
        return _with_enriched(PROMPT_FORMULA_TABLE.format(
            formula_id=formula_elem["element_id"],
            formula_variables=extract_formula_variables((formula_elem.get("content", "") or "")[:1200]),
            formula_context=_context(formula_elem),
            tbl_id=table_elem["element_id"],
            tbl_caption=(table_elem.get("caption", "") or "")[:400],
            tbl_headers=extract_table_headers((table_elem.get("content", "") or ""), max_chars=150),
            tbl_context=_context(table_elem),
            edge_context=edge_text,
            latex_bridge=latex_bridge_section,
        ))
    elif template_name.startswith("real_user_"):
        # Real-user templates use a generic two-element layout
        style_key = template_name[len("real_user_"):]
        ru_template = _REAL_USER_TEMPLATES.get(style_key, PROMPT_REAL_USER_FACTUAL)
        formatted = ru_template.format(
            elem_a_id=elem_a["element_id"],
            elem_a_type=elem_a.get("element_type", "element"),
            elem_a_caption=(elem_a.get("caption", "") or "")[:400],
            elem_a_context=_context(elem_a),
            elem_b_id=elem_b["element_id"],
            elem_b_type=elem_b.get("element_type", "element"),
            elem_b_caption=(elem_b.get("caption", "") or "")[:400],
            elem_b_context=_context(elem_b),
            edge_context=edge_text,
            latex_bridge=latex_bridge_section,
        )
        # Inject formula grounding constraint for figure+formula pairs
        pair_type = pair.get("pair_type", "")
        if pair_type == "figure+formula" and formula_elem:
            formula_vars = extract_formula_variables((formula_elem.get("content", "") or "")[:1200])
            formatted += (
                "\n\n## FORMULA GROUNDING (MANDATORY)"
                "\nOne of the elements above is a mathematical formula. "
                "Your answer MUST explicitly reference at least one specific "
                "symbol, variable name, or mathematical term from the formula "
                f"(e.g. {formula_vars[:120] if formula_vars else 'loss function, gradient, coefficient'}). "
                "An answer that only discusses the visual element without "
                "grounding in the formula's notation will be rejected."
            )
        # Inject formula grounding for formula+table pairs too
        elif pair_type == "formula+table" and formula_elem:
            formula_vars = extract_formula_variables((formula_elem.get("content", "") or "")[:1200])
            formatted += (
                "\n\n## FORMULA GROUNDING (MANDATORY)"
                "\nOne of the elements above is a mathematical formula. "
                "Your answer MUST explicitly reference at least one specific "
                "symbol, variable name, or mathematical term from the formula "
                f"(e.g. {formula_vars[:120] if formula_vars else 'loss function, gradient, coefficient'}). "
                "An answer that only discusses the tabular data without "
                "grounding in the formula's notation will be rejected."
            )
        return _with_enriched(formatted)

    return ""


# ──────────────────────────────────────────────────────────────
# API call
# ──────────────────────────────────────────────────────────────

# SSE 流式收集：逐行拼 content delta，从末尾 chunk 取 token usage
def _collect_company_stream(stream_generator) -> Tuple[str, int, int]:
    """Collect content and token usage from company API SSE stream.

    Returns (text, input_tokens, output_tokens).
    """
    content_parts: List[str] = []
    in_tok, out_tok = 0, 0
    line_count = 0
    debug_lines: List[str] = []

    for line in stream_generator:
        line_count += 1
        raw_repr = repr(line)[:200] if line_count <= 5 else None
        if raw_repr:
            debug_lines.append(raw_repr)

        # Handle bytes
        if isinstance(line, bytes):
            line = line.decode("utf-8", errors="replace")
        line = line.strip() if isinstance(line, str) else str(line).strip()

        if not line or not line.startswith("data: "):
            continue
        data = line[6:].strip()
        if data == "[DONE]":
            continue
        try:
            parsed = json.loads(data)
            # Extract token usage from the final chunk
            if "usage" in parsed and parsed["usage"]:
                in_tok = parsed["usage"].get("prompt_tokens", 0)
                out_tok = parsed["usage"].get("completion_tokens", 0)
            choices = parsed.get("choices") or []
            if not choices:
                continue
            delta = choices[0].get("delta", {})
            if c := delta.get("content"):
                content_parts.append(c)
        except (json.JSONDecodeError, KeyError, IndexError):
            continue

    text = "".join(content_parts)
    if not text and debug_lines:
        print(f"\n  [DEBUG] stream had {line_count} lines, first 5:")
        for dl in debug_lines:
            print(f"    {dl}")
    if in_tok == 0 and out_tok == 0 and content_parts:
        print("  [WARN] No token usage found in stream response. "
              "Ensure stream_options.include_usage is enabled or "
              "wrap_requests_call injects it automatically.")
    return text, in_tok, out_tok


# Global company API config (set from args in main)
_COMPANY_API_URL: str = ""
_COMPANY_API_KEY: str = ""


# 三路分发：anthropic(原生) / openai(SDK) / company(yunwu.ai SSE)
def call_api(
    client: Any,
    model: str,
    prompt: str,
    images: List[Optional[Tuple[str, str]]],
    provider: str = "anthropic",
) -> Tuple[Optional[str], int, int]:
    """Call provider API. Returns (text, input_tokens, output_tokens)."""
    if provider == "openai":
        content: List[Dict[str, Any]] = []
        for img in images:
            if img is None:
                continue
            b64, mime = img
            content.append({
                "type": "image_url",
                "image_url": {"url": f"data:{mime};base64,{b64}"},
            })
        content.append({"type": "text", "text": prompt})

        r = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": content},
            ],
            max_tokens=1536,
            temperature=0.4,
        )
        msg = r.choices[0].message.content if r.choices else ""
        if isinstance(msg, list):
            text = "".join(part.get("text", "") for part in msg if isinstance(part, dict))
        else:
            text = str(msg or "")
        in_tok = int(getattr(getattr(r, "usage", None), "prompt_tokens", 0) or 0)
        out_tok = int(getattr(getattr(r, "usage", None), "completion_tokens", 0) or 0)
        return text, in_tok, out_tok

    if provider == "company":
        from local_api_logger import wrap_requests_call

        # Build OpenAI-compatible content array (yunwu.ai is OpenAI-compat)
        user_content: List[Dict[str, Any]] = []
        for img in images:
            if img is None:
                continue
            b64, mime = img
            user_content.append({
                "type": "image_url",
                "image_url": {"url": f"data:{mime};base64,{b64}"},
            })
        user_content.append({"type": "text", "text": prompt})

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {_COMPANY_API_KEY}",
        }
        payload = {
            "model": model,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_content},
            ],
            "max_tokens": 1536,
            "temperature": 0.4,
            "stream": True,
            "stream_options": {"include_usage": True},
        }

        stream = wrap_requests_call(
            model=model,
            url=_COMPANY_API_URL,
            headers=headers,
            payload=payload,
            user="l1_dual_evidence",
            verify=False,
        )
        text, in_tok, out_tok = _collect_company_stream(stream)
        return text, in_tok, out_tok

    # Default: anthropic
    content_aa: List[Dict[str, Any]] = []
    for img in images:
        if img is not None:
            b64, mime = img
            content_aa.append({
                "type": "image",
                "source": {"type": "base64", "media_type": mime, "data": b64},
            })
    content_aa.append({"type": "text", "text": prompt})

    r = client.messages.create(
        model=model,
        system=SYSTEM_PROMPT,
        max_tokens=1536,
        temperature=0.4,
        messages=[{"role": "user", "content": content_aa}],
    )
    return (
        r.content[0].text,
        r.usage.input_tokens,
        r.usage.output_tokens,
    )


# 从 LLM 输出提取 JSON：先尝试整体 loads，再用括号配对找首个 {} 对象
def parse_json(txt: Optional[str]) -> Optional[Dict[str, Any]]:
    def _extract_first_json_object(text: str) -> Optional[str]:
        """Extract first balanced JSON object from mixed text."""
        for start_idx, ch in enumerate(text):
            if ch != "{":
                continue
            depth = 0
            in_string = False
            escape = False
            for i in range(start_idx, len(text)):
                c = text[i]
                if in_string:
                    if escape:
                        escape = False
                    elif c == "\\":
                        escape = True
                    elif c == '"':
                        in_string = False
                    continue
                if c == '"':
                    in_string = True
                elif c == "{":
                    depth += 1
                elif c == "}":
                    depth -= 1
                    if depth == 0:
                        return text[start_idx:i + 1]
        return None

    if not txt:
        return None
    t = txt.strip()
    if t.startswith("```"):
        t = re.sub(r"^```(?:json)?\s*", "", t).strip()
        t = re.sub(r"\s*```$", "", t).strip()
    try:
        return json.loads(t)
    except Exception:
        obj_text = _extract_first_json_object(t)
        if obj_text:
            try:
                return json.loads(obj_text)
            except Exception:
                pass
    return None


# ──────────────────────────────────────────────────────────────
# Path normalization
# ──────────────────────────────────────────────────────────────

REPO_ROOTS = [
    "/home/d00855555/query_myx/data-process-test/",
    "/projects/_hdd/myyyx1/data-process-test/",
    "/projects/myyyx1/data-process-test/",
]


# 将集群绝对路径统一为 data/ 开头的相对路径，写入 JSONL 可移植
def normalize_path(img_path: str) -> str:
    normed = img_path.replace("\\", "/")
    for root in REPO_ROOTS:
        if normed.startswith(root):
            return normed[len(root):]
    # Generic fallback: find '/data/' and keep relative path from there
    idx = normed.find("/data/")
    if idx >= 0:
        return normed[idx + 1:]  # 'data/...'
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
    ap.add_argument(
        "--provider",
        choices=["anthropic", "openai", "company"],
        default="company",
        help="LLM provider backend (company = OpenAI-compat proxy via local_api_logger)",
    )
    ap.add_argument("--model", default=None, help="Model name (default: auto per provider)")
    ap.add_argument(
        "--company-api-url",
        default=os.environ.get("COMPANY_API_URL", ""),
        help="Company API endpoint URL (default: $COMPANY_API_URL from .env)",
    )
    ap.add_argument(
        "--company-api-key",
        default=os.environ.get("COMPANY_API_KEY", ""),
        help="Company API key (default: $COMPANY_API_KEY)",
    )
    ap.add_argument("--limit", type=int, default=0, help="Limit pairs (0=all)")
    ap.add_argument(
        "--shuffle",
        action="store_true",
        default=False,
        help="Shuffle candidate pairs before processing (improves doc diversity when using --limit)",
    )
    ap.add_argument("--delay", type=float, default=0.5, help="Seconds between API calls")
    ap.add_argument("--dry-run", action="store_true", help="Print prompts without calling API")
    ap.add_argument("--no-images", action="store_true", help="Skip sending images")
    ap.add_argument(
        "--query-style",
        choices=["academic", "real_user", "mixed"],
        default="academic",
        help=(
            "Query generation style: "
            "'academic' (default, backward-compat dual-evidence PhD persona), "
            "'real_user' (natural-language reader queries, 5 rotating sub-types), "
            "'mixed' (50%% academic / 50%% real_user by pair hash)"
        ),
    )
    ap.add_argument(
        "--use-persona",
        action="store_true",
        default=False,
        help=(
            "Inject a PersonaHub persona prefix into every prompt, replacing the "
            "default 'You are a PhD student…' role line with a diverse reader "
            "persona from data/personahub_academic_personas.json (50 personas "
            "curated following PersonaHub methodology, Ge et al. 2024, "
            "arXiv:2406.20094). Persona assigned deterministically by pair_id "
            "hash for reproducibility. "
            "Compatible with all --query-style values."
        ),
    )
    ap.add_argument(
        "--reference-graph",
        default="data/latex_reference_graph.json",
        help=(
            "Path to latex_reference_graph.json for bridge paragraph text "
            "resolution (P0 enhancement). Provides actual LaTeX paragraph "
            "context for L3 reasoning chain queries instead of empty placeholders."
        ),
    )
    ap.add_argument(
        "--topology-candidates",
        default="data/latex_hub_multihop_candidates.json",
        help=(
            "Path to topology candidates JSON used for element→label mapping. "
            "Set this to the same section-aware topology family as --reference-graph "
            "to avoid mixing old and new graph materials."
        ),
    )
    ap.add_argument(
        "--section-enrich",
        default="",
        help=(
            "Optional section/subsection enrichment JSON. When provided, any "
            "section nodes present in the candidate path will be summarized into "
            "the prompt as additional section-level semantic context."
        ),
    )
    args = ap.parse_args()

    # Resolve model default per provider
    if args.model is None:
        if args.provider == "anthropic":
            args.model = "claude-sonnet-4-5-20250929"
        elif args.provider == "openai":
            args.model = "gpt-4o"
        else:  # company
            args.model = "gpt-5.4"

    # Load candidates
    cand_path = Path(args.candidates)
    if not cand_path.exists():
        print(f"ERROR: {cand_path} not found. Run select_multihop_candidates.py first.")
        sys.exit(1)
    cand_data = json.loads(cand_path.read_text(encoding="utf-8"))
    pairs = cand_data.get("pairs", [])
    if args.shuffle:
        random.seed(42)  # deterministic shuffle for reproducibility
        random.shuffle(pairs)
    if args.limit > 0:
        pairs = pairs[:args.limit]

    # P0: Load reference graph for bridge paragraph text resolution
    ref_graph_path = Path(args.reference_graph)
    if not ref_graph_path.is_absolute():
        ref_graph_path = PROJECT_ROOT / ref_graph_path
    topo_cand_path = Path(args.topology_candidates)
    if not topo_cand_path.is_absolute():
        topo_cand_path = PROJECT_ROOT / topo_cand_path
    if ref_graph_path.exists():
        print(f"Loading reference graph for bridge text resolution: {ref_graph_path}")
        load_reference_graph_bridge_texts(
            str(ref_graph_path),
            topology_candidates_path=str(topo_cand_path) if topo_cand_path.exists() else "",
        )
        print(f"  Loaded bridge texts for {len(_BRIDGE_TEXT_CACHE)} documents")
        print(f"  Element→label mappings: {len(_ELEMENT_TO_LABELS)} elements")
    else:
        print(f"WARNING: Reference graph not found at {ref_graph_path}")
        print(f"  L3 bridge texts will fall back to hub_semantic_summary")

    section_enrich_path = Path(args.section_enrich) if args.section_enrich else None
    if section_enrich_path:
        if not section_enrich_path.is_absolute():
            section_enrich_path = PROJECT_ROOT / section_enrich_path
        if section_enrich_path.exists():
            load_section_enrichments(str(section_enrich_path))
            print(f"  Loaded section enrichments: {len(_SECTION_ENRICH_CACHE)} sections")
        else:
            print(f"WARNING: section enrich file not found at {section_enrich_path}")

    print(f"\nDual-Evidence L1 Query Generation (v4.5+bridge)")
    print(f"  Candidates: {len(pairs)}")
    print(f"  Provider: {args.provider}")
    print(f"  Model: {args.model}")
    print(f"  Query style: {args.query_style}")
    if args.use_persona:
        _personas = _load_personahub_personas()
        print(f"  PersonaHub:  enabled ({len(_personas)} personas loaded)")
    else:
        print(f"  PersonaHub:  disabled")
    print(f"  Shuffle:     {'enabled (seed=42)' if args.shuffle else 'disabled'}")
    print(f"  Images: {'disabled' if args.no_images else 'enabled'}")
    print(f"  Output: {args.output}")
    print()

    # Initialize client
    client = None
    if not args.dry_run:
        if args.provider == "company":
            global _COMPANY_API_URL, _COMPANY_API_KEY
            _COMPANY_API_KEY = args.company_api_key
            _COMPANY_API_URL = args.company_api_url
            if not _COMPANY_API_URL:
                print("ERROR: Company API URL not set. Use --company-api-url or set COMPANY_API_URL in .env")
                sys.exit(1)
            if not _COMPANY_API_KEY:
                print("ERROR: Company API key not set. Use --company-api-key or set COMPANY_API_KEY in .env")
                sys.exit(1)
            print(f"  Company API: {_COMPANY_API_URL}")
            # client stays None; company provider uses wrap_requests_call directly
        elif args.provider == "openai":
            api_key = os.environ.get("OPENAI_API_KEY")
            if not api_key:
                print("ERROR: OPENAI_API_KEY not set. Run: export $(grep -v '^#' .env | xargs)")
                sys.exit(1)
            from openai import OpenAI

            client = OpenAI(api_key=api_key)
        else:
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
            effective_query_style = resolve_query_style(args.query_style, str(pair.get("pair_id", "")))
            template_name = select_template(pair, effective_query_style)

            # Build prompt
            prompt = build_prompt(pair, effective_query_style, use_persona=args.use_persona)
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
                raw, in_tok, out_tok = call_api(
                    client=client,
                    model=args.model,
                    prompt=prompt,
                    images=images,
                    provider=args.provider,
                )
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
                if raw:
                    print(f"  [DEBUG] raw response ({len(raw)} chars): {raw[:500]}")
                else:
                    print("  [DEBUG] raw response is empty/None")
                parse_failed += 1
                continue

            queries = obj.get("queries", [])
            if not queries:
                print("NO QUERIES")
                parse_failed += 1
                continue

            pair_kept = 0
            pair_failed = 0
            opening_counts: Dict[str, int] = defaultdict(int)
            for q_obj in queries:
                sig = query_opening_signature(q_obj.get("query", ""))
                if sig:
                    opening_counts[sig] += 1
            pair_has_short, pair_has_long = has_length_mix(queries)
            pair_has_length_mix = pair_has_short and pair_has_long

            is_l3 = pair.get("reasoning_chain_target", False)

            for q_obj in queries:
                # Route to the appropriate QC function based on query style
                is_real_user_style = effective_query_style == "real_user"
                _pair_id_str = str(pair.get("pair_id", ""))
                effective_persona_id = resolve_persona_id(_pair_id_str) if args.use_persona else "none"
                effective_persona_text = resolve_persona(_pair_id_str) if args.use_persona else ""
                if is_real_user_style:
                    issues, metrics = qc_real_user_query(q_obj, pair, persona=effective_persona_id)
                else:
                    issues, metrics = qc_multihop_query(q_obj, pair)
                    sig = query_opening_signature(q_obj.get("query", ""))
                    if sig:
                        metrics["opening_signature"] = sig
                        if opening_counts.get(sig, 0) > 1:
                            issues.append("opening_repetition")
                    metrics["pair_has_short_query"] = pair_has_short
                    metrics["pair_has_long_query"] = pair_has_long
                    # Level 3 generates 1 query, not 2 — skip length mix check
                    if not pair_has_length_mix and not is_l3:
                        issues.append("length_mix_missing")
                # Level 3 relaxation (Direction B): demote certain issues to advisory warnings
                if is_l3:
                    L3_SOFT_ISSUES = {
                        "formula_symbol_grounding_missing",
                        "architecture_intent_missing",
                        "missing_reasoning_chain",  # L3 uses reasoning_steps[] not reasoning_chain text
                        # NOTE: pseudo_multihop_parallel REMOVED from soft issues (P3)
                        # L3 queries MUST be serial, not parallel
                    }
                    l3_demoted = [i for i in issues if i in L3_SOFT_ISSUES]
                    if l3_demoted:
                        issues = [i for i in issues if i not in L3_SOFT_ISSUES]
                        metrics["l3_demoted_warnings"] = l3_demoted

                metrics["query_style"] = effective_query_style
                metrics["persona"] = effective_persona_id

                # M4 reasoning depth analysis (advisory, not hard-fail for existing data)
                rd_issues, rd_metrics = qc_reasoning_depth(q_obj, pair, min_depth=3)
                metrics["m4_reasoning_depth"] = rd_metrics.get("reasoning_depth", 2)
                metrics["m4_reasoning_structure"] = rd_metrics.get("reasoning_structure", "parallel")
                metrics["m4_is_true_multihop"] = rd_metrics.get("is_true_multihop", False)
                metrics["m4_causal_link_count"] = rd_metrics.get("causal_link_count", 0)
                metrics["m4_step_deletion_proxy"] = rd_metrics.get("step_deletion_proxy", False)
                if rd_metrics.get("has_explicit_reasoning_steps"):
                    # Hard-fail explicit reasoning_steps that don't pass schema validation
                    issues.extend(rd_issues)
                    metrics["m4_explicit_step_issues"] = rd_issues
                else:
                    # Advisory only for backward-compat dual-evidence queries
                    metrics["m4_depth_advisory_issues"] = rd_issues

                # Normalize image paths
                img_a_path = normalize_path(pair["element_a"].get("image_path", "") or "")
                img_b_path = normalize_path(pair["element_b"].get("image_path", "") or "")

                entry = {
                    "query_id": f"l{'3' if is_l3 else '1'}_de_{doc_id}_{query_idx:04d}",
                    "difficulty_level": 3 if is_l3 else 2,
                    "difficulty_label": "reasoning_chain" if is_l3 else "dual_evidence",
                    "reasoning_chain": q_obj.get("reasoning_chain", ""),
                    "query": q_obj.get("query", ""),
                    "answer": q_obj.get("answer", ""),
                    "doc_id": doc_id,
                    "pair_id": pair["pair_id"],
                    "query_length_bucket": query_length_bucket(q_obj.get("query", "")),
                    "element_ids": [pair["element_a_id"], pair["element_b_id"]],
                    "element_a_type": pair["element_a_type"],
                    "element_b_type": pair["element_b_type"],
                    "pair_type": pair_type,
                    "hop_distance": hop,
                    "path": pair.get("path", []),
                    "reasoning_steps": q_obj.get("reasoning_steps", []),  # M4 Schema 1: explicit reasoning chain steps
                    "reasoning_depth": metrics.get("m4_reasoning_depth", 2),
                    "reasoning_structure": metrics.get("m4_reasoning_structure", "parallel"),
                    "dual_evidence": True,   # v4: renamed from multi_hop (path_len always 2 for single-doc pairs)
                    "cross_modal": True,
                    "query_style": effective_query_style,
                    "persona": effective_persona_id,
                    "image_paths": [p for p in [img_a_path, img_b_path] if p],
                    "quality_tier": pair.get("quality_tier", "unknown"),
                    "query_type": q_obj.get("query_type", "unknown"),
                    "query_intent": metrics.get("query_intent", "objective"),
                    "required_evidence_spans": q_obj.get("required_evidence_spans", []),
                    "visual_anchors": q_obj.get("visual_anchors", []),
                    "text_evidence": q_obj.get("text_evidence", ""),
                    "bridge_quality": score_bridge_quality(
                        "\n".join(resolve_bridge_texts_for_path(pair))
                    ) if is_l3 else None,
                    "graph_path": build_graph_path_description(pair) if is_l3 else None,
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
    print(f"Dual-Evidence L1 Generation Summary (v4.5)")
    print(f"{'='*60}")
    print(f"  Query style:           {args.query_style}")
    print(f"  PersonaHub:            {'enabled' if args.use_persona else 'disabled'}")
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

    log_run(
        script="generate_multihop_l1_queries",
        model=f"{args.provider}:{args.model}",
        purpose=(
            f"L1 dual-evidence query generation — "
            f"{kept}/{query_idx} QC pass from {len(pairs)} pairs → {out_path.name}"
        ),
        input_tokens=total_input_tokens,
        output_tokens=total_output_tokens,
        extra={
            "pairs_processed": len(pairs),
            "queries_written": query_idx,
            "qc_pass":         kept,
            "qc_fail":         qc_failed_count,
            "parse_failures":  parse_failed,
            "output":          str(out_path),
        },
    )


if __name__ == "__main__":
    main()
