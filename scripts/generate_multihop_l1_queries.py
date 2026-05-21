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

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.token_logger import log_run
from src.utils.pair_filters import filter_intra_doc_pairs

# ── Shared modules (Phase 1-2 refactor) ──────────────────────────────────────
from src.utils.text_utils import (
    content_tokens as _content_tokens,
    extract_formula_symbol_terms as _extract_formula_symbol_terms,
    extract_formula_variables,
    extract_math_regions as _extract_math_regions,
    extract_table_headers,
    number_tokens as _number_tokens,
)
from src.utils.image_utils import encode_image, _fallback_image_path
from src.api import (
    call_llm,
    collect_company_stream as _collect_company_stream,
    parse_json,
    set_company_credentials,
    get_company_credentials,
)
from src.qc.checks import (
    anchor_leak_jaccard,
    anchor_overlap_tokens,
    anchor_token_copy_count,
    answer_text_evidence_overlap as _answer_text_evidence_overlap,
    check_evidence_spans,
    check_single_element_answer as _check_single_element_answer,
    formula_symbol_hit as _formula_symbol_hit,
    has_architecture_intent,
    has_length_mix,
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
    is_noisy_enrichment as _is_noisy_enrichment,
    is_yes_no_answer,
    is_yes_no_question,
    query_length_bucket,
    query_opening_signature,
    query_word_count,
)
from src.qc.constants import (
    ANCHOR_LEAK_THRESHOLD,
    ANSWER_BALANCE_THRESHOLD,
    ARCHITECTURE_INTENT_TERMS,
    ARCHITECTURE_KEYWORDS,
    BAD_META_PATTERNS,
    CROSS_MODAL_OPERATOR_PATTERNS,
    CROSS_MODAL_OPERATORS,
    ENTITY_AMNESTY_TERMS,
    LONG_QUERY_MIN_WORDS,
    MAX_QUERY_WORDS,
    METRIC_CONCEPT_TERMS,
    MIN_OVERLAP_BY_TYPE,
    QUERY_SHORTCUT_PATTERNS,
    RELATION_CONNECTORS,
    SHORT_QUERY_MAX_WORDS,
    SHORT_QUERY_MIN_WORDS,
    STRUCTURAL_CONCEPT_TERMS,
    TEMPLATE_COLLAPSE_PATTERNS,
    TEMPLATED_QUERY_OPENINGS,
    TEXT_EVIDENCE_OVERLAP_WARN_THRESHOLD,
    YES_NO_STARTERS,
)
from src.qc.pipelines import qc_multihop_query, qc_real_user_query
from src.qc.reasoning import (
    classify_query_intent,
    classify_reasoning_structure,
    qc_reasoning_depth,
)
from src.qc.llm_judge import run_llm_qc

from src.prompts.templates import (
    SYSTEM_PROMPT,
    PROMPT_FIGURE_TABLE_1HOP,
    PROMPT_FIGURE_TABLE_2HOP,
    PROMPT_FIGURE_FORMULA,
    PROMPT_FORMULA_TABLE,
    PROMPT_3STEP_REASONING_CHAIN,
    PROMPT_REAL_USER_FACTUAL,
    PROMPT_REAL_USER_SUMMARY,
    PROMPT_REAL_USER_COMPARISON,
    PROMPT_REAL_USER_HOW_WORKS,
    PROMPT_REAL_USER_WHAT_IF,
    REAL_USER_TEMPLATES as _REAL_USER_TEMPLATES,
    REAL_USER_STYLE_CYCLE as _REAL_USER_STYLE_CYCLE,
)
from src.prompts.personas import (
    load_personahub_personas as _load_personahub_personas,
    resolve_persona_entry as _resolve_persona_entry,
    resolve_persona,
    resolve_persona_id,
    inject_persona_prefix,
)
from src.prompts.styles import (
    resolve_query_style,
    select_template,
)

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
# PersonaHub + Prompt templates + Query styles — now in src.prompts.*
# See imports at top of file.
# ──────────────────────────────────────────────────────────────

# ──────────────────────────────────────────────────────────────
# QC infrastructure — now delegated to src.qc.*
# All QC constants, check functions, pipelines, and reasoning
# classifiers are imported at the top of this file from:
#   src.qc.constants   — thresholds, patterns, vocabulary
#   src.qc.checks      — 25+ atomic check functions
#   src.qc.pipelines   — qc_multihop_query, qc_real_user_query
#   src.qc.reasoning   — classify_*, qc_reasoning_depth
# Inline definitions removed; the imports above provide identical behaviour.
# ──────────────────────────────────────────────────────────────

# ──────────────────────────────────────────────────────────────
# Image encoding — moved to src/utils/image_utils.py
# encode_image() and _fallback_image_path() are imported at the top.
# ──────────────────────────────────────────────────────────────


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
# C1: Enrichment noise filter — moved to src/qc/checks.py
# is_noisy_enrichment() is imported at the top (as _is_noisy_enrichment).
# ──────────────────────────────────────────────────────────────


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


def synthesize_reasoning_chain_text(reasoning_steps: List[Dict[str, Any]]) -> str:
    """Build legacy reasoning_chain text from structured L3 reasoning_steps."""
    parts: List[str] = []
    for step in reasoning_steps:
        claim = str(step.get("produces_claim", "") or "").strip()
        span = str(step.get("evidence_span", "") or "").strip()
        if claim:
            parts.append(claim)
        elif span:
            parts.append(span)
    return " ".join(parts)



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

        prompt = PROMPT_3STEP_REASONING_CHAIN.format(
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
        )
        if query_style == "real_user":
            prompt = prompt.replace(
                "## YOUR TASK\n\n",
                (
                    "## YOUR TASK\n\n"
                    "STYLE VARIANT: Phrase the query like a natural reader's "
                    "question, but keep the exact 3-step reasoning-path JSON "
                    "schema and all grounding requirements below.\n\n"
                ),
                1,
            )
        return _with_enriched(prompt)

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
# API call — moved to src/api/__init__.py
# call_llm(), collect_company_stream(), parse_json(),
# set_company_credentials(), get_company_credentials() are imported at the top.
# ──────────────────────────────────────────────────────────────


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
        default="data/01_graphs/latex_cross_modal_pairs.json",
        help="Input candidate pairs (latex_cross_modal_pairs.json or multihop_l1_candidates.json)",
    )
    ap.add_argument(
        "--output",
        default="data/03_queries/l1_dual_evidence_queries_v3.jsonl",
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
        "--skip-llm-qc",
        action="store_true",
        default=False,
        help="Skip LLM-based QC (ablation + grounding). Rule-based QC still runs.",
    )
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
        default="data/01_graphs/latex_reference_graph.json",
        help=(
            "Path to latex_reference_graph.json for bridge paragraph text "
            "resolution (P0 enhancement). Provides actual LaTeX paragraph "
            "context for L3 reasoning chain queries instead of empty placeholders."
        ),
    )
    ap.add_argument(
        "--topology-candidates",
        default="data/01_graphs/latex_hub_multihop_candidates.json",
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
    ap.add_argument(
        "--skip-done",
        default="",
        metavar="JSONL",
        help=(
            "Path to an existing output .jsonl file. Any pair_id already present "
            "in that file will be skipped (resume / incremental run support). "
            "Useful when a previous run was interrupted mid-way."
        ),
    )
    ap.add_argument(
        "--checkpoint-every",
        type=int,
        default=1,
        help=(
            "Force fsync and write a checkpoint JSON every N generated rows. "
            "Default 1 maximizes durability for long API jobs."
        ),
    )
    ap.add_argument(
        "--allow-cross-doc-candidates",
        action="store_true",
        help=(
            "Experimental Track A only: bypass the strict intra-doc candidate "
            "filter so paragraph-mediated cross-doc pairs can be dry-run or judged."
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
    raw_pairs = cand_data.get("pairs", [])
    if args.allow_cross_doc_candidates:
        pairs = list(raw_pairs)
        print("  strict intra-doc filter: bypassed by --allow-cross-doc-candidates")
    else:
        pairs, intra_stats = filter_intra_doc_pairs(raw_pairs)
        if len(pairs) != len(raw_pairs):
            print(
                "  strict intra-doc filter:"
                f" removed {len(raw_pairs) - len(pairs)} pairs"
                f" (flag={intra_stats.get('drop_cross_doc_flag', 0)},"
                f" mixed={intra_stats.get('drop_mixed_doc_ids', 0)},"
                f" missing_doc={intra_stats.get('drop_missing_doc_ids', 0)})"
            )
    if args.shuffle:
        random.seed(42)  # deterministic shuffle for reproducibility
        random.shuffle(pairs)
    if args.limit > 0:
        pairs = pairs[:args.limit]

    # Resume support: skip pair_ids already present in a previous output file
    if args.skip_done:
        skip_path = Path(args.skip_done)
        if not skip_path.is_absolute():
            skip_path = PROJECT_ROOT / skip_path
        if skip_path.exists():
            import json as _json
            done_ids: set = set()
            with open(skip_path) as _sf:
                for _line in _sf:
                    _line = _line.strip()
                    if _line:
                        try:
                            done_ids.add(_json.loads(_line)["pair_id"])
                        except (KeyError, ValueError):
                            pass
            before = len(pairs)
            pairs = [p for p in pairs if p["pair_id"] not in done_ids]
            print(f"  --skip-done: {len(done_ids)} pair_ids found in {skip_path.name}, "
                  f"skipping {before - len(pairs)} pairs → {len(pairs)} remaining")
        else:
            print(f"  WARNING: --skip-done file not found: {skip_path}, running all pairs")

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
            set_company_credentials(args.company_api_url, args.company_api_key)
            _cmp_url, _cmp_key = get_company_credentials()
            if not _cmp_url:
                print("ERROR: Company API URL not set. Use --company-api-url or set COMPANY_API_URL in .env")
                sys.exit(1)
            if not _cmp_key:
                print("ERROR: Company API key not set. Use --company-api-key or set COMPANY_API_KEY in .env")
                sys.exit(1)
            print(f"  Company API: {_cmp_url}")
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
    checkpoint_path = out_path.with_name(out_path.stem + "_checkpoint.json")

    total_input_tokens = 0
    total_output_tokens = 0
    kept = 0
    qc_failed_count = 0
    parse_failed = 0
    query_idx = 0

    # Stats
    type_stats = defaultdict(int)
    qc_issue_stats = defaultdict(int)

    def _durable_checkpoint(reason: str, last_pair_id: str = "") -> None:
        """Persist progress so API quota loss or preemption cannot lose rows."""
        if args.dry_run:
            return
        try:
            f.flush()
            os.fsync(f.fileno())
            if pass_path and fp and not fp.closed:
                fp.flush()
                os.fsync(fp.fileno())
        except Exception as exc:
            print(f"  [WARN] checkpoint fsync failed: {exc}")
        checkpoint = {
            "output": str(out_path),
            "pass_output": str(pass_path) if pass_path else "",
            "reason": reason,
            "last_pair_id": last_pair_id,
            "queries_written": query_idx,
            "qc_pass": kept,
            "qc_failed": qc_failed_count,
            "parse_failed": parse_failed,
            "input_tokens": total_input_tokens,
            "output_tokens": total_output_tokens,
            "updated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }
        tmp_path = checkpoint_path.with_suffix(checkpoint_path.suffix + ".tmp")
        tmp_path.write_text(json.dumps(checkpoint, ensure_ascii=False, indent=2), encoding="utf-8")
        tmp_path.replace(checkpoint_path)

    # Dry-run should never mutate output files.
    _file_mode = "a" if args.skip_done else "w"
    out_stream = open(os.devnull, "w", encoding="utf-8") if args.dry_run else out_path.open(_file_mode, encoding="utf-8")
    pass_stream = open(os.devnull, "w", encoding="utf-8") if (args.dry_run or not pass_path) else pass_path.open(_file_mode, encoding="utf-8")

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
                raw, in_tok, out_tok = call_llm(
                    client=client,
                    model=args.model,
                    prompt=prompt,
                    images=images,
                    provider=args.provider,
                    system_prompt=SYSTEM_PROMPT,
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
                is_real_user_style = effective_query_style == "real_user" and not is_l3
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

                # LLM QC — 独立于规则 QC 运行（伪多跳和幻觉是规则抓不到的）
                if not args.dry_run and not args.skip_llm_qc:
                    llm_images = []
                    if not args.no_images:
                        from src.utils.image_utils import encode_image as _enc
                        llm_images = [
                            _enc(pair["element_a"].get("image_path")),
                            _enc(pair["element_b"].get("image_path")),
                        ]
                    llm_issues, llm_metrics, llm_in, llm_out = run_llm_qc(
                        obj=q_obj,
                        pair=pair,
                        client=client,
                        model=args.model,
                        provider=args.provider,
                        images=llm_images,
                        dry_run=args.dry_run,
                        skip_ablation=False,
                        skip_grounding=False,
                    )
                    total_input_tokens += llm_in
                    total_output_tokens += llm_out
                    issues.extend(llm_issues)
                    metrics.update(llm_metrics)

                reasoning_steps = q_obj.get("reasoning_steps", []) or []
                reasoning_chain = str(q_obj.get("reasoning_chain", "") or "").strip()
                if is_l3 and not reasoning_chain and reasoning_steps:
                    reasoning_chain = synthesize_reasoning_chain_text(reasoning_steps)

                entry = {
                    "query_id": f"l{'3' if is_l3 else '1'}_de_{doc_id}_{query_idx:04d}",
                    "difficulty_level": 3 if is_l3 else 2,
                    "difficulty_label": "reasoning_chain" if is_l3 else "dual_evidence",
                    "reasoning_chain": reasoning_chain,
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
                    "reasoning_steps": reasoning_steps,  # M4 Schema 1: explicit reasoning chain steps
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
                f.flush()
                # Also write to pass-only file if enabled
                if pass_path and entry["qc_pass"]:
                    fp.write(json.dumps(entry, ensure_ascii=False) + "\n")
                    fp.flush()
                query_idx += 1
                if args.checkpoint_every > 0 and query_idx % args.checkpoint_every == 0:
                    _durable_checkpoint("periodic", str(pair.get("pair_id", "")))

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

        _durable_checkpoint("final")

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
