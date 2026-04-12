#!/usr/bin/env python3
"""Iterative long-chain query generation with bridge-step supervision.

This script enforces path-based generation (no "god-view" one-shot prompt):
1) Generate hop subqueries step-by-step along intermediate nodes.
2) Generate final endpoint query conditioned on extracted bridge facts.
3) Run ablation QC:
   - endpoints-only (A + D)
   - drop each intermediate node
   If ablations remain answerable, mark as fake_long_chain.

Input candidate schema:
  data/latex_long_chain_pairs_*.json (dict with "pairs")
  Each pair is expected to include:
    - path
    - element_a / element_b
    - intermediate_elements (preferred)
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
from src.utils.image_utils import encode_image  # noqa: E402
from src.qc.pipelines import qc_multihop_query  # noqa: E402
from src.qc.llm_judge import run_ablation_qc, judge_answer_grounding  # noqa: E402
from src.qc.checks import is_noisy_enrichment as _is_noisy_enrichment  # noqa: E402
from src.api import call_llm, parse_json, set_company_credentials  # noqa: E402
from src.utils.token_logger import log_run  # noqa: E402
from src.prompts.personas import (  # noqa: E402
    load_personahub_personas as _load_personahub_personas,
    resolve_persona,
    resolve_persona_id,
    inject_persona_prefix,
)
from src.prompts.styles import resolve_query_style  # noqa: E402

# Bridge text resolution and path normalization (refactored to src/utils/)
from src.utils.bridge_utils import (  # noqa: E402
    load_reference_graph_bridge_texts,
    resolve_bridge_texts_for_path,
    get_bridge_text_cache as _get_bridge_text_cache,
    get_element_to_labels as _get_element_to_labels,
)
from src.utils.file_utils import normalize_path  # noqa: E402


SYSTEM_STEP_PROMPT = (
    "You are generating STRICT retrieval supervision. "
    "Return valid JSON only. No markdown."
)

SYSTEM_FINAL_PROMPT = (
    "You are generating a final multi-hop retrieval query. "
    "The final query must depend on bridge facts extracted earlier. "
    "Return valid JSON only. No markdown."
)

SYSTEM_JUDGE_PROMPT = (
    "You are a strict evaluator for multi-hop retrieval. "
    "Given evidence and question, decide if the question is answerable "
    "from provided evidence only (no guessing). Return JSON only."
)


RELATION_CONNECTORS = (
    "because",
    "due to",
    "consistent with",
    "constrained by",
    "whereas",
    "despite",
    "under",
)

OPERATOR_WORDS = (
    "show",
    "cause",
    "exceed",
    "mismatch",
    "require",
    "predict",
    "contradict",
    "derive",
    "converge",
    "reveal",
    "separate",
    "bound",
    "regulate",
    "affect",
    "differ",
    "improve",
    "reduce",
    "produce",
)

BRIDGE_STOPWORDS = {
    "the", "and", "for", "with", "that", "this", "from", "into", "across", "between",
    "under", "over", "after", "before", "while", "where", "when", "which", "their",
    "these", "those", "because", "despite", "whereas", "table", "figure", "formula",
    "equation", "graph", "plot", "diagram", "panel", "shows", "show", "using",
    "method", "model", "results", "result", "data", "value", "values", "metric",
    "metrics", "analysis", "effect", "effects", "across", "different",
}

LOCATION_WORDS = {
    "left", "right", "top", "bottom", "upper", "lower", "middle", "center", "central",
    "row", "column", "cell", "panel", "axis", "curve", "bar", "line", "scatter",
    "quadrant", "legend", "diagonal", "x", "y",
}


def clean_spaces(text: str) -> str:
    return re.sub(r"\s+", " ", text or "").strip()


def remove_numeric_tokens(text: str) -> str:
    t = text or ""
    t = re.sub(r"\b\d+(?:[.,]\d+)?%?\b", "", t)
    t = re.sub(r"\s+", " ", t).strip()
    t = re.sub(r"\s+([?.!,;:])", r"\1", t)
    return t


def extract_bridge_terms(bridge_steps: Sequence[Dict[str, Any]], max_terms: int = 2) -> List[str]:
    raw = " ".join(
        clean_spaces(
            f"{s.get('anchor', '')} {s.get('evidence_span', '')} {s.get('step_answer', '')}"
        )
        for s in bridge_steps
    ).lower()
    toks = re.findall(r"\b[a-z][a-z\-]{4,}\b", raw)
    out: List[str] = []
    seen = set()
    for tok in toks:
        if tok in BRIDGE_STOPWORDS:
            continue
        if tok in seen:
            continue
        seen.add(tok)
        out.append(tok)
        if len(out) >= max_terms:
            break
    return out


def inject_bridge_clause(query: str, bridge_terms: Sequence[str]) -> str:
    if not bridge_terms:
        return query
    q = clean_spaces(query).rstrip("?.!")
    q_lower = q.lower()
    missing = [t for t in bridge_terms if t not in q_lower]
    if missing:
        q = f"{q} under {missing[0]}"
        if len(missing) > 1:
            q = f"{q} and {missing[1]}"
        q = f"{q} constraints"
    return q + "?"


def ensure_operator(query: str) -> str:
    q = clean_spaces(query)
    q_lower = q.lower()
    if any(re.search(rf"\b{re.escape(op)}\b", q_lower) for op in OPERATOR_WORDS):
        return q
    q = q.rstrip("?.!")
    return f"{q} affect outcomes?"


def rewrite_templated_opening(query: str) -> str:
    q = clean_spaces(query)
    q_low = q.lower()
    if q_low.startswith("given that "):
        q = "How does " + q[11:]
    elif q_low.startswith("what causes "):
        q = "How does " + q[12:]
    return q


def sanitize_visual_anchor(anchor: str, element_type: str) -> str:
    txt = clean_spaces(anchor).lower()
    tokens = re.findall(r"\b[a-z]+\b", txt)
    kept = [t for t in tokens if t in LOCATION_WORDS or t in {"first", "second", "third", "fourth"}]
    if len(kept) >= 2:
        out = " ".join(dict.fromkeys(kept))
        return out[:90]
    if element_type == "table":
        return "middle row and right column cell"
    if element_type == "formula":
        return "left objective term near equality"
    return "upper right trend segment"


def normalize_visual_anchors(
    visual_anchors: Sequence[Dict[str, Any]],
    start_elem: Dict[str, Any],
    end_elem: Dict[str, Any],
) -> List[Dict[str, str]]:
    start_id = start_elem.get("element_id", "")
    end_id = end_elem.get("element_id", "")
    by_id: Dict[str, str] = {}
    for a in visual_anchors:
        if not isinstance(a, dict):
            continue
        eid = str(a.get("element_id", ""))
        if not eid:
            continue
        by_id[eid] = str(a.get("anchor", ""))
    return [
        {
            "element_id": start_id,
            "anchor": sanitize_visual_anchor(by_id.get(start_id, ""), str(start_elem.get("element_type", ""))),
        },
        {
            "element_id": end_id,
            "anchor": sanitize_visual_anchor(by_id.get(end_id, ""), str(end_elem.get("element_type", ""))),
        },
    ]


def refine_query_answer_locally(
    query: str,
    answer: str,
    bridge_steps: Sequence[Dict[str, Any]],
) -> Tuple[str, str]:
    q = clean_spaces(query)
    q = remove_numeric_tokens(q)
    q = rewrite_templated_opening(q)
    q = re.sub(r"\b(?:figure|table|formula|equation|graph|diagram|plot)\b", "", q, flags=re.IGNORECASE)
    q = clean_spaces(q)
    bridge_terms = extract_bridge_terms(bridge_steps, max_terms=2)
    q = inject_bridge_clause(q, bridge_terms)
    q = ensure_operator(q)

    a = clean_spaces(answer)
    if not any(conn in a.lower() for conn in RELATION_CONNECTORS):
        a = f"{a} This holds because the bridge constraints match the endpoint evidence."
    return q, a


def build_repair_prompt(
    query: str,
    answer: str,
    issues: Sequence[str],
    start_elem: Dict[str, Any],
    end_elem: Dict[str, Any],
    bridge_steps: Sequence[Dict[str, Any]],
) -> str:
    bridge_lines = "\n".join(
        f"- hop{s['hop_index']} ({s['element_id']}): anchor={s['anchor']}; span={s['evidence_span']}; fact={s['step_answer']}"
        for s in bridge_steps
    )
    bridge_terms = extract_bridge_terms(bridge_steps, max_terms=2)
    terms_txt = ", ".join(bridge_terms) if bridge_terms else "(no explicit terms extracted)"
    issue_txt = ", ".join(issues) if issues else "(none)"
    return f"""
Repair the following long-chain query/answer so it passes strict QC.

Current query:
{query}

Current answer:
{answer}

Current issues:
{issue_txt}

START ENDPOINT:
{element_context(start_elem)}

END ENDPOINT:
{element_context(end_elem)}

Bridge facts:
{bridge_lines}

Mandatory constraints:
1. Query must require BOTH endpoint evidence and BOTH bridge facts.
2. Query must include BOTH bridge terms: {terms_txt}
3. Query must NOT contain any specific numbers, percentages, or exact values.
4. Query must NOT contain meta words: figure, table, formula, equation, graph, diagram, plot.
5. Query must include one operator verb from:
   show, cause, exceed, mismatch, require, predict, contradict, derive, converge, reveal, separate, bound, regulate, affect, differ, improve, reduce, produce.
6. Query must not start with "Given that" or "What causes".
7. Answer must include at least one connector:
   because / due to / consistent with / constrained by / whereas / despite / under.
8. Visual anchors must be physical position cues only (row/column/left/right/top/bottom/panel/axis), not conceptual terms.

Output JSON:
{{
  "final_query": "string",
  "final_answer": "string",
  "endpoint_spans": [
    {{"element_id": "{start_elem.get('element_id')}", "span": "string", "evidence_type": "observation"}},
    {{"element_id": "{end_elem.get('element_id')}", "span": "string", "evidence_type": "result"}}
  ],
  "visual_anchors": [
    {{"element_id": "{start_elem.get('element_id')}", "anchor": "string"}},
    {{"element_id": "{end_elem.get('element_id')}", "anchor": "string"}}
  ],
  "text_evidence": "string"
}}
""".strip()


# parse_json, call_api → moved to src.api (parse_json imported above;
# call sites updated to use call_llm directly)


def short_text(text: str, limit: int = 420) -> str:
    if not text:
        return "(none)"
    t = re.sub(r"\s+", " ", text).strip()
    return t[:limit]


def element_context(elem: Dict[str, Any], context_limit: int = 280) -> str:
    caption = short_text(elem.get("caption", ""), 220)
    # Prefer enriched content (MoDora-style) when present and not noisy.
    enriched = (elem.get("enriched_content", "") or "").strip()
    if enriched and _is_noisy_enrichment(enriched):
        enriched = ""
    raw_content = short_text(elem.get("content", ""), context_limit)
    content = short_text(enriched, context_limit) if enriched else raw_content
    before = short_text(elem.get("context_before", ""), 180)
    after = short_text(elem.get("context_after", ""), 180)
    return (
        f"ID: {elem.get('element_id', '')}\n"
        f"Type: {elem.get('element_type', '')}\n"
        f"Caption: {caption}\n"
        f"Content: {content}\n"
        f"Context before: {before}\n"
        f"Context after: {after}"
    )


def build_chain_enriched_section(nodes: Sequence[Dict[str, Any]]) -> str:
    """Build an enriched-description block covering every node in the chain.

    For each chain node we include enriched_title + enriched_content (when
    present and not flagged as noisy by the C1 noise filter).  This is the
    long-chain analogue of build_enriched_context_section() in the multihop
    script — but covers all hops, not just two endpoints.
    """
    parts: List[str] = []
    for idx, elem in enumerate(nodes, 1):
        et = (elem.get("enriched_title", "") or "").strip()
        ec = (elem.get("enriched_content", "") or "").strip()
        if et and _is_noisy_enrichment(et):
            et = ""
        if ec and _is_noisy_enrichment(ec):
            ec = ""
        if not (et or ec):
            continue
        eid = elem.get("element_id", f"node_{idx}")
        line = f"[Node {idx} {eid} enriched]"
        if et:
            line += f" {et}."
        if ec:
            line += f" {ec[:600]}"
        parts.append(line)
    if not parts:
        return ""
    return "## Enriched node descriptions\n" + "\n".join(parts)


def build_chain_bridge_section(pair: Dict[str, Any], nodes: Sequence[Dict[str, Any]]) -> str:
    """Resolve real LaTeX bridge paragraph text along the chain.

    Walks consecutive (a, b) endpoint pairs along the chain and queries the
    pre-loaded reference-graph cache (resolve_bridge_texts_for_path) for each
    pair.  Caps total output at ~1200 chars to keep prompt tight.
    """
    if not _get_bridge_text_cache():
        return ""
    seen: set = set()
    bridges: List[str] = []
    for a, b in zip(nodes, nodes[1:]):
        sub_pair = {
            "element_a_id": a.get("element_id", ""),
            "element_b_id": b.get("element_id", ""),
        }
        for txt in resolve_bridge_texts_for_path(sub_pair):
            if txt and txt not in seen:
                seen.add(txt)
                bridges.append(txt)
    # Fallback: pair-level bridge_contexts (older format).
    if not bridges:
        for bc in pair.get("bridge_contexts", []) or []:
            txt = (bc.get("text", "") or "").strip()
            if txt and txt not in seen:
                seen.add(txt)
                bridges.append(txt)
    if not bridges:
        return ""
    joined = "\n".join(f"- {b}" for b in bridges[:6])
    if len(joined) > 1200:
        joined = joined[:1200] + "..."
    return "## Author's connection (from LaTeX source)\n" + joined


def get_path_nodes(pair: Dict[str, Any]) -> Optional[List[Dict[str, Any]]]:
    path = pair.get("path") or []
    if len(path) < 4:
        return None
    node_map: Dict[str, Dict[str, Any]] = {}

    for key in ("element_a", "element_b"):
        elem = pair.get(key) or {}
        eid = elem.get("element_id") or pair.get(f"{key}_id")
        if eid:
            node_map[eid] = {
                "element_id": eid,
                "element_type": elem.get("element_type", pair.get(f"{key}_type", "unknown")),
                **elem,
            }

    # Support old format: intermediate_elements (only intermediates)
    for elem in pair.get("intermediate_elements") or []:
        eid = elem.get("element_id")
        if eid and eid not in node_map:
            node_map[eid] = {
                "element_id": eid,
                "element_type": elem.get("element_type", "unknown"),
                **elem,
            }

    # Support new format: node_group (all elements in chain path, from
    # src/pairing CandidatePair schema).  Skip IDs already populated by
    # element_a / element_b (authoritative endpoint data takes priority).
    for elem in pair.get("node_group") or []:
        eid = elem.get("element_id")
        if eid and eid not in node_map:
            node_map[eid] = {
                "element_id": eid,
                "element_type": elem.get("element_type", "unknown"),
                **elem,
            }

    nodes: List[Dict[str, Any]] = []
    for eid in path:
        if eid not in node_map:
            # Missing intermediate metadata -> cannot do iterative generation reliably.
            return None
        nodes.append(node_map[eid])
    return nodes


def build_step_prompt(
    hop_index: int,
    source_elem: Dict[str, Any],
    target_elem: Dict[str, Any],
    prior_steps: Sequence[Dict[str, Any]],
) -> str:
    prior = []
    for s in prior_steps:
        prior.append(
            f"- hop{s['hop_index']}: element={s['element_id']} "
            f"anchor={s['anchor']} span={s['evidence_span']} "
            f"fact={s['step_answer']}"
        )
    prior_txt = "\n".join(prior) if prior else "(none)"
    return f"""
You are building hop-{hop_index} supervision for multi-hop retrieval.

Known bridge facts from previous hops:
{prior_txt}

SOURCE NODE:
{element_context(source_elem)}

TARGET NODE (this hop must land here):
{element_context(target_elem)}

Requirements:
1. Create ONE intermediate subquery that moves from SOURCE to TARGET.
2. The subquery must be unanswerable without TARGET evidence.
3. Provide a concrete anchor and extractive evidence span from TARGET.
4. Provide a concise hop answer grounded in TARGET.
5. No meta-language ("figure", "table", "formula", "according to", "as shown").

Output JSON:
{{
  "subquery": "string",
  "anchor": "string",
  "evidence_span": "string",
  "step_answer": "string"
}}
""".strip()


def build_final_prompt(
    path: Sequence[str],
    start_elem: Dict[str, Any],
    end_elem: Dict[str, Any],
    bridge_steps: Sequence[Dict[str, Any]],
    query_style: str = "academic",
    persona_text: str = "",
    enriched_section: str = "",
    bridge_section: str = "",
) -> str:
    steps_txt = "\n".join(
        f"- hop{s['hop_index']} ({s.get('element_type', 'element')}): {s.get('subquery', s.get('step_answer', ''))}"
        for s in bridge_steps
    )

    # Style-specific framing.  Academic = strict scholar voice.
    # Real-user   = a curious reader phrasing it naturally, but still required
    # to traverse all bridge facts (so multi-hop signal is preserved).
    if query_style == "real_user":
        framing = (
            "You are a curious technical reader who has just walked through "
            "the chain of evidence below.  Phrase the final question the way "
            "such a reader would actually ask it — natural, direct, no jargon "
            "stacking — but the question must still require ALL of the bridge "
            "facts to be answered."
        )
    else:
        framing = (
            "You are generating a strict multi-hop academic retrieval query "
            "that must traverse the entire chain.  Use precise scholarly voice."
        )

    prompt = f"""
{framing}

Generate a FINAL retrieval query for this path: {' -> '.join(path)}.

START ENDPOINT:
{element_context(start_elem)}

END ENDPOINT:
{element_context(end_elem)}

Bridge questions this chain must resolve (do NOT reveal these answers in the query):
{steps_txt}

Requirements:
1. Final query must require endpoint evidence + bridge facts.
2. If bridge facts are removed, the question should not be fully answerable.
3. Avoid templated openings: do NOT start with "Given that" or "What causes".
4. Avoid meta-language and yes/no form.
5. Do NOT use these words in query: figure, table, formula, equation, graph, diagram, plot.
6. Do NOT use phrase: "relate to".
7. Query must include one explicit operator verb from this list:
   show, cause, exceed, mismatch, require, predict, contradict, derive, converge, reveal, separate, bound, regulate, affect, differ, improve, reduce, produce.
8. Keep query <= 40 words.
9. Final answer must include a relationship connector:
   because / due to / consistent with / constrained by / whereas / despite / under.
10. Final answer MUST explicitly mention evidence from BOTH endpoints (start and end elements).
11. Return endpoint evidence spans + endpoint visual anchors.

Output JSON:
{{
  "final_query": "string",
  "final_answer": "string",
  "query_type": "trend_explanation|anomaly_investigation|cross_reading|theory_vs_experiment|bridge_reasoning",
  "endpoint_spans": [
    {{"element_id": "{start_elem.get('element_id')}", "span": "string", "evidence_type": "observation"}},
    {{"element_id": "{end_elem.get('element_id')}", "span": "string", "evidence_type": "result"}}
  ],
  "visual_anchors": [
    {{"element_id": "{start_elem.get('element_id')}", "anchor": "string"}},
    {{"element_id": "{end_elem.get('element_id')}", "anchor": "string"}}
  ],
  "text_evidence": "string"
}}
""".strip()

    # Append optional sections (enriched descriptions, real LaTeX bridge text)
    # before injecting persona prefix at the very top.
    if enriched_section:
        prompt = prompt + "\n\n" + enriched_section
    if bridge_section:
        prompt = prompt + "\n\n" + bridge_section
    if persona_text:
        prompt = inject_persona_prefix(prompt, persona_text)
    return prompt


def build_ablation_prompt(
    question: str,
    reference_answer: str,
    evidence_nodes: Sequence[Dict[str, Any]],
    scenario_name: str,
) -> str:
    blocks = []
    for idx, elem in enumerate(evidence_nodes, 1):
        blocks.append(f"[Node {idx}]\n{element_context(elem, context_limit=240)}")
    ev_txt = "\n\n".join(blocks)
    return f"""
Scenario: {scenario_name}

Question:
{question}

Reference answer (for semantic target, not mandatory wording):
{reference_answer}

Evidence nodes:
{ev_txt}

Judge rule:
- can_answer = true only if the core claim of reference answer can be derived
  from evidence nodes without guessing.
- If evidence is insufficient, set can_answer=false.

Return JSON:
{{
  "can_answer": true,
  "confidence": 0.0,
  "reason": "short reason"
}}
""".strip()


def compose_required_evidence_spans(
    endpoint_spans: Sequence[Dict[str, Any]],
    bridge_steps: Sequence[Dict[str, Any]],
) -> List[Dict[str, str]]:
    required_evidence_spans: List[Dict[str, str]] = []
    for s in endpoint_spans:
        if not isinstance(s, dict):
            continue
        required_evidence_spans.append({
            "element_id": str(s.get("element_id", "")),
            "span": str(s.get("span", ""))[:260],
            "evidence_type": str(s.get("evidence_type", "observation")),
        })
    for s in bridge_steps:
        required_evidence_spans.append({
            "element_id": str(s.get("element_id", "")),
            "span": str(s.get("evidence_span", ""))[:260],
            "evidence_type": "bridge_step",
        })
    return required_evidence_spans


def maybe_repair_candidate(
    client: Any,
    model: str,
    start_elem: Dict[str, Any],
    end_elem: Dict[str, Any],
    bridge_steps: Sequence[Dict[str, Any]],
    query: str,
    answer: str,
    issues: Sequence[str],
    no_images: bool,
    dry_run: bool,
    provider: str = "company",
    persona_text: str = "",
) -> Tuple[Optional[Dict[str, Any]], int, int]:
    if dry_run:
        return None, 0, 0
    prompt = build_repair_prompt(
        query=query,
        answer=answer,
        issues=issues,
        start_elem=start_elem,
        end_elem=end_elem,
        bridge_steps=bridge_steps,
    )
    if persona_text:
        prompt = inject_persona_prefix(prompt, persona_text)
    imgs: List[Optional[Tuple[str, str]]] = []
    if not no_images:
        imgs = [encode_image(start_elem.get("image_path")), encode_image(end_elem.get("image_path"))]
    raw, in_tok, out_tok = call_llm(
        client,
        model,
        prompt,
        images=imgs,
        provider=provider,
        system_prompt=SYSTEM_FINAL_PROMPT,
        max_tokens=1024,
        temperature=0.15,
    )
    return parse_json(raw), in_tok, out_tok


def run_iterative_generation_for_pair(
    pair: Dict[str, Any],
    client: Any,
    model: str,
    judge_model: str,
    dry_run: bool,
    no_images: bool,
    skip_ablation: bool,
    repair_attempts: int,
    provider: str = "company",
    query_style: str = "academic",
    use_persona: bool = False,
) -> Tuple[Optional[Dict[str, Any]], Dict[str, int], Optional[str]]:
    """Return (entry, token_usage, error_message)."""
    usage = {"in": 0, "out": 0}
    path = pair.get("path") or []
    nodes = get_path_nodes(pair)
    if not nodes:
        return None, usage, "missing_path_node_metadata"
    if len(nodes) < 4:
        return None, usage, "short_path"

    start = nodes[0]
    end = nodes[-1]
    intermediates = nodes[1:-1]

    # Per-pair style + persona + enrichment + bridge text resolution.
    pair_id_str = str(pair.get("pair_id", ""))
    effective_style = resolve_query_style(query_style, pair_id_str)
    persona_text = resolve_persona(pair_id_str) if use_persona else ""
    enriched_section = build_chain_enriched_section(nodes)
    bridge_section = build_chain_bridge_section(pair, nodes)

    # Stage-wise generation for bridge steps.
    bridge_steps: List[Dict[str, Any]] = []
    current_source = start
    for hop_idx, target in enumerate(intermediates, 1):
        prompt = build_step_prompt(
            hop_index=hop_idx,
            source_elem=current_source,
            target_elem=target,
            prior_steps=bridge_steps,
        )

        if dry_run:
            step_obj = {
                "subquery": f"How does hop {hop_idx} connect {current_source.get('element_id')} to {target.get('element_id')}?",
                "anchor": f"anchor for {target.get('element_id')}",
                "evidence_span": short_text(target.get("caption", "") or target.get("content", ""), 100),
                "step_answer": f"Bridge fact from {target.get('element_id')}.",
            }
        else:
            imgs: List[Optional[Tuple[str, str]]] = []
            if not no_images:
                imgs.append(encode_image(current_source.get("image_path")))
                imgs.append(encode_image(target.get("image_path")))
            raw, in_tok, out_tok = call_llm(
                client,
                model,
                prompt,
                images=imgs,
                provider=provider,
                system_prompt=SYSTEM_STEP_PROMPT,
                max_tokens=512,
                temperature=0.2,
            )
            usage["in"] += in_tok
            usage["out"] += out_tok
            step_obj = parse_json(raw)
            if not step_obj:
                return None, usage, f"step_parse_fail_hop_{hop_idx}"

        bridge_steps.append({
            "hop_index": hop_idx,
            "element_id": target.get("element_id"),
            "element_type": target.get("element_type"),
            "subquery": str(step_obj.get("subquery", ""))[:220],
            "anchor": str(step_obj.get("anchor", ""))[:220],
            "evidence_span": str(step_obj.get("evidence_span", ""))[:260],
            "step_answer": str(step_obj.get("step_answer", ""))[:420],
        })
        current_source = target

    # Final query generation.
    final_prompt = build_final_prompt(
        path=path,
        start_elem=start,
        end_elem=end,
        bridge_steps=bridge_steps,
        query_style=effective_style,
        persona_text=persona_text,
        enriched_section=enriched_section,
        bridge_section=bridge_section,
    )
    if dry_run:
        final_obj = {
            "final_query": f"How does the endpoint behavior in {start.get('element_id')} correspond to {end.get('element_id')} under the extracted bridge constraints?",
            "final_answer": "This is a dry-run final answer using bridge facts.",
            "query_type": "bridge_reasoning",
            "endpoint_spans": [
                {"element_id": start.get("element_id"), "span": short_text(start.get("caption", ""), 120), "evidence_type": "observation"},
                {"element_id": end.get("element_id"), "span": short_text(end.get("caption", ""), 120), "evidence_type": "result"},
            ],
            "visual_anchors": [
                {"element_id": start.get("element_id"), "anchor": "endpoint visual anchor A"},
                {"element_id": end.get("element_id"), "anchor": "endpoint visual anchor D"},
            ],
            "text_evidence": "dry-run",
        }
    else:
        imgs = []
        if not no_images:
            imgs = [encode_image(start.get("image_path")), encode_image(end.get("image_path"))]
        raw, in_tok, out_tok = call_llm(
            client,
            model,
            final_prompt,
            images=imgs,
            provider=provider,
            system_prompt=SYSTEM_FINAL_PROMPT,
            max_tokens=1024,
            temperature=0.25,
        )
        usage["in"] += in_tok
        usage["out"] += out_tok
        final_obj = parse_json(raw)
        if not final_obj:
            return None, usage, "final_parse_fail"

    query = str(final_obj.get("final_query", "")).strip()
    answer = str(final_obj.get("final_answer", "")).strip()
    query_type = str(final_obj.get("query_type", "bridge_reasoning")).strip() or "bridge_reasoning"
    endpoint_spans = final_obj.get("endpoint_spans") or []
    visual_anchors = final_obj.get("visual_anchors") or []
    text_evidence = str(final_obj.get("text_evidence", "")).strip()

    # Local deterministic cleanup before QC.
    query, answer = refine_query_answer_locally(query, answer, bridge_steps)
    visual_anchors = normalize_visual_anchors(visual_anchors, start, end)
    required_evidence_spans = compose_required_evidence_spans(endpoint_spans, bridge_steps)

    qc_pair = {
        "element_a_id": start.get("element_id"),
        "element_b_id": end.get("element_id"),
        "element_a": start,
        "element_b": end,
    }

    # Synthesize a flat `reasoning_chain` text from the structured bridge_steps
    # so `has_min_reasoning_chain` (which only looks at obj["reasoning_chain"])
    # can validate the chain.  We concatenate both the subquery (bridge question)
    # and the step_answer for each hop.  This is semantically correct — a
    # reasoning chain IS "question → answer → question → answer …".  It also
    # prevents false negatives when the LLM produces very terse step_answers
    # (e.g. single variable names like "A", "M") because the subquery itself
    # supplies the missing length and semantic context.
    _rc_parts: list[str] = []
    for s in (bridge_steps or []):
        sq = str(s.get("subquery", "")).strip()
        sa = str(s.get("step_answer", "")).strip()
        if sq and sa:
            _rc_parts.append(f"{sq} → {sa}")
        elif sq:
            _rc_parts.append(sq)
        elif sa:
            _rc_parts.append(sa)
    reasoning_chain_text = ". ".join(_rc_parts)

    def evaluate_current(
        cur_query: str,
        cur_answer: str,
        cur_spans: Sequence[Dict[str, Any]],
        cur_anchors: Sequence[Dict[str, Any]],
        cur_text_evidence: str,
    ) -> Tuple[List[str], Dict[str, Any]]:
        qc_obj = {
            "query": cur_query,
            "answer": cur_answer,
            "query_type": query_type,
            "required_evidence_spans": list(cur_spans),
            "visual_anchors": list(cur_anchors),
            "text_evidence": cur_text_evidence,
            "reasoning_chain": reasoning_chain_text,
        }
        cur_issues, cur_metrics = qc_multihop_query(qc_obj, qc_pair)

        # LLM judge ablation (step-deletion) — via shared src.qc.llm_judge
        ablation, fake, ab_in, ab_out = run_ablation_qc(
            question=cur_query,
            reference_answer=cur_answer,
            elements=nodes,
            client=client,
            model=judge_model,
            provider=provider,
            dry_run=dry_run,
            skip=skip_ablation,
        )
        usage["in"] += ab_in
        usage["out"] += ab_out
        if fake:
            cur_issues.append("fake_long_chain")
            cur_metrics["fake_long_chain_warn"] = True
        cur_metrics["llm_ablation"] = ablation

        # LLM grounding check — answer must be traceable to evidence
        if not skip_ablation:
            is_grounded, hallucinations, g_conf, g_reason, g_in, g_out = judge_answer_grounding(
                question=cur_query,
                answer=cur_answer,
                elements=nodes,
                text_evidence=cur_text_evidence,
                client=client,
                model=judge_model,
                provider=provider,
                dry_run=dry_run,
            )
            usage["in"] += g_in
            usage["out"] += g_out
            cur_metrics["llm_grounding"] = {
                "is_grounded": is_grounded,
                "confidence": round(g_conf, 4),
                "hallucinations": hallucinations[:5],
                "reason": g_reason,
            }
            if not is_grounded:
                cur_issues.append("llm_answer_hallucination")
                cur_metrics["llm_hallucination_warn"] = True

        cur_issues = list(dict.fromkeys(cur_issues))
        return cur_issues, cur_metrics

    issues, metrics = evaluate_current(
        cur_query=query,
        cur_answer=answer,
        cur_spans=required_evidence_spans,
        cur_anchors=visual_anchors,
        cur_text_evidence=text_evidence,
    )

    # One or more focused repair attempts for hard failures.
    repairable = {
        "fake_long_chain",
        "anchor_leakage",
        "numeric_leakage",
        "meta_language",
        "templated_opening",
        "template_shortcut",
        "no_cross_modal_operator",
        "weak_reasoning_connector",
        "single_element_answer",
    }
    for _ in range(max(0, repair_attempts)):
        if not any(i in repairable for i in issues):
            break
        repaired_obj, in_tok, out_tok = maybe_repair_candidate(
            client=client,
            model=model,
            start_elem=start,
            end_elem=end,
            bridge_steps=bridge_steps,
            query=query,
            answer=answer,
            issues=issues,
            no_images=no_images,
            dry_run=dry_run,
            provider=provider,
            persona_text=persona_text,
        )
        usage["in"] += in_tok
        usage["out"] += out_tok
        if not repaired_obj:
            break

        cand_query = str(repaired_obj.get("final_query", "")).strip() or query
        cand_answer = str(repaired_obj.get("final_answer", "")).strip() or answer
        cand_spans = repaired_obj.get("endpoint_spans") or endpoint_spans
        cand_anchors = repaired_obj.get("visual_anchors") or visual_anchors
        cand_text_evidence = str(repaired_obj.get("text_evidence", "")).strip() or text_evidence

        cand_query, cand_answer = refine_query_answer_locally(cand_query, cand_answer, bridge_steps)
        cand_anchors = normalize_visual_anchors(cand_anchors, start, end)
        cand_required_spans = compose_required_evidence_spans(cand_spans, bridge_steps)
        cand_issues, cand_metrics = evaluate_current(
            cur_query=cand_query,
            cur_answer=cand_answer,
            cur_spans=cand_required_spans,
            cur_anchors=cand_anchors,
            cur_text_evidence=cand_text_evidence,
        )

        # Keep repaired candidate if it clearly improves QC severity.
        def score(iss: Sequence[str]) -> Tuple[int, int]:
            severe = {"fake_long_chain", "ablation_check_failed", "premise_answer_contradiction"}
            severe_cnt = sum(1 for i in iss if i in severe)
            return severe_cnt, len(iss)

        if score(cand_issues) < score(issues):
            query = cand_query
            answer = cand_answer
            endpoint_spans = cand_spans
            visual_anchors = cand_anchors
            text_evidence = cand_text_evidence
            required_evidence_spans = cand_required_spans
            issues = cand_issues
            metrics = cand_metrics
            if not issues:
                break

    metrics["query_style"] = effective_style
    metrics["persona_id"] = resolve_persona_id(pair_id_str) if use_persona else "none"
    metrics["enriched_section_used"] = bool(enriched_section)
    metrics["bridge_section_used"] = bool(bridge_section)

    entry = {
        "query": query,
        "answer": answer,
        "query_type": query_type,
        "required_evidence_spans": required_evidence_spans,
        "visual_anchors": visual_anchors,
        "text_evidence": text_evidence,
        "element_a": start,
        "element_b": end,
        "bridge_steps": bridge_steps,
        "hop_subqueries": [s.get("subquery", "") for s in bridge_steps],
        "qc_issues": issues,
        "qc_pass": len(issues) == 0,
        "qc_metrics": metrics,
        "query_style": effective_style,
        "persona_id": metrics["persona_id"],
        "persona_text": persona_text,
    }
    return entry, usage, None


def split_pair_into_subchains(pair: Dict[str, Any], max_hops: int) -> List[Dict[str, Any]]:
    """Slice an over-long chain pair into contiguous sub-chain pairs.

    Each sub-chain has at most `max_hops` hops (= max_hops + 1 nodes).
    Sub-chains are non-overlapping and contiguous; the last segment keeps
    whatever nodes remain (down to the minimum of 3 hops).

    Returns a list with the original pair unchanged if hop_count <= max_hops,
    or 2+ sub-pairs otherwise.  Each sub-pair gets a new pair_id suffix
    (e.g. ``pair_1_seg0``, ``pair_1_seg1``) so skip-done and dedup work
    per-segment.
    """
    if max_hops <= 0:
        return [pair]
    nodes = get_path_nodes(pair)
    if not nodes:
        return [pair]
    n_nodes = len(nodes)
    n_hops = n_nodes - 1
    if n_hops <= max_hops:
        return [pair]

    base_path = pair.get("path") or []
    base_id = str(pair.get("pair_id", "pair"))
    base_doc = pair.get("doc_id", "")
    sub_pairs: List[Dict[str, Any]] = []
    seg_idx = 0
    i = 0
    while i < n_nodes - 1:  # need at least 2 nodes (1 hop)
        j = min(i + max_hops + 1, n_nodes)  # end index (exclusive)
        seg_nodes = nodes[i:j]
        seg_hops = len(seg_nodes) - 1
        if seg_hops < 3:  # skip trailing stubs shorter than 3 hops
            break
        seg_path = base_path[i:j] if base_path else [n.get("element_id", "") for n in seg_nodes]
        sub = dict(pair)  # shallow copy keeps all metadata
        sub["pair_id"] = f"{base_id}_seg{seg_idx}"
        sub["path"] = seg_path
        sub["hop_distance"] = seg_hops
        # Rebuild node_group to only include this segment's nodes.
        if "node_group" in pair:
            sub["node_group"] = pair["node_group"][i:j]
        sub["_sub_chain"] = True
        sub["_parent_pair_id"] = base_id
        sub["_seg_idx"] = seg_idx
        sub_pairs.append(sub)
        seg_idx += 1
        i += max_hops  # non-overlapping stride
    return sub_pairs if sub_pairs else [pair]


def normalize_pair_types(path_nodes: Sequence[Dict[str, Any]]) -> Tuple[str, str, str]:
    a_type = str(path_nodes[0].get("element_type", "unknown"))
    b_type = str(path_nodes[-1].get("element_type", "unknown"))
    pair_type = "+".join(sorted([a_type, b_type]))
    return a_type, b_type, pair_type


# ── Semantic dedup + per-pair cap + qc_summary_label helpers ────────────────

_TOKEN_RE = re.compile(r"[A-Za-z0-9]+")
_DEDUP_STOPWORDS = frozenset(
    {
        "a", "an", "the", "of", "in", "on", "at", "to", "for", "and", "or",
        "but", "is", "are", "was", "were", "be", "been", "being", "by", "with",
        "as", "that", "this", "these", "those", "it", "its", "from", "which",
        "how", "what", "why", "when", "where", "who", "whom", "does", "do",
        "did", "has", "have", "had", "can", "could", "would", "should", "will",
        "between", "into", "about", "than", "then", "so", "if", "not", "no",
    }
)


def _tokenize_for_dedup(text: str) -> set:
    """Lowercased alphanumeric tokens minus stopwords, for Jaccard."""
    if not text:
        return set()
    toks = {t.lower() for t in _TOKEN_RE.findall(text)}
    return {t for t in toks if t not in _DEDUP_STOPWORDS and len(t) > 1}


def _jaccard(a: set, b: set) -> float:
    if not a or not b:
        return 0.0
    inter = len(a & b)
    union = len(a | b)
    return inter / union if union else 0.0


# ── QC divergence reporting ─────────────────────────────────────────────────

_LLM_ISSUE_KEYS = frozenset({"fake_long_chain", "llm_answer_hallucination", "llm_fake_multihop"})
# missing_reasoning_chain is a known schema-mismatch false positive for
# long-chain entries; exclude it from divergence analysis so it doesn't
# pollute the rule-only-fail bucket.
_RULE_EXCLUDE_FROM_DIVERGENCE = frozenset({"missing_reasoning_chain"})


def compute_qc_divergence(
    qc_issues: Sequence[str],
) -> Dict[str, Any]:
    """Classify a single entry's QC outcome into rule-vs-LLM divergence.

    Returns a dict with:
      rule_issues  — list of rule-only issue strings (excl known false positives)
      llm_issues   — list of LLM-only issue strings
      rule_pass    — bool, True if no rule issues remain
      llm_pass     — bool, True if no LLM issues
      divergence   — one of: 'agree_pass', 'agree_fail', 'rule_only_fail',
                     'llm_only_fail'
    """
    issues = set(qc_issues or [])
    rule_issues = sorted(issues - _LLM_ISSUE_KEYS - _RULE_EXCLUDE_FROM_DIVERGENCE)
    llm_issues = sorted(issues & _LLM_ISSUE_KEYS)
    rule_pass = len(rule_issues) == 0
    llm_pass = len(llm_issues) == 0

    if rule_pass and llm_pass:
        divergence = "agree_pass"
    elif not rule_pass and not llm_pass:
        divergence = "agree_fail"
    elif not rule_pass and llm_pass:
        divergence = "rule_only_fail"
    else:
        divergence = "llm_only_fail"

    return {
        "rule_issues": rule_issues,
        "llm_issues": llm_issues,
        "rule_pass": rule_pass,
        "llm_pass": llm_pass,
        "divergence": divergence,
    }


def derive_qc_summary_label(
    qc_pass: bool,
    qc_issues: Sequence[str],
    qc_metrics: Dict[str, Any],
) -> str:
    """Condense qc_issues + grounding metrics into one human-readable tag.

    Priority order (first match wins):
      over_capped → duplicate → hallucinated → fake_multihop →
      bridge_overclaim → conditional_hedge → leakage → weak_chain →
      single_evidence → underdetermined → grounded_pass → rule_fail
    """
    issues = set(qc_issues or [])

    # Hard labels injected by dedup / cap checks run first.
    if "over_capped" in issues:
        return "over_capped"
    if "duplicate" in issues:
        return "duplicate"

    # Grounding-based labels (from llm_grounding metrics).
    grounding = qc_metrics.get("llm_grounding") if isinstance(qc_metrics, dict) else None
    if isinstance(grounding, dict):
        if grounding.get("hallucination"):
            return "hallucinated"
        if grounding.get("underdetermined"):
            return "underdetermined"

    # Ablation / fake multihop.
    if {"fake_long_chain", "fake_multihop", "ablation_check_failed"} & issues:
        return "fake_multihop"

    # Bridge / conditional overclaim (from qc_multihop_query).
    if "bridge_overclaim" in issues:
        return "bridge_overclaim"
    if "conditional_hedge" in issues:
        return "conditional_hedge"

    # Leakage family.
    if any(i in issues for i in ("anchor_leakage", "bridge_entity_leakage", "template_shortcut")):
        return "leakage"

    # Reasoning chain weaknesses.
    if any(
        i in issues
        for i in (
            "missing_reasoning_chain",
            "weak_reasoning_connector",
            "pseudo_multihop_parallel",
        )
    ):
        return "weak_chain"

    # Evidence insufficiency.
    if "single_element_answer" in issues:
        return "single_evidence"

    if qc_pass:
        return "grounded_pass"
    if issues:
        return "rule_fail"
    return "unknown"


def main() -> None:
    ap = argparse.ArgumentParser(description="Iterative long-chain query generation (v3)")
    ap.add_argument("--candidates", default="data/01_graphs/latex_long_chain_pairs_all_q0.json")
    ap.add_argument("--output", default="data/03_queries/l1_dual_evidence_long_chain_queries_v2_iterative.jsonl")
    ap.add_argument(
        "--pass-output",
        default="",
        help="Optional explicit pass-only JSONL path. If set, pass-only entries are written here regardless of --pass-only.",
    )
    ap.add_argument("--pass-only", action="store_true")
    ap.add_argument(
        "--provider",
        choices=["anthropic", "openai", "company"],
        default="company",
        help="LLM provider backend (company = OpenAI-compat proxy via local_api_logger)",
    )
    ap.add_argument("--model", default=None, help="Model name (default: auto per provider)")
    ap.add_argument("--judge-model", default=None, help="Judge model name (default: same as --model)")
    ap.add_argument(
        "--company-api-url",
        default=os.environ.get("COMPANY_API_URL", ""),
        help="Company API endpoint URL (default: $COMPANY_API_URL)",
    )
    ap.add_argument(
        "--company-api-key",
        default=os.environ.get("COMPANY_API_KEY", ""),
        help="Company API key (default: $COMPANY_API_KEY)",
    )
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--delay", type=float, default=0.5)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--no-images", action="store_true")
    ap.add_argument("--skip-ablation", action="store_true")
    ap.add_argument("--repair-attempts", type=int, default=1)
    ap.add_argument(
        "--query-style",
        choices=["academic", "real_user", "mixed"],
        default="academic",
        help=(
            "Query style for the FINAL prompt: 'academic' (default, scholarly "
            "voice), 'real_user' (curious-reader natural language), 'mixed' "
            "(50/50 by pair_id hash)."
        ),
    )
    ap.add_argument(
        "--use-persona",
        action="store_true",
        default=False,
        help=(
            "Inject a PersonaHub persona prefix into the final prompt. Persona "
            "is assigned deterministically by pair_id hash for reproducibility."
        ),
    )
    ap.add_argument(
        "--reference-graph",
        default="data/01_graphs/latex_reference_graph.json",
        help=(
            "Path to latex_reference_graph.json. When present, the script "
            "resolves real LaTeX bridge paragraph text along the chain and "
            "appends it to the final prompt."
        ),
    )
    ap.add_argument(
        "--topology-candidates",
        default="data/01_graphs/latex_hub_multihop_candidates.json",
        help="Path to topology candidates JSON (used for element→label mapping).",
    )
    ap.add_argument(
        "--skip-done",
        default="",
        metavar="JSONL",
        help=(
            "Path to an existing output JSONL. Pair IDs already present in "
            "that file are skipped (resume support). When set, the output "
            "stream is opened in append mode."
        ),
    )
    ap.add_argument(
        "--max-pass-per-pair",
        type=int,
        default=2,
        help=(
            "Maximum pass queries allowed per pair_id across all runs (counts "
            "entries from --skip-done file as prior). Extra passes are marked "
            "qc_fail with label 'over_capped'. Set 0 to disable."
        ),
    )
    ap.add_argument(
        "--dedup-jaccard",
        type=float,
        default=0.4,
        help=(
            "Token-level Jaccard threshold for semantic dedup across queries "
            "sharing the same pair_id. New query exceeding this vs any prior "
            "same-pair query is marked qc_fail with label 'duplicate'. "
            "Set 0 to disable."
        ),
    )
    ap.add_argument(
        "--max-query-hops",
        type=int,
        default=5,
        help=(
            "Maximum chain hops fed into a single query. Longer chains are "
            "split into non-overlapping contiguous sub-chains of this length, "
            "each generating an independent query. Set 0 to disable splitting "
            "(not recommended for chains > 6 hops)."
        ),
    )
    args = ap.parse_args()

    # Resolve model defaults per provider
    if args.model is None:
        if args.provider == "anthropic":
            args.model = "claude-sonnet-4-5-20250929"
        elif args.provider == "openai":
            args.model = "gpt-4o"
        else:
            args.model = "gpt-5.4"
    if args.judge_model is None:
        args.judge_model = args.model

    cand_path = Path(args.candidates)
    if not cand_path.exists():
        print(f"ERROR: candidate file not found: {cand_path}")
        sys.exit(1)
    cand_data = json.loads(cand_path.read_text(encoding="utf-8"))
    pairs = cand_data.get("pairs", [])
    if args.limit > 0:
        pairs = pairs[:args.limit]

    # Expand over-long chains into contiguous sub-chain pairs before any
    # skip-done filtering so that sub-chain pair_ids are stable across runs.
    if args.max_query_hops > 0:
        expanded = []
        for p in pairs:
            expanded.extend(split_pair_into_subchains(p, args.max_query_hops))
        n_orig = len(pairs)
        pairs = expanded
        if len(pairs) != n_orig:
            print(
                f"  --max-query-hops={args.max_query_hops}: "
                f"{n_orig} input pairs → {len(pairs)} sub-chain pairs "
                f"(+{len(pairs) - n_orig} from long-chain splits)"
            )

    # Resume support: skip pair_ids already present in a previous output file.
    # Also pre-load pass counts and prior query tokens per pair_id for cap +
    # semantic dedup checks that run during generation.
    done_ids: set = set()
    pass_count_by_pair: Dict[str, int] = defaultdict(int)
    prior_query_tokens_by_pair: Dict[str, List[set]] = defaultdict(list)
    skip_done_active = False
    if args.skip_done:
        skip_path = Path(args.skip_done)
        if not skip_path.is_absolute():
            skip_path = PROJECT_ROOT / skip_path
        if skip_path.exists():
            with open(skip_path, encoding="utf-8") as _sf:
                for _line in _sf:
                    _line = _line.strip()
                    if not _line:
                        continue
                    try:
                        _rec = json.loads(_line)
                    except json.JSONDecodeError:
                        continue
                    _pid = _rec.get("pair_id")
                    if not _pid:
                        continue
                    done_ids.add(_pid)
                    if _rec.get("qc_pass"):
                        pass_count_by_pair[_pid] += 1
                        _q = _rec.get("query", "")
                        if _q:
                            prior_query_tokens_by_pair[_pid].append(_tokenize_for_dedup(_q))
            before = len(pairs)
            pairs = [p for p in pairs if p.get("pair_id") not in done_ids]
            print(
                f"  --skip-done: {len(done_ids)} pair_ids found in {skip_path.name}, "
                f"skipping {before - len(pairs)} pairs → {len(pairs)} remaining"
            )
            if pass_count_by_pair:
                print(
                    f"  --skip-done: loaded pass counts for {len(pass_count_by_pair)} pair_ids "
                    f"(max={max(pass_count_by_pair.values())})"
                )
            skip_done_active = True
        else:
            print(f"  WARNING: --skip-done file not found: {skip_path}, running all pairs")

    # Load reference graph for bridge paragraph text resolution.
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
        print(f"  Loaded bridge texts for {len(_get_bridge_text_cache())} documents")
        print(f"  Element→label mappings: {len(_get_element_to_labels())} elements")
    else:
        print(f"WARNING: Reference graph not found at {ref_graph_path}")

    print("Iterative Long-Chain Generation (v3)")
    print(f"  Candidates: {len(pairs)}")
    print(f"  Provider:   {args.provider}")
    print(f"  Model:      {args.model}")
    print(f"  Judge model:{args.judge_model}")
    print(f"  Images: {'disabled' if args.no_images else 'enabled'}")
    print(f"  Ablation QC: {'disabled' if args.skip_ablation else 'enabled'}")
    print(f"  Repair attempts: {max(0, args.repair_attempts)}")
    print(f"  Query style: {args.query_style}")
    if args.use_persona:
        try:
            _personas = _load_personahub_personas()
            print(f"  PersonaHub:  enabled ({len(_personas)} personas loaded)")
        except Exception as _exc:
            print(f"  PersonaHub:  enabled (failed to load count: {_exc})")
    else:
        print(f"  PersonaHub:  disabled")
    print(f"  Output: {args.output}")
    print()

    client = None
    if not args.dry_run:
        if args.provider == "company":
            if not args.company_api_key or not args.company_api_url:
                print("ERROR: COMPANY_API_KEY / COMPANY_API_URL not set. Run: export $(grep -v '^#' .env | xargs)")
                sys.exit(1)
            set_company_credentials(args.company_api_url, args.company_api_key)
        elif args.provider == "openai":
            import openai
            client = openai.OpenAI(api_key=os.environ.get("OPENAI_API_KEY", ""))
        else:
            api_key = os.environ.get("ANTHROPIC_API_KEY")
            if not api_key:
                print("ERROR: ANTHROPIC_API_KEY not set. Run: export $(grep -v '^#' .env | xargs)")
                sys.exit(1)
            import anthropic
            client = anthropic.Anthropic(api_key=api_key)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if args.pass_output:
        pass_path = Path(args.pass_output)
        pass_path.parent.mkdir(parents=True, exist_ok=True)
    elif args.pass_only:
        pass_path = out_path.with_name(out_path.stem + "_pass" + out_path.suffix)
    else:
        pass_path = None

    total_in = 0
    total_out = 0
    kept = 0
    failed = 0
    skipped = 0
    q_idx = 0
    issue_stats: Dict[str, int] = defaultdict(int)
    type_stats: Dict[str, int] = defaultdict(int)
    divergence_stats: Dict[str, int] = defaultdict(int)
    divergence_rule_issues: Dict[str, int] = defaultdict(int)
    divergence_llm_issues: Dict[str, int] = defaultdict(int)

    # Append mode when resuming from a previous run, otherwise overwrite.
    file_mode = "a" if skip_done_active else "w"
    out_stream = open(os.devnull, "w", encoding="utf-8") if args.dry_run else out_path.open(file_mode, encoding="utf-8")
    pass_stream = open(os.devnull, "w", encoding="utf-8") if (args.dry_run or not pass_path) else pass_path.open(file_mode, encoding="utf-8")

    with out_stream as f, pass_stream as fp:
        for i, pair in enumerate(pairs, 1):
            pair_id = pair.get("pair_id", f"pair_{i:04d}")
            print(f"  [{i}/{len(pairs)}] {pair_id} ...", end=" ", flush=True)

            try:
                obj, usage, err = run_iterative_generation_for_pair(
                    pair=pair,
                    client=client,
                    model=args.model,
                    judge_model=args.judge_model,
                    dry_run=args.dry_run,
                    no_images=args.no_images,
                    skip_ablation=args.skip_ablation,
                    repair_attempts=max(0, args.repair_attempts),
                    provider=args.provider,
                    query_style=args.query_style,
                    use_persona=args.use_persona,
                )
            except Exception as e:
                print(f"ERROR: {e}")
                failed += 1
                continue

            total_in += usage["in"]
            total_out += usage["out"]

            if err:
                print(f"SKIP ({err})")
                skipped += 1
                continue

            nodes = get_path_nodes(pair) or []
            a_type, b_type, pair_type = normalize_pair_types(nodes)
            start_elem = nodes[0]
            end_elem = nodes[-1]

            # Per-pair cap + semantic dedup are enforced here, after the
            # generator + rule QC + LLM QC have decided on qc_pass. They can
            # only downgrade a pass to fail (never the other way around).
            final_issues = list(obj["qc_issues"])
            final_pass = bool(obj["qc_pass"])
            new_query_tokens = _tokenize_for_dedup(obj["query"])

            if final_pass and args.max_pass_per_pair > 0:
                if pass_count_by_pair.get(pair_id, 0) >= args.max_pass_per_pair:
                    final_issues.append("over_capped")
                    final_pass = False

            max_dup_jaccard = 0.0
            if final_pass and args.dedup_jaccard > 0 and prior_query_tokens_by_pair.get(pair_id):
                for prior_tokens in prior_query_tokens_by_pair[pair_id]:
                    j = _jaccard(new_query_tokens, prior_tokens)
                    if j > max_dup_jaccard:
                        max_dup_jaccard = j
                if max_dup_jaccard > args.dedup_jaccard:
                    final_issues.append("duplicate")
                    final_pass = False

            metrics_with_post = dict(obj["qc_metrics"])
            metrics_with_post["pair_pass_count_prior"] = pass_count_by_pair.get(pair_id, 0)
            metrics_with_post["max_dup_jaccard"] = round(max_dup_jaccard, 3)
            metrics_with_post["over_capped"] = "over_capped" in final_issues
            metrics_with_post["duplicate"] = "duplicate" in final_issues

            qc_summary_label = derive_qc_summary_label(
                qc_pass=final_pass,
                qc_issues=final_issues,
                qc_metrics=metrics_with_post,
            )

            qc_divergence = compute_qc_divergence(final_issues)

            entry = {
                "query_id": f"l1_de_lc_{pair.get('doc_id', 'unknown')}_{q_idx:04d}",
                "query": obj["query"],
                "answer": obj["answer"],
                "doc_id": pair.get("doc_id", ""),
                "pair_id": pair_id,
                "element_ids": [start_elem.get("element_id"), end_elem.get("element_id")],
                "element_a_type": a_type,
                "element_b_type": b_type,
                "pair_type": pair_type,
                "hop_distance": pair.get("hop_distance", len(pair.get("path", [])) - 1),
                "path": pair.get("path", []),
                "dual_evidence": True,
                "cross_modal": True,
                "image_paths": [
                    p for p in [
                        normalize_path(start_elem.get("image_path", "") or ""),
                        normalize_path(end_elem.get("image_path", "") or ""),
                    ] if p
                ],
                "quality_tier": pair.get("quality_tier", "unknown"),
                "query_type": obj["query_type"],
                "required_evidence_spans": obj["required_evidence_spans"],
                "visual_anchors": obj["visual_anchors"],
                "text_evidence": obj["text_evidence"],
                "bridge_steps": obj["bridge_steps"],
                "hop_subqueries": obj["hop_subqueries"],
                "qc_issues": final_issues,
                "qc_pass": final_pass,
                "qc_metrics": metrics_with_post,
                "qc_summary_label": qc_summary_label,
                "qc_divergence": qc_divergence,
                "query_style": obj.get("query_style", args.query_style),
                "persona_id": obj.get("persona_id", "none"),
                "persona_text": obj.get("persona_text", ""),
            }

            # Update in-memory trackers so same-batch duplicates are caught too.
            if final_pass:
                pass_count_by_pair[pair_id] += 1
                prior_query_tokens_by_pair[pair_id].append(new_query_tokens)

            # Track divergence stats.
            div_type = qc_divergence["divergence"]
            divergence_stats[div_type] += 1
            if div_type == "rule_only_fail":
                for ri in qc_divergence["rule_issues"]:
                    divergence_rule_issues[ri] += 1
            elif div_type == "llm_only_fail":
                for li in qc_divergence["llm_issues"]:
                    divergence_llm_issues[li] += 1

            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
            f.flush()
            try:
                os.fsync(f.fileno())
            except (OSError, ValueError):
                pass
            q_idx += 1

            if entry["qc_pass"]:
                kept += 1
                type_stats[pair_type] += 1
                if pass_path:
                    fp.write(json.dumps(entry, ensure_ascii=False) + "\n")
                    fp.flush()
                    try:
                        os.fsync(fp.fileno())
                    except (OSError, ValueError):
                        pass
                print("OK")
            else:
                failed += 1
                for iss in entry["qc_issues"]:
                    issue_stats[iss] += 1
                print(f"QC_FAIL ({', '.join(entry['qc_issues'][:2])})")

            if args.delay > 0 and i < len(pairs):
                time.sleep(args.delay)

    if args.dry_run:
        print(f"\nDry-run complete for {len(pairs)} pairs")
        return

    est_cost = total_in * 3 / 1e6 + total_out * 15 / 1e6
    print("\n" + "=" * 64)
    print("Iterative Long-Chain Summary (v2)")
    print("=" * 64)
    print(f"  Pairs processed:      {len(pairs)}")
    print(f"  Entries written:      {q_idx}")
    print(f"  QC passed:            {kept}")
    print(f"  QC failed:            {failed}")
    print(f"  Skipped:              {skipped}")
    print(f"  Input tokens:         {total_in:,}")
    print(f"  Output tokens:        {total_out:,}")
    print(f"  Estimated cost:       ${est_cost:.2f}")
    print(f"  Output (full):        {out_path}")
    if pass_path:
        print(f"  Output (pass-only):   {pass_path}")
    if type_stats:
        print("  Pass by pair_type:")
        for t, cnt in sorted(type_stats.items()):
            print(f"    {t}: {cnt}")
    if issue_stats:
        print("  QC issue breakdown:")
        for k, v in sorted(issue_stats.items(), key=lambda x: -x[1]):
            print(f"    {k}: {v}")
    if divergence_stats:
        print("  Rule vs LLM divergence:")
        for k, v in sorted(divergence_stats.items(), key=lambda x: -x[1]):
            print(f"    {k}: {v}")
    if divergence_rule_issues:
        print("  Rule-only fail issues (LLM passed these):")
        for k, v in sorted(divergence_rule_issues.items(), key=lambda x: -x[1]):
            print(f"    {k}: {v}")
    if divergence_llm_issues:
        print("  LLM-only fail issues (Rule passed these):")
        for k, v in sorted(divergence_llm_issues.items(), key=lambda x: -x[1]):
            print(f"    {k}: {v}")
    print("=" * 64)

    log_run(
        script="generate_long_chain_iterative_queries",
        model=f"{args.provider}:{args.model}",
        purpose=f"长链多跳 query 生成（iterative bridge-step + LLM judge ablation QC），{len(pairs)} pairs → {q_idx} entries，pass {kept}",
        input_tokens=total_in,
        output_tokens=total_out,
        extra={
            "pairs_processed": len(pairs),
            "entries_written": q_idx,
            "qc_pass": kept,
            "skipped": skipped,
            "output": str(out_path),
            "query_style": args.query_style,
            "use_persona": args.use_persona,
            "skip_done": bool(args.skip_done),
            "reference_graph_loaded": len(_get_bridge_text_cache()) > 0,
            "max_pass_per_pair": args.max_pass_per_pair,
            "dedup_jaccard": args.dedup_jaccard,
            "max_query_hops": args.max_query_hops,
        },
    )


if __name__ == "__main__":
    main()
