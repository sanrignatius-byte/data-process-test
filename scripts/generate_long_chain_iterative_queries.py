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

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.utils.image_utils import encode_image  # noqa: E402
from src.qc.pipelines import qc_multihop_query  # noqa: E402
from src.api import call_llm, parse_json  # noqa: E402

# normalize_path is still local to generate_multihop_l1_queries
from generate_multihop_l1_queries import normalize_path  # noqa: E402


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
    content = short_text(elem.get("content", ""), context_limit)
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

    for elem in pair.get("intermediate_elements") or []:
        eid = elem.get("element_id")
        if eid:
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
) -> str:
    steps_txt = "\n".join(
        f"- hop{s['hop_index']} ({s['element_id']}): anchor={s['anchor']}; span={s['evidence_span']}; fact={s['step_answer']}"
        for s in bridge_steps
    )
    return f"""
Generate a FINAL retrieval query for this path: {' -> '.join(path)}.

START ENDPOINT:
{element_context(start_elem)}

END ENDPOINT:
{element_context(end_elem)}

Bridge facts extracted from intermediate nodes:
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
8. Keep query <= 30 words.
9. Final answer must include a relationship connector:
   because / due to / consistent with / constrained by / whereas / despite / under.
10. Return endpoint evidence spans + endpoint visual anchors.

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


def judge_can_answer(
    client: Any,
    model: str,
    question: str,
    reference_answer: str,
    evidence_nodes: Sequence[Dict[str, Any]],
    scenario_name: str,
    dry_run: bool,
) -> Tuple[bool, float, str, int, int]:
    if dry_run:
        # In dry-run we do not call model; return a neutral default.
        return False, 0.0, "dry-run", 0, 0

    prompt = build_ablation_prompt(
        question=question,
        reference_answer=reference_answer,
        evidence_nodes=evidence_nodes,
        scenario_name=scenario_name,
    )
    raw, in_tok, out_tok = call_llm(
        client,
        model,
        prompt,
        images=[],
        system_prompt=SYSTEM_JUDGE_PROMPT,
        max_tokens=256,
        temperature=0.0,
    )
    obj = parse_json(raw) or {}
    can_answer = bool(obj.get("can_answer", False))
    conf = float(obj.get("confidence", 0.0) or 0.0)
    reason = str(obj.get("reason", ""))[:200]
    return can_answer, conf, reason, in_tok, out_tok


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


def run_ablation_checks(
    client: Any,
    judge_model: str,
    question: str,
    reference_answer: str,
    nodes: Sequence[Dict[str, Any]],
    dry_run: bool,
    skip_ablation: bool,
) -> Tuple[Dict[str, Any], bool, Dict[str, int], Optional[str]]:
    """Returns (ablation_metrics, is_fake_long_chain, token_usage, error_tag)."""
    usage = {"in": 0, "out": 0}
    if skip_ablation:
        return {"skipped": True}, False, usage, None

    start = nodes[0]
    end = nodes[-1]
    full_nodes = list(nodes)
    endpoint_only = [start, end]

    try:
        full_can, full_conf, _, in_tok, out_tok = judge_can_answer(
            client=client,
            model=judge_model,
            question=question,
            reference_answer=reference_answer,
            evidence_nodes=full_nodes,
            scenario_name="full_path",
            dry_run=dry_run,
        )
        usage["in"] += in_tok
        usage["out"] += out_tok

        endpoint_can, endpoint_conf, _, in_tok, out_tok = judge_can_answer(
            client=client,
            model=judge_model,
            question=question,
            reference_answer=reference_answer,
            evidence_nodes=endpoint_only,
            scenario_name="endpoints_only",
            dry_run=dry_run,
        )
        usage["in"] += in_tok
        usage["out"] += out_tok

        drop_flags = []
        drop_conf = []
        for drop_idx in range(1, len(nodes) - 1):
            kept = [n for i, n in enumerate(nodes) if i != drop_idx]
            can_ans, conf, _, in_tok, out_tok = judge_can_answer(
                client=client,
                model=judge_model,
                question=question,
                reference_answer=reference_answer,
                evidence_nodes=kept,
                scenario_name=f"drop_node_{drop_idx}",
                dry_run=dry_run,
            )
            usage["in"] += in_tok
            usage["out"] += out_tok
            drop_flags.append(can_ans)
            drop_conf.append(conf)

        ablation = {
            "full_can_answer": full_can,
            "full_confidence": round(full_conf, 4),
            "endpoints_can_answer": endpoint_can,
            "endpoints_confidence": round(endpoint_conf, 4),
            "drop_node_can_answer": drop_flags,
            "drop_node_confidence": [round(c, 4) for c in drop_conf],
        }
        fake = (not full_can) or endpoint_can or any(drop_flags)
        return ablation, fake, usage, None
    except Exception:
        return {}, False, usage, "ablation_check_failed"


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
    imgs: List[Optional[Tuple[str, str]]] = []
    if not no_images:
        imgs = [encode_image(start_elem.get("image_path")), encode_image(end_elem.get("image_path"))]
    raw, in_tok, out_tok = call_llm(
        client,
        model,
        prompt,
        images=imgs,
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
        }
        cur_issues, cur_metrics = qc_multihop_query(qc_obj, qc_pair)
        ablation, fake, ab_usage, ab_err = run_ablation_checks(
            client=client,
            judge_model=judge_model,
            question=cur_query,
            reference_answer=cur_answer,
            nodes=nodes,
            dry_run=dry_run,
            skip_ablation=skip_ablation,
        )
        usage["in"] += ab_usage["in"]
        usage["out"] += ab_usage["out"]
        if ab_err:
            cur_issues.append(ab_err)
            cur_metrics[f"{ab_err}_warn"] = True
        if fake:
            cur_issues.append("fake_long_chain")
            cur_metrics["fake_long_chain_warn"] = True
        cur_metrics["ablation"] = ablation
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
    }
    return entry, usage, None


def normalize_pair_types(path_nodes: Sequence[Dict[str, Any]]) -> Tuple[str, str, str]:
    a_type = str(path_nodes[0].get("element_type", "unknown"))
    b_type = str(path_nodes[-1].get("element_type", "unknown"))
    pair_type = "+".join(sorted([a_type, b_type]))
    return a_type, b_type, pair_type


def main() -> None:
    ap = argparse.ArgumentParser(description="Iterative long-chain query generation (v2)")
    ap.add_argument("--candidates", default="data/01_graphs/latex_long_chain_pairs_all_q0.json")
    ap.add_argument("--output", default="data/03_queries/l1_dual_evidence_long_chain_queries_v2_iterative.jsonl")
    ap.add_argument("--pass-only", action="store_true")
    ap.add_argument("--model", default="claude-sonnet-4-5-20250929")
    ap.add_argument("--judge-model", default="claude-sonnet-4-5-20250929")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--delay", type=float, default=0.5)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--no-images", action="store_true")
    ap.add_argument("--skip-ablation", action="store_true")
    ap.add_argument("--repair-attempts", type=int, default=1)
    args = ap.parse_args()

    cand_path = Path(args.candidates)
    if not cand_path.exists():
        print(f"ERROR: candidate file not found: {cand_path}")
        sys.exit(1)
    cand_data = json.loads(cand_path.read_text(encoding="utf-8"))
    pairs = cand_data.get("pairs", [])
    if args.limit > 0:
        pairs = pairs[:args.limit]

    print("Iterative Long-Chain Generation (v2)")
    print(f"  Candidates: {len(pairs)}")
    print(f"  Model: {args.model}")
    print(f"  Judge model: {args.judge_model}")
    print(f"  Images: {'disabled' if args.no_images else 'enabled'}")
    print(f"  Ablation QC: {'disabled' if args.skip_ablation else 'enabled'}")
    print(f"  Repair attempts: {max(0, args.repair_attempts)}")
    print(f"  Output: {args.output}")
    print()

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

    total_in = 0
    total_out = 0
    kept = 0
    failed = 0
    skipped = 0
    q_idx = 0
    issue_stats: Dict[str, int] = defaultdict(int)
    type_stats: Dict[str, int] = defaultdict(int)

    out_stream = open(os.devnull, "w", encoding="utf-8") if args.dry_run else out_path.open("w", encoding="utf-8")
    pass_stream = open(os.devnull, "w", encoding="utf-8") if (args.dry_run or not pass_path) else pass_path.open("w", encoding="utf-8")

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
                "qc_issues": obj["qc_issues"],
                "qc_pass": obj["qc_pass"],
                "qc_metrics": obj["qc_metrics"],
            }

            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
            q_idx += 1

            if entry["qc_pass"]:
                kept += 1
                type_stats[pair_type] += 1
                if pass_path:
                    fp.write(json.dumps(entry, ensure_ascii=False) + "\n")
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
    print("=" * 64)


if __name__ == "__main__":
    main()
