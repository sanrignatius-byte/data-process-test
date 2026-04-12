#!/usr/bin/env python3
"""方案C pilot v3：长链找路，压缩桥生成，全链做 QC。

核心思路（方案C生成范式 v1）：
  1. discovery path: 用图里的完整路径发现真实多跳可达性
  2. generation chain: 从完整路径里压缩出 1-3 个关键 bridge/intermediate
  3. qc chain: 规则 QC 看 endpoints；LLM QC 用完整链（含 bridge 文本）做必要性验证

相较 v2 的关键变化：
  - Prompt 不再把“长链 = 全量喂给模型”，而是显式区分 full path / compressed chain
  - bridge 可以是中间 element，也可以是 paragraph/section 的结构化桥文本
  - ablation / grounding 使用完整链的 synthetic bridge elements，而不是只看两个端点
  - 接入 log_run()，满足 token 记录铁律

用法:
    source .env
    python scripts/pilot_method_c.py --api-key "$COMPANY_API_KEY" --num-samples 3

输出:
    data/03_queries/pilot_method_c_v3.json
"""

from __future__ import annotations

import argparse
import json
import random
import re
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.api import call_llm, parse_json, set_company_credentials
from src.qc.llm_judge import run_llm_qc
from src.qc.pipelines import qc_multihop_query
from src.utils.image_utils import encode_image
from src.utils.token_logger import log_run


# ── Load enriched pairs ──────────────────────────────────────────────────────

def load_enriched_pairs(
    enriched_path: Optional[str] = None,
    single_doc_only: bool = True,
) -> List[Dict[str, Any]]:
    """加载 enriched hub candidate pairs。"""
    if enriched_path is None:
        enriched_path = str(
            PROJECT_ROOT / "data" / "02_enriched" / "hub_candidates_enriched_v4.json"
        )
    with open(enriched_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    pairs = data.get("pairs", [])
    if single_doc_only:
        def _is_same_doc(p: Dict[str, Any]) -> bool:
            a_doc = p["element_a_id"].rsplit("_", 2)[0]
            b_doc = p["element_b_id"].rsplit("_", 2)[0]
            return a_doc == b_doc
        pairs = [p for p in pairs if _is_same_doc(p)]
    return pairs


def sample_diverse(
    pairs: List[Dict[str, Any]],
    n: int = 10,
    per_doc: int = 2,
) -> List[Dict[str, Any]]:
    """多样性采样：每个 doc 最多 per_doc 个。"""
    by_doc: Dict[str, List[Dict[str, Any]]] = {}
    for p in pairs:
        by_doc.setdefault(p["doc_id"], []).append(p)
    sampled = []
    for _, doc_pairs in by_doc.items():
        random.shuffle(doc_pairs)
        sampled.extend(doc_pairs[:per_doc])
    random.shuffle(sampled)
    return sampled[:n]


# ── Method C view building ───────────────────────────────────────────────────

def _clean_text(text: Any) -> str:
    return " ".join(str(text or "").split()).strip()


def _truncate(text: Any, limit: int = 240) -> str:
    return _clean_text(text)[:limit]


def _elem_desc(elem: Dict[str, Any], max_len: int = 250) -> str:
    """从 enriched element 提取最佳描述。"""
    ec = _clean_text(elem.get("enriched_content"))
    if ec:
        return ec[:max_len]
    cap = _clean_text(elem.get("caption"))
    ctx = _clean_text(elem.get("context_before"))
    raw = _clean_text(elem.get("content"))
    pieces = [p for p in [cap, ctx, raw] if p]
    return " ".join(pieces)[:max_len] if pieces else f"[{elem.get('element_type', 'element')}]"


def _node_type_from_id(node_id: str) -> str:
    if "::p::" in node_id:
        return "paragraph"
    if "::sec::" in node_id:
        return "section"
    if "::subsec::" in node_id:
        return "subsection"
    if "::subsubsec::" in node_id:
        return "subsubsection"
    if "_figure_" in node_id:
        return "figure"
    if "_table_" in node_id:
        return "table"
    if "_formula_" in node_id:
        return "formula"
    return "bridge"


def _edge_context_texts(pair: Dict[str, Any]) -> List[str]:
    texts: List[str] = []
    seen = set()
    for ec in pair.get("edge_contexts", []) or []:
        if isinstance(ec, str):
            text = _clean_text(ec)
        elif isinstance(ec, dict):
            text = _clean_text(
                ec.get("text")
                or ec.get("context")
                or ec.get("context_snippet")
                or ec.get("ref_text")
            )
        else:
            text = ""
        if len(text) >= 20 and text not in seen:
            seen.add(text)
            texts.append(text)
    return texts


def _hub_summary_text(pair: Dict[str, Any]) -> str:
    summary = _clean_text(pair.get("hub_semantic_summary"))
    if not summary:
        return ""
    summary = re.sub(r"\[[A-Z ]+\]\s*", "", summary)
    summary = summary.replace(" | ", " ")
    return summary.strip()


def _make_synthetic_bridge_element(
    doc_id: str,
    bridge_id: str,
    bridge_type: str,
    text: str,
    source_node_id: str,
    title: str,
) -> Dict[str, Any]:
    body = _truncate(text, 800)
    return {
        "element_id": f"{doc_id}_{bridge_id}",
        "element_type": bridge_type,
        "caption": title,
        "content": body,
        "context_before": "",
        "context_after": "",
        "enriched_title": title,
        "enriched_content": body,
        "image_path": None,
        "source_node_id": source_node_id,
        "is_synthetic_bridge": True,
    }


def _bridge_priority(
    node: Dict[str, Any],
    start_type: str,
    end_type: str,
    position: int,
    total_internal: int,
) -> float:
    score = 0.0
    elem = node["element"]
    node_type = str(node.get("element_type", "bridge"))
    if node["kind"] == "element":
        score += 10.0
        if _clean_text(elem.get("enriched_content")):
            score += 4.0
        if elem.get("image_path"):
            score += 1.0
        if node_type not in {start_type, end_type}:
            score += 2.0
    else:
        score += 6.0
        if len(node.get("summary", "")) >= 120:
            score += 2.0
        if node_type in {"paragraph", "section", "subsection", "subsubsection"}:
            score += 1.0

    center = (total_internal + 1) / 2 if total_internal else 1.0
    score += max(0.0, 1.5 - abs(position - center))
    return round(score, 3)


def build_method_c_view(
    pair: Dict[str, Any],
    max_bridge_nodes: int = 2,
) -> Dict[str, Any]:
    """构造方案C的三个视图：

    - full_chain_nodes: discovery 用的完整路径
    - generation_bridges: prompt 真正使用的压缩桥
    - qc_elements: 给 LLM judge 的完整链元素（含 synthetic bridge）
    """
    start = dict(pair["element_a"])
    end = dict(pair["element_b"])
    start_id = pair["element_a_id"]
    end_id = pair["element_b_id"]
    path = list(pair.get("path") or [start_id, end_id])

    elem_by_id: Dict[str, Dict[str, Any]] = {
        start_id: start,
        end_id: end,
    }
    for node in pair.get("node_group", []) or []:
        eid = node.get("element_id")
        if eid:
            elem_by_id.setdefault(eid, node)

    edge_texts = _edge_context_texts(pair)
    hub_summary = _hub_summary_text(pair)
    seed = _clean_text(pair.get("hub_metadata", {}).get("short_query_seed"))
    fallback_bridge_text = edge_texts[0] if edge_texts else (hub_summary or seed)
    if not fallback_bridge_text:
        fallback_bridge_text = "The endpoints are connected through an internal bridge in the document graph."

    full_chain_nodes: List[Dict[str, Any]] = []
    for idx, node_id in enumerate(path):
        if node_id in elem_by_id:
            elem = dict(elem_by_id[node_id])
            full_chain_nodes.append({
                "node_id": node_id,
                "kind": "element",
                "element": elem,
                "element_id": elem.get("element_id", node_id),
                "element_type": elem.get("element_type", "element"),
                "summary": _elem_desc(elem, 240),
            })
            continue

        bridge_type = _node_type_from_id(node_id)
        bridge_text = ""
        if idx > 0 and edge_texts:
            bridge_text = edge_texts[min(idx - 1, len(edge_texts) - 1)]
        if not bridge_text:
            bridge_text = fallback_bridge_text
        title = f"{bridge_type} bridge context"
        synth = _make_synthetic_bridge_element(
            doc_id=pair["doc_id"],
            bridge_id=f"bridge_{idx}",
            bridge_type=bridge_type,
            text=bridge_text,
            source_node_id=node_id,
            title=title,
        )
        full_chain_nodes.append({
            "node_id": node_id,
            "kind": "bridge",
            "element": synth,
            "element_id": synth["element_id"],
            "element_type": synth["element_type"],
            "summary": _truncate(bridge_text, 240),
        })

    if len(full_chain_nodes) <= 2:
        synth = _make_synthetic_bridge_element(
            doc_id=pair["doc_id"],
            bridge_id="bridge_fallback",
            bridge_type="bridge",
            text=fallback_bridge_text,
            source_node_id="fallback",
            title="bridge reasoning context",
        )
        full_chain_nodes = [
            {
                "node_id": start_id,
                "kind": "element",
                "element": start,
                "element_id": start_id,
                "element_type": start.get("element_type", "element"),
                "summary": _elem_desc(start, 240),
            },
            {
                "node_id": "fallback",
                "kind": "bridge",
                "element": synth,
                "element_id": synth["element_id"],
                "element_type": synth["element_type"],
                "summary": _truncate(fallback_bridge_text, 240),
            },
            {
                "node_id": end_id,
                "kind": "element",
                "element": end,
                "element_id": end_id,
                "element_type": end.get("element_type", "element"),
                "summary": _elem_desc(end, 240),
            },
        ]

    internal = full_chain_nodes[1:-1]
    scored_internal: List[Tuple[float, Dict[str, Any]]] = []
    seen_bridge_keys = set()
    for pos, node in enumerate(internal, start=1):
        key = (node["kind"], node.get("element_type", ""), node.get("summary", "")[:160])
        if key in seen_bridge_keys:
            continue
        seen_bridge_keys.add(key)
        scored_internal.append((
            _bridge_priority(
                node,
                start_type=str(start.get("element_type", "")),
                end_type=str(end.get("element_type", "")),
                position=pos,
                total_internal=len(internal),
            ),
            node,
        ))
    scored_internal.sort(key=lambda x: (-x[0], x[1]["element_id"]))

    generation_bridges = [node for _, node in scored_internal[:max(1, max_bridge_nodes)]]
    compressed_chain_nodes = [full_chain_nodes[0]] + generation_bridges + [full_chain_nodes[-1]]

    return {
        "full_path_ids": path,
        "full_chain_nodes": full_chain_nodes,
        "generation_bridges": generation_bridges,
        "compressed_chain_nodes": compressed_chain_nodes,
        "compressed_bridge_count": len(generation_bridges),
        "qc_elements": [node["element"] for node in full_chain_nodes],
        "compression_summary": {
            "full_path_length": len(path),
            "full_internal_nodes": len(internal),
            "compressed_bridge_count": len(generation_bridges),
            "selected_bridge_ids": [node["element_id"] for node in generation_bridges],
            "selected_bridge_types": [node["element_type"] for node in generation_bridges],
        },
    }


def _bridge_prompt_section(method_c_view: Dict[str, Any]) -> str:
    lines = []
    for idx, node in enumerate(method_c_view["generation_bridges"], start=1):
        lines.append(f"BRIDGE {idx} ({node['element_type']})")
        lines.append(f"ID: {node['element_id']}")
        lines.append(node["summary"])
        lines.append("")
    return "\n".join(lines).strip()


def _chain_ids(nodes: Sequence[Dict[str, Any]]) -> str:
    return " -> ".join(node["element_id"] for node in nodes)


# ── Prompt building ──────────────────────────────────────────────────────────

def build_prompt(
    pair: Dict[str, Any],
    method_c_view: Dict[str, Any],
) -> str:
    """用方案C生成范式构建 prompt。"""
    ea = pair["element_a"]
    eb = pair["element_b"]
    hub_meta = pair.get("hub_metadata", {})
    hop = pair.get("hop_distance", 2)
    pair_type = pair.get("pair_type", "unknown")

    desc_a = _elem_desc(ea, 320)
    desc_b = _elem_desc(eb, 320)
    type_a = ea.get("element_type", "element")
    type_b = eb.get("element_type", "element")
    id_a = ea.get("element_id", "")
    id_b = eb.get("element_id", "")

    edge_texts = _edge_context_texts(pair)
    edge_ctx = ""
    if edge_texts:
        edge_ctx = "GRAPH BRIDGE CONTEXTS:\n" + "\n".join(
            f"  - {text[:150]}" for text in edge_texts[:3]
        ) + "\n"

    hub_summary = _hub_summary_text(pair)
    path_hint = ""
    if hub_summary:
        path_hint = (
            "DISCOVERY NOTE (hidden reasoning support, not user-facing):\n"
            f"{hub_summary[:500]}\n"
        )

    seed = _clean_text(hub_meta.get("short_query_seed"))
    seed_hint = ""
    if seed:
        seed_hint = (
            "REFERENCE SEED (rephrase freely, do NOT copy):\n"
            f"\"{seed[:120]}\"\n"
        )

    compressed_chain_ids = _chain_ids(method_c_view["compressed_chain_nodes"])
    bridge_section = _bridge_prompt_section(method_c_view)

    prompt = f"""Generate a natural multi-hop academic QA pair using Method C.

Method C policy:
- The full discovery path was used to FIND a valid route in the graph.
- You must NOT expose the full path directly in the final question.
- Instead, use the compressed bridge chain below to write a genuinely multi-hop query.
- The final question must require Endpoint A + at least one bridge finding + Endpoint B.

=== ENDPOINT A ({type_a}) ===
ID: {id_a}
{desc_a}

=== ENDPOINT B ({type_b}) ===
ID: {id_b}
{desc_b}

=== COMPRESSED GENERATION CHAIN ===
Discovery hop count: {hop}
Pair type: {pair_type}
Full discovery path IDs: {" -> ".join(method_c_view["full_path_ids"])}
Compressed generation chain IDs: {compressed_chain_ids}

{bridge_section}

{edge_ctx}{path_hint}{seed_hint}=== HARD CONSTRAINTS (violating any = instant rejection) ===

1. QUESTION (<= 40 words):
   - Ask about the scientific relationship between Endpoint A and Endpoint B
   - The answer MUST require information from BOTH endpoints and at least one bridge finding
   - Do NOT use meta-language words: "figure", "table", "formula", "equation",
     "graph", "plot", "diagram", "chart", "panel", "subfigure", "column", "row"
   - Do NOT use template openings: "How does X relate to Y", "What is the
     relationship between X and Y", "In what way does X impact Y"
   - Do NOT use bare deictic pronouns without clear referent: "this", "that",
     "these", "those" (at start of question)
   - Do NOT embed specific numbers or values from the elements into the question
   - Use natural, domain-specific language a researcher would actually ask

2. ANSWER (serial chain reasoning):
   - Must be >= 35 words
   - Must explicitly follow endpoint A -> bridge finding(s) -> endpoint B
   - Each step MUST depend on the previous step
   - Must mention evidence from BOTH endpoints
   - Do not mention path IDs or paragraph IDs

3. TEXT_EVIDENCE (supporting excerpt):
   - A passage (50-180 words) that a reader could use to verify the answer
   - Must NOT be a copy of the answer
   - Should combine endpoint evidence with the bridge context in different wording

=== OUTPUT FORMAT (valid JSON, no markdown) ===
{{
  "query": "natural question under 40 words",
  "answer": "serial chain reasoning >= 35 words with explicit dependencies",
  "text_evidence": "supporting passage from paper context, 50-180 words",
  "reasoning_chain": [
    {{"step": 1, "element_id": "{id_a}", "finding": "...", "depends_on": null}},
    {{"step": 2, "element_id": "bridge_1", "finding": "...", "depends_on": 1}},
    {{"step": 3, "element_id": "{id_b}", "finding": "...", "depends_on": 2}}
  ],
  "visual_anchors": [
    {{"element_id": "{id_a}", "anchor_text": "brief anchor <= 60 chars"}},
    {{"element_id": "{id_b}", "anchor_text": "brief anchor <= 60 chars"}}
  ],
  "hop_count": {hop},
  "compressed_bridge_count": {method_c_view["compressed_bridge_count"]},
  "mentions_both_endpoints": true
}}"""
    return prompt.strip()


# ── Generation ───────────────────────────────────────────────────────────────

def generate_qa(prompt: str, model: str) -> Tuple[Optional[str], int, int]:
    """调用公司 API 生成 query/answer。"""
    return call_llm(
        client=None,
        model=model,
        prompt=prompt,
        provider="company",
        system_prompt=(
            "You are an expert at generating Method C multi-hop academic QA. "
            "Return valid JSON only. Follow the constraints exactly."
        ),
        max_tokens=1024,
        temperature=0.3,
        user_tag="pilot_method_c_v3",
    )


# ── Format adapters ──────────────────────────────────────────────────────────

def build_qc_obj(
    parsed: Dict[str, Any],
    enriched_pair: Dict[str, Any],
    method_c_view: Dict[str, Any],
) -> Dict[str, Any]:
    """构造标准 qc_obj。"""
    chain = parsed.get("reasoning_chain", [])

    rc_parts = []
    span_map: Dict[str, str] = {}
    for step in chain:
        step_eid = str(step.get("element_id") or step.get("element") or "").strip()
        finding = str(step.get("finding", "")).strip()
        if finding:
            rc_parts.append(f"Step {step.get('step', '?')}: {finding}")
        if step_eid and finding:
            span_map[step_eid] = finding
    reasoning_chain_text = ". ".join(rc_parts) if rc_parts else parsed.get("answer", "")

    anchors = parsed.get("visual_anchors", [])
    if not anchors or len(anchors) < 2:
        ea = enriched_pair["element_a"]
        eb = enriched_pair["element_b"]
        anchors = [
            {
                "element_id": ea["element_id"],
                "anchor_text": (ea.get("enriched_title") or ea.get("caption") or "")[:60],
            },
            {
                "element_id": eb["element_id"],
                "anchor_text": (eb.get("enriched_title") or eb.get("caption") or "")[:60],
            },
        ]

    # Ensure endpoint spans are always present for rule QC.
    endpoint_defaults = {
        enriched_pair["element_a_id"]: _truncate(_elem_desc(enriched_pair["element_a"], 120), 120),
        enriched_pair["element_b_id"]: _truncate(_elem_desc(enriched_pair["element_b"], 120), 120),
    }
    for endpoint_id, default_span in endpoint_defaults.items():
        span_map.setdefault(endpoint_id, default_span)

    spans = [{"element_id": eid, "span": span} for eid, span in span_map.items()]

    llm_evidence = str(parsed.get("text_evidence", "")).strip()
    answer = parsed.get("answer", "")
    if len(llm_evidence) >= 40 and llm_evidence != answer:
        text_evidence = llm_evidence
    else:
        parts = []
        for node in method_c_view["compressed_chain_nodes"]:
            elem = node["element"]
            ec = _clean_text(elem.get("enriched_content"))
            if ec:
                parts.append(ec[:220])
                continue
            cap = _clean_text(elem.get("caption"))
            raw = _clean_text(elem.get("content"))
            if cap:
                parts.append(cap[:140])
            if raw:
                parts.append(raw[:180])
        text_evidence = " ".join(parts)[:520]

    return {
        "query": parsed.get("query", ""),
        "answer": answer,
        "query_type": "bridge_reasoning",
        "required_evidence_spans": spans,
        "visual_anchors": anchors,
        "text_evidence": text_evidence,
        "reasoning_chain": reasoning_chain_text,
    }


def build_qc_pair(
    enriched_pair: Dict[str, Any],
    method_c_view: Dict[str, Any],
) -> Dict[str, Any]:
    """构造标准 qc_pair，沿用 generate_l1 的 pair schema。"""
    return {
        "element_a_id": enriched_pair["element_a_id"],
        "element_b_id": enriched_pair["element_b_id"],
        "element_a": enriched_pair["element_a"],
        "element_b": enriched_pair["element_b"],
        "intermediate_elements": list(method_c_view["qc_elements"][1:-1]),
        "element_a_type": enriched_pair["element_a_type"],
        "element_b_type": enriched_pair["element_b_type"],
        "pair_type": enriched_pair["pair_type"],
    }

def build_result_method_c_summary(method_c_view: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "full_path_ids": method_c_view["full_path_ids"],
        "compressed_bridge_count": method_c_view["compressed_bridge_count"],
        "compressed_chain_ids": [node["element_id"] for node in method_c_view["compressed_chain_nodes"]],
        "compressed_chain_types": [node["element_type"] for node in method_c_view["compressed_chain_nodes"]],
        "compressed_bridge_summaries": [node["summary"] for node in method_c_view["generation_bridges"]],
        "compression_summary": method_c_view["compression_summary"],
    }


# ── Main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Pilot Method C v3 — discovery path + compressed bridge generation")
    parser.add_argument("--api-url", default="https://az.gptplus5.com/v1/chat/completions")
    parser.add_argument("--api-key", required=True)
    parser.add_argument("--model", default="gpt-4.1")
    parser.add_argument("--judge-model", default="gpt-4.1")
    parser.add_argument("--num-samples", type=int, default=10)
    parser.add_argument("--output", default="data/03_queries/pilot_method_c_v3.json")
    parser.add_argument("--enriched-path", default=None)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--skip-llm-qc", action="store_true")
    parser.add_argument("--min-hop", type=int, default=2)
    parser.add_argument("--max-hop", type=int, default=4)
    parser.add_argument("--max-bridge-nodes", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    set_company_credentials(args.api_url, args.api_key)

    print("=== Pilot Method C v3 — Discovery Path + Compressed Bridge Generation ===\n")

    print("Loading enriched pairs...")
    pairs = load_enriched_pairs(args.enriched_path, single_doc_only=True)
    print(f"  Single-doc enriched pairs: {len(pairs)}")

    filtered = [
        p for p in pairs
        if args.min_hop <= p.get("hop_distance", 0) <= args.max_hop
    ]
    print(f"  After hop filter ({args.min_hop}-{args.max_hop}): {len(filtered)}")

    sampled = sample_diverse(filtered, n=args.num_samples, per_doc=2)
    print(f"  Sampled: {len(sampled)}")

    results = []
    total_gen_in, total_gen_out = 0, 0
    total_qc_in, total_qc_out = 0, 0

    for i, pair in enumerate(sampled):
        print(f"\n{'=' * 60}")
        print(
            f"[{i + 1}/{len(sampled)}] pair_id={pair['pair_id']}, "
            f"doc={pair['doc_id']}, hop={pair['hop_distance']}, type={pair['pair_type']}"
        )
        print(f"  A: {pair['element_a_id']} ({pair['element_a_type']})")
        print(f"  B: {pair['element_b_id']} ({pair['element_b_type']})")

        method_c_view = build_method_c_view(pair, max_bridge_nodes=args.max_bridge_nodes)
        print(
            "  Method C:"
            f" full_path={len(method_c_view['full_path_ids'])} nodes,"
            f" compressed_bridges={method_c_view['compressed_bridge_count']}"
        )
        prompt = build_prompt(pair, method_c_view)

        if args.dry_run:
            print(f"\n--- Prompt (first 900 chars) ---\n{prompt[:900]}...\n")
            results.append({
                "pair_id": pair["pair_id"],
                "method_c": build_result_method_c_summary(method_c_view),
                "prompt": prompt,
                "dry_run": True,
            })
            continue

        try:
            raw_text, gen_in, gen_out = generate_qa(prompt, args.model)
            total_gen_in += gen_in
            total_gen_out += gen_out
            print(f"  Generation: in={gen_in}, out={gen_out}")
        except Exception as e:
            print(f"  [ERROR] Generation failed: {e}")
            results.append({"pair_id": pair["pair_id"], "error": str(e)})
            continue

        parsed = parse_json(raw_text)
        if not parsed:
            print(f"  [ERROR] JSON parse failed, raw: {(raw_text or '')[:120]}...")
            results.append({
                "pair_id": pair["pair_id"],
                "raw_response": raw_text,
                "parse_error": True,
            })
            continue

        query_text = parsed.get("query", "")
        answer_text = parsed.get("answer", "")
        te_text = parsed.get("text_evidence", "")
        print(f"  Q: {query_text[:90]}")
        print(f"  A: {answer_text[:90]}")
        print(f"  TE: {te_text[:90]}")

        qc_obj = build_qc_obj(parsed, pair, method_c_view)
        qc_pair = build_qc_pair(pair, method_c_view)

        rule_issues, rule_metrics = qc_multihop_query(qc_obj, qc_pair)
        print(f"  Rule QC: {rule_issues or '✓ clean'}")

        llm_issues: List[str] = []
        llm_metrics: Dict[str, Any] = {}
        if not args.skip_llm_qc:
            try:
                llm_images = [
                    encode_image(pair["element_a"].get("image_path")),
                    encode_image(pair["element_b"].get("image_path")),
                ]
                llm_issues, llm_metrics, ab_in, ab_out = run_llm_qc(
                    obj=qc_obj,
                    pair=qc_pair,
                    client=None,
                    model=args.judge_model,
                    provider="company",
                    images=llm_images,
                )
                total_qc_in += ab_in
                total_qc_out += ab_out
                ab_view = llm_metrics.get("llm_ablation", {})
                gr_view = llm_metrics.get("llm_grounding", {})
                print(
                    f"  LLM QC: issues={llm_issues or '[]'}, "
                    f"single={ab_view.get('single_element_can_answer', [])}, "
                    f"grounded={gr_view.get('is_grounded', 'n/a')}"
                )
            except Exception as e:
                print(f"  [ERROR] LLM QC failed: {e}")
                llm_metrics = {"llm_qc_error": str(e)[:200]}
        else:
            print("  LLM QC: skipped")

        all_issues = list(dict.fromkeys(rule_issues + llm_issues))
        passed = len(all_issues) == 0
        verdict = "✓ PASS" if passed else "✗ FAIL"
        print(f"  ── VERDICT: {verdict} ── issues={all_issues}")

        results.append({
            "pair_id": pair["pair_id"],
            "pair_info": {
                "doc_id": pair["doc_id"],
                "hop_distance": pair["hop_distance"],
                "pair_type": pair["pair_type"],
                "element_a_id": pair["element_a_id"],
                "element_b_id": pair["element_b_id"],
            },
            "method_c": build_result_method_c_summary(method_c_view),
            "generated": parsed,
            "qc_obj_text_evidence_len": len(qc_obj.get("text_evidence", "")),
            "qc": {
                "passed": passed,
                "all_issues": all_issues,
                "rule_issues": rule_issues,
                "llm_issues": llm_issues,
                "rule_metrics": rule_metrics,
                "llm_metrics": llm_metrics,
            },
            "tokens": {"gen_in": gen_in, "gen_out": gen_out},
        })

        time.sleep(0.5)

    if not args.dry_run:
        gen_results = [r for r in results if "qc" in r]
        n = len(gen_results)
        if n == 0:
            print("\n[WARN] No results with QC")
        else:
            passed = sum(1 for r in gen_results if r["qc"]["passed"])
            print(f"\n{'=' * 60}")
            print("=== SUMMARY ===")
            print(f"Total:  {n}")
            print(f"Passed: {passed} ({100 * passed / n:.1f}%)")
            print(f"Failed: {n - passed}")
            print(f"Gen tokens:  in={total_gen_in}, out={total_gen_out}")
            print(f"QC tokens:   in={total_qc_in}, out={total_qc_out}")
            est = (total_gen_in + total_qc_in) * 2 / 1e6 + (total_gen_out + total_qc_out) * 8 / 1e6
            print(f"Total cost:  ~${est:.4f}")

            issue_counts: Dict[str, int] = {}
            for r in gen_results:
                for iss in r["qc"]["all_issues"]:
                    issue_counts[iss] = issue_counts.get(iss, 0) + 1
            if issue_counts:
                print("\nIssue breakdown:")
                for iss, cnt in sorted(issue_counts.items(), key=lambda x: -x[1]):
                    print(f"  {iss}: {cnt}/{n} ({100 * cnt / n:.0f}%)")

            te_lens = [r.get("qc_obj_text_evidence_len", 0) for r in gen_results]
            if te_lens:
                print(
                    f"\ntext_evidence lengths: min={min(te_lens)}, "
                    f"max={max(te_lens)}, avg={sum(te_lens) / len(te_lens):.0f}"
                )

            overlaps = [r["qc"]["rule_metrics"].get("text_evidence_overlap", -1) for r in gen_results]
            overlaps = [o for o in overlaps if o >= 0]
            if overlaps:
                print(
                    f"text_evidence_overlap: min={min(overlaps):.3f}, "
                    f"max={max(overlaps):.3f}, avg={sum(overlaps) / len(overlaps):.3f}"
                )

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "metadata": {
                    "version": "v3_method_c_compressed_chain",
                    "model": args.model,
                    "judge_model": args.judge_model,
                    "num_samples": len(results),
                    "tokens_gen": {"in": total_gen_in, "out": total_gen_out},
                    "tokens_qc": {"in": total_qc_in, "out": total_qc_out},
                    "skip_llm_qc": args.skip_llm_qc,
                    "min_hop": args.min_hop,
                    "max_hop": args.max_hop,
                    "max_bridge_nodes": args.max_bridge_nodes,
                    "seed": args.seed,
                },
                "results": results,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )
    print(f"\nResults saved to: {output_path}")

    log_run(
        script="pilot_method_c.py",
        model=args.model,
        input_tokens=total_gen_in + total_qc_in,
        output_tokens=total_gen_out + total_qc_out,
        purpose="Scheme C pilot v3 compressed-bridge generation",
        extra={
            "pairs_processed": len(results),
            "queries_written": sum(1 for r in results if "generated" in r),
            "qc_pass": sum(1 for r in results if r.get("qc", {}).get("passed")),
            "qc_fail": sum(1 for r in results if "qc" in r and not r["qc"]["passed"]),
            "parse_failures": sum(1 for r in results if r.get("parse_error")),
            "output": str(output_path),
            "dry_run": args.dry_run,
            "skip_llm_qc": args.skip_llm_qc,
            "max_bridge_nodes": args.max_bridge_nodes,
        },
    )


if __name__ == "__main__":
    main()
