"""多模态 LLM Judge —— 用大模型做最终质量关卡。

规则 QC（checks.py / pipelines.py）只能做表面检查，抓不住两类核心缺陷：
  1. 伪多跳（Fake multi-hop）：query 表面看起来跨元素，但其实只靠一个元素就能回答
  2. 答案幻觉（Answer hallucination）：answer 声称来自 evidence，但实际内容对不上

本模块把这两类检查做成标准接口，支持多模态图片输入，
provider 路由完全复用 src.api.call_llm，所以 company / openai / anthropic 三条路都走得通。

对外 API：
  - judge_single_element_batch() → 一次调用判断所有元素是否单独能回答 query（合并 ablation）
  - judge_answer_grounding()     → 给定 evidence + images，判断 answer 是否可从证据推出
  - run_llm_qc()                 → 两项合并，**恰好 2 次 LLM 调用**，返回 (issues, metrics)

注意：LLM judge 只在规则 QC pass 之后运行，成本固定 2 次 LLM 调用/query（dry_run=True 时跳过）。
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple

from src.api import call_llm, parse_json


# ── System prompts ────────────────────────────────────────────────────────────

_SYSTEM_NECESSITY = (
    "You are a strict multi-hop retrieval evaluator. "
    "For each evidence element listed, decide independently whether that single element "
    "alone is sufficient to answer the question. "
    "Do not guess or use outside knowledge. Return valid JSON only, no markdown."
)

_SYSTEM_GROUNDING = (
    "You are an answer-grounding evaluator for multi-hop reasoning queries. "
    "Given a question, an answer, and evidence elements (text + optional images), "
    "decide whether the factual claims in the answer can be supported by or reasonably "
    "inferred from the evidence. "
    "Flag hallucinations ONLY for claims that directly contradict the evidence or introduce "
    "specific numbers/names that cannot be traced or inferred from the evidence at all. "
    "Allow reasonable synthesis and inference across evidence elements. "
    "Return valid JSON only, no markdown."
)


# ── Prompt builders ───────────────────────────────────────────────────────────

def _element_block(elem: Dict[str, Any], idx: int, context_limit: int = 800) -> str:
    def _short(text: str, limit: int) -> str:
        t = " ".join((text or "").split())
        return t[:limit]

    lines = [f"[Element {idx}]"]
    lines.append(f"ID: {elem.get('element_id', '')}")
    lines.append(f"Type: {elem.get('element_type', 'unknown')}")
    if elem.get("caption"):
        lines.append(f"Caption: {_short(elem['caption'], 300)}")
    if elem.get("content"):
        lines.append(f"Content: {_short(elem['content'], context_limit)}")
    if elem.get("context_before"):
        lines.append(f"Context before: {_short(elem['context_before'], 300)}")
    if elem.get("context_after"):
        lines.append(f"Context after: {_short(elem['context_after'], 300)}")
    return "\n".join(lines)


def _build_batch_necessity_prompt(
    question: str,
    reference_answer: str,
    elements: Sequence[Dict[str, Any]],
) -> str:
    """Build a single prompt that tests all elements at once (one call replaces N calls)."""
    blocks = "\n\n".join(_element_block(e, i + 1) for i, e in enumerate(elements))
    indices = list(range(1, len(elements) + 1))
    return (
        f"Question:\n{question}\n\n"
        f"Reference answer (semantic target):\n{reference_answer}\n\n"
        f"Evidence elements (evaluate each independently):\n{blocks}\n\n"
        "For EACH element, decide: can this single element alone fully answer the question "
        "without any of the other elements?\n"
        "- can_answer = true ONLY if the core answer claim can be derived from that element alone.\n"
        "- Do not guess or use outside knowledge.\n\n"
        f"Return JSON array with one entry per element (indices {indices}):\n"
        '[{"element_index": 1, "can_answer": true/false, "confidence": 0.0-1.0, "reason": "short"}, ...]'
    )


def _build_grounding_prompt(
    question: str,
    answer: str,
    elements: Sequence[Dict[str, Any]],
    text_evidence: str,
) -> str:
    blocks = "\n\n".join(_element_block(e, i + 1) for i, e in enumerate(elements))
    ev_section = f"\nAdditional text evidence:\n{text_evidence}" if text_evidence else ""
    return (
        f"Question:\n{question}\n\n"
        f"Answer to evaluate:\n{answer}\n\n"
        f"Evidence elements:\n{blocks}{ev_section}\n\n"
        "Grounding rules:\n"
        "- is_grounded = true if the answer's claims can be supported by or reasonably "
        "inferred/synthesized from the evidence (inference across elements is allowed).\n"
        "- is_grounded = false ONLY if the answer introduces specific numbers, names, or "
        "conclusions that directly contradict the evidence or have no basis in it whatsoever.\n"
        "- Do NOT flag claims as hallucinations if they are reasonable inferences or "
        "syntheses from the provided evidence, even if not word-for-word present.\n"
        "- List specific hallucinated claims in 'hallucinations' (empty list if none).\n\n"
        'Return JSON: {"is_grounded": true/false, "confidence": 0.0-1.0, '
        '"hallucinations": ["claim1", ...], "reason": "short reason"}'
    )


# ── Core judge functions ──────────────────────────────────────────────────────

def judge_single_element_batch(
    question: str,
    reference_answer: str,
    elements: Sequence[Dict[str, Any]],
    client: Any,
    model: str,
    provider: str,
    images: Optional[List[Optional[Tuple[str, str]]]] = None,
    dry_run: bool = False,
) -> Tuple[List[bool], List[float], int, int]:
    """一次 LLM 调用判断所有元素是否单独能回答 query。

    返回 (can_answer_list, confidence_list, in_tokens, out_tokens)。
    列表长度与 elements 相同，顺序对应。
    """
    if dry_run:
        return [False] * len(elements), [0.0] * len(elements), 0, 0

    prompt = _build_batch_necessity_prompt(question, reference_answer, elements)
    raw, in_tok, out_tok = call_llm(
        client, model, prompt,
        images=images or [],
        provider=provider,
        system_prompt=_SYSTEM_NECESSITY,
        max_tokens=128 * len(elements),
        temperature=0.0,
        user_tag="llm_qc_necessity",
    )
    results = parse_json(raw)
    if not isinstance(results, list):
        results = []

    can_list = [False] * len(elements)
    conf_list = [0.0] * len(elements)
    for item in results:
        if not isinstance(item, dict):
            continue
        idx = int(item.get("element_index", 0)) - 1
        if 0 <= idx < len(elements):
            can_list[idx] = bool(item.get("can_answer", False))
            conf_list[idx] = float(item.get("confidence", 0.0) or 0.0)
    return can_list, conf_list, in_tok, out_tok


def judge_answer_grounding(
    question: str,
    answer: str,
    elements: Sequence[Dict[str, Any]],
    text_evidence: str,
    client: Any,
    model: str,
    provider: str,
    images: Optional[List[Optional[Tuple[str, str]]]] = None,
    dry_run: bool = False,
) -> Tuple[bool, List[str], float, str, int, int]:
    """判断 answer 是否完全来自 evidence，检测幻觉。

    返回 (is_grounded, hallucinations, confidence, reason, in_tokens, out_tokens)。
    """
    if dry_run:
        return True, [], 1.0, "dry-run", 0, 0

    prompt = _build_grounding_prompt(question, answer, elements, text_evidence)
    raw, in_tok, out_tok = call_llm(
        client, model, prompt,
        images=images or [],
        provider=provider,
        system_prompt=_SYSTEM_GROUNDING,
        max_tokens=384,
        temperature=0.0,
        user_tag="llm_qc_grounding",
    )
    obj = parse_json(raw) or {}
    return (
        bool(obj.get("is_grounded", True)),
        list(obj.get("hallucinations", []) or []),
        float(obj.get("confidence", 0.0) or 0.0),
        str(obj.get("reason", ""))[:200],
        in_tok,
        out_tok,
    )


# ── Ablation check (single-call batch version) ───────────────────────────────

def run_ablation_qc(
    question: str,
    reference_answer: str,
    elements: Sequence[Dict[str, Any]],
    client: Any,
    model: str,
    provider: str,
    images: Optional[List[Optional[Tuple[str, str]]]] = None,
    dry_run: bool = False,
    skip: bool = False,
) -> Tuple[Dict[str, Any], bool, int, int]:
    """Fake-multihop detection using a single batched LLM call.

    Tests whether any single element alone can answer the question.
    Uses judge_single_element_batch() — exactly 1 LLM call regardless of element count.

    判定为 fake_multihop 的条件：任意单个元素独自就能回答 query。
    不做 drop-intermediate 测试（节省调用次数，主要信号来自 single-element）。

    返回 (ablation_metrics, is_fake, total_in_tok, total_out_tok)。
    """
    if skip:
        return {"skipped": True}, False, 0, 0

    can_list, conf_list, in_tok, out_tok = judge_single_element_batch(
        question, reference_answer, list(elements),
        client, model, provider, images, dry_run,
    )
    metrics = {
        "single_element_can_answer": can_list,
        "single_element_confidence": [round(c, 4) for c in conf_list],
    }
    is_fake = any(can_list)
    return metrics, is_fake, in_tok, out_tok


# ── Main entry point ──────────────────────────────────────────────────────────

def run_llm_qc(
    obj: Dict[str, Any],
    pair: Dict[str, Any],
    client: Any,
    model: str,
    provider: str,
    images: Optional[List[Optional[Tuple[str, str]]]] = None,
    dry_run: bool = False,
    skip_ablation: bool = False,
    skip_grounding: bool = False,
) -> Tuple[List[str], Dict[str, Any], int, int]:
    """多模态 LLM QC 主入口 —— 规则 QC 通过后再调这个。

    运行两项检查：
      1. Ablation（step-deletion）：移除任意 element 后 query 变不可回答 → fake_multihop
      2. Answer grounding：answer 里的事实都能在 evidence 里找到 → answer_hallucination

    参数：
      obj   — 生成的 query 对象（含 query / answer / text_evidence / visual_anchors）
      pair  — 候选 pair（含 element_a / element_b，以及 bridge/intermediate elements）
      images — 已编码的图片列表（base64, mime），按 element 顺序排列

    返回 (issues, metrics, in_tokens, out_tokens)。
    issues 直接追加到规则 QC 的 issues 列表里。
    """
    issues: List[str] = []
    metrics: Dict[str, Any] = {}
    total_in, total_out = 0, 0

    question = obj.get("query", "")
    answer = obj.get("answer", "")
    text_evidence = obj.get("text_evidence", "")

    # 收集所有 elements：element_a / element_b / intermediate_elements（如有）
    elements: List[Dict[str, Any]] = []
    if pair.get("element_a"):
        elements.append(pair["element_a"])
    for ie in (pair.get("intermediate_elements") or []):
        elements.append(ie)
    if pair.get("element_b"):
        elements.append(pair["element_b"])

    if not elements or not question or not answer:
        metrics["llm_qc_skipped"] = "missing_fields"
        return issues, metrics, total_in, total_out

    # ── 1. Ablation: step-deletion test ──────────────────────────────────────
    try:
        ablation_metrics, is_fake, in_tok, out_tok = run_ablation_qc(
            question=question,
            reference_answer=answer,
            elements=elements,
            client=client,
            model=model,
            provider=provider,
            images=images,
            dry_run=dry_run,
            skip=skip_ablation,
        )
        total_in += in_tok; total_out += out_tok
        metrics["llm_ablation"] = ablation_metrics
        if is_fake:
            issues.append("llm_fake_multihop")
            metrics["llm_fake_multihop_warn"] = True
    except Exception as e:
        metrics["llm_ablation"] = {"error": str(e)[:200]}

    # ── 2. Answer grounding check ─────────────────────────────────────────────
    if not skip_grounding:
        try:
            is_grounded, hallucinations, grounding_conf, grounding_reason, in_tok, out_tok = judge_answer_grounding(
                question=question,
                answer=answer,
                elements=elements,
                text_evidence=text_evidence,
                client=client,
                model=model,
                provider=provider,
                images=images,
                dry_run=dry_run,
            )
            total_in += in_tok; total_out += out_tok
            metrics["llm_grounding"] = {
                "is_grounded": is_grounded,
                "confidence": round(grounding_conf, 4),
                "hallucinations": hallucinations[:5],
                "reason": grounding_reason,
            }
            if not is_grounded:
                issues.append("llm_answer_hallucination")
                metrics["llm_hallucination_warn"] = True
        except Exception as e:
            metrics["llm_grounding"] = {"error": str(e)[:200]}

    return issues, metrics, total_in, total_out
