"""多模态 LLM Judge —— 用大模型做最终质量关卡。

规则 QC（checks.py / pipelines.py）只能做表面检查，抓不住两类核心缺陷：
  1. 伪多跳（Fake multi-hop）：query 表面看起来跨元素，但其实只靠一个元素就能回答
  2. 答案幻觉（Answer hallucination）：answer 声称来自 evidence，但实际内容对不上

本模块把这两类检查做成标准接口，支持多模态图片输入，
provider 路由完全复用 src.api.call_llm，所以 company / openai / anthropic 三条路都走得通。

对外 API：
  - judge_evidence_necessity()  → 每个元素单独移除，验证 query 是否变得不可回答
  - judge_answer_grounding()    → 给定 evidence + images，判断 answer 是否可从证据推出
  - run_llm_qc()                → 两项合并，返回 (issues, metrics)，格式和规则 QC 一致

注意：LLM judge 只在规则 QC pass 之后运行，成本约 3-5 次额外 LLM 调用/query。
dry_run=True 时全部跳过（返回空 issues），不影响本地调试。
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple

from src.api import call_llm, parse_json


# ── System prompts ────────────────────────────────────────────────────────────

_SYSTEM_NECESSITY = (
    "You are a strict multi-hop retrieval evaluator. "
    "Given a question, a reference answer, and a set of evidence elements "
    "(text + optional images), decide whether the question can be answered "
    "using ONLY the provided evidence. "
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


def _build_necessity_prompt(
    question: str,
    reference_answer: str,
    elements: Sequence[Dict[str, Any]],
    scenario_name: str,
) -> str:
    blocks = "\n\n".join(_element_block(e, i + 1) for i, e in enumerate(elements))
    return (
        f"Scenario: {scenario_name}\n\n"
        f"Question:\n{question}\n\n"
        f"Reference answer (semantic target, not required wording):\n{reference_answer}\n\n"
        f"Evidence elements:\n{blocks}\n\n"
        "Judge rules:\n"
        "- can_answer = true ONLY if the core claim of the reference answer can be "
        "fully derived from the provided evidence without guessing.\n"
        "- If any key fact is missing, set can_answer = false.\n\n"
        'Return JSON: {"can_answer": true/false, "confidence": 0.0-1.0, "reason": "short reason"}'
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

def judge_evidence_necessity(
    question: str,
    reference_answer: str,
    elements: Sequence[Dict[str, Any]],
    scenario_name: str,
    client: Any,
    model: str,
    provider: str,
    images: Optional[List[Optional[Tuple[str, str]]]] = None,
    dry_run: bool = False,
) -> Tuple[bool, float, str, int, int]:
    """判断给定 elements 子集是否足以回答 question。

    返回 (can_answer, confidence, reason, in_tokens, out_tokens)。
    dry_run=True 时不调 API，直接返回 (False, 0.0, 'dry-run', 0, 0)。
    """
    if dry_run:
        return False, 0.0, "dry-run", 0, 0

    prompt = _build_necessity_prompt(question, reference_answer, elements, scenario_name)
    raw, in_tok, out_tok = call_llm(
        client, model, prompt,
        images=images or [],
        provider=provider,
        system_prompt=_SYSTEM_NECESSITY,
        max_tokens=256,
        temperature=0.0,
        user_tag="llm_qc_necessity",
    )
    obj = parse_json(raw) or {}
    return (
        bool(obj.get("can_answer", False)),
        float(obj.get("confidence", 0.0) or 0.0),
        str(obj.get("reason", ""))[:200],
        in_tok,
        out_tok,
    )


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


# ── Ablation check (step-deletion test) ──────────────────────────────────────

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
    """Step-deletion test: 每次移除一个 element，验证 query 是否变得不可回答。

    判定为 fake_multihop 的条件：
      - 单个元素就能回答（single_element_can_answer 任意 True）
      - 移除任意中间节点后仍能回答（drop_element_can_answer 任意 True，仅 3+ 元素）

    注意：不再用 full_can_answer=False 作为判定条件。
    因为 judge 只能看到截断的 element 片段，无法获得生成时的完整 bridge 上下文，
    导致 full_can_answer 经常误报 False（高置信度假阴性）。
    真正的 fake multi-hop 特征是"单元素就能答"，而不是"全集答不了"。

    返回 (ablation_metrics, is_fake, total_in_tok, total_out_tok)。
    """
    if skip:
        return {"skipped": True}, False, 0, 0

    elems = list(elements)
    total_in, total_out = 0, 0

    # 1. Full set — should be answerable
    full_can, full_conf, _, in_tok, out_tok = judge_evidence_necessity(
        question, reference_answer, elems, "full_set",
        client, model, provider, images, dry_run,
    )
    total_in += in_tok; total_out += out_tok

    # 2. Single-element tests — each element alone should NOT be enough
    #    (For 2-element pairs, this replaces the endpoints_only test which
    #     would be identical to the full set and always trigger false positive)
    single_flags: List[bool] = []
    single_confs: List[float] = []
    for idx, elem in enumerate(elems):
        can, conf, _, in_tok, out_tok = judge_evidence_necessity(
            question, reference_answer, [elem], f"element_{idx}_only",
            client, model, provider, images, dry_run,
        )
        total_in += in_tok; total_out += out_tok
        single_flags.append(can)
        single_confs.append(conf)

    # 3. Drop each intermediate element (only meaningful for 3+ elements)
    drop_flags: List[bool] = []
    drop_confs: List[float] = []
    if len(elems) >= 3:
        for drop_idx in range(1, len(elems) - 1):
            kept = [e for i, e in enumerate(elems) if i != drop_idx]
            can, conf, _, in_tok, out_tok = judge_evidence_necessity(
                question, reference_answer, kept, f"drop_element_{drop_idx}",
                client, model, provider, images, dry_run,
            )
            total_in += in_tok; total_out += out_tok
            drop_flags.append(can)
            drop_confs.append(conf)

    metrics = {
        "full_can_answer": full_can,
        "full_confidence": round(full_conf, 4),
        "single_element_can_answer": single_flags,
        "single_element_confidence": [round(c, 4) for c in single_confs],
        "drop_element_can_answer": drop_flags,
        "drop_element_confidence": [round(c, 4) for c in drop_confs],
    }
    # fake = 单个元素就能答 OR 去掉中间节点仍能答
    # 不再包含 (not full_can)：judge 看不到完整 bridge 上下文，full_can=False 假阳性极高
    is_fake = any(single_flags) or any(drop_flags)
    return metrics, is_fake, total_in, total_out


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
