#!/usr/bin/env python3
"""Experimental M4 multiturn-app generator.

Problem: Method C generates "Why would X support Y?" research-comprehension
questions that almost never pass QC (historical 0-4/48 pass rates).  The user
wants "applied problem" multi-turn tasks where:
  - Turn 1 establishes a concrete scenario/state from the first multimodal element
  - Turn 2 asks a dependent follow-up that requires the second element to solve
  - Evidence is chunk/table/image level, answer is applied reasoning

This generator takes existing enriched M4 material packs and re-prompts the LLM
to produce an applied-problem two-turn dialogue instead of a single research-
relation question.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.api import call_llm, set_company_credentials  # noqa: E402
from src.qc.pipelines import qc_multihop_query  # noqa: E402
from src.qc.llm_judge import run_llm_qc  # noqa: E402

DEFAULT_MATERIALS = ROOT / "data/05_eval/m4_enriched_materials_latest/m4_material_pack.jsonl"
DEFAULT_OUT_ROOT = ROOT / "data/05_eval"
DEFAULT_MODEL = "gpt-5.4"

APPLIED_PROMPT = """You are generating a two-turn multi-hop applied problem from multimodal academic elements.
The goal is NOT to ask "what is the scientific relationship between A and B".
The goal IS to construct a realistic research scenario where a reader must:
  1. First understand a concrete setup from Endpoint A (Turn 1)
  2. Then solve a dependent problem using Endpoint B (Turn 2)

Think of it like a textbook exercise: Turn 1 gives the context, Turn 2 asks
a question that can only be answered by combining Turn 1's setup with Turn 2's
new evidence.

=== ENDPOINT A ===
Element type: {type_a}
Description: {desc_a}

=== ENDPOINT B ===
Element type: {type_b}
Description: {desc_b}

=== BRIDGE CONTEXT ===
{bridge_text}

=== CONSTRAINTS ===

1. TURN 1 (user message, 20-60 words):
   - Set up a specific, grounded scenario using information FROM Endpoint A
   - Include concrete numbers, settings, or conditions visible in Endpoint A
   - Do NOT ask a question yet — just establish the context
   - Use natural domain language (no "figure", "table", "equation", "chart")
   - Example: "A researcher trains a 7B model with LoRA (r=16) on MMLU and
     observes the per-category accuracy distribution shown in the results."

2. TURN 2 (user message, 15-45 words):
   - Ask a concrete question that requires BOTH the Turn-1 setup AND Endpoint B
   - The question should involve comparison, estimation, selection, or prediction
   - It must be impossible to answer without Turn 1's context
   - Use natural language, NOT "based on the figure" or "according to the table"
   - Example: "If they switch to the architecture in Endpoint B while keeping
     the same data budget, which categories would improve and which would regress?"

3. ASSISTANT ANSWER (Turn 1, 30-100 words):
   - Restate and confirm the setup from Endpoint A
   - Include 1-2 specific data points or parameter settings from Endpoint A
   - End by signaling readiness for the follow-up

4. ASSISTANT ANSWER (Turn 2, 40-200 words):
   - Walk through the reasoning step by step
   - Step 1: what does Endpoint B tell us? (cite specific evidence)
   - Step 2: how does this interact with the Turn-1 setup?
   - Step 3: what is the concrete answer to Turn 2's question?
   - Must reference BOTH endpoints' content

5. EVIDENCE CHUNKS (list of 3-6 items):
   - Each item: type (figure/table/formula/text), source doc, short label,
     what specific information it provides
   - Must include at least one from Endpoint A and one from Endpoint B

6. SEARCH CLAUSES (list of 3-5 items):
   - Realistic search queries a researcher would type to find the evidence
   - Should be diverse: some about Endpoint A's topic, some about Endpoint B's,
     some about the bridge concept

=== HARD REJECTIONS ===
- Do NOT use meta words: "figure", "table", "equation", "formula", "chart",
  "graph", "plot", "diagram", "panel", "subfigure"
- Do NOT use template openings: "How does X relate to Y", "What is the
  relationship", "In what way does X impact Y", "Why would X support Y"
- Do NOT use bare deictic pronouns at the start of any message: "this",
  "that", "these", "those"
- Do NOT ask "explain the relationship" or "why is X important for Y"
- The answer must NOT be just a summary of what the two endpoints say
- The question must be ANSWERABLE — not an open-ended research question

=== OUTPUT FORMAT (valid JSON only, no markdown) ===
{{
  "history": [
    {{"role": "user", "message": "...turn 1 setup..."}},
    {{"role": "assistant", "message": "...turn 1 confirmation..."}}
  ],
  "question": "...turn 2 dependent question...",
  "answer_short": "...1-2 sentence answer...",
  "answer_long": "...step-by-step applied reasoning...",
  "evidence_chunk_list": [
    {{"type": "figure", "source": "doc_id", "label": "Figure N", "provides": "what this evidence contributes"}}
  ],
  "search_clause": ["query 1", "query 2", "query 3"],
  "dependency_check": {{
    "needs_turn1_context": true,
    "uses_endpoint_a": true,
    "uses_endpoint_b": true,
    "uses_bridge": true
  }}
}}"""


def _elem_desc(elem: dict, limit: int = 400) -> str:
    parts = []
    for k in ("label", "caption", "content", "context_before", "context_after",
              "enriched_title", "enriched_content"):
        v = elem.get(k)
        if v and isinstance(v, str) and len(v.strip()) > 5:
            # Skip noisy enriched content
            if "noise" in str(v).lower() or len(str(v).split()) < 3:
                continue
            parts.append(str(v)[:limit])
    return " ".join(parts)[:limit * 2]


def _bridge_text(pair: dict) -> str:
    """Extract bridge context from the pair."""
    hub = pair.get("hub_metadata") or pair.get("method_c", {})
    summary = hub.get("hub_semantic_summary") or pair.get("hub_semantic_summary", "")
    if summary:
        return str(summary)[:600]
    # Fallback: edge contexts
    edges = pair.get("edge_contexts", [])
    if edges:
        return str(edges[0].get("context_snippet", ""))[:400]
    return "The endpoints are connected through a citation or structural bridge in the document graph."


def build_applied_prompt(pair: dict) -> str:
    ea = pair.get("element_a", {})
    eb = pair.get("element_b", {})
    return APPLIED_PROMPT.format(
        type_a=ea.get("element_type", "element"),
        type_b=eb.get("element_type", "element"),
        desc_a=_elem_desc(ea, 400),
        desc_b=_elem_desc(eb, 400),
        bridge_text=_bridge_text(pair),
    )


def parse_applied_response(raw: str) -> Optional[dict]:
    """Parse the LLM JSON response, with cleanup."""
    text = raw.strip()
    # Strip markdown code fences
    if text.startswith("```"):
        lines = text.split("\n")
        text = "\n".join(lines[1:])
        if text.endswith("```"):
            text = text[:-3]
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        # Try to find JSON object
        import re
        m = re.search(r'\{.*\}', text, re.DOTALL)
        if m:
            try:
                return json.loads(m.group())
            except json.JSONDecodeError:
                pass
    return None


def rule_qc_applied(item: dict) -> Tuple[bool, list, dict]:
    """Lightweight rule QC adapted for the applied-problem format."""
    issues = []
    metrics = {}

    history = item.get("history") or []
    question = str(item.get("question") or "")
    answer_long = str(item.get("answer_long") or "")
    answer_short = str(item.get("answer_short") or "")

    # Check required fields
    if not history or len(history) < 2:
        issues.append("missing_history")
    if not question:
        issues.append("missing_question")
    if not answer_long:
        issues.append("missing_answer_long")

    # Check meta-language ban
    meta_words = ["figure", "table", "equation", "formula", "chart",
                  "graph", "plot", "diagram", "panel", "subfigure"]
    for mw in meta_words:
        if mw.lower() in question.lower():
            issues.append(f"meta_word_in_question:{mw}")
            break

    # Check template openings ban
    template_starts = [
        "how does", "what is the relationship", "in what way does",
        "why would", "why does", "explain the relationship",
        "how are", "how is",
    ]
    q_lower = question.lower().strip()
    for ts in template_starts:
        if q_lower.startswith(ts):
            issues.append(f"template_opening:{ts}")
            break

    # Check bare deictic at start
    deictic_start = ["this ", "that ", "these ", "those "]
    for ds in deictic_start:
        if q_lower.startswith(ds):
            issues.append(f"bare_deictic:{ds.strip()}")
            break

    # Check dependency_check
    dep = item.get("dependency_check") or {}
    if not dep.get("needs_turn1_context", False):
        issues.append("dependency_not_requiring_turn1")
    if not dep.get("uses_endpoint_a", False):
        issues.append("dependency_not_using_endpoint_a")
    if not dep.get("uses_endpoint_b", False):
        issues.append("dependency_not_using_endpoint_b")

    # Evidence and search clauses
    ev = item.get("evidence_chunk_list") or []
    metrics["evidence_count"] = len(ev)
    sc = item.get("search_clause") or []
    metrics["search_clause_count"] = len(sc)

    metrics["question_word_count"] = len(question.split())
    metrics["answer_long_word_count"] = len(answer_long.split())

    passed = len(issues) == 0
    return passed, issues, metrics


def build_standard_qc_obj(parsed: dict, material: dict) -> dict:
    """Adapt multiturn-app output to the standard qc_obj for qc_multihop_query / run_llm_qc."""
    history = parsed.get("history") or []
    t1_msg = history[0].get("message", "") if history else ""
    question = parsed.get("question", "")
    answer = parsed.get("answer_long", "")
    full_query = f"{t1_msg}\n{question}"

    ev_chunks = parsed.get("evidence_chunk_list") or []
    text_evidence = "\n".join(
        f"[{c.get('type','')}] {c.get('label','')}: {c.get('provides','')}"
        for c in ev_chunks
    )

    ea = material.get("element_a") or {}
    eb = material.get("element_b") or {}
    visual_anchors = [
        {"element_id": ea.get("element_id", ""),
         "anchor_text": (ea.get("enriched_title") or ea.get("caption") or "")[:60]},
        {"element_id": eb.get("element_id", ""),
         "anchor_text": (eb.get("enriched_title") or eb.get("caption") or "")[:60]},
    ]

    return {
        "query": full_query,
        "answer": answer,
        "text_evidence": text_evidence[:1500],
        "visual_anchors": visual_anchors,
        "reasoning_chain": [],
    }


def run_standard_qc(qc_obj: dict, material: dict, args) -> Tuple[list, dict, list, dict, int, int]:
    """Run rule QC + LLM QC on a single generated item. Returns combined results."""
    all_issues = []
    all_metrics = {}
    total_in, total_out = 0, 0

    # Rule QC
    rule_issues, rule_metrics = qc_multihop_query(qc_obj, material)
    all_issues.extend(rule_issues)
    all_metrics["rule_qc"] = {"issues": rule_issues, "metrics": rule_metrics}

    # LLM QC
    if not args.skip_llm_qc:
        try:
            llm_issues, llm_metrics, ab_in, ab_out = run_llm_qc(
                obj=qc_obj,
                pair=material,
                client=None,
                model=args.model,
                provider="company",
                dry_run=False,
            )
            all_issues.extend(llm_issues)
            all_metrics["llm_qc"] = {"issues": llm_issues, "metrics": llm_metrics}
            total_in += ab_in
            total_out += ab_out
        except Exception as e:
            all_metrics["llm_qc"] = {"error": str(e)[:200]}

    return all_issues, all_metrics, rule_issues, rule_metrics, total_in, total_out


def main():
    parser = argparse.ArgumentParser(
        description="Generate M4 applied-problem multiturn tasks from enriched materials.")
    parser.add_argument("--materials", default=str(DEFAULT_MATERIALS))
    parser.add_argument("--out-root", default=str(DEFAULT_OUT_ROOT))
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--limit", type=int, default=20)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--skip-llm-qc", action="store_true")
    parser.add_argument("--skip", type=int, default=0)
    parser.add_argument("--api-url", default="")
    parser.add_argument("--api-key", default="")
    args = parser.parse_args()

    if args.api_url or args.api_key:
        set_company_credentials(args.api_url, args.api_key)

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_dir = Path(args.out_root) / f"m4_multiturn_app_smoke_{stamp}"
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Output: {out_dir}")
    print(f"Model: {args.model}")
    print(f"Mode: {'DRY-RUN' if args.dry_run else 'LIVE'}")

    # Load materials
    materials = []
    with open(args.materials) as f:
        for line in f:
            if line.strip():
                materials.append(json.loads(line))
    print(f"Loaded {len(materials)} materials")

    selected = materials[args.skip:args.skip + args.limit]
    print(f"Selected {len(selected)} for generation")

    results = []
    stats = Counter()
    total_in, total_out = 0, 0

    for i, mat in enumerate(selected):
        mid = mat.get("material_id", f"material_{i}")
        print(f"\n--- [{i+1}/{len(selected)}] {mid} ---")

        prompt = build_applied_prompt(mat)
        if args.dry_run:
            print(f"  Prompt length: {len(prompt)} chars")
            print(f"  Prompt preview:\n{prompt[:600]}...\n")
            results.append({
                "material_id": mid,
                "prompt_preview": prompt[:500],
            })
            continue

        raw, gen_in, gen_out = call_llm(
            client=None,
            model=args.model,
            prompt=prompt,
            provider="company",
            system_prompt=(
                "You are an expert at generating applied-problem multi-turn "
                "tasks from multimodal academic evidence. You write concrete, "
                "answerable problems that require cross-document reasoning."
            ),
            user_tag="m4_multiturn_app_v1",
            temperature=0.7,
        )
        total_in += gen_in
        total_out += gen_out
        stats["api_calls"] += 1

        parsed = parse_applied_response(raw)
        if not parsed:
            stats["parse_failures"] += 1
            print(f"  PARSE FAILURE: {raw[:200]}")
            results.append({"material_id": mid, "raw": raw, "parse_failed": True})
            continue

        passed, issues, metrics = rule_qc_applied(parsed)
        stats["generated"] += 1
        if passed:
            stats["rule_qc_pass"] += 1
        else:
            stats["rule_qc_fail"] += 1

        # 👇 Wire in standard qc_multihop_query + run_llm_qc
        qc_obj = build_standard_qc_obj(parsed, mat)
        std_issues, std_metrics, rule_is, rule_met, qc_in, qc_out = run_standard_qc(qc_obj, mat, args)
        total_in += qc_in
        total_out += qc_out
        if std_issues:
            stats["llm_qc_issues_found"] += 1
        else:
            stats["llm_qc_clean"] += 1

        result = {
            "material_id": mid,
            "pair_id": mat.get("pair_id", ""),
            "generated": parsed,
            "qc": {
                "passed_rule_qc": passed,
                "rule_issues": issues,
                "rule_metrics": metrics,
                "standard_issues": std_issues,
                "standard_metrics": std_metrics,
            },
            "tokens": {"gen_in": gen_in, "gen_out": gen_out, "qc_in": qc_in, "qc_out": qc_out},
        }
        results.append(result)

        q = parsed.get("question", "")
        std_status = "LLM-CLEAN" if not std_issues else f"LLM-FAIL({';'.join(std_issues[:2])})"
        print(f"  Turn1: {parsed.get('history', [{}])[0].get('message', '')[:100] if parsed.get('history') else '?'}")
        print(f"  Q: {q[:120]}")
        print(f"  Rule: {'PASS' if passed else 'FAIL'} | Std: {std_status}")

    # Save
    with open(out_dir / "generated_multiturn_smoke.jsonl", "w") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    summary = {
        "status": "ok",
        "mode": "dry_run" if args.dry_run else "live",
        "output_dir": str(out_dir),
        "source_materials": str(args.materials),
        "model": args.model,
        "requested": len(selected),
        "generated_rows": stats.get("generated", 0),
        "rule_qc_pass": stats.get("rule_qc_pass", 0),
        "rule_qc_fail": stats.get("rule_qc_fail", 0),
        "llm_qc_clean": stats.get("llm_qc_clean", 0),
        "llm_qc_issues_found": stats.get("llm_qc_issues_found", 0),
        "parse_failures": stats.get("parse_failures", 0),
        "api_calls": stats.get("api_calls", 0),
        "skip_llm_qc": args.skip_llm_qc,
        "tokens": {"in": total_in, "out": total_out},
    }
    with open(out_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    # Latest symlink
    latest = Path(args.out_root) / "m4_multiturn_app_latest"
    if latest.exists() or latest.is_symlink():
        latest.unlink()
    latest.symlink_to(out_dir, target_is_directory=True)

    print(f"\nDone. {stats.get('generated', 0)} generated, "
          f"{stats.get('rule_qc_pass', 0)} rule-pass, {stats.get('rule_qc_fail', 0)} rule-fail, "
          f"{stats.get('llm_qc_clean', 0)} llm-clean, {stats.get('llm_qc_issues_found', 0)} llm-fail")
    print(f"Tokens: {total_in} in / {total_out} out")
    print(f"Latest: {latest}")


if __name__ == "__main__":
    main()
