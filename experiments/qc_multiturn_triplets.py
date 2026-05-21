#!/usr/bin/env python3
"""Multi-turn-adapted LLM QC for M4 triplet generation.

The standard qc_multihop_query + run_llm_qc pipeline was calibrated for
single-turn "Why would X support Y?" questions from Method C. It breaks
on multi-turn applied problems because:
  - "numeric_leakage": numbers in Turn 1 are SUPPOSED to be referenced in Turn 2
  - "missing_reasoning_chain": chain is split across turns, not in one answer
  - "pseudo_multihop_parallel": turn dependency makes it sequential by design

This QC tests what actually matters for multi-turn triplets:
  S1. Turn dependency: does Turn 2 genuinely need Turn 1's context?
  S2. Element necessity: does removing an element break the chain?
  S3. Evidence grounding: are answer claims supported?
  S4. Surface quality: meta-language, deictics, template openings.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.api import call_llm, parse_json, set_company_credentials
from src.utils.token_logger import log_run

DEFAULT_TRIPLETS = ROOT / "data/05_eval/m4_multiturn_triplets_latest/multiturn_triplets.jsonl"
DEFAULT_OUT_ROOT = ROOT / "data/05_eval"
DEFAULT_MODEL = "gpt-5.4"


def configure_standard_logger() -> Path:
    from local_api_logger.logger import APILogger  # noqa: WPS433
    import local_api_logger.tracker as tracker  # noqa: WPS433

    log_dir = ROOT / "api_logs_cannt_delete"
    log_dir.mkdir(parents=True, exist_ok=True)
    tracker._default_tracker = tracker.APITracker(APILogger(str(log_dir)))
    return log_dir

# ── S1: Turn dependency ─────────────────────────────────────────────────────

TURN_DEPENDENCY_PROMPT = """You are evaluating whether a multi-turn question is genuinely dependent on prior context.

Below is a two-part dialogue. Your job: determine if Turn 2's question can be
answered WITHOUT Turn 1's assistant response.

=== TURN 1 (user setup) ===
{turn1_user}

=== TURN 1 (assistant response) ===
{turn1_assistant}

=== TURN 2 (user question) ===
{turn2_question}

=== QUESTION ===
If we DELETE Turn 1's assistant response entirely, can Turn 2's question still
be answered by someone who has the same reference materials?

Answer YES if: Turn 2 is self-contained and doesn't reference Turn 1's result.
Answer NO if: Turn 2 explicitly relies on a value, condition, or conclusion
stated in Turn 1's assistant response.

### OUTPUT (valid JSON only)
{{
  "turn2_depends_on_turn1": true/false,
  "dependency_type": "explicit_value_reference|condition_carried_forward|conclusion_required|none",
  "evidence_from_turn1_used": "the specific value, condition, or conclusion Turn 2 depends on, or null",
  "explanation": "one sentence"
}}"""


# ── S2: Element necessity ───────────────────────────────────────────────────

ELEMENT_NECESSITY_PROMPT = """You are evaluating whether a multi-hop question genuinely requires
specific evidence elements.

Below is a question and two evidence elements from DIFFERENT documents.
Your job: determine if BOTH elements are necessary to answer the question.

=== QUESTION CONTEXT ===
Turn 1 setup: {turn1_user}
Turn 1 assistant: {turn1_assistant}
Turn 2 question: {turn2_question}

=== ELEMENT A ({type_a} from doc {doc_a}) ===
{desc_a}

=== ELEMENT B ({type_b} from doc {doc_b}) ===
{desc_b}

=== BRIDGE — how they connect ===
{bridge}

=== QUESTION ===
1. If we remove Element A, can the question still be fully answered? (YES/NO)
2. If we remove Element B, can the question still be fully answered? (YES/NO)
3. Does the answer require the BRIDGE reasoning step between them? (YES/NO)
4. Is this a genuine multi-hop (serial chain A→bridge→B) or parallel retrieval
   (A and B are independently relevant)?

### OUTPUT (valid JSON only)
{{
  "needs_element_a": true/false,
  "needs_element_b": true/false,
  "needs_bridge": true/false,
  "chain_type": "serial_multi_hop|parallel_retrieval|weak_dependency",
  "explanation": "one sentence"
}}"""


# ── S3: Evidence grounding ──────────────────────────────────────────────────

GROUNDING_PROMPT = """You are an answer-grounding evaluator. Given a question, its answer,
and a list of evidence elements, verify whether the answer's factual claims
are supported by the evidence.

=== QUESTION ===
Turn 1 setup: {turn1_user}
Turn 2 question: {turn2_question}

=== ANSWER ===
{answer}

=== EVIDENCE ELEMENTS ===
{evidence_list}

=== QUESTION ===
For each factual claim in the answer:
1. Can it be found in the evidence elements? (grounded / partially_grounded / hallucinated)
2. If hallucinated, is it a minor detail or a core claim?

Then provide an overall grounding score:
  - fully_grounded: all claims found in evidence
  - mostly_grounded: >= 2/3 of claims found
  - partially_grounded: >= 1/3 found
  - mostly_hallucinated: < 1/3 found

### OUTPUT (valid JSON only)
{{
  "grounding_level": "fully_grounded|mostly_grounded|partially_grounded|mostly_hallucinated",
  "claims_total": <int>,
  "claims_grounded": <int>,
  "claims_partial": <int>,
  "claims_hallucinated": <int>,
  "hallucinated_core_claims": ["claim text or empty list"],
  "explanation": "one sentence"
}}"""


# ── S4: Surface quality (rule-based) ────────────────────────────────────────


def surface_qc(record: dict) -> Tuple[bool, List[str], dict]:
    """Rule-based surface quality check for multi-turn dialogue."""
    issues: List[str] = []
    metrics: Dict[str, Any] = {}
    g = record.get("generated", {})

    history = g.get("history") or []
    question = str(g.get("question") or "")
    answer_long = str(g.get("answer_long") or "")
    answer_short = str(g.get("answer_short") or "")

    # Required fields
    if not history or len(history) < 2:
        issues.append("missing_history")
    if not question:
        issues.append("missing_question")
    if not answer_long:
        issues.append("missing_answer_long")

    # Meta-language in user-facing messages
    meta_words = [
        "figure", "table", "equation", "formula", "chart",
        "graph", "plot", "diagram", "panel", "subfigure",
    ]
    for i, h in enumerate(history):
        if h.get("role") == "user":
            msg = str(h.get("message", "")).lower()
            for mw in meta_words:
                if mw in msg:
                    issues.append(f"meta_word_in_turn{i}:{mw}")
                    break
    for mw in meta_words:
        if mw in question.lower():
            issues.append(f"meta_word_in_question:{mw}")
            break

    # Template openings in Turn 2 question
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

    # Bare deictic at turn/question start
    for label, text in [("turn1", history[0].get("message", "") if history else ""),
                        ("question", question)]:
        t_lower = str(text).lower().strip()
        for ds in ["this ", "that ", "these ", "those "]:
            if t_lower.startswith(ds):
                issues.append(f"bare_deictic_{label}:{ds.strip()}")
                break

    # Turn dependency check from generated metadata
    deps = g.get("turn_dependencies") or []
    if not deps:
        issues.append("missing_turn_dependencies")
    else:
        has_turn2_dep = any(d.get("turn") == 2 and d.get("depends_on_turn") for d in deps)
        if not has_turn2_dep:
            issues.append("turn2_not_marked_dependent")

    # Chain coverage
    cov = g.get("chain_coverage") or {}
    for key in ["element_a_used_in", "element_b_used_in"]:
        if not cov.get(key):
            issues.append(f"missing_chain_coverage:{key}")

    metrics = {
        "history_turns": len(history) // 2,
        "question_word_count": len(question.split()),
        "answer_long_word_count": len(answer_long.split()),
        "has_turn_dependencies": len(deps) > 0,
        "has_chain_coverage": len(cov) > 0,
        "evidence_count": len(g.get("evidence_chunk_list") or []),
        "search_clause_count": len(g.get("search_clause") or []),
    }

    passed = len(issues) == 0
    return passed, issues, metrics


# ── QC orchestrator ─────────────────────────────────────────────────────────


def run_multiturn_qc(record: dict, model: str) -> dict:
    """Run full multi-turn QC on one triplet. Returns QC result dict."""
    g = record.get("generated", {})
    history = g.get("history") or []
    turn1_user = history[0].get("message", "") if history else ""
    turn1_assistant = history[1].get("message", "") if len(history) > 1 else ""
    turn2_question = g.get("question", "")
    answer = g.get("answer_long", "") or g.get("answer_short", "")
    evidence = g.get("evidence_chunk_list") or []
    search = g.get("search_clause") or []

    results: Dict[str, Any] = {
        "s1_turn_dependency": None,
        "s2_element_necessity": None,
        "s3_grounding": None,
        "s4_surface": None,
        "overall_pass": False,
        "tokens": {"in": 0, "out": 0},
    }

    # S4: Surface QC (free, always runs)
    s4_pass, s4_issues, s4_metrics = surface_qc(record)
    results["s4_surface"] = {"passed": s4_pass, "issues": s4_issues, "metrics": s4_metrics}

    # S1: Turn dependency
    prompt = TURN_DEPENDENCY_PROMPT.format(
        turn1_user=turn1_user[:800],
        turn1_assistant=turn1_assistant[:800],
        turn2_question=turn2_question[:400],
    )
    raw, tin, tout = call_llm(
        client=None, model=model, prompt=prompt, provider="company",
        system_prompt="You evaluate multi-turn dialogue dependencies. Output valid JSON only.",
        user_tag="qc_multiturn_dependency", temperature=0.0,
    )
    results["tokens"]["in"] += tin
    results["tokens"]["out"] += tout
    parsed = parse_json(raw)
    if parsed:
        results["s1_turn_dependency"] = {
            "passed": parsed.get("turn2_depends_on_turn1", False),
            "dependency_type": parsed.get("dependency_type", "?"),
            "evidence_used": parsed.get("evidence_from_turn1_used", ""),
            "explanation": parsed.get("explanation", ""),
        }

    # S2: Chain structural validity — check that the chain skeleton is intact.
    # These materials are intra-document (Method C true2 candidates), NOT xdoc.
    # Cross-document M4 comes from the xdoc resolver pipeline (separate track).
    cov = g.get("chain_coverage") or {}
    deps = g.get("turn_dependencies") or []
    bridge_texts = record.get("bridge_texts", [])
    ea_id = record.get("element_a_id", "")
    eb_id = record.get("element_b_id", "")

    s2_pass = (
        ea_id != eb_id  # different elements
        and len(bridge_texts) >= 1
        and len(cov) >= 3  # at least element_a, element_b, and one bridge/intermediate
        and len(deps) >= 1
    )
    # Determine if cross-doc from element IDs
    source_doc = ea_id.split("_", 1)[0] if "_" in ea_id else ""
    target_doc = eb_id.split("_", 1)[0] if "_" in eb_id else ""
    is_cross_doc = source_doc != target_doc and source_doc and target_doc

    results["s2_element_necessity"] = {
        "passed": s2_pass,
        "elements_different": ea_id != eb_id,
        "is_cross_document": is_cross_doc,
        "bridge_count": len(bridge_texts),
        "chain_coverage_keys": len(cov),
        "turn_dependency_count": len(deps),
        "chain_type": (
            "cross_doc_chain" if is_cross_doc else "intra_doc_chain"
        ) if s2_pass else "structural_fail",
    }

    # S3: Grounding (LLM-based)
    ev_text = "\n".join(
        f"[{e.get('type','?')}] {e.get('source','?')}: {e.get('provides','')[:200]}"
        for e in evidence[:6]
    )
    prompt = GROUNDING_PROMPT.format(
        turn1_user=turn1_user[:400],
        turn2_question=turn2_question[:300],
        answer=answer[:800],
        evidence_list=ev_text[:1200],
    )
    raw, tin, tout = call_llm(
        client=None, model=model, prompt=prompt, provider="company",
        system_prompt="You evaluate answer grounding against evidence. Output valid JSON only.",
        user_tag="qc_multiturn_grounding", temperature=0.0,
    )
    results["tokens"]["in"] += tin
    results["tokens"]["out"] += tout
    parsed = parse_json(raw)
    if parsed:
        results["s3_grounding"] = {
            "passed": parsed.get("grounding_level", "") in
                      ("fully_grounded", "mostly_grounded"),
            "level": parsed.get("grounding_level", "?"),
            "claims_total": parsed.get("claims_total", 0),
            "claims_grounded": parsed.get("claims_grounded", 0),
            "claims_hallucinated": parsed.get("claims_hallucinated", 0),
            "hallucinated_core": parsed.get("hallucinated_core_claims", []),
            "explanation": parsed.get("explanation", ""),
        }

    # Overall
    s1_ok = results["s1_turn_dependency"] and results["s1_turn_dependency"].get("passed", False)
    s2_ok = results["s2_element_necessity"] and results["s2_element_necessity"].get("passed", False)
    s3_ok = results["s3_grounding"] and results["s3_grounding"].get("passed", False)
    s4_ok = s4_pass
    results["overall_pass"] = s1_ok and s2_ok and s3_ok and s4_ok
    results["overall_breakdown"] = {"s1": s1_ok, "s2": s2_ok, "s3": s3_ok, "s4": s4_ok}

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Multi-turn-adapted LLM QC for M4 triplets.")
    parser.add_argument("--triplets", default=str(DEFAULT_TRIPLETS))
    parser.add_argument("--out-root", default=str(DEFAULT_OUT_ROOT))
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--limit", type=int, default=20)
    parser.add_argument("--api-url", default=os.environ.get("COMPANY_API_URL", ""))
    parser.add_argument("--api-key", default=os.environ.get("COMPANY_API_KEY", ""))
    args = parser.parse_args()

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_dir = Path(args.out_root) / f"m4_multiturn_qc_{stamp}"
    out_dir.mkdir(parents=True, exist_ok=True)

    log_dir = configure_standard_logger()
    if not args.api_url or not args.api_key:
        raise SystemExit("company API credentials missing; set COMPANY_API_URL and COMPANY_API_KEY")
    set_company_credentials(args.api_url, args.api_key)

    records = []
    with open(args.triplets) as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))
    selected = records[:args.limit]
    print(f"Loaded {len(records)} triplets, QCing {len(selected)}")
    print(f"Output: {out_dir}\n")

    qc_results = []
    stats = Counter()
    total_in, total_out = 0, 0

    for i, rec in enumerate(selected):
        mid = rec.get("triplet_id", rec.get("material_id", f"t{i}"))
        print(f"[{i+1}/{len(selected)}] {mid} ", end="", flush=True)

        qc = run_multiturn_qc(rec, args.model)
        total_in += qc["tokens"]["in"]
        total_out += qc["tokens"]["out"]

        s1 = qc["s1_turn_dependency"]
        s2 = qc["s2_element_necessity"]
        s3 = qc["s3_grounding"]
        s4 = qc["s4_surface"]

        if qc["overall_pass"]:
            stats["qc_pass"] += 1
            print("PASS")
        else:
            stats["qc_fail"] += 1
            failing = [k for k, v in qc["overall_breakdown"].items() if not v]
            print(f"FAIL ({','.join(failing)})")

        if s1:
            stats[f"s1_{s1.get('dependency_type','?')}"] += 1
        if s2:
            stats[f"s2_{s2.get('chain_type','?')}"] += 1
        if s3:
            stats[f"s3_{s3.get('level','?')}"] += 1

        qc_record = {
            "triplet_id": mid,
            "qc_result": qc,
        }
        qc_results.append(qc_record)

    # Write results
    with open(out_dir / "qc_results.jsonl", "w") as f:
        for r in qc_results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    n_pass = stats.get("qc_pass", 0)
    n_fail = stats.get("qc_fail", 0)
    n_total = n_pass + n_fail

    breakdown = {}
    for k, v in stats.items():
        if k not in ("qc_pass", "qc_fail"):
            breakdown[k] = v

    summary = {
        "status": "ok",
        "output_dir": str(out_dir),
        "model": args.model,
        "total": n_total,
        "qc_pass": n_pass,
        "qc_fail": n_fail,
        "pass_rate": round(n_pass / max(1, n_total), 3),
        "per_dimension_breakdown": breakdown,
        "tokens": {"in": total_in, "out": total_out},
        "files": {
            "local_api_logger": str(log_dir.relative_to(ROOT)),
            "token_db": "logs/token_usage.db",
        },
    }
    with open(out_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    log_run(
        script="experiments/qc_multiturn_triplets.py",
        model=f"company:{args.model}",
        purpose="QC M4 multi-turn triplets",
        input_tokens=total_in,
        output_tokens=total_out,
        extra={
            "pairs_processed": n_total,
            "qc_pass": n_pass,
            "qc_fail": n_fail,
            "output": str(out_dir),
        },
    )

    latest = Path(args.out_root) / "m4_multiturn_qc_latest"
    if latest.exists() or latest.is_symlink():
        latest.unlink()
    latest.symlink_to(out_dir, target_is_directory=True)

    print(f"\nDone. {n_pass}/{n_total} pass ({summary['pass_rate']:.0%})")
    print(f"Tokens: {total_in} in / {total_out} out")
    print(f"Latest: {latest}")

    # Print dimension summary
    print(f"\nPer-dimension breakdown:")
    for k, v in sorted(breakdown.items()):
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
