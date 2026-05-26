#!/usr/bin/env python3
"""Chain-to-session projection v1 (idea:006 Phase 0).

Reads L3 pass chains, runs locked-schema verbalization to extract structured
turn content, then generates 2-turn sessions with coreference.

v1 constraint: L3 pass data has empty reasoning_steps[], so we do a 2-turn
endpoint projection from the reasoning_chain + path, extracting:
  turn1_question, turn1_answer, bridge_pivot, turn2_question, turn2_answer.

This is Track A / experimental-lane. Output goes under data/05_eval/.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.api import call_llm, parse_json, set_company_credentials
from src.utils.token_logger import log_run

DEFAULT_MODEL = "gpt-5.4"

L3_FILES = [
    "data/03_queries/l3_enriched_v3_rerun2_pass.jsonl",
    "data/03_queries/l3_enriched_v3_new82_rerun2_pass.jsonl",
]

VERBALIZATION_SYSTEM = """You are a precise structural parser. Extract structured fields from a multi-hop reasoning chain.

The input describes a chain: element_a -> bridge paragraph -> element_b.
Your job is to decompose this into two conversational turns where Turn 2 CANNOT be answered without Turn 1's answer.

CRITICAL — Turn-dependency rules:
- turn1_answer MUST contain at least one specific finding with concrete numbers, named entities, or technical terms.
- turn2_question MUST explicitly reference that specific finding from turn1_answer. Use phrases like "Given that [specific finding from turn1]...", "If [specific number] holds...", "Since [named effect] is present..."
- turn2_question must be UNANSWERABLE without knowing turn1_answer. A reader who only sees turn2_question must say "I need to know what turn1 said first."
- turn2_answer must start by restating the key finding from turn1_answer, then add new evidence from element_b.

Other rules:
- No meta-language (figure, table, equation, chart, plot, diagram, paper, section).
- Use concrete observations: say "error rate is 12%" not "the performance varies".
- Keep answers concise (2-3 sentences each).

Return only valid JSON with these exact keys:
{
  "turn1_question": "...",
  "turn1_answer": "...",
  "bridge_pivot": "...",
  "turn2_question": "...",
  "turn2_answer": "..."
}"""


def load_l3_pass_chains() -> list[dict[str, Any]]:
    chains: list[dict[str, Any]] = []
    seen: set[str] = set()
    for rel in L3_FILES:
        path = ROOT / rel
        if not path.exists():
            print(f"WARNING: {rel} not found, skipping")
            continue
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                row = json.loads(line)
                qid = row.get("query_id", "")
                if qid in seen:
                    continue
                seen.add(qid)
                chains.append(row)
    return chains


def is_cross_doc(chain: dict[str, Any]) -> bool:
    docs: set[str] = set()
    for node in chain.get("path", []):
        if "::p::" not in str(node):
            doc_id = str(node).split("_", 1)[0]
            docs.add(doc_id)
    return len(docs) >= 2


def build_verbalization_prompt(chain: dict[str, Any]) -> str:
    element_ids = chain.get("element_ids", [])
    element_types = f"{chain.get('element_a_type', '?')} + {chain.get('element_b_type', '?')}"
    return f"""Decompose this reasoning chain into two conversational turns.

Element A: {element_ids[0] if len(element_ids) > 0 else '?'} ({element_types.split('+')[0].strip()})
Element B: {element_ids[1] if len(element_ids) > 1 else '?'} ({element_types.split('+')[1].strip() if '+' in element_types else '?'})

Reasoning chain:
{chain.get('reasoning_chain', '')}

Original query (for context):
{chain.get('query', '')}

Original answer (for context):
{chain.get('answer', '')}"""


def build_session_prompt(
    chain: dict[str, Any],
    verb: dict[str, Any],
) -> str:
    """Style pass: add persona + natural coreference to the raw projection."""
    element_ids = chain.get("element_ids", [])
    return f"""Rewrite this 2-turn academic dialogue to sound natural, as if a researcher asks a colleague a follow-up that genuinely depends on the first answer.

Core content to preserve:
- Turn 1 question: {verb.get('turn1_question', '')}
- Turn 1 answer: {verb.get('turn1_answer', '')}
- Turn 2 question: {verb.get('turn2_question', '')}
- Turn 2 answer: {verb.get('turn2_answer', '')}

CRITICAL RULES:
1. Turn 2 question MUST explicitly reference a specific fact, number, or named concept from Turn 1's answer. A reader seeing Turn 2 alone must think "I need Turn 1's answer first."
2. Turn 2 question MUST contain at least one referring expression (pronoun, "this finding", "that result", "given that", "since", "under those conditions") that binds to Turn 1's answer.
3. Turn 2 answer MUST begin by restating the key fact from Turn 1, then extend with new evidence.
4. NO meta-language: figure, table, equation, chart, plot, diagram, paper, study, section.
5. Use concrete observations with numbers where possible.
6. Keep each turn concise (2-3 sentences).

Return only valid JSON:
{{
  "turn1_user": "...",
  "turn1_assistant": "...",
  "turn2_user": "...",
  "turn2_assistant": "..."
}}"""


def validate_session(session: dict[str, Any] | None) -> dict[str, Any]:
    if not isinstance(session, dict):
        return {"valid": False, "reason": "not_object"}
    required = ["turn1_user", "turn1_assistant", "turn2_user", "turn2_assistant"]
    missing = [k for k in required if k not in session]
    if missing:
        return {"valid": False, "reason": f"missing:{','.join(missing)}"}

    t2 = str(session.get("turn2_user", ""))
    coref_markers = [
        "it", "they", "them", "their", "its",
        "this", "that", "these", "those",
        "such", "the same", "this finding", "that approach",
    ]
    has_coref = any(m in t2.lower().split() for m in coref_markers)
    if not has_coref:
        return {"valid": False, "reason": "no_coref_in_turn2"}

    banned = ["figure", "table", "equation", "chart", "plot", "diagram"]
    for turn_key in ["turn1_user", "turn2_user"]:
        text = str(session.get(turn_key, "")).lower()
        hits = [w for w in banned if w in text.split()]
        if hits:
            return {"valid": False, "reason": f"meta_language_in_{turn_key}:{','.join(hits)}"}

    return {"valid": True, "reason": "ok"}


def run_turn_dependency_qc(
    session: dict[str, Any],
    model: str,
    out_dir: Path,
    qc_idx: int,
) -> dict[str, Any]:
    """Test: delete turn1_assistant, re-ask turn2_user. If LLM can still answer, fail."""
    prompt = f"""You are testing whether Turn 2 genuinely depends on Turn 1's answer.

Turn 1 (user asked, and you answered — but now your answer has been ERASED):
Q: {session.get('turn1_user', '')}

Your answer to Turn 1 is GONE. You do not remember what you said.

Now Turn 2 arrives:
Q: {session.get('turn2_user', '')}

Can you answer Turn 2 without knowing what you said in Turn 1?

CRITICAL TEST: Read Turn 2 carefully. Does it reference a specific finding, number, or fact that only exists in Turn 1's answer (which is now erased)? If yes -> DEPENDENT. If Turn 2 is a generic follow-up that can be answered from general knowledge -> INDEPENDENT.

Return JSON:
{{
  "verdict": "dependent | independent",
  "explanation": "one sentence identifying what specific information Turn 2 needs from Turn 1, or why it doesn't need any"
}}"""

    raw, tin, tout = call_llm(
        client=None,
        model=model,
        provider="company",
        prompt=prompt,
        system_prompt="You are a turn-dependency evaluator. Return valid JSON only.",
        max_tokens=300,
        temperature=0.0,
        user_tag="chain_to_session_qc",
    )
    parsed = parse_json(raw or "")
    verdict = parsed.get("verdict", "parse_failed") if isinstance(parsed, dict) else "parse_failed"
    return {
        "qc_index": qc_idx,
        "verdict": verdict,
        "explanation": parsed.get("explanation", "") if isinstance(parsed, dict) else "",
        "tokens": {"in": tin, "out": tout},
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=os.environ.get("COMPANY_API_MODEL") or DEFAULT_MODEL)
    parser.add_argument("--output-dir", default="")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--max-tokens", type=int, default=1200)
    parser.add_argument("--skip-qc", action="store_true")
    parser.add_argument("--cross-doc-only", action="store_true", default=True)
    args = parser.parse_args()

    # Output dir
    if args.output_dir:
        out_dir = Path(args.output_dir)
        if not out_dir.is_absolute():
            out_dir = ROOT / out_dir
    else:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        out_dir = ROOT / f"data/05_eval/chain_to_session_v1_{stamp}"
    out_dir.mkdir(parents=True, exist_ok=True)

    latest = ROOT / "data/05_eval/chain_to_session_v1_latest"
    if latest.is_symlink() or latest.exists():
        latest.unlink()
    latest.symlink_to(out_dir.resolve())

    # Setup
    from local_api_logger.logger import APILogger
    import local_api_logger.tracker as tracker
    log_dir = ROOT / "api_logs_cannt_delete"
    log_dir.mkdir(parents=True, exist_ok=True)
    tracker._default_tracker = tracker.APITracker(APILogger(str(log_dir)))

    set_company_credentials(
        os.environ.get("COMPANY_API_URL", ""),
        os.environ.get("COMPANY_API_KEY", ""),
    )

    # Load chains
    all_chains = load_l3_pass_chains()
    print(f"Loaded {len(all_chains)} L3 pass chains")

    if args.cross_doc_only:
        chains = [c for c in all_chains if is_cross_doc(c)]
        print(f"Cross-doc filter: {len(chains)} chains")
    else:
        chains = all_chains

    if args.limit:
        chains = chains[: args.limit]
        print(f"Limited to {len(chains)} chains")

    # Phase 1: Locked verbalization
    sessions_path = out_dir / "sessions.jsonl"
    failed_verb_path = out_dir / "verbalization_failures.jsonl"
    qc_path = out_dir / "qc_results.jsonl"

    total_in = 0
    total_out = 0
    sessions: list[dict[str, Any]] = []

    print("\n--- Phase 1: Verbalization ---")
    for idx, chain in enumerate(chains, 1):
        qid = chain.get("query_id", f"unknown_{idx}")

        # Step 1a: Verbalize
        vprompt = build_verbalization_prompt(chain)
        raw_v, tin_v, tout_v = call_llm(
            client=None, model=args.model, provider="company",
            prompt=vprompt, system_prompt=VERBALIZATION_SYSTEM,
            max_tokens=args.max_tokens, temperature=0.0,
            user_tag="chain_to_session_verb",
        )
        total_in += tin_v
        total_out += tout_v
        verb = parse_json(raw_v or "")

        if not isinstance(verb, dict):
            with failed_verb_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps({"query_id": qid, "raw": raw_v}, ensure_ascii=False) + "\n")
            print(f"[{idx:03d}/{len(chains):03d}] {qid} -> VERB_PARSE_FAIL")
            continue

        # Step 1b: Style pass
        sprompt = build_session_prompt(chain, verb)
        raw_s, tin_s, tout_s = call_llm(
            client=None, model=args.model, provider="company",
            prompt=sprompt,
            system_prompt="You are a science communicator. Make academic dialogue sound natural. Return valid JSON only.",
            max_tokens=args.max_tokens, temperature=0.4,
            user_tag="chain_to_session_style",
        )
        total_in += tin_s
        total_out += tout_s
        session = parse_json(raw_s or "")

        if not isinstance(session, dict):
            with failed_verb_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps({"query_id": qid, "stage": "style", "raw": raw_s}, ensure_ascii=False) + "\n")
            print(f"[{idx:03d}/{len(chains):03d}] {qid} -> STYLE_PARSE_FAIL")
            continue

        validation = validate_session(session)
        session_row = {
            "session_id": f"c2s_v1_{qid}",
            "source_chain": qid,
            "docs": list({
                str(n).split("_", 1)[0]
                for n in chain.get("path", [])
                if "::p::" not in str(n)
            }),
            "element_ids": chain.get("element_ids", []),
            "element_types": f"{chain.get('element_a_type','?')}+{chain.get('element_b_type','?')}",
            "turn1_user": session.get("turn1_user", ""),
            "turn1_assistant": session.get("turn1_assistant", ""),
            "turn2_user": session.get("turn2_user", ""),
            "turn2_assistant": session.get("turn2_assistant", ""),
            "bridge_pivot": verb.get("bridge_pivot", ""),
            "rule_validation": validation,
            "tokens_verb": {"in": tin_v, "out": tout_v},
            "tokens_style": {"in": tin_s, "out": tout_s},
        }
        sessions.append(session_row)
        with sessions_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(session_row, ensure_ascii=False) + "\n")
        print(f"[{idx:03d}/{len(chains):03d}] {qid} -> {validation['reason']}")

    print(f"\nPhase 1 complete: {len(sessions)} sessions generated")
    print(f"Verbalization failures: {len(chains) - len(sessions)}")

    # Phase 2: Turn-dependency QC
    if not args.skip_qc and sessions:
        print("\n--- Phase 2: Turn-dependency QC ---")
        qc_results: list[dict[str, Any]] = []
        for idx, s in enumerate(sessions, 1):
            qc = run_turn_dependency_qc(s, args.model, out_dir, idx)
            total_in += qc["tokens"]["in"]
            total_out += qc["tokens"]["out"]
            qc["session_id"] = s["session_id"]
            qc_results.append(qc)
            with qc_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(qc, ensure_ascii=False) + "\n")
            print(f"QC [{idx:03d}/{len(sessions):03d}] {s['session_id']} -> {qc['verdict']}")

        dep_count = sum(1 for q in qc_results if q["verdict"] == "dependent")
        ind_count = sum(1 for q in qc_results if q["verdict"] == "independent")
        fail_count = sum(1 for q in qc_results if q["verdict"] == "parse_failed")
        print(f"QC done: dependent={dep_count}, independent={ind_count}, parse_fail={fail_count}")
    else:
        qc_results = []
        dep_count = ind_count = fail_count = 0

    # Summary
    rule_pass = sum(1 for s in sessions if s["rule_validation"]["valid"])
    rule_fail = len(sessions) - rule_pass

    summary = {
        "status": "ok",
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "model": args.model,
        "input_chains": len(chains),
        "sessions_generated": len(sessions),
        "verbalization_failures": len(chains) - len(sessions),
        "rule_qc": {"pass": rule_pass, "fail": rule_fail},
        "turn_dependency_qc": {
            "total": len(qc_results),
            "dependent": dep_count,
            "independent": ind_count,
            "parse_fail": fail_count,
            "dependency_rate": round(dep_count / len(qc_results), 4) if qc_results else 0,
        },
        "tokens": {"in": total_in, "out": total_out},
        "files": {
            "sessions": str(sessions_path.relative_to(ROOT)),
            "qc_results": str(qc_path.relative_to(ROOT)),
        },
    }
    (out_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8",
    )
    print(f"\nSummary: {summary}")
    print(f"Output: {out_dir}")
    print(f"Latest: {latest}")

    log_run(
        script="experiments/build_chain_to_session_v1.py",
        model=f"company:{args.model}",
        purpose="Chain-to-session projection v1 (idea:006 Phase 0)",
        input_tokens=total_in,
        output_tokens=total_out,
        extra={
            "input_chains": len(chains),
            "sessions_generated": len(sessions),
            "dependency_rate": summary["turn_dependency_qc"]["dependency_rate"],
            "output": str(out_dir.relative_to(ROOT)),
        },
    )


if __name__ == "__main__":
    main()
