#!/usr/bin/env python3
"""Chain-to-session projection v2 (DocTalk-inspired discourse planning).

v2 upgrade over v1:
  1. Discourse planning phase: LLM plans conversation structure BEFORE verbalization
  2. Topic shift detection + coreference seed injection
  3. Explicit dependency scaffolding ("Given [X from Turn 1], ...")
  4. Higher turn_dependency rate target: >50% (v1 was 30.6%)

Pipeline: chains → discourse plan → verbalization → style pass → turn-dep QC
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

from src.api.llm import call_llm, parse_json, set_company_credentials
from src.utils.token_logger import log_run

DEFAULT_MODEL = "gpt-5.4"

L3_FILES = [
    "data/03_queries/l3_enriched_v3_rerun2_pass.jsonl",
    "data/03_queries/l3_enriched_v3_new82_rerun2_pass.jsonl",
]

# ── Phase 0: Discourse Planning ─────────────────────────────────────────────

DISCOURSE_PLANNER_SYSTEM = """You are a discourse architect. Given a multi-hop reasoning chain,
plan the conversational flow for a 2-turn (or 3-turn if chain depth >= 3) dialogue.

Your job is to decide:
1. topic_shift: what new information does each turn introduce?
2. coreference_seeds: specific phrases Turn N+1 MUST use to reference Turn N's answer
3. dependency_graph: which turns depend on which prior turns

RULES:
- Every turn after Turn 1 MUST have at least one coreference seed that explicitly
  references a specific finding, number, or named concept from the prior turn.
- A coreference seed is a concrete phrase like "Given that the error rate is 12%..."
  or "Since demographic parity drops to 0.62..." — NOT generic like "based on this".
- If the chain has reasoning_depth >= 3, plan 3 turns. Otherwise plan 2 turns.
- Topic shifts should progress from observation → attribution → explanation/prediction.

Return only valid JSON:
{
  "num_turns": 2,
  "discourse_plan": [
    {
      "turn_id": 1,
      "topic_shift": "Establish the empirical observation from the first evidence element",
      "coreference_seeds_from_prior": [],
      "dependency": "none"
    },
    {
      "turn_id": 2,
      "topic_shift": "Attribute the observation to a specific cause, then explain mathematically",
      "coreference_seeds_from_prior": [
        "Given that [SPECIFIC FINDING from Turn 1 with numbers]...",
        "If [NAMED EFFECT from Turn 1] holds..."
      ],
      "dependency": "must_reference_turn1"
    }
  ]
}"""


def build_discourse_planning_prompt(chain: dict[str, Any]) -> str:
    reasoning = chain.get("reasoning_chain", "")
    steps = chain.get("reasoning_steps", [])
    depth = chain.get("reasoning_depth", len(steps) or 2)

    step_descriptions = ""
    for i, step in enumerate(steps):
        step_descriptions += (
            f"Step {i+1}: {step.get('evidence_type', '?')} — "
            f"{step.get('produces_claim', '')}\n"
        )
    if not step_descriptions:
        step_descriptions = f"Reasoning chain:\n{reasoning}"

    return f"""Plan the discourse for this reasoning chain ({depth} steps):

{step_descriptions}

Original query (context): {chain.get('query', '')}
Original answer (context): {chain.get('answer', '')[:500]}

Plan {min(depth, 3)} conversational turns where each turn after the first MUST
reference a concrete finding from the prior turn's answer."""


# ── Phase 1: Verbalization (discourse-guided) ───────────────────────────────

VERBALIZATION_SYSTEM_V2 = """You are a precise dialogue writer. Given a discourse plan and
reasoning chain, write natural conversational turns between a researcher (User) and
a knowledgeable colleague (Assistant).

CRITICAL RULES:
1. Every User turn after Turn 1 MUST start with one of the coreference_seeds_from_prior,
   filled in with the ACTUAL concrete finding from the previous Assistant turn.
2. Assistant answers MUST begin by restating the key fact from the prior exchange,
   then add new evidence.
3. NO meta-language: figure, table, equation, chart, plot, diagram, paper, section, study.
4. Use concrete numbers and named entities. "error is 12%" not "performance varies".
5. Keep each turn concise (2-3 sentences).

Return only valid JSON with these keys for each turn:
{
  "turns": [
    {"turn_id": 1, "user": "...", "assistant": "..."},
    {"turn_id": 2, "user": "...", "assistant": "..."}
  ]
}"""


def build_discourse_verbalization_prompt(
    chain: dict[str, Any],
    discourse_plan: dict[str, Any],
) -> str:
    return f"""Write the dialogue following this discourse plan:

{json.dumps(discourse_plan, indent=2)}

Reasoning chain:
{chain.get('reasoning_chain', '')}

Key evidence:
{chain.get('answer', '')[:600]}

IMPORTANT: Fill in the coreference_seeds with ACTUAL concrete findings from the
chain (specific numbers, named effects, technical terms). Do NOT leave placeholders."""


# ── Support functions ───────────────────────────────────────────────────────

def load_l3_pass_chains() -> list[dict[str, Any]]:
    chains: list[dict[str, Any]] = []
    seen: set[str] = set()
    for rel in L3_FILES:
        path = ROOT / rel
        if not path.exists():
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


def validate_session(session: dict[str, Any] | None) -> dict[str, Any]:
    if not isinstance(session, dict):
        return {"valid": False, "reason": "not_object"}
    turns = session.get("turns", [])
    if not turns or len(turns) < 2:
        return {"valid": False, "reason": "too_few_turns"}

    # Check coreference in non-first turns
    for turn in turns[1:]:
        t2 = str(turn.get("user", ""))
        coref_markers = [
            "it", "they", "them", "their", "its",
            "this", "that", "these", "those",
            "such", "the same", "this finding", "that approach",
            "given that", "since", "under those", "if that",
            "based on that", "following that",
        ]
        has_coref = any(m in t2.lower() for m in coref_markers)
        if not has_coref:
            return {"valid": False, "reason": f"no_coref_in_turn{turn.get('turn_id', '?')}"}

    banned = ["figure", "table", "equation", "chart", "plot", "diagram"]
    for turn in turns:
        text = str(turn.get("user", "")).lower()
        hits = [w for w in banned if w in text.split()]
        if hits:
            return {"valid": False, "reason": f"meta_language:{','.join(hits)}"}

    return {"valid": True, "reason": "ok"}


def run_turn_dependency_qc(
    session: dict[str, Any],
    model: str,
    qc_idx: int,
    total_in: int,
    total_out: int,
) -> tuple[dict[str, Any], int, int]:
    turns = session.get("turns", [])
    if len(turns) < 2:
        return {"verdict": "independent", "explanation": "only one turn"}, total_in, total_out

    # Test: erase turn1_assistant, re-ask turn2_user
    turn1 = turns[0]
    turn2 = turns[1]
    prompt = f"""Multi-turn dependency test.

Turn 1 (User asked, Assistant answered — but the answer is now ERASED):
Q: {turn1.get('user', '')}

The Assistant's response to Turn 1 has been deleted.

Now Turn 2 arrives:
Q: {turn2.get('user', '')}

Can you answer Turn 2 without knowing what the Assistant said in Turn 1?

CRITICAL: Does Turn 2's question contain a reference (pronoun, "this finding", "given that X",
"since Y", "under those conditions") that can ONLY be resolved by knowing Turn 1's answer?
If yes → dependent. If Turn 2 is self-contained → independent.

Return JSON:
{{
  "verdict": "dependent | independent",
  "explanation": "one sentence identifying the specific dependency or lack thereof"
}}"""

    raw, tin, tout = call_llm(
        client=None,
        model=model,
        provider="company",
        prompt=prompt,
        system_prompt="You are a turn-dependency evaluator. Return valid JSON only.",
        max_tokens=200,
        temperature=0.0,
        user_tag="chain_to_session_v2_qc",
    )
    total_in += tin
    total_out += tout
    parsed = parse_json(raw or "")
    verdict = parsed.get("verdict", "parse_failed") if isinstance(parsed, dict) else "parse_failed"
    return {
        "qc_index": qc_idx,
        "verdict": verdict,
        "explanation": parsed.get("explanation", "") if isinstance(parsed, dict) else "",
        "tokens": {"in": tin, "out": tout},
    }, total_in, total_out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=os.environ.get("COMPANY_API_MODEL") or DEFAULT_MODEL)
    parser.add_argument("--output-dir", default="")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--max-tokens", type=int, default=1200)
    parser.add_argument("--skip-qc", action="store_true")
    parser.add_argument("--cross-doc-only", action="store_true", default=True)
    parser.add_argument("--skip-discourse-planning", action="store_true",
                        help="Skip discourse planning (fall back to v1-style verbalization)")
    args = parser.parse_args()

    # Output dir
    if args.output_dir:
        out_dir = Path(args.output_dir)
        if not out_dir.is_absolute():
            out_dir = ROOT / out_dir
    else:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        out_dir = ROOT / f"data/05_eval/chain_to_session_v2_{stamp}"
    out_dir.mkdir(parents=True, exist_ok=True)

    latest = ROOT / "data/05_eval/chain_to_session_v2_latest"
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

    # Output files
    sessions_path = out_dir / "sessions.jsonl"
    plans_path = out_dir / "discourse_plans.jsonl"
    failures_path = out_dir / "failures.jsonl"
    qc_path = out_dir / "qc_results.jsonl"

    total_in = 0
    total_out = 0
    stats = Counter()

    for idx, chain in enumerate(chains, 1):
        qid = chain.get("query_id", f"unknown_{idx}")

        try:
            # Phase 0: Discourse planning
            disc_plan = None
            if not args.skip_discourse_planning:
                plan_prompt = build_discourse_planning_prompt(chain)
                raw_plan, tin_p, tout_p = call_llm(
                    client=None, model=args.model, provider="company",
                    prompt=plan_prompt, system_prompt=DISCOURSE_PLANNER_SYSTEM,
                    max_tokens=400, temperature=0.0,
                    user_tag="discourse_planner",
                )
                total_in += tin_p
                total_out += tout_p
                disc_plan = parse_json(raw_plan or "")

                if isinstance(disc_plan, dict):
                    with plans_path.open("a", encoding="utf-8") as f:
                        f.write(json.dumps({"query_id": qid, "plan": disc_plan},
                                          ensure_ascii=False) + "\n")
                        f.flush()
                    stats["plans_ok"] += 1
                else:
                    stats["plans_parse_fail"] += 1
                    disc_plan = None  # fall through to direct verbalization

            # Phase 1: Verbalization
            if disc_plan:
                verb_prompt = build_discourse_verbalization_prompt(chain, disc_plan)
            else:
                # Fallback: v1-style
                verb_prompt = build_discourse_planning_prompt(chain)

            raw_verb, tin_v, tout_v = call_llm(
                client=None, model=args.model, provider="company",
                prompt=verb_prompt,
                system_prompt=VERBALIZATION_SYSTEM_V2 if disc_plan else
                    "Decompose this reasoning chain into conversational turns. Return JSON.",
                max_tokens=args.max_tokens, temperature=0.0,
                user_tag="chain_to_session_v2_verb",
            )
            total_in += tin_v
            total_out += tout_v
            session = parse_json(raw_verb or "")

            if not isinstance(session, dict) or "turns" not in session:
                with failures_path.open("a", encoding="utf-8") as f:
                    f.write(json.dumps(
                        {"query_id": qid, "error": "parse_fail", "raw": raw_verb[:500]},
                        ensure_ascii=False) + "\n")
                stats["verbalization_fail"] += 1
                print(f"[{idx:03d}/{len(chains):03d}] {qid} -> VERB_PARSE_FAIL")
                continue

            # Validate
            validation = validate_session(session)
            if not validation["valid"]:
                stats[f"validate_fail:{validation['reason']}"] += 1
                stats["validate_fail"] += 1
                continue

            # Phase 2: Turn-dependency QC
            if not args.skip_qc:
                qc_result, total_in, total_out = run_turn_dependency_qc(
                    session, args.model, idx, total_in, total_out
                )
                session["turn_dependency_qc"] = qc_result
                if qc_result["verdict"] == "dependent":
                    stats["dependent"] += 1
                else:
                    stats["independent"] += 1
            else:
                stats["qc_skipped"] += 1

            session["query_id"] = qid
            session["has_discourse_plan"] = disc_plan is not None
            stats["sessions_ok"] += 1

            with sessions_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(session, ensure_ascii=False) + "\n")
                f.flush()

            if idx % 20 == 0 or idx == len(chains):
                dep_rate = stats["dependent"] / max(stats["dependent"] + stats["independent"], 1)
                print(
                    f"[{idx:03d}/{len(chains):03d}] "
                    f"ok={stats['sessions_ok']} "
                    f"dependent={stats['dependent']} "
                    f"dep_rate={dep_rate:.2%} "
                    f"(tok_in={total_in}, tok_out={total_out})"
                )

        except Exception as e:
            stats["exception"] += 1
            with failures_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(
                    {"query_id": qid, "error": str(e)[:300]},
                    ensure_ascii=False) + "\n")
            print(f"[{idx:03d}/{len(chains):03d}] {qid} -> EXCEPTION: {e}")

    # Summary
    summary = {
        "status": "ok",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "model": args.model,
        "input_chains": len(chains),
        "sessions_generated": stats["sessions_ok"],
        "verbalization_failures": stats["verbalization_fail"],
        "discourse_plans_ok": stats["plans_ok"],
        "turn_dependency_qc": {
            "total": stats["sessions_ok"],
            "dependent": stats["dependent"],
            "independent": stats["independent"],
            "dependency_rate": round(
                stats["dependent"] / max(stats["dependent"] + stats["independent"], 1), 4
            ),
        },
        "tokens": {"in": total_in, "out": total_out},
        "files": {
            "sessions": str(sessions_path.relative_to(ROOT)),
            "discourse_plans": str(plans_path.relative_to(ROOT)),
            "failures": str(failures_path.relative_to(ROOT)),
        },
    }
    summary_path = out_dir / "summary.json"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(f"\nSummary written to {summary_path}")
    dep_rate = summary["turn_dependency_qc"]["dependency_rate"]
    print(f"Turn dependency rate: {dep_rate:.2%}")
    print(f"Sessions: {stats['sessions_ok']}")
    print(f"Discourse plans: {stats['plans_ok']}")

    log_run(
        script="build_chain_to_session_v2",
        model=f"company:{args.model}",
        purpose=f"DocTalk discourse planning → {len(chains)} chains → {stats['sessions_ok']} sessions",
        input_tokens=total_in,
        output_tokens=total_out,
        extra={
            "chains_processed": len(chains),
            "sessions_generated": stats["sessions_ok"],
            "dependency_rate": dep_rate,
            "output": str(out_dir),
        },
    )


if __name__ == "__main__":
    main()
