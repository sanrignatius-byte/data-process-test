#!/usr/bin/env python3
"""Generate multi-turn conversation sessions from existing L2/L3 pass queries.

Each single-hop L2 query → 2-turn session.
Each multi-hop L3 query (with reasoning_steps) → 3-turn session.

Strategy:
  - Turn 1..N-1: decomposed sub-questions, each grounded in one evidence element,
    with natural coreference/ellipsis linking turns
  - Turn N (final): the original complex query, which synthesizes all prior turns

The LLM generates the decomposed turns given the full query, answer, and evidence.

Usage:
    python scripts/generate_multiturn_sessions.py \
        --l2 data/m2/l2_production_2026-03-26_section_enriched_pass.jsonl \
        --l3 data/m2/l3_production_2026-03-26_section_enriched_pass.jsonl \
        --output data/m2/multiturn_sessions_v1.jsonl \
        --provider company \
        --model gpt-5.4 \
        --pass-only

Iron Rule: calls log_run() at end per CLAUDE.md requirement.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

from src.utils.token_logger import log_run  # noqa: E402

# ---------------------------------------------------------------------------
# Global state for company provider
# ---------------------------------------------------------------------------
_COMPANY_API_KEY: Optional[str] = None
_COMPANY_API_URL: Optional[str] = None

SYSTEM_PROMPT = (
    "You are simulating a researcher reading an academic paper with a specific analytical goal. "
    "You generate natural multi-turn conversations that build toward a complex inference step by step, "
    "using pronouns and demonstratives to reference prior turns so each question is only answerable "
    "in context."
)

# ---------------------------------------------------------------------------
# Prompt templates — Role-play Prompting (LLM-as-Simulator)
# Improvement 1: user persona with a stated goal; forces state-tracking dependency
# ---------------------------------------------------------------------------

PROMPT_DECOMPOSE_L3 = """You are simulating a PhD researcher who is reading a paper and progressively building toward a complex conclusion. The researcher's FINAL GOAL is to answer the question below, but they reach it by asking step-by-step questions — each turn depends on the previous turn's answer.

FINAL GOAL (Turn 3 — do NOT modify):
{final_query}

FINAL ANSWER (oracle, for your reference only — do not reveal it early):
{final_answer}

REASONING PATH (the logical steps the researcher will traverse):
{steps_text}

YOUR TASK: Generate Turn 1 and Turn 2 for this researcher.

STRICT RULES:
1. Turn 1: A focused question about Step 1's evidence. It must be answerable WITHOUT any prior context.
   — Do NOT mention the final goal or Step 2/3 evidence.
   — The answer should be a single factual observation from that evidence.

2. Turn 2: The researcher has now seen Turn 1's answer. They use it as a stepping stone.
   — MUST use a coreference expression (e.g. "this imbalance", "that pattern", "given what you found", "does this extend to...") to reference Turn 1's answer.
   — Asks about Step 2's evidence in light of Turn 1's finding.
   — If isolated from Turn 1's answer, Turn 2 must be UNANSWERABLE or ambiguous.

3. Turn 3 = EXACTLY the original final goal above.

INTENT SHIFT TYPE for this session: {intent_shift}
— "drill_down": Turn 2 zooms deeper into a specific detail from Turn 1's answer.
— "bridging": Turn 2 connects Turn 1's local finding to a broader mechanism.
— "contrastive": Turn 2 asks whether Turn 1's pattern holds or inverts in different evidence.

OUTPUT FORMAT (JSON only, no extra text):
{{
  "turn1_query": "...",
  "turn1_answer": "...",
  "turn1_evidence_element_id": "{step1_eid}",
  "turn1_coreference_type": "none",
  "turn2_query": "...",
  "turn2_answer": "...",
  "turn2_evidence_element_id": "{step2_eid}",
  "turn2_coreference_type": "pronoun_reference|demonstrative_reference|ellipsis",
  "turn2_context_dependency": "high|medium"
}}"""

PROMPT_DECOMPOSE_L2 = """You are simulating a PhD researcher who is reading a paper with a specific analytical goal. They ask a preliminary question first, then — using what they learned — ask the full synthesis question.

FINAL GOAL (Turn 2 — do NOT modify):
{final_query}

FINAL ANSWER (oracle, for your reference only):
{final_answer}

EVIDENCE SPAN 1 [{eid1}]:
{span1}

EVIDENCE SPAN 2 [{eid2}]:
{span2}

YOUR TASK: Generate Turn 1 for this researcher.

STRICT RULES:
1. Turn 1: A focused question about Evidence Span 1 only.
   — Must be answerable from Span 1 alone, without knowing the final goal.
   — Must NOT reveal or anticipate the answer to Turn 2.
   — Keep it to a single factual or observational question.

2. Turn 2 = EXACTLY the original final goal above (do not modify).

3. If Turn 1's query is extracted and given to a search engine alone, it should retrieve Span 1,
   but NOT directly answer the final goal.

INTENT SHIFT TYPE: {intent_shift}
— "drill_down": Turn 1 zooms into a specific detail of Span 1 that becomes a premise for Turn 2.
— "bridging": Turn 1 establishes a local fact; Turn 2 applies it to a larger cross-document claim.

OUTPUT FORMAT (JSON only, no extra text):
{{
  "turn1_query": "...",
  "turn1_answer": "...",
  "turn1_evidence_element_id": "{eid1}",
  "turn1_coreference_type": "none",
  "turn1_context_dependency": "none"
}}"""


# ---------------------------------------------------------------------------
# API helpers
# ---------------------------------------------------------------------------

def _collect_company_stream(stream_generator) -> Tuple[str, int, int]:
    text_parts = []
    in_tok = out_tok = 0
    for chunk in stream_generator:
        if not isinstance(chunk, str):
            continue
        for line in chunk.splitlines():
            line = line.strip()
            if not line or not line.startswith("data:"):
                continue
            raw = line[5:].strip()
            if raw == "[DONE]":
                continue
            try:
                obj = json.loads(raw)
            except Exception:
                continue
            for ch in (obj.get("choices") or []):
                delta = ch.get("delta") or {}
                txt = delta.get("content") or ""
                if txt:
                    text_parts.append(txt)
            usage = obj.get("usage") or {}
            if usage:
                in_tok = usage.get("prompt_tokens", in_tok)
                out_tok = usage.get("completion_tokens", out_tok)
    return "".join(text_parts), in_tok, out_tok


def call_api(
    client: Any,
    model: str,
    prompt: str,
    provider: str = "company",
) -> Tuple[Optional[str], int, int]:
    if provider == "openai":
        r = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            max_tokens=1024,
            temperature=0.4,
        )
        text = r.choices[0].message.content if r.choices else ""
        in_tok = int(getattr(getattr(r, "usage", None), "prompt_tokens", 0) or 0)
        out_tok = int(getattr(getattr(r, "usage", None), "completion_tokens", 0) or 0)
        return text, in_tok, out_tok

    if provider == "company":
        from local_api_logger import wrap_requests_call
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {_COMPANY_API_KEY}",
        }
        payload = {
            "model": model,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            "max_tokens": 1024,
            "temperature": 0.4,
            "stream": True,
            "stream_options": {"include_usage": True},
        }
        stream = wrap_requests_call(
            model=model,
            url=_COMPANY_API_URL,
            headers=headers,
            payload=payload,
            user="multiturn_sessions",
            verify=False,
        )
        text, in_tok, out_tok = _collect_company_stream(stream)
        return text, in_tok, out_tok

    # anthropic
    r = client.messages.create(
        model=model,
        max_tokens=1024,
        temperature=0.4,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": prompt}],
    )
    text = "".join(b.text for b in r.content if hasattr(b, "text"))
    in_tok = int(getattr(getattr(r, "usage", None), "input_tokens", 0) or 0)
    out_tok = int(getattr(getattr(r, "usage", None), "output_tokens", 0) or 0)
    return text, in_tok, out_tok


# ---------------------------------------------------------------------------
# JSON extraction
# ---------------------------------------------------------------------------

def extract_json(text: str) -> Optional[dict]:
    if not text:
        return None
    # Try direct parse
    try:
        return json.loads(text.strip())
    except Exception:
        pass
    # Extract first {...} block
    m = re.search(r'\{[\s\S]+\}', text)
    if m:
        try:
            return json.loads(m.group())
        except Exception:
            pass
    return None


# ---------------------------------------------------------------------------
# Context-Isolation score (Improvement 2)
# Proxy for the "Evaluation Pitfall" test: an intermediate turn q_t should be
# UNANSWERABLE if isolated from session history.
# We use Jaccard overlap between q_t tokens and the *final* query tokens as
# a proxy: high overlap → q_t is independently solvable (bad); low → context-dependent (good).
# Threshold: if Jaccard(q_t, q_final) > 0.35, the turn is likely independently solvable.
# ---------------------------------------------------------------------------

def _tokenize(text: str) -> set:
    return set(re.findall(r"[a-z][a-z0-9_-]{1,}", (text or "").lower()))


def context_isolation_score(turn_query: str, final_query: str, evidence_span: str) -> float:
    """Return a context-dependency score in [0, 1].
    Score close to 1 = turn is context-dependent (good).
    Score close to 0 = turn is independently solvable from evidence alone (bad).

    Logic:
    - High Jaccard with the final query → turn leaks final intent → low dependency score
    - High Jaccard with the evidence span → turn is directly grounded → low dependency score
    - Low overlap with both → turn requires prior context → high dependency score
    """
    turn_toks = _tokenize(turn_query)
    if not turn_toks:
        return 0.0

    final_toks = _tokenize(final_query)
    evidence_toks = _tokenize(evidence_span)

    def jaccard(a: set, b: set) -> float:
        if not a or not b:
            return 0.0
        return len(a & b) / len(a | b)

    leak_to_final = jaccard(turn_toks, final_toks)
    grounded_in_evidence = jaccard(turn_toks, evidence_toks)

    # Dependency score: penalise if the turn by itself could answer the final query,
    # or if the turn's tokens match the evidence so closely it's independently retrievable
    dependency = 1.0 - max(leak_to_final * 0.6, grounded_in_evidence * 0.4)
    return round(max(0.0, min(1.0, dependency)), 3)


# ---------------------------------------------------------------------------
# QC
# ---------------------------------------------------------------------------

def qc_session(session: dict) -> Tuple[bool, List[str]]:
    issues = []
    turns = session.get("turns", [])
    if not turns:
        issues.append("no_turns")
        return False, issues

    final_query = turns[-1].get("query", "") if turns else ""
    final_evidence_span = " ".join(
        str(s.get("span", "")) for s in (session.get("source_evidence_spans") or [])
    )

    for t in turns[:-1]:  # all but final turn
        q = t.get("query", "")
        a = t.get("answer", "")
        if len(q.split()) < 5:
            issues.append(f"turn{t['turn_id']}_query_too_short")
        if len(a.split()) < 5:
            issues.append(f"turn{t['turn_id']}_answer_too_short")
        if not t.get("evidence_element_ids"):
            issues.append(f"turn{t['turn_id']}_missing_evidence_id")

        # Context-Isolation check (Improvement 2)
        cis = context_isolation_score(q, final_query, final_evidence_span)
        t["context_isolation_score"] = cis
        if cis < 0.35:
            # Turn is too close to the final query — independently solvable
            issues.append(f"turn{t['turn_id']}_context_isolation_fail(score={cis})")

    final = turns[-1]
    if not final.get("query"):
        issues.append("missing_final_query")

    # Check turn 2+ references prior context via coreference word
    for t in turns[1:-1]:
        q_low = (t.get("query") or "").lower()
        coreference_words = ["this", "that", "these", "those", "such", "the above",
                             "the finding", "the pattern", "the result", "the imbalance",
                             "it ", "they ", "its ", "given ", "building on", "based on"]
        if not any(w in q_low for w in coreference_words):
            issues.append(f"turn{t['turn_id']}_missing_coreference")

    return len(issues) == 0, issues


# ---------------------------------------------------------------------------
# Session builders
# ---------------------------------------------------------------------------

def build_steps_text(steps: List[dict]) -> str:
    lines = []
    for s in steps:
        lines.append(
            f"Step {s.get('step_id', '?')}: [{s.get('evidence_element_id', '?')}] "
            f"{s.get('produces_claim', '')} | evidence: {s.get('evidence_span', '')}"
        )
    return "\n".join(lines)


# Improvement 3: intent shift cycling — rotate across session types for diversity
_INTENT_SHIFTS_L3 = ["drill_down", "bridging", "contrastive"]
_INTENT_SHIFTS_L2 = ["drill_down", "bridging"]


def _pick_intent_shift(shifts: List[str], idx: int) -> str:
    return shifts[idx % len(shifts)]


def generate_l3_session(
    q: dict,
    client: Any,
    model: str,
    provider: str,
    delay: float,
    session_idx: int = 0,
) -> Tuple[Optional[dict], int, int]:
    steps = q.get("reasoning_steps") or []
    if len(steps) < 3:
        return None, 0, 0

    step1 = steps[0]
    step2 = steps[1]
    intent_shift = _pick_intent_shift(_INTENT_SHIFTS_L3, session_idx)

    prompt = PROMPT_DECOMPOSE_L3.format(
        final_query=q["query"],
        final_answer=q["answer"],
        steps_text=build_steps_text(steps),
        step1_eid=step1.get("evidence_element_id", ""),
        step2_eid=step2.get("evidence_element_id", ""),
        intent_shift=intent_shift,
    )

    text, in_tok, out_tok = call_api(client, model, prompt, provider)
    time.sleep(delay)

    parsed = extract_json(text)
    if not parsed:
        return None, in_tok, out_tok

    turns = [
        {
            "turn_id": 1,
            "query": parsed.get("turn1_query", ""),
            "answer": parsed.get("turn1_answer", ""),
            "evidence_element_ids": [parsed.get("turn1_evidence_element_id", step1.get("evidence_element_id", ""))],
            "coreference_type": parsed.get("turn1_coreference_type", "none"),
            "depends_on_turns": [],
        },
        {
            "turn_id": 2,
            "query": parsed.get("turn2_query", ""),
            "answer": parsed.get("turn2_answer", ""),
            "evidence_element_ids": [parsed.get("turn2_evidence_element_id", step2.get("evidence_element_id", ""))],
            "coreference_type": parsed.get("turn2_coreference_type", "pronoun_reference"),
            "context_dependency": parsed.get("turn2_context_dependency", "high"),
            "depends_on_turns": [1],
        },
        {
            "turn_id": 3,
            "query": q["query"],
            "answer": q["answer"],
            "evidence_element_ids": [s.get("evidence_element_id", "") for s in steps],
            "coreference_type": "synthesis",
            "depends_on_turns": [1, 2],
        },
    ]

    # Provide source evidence spans so QC can compute context_isolation_score
    source_spans = q.get("required_evidence_spans") or []

    session = {
        "session_id": f"mt_{q.get('query_id', q.get('pair_id', ''))}",
        "source_query_id": q.get("query_id", ""),
        "source_pair_id": q.get("pair_id", ""),
        "source_level": "l3",
        "intent_shift_type": intent_shift,
        "total_turns": 3,
        "turns": turns,
        "source_evidence_spans": source_spans,
    }
    qc_pass, qc_issues = qc_session(session)
    session["qc"] = {"pass": qc_pass, "issues": qc_issues}
    # Remove internal field from final output
    session.pop("source_evidence_spans", None)
    return session, in_tok, out_tok


def generate_l2_session(
    q: dict,
    client: Any,
    model: str,
    provider: str,
    delay: float,
    session_idx: int = 0,
) -> Tuple[Optional[dict], int, int]:
    spans = q.get("required_evidence_spans") or []
    if len(spans) < 2:
        return None, 0, 0

    s1, s2 = spans[0], spans[1]
    intent_shift = _pick_intent_shift(_INTENT_SHIFTS_L2, session_idx)

    prompt = PROMPT_DECOMPOSE_L2.format(
        final_query=q["query"],
        final_answer=q["answer"],
        eid1=s1.get("element_id", ""),
        span1=s1.get("span", s1.get("content", "")),
        eid2=s2.get("element_id", ""),
        span2=s2.get("span", s2.get("content", "")),
        intent_shift=intent_shift,
    )

    text, in_tok, out_tok = call_api(client, model, prompt, provider)
    time.sleep(delay)

    parsed = extract_json(text)
    if not parsed:
        return None, in_tok, out_tok

    turns = [
        {
            "turn_id": 1,
            "query": parsed.get("turn1_query", ""),
            "answer": parsed.get("turn1_answer", ""),
            "evidence_element_ids": [parsed.get("turn1_evidence_element_id", s1.get("element_id", ""))],
            "coreference_type": parsed.get("turn1_coreference_type", "none"),
            "depends_on_turns": [],
        },
        {
            "turn_id": 2,
            "query": q["query"],
            "answer": q["answer"],
            "evidence_element_ids": [s.get("element_id", "") for s in spans],
            "coreference_type": "synthesis",
            "depends_on_turns": [1],
        },
    ]

    session = {
        "session_id": f"mt_{q.get('query_id', q.get('pair_id', ''))}",
        "source_query_id": q.get("query_id", ""),
        "source_pair_id": q.get("pair_id", ""),
        "source_level": "l2",
        "intent_shift_type": intent_shift,
        "total_turns": 2,
        "turns": turns,
        "source_evidence_spans": spans,
    }
    qc_pass, qc_issues = qc_session(session)
    session["qc"] = {"pass": qc_pass, "issues": qc_issues}
    session.pop("source_evidence_spans", None)
    return session, in_tok, out_tok


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------

def load_jsonl(path: Path) -> List[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_jsonl(rows: List[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Generate multi-turn sessions from L2/L3 queries")
    parser.add_argument("--l2", type=Path,
                        default=_PROJECT_ROOT / "data/m2/l2_production_2026-03-26_section_enriched_pass.jsonl",
                        help="L2 pass queries input")
    parser.add_argument("--l3", type=Path,
                        default=_PROJECT_ROOT / "data/m2/l3_production_2026-03-26_section_enriched_pass.jsonl",
                        help="L3 pass queries input")
    parser.add_argument("--output", type=Path,
                        default=_PROJECT_ROOT / "data/m2/multiturn_sessions_v1.jsonl",
                        help="Output JSONL file")
    parser.add_argument("--pass-only", action="store_true",
                        help="Write only QC-passing sessions to output")
    parser.add_argument("--provider", choices=["anthropic", "openai", "company"], default="company")
    parser.add_argument("--model", default="gpt-5.4")
    parser.add_argument("--delay", type=float, default=0.5, help="Delay between API calls (s)")
    parser.add_argument("--limit", type=int, default=None, help="Max sessions to generate (per level)")
    parser.add_argument("--dry-run", action="store_true", help="Print prompts without calling API")
    parser.add_argument("--company-api-key", default=None)
    parser.add_argument("--company-api-url", default=None)
    args = parser.parse_args()

    global _COMPANY_API_KEY, _COMPANY_API_URL

    # Load environment
    env_path = _PROJECT_ROOT / ".env"
    if env_path.exists():
        for line in env_path.read_text().splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip())

    # Setup client
    client = None
    if args.provider == "company":
        _COMPANY_API_KEY = args.company_api_key or os.environ.get("COMPANY_API_KEY")
        _COMPANY_API_URL = args.company_api_url or os.environ.get("COMPANY_API_URL", "https://yunwu.ai/v1/chat/completions")
        if not args.dry_run:
            if not _COMPANY_API_KEY:
                print("ERROR: COMPANY_API_KEY not set")
                sys.exit(1)
            if not _COMPANY_API_URL:
                print("ERROR: COMPANY_API_URL not set")
                sys.exit(1)
    elif args.provider == "openai":
        from openai import OpenAI
        client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
    else:
        import anthropic
        client = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))

    # Load queries
    l2_queries: List[dict] = []
    l3_queries: List[dict] = []
    if args.l2 and args.l2.exists():
        l2_queries = load_jsonl(args.l2)
        print(f"Loaded {len(l2_queries)} L2 queries from {args.l2}")
    if args.l3 and args.l3.exists():
        l3_queries = load_jsonl(args.l3)
        print(f"Loaded {len(l3_queries)} L3 queries from {args.l3}")

    if args.limit:
        l2_queries = l2_queries[:args.limit]
        l3_queries = l3_queries[:args.limit]

    total_in_tok = total_out_tok = 0
    sessions: List[dict] = []
    n_pass = n_fail = n_skip = 0

    # Process L3 (3-turn sessions)
    print(f"\nGenerating L3 sessions ({len(l3_queries)} queries)...")
    for i, q in enumerate(l3_queries):
        if args.dry_run:
            steps = q.get("reasoning_steps") or []
            if len(steps) >= 3:
                prompt = PROMPT_DECOMPOSE_L3.format(
                    final_query=q["query"],
                    final_answer=q["answer"],
                    steps_text=build_steps_text(steps),
                    step1_eid=steps[0].get("evidence_element_id", ""),
                    step2_eid=steps[1].get("evidence_element_id", ""),
                    intent_shift=_pick_intent_shift(_INTENT_SHIFTS_L3, i),
                )
                print(f"\n--- L3 [{i}] {q.get('query_id', '')} ({_pick_intent_shift(_INTENT_SHIFTS_L3, i)}) ---")
                print(prompt[:500], "...")
            continue

        session, in_tok, out_tok = generate_l3_session(q, client, args.model, args.provider, args.delay, session_idx=i)
        total_in_tok += in_tok
        total_out_tok += out_tok

        if session is None:
            n_skip += 1
            continue

        sessions.append(session)
        if session["qc"]["pass"]:
            n_pass += 1
        else:
            n_fail += 1

        if (i + 1) % 10 == 0:
            print(f"  L3 {i+1}/{len(l3_queries)} | pass={n_pass} fail={n_fail} skip={n_skip}", flush=True)

    # Process L2 (2-turn sessions)
    print(f"\nGenerating L2 sessions ({len(l2_queries)} queries)...")
    for i, q in enumerate(l2_queries):
        if args.dry_run:
            spans = q.get("required_evidence_spans") or []
            if len(spans) >= 2:
                s1, s2 = spans[0], spans[1]
                prompt = PROMPT_DECOMPOSE_L2.format(
                    final_query=q["query"],
                    final_answer=q["answer"],
                    eid1=s1.get("element_id", ""),
                    span1=s1.get("span", ""),
                    eid2=s2.get("element_id", ""),
                    span2=s2.get("span", ""),
                    intent_shift=_pick_intent_shift(_INTENT_SHIFTS_L2, i),
                )
                print(f"\n--- L2 [{i}] {q.get('query_id', '')} ({_pick_intent_shift(_INTENT_SHIFTS_L2, i)}) ---")
                print(prompt[:400], "...")
            continue

        session, in_tok, out_tok = generate_l2_session(q, client, args.model, args.provider, args.delay, session_idx=i)
        total_in_tok += in_tok
        total_out_tok += out_tok

        if session is None:
            n_skip += 1
            continue

        sessions.append(session)
        if session["qc"]["pass"]:
            n_pass += 1
        else:
            n_fail += 1

        if (i + 1) % 10 == 0:
            print(f"  L2 {i+1}/{len(l2_queries)} | pass={n_pass} fail={n_fail} skip={n_skip}", flush=True)

    if args.dry_run:
        print("\nDRY RUN complete. No sessions written.")
        return

    # Write output
    to_write = [s for s in sessions if s["qc"]["pass"]] if args.pass_only else sessions
    write_jsonl(to_write, args.output)
    print(f"\nWrote {len(to_write)} sessions to {args.output}")
    print(f"pass={n_pass} fail={n_fail} skip={n_skip} total_turns={sum(s['total_turns'] for s in to_write)}")

    # Also write pass-only subset if not already doing so
    if not args.pass_only:
        pass_path = args.output.with_name(args.output.stem + "_pass" + args.output.suffix)
        pass_sessions = [s for s in sessions if s["qc"]["pass"]]
        write_jsonl(pass_sessions, pass_path)
        print(f"Wrote {len(pass_sessions)} pass sessions to {pass_path}")

    # Iron Rule: log token usage
    log_run(
        script="generate_multiturn_sessions",
        model=f"{args.provider}:{args.model}",
        purpose="Generate multi-turn conversation sessions from L2/L3 pass queries",
        input_tokens=total_in_tok,
        output_tokens=total_out_tok,
        extra={
            "l2_queries": len(l2_queries),
            "l3_queries": len(l3_queries),
            "sessions_generated": len(sessions),
            "sessions_pass": n_pass,
            "sessions_fail": n_fail,
            "sessions_skip": n_skip,
            "output": str(args.output),
        },
    )


if __name__ == "__main__":
    main()
