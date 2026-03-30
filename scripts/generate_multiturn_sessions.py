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
    "You are a research assistant helping to create multi-turn reading comprehension "
    "conversations about academic papers. You generate natural, coherent dialogue turns "
    "that build toward a complex reasoning question step by step."
)

# ---------------------------------------------------------------------------
# Prompt templates
# ---------------------------------------------------------------------------

PROMPT_DECOMPOSE_L3 = """You are given a complex multi-hop question about academic papers, its answer, and a 3-step reasoning chain with evidence elements.

Your task: Generate turns 1 and 2 of a 3-turn conversation session where turn 3 is the original question.

ORIGINAL QUESTION (Turn 3 / Final):
{final_query}

ORIGINAL ANSWER:
{final_answer}

REASONING STEPS:
{steps_text}

REQUIREMENTS:
1. Turn 1 asks ONLY about the evidence in step 1. It must be answerable independently without any context.
2. Turn 2 uses natural coreference to reference turn 1's answer (use "this pattern", "that finding", "this imbalance", etc.), then asks about the evidence in step 2 and how it relates.
3. Turn 3 is EXACTLY the original question above (do not modify it).
4. Each turn's answer must be a complete, self-contained sentence or short paragraph.
5. Turns 1-2 answers must be fully supported by the cited evidence span only.
6. Turn 2 query must reference the prior turn naturally (ellipsis or pronoun).

OUTPUT FORMAT (JSON only, no extra text):
{{
  "turn1_query": "...",
  "turn1_answer": "...",
  "turn1_evidence_element_id": "{step1_eid}",
  "turn1_coreference_type": "none",
  "turn2_query": "...",
  "turn2_answer": "...",
  "turn2_evidence_element_id": "{step2_eid}",
  "turn2_coreference_type": "pronoun_reference|demonstrative_reference|ellipsis"
}}"""

PROMPT_DECOMPOSE_L2 = """You are given a dual-evidence question about academic papers, its answer, and two evidence spans.

Your task: Generate turn 1 of a 2-turn conversation session where turn 2 is the original question.

ORIGINAL QUESTION (Turn 2 / Final):
{final_query}

ORIGINAL ANSWER:
{final_answer}

EVIDENCE SPAN 1 ({eid1}):
{span1}

EVIDENCE SPAN 2 ({eid2}):
{span2}

REQUIREMENTS:
1. Turn 1 asks ONLY about evidence span 1. It must be a specific, focused question answerable from that span alone.
2. Turn 2 is EXACTLY the original question above (do not modify it).
3. Turn 1 answer must be a complete, self-contained sentence fully supported by span 1.
4. Turn 1 query must NOT leak the answer to turn 2.

OUTPUT FORMAT (JSON only, no extra text):
{{
  "turn1_query": "...",
  "turn1_answer": "...",
  "turn1_evidence_element_id": "{eid1}",
  "turn1_coreference_type": "none"
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
# QC
# ---------------------------------------------------------------------------

def qc_session(session: dict) -> Tuple[bool, List[str]]:
    issues = []
    turns = session.get("turns", [])
    if not turns:
        issues.append("no_turns")
        return False, issues

    for t in turns[:-1]:  # all but final turn
        q = t.get("query", "")
        a = t.get("answer", "")
        if len(q.split()) < 5:
            issues.append(f"turn{t['turn_id']}_query_too_short")
        if len(a.split()) < 5:
            issues.append(f"turn{t['turn_id']}_answer_too_short")
        if not t.get("evidence_element_id"):
            issues.append(f"turn{t['turn_id']}_missing_evidence_id")

    final = turns[-1]
    if not final.get("query"):
        issues.append("missing_final_query")

    # Check turn 2+ references prior context via coreference word
    for t in turns[1:-1]:
        q_low = (t.get("query") or "").lower()
        coreference_words = ["this", "that", "these", "those", "such", "the above",
                             "the finding", "the pattern", "the result", "the imbalance",
                             "it ", "they ", "its "]
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


def generate_l3_session(
    q: dict,
    client: Any,
    model: str,
    provider: str,
    delay: float,
) -> Tuple[Optional[dict], int, int]:
    steps = q.get("reasoning_steps") or []
    if len(steps) < 3:
        return None, 0, 0

    step1 = steps[0]
    step2 = steps[1]
    prompt = PROMPT_DECOMPOSE_L3.format(
        final_query=q["query"],
        final_answer=q["answer"],
        steps_text=build_steps_text(steps),
        step1_eid=step1.get("evidence_element_id", ""),
        step2_eid=step2.get("evidence_element_id", ""),
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

    session = {
        "session_id": f"mt_{q.get('query_id', q.get('pair_id', ''))}",
        "source_query_id": q.get("query_id", ""),
        "source_pair_id": q.get("pair_id", ""),
        "source_level": "l3",
        "total_turns": 3,
        "turns": turns,
    }
    qc_pass, qc_issues = qc_session(session)
    session["qc"] = {"pass": qc_pass, "issues": qc_issues}
    return session, in_tok, out_tok


def generate_l2_session(
    q: dict,
    client: Any,
    model: str,
    provider: str,
    delay: float,
) -> Tuple[Optional[dict], int, int]:
    spans = q.get("required_evidence_spans") or []
    if len(spans) < 2:
        return None, 0, 0

    s1, s2 = spans[0], spans[1]
    prompt = PROMPT_DECOMPOSE_L2.format(
        final_query=q["query"],
        final_answer=q["answer"],
        eid1=s1.get("element_id", ""),
        span1=s1.get("span", s1.get("content", "")),
        eid2=s2.get("element_id", ""),
        span2=s2.get("span", s2.get("content", "")),
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
        "total_turns": 2,
        "turns": turns,
    }
    qc_pass, qc_issues = qc_session(session)
    session["qc"] = {"pass": qc_pass, "issues": qc_issues}
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
                )
                print(f"\n--- L3 [{i}] {q.get('query_id', '')} ---")
                print(prompt[:500], "...")
            continue

        session, in_tok, out_tok = generate_l3_session(q, client, args.model, args.provider, args.delay)
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
                )
                print(f"\n--- L2 [{i}] {q.get('query_id', '')} ---")
                print(prompt[:400], "...")
            continue

        session, in_tok, out_tok = generate_l2_session(q, client, args.model, args.provider, args.delay)
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
