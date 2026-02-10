#!/usr/bin/env python3
"""Generate L2 (cross-document) queries from candidate document pairs."""

from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

BAD_META_PATTERNS = [
    r"according to document", r"in document [ab]", r"from doc(ument)? [ab]",
]


def qc_l2_query(obj: Dict[str, Any]) -> List[str]:
    issues: List[str] = []
    q = obj.get("query", "").lower()
    a = obj.get("answer", "").lower()
    if any(re.search(p, q) for p in BAD_META_PATTERNS):
        issues.append("meta_language_query")
    if any(re.search(p, a) for p in BAD_META_PATTERNS):
        issues.append("meta_language_answer")
    refs = obj.get("evidence_refs", [])
    docs = {x.get("doc_id") for x in refs if isinstance(x, dict)}
    if len(docs) < 2:
        issues.append("missing_dual_doc_evidence")
    if len(obj.get("turns", [])) < 1:
        issues.append("empty_turns")
    if not obj.get("query"):
        issues.append("empty_query")
    return issues


def build_prompt(pair: Dict[str, Any]) -> str:
    shared = ", ".join(pair.get("shared_entities", [])[:8])
    examples = pair.get("evidence_examples", [])
    doc_a = pair["doc_a"]
    doc_b = pair["doc_b"]

    ex_text = []
    for i, e in enumerate(examples, 1):
        ex_text.append(
            f"Example {i} ({e.get('doc_id')}):\n"
            f"- query: {e.get('query','')}\n"
            f"- visual_anchor: {e.get('visual_anchor','')}\n"
            f"- text_evidence: {e.get('text_evidence','')}"
        )

    return f"""
You are generating ONE L2 cross-document query.

Document A: {doc_a}
Document B: {doc_b}
Shared concepts: {shared}

Reference evidence snippets:
{chr(10).join(ex_text)}

Requirements:
1) Query MUST require information from BOTH documents.
2) Do NOT use meta-language (e.g., "According to document A...").
3) Prefer a concrete comparison/synthesis question, not vague why-only speculation.
4) Output JSON only.
5) If the shared concepts do not support a meaningful comparison/synthesis, output:
   {{"status": "NULL", "reason": "..."}}

BAD example:
{{"query":"According to document A ... and document B ..."}}

GOOD example:
{{"query":"How does the reported behavior of <concept> differ when ...?"}}

Output format:
{{
  "turns": ["..."],
  "query": "...",
  "answer": "...",
  "evidence_refs": [
    {{"doc_id":"{doc_a}", "anchor":"..."}},
    {{"doc_id":"{doc_b}", "anchor":"..."}}
  ]
}}
""".strip()


def call_llm(provider: str, model: str, prompt: str) -> Optional[str]:
    try:
        if provider == "anthropic":
            import anthropic
            client = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))
            r = client.messages.create(
                model=model,
                max_tokens=800,
                messages=[{"role": "user", "content": prompt}],
            )
            return r.content[0].text
        if provider == "openai":
            from openai import OpenAI
            client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
            r = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                response_format={"type": "json_object"},
                max_tokens=800,
            )
            return r.choices[0].message.content
    except Exception as e:
        print(f"LLM error: {e}")
    return None


def parse_json(txt: Optional[str]) -> Optional[Dict[str, Any]]:
    if not txt:
        return None
    t = txt.strip()
    if t.startswith("```"):
        t = re.sub(r"^```(?:json)?", "", t).strip()
        t = re.sub(r"```$", "", t).strip()
    try:
        return json.loads(t)
    except Exception:
        return None


def main() -> None:
    ap = argparse.ArgumentParser(description="Generate L2 queries from candidate pairs")
    ap.add_argument("--pairs", default="data/l2_candidate_pairs_v1.json")
    ap.add_argument("--output", default="data/l2_queries_v1.jsonl")
    ap.add_argument("--limit", type=int, default=50)
    ap.add_argument("--provider", choices=["anthropic", "openai"], default="anthropic")
    ap.add_argument("--model", default="claude-sonnet-4-20250514")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    pair_data = json.loads(Path(args.pairs).read_text(encoding="utf-8"))
    pairs = pair_data.get("pairs", [])[: args.limit]
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)

    kept = 0
    with out.open("w", encoding="utf-8") as f:
        for i, p in enumerate(pairs):
            prompt = build_prompt(p)
            if args.dry_run:
                print(f"\n--- pair {i+1}: {p['doc_a']} vs {p['doc_b']} ---")
                print(prompt[:600] + "\n...")
                continue

            raw = call_llm(args.provider, args.model, prompt)
            obj = parse_json(raw)
            if not obj:
                continue

            if obj.get("status") == "NULL":
                continue

            obj["pair"] = {"doc_a": p["doc_a"], "doc_b": p["doc_b"]}
            obj["shared_entities"] = p.get("shared_entities", [])
            issues = qc_l2_query(obj)
            obj["qc_issues"] = issues
            obj["qc_pass"] = len(issues) == 0

            if obj["qc_pass"]:
                kept += 1
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")

    if args.dry_run:
        print(f"Dry-run complete for {len(pairs)} pairs")
    else:
        print(f"Generated {len(pairs)} rows, qc_pass={kept}")


if __name__ == "__main__":
    main()
