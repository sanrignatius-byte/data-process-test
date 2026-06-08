#!/usr/bin/env python3
"""
Lightweight query generator for product documentation pairs.

Unlike the full generate_multihop_l1_queries.py (designed for academic papers
with figures/tables/formulas), this script handles text-heavy product pages
with simple section+paragraph pairs. Uses a minimal prompt that asks for
factual dual-evidence questions.

Usage:
  python scripts/generate_huawei_queries.py \
    --candidates data/03_queries/huawei_l1_candidates.json \
    --output data/03_queries/huawei_l1_queries.jsonl \
    --limit 20
"""

import argparse, json, os, re, sys, time
from pathlib import Path
from typing import Any, Dict, List

import openai

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

# ── Prompt template for product documentation pairs ──
SYSTEM_PROMPT = (
    "You are a technical documentation analyst. "
    "Your task is to generate retrieval queries that require reading "
    "multiple sections of a document to answer correctly. "
    "Output valid JSON only, no markdown fences."
)

PROMPT_PRODUCT_DOC = """You are reading a technical product document. Below are two sections that together contain information needed to answer a question.

## Section A ({elem_a_type})
{elem_a_text}

## Section B ({elem_b_type})
{elem_b_text}

## YOUR TASK
Generate 2 questions that require BOTH sections to answer correctly.
The questions should test whether someone has read and understood both sections.

## RULES
1. Each question must need information from BOTH sections — removing either section should make the question unanswerable.
2. Questions should be natural, like what a user or engineer would actually ask about this product.
3. Max 30 words per question. Answer max 3 sentences.
4. DO NOT use meta-language like "according to section A" or "as shown above".
5. DO NOT make up information not present in the text.
6. Each question should have a different focus (e.g., one about configuration, one about behavior).

## Output format (JSON only):
{{
  "queries": [
    {{
      "query": "natural question requiring both sections (max 30 words)",
      "answer": "direct answer citing details from both sections (max 3 sentences)",
      "query_type": "dual_evidence_factual",
      "required_evidence_spans": [
        {{"element_id": "{elem_a_id}", "span": "key detail from section A", "evidence_type": "text"}},
        {{"element_id": "{elem_b_id}", "span": "key detail from section B", "evidence_type": "text"}}
      ]
    }},
    {{
      "query": "second question, different focus (max 30 words)",
      "answer": "direct answer citing details from both sections (max 3 sentences)",
      "query_type": "dual_evidence_factual",
      "required_evidence_spans": [
        {{"element_id": "{elem_a_id}", "span": "key detail from section A", "evidence_type": "text"}},
        {{"element_id": "{elem_b_id}", "span": "key detail from section B", "evidence_type": "text"}}
      ]
    }}
  ]
}}"""


def self_parse_json(text: str) -> dict | None:
    """Parse JSON from LLM response, handling markdown fences."""
    import re
    text = (text or "").strip()
    # Remove markdown fences
    text = re.sub(r'^```(?:json)?\s*\n?', '', text)
    text = re.sub(r'\n?```\s*$', '', text)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        # Try extracting JSON object
        m = re.search(r'\{[^{}]*\}', text, re.DOTALL)
        if m:
            try:
                return json.loads(m.group())
            except json.JSONDecodeError:
                pass
        return None


def get_element_text(el: dict) -> str:
    """Get best available text content from an element."""
    # Try enriched_content first
    enriched = (el.get("enriched_content") or "").strip()
    if enriched and len(enriched) > 50:
        return enriched[:1200]

    # Try content
    content = (el.get("content") or "").strip()
    if content and len(content) > 20:
        return content[:1200]

    # Try label/caption
    label = (el.get("label") or el.get("caption") or "").strip()
    if label:
        return label[:1200]

    return "(no text content available)"


def build_prompt(pair: dict) -> str:
    """Build minimal prompt for a product doc pair."""
    ea = pair.get("element_a", {})
    eb = pair.get("element_b", {})

    return PROMPT_PRODUCT_DOC.format(
        elem_a_type=ea.get("element_type", "element"),
        elem_a_id=ea.get("element_id", ""),
        elem_a_text=get_element_text(ea),
        elem_b_type=eb.get("element_type", "element"),
        elem_b_id=eb.get("element_id", ""),
        elem_b_text=get_element_text(eb),
    )


def generate_queries(
    candidates_path: str,
    output_path: str,
    provider: str = "company",
    model: str = None,
    limit: int = 0,
    delay: float = 0.5,
    dry_run: bool = False,
) -> dict:
    """Generate queries for candidate pairs."""
    candidates = json.loads(Path(candidates_path).read_text(encoding="utf-8"))
    pairs = candidates.get("pairs", [])
    if limit > 0:
        pairs = pairs[:limit]

    print(f"Product Doc Query Generator")
    print(f"  Candidates: {len(pairs)}")
    print(f"  Provider: {provider}")
    print(f"  Model: {model or 'auto'}")
    print(f"  Output: {output_path}")

    if dry_run:
        for i, pair in enumerate(pairs[:5]):
            print(f"\n--- Pair {i+1}: {pair['pair_id'][:60]} ---")
            print(build_prompt(pair)[:500])
        return {}

    # Set up OpenAI-compatible client
    api_url = os.environ.get("COMPANY_API_URL", "")
    api_key = os.environ.get("COMPANY_API_KEY", "")
    if not api_url or not api_key:
        print("ERROR: COMPANY_API_URL and COMPANY_API_KEY must be set")
        sys.exit(1)
    
    client = openai.OpenAI(base_url=api_url.rsplit("/v1", 1)[0] + "/v1", api_key=api_key)
    effective_model = model or "gpt-5.4"

    outputs = []
    ok = fail = empty = 0
    total_in = total_out = 0

    for i, pair in enumerate(pairs):
        pair_id = pair.get("pair_id", f"pair_{i}")
        ptype = pair.get("pair_type", "?")
        prompt = build_prompt(pair)

        if dry_run:
            print(f"\n  [{i+1}/{len(pairs)}] {pair_id[:50]} ({ptype})")
            print(f"  Prompt: {prompt[:300]}...")
            continue

        try:
            response = client.chat.completions.create(
                model=effective_model,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.3,
                max_tokens=800,
            )
            raw = response.choices[0].message.content or ""
            total_in += response.usage.prompt_tokens if response.usage else 0
            total_out += response.usage.completion_tokens if response.usage else 0
        except Exception as e:
            fail += 1
            print(f"  [{i+1}/{len(pairs)}] API ERROR: {e}")
            continue

        obj = self_parse_json(raw)

        if not obj:
            empty += 1
            print(f"  [{i+1}/{len(pairs)}] PARSE FAIL: {raw[:100]}")
            continue

        queries = obj.get("queries", [])
        if not queries:
            empty += 1
            print(f"  [{i+1}/{len(pairs)}] NO QUERIES")
            continue

        # Attach pair metadata
        for q in queries:
            q["pair_id"] = pair_id
            q["doc_id"] = pair.get("doc_id", "")
            q["pair_type"] = ptype
            q["element_a_id"] = pair.get("element_a_id", "")
            q["element_b_id"] = pair.get("element_b_id", "")
            q["element_a_type"] = pair.get("element_a_type", "")
            q["element_b_type"] = pair.get("element_b_type", "")

        outputs.extend(queries)
        ok += 1
        print(f"  [{i+1}/{len(pairs)}] OK: {len(queries)} queries from {pair_id[:50]}")

        time.sleep(delay)

    # Write output
    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        for q in outputs:
            f.write(json.dumps(q, ensure_ascii=False) + "\n")



    summary = {
        "total_pairs": len(pairs),
        "ok": ok,
        "parse_fail": fail,
        "empty_queries": empty,
        "total_queries": len(outputs),
        "input_tokens": total_in,
        "output_tokens": total_out,
    }

    print(f"\n{'='*50}")
    print(f"  Summary: {ok} ok, {fail} fail, {empty} empty")
    print(f"  Queries generated: {len(outputs)}")
    print(f"  Tokens: {total_in} in, {total_out} out")
    print(f"  Output: {output_path}")

    return summary


def main():
    ap = argparse.ArgumentParser(description="Lightweight query generator for product docs")
    ap.add_argument("--candidates", default="data/03_queries/huawei_l1_candidates.json")
    ap.add_argument("--output", default="data/03_queries/huawei_l1_queries.jsonl")
    ap.add_argument("--provider", default="company")
    ap.add_argument("--model", default=None)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--delay", type=float, default=0.5)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    generate_queries(
        candidates_path=args.candidates,
        output_path=args.output,
        provider=args.provider,
        model=args.model,
        limit=args.limit,
        delay=args.delay,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    main()
