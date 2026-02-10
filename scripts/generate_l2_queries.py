#!/usr/bin/env python3
"""
Generate L2 cross-document queries from candidate pairs.

Takes L2 candidate pairs (from build_l2_candidates.py) and generates
cross-document queries using Claude API. Each query must require
evidence from BOTH documents to answer.

Modes:
  --dry-run : validate prompt construction, no API calls
  (default) : call Claude API and generate queries
"""

import argparse
import base64
import json
import os
import re
import sys
import time
from pathlib import Path

SYSTEM_PROMPT = """You are a data annotator creating cross-document retrieval training data.
You will receive evidence from TWO different research papers (Document A and Document B)
that share a common concept. Generate queries that REQUIRE both documents to answer.
Output valid JSON only, no other text, no markdown fences."""

USER_PROMPT_TEMPLATE = """Generate cross-document retrieval queries for these two related papers.

## Document A ({doc_a_id})
Figure caption: {doc_a_caption}
Visual elements: {doc_a_visual_anchor}
Key finding: {doc_a_answer}
Text evidence: {doc_a_text_evidence}

## Document B ({doc_b_id})
Figure caption: {doc_b_caption}
Visual elements: {doc_b_visual_anchor}
Key finding: {doc_b_answer}
Text evidence: {doc_b_text_evidence}

## Shared concepts: {shared_entities}

RULES — read carefully:

1. Each query is ONE question (max 25 words). No "and" joining two sub-questions.
2. The query must REQUIRE information from BOTH Document A AND Document B to answer.
   - Removing Document A's figure must make it unanswerable.
   - Removing Document B's text/figure must also make it unanswerable.
3. Include a specific visual anchor from at least one document (color, position, label, curve shape).
4. NEVER use meta-words: "Document A", "Document B", "the text", "the paper", "figure shows", "according to".
   Instead, refer to content by its actual subject matter.
5. Prefer these cross-document relationship types:
   - COMPARISON: "Why does method X achieve Y in one setting but Z in another?"
   - CONTRADICTION: "How can both findings be true given different assumptions?"
   - AGGREGATION: "What combined pattern emerges from both experimental setups?"
   - EVOLUTION: "How does the later approach improve on the earlier result?"
6. Each query must have evidence_a (from Doc A) and evidence_b (from Doc B).
7. The answer must cite specific facts from BOTH documents.

BAD query (only needs one document):
"What accuracy does the model achieve on German Credit?"

GOOD query (needs both):
"Why does the accuracy-fairness tradeoff curve flatten at 0.85 for logistic regression when a different study reports a steeper decline beyond that threshold?"

GOOD query (cross-document comparison):
"Does the sharp drop in the blue DI curve after threshold 0.5 contradict the claim that post-processing preserves utility above 0.8?"

Output JSON:
{{
  "queries": [
    {{
      "query": "single cross-document question, max 25 words",
      "answer": "must reference facts from BOTH documents, max 3 sentences",
      "evidence_a": {{
        "visual_anchor": "specific visual element from Doc A's figure",
        "text_evidence": "relevant text passage from Doc A"
      }},
      "evidence_b": {{
        "visual_anchor": "specific visual element from Doc B's figure (or 'N/A' if text-only)",
        "text_evidence": "relevant text passage from Doc B"
      }},
      "cross_doc_relationship": "comparison|contradiction|aggregation|evolution",
      "query_type": "value_context|comparison_explanation|anomaly_cause|visual_definition"
    }}
  ]
}}"""

# QC gate: meta-language patterns to ban
META_WORDS = [
    "document a", "document b", "doc a", "doc b", "the text",
    "the paper", "the caption", "the figure", "according to",
    "as mentioned", "as stated", "as described", "the section",
    "the paragraph", "first paper", "second paper",
]

BAN_PREFIXES = [
    "how does figure", "what does figure", "what does the figure",
    "what is shown", "what is depicted", "compare document",
]


def qc_l2_query(query_obj: dict) -> tuple[bool, list[str]]:
    """Quality-check a single L2 query. Returns (pass, [reasons])."""
    issues = []
    query = query_obj.get("query", "").lower().strip()

    # Meta-language
    for mw in META_WORDS:
        if mw in query:
            issues.append(f"meta_language:{mw}")
    for bp in BAN_PREFIXES:
        if query.startswith(bp):
            issues.append(f"banned_prefix:{bp}")

    # Evidence completeness
    ev_a = query_obj.get("evidence_a", {})
    ev_b = query_obj.get("evidence_b", {})
    if not ev_a.get("text_evidence") or len(ev_a.get("text_evidence", "")) < 20:
        issues.append("weak_evidence_a")
    if not ev_b.get("text_evidence") or len(ev_b.get("text_evidence", "")) < 20:
        issues.append("weak_evidence_b")

    # Visual anchor
    has_anchor = bool(
        (ev_a.get("visual_anchor", "").strip() and
         len(ev_a.get("visual_anchor", "")) > 5) or
        (ev_b.get("visual_anchor", "").strip() and
         len(ev_b.get("visual_anchor", "")) > 5)
    )
    if not has_anchor:
        issues.append("no_visual_anchor")

    # Query length
    word_count = len(query.split())
    if word_count > 35:
        issues.append(f"too_long:{word_count}")

    passed = len(issues) == 0
    return passed, issues


def build_prompt(pair: dict) -> str:
    """Build the generation prompt from a candidate pair."""
    return USER_PROMPT_TEMPLATE.format(
        doc_a_id=pair["doc_a"],
        doc_a_caption=pair.get("doc_a_caption", "(no caption)"),
        doc_a_visual_anchor=pair.get("doc_a_visual_anchor", "(none)"),
        doc_a_answer=pair.get("doc_a_answer", "(none)"),
        doc_a_text_evidence=pair.get("doc_a_text_evidence", "(none)"),
        doc_b_id=pair["doc_b"],
        doc_b_caption=pair.get("doc_b_caption", "(no caption)"),
        doc_b_visual_anchor=pair.get("doc_b_visual_anchor", "(none)"),
        doc_b_answer=pair.get("doc_b_answer", "(none)"),
        doc_b_text_evidence=pair.get("doc_b_text_evidence", "(none)"),
        shared_entities=", ".join(pair.get("shared_entities", [])[:8]),
    )


def call_api(client, prompt: str, model: str,
             image_a_path: str = None, image_b_path: str = None) -> dict:
    """Call Claude API with optional images."""
    content = []

    # Add images if available
    for img_path in [image_a_path, image_b_path]:
        if img_path and Path(img_path).exists():
            ext = Path(img_path).suffix.lower().lstrip(".")
            mime = {"jpg": "image/jpeg", "jpeg": "image/jpeg",
                    "png": "image/png"}.get(ext, "image/jpeg")
            with open(img_path, "rb") as f:
                img_b64 = base64.b64encode(f.read()).decode("utf-8")
            content.append({
                "type": "image",
                "source": {"type": "base64", "media_type": mime, "data": img_b64},
            })

    content.append({"type": "text", "text": prompt})

    response = client.messages.create(
        model=model,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": content}],
        temperature=0.7,
        max_tokens=2048,
    )

    raw = response.content[0].text.strip()
    parsed = None
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        json_match = re.search(r'\{.*\}', raw, re.DOTALL)
        if json_match:
            try:
                parsed = json.loads(json_match.group())
            except json.JSONDecodeError:
                pass

    return {
        "raw_response": raw,
        "parsed": parsed,
        "usage": {
            "input_tokens": response.usage.input_tokens,
            "output_tokens": response.usage.output_tokens,
        },
    }


def resolve_image_path(path: str) -> str:
    """Try to resolve image path: check relative, then absolute."""
    if Path(path).exists():
        return path
    # Try relative from repo root
    rel = Path(path)
    if rel.exists():
        return str(rel)
    return path  # return as-is, may not exist


def main():
    parser = argparse.ArgumentParser(
        description="Generate L2 cross-document queries via Claude API")
    parser.add_argument("--pairs", default="data/l2_candidate_pairs_v1.json",
                        help="Candidate pairs JSON")
    parser.add_argument("--output", default="data/l2_queries_v1.jsonl",
                        help="Output JSONL")
    parser.add_argument("--model", default="claude-sonnet-4-5-20250929",
                        help="Anthropic model")
    parser.add_argument("--limit", type=int, default=50,
                        help="Max pairs to process")
    parser.add_argument("--delay", type=float, default=0.5,
                        help="Seconds between API calls")
    parser.add_argument("--dry-run", action="store_true",
                        help="Validate prompts without calling API")
    args = parser.parse_args()

    with open(args.pairs) as f:
        data = json.load(f)

    pairs = data["pairs"][:args.limit]
    print(f"Processing {len(pairs)} candidate pairs"
          f"{' (DRY RUN)' if args.dry_run else ''}")

    if not args.dry_run:
        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            print("ERROR: ANTHROPIC_API_KEY not set.")
            print("Run: export $(grep -v '^#' .env | xargs)")
            sys.exit(1)
        import anthropic
        client = anthropic.Anthropic(api_key=api_key)

    results = []
    total_input_tokens = 0
    total_output_tokens = 0
    total_queries = 0
    total_passed_qc = 0
    total_failed_qc = 0

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as out_f:
        for i, pair in enumerate(pairs):
            print(f"  [{i+1}/{len(pairs)}] {pair['doc_a']} × {pair['doc_b']} "
                  f"(score={pair['score']:.1f})...", end=" ", flush=True)

            prompt = build_prompt(pair)

            if args.dry_run:
                print(f"OK (prompt: {len(prompt)} chars)")
                continue

            try:
                img_a = resolve_image_path(pair.get("doc_a_image_path", ""))
                img_b = resolve_image_path(pair.get("doc_b_image_path", ""))

                resp = call_api(client, prompt, args.model, img_a, img_b)
                total_input_tokens += resp["usage"]["input_tokens"]
                total_output_tokens += resp["usage"]["output_tokens"]

                parsed = resp.get("parsed")
                if not parsed or "queries" not in parsed:
                    print("FAIL (no queries parsed)")
                    continue

                queries = parsed["queries"]
                kept = 0
                for q in queries:
                    passed, issues = qc_l2_query(q)

                    entry = {
                        "query_id": f"l2_{pair['doc_a']}_{pair['doc_b']}_{total_queries}",
                        "query": q.get("query", ""),
                        "answer": q.get("answer", ""),
                        "doc_a": pair["doc_a"],
                        "doc_b": pair["doc_b"],
                        "doc_a_figure_id": pair["doc_a_figure_id"],
                        "doc_b_figure_id": pair["doc_b_figure_id"],
                        "doc_a_image_path": pair.get("doc_a_image_path", ""),
                        "doc_b_image_path": pair.get("doc_b_image_path", ""),
                        "evidence_a": q.get("evidence_a", {}),
                        "evidence_b": q.get("evidence_b", {}),
                        "shared_entities": pair["shared_entities"][:8],
                        "cross_doc_relationship": q.get("cross_doc_relationship", ""),
                        "query_type": q.get("query_type", ""),
                        "qc_passed": passed,
                        "qc_issues": issues,
                    }
                    out_f.write(json.dumps(entry, ensure_ascii=False) + "\n")
                    total_queries += 1
                    if passed:
                        total_passed_qc += 1
                        kept += 1
                    else:
                        total_failed_qc += 1

                print(f"OK ({len(queries)} gen, {kept} passed QC)")

            except Exception as e:
                print(f"ERROR: {e}")

            if args.delay > 0 and i < len(pairs) - 1:
                time.sleep(args.delay)

    # Summary
    if not args.dry_run:
        est_cost = total_input_tokens * 3 / 1e6 + total_output_tokens * 15 / 1e6
        print(f"\n{'='*60}")
        print(f"L2 Generation Summary")
        print(f"{'='*60}")
        print(f"  Pairs processed:     {len(pairs)}")
        print(f"  Total queries:       {total_queries}")
        print(f"  Passed QC:           {total_passed_qc}")
        print(f"  Failed QC:           {total_failed_qc}")
        print(f"  QC pass rate:        {total_passed_qc/max(total_queries,1)*100:.1f}%")
        print(f"  Input tokens:        {total_input_tokens:,}")
        print(f"  Output tokens:       {total_output_tokens:,}")
        print(f"  Est. cost:           ${est_cost:.2f}")
        print(f"  Output:              {args.output}")
        print(f"{'='*60}")
    else:
        print(f"\nDry run complete. {len(pairs)} prompts validated.")
        print("Remove --dry-run to call API.")


if __name__ == "__main__":
    main()
