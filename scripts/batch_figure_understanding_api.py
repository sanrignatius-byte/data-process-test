#!/usr/bin/env python3
"""
Batch Figure Understanding via Anthropic Claude API.

No GPU needed — runs on login node. Processes figure-text pairs
and generates L1 cross-modal queries with strict QC.
"""

import argparse
import base64
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Dict

import anthropic


SYSTEM_PROMPT = "You are a data annotator creating cross-modal retrieval training data. Output valid JSON only, no other text, no markdown fences."

USER_PROMPT_TEMPLATE = """Generate cross-modal retrieval queries for this academic figure.

Caption: {caption}
Text before: {context_before}
Text after: {context_after}
References: {references}

RULES — read carefully:

1. Each query is ONE question (max 25 words, no "and" joining two sub-questions).
2. The query must FUSE figure and text: changing the figure must change the answer, AND removing the text must also make it unanswerable.
3. Include a specific visual anchor (color, position, label, value) in the query itself.
4. NEVER use meta-words: "text", "caption", "figure", "paper", "section", "according to", "as mentioned". Refer to content directly.
5. Each of the 3 queries must cite a DIFFERENT text passage as evidence.
6. Prefer comparison/trend/anomaly queries over pure value-reading.

BAD query (concatenated, meta-language):
"What accuracy does RLR reach at 0.95 in the plot, and what does the text state about algorithm differences?"

GOOD query (fused, no meta-language):
"Does RLR's 0.68 accuracy at fairness=0.95 support the claim that repair performance varies across algorithms?"

GOOD query (comparison):
"Why does the solid blue curve overtake the dashed red one only after epoch 12, given that both use the same base architecture?"

GOOD query (anomaly):
"What causes the sharp spike at step 5000 in the green loss curve, despite the stated constant learning rate?"

Output JSON:
{{
  "figure_type": "plot|diagram|architecture|table|example|photo|other",
  "visual_elements": ["element with value/position", "..."],
  "queries": [
    {{
      "query": "single fused question, max 25 words, no meta-language",
      "answer": "factual, max 2 sentences",
      "visual_anchor": "specific element (e.g. 'red dashed line at y=0.85')",
      "text_evidence": "direct quote from the provided context, min 50 chars, different per query",
      "query_type": "value_context|comparison_explanation|anomaly_cause|visual_definition"
    }}
  ]
}}"""


def encode_image_base64(image_path: str) -> str:
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def build_prompt(pair: Dict) -> str:
    refs = pair.get("referring_paragraphs", [])
    refs_text = "\n".join(f"- {r[:300]}" for r in refs[:3]) if refs else "(none)"
    return USER_PROMPT_TEMPLATE.format(
        caption=pair.get("caption", "(no caption)"),
        context_before=pair.get("context_before", "(none)")[:500],
        context_after=pair.get("context_after", "(none)")[:500],
        references=refs_text,
    )


def process_pair(client: anthropic.Anthropic, pair: Dict, model: str) -> Dict:
    """Process a single figure-text pair via Anthropic API."""
    img_path = pair["image_path"]
    ext = Path(img_path).suffix.lower().lstrip(".")
    mime = {"jpg": "image/jpeg", "jpeg": "image/jpeg", "png": "image/png"}.get(ext, "image/jpeg")
    img_b64 = encode_image_base64(img_path)

    text_prompt = build_prompt(pair)

    response = client.messages.create(
        model=model,
        system=SYSTEM_PROMPT,
        messages=[
            {"role": "user", "content": [
                {"type": "image", "source": {"type": "base64", "media_type": mime, "data": img_b64}},
                {"type": "text", "text": text_prompt},
            ]},
        ],
        temperature=0.7,
        max_tokens=2048,
    )

    raw = response.content[0].text.strip()
    parsed = None
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        # Try extracting JSON from markdown fences or mixed text
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


def main():
    parser = argparse.ArgumentParser(description="Batch figure understanding via Anthropic Claude API")
    parser.add_argument("--input", required=True, help="figure_text_pairs.json")
    parser.add_argument("--output", required=True, help="Output JSON path")
    parser.add_argument("--model", default="claude-sonnet-4-5-20250929", help="Anthropic model")
    parser.add_argument("--min-quality", type=float, default=0.5)
    parser.add_argument("--max-pairs", type=int, default=0, help="Limit pairs (0=all)")
    parser.add_argument("--delay", type=float, default=0.5, help="Seconds between API calls")
    args = parser.parse_args()

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        print("ERROR: ANTHROPIC_API_KEY not set. Run: export $(grep -v '^#' .env | xargs)")
        sys.exit(1)

    client = anthropic.Anthropic(api_key=api_key)

    # Load pairs
    with open(args.input) as f:
        all_pairs = json.load(f)

    pairs_to_process = []
    for doc_id, pairs in all_pairs.items():
        for pair in pairs:
            if pair.get("quality_score", 0) >= args.min_quality:
                pair["_doc_id"] = doc_id
                img = Path(pair["image_path"])
                if img.exists() and img.stat().st_size > 1000:
                    pairs_to_process.append(pair)

    if args.max_pairs > 0:
        pairs_to_process = pairs_to_process[:args.max_pairs]

    print(f"Processing {len(pairs_to_process)} pairs with {args.model}...")

    # Process
    results = []
    total_input_tokens = 0
    total_output_tokens = 0

    for i, pair in enumerate(pairs_to_process):
        print(f"  [{i+1}/{len(pairs_to_process)}] {pair['figure_id']}...", end=" ", flush=True)

        try:
            resp = process_pair(client, pair, args.model)
            parsed = resp["parsed"]
            total_input_tokens += resp["usage"]["input_tokens"]
            total_output_tokens += resp["usage"]["output_tokens"]

            result = {
                "doc_id": pair["_doc_id"],
                "figure_id": pair["figure_id"],
                "figure_number": pair.get("figure_number"),
                "image_path": pair["image_path"],
                "caption": pair.get("caption", ""),
                "original_quality_score": pair.get("quality_score", 0),
                "raw_response": resp["raw_response"],
                "parsed": parsed,
            }

            if parsed and ("queries" in parsed or "cross_modal_queries" in parsed):
                result["queries"] = parsed.get("queries", parsed.get("cross_modal_queries", []))
                result["figure_type_mllm"] = parsed.get("figure_type", "unknown")
                result["visual_elements"] = parsed.get("visual_elements", [])
                print(f"OK ({len(result.get('queries', []))} queries)")
            else:
                print("FAIL (no queries parsed)")

            results.append(result)

        except Exception as e:
            print(f"ERROR: {e}")
            results.append({
                "doc_id": pair["_doc_id"],
                "figure_id": pair["figure_id"],
                "image_path": pair["image_path"],
                "error": str(e),
            })

        if args.delay > 0 and i < len(pairs_to_process) - 1:
            time.sleep(args.delay)

    # Save full results
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    # Save filtered queries JSONL
    queries_path = output_path.with_name("l1_cross_modal_queries_v3.jsonl")
    query_count = 0
    dropped = 0

    # Meta-language ban list
    meta_words = ["the text", "the caption", "the paper",
                  "according to", "as mentioned", "as stated",
                  "as described", "the section", "the paragraph"]
    ban_prefixes = ["how does figure", "what does figure",
                    "what does the figure", "what is shown", "what is depicted"]

    repo_roots = ["/projects/_hdd/myyyx1/data-process-test/",
                  "/projects/myyyx1/data-process-test/"]

    with open(queries_path, 'w', encoding='utf-8') as f:
        for result in results:
            for q in result.get("queries", []):
                visual_anchor = q.get("visual_anchor", "")
                text_evidence = q.get("text_evidence", "")
                query_text = q.get("query", "")
                query_lower = query_text.lower().strip()

                # Ban meta-language
                if any(mw in query_lower for mw in meta_words):
                    dropped += 1
                    continue
                # Ban shallow patterns
                if any(query_lower.startswith(bp) for bp in ban_prefixes):
                    dropped += 1
                    continue
                # Require visual anchor
                if not visual_anchor or len(visual_anchor) < 5:
                    dropped += 1
                    continue
                # Require substantial text evidence
                if not text_evidence or len(text_evidence) < 50:
                    dropped += 1
                    continue

                # Normalize image path
                img_path = result.get("image_path", "")
                for root in repo_roots:
                    if img_path.startswith(root):
                        img_path = img_path[len(root):]
                        break

                entry = {
                    "query_id": f"l1_{result['doc_id']}_{result['figure_id']}_{query_count}",
                    "query": query_text,
                    "answer": q.get("answer", ""),
                    "doc_id": result["doc_id"],
                    "figure_id": result["figure_id"],
                    "figure_number": result.get("figure_number"),
                    "image_path": img_path,
                    "caption": result.get("caption", ""),
                    "figure_type": result.get("figure_type_mllm", "unknown"),
                    "visual_anchor": visual_anchor,
                    "text_evidence": text_evidence,
                    "query_type": q.get("query_type", "unknown"),
                }
                f.write(json.dumps(entry, ensure_ascii=False) + '\n')
                query_count += 1

    # Summary
    success = sum(1 for r in results if r.get("parsed"))
    failed = sum(1 for r in results if not r.get("parsed") and "error" not in r)
    errors = sum(1 for r in results if "error" in r)

    # Claude Sonnet 4.5: $3/M input, $15/M output
    est_cost = total_input_tokens * 3 / 1e6 + total_output_tokens * 15 / 1e6

    print(f"\n{'='*60}")
    print(f"Results Summary")
    print(f"{'='*60}")
    print(f"  Total pairs:           {len(pairs_to_process)}")
    print(f"  Successful parses:     {success}")
    print(f"  Failed parses:         {failed}")
    print(f"  API errors:            {errors}")
    print(f"  Queries generated:     {query_count + dropped}")
    print(f"  Queries dropped (QC):  {dropped}")
    print(f"  Queries kept:          {query_count}")
    print(f"  Input tokens:          {total_input_tokens:,}")
    print(f"  Output tokens:         {total_output_tokens:,}")
    print(f"  Est. cost:             ${est_cost:.2f}")
    print(f"  Output:                {output_path}")
    print(f"  Queries:               {queries_path}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
