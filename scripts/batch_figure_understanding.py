#!/usr/bin/env python3
"""
Batch Figure Understanding with Qwen3-VL via vLLM.

Processes figure-text pairs: classifies figures, generates descriptions,
and produces intra-document cross-modal queries (L1).
"""

import argparse
import json
import re
import sys
from pathlib import Path
from typing import List, Dict, Optional

from PIL import Image
from vllm import LLM, SamplingParams


FIGURE_UNDERSTANDING_PROMPT = """Generate cross-modal retrieval queries for this academic figure.

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
→ Must see figure (read RLR's value) AND know the claim (from surrounding discussion) to answer.

GOOD query (comparison):
"Why does the solid blue curve overtake the dashed red one only after epoch 12, given that both use the same base architecture?"
→ Must see figure (crossover point) AND know architecture details to answer.

GOOD query (anomaly):
"What causes the sharp spike at step 5000 in the green loss curve, despite the stated constant learning rate?"
→ Must see figure (spike location) AND know training setup to answer.

Output JSON only:
```json
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
}}
```"""


def build_prompt(pair: Dict) -> str:
    """Build the prompt for a single figure-text pair."""
    refs = pair.get("referring_paragraphs", [])
    refs_text = "\n".join(f"- {r[:300]}" for r in refs[:3]) if refs else "(none)"

    return FIGURE_UNDERSTANDING_PROMPT.format(
        caption=pair.get("caption", "(no caption)"),
        context_before=pair.get("context_before", "(none)")[:500],
        context_after=pair.get("context_after", "(none)")[:500],
        references=refs_text,
    )


def process_figures(
    model_name: str,
    tp_size: int,
    pairs_file: str,
    output_file: str,
    model_cache: str,
    max_model_len: int = 32768,
    min_quality: float = 0.5,
):
    """Process all figure-text pairs with Qwen3-VL."""

    # Load pairs
    with open(pairs_file) as f:
        all_pairs = json.load(f)

    # Flatten and filter
    pairs_to_process = []
    for doc_id, pairs in all_pairs.items():
        for pair in pairs:
            if pair.get("quality_score", 0) >= min_quality:
                pair["_doc_id"] = doc_id
                pairs_to_process.append(pair)

    print(f"Loaded {len(pairs_to_process)} pairs (quality >= {min_quality})")

    # Verify images exist
    valid_pairs = []
    for pair in pairs_to_process:
        img_path = Path(pair["image_path"])
        if img_path.exists() and img_path.stat().st_size > 1000:
            valid_pairs.append(pair)
        else:
            print(f"  Skipping {pair['figure_id']}: image not found or too small")

    print(f"Valid pairs with images: {len(valid_pairs)}")

    # Initialize vLLM
    print(f"\nLoading {model_name} with TP={tp_size}...")
    llm = LLM(
        model=model_name,
        tensor_parallel_size=tp_size,
        max_model_len=max_model_len,
        trust_remote_code=True,
        dtype="auto",
        gpu_memory_utilization=0.90,
        download_dir=model_cache,
        limit_mm_per_prompt={"image": 4},
    )

    sampling_params = SamplingParams(
        temperature=0.7,
        top_p=0.8,
        top_k=20,
        max_tokens=8192,  # Thinking mode needs ~4K for <think> + ~2K for JSON output
    )

    # Build prompts with images (PIL Image loading - required by vLLM)
    print(f"\nBuilding {len(valid_pairs)} prompts...")
    prompts = []
    for pair in valid_pairs:
        text_prompt = build_prompt(pair)
        img = Image.open(pair["image_path"])

        prompts.append({
            "prompt": f"<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n<|vision_start|><|image_pad|><|vision_end|>\n{text_prompt}<|im_end|>\n<|im_start|>assistant\n",
            "multi_modal_data": {
                "image": img,
            },
        })

    # Process all at once (vLLM handles batching internally)
    print(f"\nRunning inference on {len(prompts)} prompts...")
    outputs = llm.generate(
        prompts,
        sampling_params,
    )

    # Parse results
    results = []
    success = 0
    failed = 0

    for pair, output in zip(valid_pairs, outputs):
        generated_text = output.outputs[0].text.strip()

        # Strip <think>...</think> block from Thinking mode
        clean_text = re.sub(r'<think>.*?</think>', '', generated_text, flags=re.DOTALL).strip()

        # Try to parse JSON from response
        parsed = None
        try:
            # Try ```json ... ``` block first
            json_match = re.search(r'```json\s*(\{.*?\})\s*```', clean_text, re.DOTALL)
            if json_match:
                parsed = json.loads(json_match.group(1))
            else:
                # Fallback: find outermost JSON object
                json_start = clean_text.find('{')
                json_end = clean_text.rfind('}') + 1
                if json_start >= 0 and json_end > json_start:
                    parsed = json.loads(clean_text[json_start:json_end])
            if parsed:
                success += 1
        except json.JSONDecodeError:
            failed += 1

        result = {
            "doc_id": pair["_doc_id"],
            "figure_id": pair["figure_id"],
            "figure_number": pair.get("figure_number"),
            "image_path": pair["image_path"],
            "caption": pair.get("caption", ""),
            "original_quality_score": pair.get("quality_score", 0),
            "raw_response": generated_text,
            "parsed": parsed,
        }

        if parsed and ("queries" in parsed or "cross_modal_queries" in parsed):
            result["queries"] = parsed.get("queries", parsed.get("cross_modal_queries", []))
            result["figure_type_mllm"] = parsed.get("figure_type", "unknown")
            result["visual_elements"] = parsed.get("visual_elements", [])

        results.append(result)

    # Save results
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    # Also save queries in a flat JSONL format for easy evaluation
    queries_path = output_path.with_name("l1_cross_modal_queries_v3.jsonl")
    query_count = 0
    dropped = 0
    with open(queries_path, 'w', encoding='utf-8') as f:
        for result in results:
            for q in result.get("queries", []):
                visual_anchor = q.get("visual_anchor", "")
                text_evidence = q.get("text_evidence", "")
                query_text = q.get("query", "")

                # Hard validation: drop queries that fail cross-modal checks
                query_lower = query_text.lower().strip()

                # Ban meta-language anywhere in query
                meta_words = ["the text", "the caption", "the paper",
                              "according to", "as mentioned", "as stated",
                              "as described", "the section", "the paragraph"]
                if any(mw in query_lower for mw in meta_words):
                    dropped += 1
                    continue

                # Ban shallow patterns
                ban_prefixes = [
                    "how does figure", "what does figure",
                    "what does the figure", "what is shown",
                    "what is depicted",
                ]
                if any(query_lower.startswith(bp) for bp in ban_prefixes):
                    dropped += 1
                    continue

                # Require visual anchor (min 5 chars)
                if not visual_anchor or len(visual_anchor) < 5:
                    dropped += 1
                    continue

                # Require substantial text evidence (min 50 chars per feedback)
                if not text_evidence or len(text_evidence) < 50:
                    dropped += 1
                    continue

                # Normalize image path to relative
                img_path = result["image_path"]
                repo_root = "/projects/_hdd/myyyx1/data-process-test/"
                repo_root_alt = "/projects/myyyx1/data-process-test/"
                if img_path.startswith(repo_root):
                    img_path = img_path[len(repo_root):]
                elif img_path.startswith(repo_root_alt):
                    img_path = img_path[len(repo_root_alt):]

                entry = {
                    "query_id": f"l1_{result['doc_id']}_{result['figure_id']}_{query_count}",
                    "query": query_text,
                    "answer": q.get("answer", ""),
                    "doc_id": result["doc_id"],
                    "figure_id": result["figure_id"],
                    "figure_number": result.get("figure_number"),
                    "image_path": img_path,
                    "caption": result["caption"],
                    "figure_type": result.get("figure_type_mllm", "unknown"),
                    "visual_anchor": visual_anchor,
                    "text_evidence": text_evidence,
                    "query_type": q.get("query_type", "unknown"),
                }
                f.write(json.dumps(entry, ensure_ascii=False) + '\n')
                query_count += 1

    # Print summary
    print(f"\n{'='*60}")
    print(f"Results Summary")
    print(f"{'='*60}")
    print(f"  Total pairs processed: {len(valid_pairs)}")
    print(f"  Successful parses:     {success}")
    print(f"  Failed parses:         {failed}")
    print(f"  No JSON found:         {len(valid_pairs) - success - failed}")
    print(f"  Queries generated:     {query_count + dropped}")
    print(f"  Queries dropped (QC):  {dropped}")
    print(f"  Queries kept:          {query_count}")
    print(f"  Output:                {output_path}")
    print(f"  Queries:               {queries_path}")
    print(f"{'='*60}")


def main():
    parser = argparse.ArgumentParser(description="Batch figure understanding with Qwen3-VL")
    parser.add_argument("--model", default="/projects/myyyx1/data-process-test/Qwen3-VL-30B-A3B-Thinking")
    parser.add_argument("--tp-size", type=int, default=4)
    parser.add_argument("--input", required=True, help="figure_text_pairs.json from Step 0")
    parser.add_argument("--output", required=True, help="Output JSON path")
    parser.add_argument("--model-cache", default="/projects/myyyx1/model_cache")
    parser.add_argument("--max-model-len", type=int, default=32768)
    parser.add_argument("--min-quality", type=float, default=0.5,
                        help="Minimum quality score to process")

    args = parser.parse_args()

    process_figures(
        model_name=args.model,
        tp_size=args.tp_size,
        pairs_file=args.input,
        output_file=args.output,
        model_cache=args.model_cache,
        max_model_len=args.max_model_len,
        min_quality=args.min_quality,
    )


if __name__ == "__main__":
    main()
