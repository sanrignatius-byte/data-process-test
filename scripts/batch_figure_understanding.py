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


FIGURE_UNDERSTANDING_PROMPT = """You are creating cross-modal retrieval training data. Generate queries that are IMPOSSIBLE to answer without BOTH the figure AND the text.

**Caption**: {caption}
**Text before figure**: {context_before}
**Text after figure**: {context_after}
**Referring paragraphs**: {references}

## Step 1: Inventory visual elements
List every concrete element visible in the image:
- Plots: axis labels, units, data series names, key values (peaks/valleys/crossovers), line styles, colors
- Diagrams: node labels, arrow directions, module names, connection types
- Tables: headers, specific cell values
- Photos/examples: objects, text, UI elements, numbers shown

## Step 2: Generate 3 cross-modal queries

BLINDFOLD TEST (mandatory): Covering the figure makes the query unanswerable. Covering the text also makes it unanswerable. Both are needed.

EVERY query MUST contain a SPECIFIC visual anchor — a concrete element you can point to:
- GOOD: "the red dashed curve peaking at y=0.92", "the node labeled Encoder", "the bar reaching 85.3%"
- BAD: "the figure", "the trend", "the results shown"

BANNED (auto-reject):
- "According to the text, ..." or "As mentioned in the text, ..."
- "How does Figure N represent/illustrate/show ..."
- "What does Figure N depict/display ..."
- Any query answerable from caption alone or text alone

QUERY TYPES (use these):
1. Value+Context: Read a specific value from the figure, connect to a textual condition. "What accuracy does Method-A reach at k=5 [from figure], and what design choice in Section 3 explains this [from text]?"
2. Comparison+Explanation: Compare two visual elements, explain via text. "The blue line overtakes the red at epoch 12 [from figure]—what training change in the text causes this crossover?"
3. Anomaly+Cause: Spot something unexpected in the figure, find cause in text. "The loss spikes at step 5000 [from figure]. What preprocessing issue in Section 4.1 explains this [from text]?"
4. Visual+Definition: A visual element whose meaning requires textual context. "The shaded region between x=0.2 and x=0.8 [from figure] corresponds to what concept defined in Equation 3 [from text]?"

## Output: JSON only, no other text
```json
{{
  "figure_type": "plot|diagram|architecture|table|example|photo|other",
  "visual_elements": ["element1 with value/position", "element2", "..."],
  "cross_modal_queries": [
    {{
      "query": "...",
      "answer": "concise, factual, max 2 sentences",
      "visual_anchor": "exact visual element referenced (e.g. 'red line peaking at 0.92 around x=50')",
      "text_evidence": "quote or close paraphrase from the provided text needed to answer",
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
        max_tokens=4096,  # Thinking mode needs more tokens for <think> + JSON output
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

        if parsed and "cross_modal_queries" in parsed:
            result["queries"] = parsed["cross_modal_queries"]
            result["figure_type_mllm"] = parsed.get("figure_type", "unknown")
            result["visual_elements"] = parsed.get("visual_elements", [])

        results.append(result)

    # Save results
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    # Also save queries in a flat JSONL format for easy evaluation
    queries_path = output_path.with_name("l1_cross_modal_queries_v2.jsonl")
    query_count = 0
    dropped = 0
    with open(queries_path, 'w', encoding='utf-8') as f:
        for result in results:
            for q in result.get("queries", []):
                visual_anchor = q.get("visual_anchor", "")
                text_evidence = q.get("text_evidence", "")
                query_text = q.get("query", "")

                # Hard validation: drop queries that fail basic cross-modal checks
                ban_prefixes = [
                    "according to the text",
                    "as mentioned in the text",
                    "how does figure",
                    "what does figure",
                    "what does the figure",
                ]
                query_lower = query_text.lower().strip()
                if any(query_lower.startswith(bp) for bp in ban_prefixes):
                    dropped += 1
                    continue
                if not visual_anchor or len(visual_anchor) < 5:
                    dropped += 1
                    continue
                if not text_evidence or len(text_evidence) < 10:
                    dropped += 1
                    continue

                entry = {
                    "query_id": f"l1_{result['doc_id']}_{result['figure_id']}_{query_count}",
                    "query": query_text,
                    "answer": q.get("answer", ""),
                    "doc_id": result["doc_id"],
                    "figure_id": result["figure_id"],
                    "figure_number": result.get("figure_number"),
                    "image_path": result["image_path"],
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
