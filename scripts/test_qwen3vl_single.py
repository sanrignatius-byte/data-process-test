#!/usr/bin/env python3
"""
Quick smoke test: run Qwen3-VL on 2 figures to verify everything works.
"""

import json
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def main():
    # Load figure-text pairs
    pairs_file = project_root / "data" / "figure_text_pairs.json"
    with open(pairs_file) as f:
        all_pairs = json.load(f)

    # Pick 2 high-quality pairs with images
    test_pairs = []
    for doc_id, pairs in all_pairs.items():
        for pair in pairs:
            img = Path(pair["image_path"])
            if (pair.get("quality_score", 0) >= 0.8
                    and pair.get("caption")
                    and img.exists()
                    and img.stat().st_size > 5000):
                test_pairs.append(pair)
                if len(test_pairs) >= 2:
                    break
        if len(test_pairs) >= 2:
            break

    print(f"Selected {len(test_pairs)} test pairs:")
    for p in test_pairs:
        print(f"  - {p['figure_id']}: {p['caption'][:80]}")
        print(f"    Image: {p['image_path']} ({Path(p['image_path']).stat().st_size / 1024:.1f} KB)")

    # Build prompts
    from scripts.batch_figure_understanding import build_prompt

    print("\n" + "=" * 60)
    print("Loading Qwen3-VL-32B with vLLM...")
    print("=" * 60)

    from vllm import LLM, SamplingParams

    llm = LLM(
        model="Qwen/Qwen3-VL-32B-Instruct",
        tensor_parallel_size=4,
        max_model_len=16384,
        trust_remote_code=True,
        dtype="auto",
        gpu_memory_utilization=0.90,
        download_dir="/projects/myyyx1/model_cache",
        limit_mm_per_prompt={"image": 4},
    )

    sampling_params = SamplingParams(
        temperature=0.7,
        top_p=0.8,
        top_k=20,
        max_tokens=2048,
    )

    prompts = []
    for pair in test_pairs:
        text_prompt = build_prompt(pair)
        prompts.append({
            "prompt": f"<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n<|vision_start|><|image_pad|><|vision_end|>\n{text_prompt}<|im_end|>\n<|im_start|>assistant\n",
            "multi_modal_data": {
                "image": pair["image_path"],
            },
        })

    print(f"\nRunning inference on {len(prompts)} images...")
    outputs = llm.generate(prompts, sampling_params)

    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)

    for pair, output in zip(test_pairs, outputs):
        text = output.outputs[0].text
        print(f"\n{'─' * 60}")
        print(f"Figure: {pair['figure_id']}")
        print(f"Caption: {pair['caption'][:100]}")
        print(f"Response ({len(text)} chars):")
        print(text[:2000])

    print("\n✅ Smoke test passed!")


if __name__ == "__main__":
    main()
