#!/bin/bash
#SBATCH -p cluster02
#SBATCH -C gpu
#SBATCH --job-name=qwen3vl_test
#SBATCH --output=logs/qwen3vl_test_%j.out
#SBATCH --error=logs/qwen3vl_test_%j.err
#SBATCH --time=0:30:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gpus=4

# Quick smoke test: Qwen3-VL-30B-A3B-Thinking (MoE, 30B total → need 4 GPUs TP)

set -euo pipefail

module load Miniforge3

REPO_ROOT=/projects/myyyx1/data-process-test
MODEL_PATH="${REPO_ROOT}/Qwen3-VL-30B-A3B-Thinking"
cd "$REPO_ROOT"
mkdir -p logs

echo "Node: $(hostname) | Start: $(date)"

conda run -n minerU python -c "
import torch
print(f'CUDA: {torch.cuda.is_available()}, GPUs: {torch.cuda.device_count()}')
for i in range(torch.cuda.device_count()):
    print(f'  {i}: {torch.cuda.get_device_name(i)} {torch.cuda.get_device_properties(i).total_memory/1024**3:.1f}GB')
"

echo ""
echo "Model: ${MODEL_PATH}"
echo "Running smoke test..."

conda run -n minerU python -c "
import json
from pathlib import Path

# Pick 1 high-quality test image
pairs_file = Path('/projects/myyyx1/data-process-test/data/figure_text_pairs.json')
with open(pairs_file) as f:
    all_pairs = json.load(f)

test_pair = None
for doc_id, pairs in all_pairs.items():
    for p in pairs:
        img = Path(p['image_path'])
        if p.get('quality_score', 0) >= 0.8 and p.get('caption') and img.exists() and img.stat().st_size > 5000:
            test_pair = p
            break
    if test_pair:
        break

print(f'Test figure: {test_pair[\"figure_id\"]}')
print(f'Caption: {test_pair[\"caption\"][:100]}')
print(f'Image: {test_pair[\"image_path\"]} ({Path(test_pair[\"image_path\"]).stat().st_size/1024:.1f} KB)')
print()

from vllm import LLM, SamplingParams

print('Loading Qwen3-VL-30B-A3B-Thinking from local path...')
llm = LLM(
    model='${MODEL_PATH}',
    tensor_parallel_size=4,
    max_model_len=8192,
    trust_remote_code=True,
    dtype='auto',
    gpu_memory_utilization=0.90,
    limit_mm_per_prompt={'image': 4},
)

caption = test_pair.get('caption', '(none)')
ctx_before = test_pair.get('context_before', '(none)')[:300]
ctx_after = test_pair.get('context_after', '(none)')[:300]

prompt_text = f'''Analyze this figure from an academic paper.

Caption: {caption}
Text before figure: {ctx_before}
Text after figure: {ctx_after}

1. What type of figure is this? (architecture/plot/table/diagram/example)
2. Describe what the figure shows in detail.
3. Generate one query that requires BOTH the figure AND the text to answer.'''

params = SamplingParams(temperature=0.7, top_p=0.8, max_tokens=1024)

print('Running inference on 1 image...')
from PIL import Image
img = Image.open(test_pair['image_path'])
outputs = llm.generate([{
    'prompt': '<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n<|vision_start|><|image_pad|><|vision_end|>\n' + prompt_text + '<|im_end|>\n<|im_start|>assistant\n',
    'multi_modal_data': {'image': img},
}], sampling_params=params)

print()
print('=' * 60)
print('RESULT:')
print('=' * 60)
print(outputs[0].outputs[0].text)
print()
print('Smoke test complete!')
"

echo ""
echo "Done: $(date)"
