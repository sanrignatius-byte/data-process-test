#!/bin/bash
#SBATCH -p cluster02
#SBATCH -C gpu
#SBATCH --job-name=qwen3vl_batch
#SBATCH --output=logs/qwen3vl_batch_%j.out
#SBATCH --error=logs/qwen3vl_batch_%j.err
#SBATCH --time=2:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gpus=4

# Batch figure understanding v2: improved prompt with visual anchors + blindfold test
# Generates L1 intra-document cross-modal queries with built-in QC

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
echo "Running batch figure understanding..."

conda run -n minerU python scripts/batch_figure_understanding.py \
    --model "${MODEL_PATH}" \
    --tp-size 4 \
    --input "${REPO_ROOT}/data/figure_text_pairs.json" \
    --output "${REPO_ROOT}/data/figure_descriptions_v2.json" \
    --max-model-len 8192 \
    --min-quality 0.5

echo ""
echo "Running validation on generated queries..."
conda run -n minerU python scripts/validate_queries.py \
    "${REPO_ROOT}/data/l1_cross_modal_queries_v2.jsonl" \
    --output "${REPO_ROOT}/data/validation_report_v2.json"

echo ""
echo "Done: $(date)"
