#!/bin/bash
#SBATCH -p cluster02
#SBATCH -C gpu
#SBATCH --job-name=qwen3vl_serve
#SBATCH --output=logs/qwen3vl_serve_%j.out
#SBATCH --error=logs/qwen3vl_serve_%j.err
#SBATCH --time=8:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --gpus=4

# ============================================================
# Step 4: 部署 Qwen3-VL-32B 用 vLLM 做多模态推理
#
# 两种模式:
#   serve  — 启动 OpenAI 兼容 API server (默认)
#   batch  — 直接批量处理 figure-text pairs
# ============================================================

set -euo pipefail

module load Miniforge3

REPO_ROOT=${REPO_ROOT:-/projects/myyyx1/data-process-test}
CONDA_ENV=${CONDA_ENV:-minerU}
MODEL=${MODEL:-/projects/myyyx1/data-process-test/Qwen3-VL-30B-A3B-Thinking}
MODEL_CACHE=${MODEL_CACHE:-/projects/myyyx1/model_cache}
TP_SIZE=${TP_SIZE:-4}
PORT=${PORT:-8000}
MAX_MODEL_LEN=${MAX_MODEL_LEN:-8192}
MODE=${MODE:-serve}

cd "$REPO_ROOT"
mkdir -p logs "$MODEL_CACHE"

export HF_HOME="$MODEL_CACHE"
export HUGGINGFACE_HUB_CACHE="$MODEL_CACHE"

echo "=========================================="
echo "Qwen3-VL-32B Deployment"
echo "Start time: $(date)"
echo "Mode: $MODE"
echo "Model: $MODEL"
echo "TP size: $TP_SIZE"
echo "Port: $PORT"
echo "Node: $(hostname)"
echo "=========================================="

# 显示 GPU 信息
conda run -n "$CONDA_ENV" python -c "
import torch
print(f'CUDA available: {torch.cuda.is_available()}')
print(f'GPU count: {torch.cuda.device_count()}')
for i in range(torch.cuda.device_count()):
    name = torch.cuda.get_device_name(i)
    mem = torch.cuda.get_device_properties(i).total_memory / 1024**3
    print(f'  GPU {i}: {name}, {mem:.1f} GB')
"

if [ "$MODE" = "serve" ]; then
    echo ""
    echo "Starting vLLM OpenAI-compatible server..."
    echo "API will be available at: http://$(hostname):${PORT}/v1"
    echo ""

    conda run -n "$CONDA_ENV" python -m vllm.entrypoints.openai.api_server \
        --model "$MODEL" \
        --tensor-parallel-size "$TP_SIZE" \
        --max-model-len "$MAX_MODEL_LEN" \
        --port "$PORT" \
        --trust-remote-code \
        --dtype auto \
        --gpu-memory-utilization 0.90 \
        --limit-mm-per-prompt image=4 \
        --download-dir "$MODEL_CACHE"

elif [ "$MODE" = "batch" ]; then
    echo ""
    echo "Running batch figure understanding..."
    echo ""

    conda run -n "$CONDA_ENV" python scripts/batch_figure_understanding.py \
        --model "$MODEL" \
        --tp-size "$TP_SIZE" \
        --input data/figure_text_pairs.json \
        --output data/figure_descriptions.json \
        --model-cache "$MODEL_CACHE"
fi

echo "=========================================="
echo "Done: $(date)"
echo "=========================================="
