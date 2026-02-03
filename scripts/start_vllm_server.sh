#!/bin/bash
# Start vLLM server for Qwen3-VL multimodal query generation
# Designed for NTU EEE Cluster with A2000 GPUs

set -e

# Configuration
MODEL=${MODEL:-"Qwen/Qwen2.5-VL-7B-Instruct"}
PORT=${PORT:-8000}
GPU_MEMORY_UTILIZATION=${GPU_MEMORY_UTILIZATION:-0.85}
MAX_MODEL_LEN=${MAX_MODEL_LEN:-4096}
TENSOR_PARALLEL_SIZE=${TENSOR_PARALLEL_SIZE:-1}

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}  vLLM Server for Qwen3-VL${NC}"
echo -e "${GREEN}========================================${NC}"
echo ""
echo -e "Model: ${YELLOW}${MODEL}${NC}"
echo -e "Port: ${YELLOW}${PORT}${NC}"
echo -e "GPU Memory Utilization: ${YELLOW}${GPU_MEMORY_UTILIZATION}${NC}"
echo ""

# Check if vLLM is installed
if ! command -v vllm &> /dev/null; then
    echo -e "${RED}Error: vLLM not installed${NC}"
    echo "Install with: pip install vllm"
    exit 1
fi

# Check CUDA availability
if ! command -v nvidia-smi &> /dev/null; then
    echo -e "${YELLOW}Warning: nvidia-smi not found, GPU may not be available${NC}"
fi

# Start server
echo -e "${GREEN}Starting vLLM server...${NC}"
echo ""

# For Qwen2.5-VL, we need to enable multimodal support
python -m vllm.entrypoints.openai.api_server \
    --model "$MODEL" \
    --port "$PORT" \
    --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION" \
    --max-model-len "$MAX_MODEL_LEN" \
    --tensor-parallel-size "$TENSOR_PARALLEL_SIZE" \
    --trust-remote-code \
    --dtype auto \
    --api-key "dummy" \
    2>&1 | tee logs/vllm_server.log
