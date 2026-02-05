#!/bin/bash
#SBATCH -p cluster02
#SBATCH --job-name=m4_query_gen
#SBATCH --output=logs/m4_query_%j.out
#SBATCH --error=logs/m4_query_%j.err
#SBATCH --time=4:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G

# ============================================================
# Step 3: 生成M4跨文档Query
# 需要API密钥，从环境变量读取
# ============================================================

set -euo pipefail

module load Miniforge3

REPO_ROOT=${REPO_ROOT:-/projects/myyyx1/data-process-test}
CONDA_ENV=${CONDA_ENV:-minerU}
MINERU_INPUT_DIR=${MINERU_INPUT_DIR:-data/mineru_output}
QUERY_OUTPUT_DIR=${QUERY_OUTPUT_DIR:-data/queries_output/m4_queries}
MAX_DOCS=${MAX_DOCS:-20}
NUM_QUERIES=${NUM_QUERIES:-50}
PROVIDER=${PROVIDER:-anthropic}

cd "$REPO_ROOT"
mkdir -p logs "$(dirname "$QUERY_OUTPUT_DIR")"

echo "=========================================="
echo "Starting M4 Query Generation"
echo "Start time: $(date)"
echo "Input: $MINERU_INPUT_DIR"
echo "Output: $QUERY_OUTPUT_DIR"
echo "=========================================="

echo "Parsed documents available:"
find "$MINERU_INPUT_DIR" -mindepth 1 -maxdepth 1 -type d | wc -l

echo ""
echo "Running dry-run to check entity statistics..."
conda run -n "$CONDA_ENV" python scripts/generate_m4_queries.py \
    --input "$MINERU_INPUT_DIR" \
    --dry-run

echo ""
echo "=========================================="
echo "Generating M4 queries..."
echo "=========================================="

conda run -n "$CONDA_ENV" python scripts/generate_m4_queries.py \
    --input "$MINERU_INPUT_DIR" \
    --output "$QUERY_OUTPUT_DIR" \
    --max-docs "$MAX_DOCS" \
    --num-queries "$NUM_QUERIES" \
    --provider "$PROVIDER"

echo "=========================================="
echo "Generation complete: $(date)"
echo "Output files:"
ls -la "$(dirname "$QUERY_OUTPUT_DIR")"
echo "=========================================="
