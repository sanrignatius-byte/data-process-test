#!/bin/bash
#SBATCH -p cluster02
#SBATCH -C gpu
#SBATCH --job-name=mineru_parse
#SBATCH --output=logs/parse_%j.out
#SBATCH --error=logs/parse_%j.err
#SBATCH --time=24:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --gpus=4

# ============================================================
# Step 2: 使用MinerU解析PDF
# ============================================================

set -euo pipefail

module load Miniforge3

REPO_ROOT=${REPO_ROOT:-/projects/myyyx1/data-process-test}
CONDA_ENV=${CONDA_ENV:-minerU}
INPUT_PDF_DIR=${INPUT_PDF_DIR:-data/raw_pdfs}
MINERU_OUTPUT_DIR=${MINERU_OUTPUT_DIR:-data/mineru_output}
WORKERS=${WORKERS:-4}
DEVICES=${DEVICES:-"cuda:0 cuda:1 cuda:2 cuda:3"}
TIMEOUT=${TIMEOUT:-900}

cd "$REPO_ROOT"
mkdir -p logs "$MINERU_OUTPUT_DIR"

echo "=========================================="
echo "Starting MinerU parsing"
echo "Start time: $(date)"
echo "Input: $INPUT_PDF_DIR"
echo "Output: $MINERU_OUTPUT_DIR"
echo "=========================================="

PDF_COUNT=$(find "$INPUT_PDF_DIR" -maxdepth 1 -name '*.pdf' | wc -l)
echo "Total: $PDF_COUNT PDFs"

if [ "$PDF_COUNT" -eq 0 ]; then
    echo "ERROR: No PDF files found in $INPUT_PDF_DIR"
    exit 1
fi

read -r -a DEVICE_ARR <<< "$DEVICES"
conda run -n "$CONDA_ENV" python scripts/parse_only.py \
    --input "$INPUT_PDF_DIR" \
    --output "$MINERU_OUTPUT_DIR" \
    --workers "$WORKERS" \
    --devices "${DEVICE_ARR[@]}" \
    --timeout "$TIMEOUT"

echo "=========================================="
echo "Parsing complete: $(date)"
echo "Parsed documents:"
find "$MINERU_OUTPUT_DIR" -mindepth 1 -maxdepth 1 -type d | wc -l
echo "=========================================="
