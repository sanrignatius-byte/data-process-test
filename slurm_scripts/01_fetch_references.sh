#!/bin/bash
#SBATCH -p cluster02
#SBATCH --job-name=fetch_refs
#SBATCH --output=logs/fetch_refs_%j.out
#SBATCH --error=logs/fetch_refs_%j.err
#SBATCH --time=4:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=12G

# ============================================================
# Step 1: 按arXiv ID下载其真实引用文献PDF
# 支持通过环境变量覆盖默认参数
# ============================================================

set -euo pipefail

module load Miniforge3

REPO_ROOT=${REPO_ROOT:-/projects/myyyx1/data-process-test}
CONDA_ENV=${CONDA_ENV:-minerU}
ARXIV_ID=${ARXIV_ID:-2501.09959}
OUTPUT_DIR=${OUTPUT_DIR:-data/raw_pdfs}
MAX_REFERENCES=${MAX_REFERENCES:-200}
MIN_CITATIONS=${MIN_CITATIONS:-0}
API_KEY=${SEMANTIC_SCHOLAR_API_KEY:-}

cd "$REPO_ROOT"
mkdir -p logs "$OUTPUT_DIR"

echo "=========================================="
echo "Downloading references for arXiv: ${ARXIV_ID}"
echo "Start time: $(date)"
echo "Repo root: ${REPO_ROOT}"
echo "Output: ${OUTPUT_DIR}"
echo "Max refs: ${MAX_REFERENCES}, Min citations: ${MIN_CITATIONS}"
echo "=========================================="

CMD=(
    conda run -n "$CONDA_ENV" python scripts/download_references_by_arxiv.py
    --arxiv-id "$ARXIV_ID"
    --output "$OUTPUT_DIR"
    --max-references "$MAX_REFERENCES"
    --min-citations "$MIN_CITATIONS"
)

if [[ -n "$API_KEY" ]]; then
    CMD+=(--api-key "$API_KEY")
fi

"${CMD[@]}"

echo "=========================================="
echo "Download complete: $(date)"
echo "Downloaded PDFs:"
find "$OUTPUT_DIR" -maxdepth 1 -name '*.pdf' | wc -l
echo "Report: ${OUTPUT_DIR}/reference_download_report.json"
echo "=========================================="
