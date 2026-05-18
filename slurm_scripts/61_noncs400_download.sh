#!/bin/bash
#SBATCH --job-name=noncs400_dl
#SBATCH --partition=cluster02
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=24G
#SBATCH --time=1-12:00:00
#SBATCH --output=logs/noncs400_download_%j.out
#SBATCH --error=logs/noncs400_download_%j.err

set -euo pipefail

echo "=========================================="
echo "noncs400 arXiv download started: $(date)"
echo "Job ID: ${SLURM_JOBID:-manual}"
echo "Node:   $(hostname)"
echo "=========================================="

PROJECT_DIR="/projects/myyyx1/data-process-test"
cd "$PROJECT_DIR"
mkdir -p logs

module load Miniforge3 2>/dev/null || true
source /cluster/apps/software/Miniforge3/24.11.3-1/etc/profile.d/conda.sh 2>/dev/null || true
conda activate base 2>/dev/null || true

BASE="data/00_raw"
SEED_SRC="${BASE}/noncs400_seed_latex_sources"
PDF_DIR="${BASE}/noncs400_pdfs"
LATEX_DIR="${BASE}/noncs400_latex_sources"
MINERU_OUT="${BASE}/noncs400_mineru_output"

python -u scripts/collect_noncs_arxiv_review_refs.py \
  --target 400 \
  --refs-per-seed 20 \
  --max-seeds 35 \
  --candidate-limit 900 \
  --search-results-per-topic 40 \
  --api-delay 4.0 \
  --api-retries 6 \
  --download-delay 3.0 \
  --progress-interval 20 \
  --seed-source-dir "$SEED_SRC" \
  --pdf-dir "$PDF_DIR" \
  --latex-dir "$LATEX_DIR" \
  --mineru-output-dir "$MINERU_OUT" \
  --seed-manifest "${BASE}/noncs400_seed_manifest.json" \
  --ids-output "${BASE}/noncs400_arxiv_ids.txt" \
  --success-ids-output "${BASE}/noncs400_success_ids.txt" \
  --manifest "${BASE}/noncs400_manifest.json" \
  2>&1 | tee -a "logs/noncs400_download_${SLURM_JOBID:-manual}.log"

echo "=========================================="
echo "noncs400 arXiv download finished: $(date)"
echo "=========================================="
