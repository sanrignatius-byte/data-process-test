#!/bin/bash
#SBATCH --job-name=dl_refs_b2
#SBATCH --output=logs/download_refs_batch2_%j.out
#SBATCH --error=logs/download_refs_batch2_%j.err
#SBATCH --partition=cluster02
#SBATCH --cpus-per-task=2
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1

# ============================================================
# Batch download referenced papers (PDF + LaTeX) from arXiv
# Input: arXiv IDs extracted from 25 seed survey papers' LaTeX
# Output: data/00_raw/pdfs_batch2/ + data/00_raw/latex_sources_batch2/
# ============================================================

echo "=========================================="
echo "Reference download started: $(date)"
echo "Job ID: $SLURM_JOBID"
echo "Node: $HOSTNAME"
echo "=========================================="

cd /projects/myyyx1/data-process-test || exit 1

# Activate conda env (base has requests)
module load Miniforge3 2>/dev/null || true
source activate base 2>/dev/null || true

# Create log dir
mkdir -p logs

# Run the batch download script
# -u: unbuffered Python output for real-time SLURM logging
python -u scripts/batch_download_references.py \
    --save-ids \
    2>&1 | tee -a "logs/download_refs_batch2_${SLURM_JOBID}.log"

echo "=========================================="
echo "Reference download finished: $(date)"
echo "=========================================="
