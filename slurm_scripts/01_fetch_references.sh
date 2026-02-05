#!/bin/bash
#SBATCH -p cluster02
#SBATCH --job-name=fetch_refs
#SBATCH --output=logs/fetch_refs_%j.out
#SBATCH --error=logs/fetch_refs_%j.err
#SBATCH --time=2:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G

# ============================================================
# Step 1: 获取Survey引用的论文
# Survey: Multi-Turn Interaction Capabilities of LLMs (2501.09959)
# ============================================================

module load Miniforge3

cd /projects/myyyx1/data-process-test

echo "=========================================="
echo "Fetching references from Survey: 2501.09959"
echo "Start time: $(date)"
echo "=========================================="

# 获取引用并下载PDF
conda run -n minerU python scripts/fetch_survey_references.py \
    --arxiv-id 2501.09959 \
    --output data/raw_pdfs \
    --max-papers 50 \
    --min-citations 5

echo "=========================================="
echo "Download complete: $(date)"
echo "Downloaded papers:"
ls -la data/raw_pdfs/*.pdf 2>/dev/null | wc -l
echo "=========================================="
