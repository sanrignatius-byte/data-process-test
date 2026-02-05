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

module load Miniforge3

cd /projects/myyyx1/data-process-test

echo "=========================================="
echo "Starting MinerU parsing"
echo "Start time: $(date)"
echo "Input: data/raw_pdfs"
echo "Output: data/mineru_output"
echo "=========================================="

# 检查输入文件
echo "PDF files to parse:"
ls -la data/raw_pdfs/*.pdf 2>/dev/null | head -20
PDF_COUNT=$(ls data/raw_pdfs/*.pdf 2>/dev/null | wc -l)
echo "Total: $PDF_COUNT PDFs"

if [ "$PDF_COUNT" -eq 0 ]; then
    echo "ERROR: No PDF files found in data/raw_pdfs/"
    exit 1
fi

# 运行MinerU解析
conda run -n minerU python scripts/parse_only.py \
    --input data/raw_pdfs \
    --output data/mineru_output \
    --workers 4 \
    --devices cuda:0 cuda:1 cuda:2 cuda:3 \
    --timeout 900

echo "=========================================="
echo "Parsing complete: $(date)"
echo "Parsed documents:"
ls -d data/mineru_output/*/ 2>/dev/null | wc -l
echo "=========================================="
