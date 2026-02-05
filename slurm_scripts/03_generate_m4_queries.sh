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

module load Miniforge3

cd /projects/myyyx1/data-process-test

echo "=========================================="
echo "Starting M4 Query Generation"
echo "Start time: $(date)"
echo "Input: data/mineru_output"
echo "Output: data/queries_output"
echo "=========================================="

# 检查解析输出
echo "Parsed documents available:"
ls -d data/mineru_output/*/ 2>/dev/null | wc -l

# 先运行dry-run查看统计
echo ""
echo "Running dry-run to check entity statistics..."
conda run -n minerU python scripts/generate_m4_queries.py \
    --input data/mineru_output \
    --dry-run

echo ""
echo "=========================================="
echo "Generating M4 queries..."
echo "=========================================="

# 生成M4 Query
conda run -n minerU python scripts/generate_m4_queries.py \
    --input data/mineru_output \
    --output data/queries_output/m4_queries \
    --max-docs 20 \
    --num-queries 50 \
    --provider anthropic

echo "=========================================="
echo "Generation complete: $(date)"
echo "Output files:"
ls -la data/queries_output/
echo "=========================================="
