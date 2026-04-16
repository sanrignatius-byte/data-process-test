#!/bin/bash
#SBATCH --job-name=dense_retr
#SBATCH --partition=cluster02
#SBATCH --qos=msc
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --gres=gpu:a6000:1
#SBATCH --time=02:00:00
#SBATCH --output=logs/dense_retrieval_%j.out
#SBATCH --error=logs/dense_retrieval_%j.err

echo "============================================"
echo "Job: Dense Retrieval Baseline (Qwen3-Embedding-4B)"
echo "============================================"
echo "Job ID:  $SLURM_JOB_ID"
echo "Node:    $(hostname)"
echo "GPU:     $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo N/A)"
echo "Start:   $(date)"
echo "============================================"

cd /projects/myyyx1/data-process-test
source /cluster/apps/software/Miniforge3/24.11.3-1/etc/profile.d/conda.sh
conda activate /projects/myyyx1/envs/minerU

mkdir -p logs data/05_eval/dense_retrieval

DATA_DIR="data/03_queries/M4query_v1"
OUTPUT_DIR="data/05_eval/dense_retrieval"

echo ""
echo "Qwen3-Embedding-4B baseline on M4query_v1..."
python scripts/eval_dense_retrieval.py \
    --data-dir "$DATA_DIR" \
    --model-path "models/Qwen3-Embedding-4B" \
    --output "$OUTPUT_DIR/eval_report_qwen3_4B.json" \
    --save-ranking "$OUTPUT_DIR/ranking_qwen3_4B.jsonl" \
    --batch-size 8 \
    --max-length 512 \
    --top-k 100 \
    --device cuda

echo "Done: $(date)"
echo "============================================"
echo "Job Complete: $(date)"
echo "Results in: $OUTPUT_DIR/"
echo "============================================"
