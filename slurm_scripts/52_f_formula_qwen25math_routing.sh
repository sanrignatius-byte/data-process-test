#!/bin/bash
#SBATCH --job-name=f_formula_qwen25math
#SBATCH --partition=cluster02
#SBATCH --qos=msc
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --gres=gpu:a6000:1
#SBATCH --time=02:00:00
#SBATCH --output=logs/f_formula_qwen25math_%j.out
#SBATCH --error=logs/f_formula_qwen25math_%j.err

echo "============================================"
echo "Job: F-formula Mode B — Qwen2.5-Math-7B routing + RRF"
echo "============================================"
echo "Job ID:  $SLURM_JOB_ID"
echo "Node:    $(hostname)"
echo "GPU:     $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo N/A)"
echo "Start:   $(date)"
echo "============================================"

cd /projects/myyyx1/data-process-test
source /cluster/apps/software/Miniforge3/24.11.3-1/etc/profile.d/conda.sh
conda activate /projects/myyyx1/envs/minerU

mkdir -p logs

CORPUS="data/05_eval/dense_retrieval/rebuilt_20260417/augmented/corpus_v1_enriched.jsonl"
QUERIES="data/03_queries/M4query_v1/queries.jsonl"
QRELS="data/03_queries/M4query_v1/qrels.jsonl"
BASELINE="data/05_eval/dense_retrieval/rebuilt_20260417/ranking_v1_enriched.jsonl"
SMOKE50="data/03_queries/M4query_smoke50/queries.jsonl"
OUT_DIR="data/05_eval/dense_retrieval/qwen25math_formula_routing"
MODEL="Qwen/Qwen2.5-Math-7B"

echo ""
echo "=== Encode + RRF fuse + eval ==="
python -u scripts/eval_formula_qwen25math_routing.py \
    --corpus "$CORPUS" \
    --queries "$QUERIES" \
    --qrels "$QRELS" \
    --baseline-ranking "$BASELINE" \
    --smoke50-queries "$SMOKE50" \
    --model-path "$MODEL" \
    --output-dir "$OUT_DIR" \
    --passage-max-length 384 \
    --query-max-length 128 \
    --batch-size 4 \
    --rrf-ks "10,20,60"

echo ""
echo "=== Done ==="
echo "End: $(date)"
echo ""
echo "Headline numbers (formula bucket on smoke50 should be the focus):"
python -c "import json,sys; r=json.load(open('$OUT_DIR/eval_report.json')); \
print('overall full:', r['overall']); \
print('formula bucket overall:', r['formula_bucket_overall']); \
print('formula bucket smoke50:', r['formula_bucket_smoke50']); \
print('baselines:', r['baselines_for_compare'])"
