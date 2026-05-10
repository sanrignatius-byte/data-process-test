#!/bin/bash
#SBATCH --job-name=fail_full_rank
#SBATCH --partition=cluster02
#SBATCH --qos=msc
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --gres=gpu:a6000:1
#SBATCH --time=01:00:00
#SBATCH --output=logs/fail_full_rank_%j.out
#SBATCH --error=logs/fail_full_rank_%j.err

# ============================================================
# 44_failure_full_rank.sh
#
# Phase B of CEILING_PROFILING_PLAN_20260503.md
# Re-run dense retrieval on M4query_v1 / v1_enriched corpus with
# --top-k 2809 (full corpus). Produce full ranking for all 473
# queries, then post-process to extract rank of every missed qrel
# for the 121 partial+zero query subset.
#
# Outputs:
#   data/05_eval/failure_analysis/full_ranking.jsonl         (raw)
#   data/05_eval/failure_analysis/missed_qrel_ranks.json     (sliced)
#   data/05_eval/failure_analysis/decision_tables.md         (T6 + T7)
# ============================================================

set -euo pipefail

echo "============================================"
echo "Job: failure_full_rank"
echo "Job ID:  ${SLURM_JOB_ID:-local}"
echo "Node:    $(hostname)"
echo "Start:   $(date)"
echo "============================================"

cd /projects/myyyx1/data-process-test
source /cluster/apps/software/Miniforge3/24.11.3-1/etc/profile.d/conda.sh
conda activate /projects/myyyx1/envs/minerU

mkdir -p logs data/05_eval/failure_analysis

MODEL="models/Qwen3-Embedding-4B"
AUG_DIR="data/05_eval/dense_retrieval/rebuilt_20260417/augmented"
OUT_DIR="data/05_eval/failure_analysis"

echo ""
echo "=== Step 1/2: full-corpus dense retrieval (top-k 2809) ==="
python scripts/eval_dense_retrieval.py \
    --data-dir "$AUG_DIR" \
    --queries "$AUG_DIR/queries.jsonl" \
    --corpus "$AUG_DIR/corpus_v1_enriched.jsonl" \
    --qrels "$AUG_DIR/qrels_v1.jsonl" \
    --model-path "$MODEL" \
    --output "$OUT_DIR/eval_report_full_rank.json" \
    --save-ranking "$OUT_DIR/full_ranking.jsonl" \
    --batch-size 8 \
    --max-length 512 \
    --top-k 2809 \
    --device cuda
echo "full-rank dense done: $(date)"

echo ""
echo "=== Step 2/2: post-process — missed qrel rank lookup + decision tables ==="
python scripts/analyze_missed_qrel_rank.py \
    --full-ranking "$OUT_DIR/full_ranking.jsonl" \
    --qrels "$AUG_DIR/qrels_v1.jsonl" \
    --target-qids "$OUT_DIR/target_qids.json" \
    --elements-graph data/01_graphs/multimodal_elements.json \
    --out-ranks "$OUT_DIR/missed_qrel_ranks.json" \
    --out-tables "$OUT_DIR/decision_tables.md"
echo "post-process done: $(date)"

echo ""
echo "============================================"
echo "End:     $(date)"
echo "============================================"
