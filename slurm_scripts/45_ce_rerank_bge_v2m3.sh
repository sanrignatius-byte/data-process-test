#!/bin/bash
#SBATCH --job-name=ce_rerank_bge
#SBATCH --partition=cluster02
#SBATCH --qos=msc
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --gres=gpu:a6000:1
#SBATCH --time=01:30:00
#SBATCH --output=logs/ce_rerank_bge_%j.out
#SBATCH --error=logs/ce_rerank_bge_%j.err

# ============================================================
# 45_ce_rerank_bge_v2m3.sh
#
# Pilot triggered by R2 of CEILING_DECISION_20260503.md.
# Rerank dense top-500 with BAAI/bge-reranker-v2-m3 on M4query_v1.
#
# Strong baselines to beat:
#   - dense R@10           = 0.6195
#   - graph rerank R@10    = 0.6913 (graph_explicit_only_fixed/static_plus_neighbor)
#
# Success bar: R@10 >= 0.72 (primary), >= 0.75 (stretch).
# Theoretical ceiling (R@500 of dense) = 0.9577.
# ============================================================

set -euo pipefail

echo "============================================"
echo "Job: ce_rerank_bge"
echo "Job ID:  ${SLURM_JOB_ID:-local}"
echo "Node:    $(hostname)"
echo "Start:   $(date)"
echo "============================================"

cd /projects/myyyx1/data-process-test
source /cluster/apps/software/Miniforge3/24.11.3-1/etc/profile.d/conda.sh
conda activate /projects/myyyx1/envs/minerU

export HF_HOME=/projects/myyyx1/models/hf_cache
export TOKENIZERS_PARALLELISM=false

CE_MODEL_PATH="/projects/myyyx1/models/hf_cache/models--BAAI--bge-reranker-v2-m3/snapshots/953dc6f6f85a1b2dbfca4c34a2796e7dde08d41e"

mkdir -p logs data/05_eval/cross_encoder_rerank

AUG_DIR="data/05_eval/dense_retrieval/rebuilt_20260417/augmented"
OUT_DIR="data/05_eval/cross_encoder_rerank/bge_v2m3_top500"
mkdir -p "$OUT_DIR"

echo ""
echo "=== Cross-encoder rerank (top-500, max_length=7500) ==="
python scripts/cross_encoder_rerank.py \
    --ranking-in    data/05_eval/failure_analysis/full_ranking.jsonl \
    --corpus        "$AUG_DIR/corpus_v1_enriched.jsonl" \
    --queries       "$AUG_DIR/queries.jsonl" \
    --qrels         "$AUG_DIR/qrels_v1.jsonl" \
    --model         "$CE_MODEL_PATH" \
    --rerank-topn   500 \
    --max-length    2048 \
    --batch-size    64 \
    --device        cuda \
    --ranking-out   "$OUT_DIR/ranking_ce_bge_v2m3.jsonl" \
    --metrics-out   "$OUT_DIR/metrics.json"
echo "rerank done: $(date)"

echo ""
echo "============================================"
echo "End:     $(date)"
echo "============================================"
