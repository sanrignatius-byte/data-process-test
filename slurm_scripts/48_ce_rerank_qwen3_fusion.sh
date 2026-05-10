#!/bin/bash
#SBATCH --job-name=qwen3_ce_fusion
#SBATCH --partition=cluster02
#SBATCH --qos=msc
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --gres=gpu:a6000:1
#SBATCH --time=08:00:00
#SBATCH --output=logs/qwen3_ce_fusion_%j.out
#SBATCH --error=logs/qwen3_ce_fusion_%j.err

# ============================================================
# 48_ce_rerank_qwen3_fusion.sh
#
# Anchor-corpus reranker family swap:
#   Qwen/Qwen3-Reranker-4B on dense top-500 via vLLM, then
#   posthoc dense/graph/CE RRF fusion + modality-bias guard.
#
# Strong baselines to beat:
#   - dense R@10        = 0.6195
#   - graph R@10        = 0.6913
# Primary success bar:
#   - any CE/fusion R@10 >= 0.7200
# Partial success:
#   - any CE/fusion R@10 > 0.6913 graph ceiling
# ============================================================

set -euo pipefail

echo "============================================"
echo "Job: qwen3_ce_fusion"
echo "Job ID:  ${SLURM_JOB_ID:-local}"
echo "Node:    $(hostname)"
echo "Start:   $(date)"
echo "============================================"

cd /projects/myyyx1/data-process-test
source /cluster/apps/software/Miniforge3/24.11.3-1/etc/profile.d/conda.sh
conda activate /projects/myyyx1/envs/minerU

export HF_HOME=/projects/myyyx1/models/hf_cache
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

mkdir -p logs data/05_eval/cross_encoder_rerank

CE_MODEL="${CE_MODEL:-Qwen/Qwen3-Reranker-4B}"
PROMPT="Given a scientific paper evidence-retrieval query, determine whether the passage is relevant evidence from an academic paper. Figures, tables, formulas, captions, and surrounding paper context are all valid evidence; do not prefer fluent natural-language paragraphs solely because they look more textual."

AUG_DIR="data/05_eval/dense_retrieval/rebuilt_20260417/augmented"
DENSE_RANKING="data/05_eval/failure_analysis/full_ranking.jsonl"
GRAPH_RANKING="data/05_eval/dense_retrieval/rebuilt_20260417/graph_explicit_only_fixed/ranking_graph_static_plus_neighbor.jsonl"
OUT_DIR="data/05_eval/cross_encoder_rerank/qwen3_reranker_4b_anchor_top500"
DRY_DIR="$OUT_DIR/dryrun10"
mkdir -p "$OUT_DIR" "$DRY_DIR"

echo ""
echo "=== Environment ==="
which python
python - <<'PY'
import torch
print("torch", torch.__version__, "cuda", torch.cuda.is_available())
if torch.cuda.is_available():
    print("gpu", torch.cuda.get_device_name(0))
PY

echo ""
echo "=== Stage 0: dry-run 10 queries (vLLM model load + score path) ==="
date
python scripts/qwen3_vllm_rerank.py \
    --ranking-in    "$DENSE_RANKING" \
    --corpus        "$AUG_DIR/corpus_v1_enriched.jsonl" \
    --queries       "$AUG_DIR/queries.jsonl" \
    --qrels         "$AUG_DIR/qrels_v1.jsonl" \
    --model         "$CE_MODEL" \
    --instruction   "$PROMPT" \
    --rerank-topn   500 \
    --max-model-len 4096 \
    --gpu-memory-utilization 0.84 \
    --batch-size    64 \
    --limit-queries 10 \
    --ranking-out   "$DRY_DIR/ranking_ce_qwen3_dryrun10.jsonl" \
    --metrics-out   "$DRY_DIR/metrics_dryrun10.json" \
    --scores-out    "$DRY_DIR/scores_dryrun10.jsonl"
echo "dry-run done: $(date)"

echo ""
echo "=== Stage 1: full Qwen3-Reranker-4B rerank (top-500) ==="
date
python scripts/qwen3_vllm_rerank.py \
    --ranking-in    "$DENSE_RANKING" \
    --corpus        "$AUG_DIR/corpus_v1_enriched.jsonl" \
    --queries       "$AUG_DIR/queries.jsonl" \
    --qrels         "$AUG_DIR/qrels_v1.jsonl" \
    --model         "$CE_MODEL" \
    --instruction   "$PROMPT" \
    --rerank-topn   500 \
    --max-model-len 4096 \
    --gpu-memory-utilization 0.84 \
    --batch-size    64 \
    --ranking-out   "$OUT_DIR/ranking_ce_qwen3_reranker_4b.jsonl" \
    --metrics-out   "$OUT_DIR/metrics_ce_qwen3_reranker_4b.json" \
    --scores-out    "$OUT_DIR/scores_ce_qwen3_reranker_4b.jsonl"
echo "full rerank done: $(date)"

echo ""
echo "=== Stage 2: posthoc RRF fusion + modality guard ==="
python scripts/posthoc_rerank_fusion_report.py \
    --dense-ranking "$DENSE_RANKING" \
    --graph-ranking "$GRAPH_RANKING" \
    --ce-ranking    "$OUT_DIR/ranking_ce_qwen3_reranker_4b.jsonl" \
    --corpus        "$AUG_DIR/corpus_v1_enriched.jsonl" \
    --qrels         "$AUG_DIR/qrels_v1.jsonl" \
    --rrf-ks        10,20,60 \
    --out-json      "$OUT_DIR/fusion_report.json" \
    --out-md        "$OUT_DIR/fusion_report.md"

echo ""
echo "=== Fusion report preview ==="
sed -n '1,120p' "$OUT_DIR/fusion_report.md"

echo ""
echo "============================================"
echo "qwen3_ce_fusion COMPLETE: $(date)"
echo "Output: $OUT_DIR"
echo "============================================"
