#!/bin/bash
#SBATCH --job-name=corpus_fix_rebaseline
#SBATCH --partition=cluster02
#SBATCH --qos=msc
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --gres=gpu:a6000:1
#SBATCH --time=02:00:00
#SBATCH --output=logs/corpus_fix_rebaseline_%j.out
#SBATCH --error=logs/corpus_fix_rebaseline_%j.err

# ============================================================
# 46_corpus_fix_rebaseline.sh
#
# Phase C of CORPUS_ENRICH_FIX_PLAN_20260503.md
# Re-baseline dense retrieval + graph rerank on the corpus_fix_v1
# (load_enriched_index now ingests MODORA documents.elements format,
#  297 figure passages promoted from "[Image: …]" to enriched description).
#
# Anchors to beat (rebuilt_20260417 / v1_enriched):
#   dense        R@10 0.6195  R@100 0.8636  MRR 0.6122
#   graph rerank R@10 0.6913  R@100 0.8636  MRR ~0.690 (explicit_only_fixed)
#
# Outputs:
#   data/05_eval/corpus_fix_v1/eval_dense.json
#   data/05_eval/corpus_fix_v1/ranking.jsonl
#   data/05_eval/corpus_fix_v1/graph_explicit_only/eval_report.json
# ============================================================

set -euo pipefail

echo "============================================"
echo "Job: corpus_fix_rebaseline (Phase C)"
echo "Job ID:  ${SLURM_JOB_ID:-local}"
echo "Node:    $(hostname)"
echo "GPU:     $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo N/A)"
echo "Start:   $(date)"
echo "============================================"

cd /projects/myyyx1/data-process-test
source /cluster/apps/software/Miniforge3/24.11.3-1/etc/profile.d/conda.sh
conda activate /projects/myyyx1/envs/minerU

mkdir -p logs

MODEL="models/Qwen3-Embedding-4B"
FIX_DIR="data/05_eval/corpus_fix_v1"
QRELS_ORIG="data/03_queries/M4query_v1/qrels.jsonl"   # graph rerank uses unaugmented qrels (matches 19_)
HUB="data/02_enriched/hub_candidates_enriched_v3.json"
CHUNK_GRAPH="data/01_graphs/chunk_virtual_nodes.json"

# ── Step 1/3: dense retrieval on fixed corpus (top-k 100 for graph rerank, mirrors anchor) ──
echo ""
echo "=== [1/3] dense retrieval on corpus_fix_v1 ==="
python scripts/eval_dense_retrieval.py \
    --data-dir "$FIX_DIR" \
    --queries "$FIX_DIR/queries.jsonl" \
    --corpus  "$FIX_DIR/corpus_v1_enriched.jsonl" \
    --qrels   "$FIX_DIR/qrels_v1.jsonl" \
    --model-path "$MODEL" \
    --output  "$FIX_DIR/eval_dense.json" \
    --save-ranking "$FIX_DIR/ranking.jsonl" \
    --batch-size 8 \
    --max-length 512 \
    --top-k 100 \
    --device cuda
echo "  dense done: $(date)"

# ── Step 2/3: graph rerank — explicit only (matches the prior ceiling: graph_explicit_only_fixed) ──
OUT_GA="$FIX_DIR/graph_explicit_only"
mkdir -p "$OUT_GA"
echo ""
echo "=== [2/3] graph rerank — explicit only ==="
python scripts/eval_graph_topk_rerank.py \
    --ranking "$FIX_DIR/ranking.jsonl" \
    --qrels   "$QRELS_ORIG" \
    --corpus  "$FIX_DIR/corpus_v1_enriched.jsonl" \
    --chunk-graph "$CHUNK_GRAPH" \
    --hub-candidates "$HUB" \
    --graph-sources explicit \
    --output-dir "$OUT_GA" \
    --top-k 100
echo "  graph_explicit_only done: $(date)"

# ── Step 3/3: graph rerank — explicit + same_chunk virtual (the strongest variant from 19_) ──
OUT_GC="$FIX_DIR/graph_explicit_plus_same_chunk"
mkdir -p "$OUT_GC"
echo ""
echo "=== [3/3] graph rerank — explicit + same_chunk virtual ==="
python scripts/eval_graph_topk_rerank.py \
    --ranking "$FIX_DIR/ranking.jsonl" \
    --qrels   "$QRELS_ORIG" \
    --corpus  "$FIX_DIR/corpus_v1_enriched.jsonl" \
    --chunk-graph "$CHUNK_GRAPH" \
    --hub-candidates "$HUB" \
    --graph-sources explicit virtual \
    --edge-types same_chunk \
    --output-dir "$OUT_GC" \
    --top-k 100
echo "  graph_explicit_plus_same_chunk done: $(date)"

echo ""
echo "============================================"
echo "End:     $(date)"
echo "============================================"
