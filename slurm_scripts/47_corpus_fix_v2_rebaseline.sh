#!/bin/bash
#SBATCH --job-name=corpus_fix_v2_rebaseline
#SBATCH --partition=cluster02
#SBATCH --qos=msc
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --gres=gpu:a6000:1
#SBATCH --time=02:00:00
#SBATCH --output=logs/corpus_fix_v2_rebaseline_%j.out
#SBATCH --error=logs/corpus_fix_v2_rebaseline_%j.err

# fix_v2 = additive enrichment (visual MODORA + graph caption/context concat).
# Sibling to 46_*; same eval protocol.

set -euo pipefail
cd /projects/myyyx1/data-process-test
source /cluster/apps/software/Miniforge3/24.11.3-1/etc/profile.d/conda.sh
conda activate /projects/myyyx1/envs/minerU
mkdir -p logs

MODEL="models/Qwen3-Embedding-4B"
FIX_DIR="data/05_eval/corpus_fix_v2"
QRELS_ORIG="data/03_queries/M4query_v1/qrels.jsonl"
HUB="data/02_enriched/hub_candidates_enriched_v3.json"
CHUNK_GRAPH="data/01_graphs/chunk_virtual_nodes.json"

echo "=== [1/3] dense retrieval on corpus_fix_v2 ==="
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

OUT_GA="$FIX_DIR/graph_explicit_only"
mkdir -p "$OUT_GA"
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
echo "  done: $(date)"

OUT_GC="$FIX_DIR/graph_explicit_plus_same_chunk"
mkdir -p "$OUT_GC"
echo "=== [3/3] graph rerank — explicit + same_chunk ==="
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
echo "  done: $(date)"
