#!/bin/bash
#SBATCH --job-name=dr_rebuilt
#SBATCH --partition=cluster02
#SBATCH --qos=msc
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --gres=gpu:a6000:1
#SBATCH --time=04:00:00
#SBATCH --output=logs/dense_retrieval_rebuilt_%j.out
#SBATCH --error=logs/dense_retrieval_rebuilt_%j.err

echo "============================================"
echo "Job: Dense Retrieval Rerun on rebuilt M4query_v1"
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

MODEL="models/Qwen3-Embedding-4B"
DATA_DIR="data/03_queries/M4query_v1"
OUT_DIR="data/05_eval/dense_retrieval/rebuilt_20260417"
AUG_DIR="$OUT_DIR/augmented"

mkdir -p "$OUT_DIR" "$AUG_DIR"

echo ""
echo "=== Step 1/2: rebuild augmented corpora on rebuilt M4query_v1 ==="
python scripts/build_graph_augmented_corpus.py \
    --m4-dir "$DATA_DIR" \
    --output-dir "$AUG_DIR" \
    --variants v1 v2 v3
echo "Augmented corpus rebuild done: $(date)"

echo ""
echo "=== Step 2/2: dense retrieval baseline + ablations ==="

echo ""
echo "=== baseline: rebuilt M4query_v1 ==="
python scripts/eval_dense_retrieval.py \
    --data-dir "$DATA_DIR" \
    --model-path "$MODEL" \
    --output "$OUT_DIR/eval_report_qwen3_4B.json" \
    --save-ranking "$OUT_DIR/ranking_qwen3_4B.jsonl" \
    --batch-size 8 \
    --max-length 512 \
    --top-k 100 \
    --device cuda
echo "baseline done: $(date)"

echo ""
echo "=== v1: enriched elements ==="
python scripts/eval_dense_retrieval.py \
    --data-dir "$AUG_DIR" \
    --queries "$AUG_DIR/queries.jsonl" \
    --corpus "$AUG_DIR/corpus_v1_enriched.jsonl" \
    --qrels "$AUG_DIR/qrels_v1.jsonl" \
    --model-path "$MODEL" \
    --output "$OUT_DIR/eval_report_v1_enriched.json" \
    --save-ranking "$OUT_DIR/ranking_v1_enriched.jsonl" \
    --batch-size 8 \
    --max-length 512 \
    --top-k 100 \
    --device cuda
echo "v1 done: $(date)"

echo ""
echo "=== v2: enriched + chunks ==="
python scripts/eval_dense_retrieval.py \
    --data-dir "$AUG_DIR" \
    --queries "$AUG_DIR/queries.jsonl" \
    --corpus "$AUG_DIR/corpus_v2_chunks.jsonl" \
    --qrels "$AUG_DIR/qrels_v2_chunks.jsonl" \
    --model-path "$MODEL" \
    --output "$OUT_DIR/eval_report_v2_chunks.json" \
    --save-ranking "$OUT_DIR/ranking_v2_chunks.jsonl" \
    --batch-size 8 \
    --max-length 512 \
    --top-k 100 \
    --device cuda
echo "v2 done: $(date)"

echo ""
echo "=== v3: enriched + chunks + summaries ==="
python scripts/eval_dense_retrieval.py \
    --data-dir "$AUG_DIR" \
    --queries "$AUG_DIR/queries.jsonl" \
    --corpus "$AUG_DIR/corpus_v3_summaries.jsonl" \
    --qrels "$AUG_DIR/qrels_v3_summaries.jsonl" \
    --model-path "$MODEL" \
    --output "$OUT_DIR/eval_report_v3_summaries.json" \
    --save-ranking "$OUT_DIR/ranking_v3_summaries.jsonl" \
    --batch-size 8 \
    --max-length 512 \
    --top-k 100 \
    --device cuda
echo "v3 done: $(date)"

echo ""
echo "============================================"
echo "Dense retrieval rerun complete: $(date)"
echo "Output dir: $OUT_DIR"
echo "============================================"
for v in qwen3_4B v1_enriched v2_chunks v3_summaries; do
    f="$OUT_DIR/eval_report_${v}.json"
    if [ -f "$f" ]; then
        echo "  $v:"
        python3 -c "
import json
d = json.load(open('$f'))
print(f'    R@10={d[\"recall@10\"]:.4f}  R@100={d[\"recall@100\"]:.4f}  NDCG@10={d[\"ndcg@10\"]:.4f}  MRR={d[\"mrr\"]:.4f}')
"
    fi
done
echo "============================================"
