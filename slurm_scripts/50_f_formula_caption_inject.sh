#!/bin/bash
#SBATCH --job-name=f_formula_caption
#SBATCH --partition=cluster02
#SBATCH --qos=msc
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --gres=gpu:a6000:1
#SBATCH --time=01:30:00
#SBATCH --output=logs/f_formula_caption_%j.out
#SBATCH --error=logs/f_formula_caption_%j.err

echo "============================================"
echo "Job: F-formula caption injection variant"
echo "Test claim:C11 — does NL context lift formula R@10 above 0.56?"
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
AUG_DIR="data/05_eval/dense_retrieval/rebuilt_20260417/augmented"
OUT_DIR="data/05_eval/dense_retrieval/rebuilt_20260417"

# 1) Dense retrieval on caption-injected corpus
echo ""
echo "=== Dense retrieval: corpus_v1_enriched_formula_caption ==="
python scripts/eval_dense_retrieval.py \
    --data-dir "$AUG_DIR" \
    --queries "$AUG_DIR/queries.jsonl" \
    --corpus "$AUG_DIR/corpus_v1_enriched_formula_caption.jsonl" \
    --qrels "$AUG_DIR/qrels_v1.jsonl" \
    --model-path "$MODEL" \
    --output "$OUT_DIR/eval_report_v1_formula_caption.json" \
    --save-ranking "$OUT_DIR/ranking_v1_formula_caption.jsonl" \
    --batch-size 8 \
    --max-length 512 \
    --top-k 100 \
    --device cuda
echo "dense done: $(date)"

# 2) Graph rerank on the new ranking (explicit-only, the ceiling config)
echo ""
echo "=== Graph rerank: explicit-only on caption-injected ranking ==="
GR_OUT="$OUT_DIR/graph_explicit_only_formula_caption"
mkdir -p "$GR_OUT"
python scripts/eval_graph_topk_rerank.py \
    --ranking "$OUT_DIR/ranking_v1_formula_caption.jsonl" \
    --qrels "$DATA_DIR/qrels.jsonl" \
    --corpus "$AUG_DIR/corpus_v1_enriched_formula_caption.jsonl" \
    --chunk-graph data/01_graphs/chunk_virtual_nodes.json \
    --hub-candidates data/02_enriched/hub_candidates_enriched_v3.json \
    --graph-sources explicit \
    --output-dir "$GR_OUT" \
    --top-k 100
echo "graph rerank done: $(date)"

echo ""
echo "============================================"
echo "Summary (vs baseline R@10 0.6195 / graph 0.6913):"
echo ""
for f in "$OUT_DIR/eval_report_v1_formula_caption.json" "$GR_OUT/metrics_graph_static_plus_neighbor.json"; do
    if [ -f "$f" ]; then
        echo "  $(basename $f):"
        python3 -c "
import json
d = json.load(open('$f'))
print(f'    R@1={d[\"recall@1\"]:.4f}  R@5={d[\"recall@5\"]:.4f}  R@10={d[\"recall@10\"]:.4f}  R@100={d[\"recall@100\"]:.4f}  MRR={d[\"mrr\"]:.4f}')
"
    fi
done
echo "============================================"
echo "Done: $(date)"
