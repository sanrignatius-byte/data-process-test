#!/bin/bash
#SBATCH --job-name=f_formula_math_norm
#SBATCH --partition=cluster02
#SBATCH --qos=msc
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:a6000:1
#SBATCH --time=01:00:00
#SBATCH --output=logs/f_formula_math_norm_%j.out
#SBATCH --error=logs/f_formula_math_norm_%j.err

echo "============================================"
echo "Job: F-formula Phase 2 — math-normalised LaTeX corpus"
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

BASE_DIR="data/05_eval/dense_retrieval/rebuilt_20260417"
CORPUS="data/05_eval/dense_retrieval/rebuilt_20260417/augmented/corpus_v1_math_norm.jsonl"
QRELS="data/03_queries/M4query_v1/qrels.jsonl"
QUERIES="data/03_queries/M4query_v1/queries.jsonl"
MODEL="models/Qwen3-Embedding-4B"
CHUNK_GRAPH="data/01_graphs/chunk_virtual_nodes.json"
ELEMENTS_GRAPH="data/03_queries/M4query_v1/graphs/multimodal_elements.json"
HUB="data/02_enriched/hub_candidates_enriched_v3.json"

# ── Step 1: Dense retrieval with math-normalised corpus ──
OUT_DENSE="$BASE_DIR/graph_math_norm"
mkdir -p "$OUT_DENSE"
echo ""
echo "=== [1/3] Dense retrieval (Qwen3-Embedding-4B, math-normalised corpus) ==="
python scripts/eval_dense_retrieval.py \
    --data-dir /dev/null \
    --queries "$QUERIES" \
    --corpus "$CORPUS" \
    --qrels "$QRELS" \
    --model-path "$MODEL" \
    --output "$OUT_DENSE/eval_report.json" \
    --save-ranking "$OUT_DENSE/ranking_baseline.jsonl" \
    --top-k 100
echo "  done: $(date)"

# ── Step 2: Graph rerank (explicit only — highest R@10 config) ──
echo ""
echo "=== [2/3] Graph rerank (explicit only) ==="
python scripts/eval_graph_topk_rerank.py \
    --ranking "$OUT_DENSE/ranking_baseline.jsonl" \
    --qrels "$QRELS" \
    --corpus "$CORPUS" \
    --chunk-graph "$CHUNK_GRAPH" \
    --elements-graph "$ELEMENTS_GRAPH" \
    --hub-candidates "$HUB" \
    --graph-sources explicit \
    --output-dir "$OUT_DENSE" \
    --top-k 100
echo "  done: $(date)"

# ── Step 3: Smoke50 slice ──
echo ""
echo "=== [3/3] Smoke50 per-modality slice ==="
python scripts/eval_smoke50_slice.py
echo "  done: $(date)"

echo ""
echo "============================================"
echo "Summary — Math-normalised LaTeX corpus:"
echo ""

echo "Dense baseline:"
python3 -c "
import json
d = json.load(open('$OUT_DENSE/eval_report.json'))
print(f'  R@1={d[\"recall@1\"]:.4f}  R@5={d[\"recall@5\"]:.4f}  R@10={d[\"recall@10\"]:.4f}  R@100={d[\"recall@100\"]:.4f}  MRR={d[\"mrr\"]:.4f}')
"

for method in graph_static_prior graph_neighbor_prop graph_static_plus_neighbor; do
    f="$OUT_DENSE/metrics_${method}.json"
    if [ -f "$f" ]; then
        python3 -c "
import json
d = json.load(open('$f'))
print(f'  $method  R@1={d[\"recall@1\"]:.4f}  R@5={d[\"recall@5\"]:.4f}  R@10={d[\"recall@10\"]:.4f}  MRR={d[\"mrr\"]:.4f}')
"
    fi
done

echo ""
echo "Smoke50 per-modality (from data/05_eval/smoke50/per_system_per_modality.md):"
grep -A2 "math_norm" data/05_eval/smoke50/per_system_per_modality.md 2>/dev/null || echo "  (run eval_smoke50_slice.py after job completes)"

echo ""
echo "============================================"
echo "Complete: $(date)"
echo "============================================"
