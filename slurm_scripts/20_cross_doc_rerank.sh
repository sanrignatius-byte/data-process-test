#!/bin/bash
#SBATCH --job-name=cross_doc_rerank
#SBATCH --partition=cluster02
#SBATCH --qos=msc
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --gres=gpu:a6000:1
#SBATCH --time=03:00:00
#SBATCH --output=logs/cross_doc_rerank_%j.out
#SBATCH --error=logs/cross_doc_rerank_%j.err

echo "============================================"
echo "Job: Cross-Doc Sim Edges + Rerank Ablation"
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
SUMMARY_GRAPH="data/01_graphs/llm_summary_virtual_nodes.json"
CROSS_DOC_EDGES="data/01_graphs/cross_doc_sim_edges.json"
BASE_DIR="data/05_eval/dense_retrieval/rebuilt_20260417"
RANKING="$BASE_DIR/ranking_v1_enriched.jsonl"
QRELS="data/03_queries/M4query_v1/qrels.jsonl"
CORPUS="$BASE_DIR/augmented/corpus_v1_enriched.jsonl"
HUB="data/02_enriched/hub_candidates_enriched_v3.json"
CHUNK_GRAPH="data/01_graphs/chunk_virtual_nodes.json"

# ── Step 1: Build cross-doc section similarity edges ──────────────────────────
echo ""
echo "=== Step 1: Build cross-doc section similarity edges (threshold=0.70, cite_boost=0.10) ==="
python scripts/build_cross_doc_sim_edges.py \
    --summary-graph "$SUMMARY_GRAPH" \
    --output "$CROSS_DOC_EDGES" \
    --model-path "$MODEL" \
    --level section \
    --threshold 0.70 \
    --citation-boost 0.10 \
    --top-k 5 \
    --batch-size 32 \
    --max-length 256 \
    --device cuda
echo "  build done: $(date)"

# t=0.80 with citation boost
echo ""
echo "=== Step 1b: threshold=0.80 + citation_boost=0.10 ==="
python scripts/build_cross_doc_sim_edges.py \
    --summary-graph "$SUMMARY_GRAPH" \
    --output "data/01_graphs/cross_doc_sim_edges_t80.json" \
    --model-path "$MODEL" \
    --level section \
    --threshold 0.80 \
    --citation-boost 0.10 \
    --top-k 5 \
    --batch-size 32 \
    --max-length 256 \
    --device cuda
echo "  build t80 done: $(date)"

# t=0.70 WITHOUT citation boost (ablation: pure sim only)
echo ""
echo "=== Step 1c: threshold=0.70, NO citation boost (ablation) ==="
python scripts/build_cross_doc_sim_edges.py \
    --summary-graph "$SUMMARY_GRAPH" \
    --output "data/01_graphs/cross_doc_sim_edges_nocite.json" \
    --model-path "$MODEL" \
    --level section \
    --threshold 0.70 \
    --citation-boost 0.0 \
    --top-k 5 \
    --batch-size 32 \
    --max-length 256 \
    --device cuda
echo "  build nocite done: $(date)"

# ── Step 2: Rerank experiments ─────────────────────────────────────────────────
echo ""
echo "=== Step 2: Rerank ablation with cross-doc edges ==="

# [A] explicit only (best from job 61089, reference)
OUT_REF="$BASE_DIR/graph_explicit_only_fixed"
echo "  [REF] explicit-only already done at $OUT_REF"

# [B] explicit + cross_doc (threshold=0.70)
OUT_B="$BASE_DIR/graph_explicit_plus_crossdoc_t70"
mkdir -p "$OUT_B"
echo ""
echo "  [B] explicit + cross_doc t=0.70"
python scripts/eval_graph_topk_rerank.py \
    --ranking "$RANKING" \
    --qrels "$QRELS" \
    --corpus "$CORPUS" \
    --chunk-graph "$CHUNK_GRAPH" \
    --hub-candidates "$HUB" \
    --cross-doc-edges "$CROSS_DOC_EDGES" \
    --graph-sources explicit cross_doc \
    --output-dir "$OUT_B" \
    --top-k 100
echo "  done: $(date)"

# [C] explicit + cross_doc (threshold=0.80)
OUT_C="$BASE_DIR/graph_explicit_plus_crossdoc_t80"
mkdir -p "$OUT_C"
echo ""
echo "  [C] explicit + cross_doc t=0.80"
python scripts/eval_graph_topk_rerank.py \
    --ranking "$RANKING" \
    --qrels "$QRELS" \
    --corpus "$CORPUS" \
    --chunk-graph "$CHUNK_GRAPH" \
    --hub-candidates "$HUB" \
    --cross-doc-edges "data/01_graphs/cross_doc_sim_edges_t80.json" \
    --graph-sources explicit cross_doc \
    --output-dir "$OUT_C" \
    --top-k 100
echo "  done: $(date)"

# [D] cross_doc only (no explicit)
OUT_D="$BASE_DIR/graph_crossdoc_only_t70"
mkdir -p "$OUT_D"
echo ""
echo "  [D] cross_doc only t=0.70"
python scripts/eval_graph_topk_rerank.py \
    --ranking "$RANKING" \
    --qrels "$QRELS" \
    --corpus "$CORPUS" \
    --chunk-graph "$CHUNK_GRAPH" \
    --cross-doc-edges "$CROSS_DOC_EDGES" \
    --graph-sources cross_doc \
    --output-dir "$OUT_D" \
    --top-k 100
echo "  done: $(date)"

# [E] explicit + virtual (same_chunk) + cross_doc
OUT_E="$BASE_DIR/graph_explicit_virtual_crossdoc"
mkdir -p "$OUT_E"
echo ""
echo "  [E] explicit + virtual(same_chunk) + cross_doc t=0.70"
python scripts/eval_graph_topk_rerank.py \
    --ranking "$RANKING" \
    --qrels "$QRELS" \
    --corpus "$CORPUS" \
    --chunk-graph "$CHUNK_GRAPH" \
    --hub-candidates "$HUB" \
    --cross-doc-edges "$CROSS_DOC_EDGES" \
    --graph-sources explicit virtual cross_doc \
    --edge-types same_chunk \
    --output-dir "$OUT_E" \
    --top-k 100
echo "  done: $(date)"

# [F] explicit + cross_doc t=0.70 WITHOUT citation boost (ablation)
OUT_F="$BASE_DIR/graph_explicit_plus_crossdoc_nocite"
mkdir -p "$OUT_F"
echo ""
echo "  [F] explicit + cross_doc t=0.70 (no citation boost, ablation)"
python scripts/eval_graph_topk_rerank.py \
    --ranking "$RANKING" \
    --qrels "$QRELS" \
    --corpus "$CORPUS" \
    --chunk-graph "$CHUNK_GRAPH" \
    --hub-candidates "$HUB" \
    --cross-doc-edges "data/01_graphs/cross_doc_sim_edges_nocite.json" \
    --graph-sources explicit cross_doc \
    --output-dir "$OUT_F" \
    --top-k 100
echo "  done: $(date)"

# ── Summary ────────────────────────────────────────────────────────────────────
echo ""
echo "============================================"
echo "All done: $(date)"
echo ""
echo "Baseline: R@1=0.2336  R@5=0.5275  R@10=0.6195  MRR=0.6121"
echo "Explicit-only static_prior (best ref): R@1=0.2421 R@5=0.5856 R@10=0.6448 MRR=0.6399"
echo ""
echo "Cross-doc results:"

for label in B C D E F; do
    case $label in
        B) d="$OUT_B" ;;
        C) d="$OUT_C" ;;
        D) d="$OUT_D" ;;
        E) d="$OUT_E" ;;
        F) d="$OUT_F" ;;
    esac
    echo "  [$label] $(basename $d):"
    for method in graph_static_prior graph_static_plus_neighbor; do
        f="$d/metrics_${method}.json"
        if [ -f "$f" ]; then
            python3 -c "
import json
d = json.load(open('$f'))
print(f'    $method  R@1={d[\"recall@1\"]:.4f}  R@5={d[\"recall@5\"]:.4f}  R@10={d[\"recall@10\"]:.4f}  MRR={d[\"mrr\"]:.4f}')
"
        fi
    done
done
echo "============================================"
