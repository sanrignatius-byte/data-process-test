#!/bin/bash
#SBATCH --job-name=graph_rerank_explicit
#SBATCH --partition=cluster02
#SBATCH --qos=msc
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:a6000:1
#SBATCH --time=02:00:00
#SBATCH --output=logs/graph_rerank_explicit_%j.out
#SBATCH --error=logs/graph_rerank_explicit_%j.err

echo "============================================"
echo "Job: Graph Rerank - explicit + explicit+virtual edge ablation"
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
RANKING="$BASE_DIR/ranking_v1_enriched.jsonl"
QRELS="data/03_queries/M4query_v1/qrels.jsonl"
CORPUS="$BASE_DIR/augmented/corpus_v1_enriched.jsonl"
HUB="data/02_enriched/hub_candidates_enriched_v3.json"
CHUNK_GRAPH="data/01_graphs/chunk_virtual_nodes.json"

# ── Experiment A: explicit edges only (hub bridge pairs) ──
OUT_A="$BASE_DIR/graph_explicit_only_fixed"
mkdir -p "$OUT_A"
echo ""
echo "=== [A] explicit only ==="
python scripts/eval_graph_topk_rerank.py \
    --ranking "$RANKING" \
    --qrels "$QRELS" \
    --corpus "$CORPUS" \
    --chunk-graph "$CHUNK_GRAPH" \
    --hub-candidates "$HUB" \
    --graph-sources explicit \
    --output-dir "$OUT_A" \
    --top-k 100
echo "  done: $(date)"

# ── Experiment B: explicit + all virtual edges ──
OUT_B="$BASE_DIR/graph_explicit_plus_all_virtual_fixed"
mkdir -p "$OUT_B"
echo ""
echo "=== [B] explicit + all virtual ==="
python scripts/eval_graph_topk_rerank.py \
    --ranking "$RANKING" \
    --qrels "$QRELS" \
    --corpus "$CORPUS" \
    --chunk-graph "$CHUNK_GRAPH" \
    --hub-candidates "$HUB" \
    --graph-sources explicit virtual \
    --edge-types same_chunk chunk_sequence same_section_nonseq \
    --output-dir "$OUT_B" \
    --top-k 100
echo "  done: $(date)"

# ── Experiment C: explicit + same_chunk only ──
OUT_C="$BASE_DIR/graph_explicit_plus_same_chunk_fixed"
mkdir -p "$OUT_C"
echo ""
echo "=== [C] explicit + same_chunk virtual only ==="
python scripts/eval_graph_topk_rerank.py \
    --ranking "$RANKING" \
    --qrels "$QRELS" \
    --corpus "$CORPUS" \
    --chunk-graph "$CHUNK_GRAPH" \
    --hub-candidates "$HUB" \
    --graph-sources explicit virtual \
    --edge-types same_chunk \
    --output-dir "$OUT_C" \
    --top-k 100
echo "  done: $(date)"

# ── Experiment D: explicit + chunk_sequence only ──
OUT_D="$BASE_DIR/graph_explicit_plus_chunk_sequence_fixed"
mkdir -p "$OUT_D"
echo ""
echo "=== [D] explicit + chunk_sequence virtual only ==="
python scripts/eval_graph_topk_rerank.py \
    --ranking "$RANKING" \
    --qrels "$QRELS" \
    --corpus "$CORPUS" \
    --chunk-graph "$CHUNK_GRAPH" \
    --hub-candidates "$HUB" \
    --graph-sources explicit virtual \
    --edge-types chunk_sequence \
    --output-dir "$OUT_D" \
    --top-k 100
echo "  done: $(date)"

# ── Experiment E: explicit + local virtual (same_chunk + chunk_sequence) ──
OUT_E="$BASE_DIR/graph_explicit_plus_local_virtual_fixed"
mkdir -p "$OUT_E"
echo ""
echo "=== [E] explicit + local virtual (same_chunk + chunk_sequence) ==="
python scripts/eval_graph_topk_rerank.py \
    --ranking "$RANKING" \
    --qrels "$QRELS" \
    --corpus "$CORPUS" \
    --chunk-graph "$CHUNK_GRAPH" \
    --hub-candidates "$HUB" \
    --graph-sources explicit virtual \
    --edge-types same_chunk chunk_sequence \
    --output-dir "$OUT_E" \
    --top-k 100
echo "  done: $(date)"

echo ""
echo "============================================"
echo "All graph rerank experiments complete: $(date)"
echo ""
echo "Summary (baseline R@1=0.2336, R@5=0.5275, R@10=0.6195, MRR=0.6121):"
echo ""

for label in A B C D E; do
    case $label in
        A) d="$OUT_A" ;;
        B) d="$OUT_B" ;;
        C) d="$OUT_C" ;;
        D) d="$OUT_D" ;;
        E) d="$OUT_E" ;;
    esac
    echo "  [$label] $(basename $d):"
    for method in graph_static_prior graph_neighbor_prop graph_static_plus_neighbor; do
        f="$d/metrics_${method}.json"
        if [ -f "$f" ]; then
            python3 -c "
import json
d = json.load(open('$f'))
print(f'    {\"$method\":<35} R@1={d[\"recall@1\"]:.4f}  R@5={d[\"recall@5\"]:.4f}  R@10={d[\"recall@10\"]:.4f}  MRR={d[\"mrr\"]:.4f}')
"
        fi
    done
done

echo "============================================"
