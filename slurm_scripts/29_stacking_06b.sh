#!/bin/bash
#SBATCH --job-name=stack_06b
#SBATCH --output=logs/stack_06b_%j.out
#SBATCH --error=logs/stack_06b_%j.err
#SBATCH --partition=cluster02
#SBATCH --qos=msc
#SBATCH --gres=gpu:a6000:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=01:00:00

set -e
cd /projects/myyyx1/data-process-test
source /projects/myyyx1/envs/minerU/bin/activate 2>/dev/null || true
export PYTHONPATH=/projects/myyyx1/data-process-test:$PYTHONPATH

AUG_V2="data/05_eval/dense_retrieval/augmented_v2"
RANK_V1_06B="data/05_eval/dense_retrieval/rebuilt_20260417/graph_06b_v1_explicit_v1chunk/ranking_baseline.jsonl"
HUB_CANDS="data/02_enriched/hub_candidates_enriched_v3.json"
CHUNK_V1="data/01_graphs/chunk_virtual_nodes.json"
CHUNK_V2="data/01_graphs/chunk_virtual_nodes_v2.json"
ELEMENTS="data/01_graphs/multimodal_elements.json"
CROSS_DOC="data/01_graphs/cross_doc_sim_edges.json"
OUT_BASE="data/05_eval/dense_retrieval/stacking_06b"

mkdir -p "$OUT_BASE"

# Sanity: 0.6B baseline ranking exists
if [ ! -f "$RANK_V1_06B" ]; then
    echo "[fatal] missing 0.6B v1_enriched ranking: $RANK_V1_06B" >&2
    exit 1
fi

run_rerank() {
    local name="$1"
    shift
    local outdir="$OUT_BASE/$name"
    mkdir -p "$outdir"
    echo ""
    echo "=== $name ==="
    python scripts/eval_graph_topk_rerank.py \
        --ranking "$RANK_V1_06B" \
        --qrels "$AUG_V2/qrels_v1.jsonl" \
        --corpus "$AUG_V2/corpus_v1_enriched.jsonl" \
        --hub-candidates "$HUB_CANDS" \
        --chunk-graph "$CHUNK_V2" \
        --elements-graph "$ELEMENTS" \
        --cross-doc-edges "$CROSS_DOC" \
        --output-dir "$outdir" \
        --top-k 100 \
        "$@"
}

# 1. explicit only (reference, already have but re-run for consistency)
run_rerank "explicit_only" --graph-sources explicit

# 2. explicit + virtual (all edge types)
run_rerank "explicit_plus_virtual_all" --graph-sources explicit virtual \
    --edge-types same_chunk chunk_sequence same_section_nonseq

# 3. explicit + virtual (same_chunk only)
run_rerank "explicit_plus_same_chunk" --graph-sources explicit virtual \
    --edge-types same_chunk

# 4. explicit + virtual (chunk_sequence only)
run_rerank "explicit_plus_chunk_seq" --graph-sources explicit virtual \
    --edge-types chunk_sequence

# 5. explicit + cross_doc
run_rerank "explicit_plus_crossdoc" --graph-sources explicit cross_doc

# 6. explicit + virtual (same_chunk) + cross_doc
run_rerank "explicit_plus_same_chunk_plus_crossdoc" \
    --graph-sources explicit virtual cross_doc \
    --edge-types same_chunk

# 7. virtual only (same_chunk) - see if chunk edges alone help on 0.6B
run_rerank "virtual_same_chunk_only" --graph-sources virtual \
    --edge-types same_chunk

# 8. cross_doc only
run_rerank "crossdoc_only" --graph-sources cross_doc

echo ""
echo "=== ALL DONE: $(date) ==="
echo "Summary:"
for d in "$OUT_BASE"/*/; do
    cfg=$(basename "$d")
    fp="$d/metrics_graph_static_prior.json"
    [ -f "$fp" ] || continue
    python3 -c "
import json
m=json.load(open('$fp'))
print(f\"{'$cfg':<45} sp R@1={m.get('recall@1',0):.4f} R@5={m.get('recall@5',0):.4f} R@10={m.get('recall@10',0):.4f} MRR={m.get('mrr',0):.4f}\")"
done
