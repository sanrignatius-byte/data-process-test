#!/bin/bash
#SBATCH --job-name=stack_06b_w
#SBATCH --output=logs/stack_06b_w_%j.out
#SBATCH --error=logs/stack_06b_w_%j.err
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
CHUNK_V2="data/01_graphs/chunk_virtual_nodes_v2.json"
ELEMENTS="data/01_graphs/multimodal_elements.json"
CROSS_DOC="data/01_graphs/cross_doc_sim_edges.json"
OUT_BASE="data/05_eval/dense_retrieval/stacking_06b_weighted"

mkdir -p "$OUT_BASE"

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

# ======================================================
# Series 1: explicit + virtual, scan virtual weight down
# hypothesis: virtual edges are noise; reducing their weight should recover explicit_only level
# ======================================================

run_rerank "ex1.0_vir1.0_degree" \
    --graph-sources explicit virtual --edge-types same_chunk chunk_sequence same_section_nonseq \
    --explicit-weight 1.0 --virtual-weight 1.0

run_rerank "ex1.0_vir0.5_degree" \
    --graph-sources explicit virtual --edge-types same_chunk chunk_sequence same_section_nonseq \
    --explicit-weight 1.0 --virtual-weight 0.5

run_rerank "ex1.0_vir0.2_degree" \
    --graph-sources explicit virtual --edge-types same_chunk chunk_sequence same_section_nonseq \
    --explicit-weight 1.0 --virtual-weight 0.2

run_rerank "ex1.0_vir0.1_degree" \
    --graph-sources explicit virtual --edge-types same_chunk chunk_sequence same_section_nonseq \
    --explicit-weight 1.0 --virtual-weight 0.1

# Same but with weighted prior (uses weights, not just degree)
run_rerank "ex1.0_vir0.2_weighted" \
    --graph-sources explicit virtual --edge-types same_chunk chunk_sequence same_section_nonseq \
    --explicit-weight 1.0 --virtual-weight 0.2 --prior-mode weighted

run_rerank "ex1.0_vir0.1_weighted" \
    --graph-sources explicit virtual --edge-types same_chunk chunk_sequence same_section_nonseq \
    --explicit-weight 1.0 --virtual-weight 0.1 --prior-mode weighted

# Explicit boosted up, virtual normal (amplify strong)
run_rerank "ex3.0_vir1.0_weighted" \
    --graph-sources explicit virtual --edge-types same_chunk chunk_sequence same_section_nonseq \
    --explicit-weight 3.0 --virtual-weight 1.0 --prior-mode weighted

run_rerank "ex5.0_vir1.0_weighted" \
    --graph-sources explicit virtual --edge-types same_chunk chunk_sequence same_section_nonseq \
    --explicit-weight 5.0 --virtual-weight 1.0 --prior-mode weighted

# ======================================================
# Series 2: explicit + crossdoc (crossdoc seemed inert in A)
# ======================================================

run_rerank "ex1.0_cd0.5_weighted" \
    --graph-sources explicit cross_doc \
    --explicit-weight 1.0 --crossdoc-weight 0.5 --prior-mode weighted

run_rerank "ex1.0_cd2.0_weighted" \
    --graph-sources explicit cross_doc \
    --explicit-weight 1.0 --crossdoc-weight 2.0 --prior-mode weighted

# ======================================================
# Series 3: all three sources with tuned weights (the "full stack" config)
# ======================================================

run_rerank "ex1.0_vir0.2_cd0.5_weighted" \
    --graph-sources explicit virtual cross_doc --edge-types same_chunk chunk_sequence same_section_nonseq \
    --explicit-weight 1.0 --virtual-weight 0.2 --crossdoc-weight 0.5 --prior-mode weighted

run_rerank "ex1.0_vir0.1_cd1.0_sum_weighted" \
    --graph-sources explicit virtual cross_doc --edge-types same_chunk chunk_sequence same_section_nonseq \
    --explicit-weight 1.0 --virtual-weight 0.1 --crossdoc-weight 1.0 \
    --merge-combine sum --prior-mode weighted

run_rerank "ex2.0_vir0.2_cd0.5_sum_weighted" \
    --graph-sources explicit virtual cross_doc --edge-types same_chunk chunk_sequence same_section_nonseq \
    --explicit-weight 2.0 --virtual-weight 0.2 --crossdoc-weight 0.5 \
    --merge-combine sum --prior-mode weighted

echo ""
echo "=== ALL DONE: $(date) ==="
echo ""
echo "== Summary: static_prior method =="
printf "%-45s %8s %8s %8s %8s\n" "config" "R@1" "R@5" "R@10" "MRR"
echo "--------------------------------------------------------------------------------"
# include the explicit_only baseline from series A for reference
REF="data/05_eval/dense_retrieval/stacking_06b/explicit_only/metrics_graph_static_prior.json"
if [ -f "$REF" ]; then
    python3 -c "
import json
m=json.load(open('$REF'))
print(f\"{'[ref] explicit_only_06b':<45} {m.get('recall@1',0):8.4f} {m.get('recall@5',0):8.4f} {m.get('recall@10',0):8.4f} {m.get('mrr',0):8.4f}\")"
fi
for d in "$OUT_BASE"/*/; do
    cfg=$(basename "$d")
    fp="$d/metrics_graph_static_prior.json"
    [ -f "$fp" ] || continue
    python3 -c "
import json
m=json.load(open('$fp'))
print(f\"{'$cfg':<45} {m.get('recall@1',0):8.4f} {m.get('recall@5',0):8.4f} {m.get('recall@10',0):8.4f} {m.get('mrr',0):8.4f}\")"
done

echo ""
echo "== Summary: neighbor_prop method (responds to edge weights) =="
printf "%-45s %8s %8s %8s %8s\n" "config" "R@1" "R@5" "R@10" "MRR"
echo "--------------------------------------------------------------------------------"
for d in "$OUT_BASE"/*/; do
    cfg=$(basename "$d")
    fp="$d/metrics_graph_neighbor_prop.json"
    [ -f "$fp" ] || continue
    python3 -c "
import json
m=json.load(open('$fp'))
print(f\"{'$cfg':<45} {m.get('recall@1',0):8.4f} {m.get('recall@5',0):8.4f} {m.get('recall@10',0):8.4f} {m.get('mrr',0):8.4f}\")"
done
