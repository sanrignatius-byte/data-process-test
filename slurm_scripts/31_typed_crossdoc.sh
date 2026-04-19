#!/bin/bash
#SBATCH --job-name=typed_crossdoc
#SBATCH --output=logs/typed_crossdoc_%j.out
#SBATCH --error=logs/typed_crossdoc_%j.err
#SBATCH --partition=cluster02
#SBATCH --qos=msc
#SBATCH --gres=gpu:a6000:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=02:00:00

set -e
cd /projects/myyyx1/data-process-test
source /cluster/apps/software/Miniforge3/24.11.3-1/etc/profile.d/conda.sh
conda activate /projects/myyyx1/envs/minerU
export PYTHONPATH=/projects/myyyx1/data-process-test:$PYTHONPATH
export MODELSCOPE_CACHE=models

AUG_V2="data/05_eval/dense_retrieval/augmented_v2"
RANK_V1_06B="data/05_eval/dense_retrieval/rebuilt_20260417/graph_06b_v1_explicit_v1chunk/ranking_baseline.jsonl"
HUB_CANDS="data/02_enriched/hub_candidates_enriched_v3.json"
CHUNK_V2="data/01_graphs/chunk_virtual_nodes_v2.json"
ELEMENTS="data/01_graphs/multimodal_elements.json"
CROSS_DOC_FIXED="data/01_graphs/cross_doc_sim_edges.json"
TYPED_EDGES="data/01_graphs/typed_crossdoc_edges.json"
CITATION="data/01_graphs/citation_graph.json"
OUT_BASE="data/05_eval/dense_retrieval/typed_crossdoc"

mkdir -p "$OUT_BASE"

# ============================================================
# PHASE 1: Build typed cross-doc edges (figure, formula, table, chunk)
# ============================================================
echo "=== Phase 1: Build typed cross-doc edges ==="
date
if [ -f "$TYPED_EDGES" ] && python3 -c "
import json; d=json.load(open('$TYPED_EDGES'))
assert d.get('stats',{}).get('num_edges_total',0) > 0
" 2>/dev/null; then
    echo "  Already exists, skipping build."
    python3 -c "import json; print(json.dumps(json.load(open('$TYPED_EDGES'))['stats'], indent=2))"
else
    python scripts/build_typed_crossdoc_edges.py \
        --corpus "$AUG_V2/corpus_v1_enriched.jsonl" \
        --chunk-graph "$CHUNK_V2" \
        --citation-graph "$CITATION" \
        --output "$TYPED_EDGES" \
        --model-name "Qwen/Qwen3-Embedding-0.6B" \
        --download-root models \
        --types figure formula table \
        --threshold 0.70 \
        --top-k 10 \
        --cite-boost 0.05 \
        --batch-size 32 \
        --max-length 512
fi
echo "  Phase 1 done: $(date)"

# Helper
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
        --cross-doc-edges "$CROSS_DOC_FIXED" \
        --typed-crossdoc-edges "$TYPED_EDGES" \
        --output-dir "$outdir" \
        --top-k 100 \
        "$@"
}

# ============================================================
# PHASE 2: Re-evaluate section-level cross_doc (now bug-fixed)
# ============================================================
echo ""
echo "=== Phase 2: Fixed section-level cross_doc ==="

run_rerank "crossdoc_sec_only_FIXED" \
    --graph-sources cross_doc --prior-mode weighted

run_rerank "explicit_plus_crossdoc_sec_FIXED" \
    --graph-sources explicit cross_doc \
    --explicit-weight 1.0 --crossdoc-weight 0.5 --prior-mode weighted

# ============================================================
# PHASE 3: Typed crossdoc (element-level, new)
# ============================================================
echo ""
echo "=== Phase 3: Typed crossdoc experiments ==="

# 3.1 typed only (pure new signal)
run_rerank "typed_only_all" \
    --graph-sources typed_crossdoc --prior-mode weighted

# 3.2 typed only — by single type
run_rerank "typed_only_figure" \
    --graph-sources typed_crossdoc --typed-crossdoc-types figure --prior-mode weighted

run_rerank "typed_only_formula" \
    --graph-sources typed_crossdoc --typed-crossdoc-types formula --prior-mode weighted

run_rerank "typed_only_table" \
    --graph-sources typed_crossdoc --typed-crossdoc-types table --prior-mode weighted

# 3.3 typed with citation boost weight
run_rerank "typed_only_all_boosted" \
    --graph-sources typed_crossdoc --typed-crossdoc-use-boost --prior-mode weighted

# 3.4 explicit + typed (main combo)
run_rerank "explicit_plus_typed_all_w1.0" \
    --graph-sources explicit typed_crossdoc \
    --explicit-weight 1.0 --typed-crossdoc-weight 1.0 --prior-mode weighted

run_rerank "explicit_plus_typed_all_w0.5" \
    --graph-sources explicit typed_crossdoc \
    --explicit-weight 1.0 --typed-crossdoc-weight 0.5 --prior-mode weighted

run_rerank "explicit_plus_typed_all_w0.2" \
    --graph-sources explicit typed_crossdoc \
    --explicit-weight 1.0 --typed-crossdoc-weight 0.2 --prior-mode weighted

run_rerank "explicit_plus_typed_all_w2.0" \
    --graph-sources explicit typed_crossdoc \
    --explicit-weight 1.0 --typed-crossdoc-weight 2.0 --prior-mode weighted

# 3.5 explicit + typed (citation-boosted weights)
run_rerank "explicit_plus_typed_boosted_w1.0" \
    --graph-sources explicit typed_crossdoc \
    --explicit-weight 1.0 --typed-crossdoc-weight 1.0 \
    --typed-crossdoc-use-boost --prior-mode weighted

# 3.6 explicit + typed element types only (no chunk)
run_rerank "explicit_plus_typed_elements_w1.0" \
    --graph-sources explicit typed_crossdoc \
    --typed-crossdoc-types figure formula table \
    --explicit-weight 1.0 --typed-crossdoc-weight 1.0 --prior-mode weighted

# 3.7 kitchen sink: explicit + sec_crossdoc + typed
run_rerank "explicit_plus_sec_plus_typed" \
    --graph-sources explicit cross_doc typed_crossdoc \
    --explicit-weight 1.0 --crossdoc-weight 0.5 --typed-crossdoc-weight 1.0 \
    --prior-mode weighted --merge-combine sum

run_rerank "explicit_plus_sec_plus_typed_max" \
    --graph-sources explicit cross_doc typed_crossdoc \
    --explicit-weight 1.0 --crossdoc-weight 0.5 --typed-crossdoc-weight 1.0 \
    --prior-mode weighted --merge-combine max

# ============================================================
# SUMMARY
# ============================================================
echo ""
echo "=== ALL DONE: $(date) ==="
echo ""
echo "== Baseline references =="
REF_BASE="data/05_eval/dense_retrieval/stacking_06b/explicit_only/metrics_baseline.json"
REF_EX="data/05_eval/dense_retrieval/stacking_06b/explicit_only/metrics_graph_static_prior.json"
for fp in "$REF_BASE:dense_06b_baseline" "$REF_EX:[ref]_explicit_only_06b"; do
    path="${fp%%:*}"; label="${fp##*:}"
    [ -f "$path" ] || continue
    python3 -c "
import json
m=json.load(open('$path'))
print(f\"{'$label':<48} {m.get('recall@1',0):8.4f} {m.get('recall@5',0):8.4f} {m.get('recall@10',0):8.4f} {m.get('mrr',0):8.4f}\")"
done

echo ""
echo "== static_prior (primary) =="
printf "%-48s %8s %8s %8s %8s\n" "config" "R@1" "R@5" "R@10" "MRR"
echo "-------------------------------------------------------------------------------------------"
for d in "$OUT_BASE"/*/; do
    cfg=$(basename "$d")
    fp="$d/metrics_graph_static_prior.json"
    [ -f "$fp" ] || continue
    python3 -c "
import json
m=json.load(open('$fp'))
print(f\"{'$cfg':<48} {m.get('recall@1',0):8.4f} {m.get('recall@5',0):8.4f} {m.get('recall@10',0):8.4f} {m.get('mrr',0):8.4f}\")"
done

echo ""
echo "== neighbor_prop =="
printf "%-48s %8s %8s %8s %8s\n" "config" "R@1" "R@5" "R@10" "MRR"
echo "-------------------------------------------------------------------------------------------"
for d in "$OUT_BASE"/*/; do
    cfg=$(basename "$d")
    fp="$d/metrics_graph_neighbor_prop.json"
    [ -f "$fp" ] || continue
    python3 -c "
import json
m=json.load(open('$fp'))
print(f\"{'$cfg':<48} {m.get('recall@1',0):8.4f} {m.get('recall@5',0):8.4f} {m.get('recall@10',0):8.4f} {m.get('mrr',0):8.4f}\")"
done

echo ""
echo "== static_plus_neighbor =="
printf "%-48s %8s %8s %8s %8s\n" "config" "R@1" "R@5" "R@10" "MRR"
echo "-------------------------------------------------------------------------------------------"
for d in "$OUT_BASE"/*/; do
    cfg=$(basename "$d")
    fp="$d/metrics_graph_static_plus_neighbor.json"
    [ -f "$fp" ] || continue
    python3 -c "
import json
m=json.load(open('$fp'))
print(f\"{'$cfg':<48} {m.get('recall@1',0):8.4f} {m.get('recall@5',0):8.4f} {m.get('recall@10',0):8.4f} {m.get('mrr',0):8.4f}\")"
done
