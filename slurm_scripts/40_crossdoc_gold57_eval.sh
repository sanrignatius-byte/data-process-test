#!/bin/bash
#SBATCH --job-name=crossdoc_gold57
#SBATCH --output=logs/crossdoc_gold57_%j.out
#SBATCH --error=logs/crossdoc_gold57_%j.err
#SBATCH --partition=cluster02
#SBATCH --qos=msc
#SBATCH --gres=gpu:a6000:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=03:00:00

set -euo pipefail
cd /projects/myyyx1/data-process-test
source /cluster/apps/software/Miniforge3/24.11.3-1/etc/profile.d/conda.sh
conda activate /projects/myyyx1/envs/minerU
export PYTHONPATH=/projects/myyyx1/data-process-test:${PYTHONPATH:-}
export MODELSCOPE_CACHE=models

# ── Paths ─────────────────────────────────────────────────────────────────────
M4Q="data/03_queries/M4query_v1"
REBUILT="data/05_eval/dense_retrieval/rebuilt_20260417"
RANK="$REBUILT/ranking_qwen3_4B.jsonl"
CHUNK_GRAPH_ENRICHED="data/01_graphs/paragraph_chunks_n400_trial57_enriched.json"
CITATION="data/01_graphs/citation_graph.json"
HUB_CANDS="data/02_enriched/hub_candidates_enriched_v3.json"
CHUNK_V2="data/01_graphs/chunk_virtual_nodes_v2.json"
ELEMENTS="data/01_graphs/multimodal_elements.json"
CROSS_DOC_SEC="data/01_graphs/cross_doc_sim_edges.json"

OUT_GRAPH="data/01_graphs/crossdoc_gold57.json"
OUT_PREFIX="data/01_graphs/crossdoc_gold57"          # produces _report.json _projected.json
OUT_EVAL="data/05_eval/dense_retrieval/crossdoc_gold57_eval"
mkdir -p "$OUT_EVAL"

echo "=== Phase 1: build crossdoc edges (figure/formula/table + chunk) with Qwen3-Embedding-0.6B ==="
date
if [ -f "$OUT_GRAPH" ] && python3 -c "
import json
d=json.load(open('$OUT_GRAPH'))
assert not d['metadata'].get('citation_only', False)
assert d['stats']['num_edges_total'] > 0
" 2>/dev/null; then
    echo "[skip] $OUT_GRAPH already exists (embedding build)"
    python3 -c "import json; print(json.dumps(json.load(open('$OUT_GRAPH'))['stats'], indent=2))"
else
    python scripts/build_crossdoc_gold57.py \
        --qrels "$M4Q/qrels.jsonl" \
        --corpus "$M4Q/corpus.jsonl" \
        --chunk-graph "$CHUNK_GRAPH_ENRICHED" \
        --citation-graph "$CITATION" \
        --output "$OUT_GRAPH" \
        --model-name "Qwen/Qwen3-Embedding-0.6B" \
        --download-root models \
        --types figure formula table \
        --include-chunks \
        --threshold 0.70 \
        --top-k 10 \
        --cite-boost 0.05 \
        --batch-size 32 \
        --max-length 512
fi
echo "[phase 1 done] $(date)"

echo ""
echo "=== Phase 2: validate + project chunk->element edges ==="
python scripts/validate_and_project_crossdoc.py \
    --crossdoc "$OUT_GRAPH" \
    --qrels "$M4Q/qrels.jsonl" \
    --corpus "$M4Q/corpus.jsonl" \
    --ranking "$RANK" \
    --citation-graph "$CITATION" \
    --output-prefix "$OUT_PREFIX" \
    --chunk-project-cap 20 \
    --chunk-project-min-sim 0.70 \
    --top-k 100
echo "[phase 2 done] $(date)"

PROJECTED="${OUT_PREFIX}_projected.json"
ELEMENT_ONLY="$OUT_GRAPH"   # contains figure/formula/table at element level

echo ""
echo "=== Phase 3: element-level graph rerank over rebuilt_20260417 ranking ==="
echo "    qrels = M4query_v1 (element-only; chunks don't appear in gold)"
echo "    ranking = rebuilt_20260417 (element corpus)"

run_eval() {
    local name="$1"; shift
    local outdir="$OUT_EVAL/$name"
    mkdir -p "$outdir"
    echo ""
    echo "--- $name ---"
    python scripts/eval_graph_topk_rerank.py \
        --ranking "$RANK" \
        --qrels "$M4Q/qrels.jsonl" \
        --corpus "$M4Q/corpus.jsonl" \
        --hub-candidates "$HUB_CANDS" \
        --chunk-graph "$CHUNK_V2" \
        --elements-graph "$ELEMENTS" \
        --cross-doc-edges "$CROSS_DOC_SEC" \
        --output-dir "$outdir" \
        --top-k 100 \
        "$@"
}

# 3.0 baseline (no graph) — sanity reference
run_eval "baseline_nograph" \
    --graph-sources explicit --explicit-weight 0.0 --prior-mode weighted || true

# 3.1 explicit only (reference)
run_eval "explicit_only" \
    --graph-sources explicit --prior-mode weighted

# 3.2 crossdoc: element edges only (figure/formula/table from gold57 build, NO chunk projection)
run_eval "crossdoc_elem_only" \
    --typed-crossdoc-edges "$ELEMENT_ONLY" \
    --graph-sources typed_crossdoc \
    --typed-crossdoc-types figure formula table \
    --prior-mode weighted

run_eval "crossdoc_elem_only_boosted" \
    --typed-crossdoc-edges "$ELEMENT_ONLY" \
    --graph-sources typed_crossdoc \
    --typed-crossdoc-types figure formula table \
    --typed-crossdoc-use-boost \
    --prior-mode weighted

# 3.3 crossdoc: element + chunk-mediated projection
run_eval "crossdoc_with_chunk_projection" \
    --typed-crossdoc-edges "$PROJECTED" \
    --graph-sources typed_crossdoc \
    --typed-crossdoc-types figure formula table chunk_mediated \
    --prior-mode weighted

run_eval "crossdoc_with_chunk_projection_boosted" \
    --typed-crossdoc-edges "$PROJECTED" \
    --graph-sources typed_crossdoc \
    --typed-crossdoc-types figure formula table chunk_mediated \
    --typed-crossdoc-use-boost \
    --prior-mode weighted

# 3.4 combined: explicit + crossdoc with chunk projection
run_eval "explicit_plus_crossdoc_chunk_projection" \
    --typed-crossdoc-edges "$PROJECTED" \
    --graph-sources explicit typed_crossdoc \
    --typed-crossdoc-types figure formula table chunk_mediated \
    --explicit-weight 1.0 --typed-crossdoc-weight 0.5 \
    --prior-mode weighted

run_eval "explicit_plus_crossdoc_chunk_projection_w0.2" \
    --typed-crossdoc-edges "$PROJECTED" \
    --graph-sources explicit typed_crossdoc \
    --typed-crossdoc-types figure formula table chunk_mediated \
    --explicit-weight 1.0 --typed-crossdoc-weight 0.2 \
    --prior-mode weighted

# 3.5 combined: explicit + crossdoc elem only (control — chunk projection contribution isolated)
run_eval "explicit_plus_crossdoc_elem_only" \
    --typed-crossdoc-edges "$ELEMENT_ONLY" \
    --graph-sources explicit typed_crossdoc \
    --typed-crossdoc-types figure formula table \
    --explicit-weight 1.0 --typed-crossdoc-weight 0.5 \
    --prior-mode weighted

echo ""
echo "=== ALL DONE: $(date) ==="

# ── summary ──────────────────────────────────────────────────────────────────
echo ""
echo "== Summary (static_plus_neighbor) =="
printf "%-50s %8s %8s %8s %8s %8s\n" "config" "R@1" "R@5" "R@10" "R@100" "MRR"
echo "---------------------------------------------------------------------------------------------"
for d in "$OUT_EVAL"/*/; do
    cfg=$(basename "$d")
    fp="$d/metrics_graph_static_plus_neighbor.json"
    [ -f "$fp" ] || continue
    python3 -c "
import json
m=json.load(open('$fp'))
print(f\"{'$cfg':<50} {m.get('recall@1',0):8.4f} {m.get('recall@5',0):8.4f} {m.get('recall@10',0):8.4f} {m.get('recall@100',0):8.4f} {m.get('mrr',0):8.4f}\")"
done

echo ""
echo "== Summary (static_prior) =="
printf "%-50s %8s %8s %8s %8s %8s\n" "config" "R@1" "R@5" "R@10" "R@100" "MRR"
echo "---------------------------------------------------------------------------------------------"
for d in "$OUT_EVAL"/*/; do
    cfg=$(basename "$d")
    fp="$d/metrics_graph_static_prior.json"
    [ -f "$fp" ] || continue
    python3 -c "
import json
m=json.load(open('$fp'))
print(f\"{'$cfg':<50} {m.get('recall@1',0):8.4f} {m.get('recall@5',0):8.4f} {m.get('recall@10',0):8.4f} {m.get('recall@100',0):8.4f} {m.get('mrr',0):8.4f}\")"
done
