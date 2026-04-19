#!/bin/bash
#SBATCH --job-name=summary_shadow_eval
#SBATCH --partition=cluster02
#SBATCH --qos=msc
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --gres=gpu:a6000:1
#SBATCH --time=04:00:00
#SBATCH --output=logs/summary_shadow_eval_%j.out
#SBATCH --error=logs/summary_shadow_eval_%j.err

# ============================================================
# 4.20 deliverable - S1: Summary as shadow rerank node
#
# Hypothesis: LLM section/chunk summaries SHOULD help R@1 *only* when used
# as a side index (boost children pids), NOT inserted into the corpus.
# Prior 4.16 v3_summaries failed because summaries entered the candidate pool.
#
# Steps:
#   1. Build summary shadow index (encode 2767 summaries + queries with
#      Qwen3-Embedding-0.6B; emit per-query top-K summaries).
#   2. Rerank existing 4B baseline ranking with explicit + summary shadow.
#   3. If 0.6B ranking from Job 61297 exists, rerun on it too.
#
# Compare against:
#   data/05_eval/dense_retrieval/rebuilt_20260417/graph_explicit_only_fixed/
#     metrics_graph_static_prior.json  ->  R@1=0.2421 R@5=0.5856 MRR=0.6399
# ============================================================

set -euo pipefail

echo "============================================"
echo "Job: summary_shadow_eval (4.20 S1)"
echo "Job ID:  ${SLURM_JOB_ID}"
echo "Node:    $(hostname)"
echo "Start:   $(date)"
echo "============================================"

cd /projects/myyyx1/data-process-test
source /cluster/apps/software/Miniforge3/24.11.3-1/etc/profile.d/conda.sh
conda activate /projects/myyyx1/envs/minerU

mkdir -p logs

# ─── Shared paths ────────────────────────────────────────────────────────
BASE_DIR="data/05_eval/dense_retrieval/rebuilt_20260417"
AUG_DIR_V2="data/05_eval/dense_retrieval/augmented_v2"   # output of Job 61297
QRELS="data/03_queries/M4query_v1/qrels.jsonl"
QUERIES="data/03_queries/M4query_v1/queries.jsonl"
CORPUS="$BASE_DIR/augmented/corpus_v1_enriched.jsonl"
HUB="data/02_enriched/hub_candidates_enriched_v3.json"
CHUNK_GRAPH="data/01_graphs/chunk_virtual_nodes.json"
ELEM_GRAPH="data/01_graphs/multimodal_elements.json"
SUM_GRAPH="data/01_graphs/llm_summary_virtual_nodes.json"

# ─── Model selection (use 0.6B as mentor recommended) ────────────────────
MODEL_NAME="Qwen/Qwen3-Embedding-0.6B"
DOWNLOAD_ROOT="models"

# ─── Summary shadow index path ───────────────────────────────────────────
SUM_IDX="data/01_graphs/summary_shadow_index_qwen06b.json"

# ============================================================
# STEP 1 - build summary shadow index
# ============================================================
echo ""
echo "=== Step 1: build summary shadow index ==="
python scripts/build_summary_shadow_index.py \
    --summary-graph "$SUM_GRAPH" \
    --chunk-graph   "$CHUNK_GRAPH" \
    --elements-graph "$ELEM_GRAPH" \
    --corpus        "$CORPUS" \
    --queries       "$QUERIES" \
    --model-name    "$MODEL_NAME" \
    --download-root "$DOWNLOAD_ROOT" \
    --batch-size    16 \
    --max-length    512 \
    --top-k-summaries 30 \
    --min-score     0.20 \
    --output        "$SUM_IDX"
echo "  Step 1 done: $(date)"

# ============================================================
# STEP 2 - rerank 4B baseline with summary shadow boost
#          (sweep weight to find sweet spot)
# ============================================================
RANKING_4B="$BASE_DIR/ranking_v1_enriched.jsonl"

run_rerank () {
    local TAG="$1"
    local RANKING="$2"
    local SOURCES="$3"
    local WEIGHT="$4"
    local TOPN="$5"
    local OUT_DIR="$BASE_DIR/graph_${TAG}"
    mkdir -p "$OUT_DIR"
    echo ""
    echo "  ── [$TAG]  sources=$SOURCES  w=$WEIGHT  topn=$TOPN ──"
    python scripts/eval_graph_topk_rerank.py \
        --ranking "$RANKING" \
        --qrels   "$QRELS" \
        --corpus  "$CORPUS" \
        --chunk-graph    "$CHUNK_GRAPH" \
        --elements-graph "$ELEM_GRAPH" \
        --hub-candidates "$HUB" \
        --summary-shadow-index "$SUM_IDX" \
        --summary-shadow-weight "$WEIGHT" \
        --summary-shadow-topn   "$TOPN" \
        --summary-shadow-cap    0.20 \
        --graph-sources $SOURCES \
        --output-dir "$OUT_DIR" \
        --top-k 100
    echo "  ── done [$TAG] $(date)"
}

echo ""
echo "=== Step 2: rerank on 4B ranking (existing) ==="

# A: summary alone
run_rerank "summary_only_w010_t10_4B" "$RANKING_4B" "summary" 0.10 10
run_rerank "summary_only_w015_t15_4B" "$RANKING_4B" "summary" 0.15 15
# B: explicit + summary
run_rerank "explicit_plus_summary_w010_t10_4B" "$RANKING_4B" "explicit summary" 0.10 10
run_rerank "explicit_plus_summary_w015_t15_4B" "$RANKING_4B" "explicit summary" 0.15 15
run_rerank "explicit_plus_summary_w020_t20_4B" "$RANKING_4B" "explicit summary" 0.20 20

# ============================================================
# STEP 3 - if Job 61297 finished, rerank on 0.6B ranking too
# ============================================================
RANKING_06B="$AUG_DIR_V2/ranking_v1_enriched_qwen06b.jsonl"
if [[ -s "$RANKING_06B" ]]; then
    echo ""
    echo "=== Step 3: rerank on 0.6B ranking ==="
    OUT_BASE="$AUG_DIR_V2/graph_rerank"
    mkdir -p "$OUT_BASE"

    # Need a corpus that matches the ranking. v1_enriched should be same set of pids.
    CORPUS_06B="$AUG_DIR_V2/corpus_v1_enriched.jsonl"
    if [[ ! -s "$CORPUS_06B" ]]; then
        CORPUS_06B="$CORPUS"
    fi

    run_rerank_06b () {
        local TAG="$1"; local SOURCES="$2"; local WEIGHT="$3"; local TOPN="$4"
        local OUT_DIR="$OUT_BASE/${TAG}"
        mkdir -p "$OUT_DIR"
        echo ""
        echo "  ── [0.6B/$TAG]  sources=$SOURCES  w=$WEIGHT  topn=$TOPN ──"
        python scripts/eval_graph_topk_rerank.py \
            --ranking "$RANKING_06B" \
            --qrels   "$QRELS" \
            --corpus  "$CORPUS_06B" \
            --chunk-graph    "$CHUNK_GRAPH" \
            --elements-graph "$ELEM_GRAPH" \
            --hub-candidates "$HUB" \
            --summary-shadow-index "$SUM_IDX" \
            --summary-shadow-weight "$WEIGHT" \
            --summary-shadow-topn   "$TOPN" \
            --summary-shadow-cap    0.20 \
            --graph-sources $SOURCES \
            --output-dir "$OUT_DIR" \
            --top-k 100
        echo "  ── done [0.6B/$TAG] $(date)"
    }

    run_rerank_06b "explicit_only"                  "explicit"          0.10 10
    run_rerank_06b "summary_only_w010_t10"          "summary"           0.10 10
    run_rerank_06b "summary_only_w015_t15"          "summary"           0.15 15
    run_rerank_06b "explicit_plus_summary_w010_t10" "explicit summary"  0.10 10
    run_rerank_06b "explicit_plus_summary_w015_t15" "explicit summary"  0.15 15
    run_rerank_06b "explicit_plus_summary_w020_t20" "explicit summary"  0.20 20
else
    echo ""
    echo "[skip] Step 3 skipped, 0.6B ranking not found at $RANKING_06B"
    echo "       Re-run this step manually after Job 61297 produces it."
fi

# ============================================================
# STEP 4 - comparison summary
# ============================================================
echo ""
echo "=== Step 4: comparison ==="
python - <<'PY'
import json, glob
from pathlib import Path

base = Path("data/05_eval/dense_retrieval/rebuilt_20260417")
aug2 = Path("data/05_eval/dense_retrieval/augmented_v2/graph_rerank")

def show(label, dirs):
    print(f"\n── {label} ──")
    print(f'{"variant":<48} {"R@1":>8} {"R@5":>8} {"R@10":>8} {"MRR":>8}')
    print("-" * 90)
    for d in sorted(dirs):
        rep = Path(d) / "report.json"
        if not rep.exists():
            continue
        r = json.loads(rep.read_text())
        for m, met in r["results"].items():
            print(f'{Path(d).name + "/" + m:<48} {met["recall@1"]:>8.4f} {met["recall@5"]:>8.4f} {met["recall@10"]:>8.4f} {met["mrr"]:>8.4f}')

show("4B reranks (summary shadow ablation)", glob.glob(str(base / "graph_*summary*4B")) + [str(base / "graph_explicit_only_fixed")])
if aug2.exists():
    show("0.6B reranks (summary shadow ablation)", [str(p) for p in aug2.iterdir() if p.is_dir()])
PY

echo ""
echo "============================================"
echo "summary_shadow_eval COMPLETE: $(date)"
echo "  Index:     $SUM_IDX"
echo "  4B out:    $BASE_DIR/graph_*summary*_4B/"
echo "  0.6B out:  $AUG_DIR_V2/graph_rerank/   (if Step 3 ran)"
echo "============================================"
