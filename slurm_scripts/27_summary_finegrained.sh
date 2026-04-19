#!/bin/bash
#SBATCH --job-name=summary_finegrained
#SBATCH --partition=cluster02
#SBATCH --qos=msc
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --gres=gpu:a6000:1
#SBATCH --time=05:00:00
#SBATCH --output=logs/summary_finegrained_%j.out
#SBATCH --error=logs/summary_finegrained_%j.err

# ============================================================
# 4.20 - Fine-grained fix for summary-based augmentation
#
# Principle: 细粒度是生命线 - do not homogenize siblings, do not
# drown short figure/formula passages.
#
# S1-fix (shadow boost):
#   * use ONLY chunk summaries (finer granularity than section)
#   * normalize boost by children_count^alpha (anti-group)
#   * cap max_children to exclude outlier large groups
#
# S2-fix (context aug):
#   * chunk summary prefix (not section)
#   * only inject on passages >= min_orig_words
#   * prefix <= 25 words (retrieval_hint style)
# ============================================================

set -euo pipefail

echo "============================================"
echo "Job: summary_finegrained  ID=${SLURM_JOB_ID}"
echo "Node: $(hostname)   Start: $(date)"
echo "============================================"

cd /projects/myyyx1/data-process-test
source /cluster/apps/software/Miniforge3/24.11.3-1/etc/profile.d/conda.sh
conda activate /projects/myyyx1/envs/minerU
mkdir -p logs

# ─── paths ─────────────────────────────────────────────────
BASE_DIR="data/05_eval/dense_retrieval/rebuilt_20260417"
AUG_DIR="data/05_eval/dense_retrieval/augmented_v2"
mkdir -p "$AUG_DIR"

IN_CORPUS="$BASE_DIR/augmented/corpus_v1_enriched.jsonl"
QUERIES_AUG="$BASE_DIR/augmented/queries.jsonl"
QRELS_AUG="$BASE_DIR/augmented/qrels_v1.jsonl"
QUERIES_M4="data/03_queries/M4query_v1/queries.jsonl"
QRELS_M4="data/03_queries/M4query_v1/qrels.jsonl"
RANKING_4B="$BASE_DIR/ranking_v1_enriched.jsonl"

HUB="data/02_enriched/hub_candidates_enriched_v3.json"
CHUNK_GRAPH="data/01_graphs/chunk_virtual_nodes.json"
ELEM_GRAPH="data/01_graphs/multimodal_elements.json"
SUM_GRAPH="data/01_graphs/llm_summary_virtual_nodes.json"

MODEL_NAME="Qwen/Qwen3-Embedding-0.6B"
DOWNLOAD_ROOT="models"

# ============================================================
# PART A - S2-fix: fine-grained context-augmented corpus + 0.6B eval
# ============================================================
echo ""
echo "=== PART A: fine-grained context augmentation + 0.6B eval ==="

declare -a VARIANTS=(
  "chunk_hint_mw40_pw25:chunk:hint:40:25"
  "chunk_hint_mw60_pw25:chunk:hint:60:25"
  "chunk_hint_mw80_pw25:chunk:hint:80:25"
  "chunk_summary_mw60_pw25:chunk:summary:60:25"
)

for spec in "${VARIANTS[@]}"; do
  IFS=':' read -r TAG SCOPE STYLE MINW PFXW <<< "$spec"
  OUT_CORPUS="$AUG_DIR/corpus_v1_${TAG}.jsonl"
  OUT_REPORT="$AUG_DIR/eval_${TAG}_qwen06b.json"
  OUT_RANKING="$AUG_DIR/ranking_${TAG}_qwen06b.jsonl"
  echo ""
  echo "── [A.$TAG] scope=$SCOPE style=$STYLE min_words=$MINW prefix_words=$PFXW ──"
  python scripts/build_context_aug_corpus_v2.py \
      --summary-graph "$SUM_GRAPH" \
      --chunk-graph "$CHUNK_GRAPH" \
      --elements-graph "$ELEM_GRAPH" \
      --in-corpus "$IN_CORPUS" \
      --out-corpus "$OUT_CORPUS" \
      --scope "$SCOPE" \
      --style "$STYLE" \
      --min-orig-words "$MINW" \
      --max-prefix-words "$PFXW"
  python scripts/eval_dense_retrieval.py \
      --data-dir "$AUG_DIR" \
      --queries "$QUERIES_AUG" \
      --corpus "$OUT_CORPUS" \
      --qrels "$QRELS_AUG" \
      --model-name "$MODEL_NAME" \
      --download-root "$DOWNLOAD_ROOT" \
      --output "$OUT_REPORT" \
      --save-ranking "$OUT_RANKING" \
      --batch-size 16 \
      --max-length 512 \
      --top-k 100
  echo "── done [$TAG] $(date)"
done

# ============================================================
# PART B - S1-fix: fine-grained summary shadow boost
# ============================================================
echo ""
echo "=== PART B: fine-grained shadow boost (chunk-scope + normalized) ==="

# Build two indices: chunk-only, and chunk-only with max_children<=5
IDX_CHUNK="data/01_graphs/summary_shadow_chunk_qwen06b.json"
IDX_CHUNK_MC5="data/01_graphs/summary_shadow_chunk_mc5_qwen06b.json"

python scripts/build_summary_shadow_index.py \
    --summary-graph "$SUM_GRAPH" \
    --chunk-graph "$CHUNK_GRAPH" \
    --elements-graph "$ELEM_GRAPH" \
    --corpus "$IN_CORPUS" \
    --queries "$QUERIES_M4" \
    --model-name "$MODEL_NAME" \
    --download-root "$DOWNLOAD_ROOT" \
    --batch-size 16 \
    --max-length 512 \
    --top-k-summaries 30 \
    --min-score 0.20 \
    --scope chunk \
    --output "$IDX_CHUNK"

python scripts/build_summary_shadow_index.py \
    --summary-graph "$SUM_GRAPH" \
    --chunk-graph "$CHUNK_GRAPH" \
    --elements-graph "$ELEM_GRAPH" \
    --corpus "$IN_CORPUS" \
    --queries "$QUERIES_M4" \
    --model-name "$MODEL_NAME" \
    --download-root "$DOWNLOAD_ROOT" \
    --batch-size 16 \
    --max-length 512 \
    --top-k-summaries 30 \
    --min-score 0.20 \
    --scope chunk \
    --max-children 5 \
    --output "$IDX_CHUNK_MC5"

rerank_4b () {
    local TAG="$1" IDX="$2" SRC="$3" W="$4" TOPN="$5" ALPHA="$6" MAXC="$7"
    local OUT="$BASE_DIR/graph_${TAG}"
    mkdir -p "$OUT"
    echo ""
    echo "── [B.$TAG]  idx=$(basename $IDX)  src=$SRC  w=$W topn=$TOPN alpha=$ALPHA maxc=$MAXC ──"
    python scripts/eval_graph_topk_rerank.py \
        --ranking "$RANKING_4B" \
        --qrels   "$QRELS_M4" \
        --corpus  "$IN_CORPUS" \
        --chunk-graph "$CHUNK_GRAPH" \
        --elements-graph "$ELEM_GRAPH" \
        --hub-candidates "$HUB" \
        --summary-shadow-index "$IDX" \
        --summary-shadow-weight "$W" \
        --summary-shadow-topn "$TOPN" \
        --summary-shadow-cap 0.20 \
        --summary-shadow-child-alpha "$ALPHA" \
        --summary-shadow-max-children "$MAXC" \
        --graph-sources $SRC \
        --output-dir "$OUT" \
        --top-k 100
    echo "── done [B.$TAG] $(date)"
}

# chunk-only scope, increasing child_alpha
rerank_4b "chunk_w010_t10_a1"            "$IDX_CHUNK"     "summary"           0.10 10 1.0 0
rerank_4b "chunk_w015_t15_a1"            "$IDX_CHUNK"     "summary"           0.15 15 1.0 0
rerank_4b "chunk_w020_t15_a05"           "$IDX_CHUNK"     "summary"           0.20 15 0.5 0
# chunk scope + max_children=5 as second layer of protection
rerank_4b "chunk_mc5_w015_t15_a1"        "$IDX_CHUNK_MC5" "summary"           0.15 15 1.0 0
rerank_4b "chunk_mc5_w020_t20_a1"        "$IDX_CHUNK_MC5" "summary"           0.20 20 1.0 0
# explicit + chunk shadow (best shot: keep winner signal, add fine-grained shadow)
rerank_4b "explicit_plus_chunk_w010_t10_a1" "$IDX_CHUNK"  "explicit summary"  0.10 10 1.0 0
rerank_4b "explicit_plus_chunk_w015_t15_a1" "$IDX_CHUNK"  "explicit summary"  0.15 15 1.0 0
rerank_4b "explicit_plus_chunk_mc5_w015_t15_a1" "$IDX_CHUNK_MC5" "explicit summary" 0.15 15 1.0 0

# ============================================================
# PART C - summary
# ============================================================
echo ""
echo "=== PART C: combined comparison ==="
python - <<'PY'
import json, glob
from pathlib import Path

print("\n── S2-fix (0.6B dense retrieval on augmented corpora) ──")
print(f'{"variant":<30} {"R@1":>8} {"R@5":>8} {"R@10":>8} {"R@100":>8} {"MRR":>8} {"NDCG@10":>10}')
print("-" * 95)
for rp in sorted(glob.glob("data/05_eval/dense_retrieval/augmented_v2/eval_*_qwen06b.json")):
    name = Path(rp).stem.replace("eval_","").replace("_qwen06b","")
    r = json.load(open(rp))
    m = r.get("metrics", {k:v for k,v in r.items() if isinstance(v,(int,float))})
    print(f'{name:<30} {m.get("recall@1",0):>8.4f} {m.get("recall@5",0):>8.4f} {m.get("recall@10",0):>8.4f} {m.get("recall@100",0):>8.4f} {m.get("mrr",0):>8.4f} {m.get("ndcg@10",0):>10.4f}')

print("\n── S1-fix (4B rerank with normalized chunk-shadow) vs winner ──")
print(f'{"variant/method":<55} {"R@1":>8} {"R@5":>8} {"R@10":>8} {"MRR":>8}')
print("-" * 95)
base = Path("data/05_eval/dense_retrieval/rebuilt_20260417")
dirs = sorted(glob.glob(str(base / "graph_chunk_*"))) + sorted(glob.glob(str(base / "graph_explicit_plus_chunk_*")))
dirs = [str(base / "graph_explicit_only_fixed")] + dirs
for d in dirs:
    rep = Path(d) / "report.json"
    if not rep.exists(): continue
    r = json.loads(rep.read_text())
    for mname, met in r["results"].items():
        tag = f'{Path(d).name}/{mname}'
        print(f'{tag:<55} {met["recall@1"]:>8.4f} {met["recall@5"]:>8.4f} {met["recall@10"]:>8.4f} {met["mrr"]:>8.4f}')
PY

echo ""
echo "============================================"
echo "summary_finegrained COMPLETE: $(date)"
echo "============================================"
