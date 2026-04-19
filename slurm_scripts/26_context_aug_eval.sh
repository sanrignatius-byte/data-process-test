#!/bin/bash
#SBATCH --job-name=context_aug_eval
#SBATCH --partition=cluster02
#SBATCH --qos=msc
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:a6000:1
#SBATCH --time=05:00:00
#SBATCH --output=logs/context_aug_eval_%j.out
#SBATCH --error=logs/context_aug_eval_%j.err

# ============================================================
# 4.20 deliverable - S2: Section-context augmented corpus
#
# Hypothesis: Prepending each passage with its parent section's LLM summary
# enriches the embedding without expanding the candidate pool. Orthogonal to S1.
#
# Steps:
#   1. Build augmented corpus (corpus_v1_sec_context.jsonl).
#   2. Dense retrieval eval on Qwen3-Embedding-0.6B.
#   3. Compare against v1_enriched (same corpus but no section prefix).
# ============================================================

set -euo pipefail

echo "============================================"
echo "Job: context_aug_eval (4.20 S2)"
echo "Job ID:  ${SLURM_JOB_ID}"
echo "Node:    $(hostname)"
echo "Start:   $(date)"
echo "============================================"

cd /projects/myyyx1/data-process-test
source /cluster/apps/software/Miniforge3/24.11.3-1/etc/profile.d/conda.sh
conda activate /projects/myyyx1/envs/minerU

mkdir -p logs

# ─── Paths ──────────────────────────────────────────────────────────────
BASE_DIR="data/05_eval/dense_retrieval/rebuilt_20260417"
AUG_DIR="data/05_eval/dense_retrieval/augmented_v2"
mkdir -p "$AUG_DIR"

IN_CORPUS="$BASE_DIR/augmented/corpus_v1_enriched.jsonl"
QRELS="$BASE_DIR/augmented/qrels_v1.jsonl"
QUERIES="$BASE_DIR/augmented/queries.jsonl"
SUM_GRAPH="data/01_graphs/llm_summary_virtual_nodes.json"
CHUNK_GRAPH="data/01_graphs/chunk_virtual_nodes.json"
ELEM_GRAPH="data/01_graphs/multimodal_elements.json"

MODEL_NAME="Qwen/Qwen3-Embedding-0.6B"
DOWNLOAD_ROOT="models"

# Variants: 60-word section prefix; 60-word + keywords
OUT_CORPUS_A="$AUG_DIR/corpus_v1_sec_context_w60.jsonl"
OUT_CORPUS_B="$AUG_DIR/corpus_v1_sec_context_w60_kw.jsonl"

# ============================================================
# STEP 1 - build augmented corpora
# ============================================================
echo ""
echo "=== Step 1: build context-augmented corpora ==="
python scripts/build_context_aug_corpus.py \
    --summary-graph "$SUM_GRAPH" \
    --chunk-graph "$CHUNK_GRAPH" \
    --elements-graph "$ELEM_GRAPH" \
    --in-corpus "$IN_CORPUS" \
    --out-corpus "$OUT_CORPUS_A" \
    --max-summary-words 60

python scripts/build_context_aug_corpus.py \
    --summary-graph "$SUM_GRAPH" \
    --chunk-graph "$CHUNK_GRAPH" \
    --elements-graph "$ELEM_GRAPH" \
    --in-corpus "$IN_CORPUS" \
    --out-corpus "$OUT_CORPUS_B" \
    --max-summary-words 60 \
    --include-keywords
echo "  Step 1 done: $(date)"

# ============================================================
# STEP 2 - Dense retrieval eval (Qwen3-Embedding-0.6B)
# ============================================================
echo ""
echo "=== Step 2: dense retrieval eval (0.6B) ==="

run_eval () {
    local TAG="$1"
    local CORPUS="$2"
    local OUT_REPORT="$AUG_DIR/eval_${TAG}_qwen06b.json"
    local OUT_RANKING="$AUG_DIR/ranking_${TAG}_qwen06b.jsonl"
    echo ""
    echo "  ── Eval [$TAG] ──"
    python scripts/eval_dense_retrieval.py \
        --data-dir "$AUG_DIR" \
        --queries "$QUERIES" \
        --corpus "$CORPUS" \
        --qrels "$QRELS" \
        --model-name "$MODEL_NAME" \
        --download-root "$DOWNLOAD_ROOT" \
        --output "$OUT_REPORT" \
        --save-ranking "$OUT_RANKING" \
        --batch-size 16 \
        --max-length 512 \
        --top-k 100
    echo "  ── done [$TAG] $(date)"
}

# Baseline reference (re-run to compare on identical 0.6B encoder if needed)
if [[ ! -s "$AUG_DIR/eval_v1_enriched_qwen06b.json" ]]; then
    run_eval "v1_enriched" "$IN_CORPUS"
fi
run_eval "v1_sec_context_w60"     "$OUT_CORPUS_A"
run_eval "v1_sec_context_w60_kw"  "$OUT_CORPUS_B"

# ============================================================
# STEP 3 - comparison summary
# ============================================================
echo ""
echo "=== Step 3: comparison ==="
python - <<'PY'
import json, glob
from pathlib import Path
aug = Path("data/05_eval/dense_retrieval/augmented_v2")
print()
print(f'{"variant":<35} {"R@1":>8} {"R@5":>8} {"R@10":>8} {"R@100":>8} {"MRR":>8} {"NDCG@10":>10}')
print("-" * 90)
for rp in sorted(glob.glob(str(aug / "eval_*_qwen06b.json"))):
    name = Path(rp).stem.replace("eval_", "").replace("_qwen06b", "")
    r = json.load(open(rp))
    m = r.get("metrics", {})
    print(f'{name:<35} {m.get("recall@1",0):>8.4f} {m.get("recall@5",0):>8.4f} {m.get("recall@10",0):>8.4f} {m.get("recall@100",0):>8.4f} {m.get("mrr",0):>8.4f} {m.get("ndcg@10",0):>10.4f}')
PY

echo ""
echo "============================================"
echo "context_aug_eval COMPLETE: $(date)"
echo "  Corpora: $OUT_CORPUS_A  $OUT_CORPUS_B"
echo "  Reports: $AUG_DIR/eval_*_qwen06b.json"
echo "============================================"
