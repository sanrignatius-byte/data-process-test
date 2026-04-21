#!/bin/bash
#SBATCH --job-name=chunk_eval_partial
#SBATCH --partition=cluster02
#SBATCH --qos=msc
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:a6000:1
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=logs/37d_chunk_eval_partial_%j.out
#SBATCH --error=logs/37d_chunk_eval_partial_%j.err

# ============================================================
# 37d_chunk_corpus_eval_partial_overlay.sh
#
# 目标：
#   在 key 不可用、无法补齐 enrich 的情况下，先跑一条 partial-enriched
#   exploratory lane，观察“当前 selective enrich + rebuilt chunk graph”
#   对 retrieval 的影响。
#
# 重要说明：
#   - 这条线不是 fair eval
#   - 不能用于正式 supporting claim
#   - 仅用于探索 partial overlay 是否带来方向性变化
# ============================================================

set -euo pipefail

echo "============================================"
echo "Job: chunk_corpus_eval_partial_overlay"
echo "Job ID:  ${SLURM_JOB_ID}"
echo "Node:    $(hostname)"
echo "Start:   $(date)"
echo "============================================"

cd /projects/myyyx1/data-process-test
source /cluster/apps/software/Miniforge3/24.11.3-1/etc/profile.d/conda.sh
conda activate /projects/myyyx1/envs/minerU

TRIAL_M4_DIR="data/03_queries/M4query_v1"
TRIAL_GRAPH_ELEMENTS="$TRIAL_M4_DIR/graphs/multimodal_elements.json"
PARTIAL_OVERLAY="data/02_enriched/multimodal_elements_trial57_enriched_full.json"
TRIAL_CHUNKS_N400="data/01_graphs/paragraph_chunks_n400_trial57_enriched.json"
TRIAL_CHUNKS_N500="data/01_graphs/paragraph_chunks_n500_trial57_enriched.json"
MODEL_NAME="Qwen/Qwen3-Embedding-0.6B"
DOWNLOAD_ROOT="models"

mkdir -p logs data/05_eval/chunk_corpus_n400_partial_overlay data/05_eval/chunk_corpus_n500_partial_overlay

if [ ! -f "$TRIAL_CHUNKS_N400" ] || [ ! -f "$TRIAL_CHUNKS_N500" ]; then
    echo "[error] rebuilt chunk graphs not found."
    echo "  Expected: $TRIAL_CHUNKS_N400 and $TRIAL_CHUNKS_N500"
    exit 1
fi

echo ""
echo "=== Step 1a: Build partial-overlay exploratory chunk corpus n400 ==="
python scripts/build_chunk_corpus.py \
    --chunks "$TRIAL_CHUNKS_N400" \
    --graph-elements "$TRIAL_GRAPH_ELEMENTS" \
    --enriched "$PARTIAL_OVERLAY" \
    --element-text-mode graph_plus_enriched \
    --qrels "$TRIAL_M4_DIR/qrels.jsonl" \
    --out-dir data/05_eval/chunk_corpus_n400_partial_overlay

echo "=== Step 1b: Build partial-overlay exploratory chunk corpus n500 ==="
python scripts/build_chunk_corpus.py \
    --chunks "$TRIAL_CHUNKS_N500" \
    --graph-elements "$TRIAL_GRAPH_ELEMENTS" \
    --enriched "$PARTIAL_OVERLAY" \
    --element-text-mode graph_plus_enriched \
    --qrels "$TRIAL_M4_DIR/qrels.jsonl" \
    --out-dir data/05_eval/chunk_corpus_n500_partial_overlay

run_eval() {
    local TAG="$1"
    local CORPUS="$2"
    local QRELS="$3"
    local OUT_DIR="$4"
    echo ""
    echo "  ── Eval [$TAG] ──"
    python scripts/eval_dense_retrieval.py \
        --data-dir "$TRIAL_M4_DIR" \
        --queries "$TRIAL_M4_DIR/queries.jsonl" \
        --corpus "$CORPUS" \
        --qrels "$QRELS" \
        --model-name "$MODEL_NAME" \
        --download-root "$DOWNLOAD_ROOT" \
        --output "$OUT_DIR/eval_report.json" \
        --save-ranking "$OUT_DIR/ranking.jsonl" \
        --batch-size 16 \
        --max-length 1024 \
        --top-k 100
    echo "  ── done [$TAG]: $(date)"
}

echo ""
echo "=== Step 2: Dense retrieval eval ==="
run_eval "chunk_n400_partial_overlay" \
    "data/05_eval/chunk_corpus_n400_partial_overlay/corpus.jsonl" \
    "data/05_eval/chunk_corpus_n400_partial_overlay/qrels.jsonl" \
    "data/05_eval/chunk_corpus_n400_partial_overlay"

run_eval "chunk_n500_partial_overlay" \
    "data/05_eval/chunk_corpus_n500_partial_overlay/corpus.jsonl" \
    "data/05_eval/chunk_corpus_n500_partial_overlay/qrels.jsonl" \
    "data/05_eval/chunk_corpus_n500_partial_overlay"

echo ""
echo "=== Step 3: Summary ==="
python3 - <<'PYEOF'
import json
from pathlib import Path

configs = [
    ("v1_enriched 0.6B (ref)", "data/05_eval/dense_retrieval/stacking_06b/explicit_only/metrics_baseline.json"),
    ("chunk_n400_fair", "data/05_eval/chunk_corpus_n400_fair/eval_report.json"),
    ("chunk_n500_fair", "data/05_eval/chunk_corpus_n500_fair/eval_report.json"),
    ("chunk_n400_partial_overlay", "data/05_eval/chunk_corpus_n400_partial_overlay/eval_report.json"),
    ("chunk_n500_partial_overlay", "data/05_eval/chunk_corpus_n500_partial_overlay/eval_report.json"),
]

print()
print(f"{'Variant':<32} {'R@1':>8} {'R@5':>8} {'R@10':>8} {'R@100':>8} {'MRR':>8}")
print("-" * 82)
for name, rp in configs:
    p = Path(rp)
    if not p.exists():
        print(f"{name:<32}  [missing]")
        continue
    data = json.load(open(p))
    metrics = data.get("metrics", data)
    print(f"{name:<32} {metrics.get('recall@1',0):>8.4f} {metrics.get('recall@5',0):>8.4f} "
          f"{metrics.get('recall@10',0):>8.4f} {metrics.get('recall@100',0):>8.4f} "
          f"{metrics.get('mrr',0):>8.4f}")
print()
PYEOF

echo ""
echo "============================================"
echo "chunk_corpus_eval_partial_overlay COMPLETE: $(date)"
echo "============================================"
