#!/bin/bash
#SBATCH --job-name=chunk_corpus_eval
#SBATCH --partition=cluster02
#SBATCH --qos=msc
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:a6000:1
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=logs/chunk_corpus_eval_%j.out
#SBATCH --error=logs/chunk_corpus_eval_%j.err

# ============================================================
# 37_chunk_corpus_eval.sh
#
# 验证 4.19.md §5.3 重构方向（fair trial 版）：
#   chunk 作为检索单元，先使用 full graph elements 构造公平 chunk text，
#   不依赖 selective enrich 覆盖率；
#   qrels 重映射 element_id → parent_chunk_id
#
# 对照组：
#   A. v1_enriched baseline（element 为检索单元，原始 qrels）
#   B. chunk_n400_fair（新 chunk corpus，full graph text，qrels 重映射）
#   C. chunk_n500（同上，500词 chunk）
#
# 注意：
#   - 这是 57 gold docs 的 old trial / closed-eval lane
#   - 不要接入 1040-doc production partial enrich 文件
#   - enriched 版本需等 57-doc full backfill 后再单独开一条 lane
#
# 输出目录：data/05_eval/chunk_corpus_n400_fair/  chunk_corpus_n500_fair/
# ============================================================

set -euo pipefail

echo "============================================"
echo "Job: chunk_corpus_eval"
echo "Job ID:  ${SLURM_JOB_ID}"
echo "Node:    $(hostname)"
echo "Start:   $(date)"
echo "============================================"

cd /projects/myyyx1/data-process-test
source /cluster/apps/software/Miniforge3/24.11.3-1/etc/profile.d/conda.sh
conda activate /projects/myyyx1/envs/minerU

TRIAL_M4_DIR="data/03_queries/M4query_v1"
TRIAL_GRAPH_ELEMENTS="$TRIAL_M4_DIR/graphs/multimodal_elements.json"
TRIAL_ENRICHED_OVERLAY="data/02_enriched/multimodal_elements_enriched.json"
MODEL_NAME="Qwen/Qwen3-Embedding-0.6B"
DOWNLOAD_ROOT="models"

mkdir -p logs data/05_eval/chunk_corpus_n400_fair data/05_eval/chunk_corpus_n500_fair

# ============================================================
# STEP 1 — 构建 n400 chunk corpus（已在 CPU 上验证过，可直接运行）
# ============================================================
echo ""
echo "=== Step 1a: Build chunk corpus n400 ==="
python scripts/build_chunk_corpus.py \
    --chunks data/01_graphs/paragraph_chunks_n400.json \
    --graph-elements "$TRIAL_GRAPH_ELEMENTS" \
    --enriched "$TRIAL_ENRICHED_OVERLAY" \
    --element-text-mode graph_only \
    --qrels "$TRIAL_M4_DIR/qrels.jsonl" \
    --out-dir data/05_eval/chunk_corpus_n400_fair

echo "=== Step 1b: Build chunk corpus n500 ==="
python scripts/build_chunk_corpus.py \
    --chunks data/01_graphs/paragraph_chunks_n500.json \
    --graph-elements "$TRIAL_GRAPH_ELEMENTS" \
    --enriched "$TRIAL_ENRICHED_OVERLAY" \
    --element-text-mode graph_only \
    --qrels "$TRIAL_M4_DIR/qrels.jsonl" \
    --out-dir data/05_eval/chunk_corpus_n500_fair

echo "  Step 1 done: $(date)"

# ============================================================
# STEP 2 — Dense retrieval eval（两组新实验，baseline 已有）
# v1_enriched baseline 已在 data/05_eval/dense_retrieval/stacking_06b/explicit_only/
# ============================================================
echo ""
echo "=== Step 2: Dense retrieval eval ==="

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
        --corpus  "$CORPUS" \
        --qrels   "$QRELS" \
        --model-name "$MODEL_NAME" \
        --download-root "$DOWNLOAD_ROOT" \
        --output "$OUT_DIR/eval_report.json" \
        --save-ranking "$OUT_DIR/ranking.jsonl" \
        --batch-size 16 \
        --max-length 1024 \
        --top-k 100
    echo "  ── done [$TAG]: $(date)"
}

# B: chunk n400（chunk 为单元，element 内容注入，qrels 重映射）
run_eval "chunk_n400" \
    "data/05_eval/chunk_corpus_n400_fair/corpus.jsonl" \
    "data/05_eval/chunk_corpus_n400_fair/qrels.jsonl" \
    "data/05_eval/chunk_corpus_n400_fair"

# C: chunk n500
run_eval "chunk_n500" \
    "data/05_eval/chunk_corpus_n500_fair/corpus.jsonl" \
    "data/05_eval/chunk_corpus_n500_fair/qrels.jsonl" \
    "data/05_eval/chunk_corpus_n500_fair"

# ============================================================
# STEP 3 — 汇总结果
# ============================================================
echo ""
echo "=== Step 3: Summary ==="
python3 - <<'PYEOF'
import json
from pathlib import Path

# v1_enriched 0.6B baseline 已有结果（stacking_06b 实验中的 metrics_baseline.json）
configs = [
    ("v1_enriched 0.6B (ref)",  "data/05_eval/dense_retrieval/stacking_06b/explicit_only/metrics_baseline.json"),
    ("chunk_n400_fair",         "data/05_eval/chunk_corpus_n400_fair/eval_report.json"),
    ("chunk_n500_fair",         "data/05_eval/chunk_corpus_n500_fair/eval_report.json"),
]

print()
print(f"{'Variant':<28} {'R@1':>8} {'R@5':>8} {'R@10':>8} {'R@100':>8} {'MRR':>8} {'NDCG@10':>10}")
print("-" * 82)
for name, rp in configs:
    p = Path(rp)
    if not p.exists():
        print(f"{name:<28}  [missing]")
        continue
    r = json.load(open(p))
    m = r.get("metrics", {})
    print(f"{name:<28} {m.get('recall@1',0):>8.4f} {m.get('recall@5',0):>8.4f} "
          f"{m.get('recall@10',0):>8.4f} {m.get('recall@100',0):>8.4f} "
          f"{m.get('mrr',0):>8.4f} {m.get('ndcg@10',0):>10.4f}")
print()
PYEOF

echo ""
echo "============================================"
echo "chunk_corpus_eval COMPLETE: $(date)"
echo "============================================"
