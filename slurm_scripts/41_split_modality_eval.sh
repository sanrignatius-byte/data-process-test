#!/bin/bash
#SBATCH --job-name=split_mod_eval
#SBATCH --partition=cluster02
#SBATCH --qos=msc
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:a6000:1
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=logs/41_split_modality_%j.out
#SBATCH --error=logs/41_split_modality_%j.err

# ============================================================
# 41_split_modality_eval.sh
#
# 目标：
#   分离式检索实验 — 文本 query 只查文本 chunk，
#   figure/table/formula query 只查对应模态的 corpus。
#
# 对照：
#   - 0.6B baseline（all corpus, unified retrieval）
#   - 4B baseline
#   - 0.6B split
#   - 4B split
# ============================================================

set -euo pipefail

echo "============================================"
echo "Job: split_modality_eval"
echo "Job ID:  ${SLURM_JOB_ID}"
echo "Node:    $(hostname)"
echo "Start:   $(date)"
echo "============================================"

cd /projects/myyyx1/data-process-test
source /cluster/apps/software/Miniforge3/24.11.3-1/etc/profile.d/conda.sh
conda activate /projects/myyyx1/envs/minerU

M4_DIR="data/03_queries/M4query_v1"
OUT_BASE="data/05_eval/dense_retrieval/split_modality"

mkdir -p "$OUT_BASE" "$OUT_BASE/0.6B" "$OUT_BASE/4B" logs

# ── 0.6B split ──
echo ""
echo "=== Step 1: 0.6B split modality ==="
python scripts/eval_split_modality_retrieval.py \
    --data-dir "$M4_DIR" \
    --model-path models/Qwen/Qwen3-Embedding-0.6B \
    --output-dir "$OUT_BASE/0.6B" \
    --batch-size 8 \
    --max-length 512 \
    --top-k 100

# ── 4B split ──
echo ""
echo "=== Step 2: 4B split modality ==="
python scripts/eval_split_modality_retrieval.py \
    --data-dir "$M4_DIR" \
    --model-path models/Qwen3-Embedding-4B \
    --output-dir "$OUT_BASE/4B" \
    --batch-size 4 \
    --max-length 512 \
    --top-k 100

# ── Summary ──
echo ""
echo "=== Step 3: Summary ==="
python3 - <<'PYEOF'
import json
from pathlib import Path

# Existing baselines
baselines = {
    "v1_enriched_0.6B": "data/05_eval/dense_retrieval/eval_report_v1_enriched.json",
    "v2_chunks_0.6B":   "data/05_eval/dense_retrieval/eval_report_v2_chunks.json",
    "v1_enriched_4B":   "data/05_eval/dense_retrieval/rebuilt_20260417/eval_report_v1_enriched.json",
    "v2_chunks_4B":     "data/05_eval/dense_retrieval/rebuilt_20260417/eval_report_v2_chunks.json",
}

# Split modality results
splits = {
    "split_0.6B": "data/05_eval/dense_retrieval/split_modality/0.6B/eval_report.json",
    "split_4B":   "data/05_eval/dense_retrieval/split_modality/4B/eval_report.json",
}

print()
print(f"{'Config':<25} {'R@1':>8} {'R@5':>8} {'R@10':>8} {'R@100':>8} {'MRR':>8}")
print("-" * 75)

for name, rp in {**baselines, **splits}.items():
    p = Path(rp)
    if not p.exists():
        print(f"{name:<25}  [missing]")
        continue
    data = json.load(open(p))
    metrics = data.get("metrics", data)
    print(f"{name:<25} {metrics.get('recall@1',0):>8.4f} {metrics.get('recall@5',0):>8.4f} "
          f"{metrics.get('recall@10',0):>8.4f} {metrics.get('recall@100',0):>8.4f} "
          f"{metrics.get('mrr',0):>8.4f}")
print()
PYEOF

echo ""
echo "============================================"
echo "split_modality_eval COMPLETE: $(date)"
echo "============================================"
