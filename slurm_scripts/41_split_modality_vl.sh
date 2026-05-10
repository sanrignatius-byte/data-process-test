#!/bin/bash
#SBATCH --job-name=split_vl
#SBATCH --partition=cluster02
#SBATCH --qos=msc
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:a6000:1
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=logs/41_split_modality_vl_%j.out
#SBATCH --error=logs/41_split_modality_vl_%j.err

# ============================================================
# 41_split_modality_vl.sh
#
# 分离式检索 + 真正多模态 embedding。
#
# 核心区别 vs 纯文本版：
#   - Figure/Table: Qwen3-VL-Embedding-2B 直接编码实际图像
#   - Formula/Text: 文本编码（LaTeX / 自然语言）
#   - Query: 文本编码，通过 VL 模型跨模态对齐匹配图像
#
# 所有 embedding 在统一 2048-dim 空间，真正实现
# "文本 query 找多模态 corpus"。
# ============================================================

set -euo pipefail

echo "============================================"
echo "Job: split_modality_vl"
echo "Job ID:  ${SLURM_JOB_ID}"
echo "Node:    $(hostname)"
echo "Start:   $(date)"
echo "GPU:     $(nvidia-smi --query-gpu=gpu_name --format=csv,noheader 2>/dev/null || echo N/A)"
echo "============================================"

cd /projects/myyyx1/data-process-test
source /cluster/apps/software/Miniforge3/24.11.3-1/etc/profile.d/conda.sh
conda activate /projects/myyyx1/envs/minerU

M4_DIR="data/03_queries/M4query_v1"
VL_MODEL="/projects/myyyx1/models/Qwen3-VL-Embedding-2B"
OUT_DIR="data/05_eval/dense_retrieval/split_modality_vl"

mkdir -p "$OUT_DIR" logs

echo ""
echo "=== Split Modality VL (Qwen3-VL-Embedding-2B) ==="
python scripts/eval_split_modality_vl.py \
    --data-dir "$M4_DIR" \
    --vl-model "$VL_MODEL" \
    --output-dir "$OUT_DIR" \
    --batch-size 16 \
    --text-max-length 512 \
    --image-max-length 4096 \
    --top-k 100

# ── Summary vs baselines ──
echo ""
echo "=== Summary vs Baselines ==="
python3 - <<'PYEOF'
import json
from pathlib import Path

baselines = {
    "v1_enriched_4B":   "data/05_eval/dense_retrieval/rebuilt_20260417/eval_report_v1_enriched.json",
    "split_4B_text":    "data/05_eval/dense_retrieval/split_modality/4B/eval_report.json",
    "split_VL_2B":      "data/05_eval/dense_retrieval/split_modality_vl/eval_report.json",
}

print(f"\n{'Config':<25s} {'R@1':>8s} {'R@5':>8s} {'R@10':>8s} {'R@100':>8s} {'MRR':>8s}")
print("-" * 75)

for name, rp in baselines.items():
    p = Path(rp)
    if not p.exists():
        print(f"{name:<25s}  [missing]")
        continue
    data = json.load(open(p))
    m = data.get("baseline", data.get("metrics", data))
    # For split_4B_text, the report has "baseline" key
    if "baseline" in data and isinstance(data["baseline"], dict):
        m = data["baseline"]
    print(f"{name:<25s} {m.get('recall@1',0):>8.4f} {m.get('recall@5',0):>8.4f} "
          f"{m.get('recall@10',0):>8.4f} {m.get('recall@100',0):>8.4f} "
          f"{m.get('mrr',0):>8.4f}")
print()
PYEOF

echo ""
echo "============================================"
echo "split_modality_vl COMPLETE: $(date)"
echo "============================================"
