#!/bin/bash
#SBATCH --job-name=split_vl_t5
#SBATCH --partition=cluster02
#SBATCH --qos=msc
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:a6000:1
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=logs/41b_split_modality_vl_t5_%j.out
#SBATCH --error=logs/41b_split_modality_vl_t5_%j.err

# ============================================================
# 41b_split_modality_vl_t5.sh
#
# 复跑 Job 66114 split_modality_vl，唯一变化是 transformers 版本：
#   minerU (transformers 4.57) + overlay (transformers ≥ 5.2)
#
# 根因：Qwen3-VL-Embedding-2B 的 safetensors 用 `model.language_model.*`
# 命名空间，transformers 4.x 的 Qwen3VLModel 期待 `language_model.*`，
# 28 层 decoder 全部"newly initialized" → query 编码用随机权重 → R@10=0.0021。
# Qwen 官方 PR #19 确认模型需要 transformers ≥ 5.2。
#
# 一切其他参数（M4query_v1 数据、merge configs、batch_size、max_length）
# 与 66114 完全一致，方便直接对比 R@10 是否从 0.0021 跳回正常区间。
# ============================================================

set -euo pipefail

echo "============================================"
echo "Job: split_modality_vl_t5"
echo "Job ID:  ${SLURM_JOB_ID}"
echo "Node:    $(hostname)"
echo "Start:   $(date)"
echo "GPU:     $(nvidia-smi --query-gpu=gpu_name --format=csv,noheader 2>/dev/null || echo N/A)"
echo "============================================"

cd /projects/myyyx1/data-process-test
source /cluster/apps/software/Miniforge3/24.11.3-1/etc/profile.d/conda.sh
conda activate /projects/myyyx1/envs/minerU

# Keep the production minerU env untouched. This overlay only shadows
# transformers/tokenizers/huggingface-hub for this job.
export QWEN3VL_TF5_OVERLAY="/projects/myyyx1/envs/qwen3vl_tf5_overlay"
export PYTHONPATH="${QWEN3VL_TF5_OVERLAY}${PYTHONPATH:+:${PYTHONPATH}}"
export PYTHONUNBUFFERED=1

# Sanity: confirm transformers >= 5.2 and language_model loads with weights
python3 -u - <<'PYEOF'
import transformers, sentence_transformers
from mistral_common.protocol.instruct.request import ReasoningEffort
print(f"transformers={transformers.__version__}")
print(f"sentence_transformers={sentence_transformers.__version__}")
print(f"transformers_path={transformers.__file__}")
print(f"mistral_common_ReasoningEffort={ReasoningEffort.__name__}")
v = tuple(int(x) for x in transformers.__version__.split(".")[:2])
assert v >= (5, 2), f"need transformers >= 5.2, got {transformers.__version__}"
PYEOF

M4_DIR="data/03_queries/M4query_v1"
VL_MODEL="/projects/myyyx1/models/Qwen3-VL-Embedding-2B"
OUT_DIR="data/05_eval/dense_retrieval/split_modality_vl_t5"

mkdir -p "$OUT_DIR" logs

echo ""
echo "=== Split Modality VL (Qwen3-VL-Embedding-2B, transformers≥5.2) ==="
python scripts/eval_split_modality_vl.py \
    --data-dir "$M4_DIR" \
    --vl-model "$VL_MODEL" \
    --output-dir "$OUT_DIR" \
    --batch-size 16 \
    --text-max-length 512 \
    --image-max-length 4096 \
    --top-k 100

echo ""
echo "=== Summary vs Baselines ==="
python3 - <<'PYEOF'
import json
from pathlib import Path

baselines = {
    "v1_enriched_4B":         "data/05_eval/dense_retrieval/rebuilt_20260417/eval_report_v1_enriched.json",
    "split_4B_text":          "data/05_eval/dense_retrieval/split_modality/4B/eval_report.json",
    "split_VL_2B_t4_failed":  "data/05_eval/dense_retrieval/split_modality_vl/eval_report.json",
    "split_VL_2B_t5":         "data/05_eval/dense_retrieval/split_modality_vl_t5/eval_report.json",
}

print(f"\n{'Config':<28s} {'R@1':>8s} {'R@5':>8s} {'R@10':>8s} {'R@100':>8s} {'MRR':>8s}")
print("-" * 78)

for name, rp in baselines.items():
    p = Path(rp)
    if not p.exists():
        print(f"{name:<28s}  [missing]")
        continue
    data = json.load(open(p))
    m = data.get("baseline", data.get("metrics", data))
    if "baseline" in data and isinstance(data["baseline"], dict):
        m = data["baseline"]
    print(f"{name:<28s} {m.get('recall@1',0):>8.4f} {m.get('recall@5',0):>8.4f} "
          f"{m.get('recall@10',0):>8.4f} {m.get('recall@100',0):>8.4f} "
          f"{m.get('mrr',0):>8.4f}")
print()
PYEOF

echo ""
echo "============================================"
echo "split_modality_vl_t5 COMPLETE: $(date)"
echo "============================================"
