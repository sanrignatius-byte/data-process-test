#!/bin/bash
#SBATCH --job-name=xdoc_infer
#SBATCH --partition=cluster02
#SBATCH --qos=msc
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:a6000:1
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --output=logs/77_xdoc_infer_%j.out
#SBATCH --error=logs/77_xdoc_infer_%j.err

# ============================================================
# 77_xdoc_citation_infer.sh
#
# Phase 3: Inference — run cross-document citation edge prediction
# on ALL 1,040 MinerU docs using the trained XGBoost model.
#
# Requires: 76_xdoc_citation_train.sh completed first
# ============================================================

set -euo pipefail

echo "============================================"
echo "Job: xdoc_citation_infer (Phase 3)"
echo "Job ID:  ${SLURM_JOB_ID}"
echo "Node:    $(hostname)"
echo "Start:   $(date)"
echo "GPU:     $(nvidia-smi --query-gpu=gpu_name --format=csv,noheader 2>/dev/null || echo N/A)"
echo "============================================"

cd /projects/myyyx1/data-process-test
source /cluster/apps/software/Miniforge3/24.11.3-1/etc/profile.d/conda.sh
conda activate /projects/myyyx1/envs/minerU

export PYTHONUNBUFFERED=1
mkdir -p logs

OUT_DIR="data/04_xdoc_citation"
# Use fallback chain: local Qwen3-Embedding-4B > all-MiniLM-L6-v2 (auto-download)
EMBEDDING_MODEL="all-MiniLM-L6-v2"

# ── Sanity checks ──
if [ ! -s "${OUT_DIR}/xgb_link_predictor.pkl" ]; then
    echo "ERROR: Model not found at ${OUT_DIR}/xgb_link_predictor.pkl"
    echo "Run 76_xdoc_citation_train.sh first."
    exit 1
fi

if [ ! -s "${OUT_DIR}/model_info.json" ]; then
    echo "ERROR: Model info not found at ${OUT_DIR}/model_info.json"
    exit 1
fi

echo ""
echo "=== Phase 3: Inference on all MinerU docs ==="

python -u scripts/infer_xdoc_citation_edges.py \
    --model-path "${OUT_DIR}/xgb_link_predictor.pkl" \
    --model-info "${OUT_DIR}/model_info.json" \
    --output-dir "${OUT_DIR}" \
    --embedding-model "$EMBEDDING_MODEL" \
    --batch-size 32 \
    --top-k-candidates 20

echo ""
echo "=== Inference complete: $(date) ==="

python3 -u - <<'PYEOF'
import json
from pathlib import Path
out_dir = Path("data/04_xdoc_citation")

pred_path = out_dir / "predicted_xdoc_edges.jsonl"
stats_path = out_dir / "inference_stats.json"

if pred_path.exists():
    n_lines = sum(1 for _ in open(pred_path))
    print(f"\nPredicted edges: {n_lines}")

    # Quick stats
    from collections import Counter
    src_docs = set()
    tgt_docs = set()
    probs = []
    with open(pred_path) as f:
        for line in f:
            e = json.loads(line)
            src_docs.add(e['source_doc'])
            tgt_docs.add(e['target_doc'])
            probs.append(e['probability'])

    print(f"Unique source docs: {len(src_docs)}")
    print(f"Unique target docs: {len(tgt_docs)}")
    print(f"Probability range: [{min(probs):.4f}, {max(probs):.4f}]")

    if probs:
        import numpy as np
        print(f"Probability mean: {np.mean(probs):.4f}")
        print(f"Probability percentiles: P10={np.percentile(probs,10):.4f} P50={np.percentile(probs,50):.4f} P90={np.percentile(probs,90):.4f}")
else:
    print("No predictions found!")

if stats_path.exists():
    with open(stats_path) as f:
        stats = json.load(f)
    print(f"\nInference stats: {json.dumps(stats, indent=2)}")
PYEOF

echo "Done: $(date)"
