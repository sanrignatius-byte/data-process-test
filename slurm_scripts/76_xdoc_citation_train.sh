#!/bin/bash
#SBATCH --job-name=xdoc_train
#SBATCH --partition=cluster02
#SBATCH --qos=msc
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:a6000:1
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=logs/76_xdoc_train_%j.out
#SBATCH --error=logs/76_xdoc_train_%j.err

# ============================================================
# 76_xdoc_citation_train.sh
#
# Phase 1+2: Cross-document citation link predictor training.
#
# Step 1: Extract GT citation pairs from LaTeX → align to MinerU
# Step 2: Compute features (incl. Qwen3-Embedding-4B text embeddings)
# Step 3: Train XGBoost classifier + evaluate
# ============================================================

set -euo pipefail

echo "============================================"
echo "Job: xdoc_citation_train (Phase 1+2)"
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

# ── Step 1: Extract GT pairs (if not already done) ──
if [ ! -s "${OUT_DIR}/gt_citation_pairs.jsonl" ]; then
    echo ""
    echo "=== Step 1: Extract GT citation pairs ==="
    python -u scripts/extract_xdoc_citation_pairs.py \
        --output-dir "$OUT_DIR"
else
    echo "Step 1: GT pairs already exist ($(wc -l < ${OUT_DIR}/gt_citation_pairs.jsonl) lines)"
fi

# ── Step 2: Compute features with embeddings (GPU) ──
echo ""
echo "=== Step 2: Compute passage features (GPU) ==="
python -u scripts/compute_xdoc_passage_features.py \
    --input-file "${OUT_DIR}/gt_citation_pairs.jsonl" \
    --output-dir "$OUT_DIR" \
    --embedding-model "$EMBEDDING_MODEL" \
    --neg-ratio 3 \
    --batch-size 32

# ── Step 3: Train XGBoost ──
echo ""
echo "=== Step 3: Train XGBoost link predictor ==="
python -u scripts/train_xdoc_link_predictor.py \
    --feature-file "${OUT_DIR}/features_train.npz" \
    --metadata-file "${OUT_DIR}/feature_metadata.json" \
    --pair-file "${OUT_DIR}/pair_records.jsonl" \
    --output-dir "${OUT_DIR}" \
    --n-folds 5

echo ""
echo "=== Training complete: $(date) ==="

# Quick summary
python3 -u - <<'PYEOF'
import json
from pathlib import Path
out_dir = Path("data/04_xdoc_citation")
report_path = out_dir / "training_report.json"
if report_path.exists():
    with open(report_path) as f:
        r = json.load(f)
    avg = r.get("cv_results", {}).get("average", {})
    print("\nCV Results (avg across folds):")
    for k, v in avg.items():
        if not k.endswith("_std"):
            print(f"  {k}: {v}")
    print(f"\nOptimal threshold: {r.get('optimal_threshold', 'N/A')}")
    print(f"Optimal F1: {r.get('optimal_f1', 'N/A')}")
    print("\nFeature importance:")
    for name, imp in sorted(r.get("feature_importance", {}).items(), key=lambda x: -x[1]):
        print(f"  {name}: {imp:.4f}")
PYEOF

echo "Done: $(date)"
