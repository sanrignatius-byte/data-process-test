#!/bin/bash
#SBATCH --job-name=xdoc_chunk
#SBATCH --partition=cluster02
#SBATCH --qos=msc
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:a6000:1
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=logs/78_xdoc_chunk_%j.out
#SBATCH --error=logs/78_xdoc_chunk_%j.err

# ============================================================
# 78_xdoc_citation_chunk.sh
#
# Chunk-level cross-document citation link predictor.
# Uses pre-built chunks (chunk_virtual_nodes_v2.json) instead of
# raw markdown passages. Chunks are uniform, fewer, and have
# section context + element ID mappings.
# ============================================================

set -euo pipefail

echo "============================================"
echo "Job: xdoc_citation_chunk"
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

# ── Step 1: Extract chunk-level GT pairs ──
echo ""
echo "=== Step 1: Extract chunk-level GT citation pairs ==="
python -u scripts/extract_xdoc_citation_chunks.py --output-dir "$OUT_DIR"

# ── Step 2: Compute chunk features with embeddings (GPU) ──
echo ""
echo "=== Step 2: Compute chunk features (GPU) ==="
python -u scripts/compute_xdoc_chunk_features.py \
    --input-file "${OUT_DIR}/gt_citation_chunks.jsonl" \
    --output-dir "$OUT_DIR" \
    --embedding-model "all-MiniLM-L6-v2" \
    --neg-ratio 3 \
    --batch-size 64

# ── Step 3: Train XGBoost on chunk features ──
echo ""
echo "=== Step 3: Train XGBoost (chunk level) ==="
python -u scripts/train_xdoc_link_predictor.py \
    --feature-file "${OUT_DIR}/features_chunk_train.npz" \
    --metadata-file "${OUT_DIR}/feature_chunk_metadata.json" \
    --pair-file "${OUT_DIR}/pair_chunk_records.jsonl" \
    --output-dir "${OUT_DIR}" \
    --n-folds 5

echo ""
echo "=== Chunk-level training complete: $(date) ==="

# Compare with passage-level results
python3 -u - <<'PYEOF'
import json
from pathlib import Path
out_dir = Path("data/04_xdoc_citation")

# Chunk-level
chunk_report = out_dir / "training_report.json"
if chunk_report.exists():
    with open(chunk_report) as f:
        r = json.load(f)
    avg = r.get("cv_results", {}).get("average", {})
    print("\n=== CHUNK-LEVEL RESULTS ===")
    for k, v in avg.items():
        if not k.endswith("_std"):
            print(f"  {k}: {v}")
    print(f"  Optimal threshold: {r.get('optimal_threshold', 'N/A')}")
    print(f"  Optimal F1: {r.get('optimal_f1', 'N/A')}")
    print("\nFeature importance (top 5):")
    for name, imp in sorted(r.get("feature_importance", {}).items(), key=lambda x: -x[1])[:5]:
        print(f"  {name}: {imp:.4f}")

# Passage-level for comparison
import os
passage_report_path = out_dir / "training_report.json"
# The passage model was saved to model_info.json (different from training_report which was overwritten by this run)
# Let's load the passage model info
passage_model_info = out_dir / "model_info.json"
if passage_model_info.exists():
    with open(passage_model_info) as f:
        pi = json.load(f)
    pavg = pi.get("cv_results", {}).get("average", {})
    print("\n=== PASSAGE-LEVEL (previous) ===")
    for k, v in pavg.items():
        if not k.endswith("_std"):
            print(f"  {k}: {v}")
    print(f"  Optimal threshold: {pi.get('optimal_threshold', 'N/A')}")
    print(f"  Optimal F1: {pi.get('optimal_f1', 'N/A')}")
PYEOF

echo "Done: $(date)"
