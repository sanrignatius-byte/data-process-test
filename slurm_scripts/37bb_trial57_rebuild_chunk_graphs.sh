#!/bin/bash
#SBATCH --job-name=trial57_chunk_rebuild
#SBATCH --output=logs/37bb_trial57_chunk_rebuild_%j.out
#SBATCH --error=logs/37bb_trial57_chunk_rebuild_%j.err
#SBATCH --partition=cluster02
#SBATCH --qos=msc
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=02:00:00

# ============================================================
# 37bb_trial57_rebuild_chunk_graphs.sh
#
# 目标：
#   在 trial57 backfill enrich 完成后，显式重做 chunk、重构边、重建 chunk graph。
#
# 说明：
#   - 仍然只针对 old-trial / 57 gold docs lane
#   - 输出新的 n400 / n500 chunk graph 文件，供后续 passage 构造与 graph rerank 使用
#   - elements 输入使用 trial57 full enriched overlay，确保 enriched wave 的 chunk graph
#     与旧 partial overlay / production lane 分离
# ============================================================

set -euo pipefail

echo "============================================"
echo "Job: trial57_rebuild_chunk_graphs"
echo "Job ID:  ${SLURM_JOB_ID}"
echo "Node:    $(hostname)"
echo "Start:   $(date)"
echo "============================================"

cd /projects/myyyx1/data-process-test
source /cluster/apps/software/Miniforge3/24.11.3-1/etc/profile.d/conda.sh
conda activate /projects/myyyx1/envs/minerU

TRIAL_FULL_OVERLAY="data/02_enriched/multimodal_elements_trial57_enriched_full.json"
OUT_N400="data/01_graphs/paragraph_chunks_n400_trial57_enriched.json"
OUT_N500="data/01_graphs/paragraph_chunks_n500_trial57_enriched.json"

mkdir -p logs data/01_graphs

echo ""
echo "=== Step 1a: Rebuild trial57 n400 chunk graph ==="
python scripts/build_paragraph_chunks.py \
    --chunk-size 400 \
    --elements "$TRIAL_FULL_OVERLAY" \
    --output "$OUT_N400"

echo ""
echo "=== Step 1b: Rebuild trial57 n500 chunk graph ==="
python scripts/build_paragraph_chunks.py \
    --chunk-size 500 \
    --elements "$TRIAL_FULL_OVERLAY" \
    --output "$OUT_N500"

echo ""
echo "=== Step 2: Summary ==="
python3 - <<'PYEOF'
import json
from pathlib import Path

for label, path in [
    ("n400", Path("data/01_graphs/paragraph_chunks_n400_trial57_enriched.json")),
    ("n500", Path("data/01_graphs/paragraph_chunks_n500_trial57_enriched.json")),
]:
    if not path.exists():
        print(f"{label}: [missing]")
        continue
    data = json.load(open(path))
    stats = data.get("stats", {})
    print(
        f"{label}: docs={stats.get('docs_with_chunks', 0)} "
        f"chunks={stats.get('total_chunks', 0)} "
        f"elements_assigned={stats.get('total_elements_assigned', 0)} "
        f"edges={stats.get('total_edges', 0)} "
        f"avg_words={stats.get('avg_words_per_chunk', 0)}"
    )
PYEOF

echo ""
echo "============================================"
echo "trial57_rebuild_chunk_graphs COMPLETE: $(date)"
echo "Outputs: $OUT_N400 / $OUT_N500"
echo "============================================"
