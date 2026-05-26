#!/bin/bash
#SBATCH --job-name=xdoc_chunk_infer
#SBATCH --partition=cluster02
#SBATCH --qos=msc
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:a6000:1
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --output=logs/79_xdoc_chunk_infer_%j.out
#SBATCH --error=logs/79_xdoc_chunk_infer_%j.err

# ============================================================
# 79_xdoc_chunk_infer.sh
#
# Chunk-level inference: predict cross-document citation edges
# on all MinerU docs using the chunk-trained XGBoost model.
#
# Requires: 78_xdoc_citation_chunk.sh completed first.
# ============================================================

set -euo pipefail

echo "============================================"
echo "Job: xdoc_chunk_infer"
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

if [ ! -s "${OUT_DIR}/xgb_link_predictor.pkl" ]; then
    echo "ERROR: Chunk model not found. Run 78_xdoc_citation_chunk.sh first."
    exit 1
fi

echo ""
echo "=== Chunk-level inference on all docs ==="

python -u scripts/infer_xdoc_citation_chunks.py \
    --model-path "${OUT_DIR}/xgb_link_predictor.pkl" \
    --model-info "${OUT_DIR}/model_info.json" \
    --output-dir "${OUT_DIR}" \
    --embedding-model "all-MiniLM-L6-v2" \
    --batch-size 64 \
    --top-k-candidates 15 \
    --max-edges-per-doc 200

echo ""
echo "=== Done: $(date) ==="

# Summary
python3 -u - <<'PYEOF'
import json
import numpy as np
from pathlib import Path
from collections import Counter

out_dir = Path("data/04_xdoc_citation")
pred_path = out_dir / "predicted_xdoc_edges_chunks.jsonl"

if not pred_path.exists():
    print("No predictions found!")
    exit(1)

edges = [json.loads(l) for l in open(pred_path)]
print(f"\nTotal predicted edges: {len(edges)}")

src_docs = Counter(e['source_doc'] for e in edges)
tgt_docs = Counter(e['target_doc'] for e in edges)
probs = [e['probability'] for e in edges]

print(f"Unique source docs: {len(src_docs)}")
print(f"Unique target docs: {len(tgt_docs)}")
print(f"Probability: min={min(probs):.4f} mean={np.mean(probs):.4f} max={max(probs):.4f}")
print(f"Percentiles: P10={np.percentile(probs,10):.4f} P50={np.percentile(probs,50):.4f} P90={np.percentile(probs,90):.4f}")

# Top source/target
print("\nTop source docs:")
for doc, n in src_docs.most_common(10):
    print(f"  {doc}: {n} edges")

print("\nTop target docs (most cited):")
for doc, n in tgt_docs.most_common(10):
    print(f"  {doc}: {n} citations")

# Section distribution
sections = Counter(e.get('section_title', '?')[:30] for e in edges)
print("\nTop sections:")
for s, n in sections.most_common(10):
    print(f"  {s}: {n}")

# Feature distribution
cite_patterns = [e['features']['cite_pattern'] for e in edges]
title_matches = [e['features']['title_match'] for e in edges]
text_sims = [e['features']['text_sim'] for e in edges]
print(f"\nCite pattern: mean={np.mean(cite_patterns):.4f}")
print(f"Title match: mean={np.mean(title_matches):.4f}")
print(f"Text sim: mean={np.mean(text_sims):.4f}")

# Show a few high-confidence examples
print("\n=== High-confidence examples ===")
edges_sorted = sorted(edges, key=lambda x: -x['probability'])[:10]
for e in edges_sorted:
    print(f"\n  {e['source_doc']} → {e['target_doc']} (p={e['probability']:.4f})")
    print(f"  Section: {e['section_title']}")
    print(f"  Chunk: {e['chunk_text'][:120]}...")
    print(f"  Features: {e['features']}")
PYEOF

echo "Done: $(date)"
