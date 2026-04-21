#!/bin/bash
#SBATCH --job-name=chunk_rerank_part
#SBATCH --partition=cluster02
#SBATCH --qos=msc
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=logs/38c_chunk_rerank_partial_%j.out
#SBATCH --error=logs/38c_chunk_rerank_partial_%j.err

# ============================================================
# 38c_chunk_graph_rerank_partial_overlay.sh
#
# 目标：
#   在 partial-overlay exploratory chunk dense ranking 上做 graph rerank。
#
# 重要说明：
#   - 这是 exploratory lane，不是 fair claim
# ============================================================

set -euo pipefail

echo "============================================"
echo "Job: chunk_graph_rerank_partial_overlay"
echo "Job ID:  ${SLURM_JOB_ID}"
echo "Node:    $(hostname)"
echo "Start:   $(date)"
echo "============================================"

cd /projects/myyyx1/data-process-test
source /cluster/apps/software/Miniforge3/24.11.3-1/etc/profile.d/conda.sh
conda activate /projects/myyyx1/envs/minerU

CHUNKS_V2="data/01_graphs/paragraph_chunks_n400_trial57_enriched.json"
TRIAL_HUB_CANDS="data/02_enriched/hub_candidates_enriched_v3.json"
N400_DIR="data/05_eval/chunk_corpus_n400_partial_overlay"
RANKING="$N400_DIR/ranking.jsonl"

if [ ! -f "$CHUNKS_V2" ]; then
    echo "[error] rebuilt chunk graph not found: $CHUNKS_V2"
    exit 1
fi

if [ ! -f "$RANKING" ]; then
    echo "[error] ranking file not found: $RANKING"
    echo "  Run 37d_chunk_corpus_eval_partial_overlay.sh first."
    exit 1
fi

echo ""
echo "=== Step 1: Graph rerank on partial-overlay chunk ranking ==="
python scripts/eval_chunk_graph_rerank.py \
    --ranking "$RANKING" \
    --corpus "$N400_DIR/corpus.jsonl" \
    --qrels "$N400_DIR/qrels.jsonl" \
    --chunk-graph "$CHUNKS_V2" \
    --hub-candidates "$TRIAL_HUB_CANDS" \
    --output-dir "$N400_DIR/graph_rerank" \
    --static-weight 0.15 \
    --neighbor-decay 0.20 \
    --top-k 100

echo ""
echo "=== Step 2: Summary ==="
python3 - <<'PYEOF'
import json
from pathlib import Path

print()
print(f"{'Config':<44} {'R@1':>8} {'R@5':>8} {'R@10':>8} {'MRR':>8}")
print("-" * 78)

configs = [
    ("[ref] v1_enriched 0.6B (element)", "data/05_eval/dense_retrieval/stacking_06b/explicit_only/metrics_baseline.json"),
    ("chunk_n400_fair dense", "data/05_eval/chunk_corpus_n400_fair/eval_report.json"),
    ("chunk_n400_partial_overlay dense", "data/05_eval/chunk_corpus_n400_partial_overlay/eval_report.json"),
]

for name, rp in configs:
    p = Path(rp)
    if not p.exists():
        continue
    data = json.load(open(p))
    metrics = data.get("metrics", data)
    print(f"{name:<44} {metrics.get('recall@1',0):>8.4f} {metrics.get('recall@5',0):>8.4f} {metrics.get('recall@10',0):>8.4f} {metrics.get('mrr',0):>8.4f}")

rp = Path("data/05_eval/chunk_corpus_n400_partial_overlay/graph_rerank/eval_report.json")
if rp.exists():
    data = json.load(open(rp))
    for config_name, res in data.get("results", {}).items():
        metrics = res.get("metrics", {})
        short = config_name[:44]
        print(f"{short:<44} {metrics.get('recall@1',0):>8.4f} {metrics.get('recall@5',0):>8.4f} {metrics.get('recall@10',0):>8.4f} {metrics.get('mrr',0):>8.4f}")
print()
PYEOF

echo ""
echo "============================================"
echo "chunk_graph_rerank_partial_overlay COMPLETE: $(date)"
echo "Results: $N400_DIR/graph_rerank/"
echo "============================================"
