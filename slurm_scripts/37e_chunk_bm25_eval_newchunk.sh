#!/bin/bash
#SBATCH --job-name=chunk_bm25_newchunk
#SBATCH --partition=cluster02
#SBATCH --qos=msc
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=logs/37e_chunk_bm25_%j.out
#SBATCH --error=logs/37e_chunk_bm25_%j.err

set -euo pipefail

echo "============================================"
echo "Job: chunk_bm25_eval_newchunk"
echo "Job ID:  ${SLURM_JOB_ID}"
echo "Node:    $(hostname)"
echo "Start:   $(date)"
echo "============================================"

cd /projects/myyyx1/data-process-test
source /cluster/apps/software/Miniforge3/24.11.3-1/etc/profile.d/conda.sh
conda activate /projects/myyyx1/envs/minerU

TRIAL_M4_DIR="data/03_queries/M4query_v1"

run_eval() {
    local TAG="$1"
    local CORPUS="$2"
    local QRELS="$3"
    local OUT_DIR="$4"
    echo ""
    echo "  ── BM25 [$TAG] ──"
    python scripts/eval_bm25_retrieval.py \
        --data-dir "$TRIAL_M4_DIR" \
        --queries "$TRIAL_M4_DIR/queries.jsonl" \
        --corpus "$CORPUS" \
        --qrels "$QRELS" \
        --output "$OUT_DIR/eval_bm25.json" \
        --save-ranking "$OUT_DIR/ranking_bm25.jsonl" \
        --top-k 100
    echo "  ── done [$TAG]: $(date)"
}

run_eval "chunk_n400_fair_bm25" \
    "data/05_eval/chunk_corpus_n400_fair/corpus.jsonl" \
    "data/05_eval/chunk_corpus_n400_fair/qrels.jsonl" \
    "data/05_eval/chunk_corpus_n400_fair"

run_eval "chunk_n500_fair_bm25" \
    "data/05_eval/chunk_corpus_n500_fair/corpus.jsonl" \
    "data/05_eval/chunk_corpus_n500_fair/qrels.jsonl" \
    "data/05_eval/chunk_corpus_n500_fair"

run_eval "chunk_n400_partial_overlay_bm25" \
    "data/05_eval/chunk_corpus_n400_partial_overlay/corpus.jsonl" \
    "data/05_eval/chunk_corpus_n400_partial_overlay/qrels.jsonl" \
    "data/05_eval/chunk_corpus_n400_partial_overlay"

run_eval "chunk_n500_partial_overlay_bm25" \
    "data/05_eval/chunk_corpus_n500_partial_overlay/corpus.jsonl" \
    "data/05_eval/chunk_corpus_n500_partial_overlay/qrels.jsonl" \
    "data/05_eval/chunk_corpus_n500_partial_overlay"

echo ""
echo "=== Step 2: Summary ==="
python3 - <<'PYEOF'
import json
from pathlib import Path

configs = [
    ("chunk_n400_fair_bm25", "data/05_eval/chunk_corpus_n400_fair/eval_bm25.json"),
    ("chunk_n500_fair_bm25", "data/05_eval/chunk_corpus_n500_fair/eval_bm25.json"),
    ("chunk_n400_partial_bm25", "data/05_eval/chunk_corpus_n400_partial_overlay/eval_bm25.json"),
    ("chunk_n500_partial_bm25", "data/05_eval/chunk_corpus_n500_partial_overlay/eval_bm25.json"),
]

print()
print(f"{'Variant':<28} {'R@1':>8} {'R@5':>8} {'R@10':>8} {'R@100':>8} {'MRR':>8}")
print("-" * 82)
for name, rp in configs:
    p = Path(rp)
    if not p.exists():
        print(f"{name:<28}  [missing]")
        continue
    m = json.load(open(p))
    print(f"{name:<28} {m.get('recall@1',0):>8.4f} {m.get('recall@5',0):>8.4f} "
          f"{m.get('recall@10',0):>8.4f} {m.get('recall@100',0):>8.4f} "
          f"{m.get('mrr',0):>8.4f}")
print()
PYEOF

echo ""
echo "============================================"
echo "chunk_bm25_eval_newchunk COMPLETE: $(date)"
echo "============================================"
