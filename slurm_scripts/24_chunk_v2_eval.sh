#!/bin/bash
#SBATCH --job-name=chunk_v2_eval
#SBATCH --partition=cluster02
#SBATCH --qos=msc
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:a6000:1
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --output=logs/chunk_v2_eval_%j.out
#SBATCH --error=logs/chunk_v2_eval_%j.err

# ============================================================
# 4.20 deliverable: 新 chunk 图 + Qwen3-Embedding-0.6B 验证
#
# Step 1: 重建 chunk virtual nodes（nltk 分词，target=425 词）
# Step 2: 重建 augmented corpus v2 (用新 chunks)
# Step 3: 评测 dense retrieval × {baseline / v1_enriched / v2_chunks_new}
#         × Qwen3-Embedding-0.6B
#         报告 R@1, R@5, R@10, R@100, MRR, NDCG@10
# ============================================================

set -euo pipefail

echo "============================================"
echo "Job: chunk_v2_eval (4.20 deliverable)"
echo "Job ID:  ${SLURM_JOB_ID}"
echo "Node:    $(hostname)"
echo "Start:   $(date)"
echo "============================================"

cd /projects/myyyx1/data-process-test
source /cluster/apps/software/Miniforge3/24.11.3-1/etc/profile.d/conda.sh
conda activate /projects/myyyx1/envs/minerU

mkdir -p logs data/01_graphs data/05_eval/dense_retrieval/augmented_v2

# ── Paths ────────────────────────────────────────────────────────────────────
NEW_CHUNKS="data/01_graphs/chunk_virtual_nodes_v2.json"
AUG_DIR="data/05_eval/dense_retrieval/augmented_v2"
M4_DIR="data/03_queries/M4query_v1"
MODEL_NAME="Qwen/Qwen3-Embedding-0.6B"
DOWNLOAD_ROOT="models"

# ── Step 0: ensure nltk punkt available ──────────────────────────────────────
python -c "
import nltk
for pkg in ['punkt', 'punkt_tab']:
    try:
        nltk.data.find(f'tokenizers/{pkg}')
    except LookupError:
        nltk.download(pkg, quiet=True)
print('nltk punkt ready')
"

# ============================================================
# STEP 1 — Rebuild chunks with nltk + tighter target (425 words)
# ============================================================
echo ""
echo "=== Step 1: Rebuild chunk virtual nodes (nltk, target=425) ==="

python scripts/build_chunk_virtual_nodes.py \
    --target-words 425 \
    --min-words 100 \
    --tokenizer auto \
    --output "$NEW_CHUNKS"

echo "  Step 1 done: $(date)"

python -c "
import json
d = json.load(open('$NEW_CHUNKS'))
s = d['stats']
print(f'  total docs:  {s.get(\"docs_with_chunks\")}')
print(f'  total chunks: {s.get(\"total_chunks\")}')
print(f'  total paragraphs: {s.get(\"total_paragraphs\")}')
print(f'  total edges: {s.get(\"total_edges\")}')
print(f'  avg words/chunk: {s.get(\"avg_words_per_chunk\")}')
print(f'  avg paragraphs/chunk: {s.get(\"avg_paragraphs_per_chunk\")}')
"

# ============================================================
# STEP 2 — Rebuild augmented corpus (v2_chunks_new)
# ============================================================
echo ""
echo "=== Step 2: Rebuild augmented corpus with new chunks ==="

python scripts/build_graph_augmented_corpus.py \
    --m4-dir "$M4_DIR" \
    --chunk-file "$NEW_CHUNKS" \
    --output-dir "$AUG_DIR" \
    --variants v1 v2

echo "  Step 2 done: $(date)"

# ============================================================
# STEP 3 — Dense retrieval evaluation × Qwen3-Embedding-0.6B
# ============================================================
echo ""
echo "=== Step 3: Dense retrieval (Qwen3-Embedding-0.6B) ==="

# Make sure model is downloaded once (subsequent calls reuse cache)
mkdir -p "$DOWNLOAD_ROOT"

run_eval () {
    local TAG="$1"
    local CORPUS="$2"
    local QRELS="$3"
    local OUT_REPORT="$AUG_DIR/eval_${TAG}_qwen06b.json"
    local OUT_RANKING="$AUG_DIR/ranking_${TAG}_qwen06b.jsonl"
    echo ""
    echo "  ── Eval [$TAG] ──"
    python scripts/eval_dense_retrieval.py \
        --data-dir "$AUG_DIR" \
        --queries "$AUG_DIR/queries.jsonl" \
        --corpus "$CORPUS" \
        --qrels "$QRELS" \
        --model-name "$MODEL_NAME" \
        --download-root "$DOWNLOAD_ROOT" \
        --output "$OUT_REPORT" \
        --save-ranking "$OUT_RANKING" \
        --batch-size 16 \
        --max-length 512 \
        --top-k 100
    echo "  ── done [$TAG] $(date)"
}

# Baseline (corpus_v1_enriched is the strongest baseline from 4.16)
run_eval "v1_enriched"     "$AUG_DIR/corpus_v1_enriched.jsonl"   "$AUG_DIR/qrels_v1.jsonl"
run_eval "v2_chunks_new"   "$AUG_DIR/corpus_v2_chunks.jsonl"     "$AUG_DIR/qrels_v2_chunks.jsonl"

# ============================================================
# STEP 4 — Compare results
# ============================================================
echo ""
echo "=== Step 4: Comparison summary ==="

python -c "
import json, glob
from pathlib import Path

reports = sorted(glob.glob('$AUG_DIR/eval_*_qwen06b.json'))
print()
print(f'{\"Variant\":<20} {\"R@1\":>8} {\"R@5\":>8} {\"R@10\":>8} {\"R@100\":>8} {\"MRR\":>8} {\"NDCG@10\":>10}')
print('-' * 80)
for rp in reports:
    name = Path(rp).stem.replace('eval_', '').replace('_qwen06b', '')
    r = json.load(open(rp))
    m = r.get('metrics', {})
    print(f'{name:<20} {m.get(\"recall@1\",0):>8.4f} {m.get(\"recall@5\",0):>8.4f} {m.get(\"recall@10\",0):>8.4f} {m.get(\"recall@100\",0):>8.4f} {m.get(\"mrr\",0):>8.4f} {m.get(\"ndcg@10\",0):>10.4f}')
print()
print('Note: corpus sizes differ. v1 = enriched only; v2 = enriched + new chunks.')
"

echo ""
echo "============================================"
echo "chunk_v2_eval COMPLETE: $(date)"
echo "  New chunks: $NEW_CHUNKS"
echo "  Reports:    $AUG_DIR/eval_*_qwen06b.json"
echo "============================================"
