#!/bin/bash
#SBATCH --job-name=33_para_merge
#SBATCH --output=logs/33_paragraph_merge_%j.out
#SBATCH --error=logs/33_paragraph_merge_%j.err
#SBATCH --partition=cluster02
#SBATCH --qos=msc
#SBATCH --gres=gpu:pro6000:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00

# Block 4: Paragraph-merge corpus (n=400 and n=500) + dense baseline eval
# Tests claim C_PARA: do merged chunks improve dense R@1 vs v1_enriched (0.2389)?
# Step 1: Re-chunk paragraphs (CPU, fast)
# Step 2: Build corpus v3 using new chunks (CPU)
# Step 3: Dense retrieval on v3 corpus (GPU, ~1h)

set -e
cd /projects/myyyx1/data-process-test
source /cluster/apps/software/Miniforge3/24.11.3-1/etc/profile.d/conda.sh
conda activate /projects/myyyx1/envs/minerU
export PYTHONPATH=/projects/myyyx1/data-process-test:$PYTHONPATH
export MODELSCOPE_CACHE=models

CHUNK_V2="data/01_graphs/chunk_virtual_nodes_v2.json"
PARA_N400="data/01_graphs/paragraph_chunks_n400.json"
PARA_N500="data/01_graphs/paragraph_chunks_n500.json"
OUT_N400="data/05_eval/dense_retrieval/augmented_v3_n400"
OUT_N500="data/05_eval/dense_retrieval/augmented_v3_n500"
MODEL="Qwen/Qwen3-Embedding-0.6B"

mkdir -p logs

# ============================================================
# STEP 1: Build paragraph-merged chunk graphs
# ============================================================
echo "=== Step 1: Build paragraph chunk graphs ==="
date

echo "  n=400 ..."
python scripts/build_paragraph_chunks.py \
    --input "$CHUNK_V2" \
    --chunk-size 400 \
    --output "$PARA_N400"

echo "  n=500 ..."
python scripts/build_paragraph_chunks.py \
    --input "$CHUNK_V2" \
    --chunk-size 500 \
    --output "$PARA_N500"

echo "  Step 1 done: $(date)"

# ============================================================
# STEP 2: Build augmented corpus using new paragraph chunks
# ============================================================
echo ""
echo "=== Step 2: Build augmented corpus v3 ==="

echo "  n=400 corpus ..."
python scripts/build_graph_augmented_corpus.py \
    --chunk-file "$PARA_N400" \
    --output-dir "$OUT_N400"

echo "  n=500 corpus ..."
python scripts/build_graph_augmented_corpus.py \
    --chunk-file "$PARA_N500" \
    --output-dir "$OUT_N500"

echo "  Step 2 done: $(date)"

# ============================================================
# STEP 3: Dense retrieval baseline on v3 corpus
# The key comparison: corpus_v2_chunks.jsonl (para-merged) vs
#   augmented_v2/corpus_v1_enriched.jsonl (element-level)
# Success: R@1 > 0.2389 (current v1_enriched dense baseline)
# ============================================================
echo ""
echo "=== Step 3: Dense retrieval on v3 corpora (GPU) ==="

for OUT_DIR in "$OUT_N400" "$OUT_N500"; do
    TAG=$(basename "$OUT_DIR")
    CORPUS="$OUT_DIR/corpus_v2_chunks.jsonl"
    QRELS="$OUT_DIR/qrels_v2_chunks.jsonl"

    if [ ! -f "$CORPUS" ]; then
        echo "  WARN: $CORPUS not found, skipping"
        continue
    fi

    echo ""
    echo "  Evaluating $TAG ..."
    python scripts/eval_dense_retrieval.py \
        --corpus      "$CORPUS" \
        --qrels       "$QRELS" \
        --queries     "$OUT_DIR/queries.jsonl" \
        --model-name  "$MODEL" \
        --output      "$OUT_DIR/eval_v3_chunks_qwen06b.json" \
        --ranking-out "$OUT_DIR/ranking_v3_chunks_qwen06b.jsonl" \
        --batch-size  8
done

echo ""
echo "=== ALL DONE: $(date) ==="
echo ""
echo "== Dense baseline comparison =="
printf "%-44s %8s %8s %8s\n" "corpus" "R@1" "R@10" "MRR"
echo "-----------------------------------------------------------------------"

REF="data/05_eval/dense_retrieval/augmented_v2/eval_v1_enriched_qwen06b.json"
[ -f "$REF" ] && python3 -c "
import json; m=json.load(open('$REF'))
print(f\"{'v1_enriched (ref)':<44} {m.get('recall@1',0):8.4f} {m.get('recall@10',0):8.4f} {m.get('mrr',0):8.4f}\")"

for OUT_DIR in "$OUT_N400" "$OUT_N500"; do
    TAG=$(basename "$OUT_DIR")
    fp="$OUT_DIR/eval_v3_chunks_qwen06b.json"
    [ -f "$fp" ] || continue
    python3 -c "
import json; m=json.load(open('$fp'))
print(f\"{'$TAG':<44} {m.get('recall@1',0):8.4f} {m.get('recall@10',0):8.4f} {m.get('mrr',0):8.4f}\")"
done
