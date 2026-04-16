#!/bin/bash
#SBATCH --job-name=llm_summary
#SBATCH --partition=cluster02
#SBATCH --qos=msc
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=logs/llm_summary_%j.out
#SBATCH --error=logs/llm_summary_%j.err

echo "============================================"
echo "Job 1: LLM Summary Virtual Nodes (Old 53 docs)"
echo "============================================"
echo "Job ID:  $SLURM_JOB_ID"
echo "Node:    $(hostname)"
echo "Start:   $(date)"
echo "============================================"

cd /projects/myyyx1/data-process-test
source /cluster/apps/software/Miniforge3/24.11.3-1/etc/profile.d/conda.sh
conda activate /projects/myyyx1/envs/minerU

mkdir -p logs

# Load API keys (Iron Rule 3)
if [ -f .env ]; then
    set -a && source .env && set +a
    echo "[ok] .env loaded"
else
    echo "[FATAL] .env not found" && exit 1
fi

# ── Step 1: Rebuild chunk virtual nodes ──
# Rebuild with the latest script version (adds paragraph_nodes + paragraph edges).
# Pure CPU task, ~2min for 1147 docs. Regenerating ensures paragraph_nodes
# are present so LLM summary can use fine-grained section text.
CHUNK_OUTPUT="data/01_graphs/chunk_virtual_nodes.json"
echo ""
echo "[Step 1] Building chunk virtual nodes (with paragraph nodes)..."
python scripts/build_chunk_virtual_nodes.py \
    --target-words 450 \
    --min-words 100 \
    --output "$CHUNK_OUTPUT"
echo "[Step 1] Done: $(date)"

# ── Step 2: Build LLM summary nodes (old 53 docs only) ──
echo ""
echo "[Step 2] Building LLM summary virtual nodes (53 old docs)..."
python scripts/build_llm_summary_nodes.py \
    --input "$CHUNK_OUTPUT" \
    --output "data/01_graphs/llm_summary_virtual_nodes.json" \
    --doc-id-file "data/doc_lists/old_53_docs.txt" \
    --provider company \
    --delay 0.3 \
    --flush-every 5 \
    --incremental

echo ""
echo "============================================"
echo "Job 1 Complete: $(date)"
echo "============================================"
