#!/bin/bash
#SBATCH --job-name=mineru_sB
#SBATCH --partition=cluster02
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:a6000:4
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=24:00:00
#SBATCH --nodelist=gpu-a6000-1
#SBATCH --output=logs/mineru_shardB_%j.out
#SBATCH --error=logs/mineru_shardB_%j.err

# ============================================================================
# MinerU Shard B — gpu-a6000-1, 4× A6000 GPU
# Parses second half of PDFs (711 files)
# ============================================================================

set -euo pipefail

SHARD_NAME="B"
SHARD_FILE="/projects/myyyx1/data-process-test/logs/shard_b.txt"
NUM_GPUS=4

echo "=========================================="
echo "MinerU Shard $SHARD_NAME: $(date)"
echo "Job ID:  ${SLURM_JOBID}"
echo "Node:    $(hostname)"
echo "=========================================="

# ── Env setup ──
module load Miniforge3 2>/dev/null || true
source /cluster/apps/software/Miniforge3/24.11.3-1/etc/profile.d/conda.sh 2>/dev/null || \
  source activate minerU 2>/dev/null || true
conda activate /projects/myyyx1/envs/minerU

# ── Verify GPUs ──
DETECTED_GPUS=$(nvidia-smi -L 2>/dev/null | wc -l)
echo "GPUs detected: $DETECTED_GPUS (requested: $NUM_GPUS)"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader
echo "=========================================="

# ── Paths ──
PROJECT_DIR="/projects/myyyx1/data-process-test"
MINERU_BIN="/projects/myyyx1/envs/minerU/bin/mineru"
OUTPUT_DIR="${PROJECT_DIR}/data/00_raw/mineru_output"
LOCKDIR="${PROJECT_DIR}/logs/.mineru_locks"

mkdir -p "$OUTPUT_DIR" "$LOCKDIR"

# ── Build work list (skip already-parsed from shard file) ──
WORKLIST=$(mktemp /tmp/mineru_shard${SHARD_NAME}_XXXX.txt)
SKIP=0
while IFS= read -r pdf; do
    [ -f "$pdf" ] || continue
    bname=$(basename "$pdf" .pdf)
    if find "$OUTPUT_DIR/$bname" -name "*.md" -type f 2>/dev/null | head -1 | grep -q .; then
        SKIP=$((SKIP + 1))
        continue
    fi
    echo "$pdf" >> "$WORKLIST"
done < "$SHARD_FILE"

TOTAL=$(wc -l < "$WORKLIST")
echo "Shard $SHARD_NAME: $TOTAL to parse, $SKIP skipped (already done)"

if [ "$TOTAL" -eq 0 ]; then
    echo "Nothing to do. Exiting."
    rm -f "$WORKLIST"
    exit 0
fi

# ── Worker function ──
parse_one_pdf() {
    local pdf="$1"
    local gpu_id="$2"
    local bname lockfile doc_output t_start elapsed rc
    bname=$(basename "$pdf" .pdf)
    doc_output="${OUTPUT_DIR}/${bname}"
    lockfile="${LOCKDIR}/${bname}.lock"

    # Atomic lock via mkdir
    if ! mkdir "$lockfile" 2>/dev/null; then
        echo "[GPU${gpu_id}] SKIP  ${bname}  (locked)"
        return 0
    fi
    trap "rmdir '$lockfile' 2>/dev/null" RETURN

    if find "$doc_output" -name "*.md" -type f 2>/dev/null | head -1 | grep -q .; then
        echo "[GPU${gpu_id}] SKIP  ${bname}  (already parsed)"
        return 0
    fi

    echo "[GPU${gpu_id}] START ${bname}"
    t_start=$SECONDS

    CUDA_VISIBLE_DEVICES="$gpu_id" "$MINERU_BIN" \
        -p "$pdf" \
        -o "$doc_output" \
        -m auto \
        -b pipeline \
        -l en \
        -f True \
        -t True \
        -d "cuda" \
        2>&1 | sed "s/^/[GPU${gpu_id}|${SHARD_NAME}] /"

    rc=${PIPESTATUS[0]}
    elapsed=$(( SECONDS - t_start ))

    if [ "$rc" -eq 0 ]; then
        echo "[GPU${gpu_id}|${SHARD_NAME}] DONE  ${bname}  (${elapsed}s)"
    else
        echo "[GPU${gpu_id}|${SHARD_NAME}] FAIL  ${bname}  (exit=${rc}, ${elapsed}s)"
    fi
    return 0
}
export -f parse_one_pdf
export OUTPUT_DIR MINERU_BIN LOCKDIR SHARD_NAME

# ── Dispatch: round-robin background jobs ──
echo ""
echo "Dispatching $TOTAL PDFs across $NUM_GPUS GPUs..."
echo ""

declare -A slot_pids
while IFS= read -r pdf; do
    # Wait until a GPU slot is free
    while [ "${#slot_pids[@]}" -ge "$NUM_GPUS" ]; do
        for g in "${!slot_pids[@]}"; do
            if ! kill -0 "${slot_pids[$g]}" 2>/dev/null; then
                wait "${slot_pids[$g]}" 2>/dev/null || true
                unset "slot_pids[$g]"
            fi
        done
        [ "${#slot_pids[@]}" -ge "$NUM_GPUS" ] && sleep 2
    done

    # Pick first free GPU slot
    local_gpu=0
    for (( g=0; g<NUM_GPUS; g++ )); do
        if [ -z "${slot_pids[$g]+x}" ]; then
            local_gpu=$g; break
        fi
    done

    parse_one_pdf "$pdf" "$local_gpu" &
    slot_pids[$local_gpu]=$!
done < "$WORKLIST"

# Wait for stragglers
for g in "${!slot_pids[@]}"; do
    wait "${slot_pids[$g]}" 2>/dev/null || true
done

rm -f "$WORKLIST"

# Clean stale locks
find "$LOCKDIR" -maxdepth 1 -name "*.lock" -type d -mmin +30 -exec rmdir {} \; 2>/dev/null || true

# ── Summary ──
PARSED=$(find "$OUTPUT_DIR" -name "*.md" -type f 2>/dev/null | wc -l)
echo ""
echo "=========================================="
echo "Shard $SHARD_NAME Complete: $(date)"
echo "  .md files in output dir: $PARSED"
echo "=========================================="
