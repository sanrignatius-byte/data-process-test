#!/bin/bash
# ============================================================================
# MinerU Shard Parser — Generic template
# Usage: sbatch --export=SHARD_ID=1,GPU_TYPE=a40,GPU_NODE=gpu-a40-1 slurm_scripts/08_mineru_shard.sh
#
# Required env vars (pass via --export):
#   SHARD_ID   - shard number (1,2,3,4), maps to logs/shard_N.txt
#   GPU_TYPE   - GPU model name for --gres (a40, a6000, pro6000, v100, etc.)
#   GPU_NODE   - target node name (gpu-a40-1, gpu-a6000-1, etc.)
#
# QOS limit: 2 GPUs per type per user → each job uses exactly 2 GPUs
# ============================================================================
#SBATCH --job-name=mnu_sX
#SBATCH --partition=cluster02
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=24:00:00
#SBATCH --output=logs/mineru_shard_%j.out
#SBATCH --error=logs/mineru_shard_%j.err

set -euo pipefail

# ── Validate env vars ──
: "${SHARD_ID:?ERROR: set SHARD_ID (1-4)}"
: "${GPU_TYPE:?ERROR: set GPU_TYPE (a40, a6000, etc.)}"

PROJECT_DIR="/projects/myyyx1/data-process-test"
SHARD_FILE="${PROJECT_DIR}/logs/shard_${SHARD_ID}.txt"
NUM_GPUS=2  # QOS limit per GPU type

if [ ! -f "$SHARD_FILE" ]; then
    echo "ERROR: shard file not found: $SHARD_FILE"
    exit 1
fi

echo "=========================================="
echo "MinerU Shard ${SHARD_ID} (${GPU_TYPE}): $(date)"
echo "Job ID:  ${SLURM_JOBID}"
echo "Node:    $(hostname)"
echo "Shard:   ${SHARD_FILE} ($(wc -l < "$SHARD_FILE") PDFs)"
echo "=========================================="

# ── Env setup ──
module load Miniforge3 2>/dev/null || true
source /cluster/apps/software/Miniforge3/24.11.3-1/etc/profile.d/conda.sh 2>/dev/null || \
  source activate minerU 2>/dev/null || true
conda activate /projects/myyyx1/envs/minerU

# ── Verify GPUs ──
DETECTED=$(nvidia-smi -L 2>/dev/null | wc -l)
echo "GPUs detected: $DETECTED (using: $NUM_GPUS)"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader
echo "=========================================="

# ── Paths ──
MINERU_BIN="/projects/myyyx1/envs/minerU/bin/mineru"
OUTPUT_DIR="${PROJECT_DIR}/data/00_raw/mineru_output"
LOCKDIR="${PROJECT_DIR}/logs/.mineru_locks"
mkdir -p "$OUTPUT_DIR" "$LOCKDIR"

# ── Build work list ──
WORKLIST=$(mktemp /tmp/mineru_s${SHARD_ID}_XXXX.txt)
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
echo "To parse: $TOTAL, skipped: $SKIP"

if [ "$TOTAL" -eq 0 ]; then
    echo "Nothing to do."; rm -f "$WORKLIST"; exit 0
fi

# ── Worker function ──
parse_one_pdf() {
    local pdf="$1" gpu_id="$2"
    local bname doc_output lockfile t_start elapsed rc
    bname=$(basename "$pdf" .pdf)
    doc_output="${OUTPUT_DIR}/${bname}"
    lockfile="${LOCKDIR}/${bname}.lock"

    if ! mkdir "$lockfile" 2>/dev/null; then
        echo "[S${SHARD_ID}|GPU${gpu_id}] SKIP ${bname} (locked)"
        return 0
    fi
    trap "rmdir '$lockfile' 2>/dev/null" RETURN

    if find "$doc_output" -name "*.md" -type f 2>/dev/null | head -1 | grep -q .; then
        echo "[S${SHARD_ID}|GPU${gpu_id}] SKIP ${bname} (done)"
        return 0
    fi

    echo "[S${SHARD_ID}|GPU${gpu_id}] START ${bname}"
    t_start=$SECONDS

    CUDA_VISIBLE_DEVICES="$gpu_id" "$MINERU_BIN" \
        -p "$pdf" -o "$doc_output" \
        -m auto -b pipeline -l en -f True -t True -d cuda \
        2>&1 | sed "s/^/[S${SHARD_ID}|GPU${gpu_id}] /"

    rc=${PIPESTATUS[0]}
    elapsed=$(( SECONDS - t_start ))

    if [ "$rc" -eq 0 ]; then
        echo "[S${SHARD_ID}|GPU${gpu_id}] DONE ${bname} (${elapsed}s)"
    else
        echo "[S${SHARD_ID}|GPU${gpu_id}] FAIL ${bname} (rc=${rc}, ${elapsed}s)"
    fi
    return 0
}
export -f parse_one_pdf
export OUTPUT_DIR MINERU_BIN LOCKDIR SHARD_ID

# ── Dispatch ──
echo "Dispatching $TOTAL PDFs across $NUM_GPUS GPUs..."

# Use indexed arrays instead of associative arrays (avoids nounset issues)
declare -a slot_pids
for (( i=0; i<NUM_GPUS; i++ )); do slot_pids[$i]=0; done
DONE=0

while IFS= read -r pdf; do
    # Find a free GPU slot (wait until one is free)
    local_gpu=-1
    while [ "$local_gpu" -eq -1 ]; do
        for (( g=0; g<NUM_GPUS; g++ )); do
            pid="${slot_pids[$g]}"
            if [ "$pid" -eq 0 ] || ! kill -0 "$pid" 2>/dev/null; then
                if [ "$pid" -ne 0 ]; then
                    wait "$pid" 2>/dev/null || true
                    DONE=$((DONE + 1))
                    if [ $((DONE % 10)) -eq 0 ]; then
                        echo "[S${SHARD_ID}] Progress: ${DONE}/${TOTAL} dispatched"
                    fi
                fi
                local_gpu=$g
                break
            fi
        done
        [ "$local_gpu" -eq -1 ] && sleep 2
    done

    parse_one_pdf "$pdf" "$local_gpu" &
    slot_pids[$local_gpu]=$!
done < "$WORKLIST"

# Wait for all remaining workers
for (( g=0; g<NUM_GPUS; g++ )); do
    pid="${slot_pids[$g]}"
    [ "$pid" -ne 0 ] && wait "$pid" 2>/dev/null || true
done

rm -f "$WORKLIST"
find "$LOCKDIR" -maxdepth 1 -name "*.lock" -type d -mmin +30 -exec rmdir {} \; 2>/dev/null || true

PARSED=$(find "$OUTPUT_DIR" -name "*.md" -type f 2>/dev/null | wc -l)
echo "=========================================="
echo "Shard ${SHARD_ID} Complete: $(date)"
echo "  Total .md files: $PARSED"
echo "=========================================="
