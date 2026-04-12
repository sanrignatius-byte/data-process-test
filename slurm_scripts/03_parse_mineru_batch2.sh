#!/bin/bash
#SBATCH --job-name=mineru_b2
#SBATCH --partition=cluster02
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --exclusive
#SBATCH --gres=gpu:pro6000:4
#SBATCH --time=24:00:00
#SBATCH --output=logs/mineru_parse_%j.out
#SBATCH --error=logs/mineru_parse_%j.err

# ============================================================================
# MinerU PDF Parsing — Full-node multi-GPU parallel
#
# Requests --exclusive to claim all GPUs on one PRO 6000 node.
# Runs N MinerU workers in parallel (one per GPU).
#
# Default: 4-GPU node (pro6000-[1-4]).  For 10-GPU nodes:
#   sbatch --gres=gpu:pro6000:10 slurm_scripts/03_parse_mineru_batch2.sh
#
# 见缝插针: submit to multiple nodes
#   for n in 1 2 3 4 5 6; do
#     sbatch --nodelist=gpu-pro6000-$n slurm_scripts/03_parse_mineru_batch2.sh
#   done
#
# Lock files prevent duplicate work across concurrent jobs.
# ============================================================================

set -euo pipefail

echo "=========================================="
echo "MinerU Multi-GPU Parallel Parsing: $(date)"
echo "Job ID:  ${SLURM_JOBID}"
echo "Node:    $(hostname)"
echo "=========================================="

# ── Env setup ──
module load Miniforge3 2>/dev/null || true
source /cluster/apps/software/Miniforge3/24.11.3-1/etc/profile.d/conda.sh 2>/dev/null || \
  source activate minerU 2>/dev/null || true
conda activate /projects/myyyx1/envs/minerU

# ── Detect GPUs ──
NUM_GPUS=$(nvidia-smi -L 2>/dev/null | wc -l)
if [ "$NUM_GPUS" -eq 0 ]; then
    echo "ERROR: No GPUs detected"; exit 1
fi
echo "GPUs:    $NUM_GPUS"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader
echo "=========================================="

# ── Paths ──
PROJECT_DIR="/projects/myyyx1/data-process-test"
MINERU_BIN="/projects/myyyx1/envs/minerU/bin/mineru"
PDF_DIR="${PROJECT_DIR}/data/00_raw/pdfs_batch2"
OUTPUT_DIR="${PROJECT_DIR}/data/00_raw/mineru_output"
LOCKDIR="${PROJECT_DIR}/logs/.mineru_locks"

mkdir -p "$OUTPUT_DIR" "$LOCKDIR"

# ── Build work list (skip already-parsed) ──
WORKLIST=$(mktemp /tmp/mineru_worklist_XXXX.txt)
SKIP=0
for pdf in "$PDF_DIR"/*.pdf; do
    [ -f "$pdf" ] || continue
    bname=$(basename "$pdf" .pdf)
    if find "$OUTPUT_DIR/$bname" -name "*.md" -type f 2>/dev/null | head -1 | grep -q .; then
        SKIP=$((SKIP + 1))
        continue
    fi
    echo "$pdf" >> "$WORKLIST"
done

TOTAL=$(wc -l < "$WORKLIST")
echo "PDFs to parse: $TOTAL  (skipped already-parsed: $SKIP)"

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

    # Atomic lock via mkdir — skip if another job is processing this
    if ! mkdir "$lockfile" 2>/dev/null; then
        echo "[GPU${gpu_id}] SKIP  ${bname}  (locked by another job)"
        return 0
    fi
    # Trap to clean up lock on any exit
    trap "rmdir '$lockfile' 2>/dev/null" RETURN

    # Double-check not already done
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
        2>&1 | sed "s/^/[GPU${gpu_id}] /"

    rc=${PIPESTATUS[0]}
    elapsed=$(( SECONDS - t_start ))

    if [ "$rc" -eq 0 ]; then
        echo "[GPU${gpu_id}] DONE  ${bname}  (${elapsed}s)"
    else
        echo "[GPU${gpu_id}] FAIL  ${bname}  (exit=${rc}, ${elapsed}s)"
    fi
    return "$rc"
}
export -f parse_one_pdf
export OUTPUT_DIR MINERU_BIN LOCKDIR

# ── Launch parallel workers ──
echo ""
echo "Dispatching $TOTAL PDFs across $NUM_GPUS GPUs..."
echo ""

if command -v parallel &>/dev/null; then
    # ── GNU parallel: best approach ──
    parallel --will-cite -j "$NUM_GPUS" --linebuffer --halt soon,fail=20% \
        'parse_one_pdf {1} $(( ({%} - 1) ))' \
        :::: "$WORKLIST" || true
else
    # ── Fallback: manual round-robin with background jobs ──
    echo "(GNU parallel not found — using background-job dispatch)"
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

        # Pick the first free GPU id
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
fi

rm -f "$WORKLIST"

# ── Clean up stale locks ──
find "$LOCKDIR" -maxdepth 1 -name "*.lock" -type d -mmin +30 -exec rmdir {} \; 2>/dev/null || true

# ── Summary ──
PARSED=$(find "$OUTPUT_DIR" -name "*.md" -type f 2>/dev/null | wc -l)
echo ""
echo "=========================================="
echo "MinerU Batch Complete: $(date)"
echo "  .md files in output dir: $PARSED"
echo "=========================================="
