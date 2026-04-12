#!/bin/bash
# ============================================================================
# Auto-watcher: monitors download + MinerU jobs, resubmits if they die.
# Run in background on login node:
#   nohup bash slurm_scripts/05_auto_watcher.sh > logs/auto_watcher.log 2>&1 &
#
# What it does:
#   1. Every 10 minutes, checks if download job is still running.
#      If not, checks if download is complete. If incomplete, resubmits.
#   2. Checks if MinerU job is running/queued.
#      If not, checks if there are unparsed PDFs. If so, resubmits.
#   3. Stops automatically when both tasks are complete.
# ============================================================================

set -uo pipefail

PROJECT_DIR="/projects/myyyx1/data-process-test"
PDF_DIR="${PROJECT_DIR}/data/00_raw/pdfs_batch2"
OUTPUT_DIR="${PROJECT_DIR}/data/00_raw/mineru_output"
LOGDIR="${PROJECT_DIR}/logs"
DOWNLOAD_SCRIPT="${PROJECT_DIR}/slurm_scripts/04_download_references_batch2.sh"
MINERU_SCRIPT="${PROJECT_DIR}/slurm_scripts/03_parse_mineru_batch2.sh"

CHECK_INTERVAL=600   # 10 minutes
MAX_HOURS=48         # safety timeout: 48 hours
TARGET_PDFS=2300     # approximate number of expected PDFs

mkdir -p "$LOGDIR"
cd "$PROJECT_DIR" || exit 1

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"
}

count_pdfs() {
    find "$PDF_DIR" -name "*.pdf" -type f 2>/dev/null | wc -l
}

count_parsed() {
    find "$OUTPUT_DIR" -name "*.md" -type f 2>/dev/null | wc -l
}

# Check if any job matching a name pattern is running or pending
is_job_alive() {
    local pattern="$1"
    squeue -u "$USER" -h -o "%j %T" 2>/dev/null | grep -q "$pattern"
}

get_job_id() {
    local pattern="$1"
    squeue -u "$USER" -h -o "%i %j" 2>/dev/null | grep "$pattern" | awk '{print $1}' | head -1
}

log "=========================================="
log "Auto-watcher started"
log "  Download script: $DOWNLOAD_SCRIPT"
log "  MinerU script:   $MINERU_SCRIPT"
log "  Check interval:  ${CHECK_INTERVAL}s"
log "  Safety timeout:  ${MAX_HOURS}h"
log "=========================================="

START_TIME=$(date +%s)
ITERATION=0

while true; do
    ITERATION=$((ITERATION + 1))
    NOW=$(date +%s)
    ELAPSED_HOURS=$(( (NOW - START_TIME) / 3600 ))

    if [ "$ELAPSED_HOURS" -ge "$MAX_HOURS" ]; then
        log "Safety timeout reached (${MAX_HOURS}h). Exiting."
        break
    fi

    NUM_PDFS=$(count_pdfs)
    NUM_PARSED=$(count_parsed)
    log "--- Check #$ITERATION: PDFs=$NUM_PDFS, Parsed(md)=$NUM_PARSED ---"

    # ── Phase 1: Download monitoring ──
    if is_job_alive "dl_refs"; then
        DL_JOB=$(get_job_id "dl_refs")
        log "[download] Job $DL_JOB is alive. PDFs downloaded: $NUM_PDFS"
    else
        if [ "$NUM_PDFS" -ge "$TARGET_PDFS" ]; then
            log "[download] COMPLETE. $NUM_PDFS PDFs downloaded."
        else
            log "[download] Job NOT running! Only $NUM_PDFS/$TARGET_PDFS PDFs. Resubmitting..."
            DL_NEW=$(sbatch "$DOWNLOAD_SCRIPT" 2>&1)
            log "[download] Resubmitted: $DL_NEW"
        fi
    fi

    # ── Phase 2: MinerU monitoring ──
    # Only start MinerU check if we have at least some PDFs to parse
    if [ "$NUM_PDFS" -ge 30 ]; then
        # Count how many PDFs don't have a corresponding .md yet
        UNPARSED=0
        for pdf in "$PDF_DIR"/*.pdf; do
            [ -f "$pdf" ] || continue
            bname=$(basename "$pdf" .pdf)
            if ! find "$OUTPUT_DIR/$bname" -name "*.md" -type f 2>/dev/null | head -1 | grep -q .; then
                UNPARSED=$((UNPARSED + 1))
            fi
        done

        if is_job_alive "mineru"; then
            MU_JOB=$(get_job_id "mineru")
            log "[mineru] Job $MU_JOB is alive. Unparsed: $UNPARSED"
        else
            if [ "$UNPARSED" -gt 0 ]; then
                log "[mineru] Job NOT running! $UNPARSED PDFs unparsed. Submitting..."
                MU_NEW=$(sbatch "$MINERU_SCRIPT" 2>&1)
                log "[mineru] Submitted: $MU_NEW"
            else
                log "[mineru] All $NUM_PDFS PDFs are parsed. Nothing to do."
            fi
        fi
    else
        log "[mineru] Only $NUM_PDFS PDFs available, waiting for more downloads."
    fi

    # ── Phase 3: Check if everything is done ──
    if [ "$NUM_PDFS" -ge "$TARGET_PDFS" ]; then
        UNPARSED_FINAL=0
        for pdf in "$PDF_DIR"/*.pdf; do
            [ -f "$pdf" ] || continue
            bname=$(basename "$pdf" .pdf)
            if ! find "$OUTPUT_DIR/$bname" -name "*.md" -type f 2>/dev/null | head -1 | grep -q .; then
                UNPARSED_FINAL=$((UNPARSED_FINAL + 1))
            fi
        done
        if [ "$UNPARSED_FINAL" -eq 0 ] && ! is_job_alive "mineru" && ! is_job_alive "dl_refs"; then
            log "=========================================="
            log "ALL DONE! $NUM_PDFS PDFs downloaded, $NUM_PARSED .md files generated."
            log "=========================================="
            break
        fi
    fi

    sleep "$CHECK_INTERVAL"
done

log "Auto-watcher finished."
