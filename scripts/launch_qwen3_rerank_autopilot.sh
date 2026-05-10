#!/bin/bash
# Durable launcher for the Qwen3 reranker experiment.
#
# Responsibilities:
#   1. Ensure Qwen/Qwen3-Reranker-4B is present in HF cache.
#   2. Submit slurm_scripts/48_ce_rerank_qwen3_fusion.sh.
#   3. Wait for the Slurm job to finish.
#   4. Write a compact status/result log for morning inspection.

set -euo pipefail

cd /projects/myyyx1/data-process-test

RUN_ID="$(date -u +%Y%m%dT%H%M%SZ)"
AUTOPILOT_LOG="logs/qwen3_rerank_autopilot_${RUN_ID}.log"
OUT_DIR="data/05_eval/cross_encoder_rerank/qwen3_reranker_4b_anchor_top500"
MODEL_ID="Qwen/Qwen3-Reranker-4B"

mkdir -p logs "$OUT_DIR"

{
    echo "============================================"
    echo "Qwen3 reranker autopilot"
    echo "Run ID: ${RUN_ID}"
    echo "Start:  $(date -u)"
    echo "Host:   $(hostname)"
    echo "============================================"
    echo

    export HF_HOME=/projects/myyyx1/models/hf_cache
    export TOKENIZERS_PARALLELISM=false

    echo "=== Step 1: ensure model cache ==="
    /projects/myyyx1/envs/minerU/bin/python - <<'PY'
from huggingface_hub import snapshot_download
path = snapshot_download(
    repo_id="Qwen/Qwen3-Reranker-4B",
    resume_download=True,
    max_workers=2,
)
print(path)
PY
    echo "Model cache ready: $(date -u)"
    du -sh /projects/myyyx1/models/hf_cache/hub/models--Qwen--Qwen3-Reranker-4B 2>/dev/null || true
    echo

    echo "=== Step 2: submit Slurm ==="
    SUBMIT_OUT="$(sbatch slurm_scripts/48_ce_rerank_qwen3_fusion.sh)"
    echo "$SUBMIT_OUT"
    JOB_ID="$(echo "$SUBMIT_OUT" | awk '{print $NF}')"
    echo "JOB_ID=${JOB_ID}"
    echo

    echo "=== Step 3: monitor Slurm ==="
    while squeue -j "$JOB_ID" -h >/tmp/qwen3_rerank_squeue_${JOB_ID}.txt && [ -s /tmp/qwen3_rerank_squeue_${JOB_ID}.txt ]; do
        cat /tmp/qwen3_rerank_squeue_${JOB_ID}.txt
        if [ -f "logs/qwen3_ce_fusion_${JOB_ID}.out" ]; then
            echo "--- slurm stdout tail ---"
            tail -n 20 "logs/qwen3_ce_fusion_${JOB_ID}.out" || true
        fi
        if [ -f "logs/qwen3_ce_fusion_${JOB_ID}.err" ]; then
            echo "--- slurm stderr tail ---"
            tail -n 20 "logs/qwen3_ce_fusion_${JOB_ID}.err" || true
        fi
        echo "--- $(date -u) ---"
        sleep 120
    done

    echo "Slurm job left queue: $(date -u)"
    sacct -j "$JOB_ID" --format=JobID,JobName,State,ExitCode,Elapsed,MaxRSS -P 2>/dev/null || true
    echo

    echo "=== Step 4: result preview ==="
    if [ -f "$OUT_DIR/fusion_report.md" ]; then
        sed -n '1,140p' "$OUT_DIR/fusion_report.md"
    else
        echo "fusion_report.md not found at $OUT_DIR"
        find "$OUT_DIR" -maxdepth 2 -type f -printf '%TY-%Tm-%Td %TH:%TM %s %p\n' 2>/dev/null | sort || true
    fi

    echo
    echo "Autopilot complete: $(date -u)"
} > "$AUTOPILOT_LOG" 2>&1
