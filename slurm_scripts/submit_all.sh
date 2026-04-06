#!/bin/bash
# ============================================================
# PDF 下载工作流 - 提交下载任务
#
# 用法:
#   ./slurm_scripts/submit_all.sh --arxiv-id 2501.09959
#   ./slurm_scripts/submit_all.sh --arxiv-id 2501.09959 --max-references 300
#
# 常用状态查看:
#   squeue -u $USER
#   sacct -j <jobid> --format=JobID,State,Elapsed,MaxRSS
# ============================================================

set -euo pipefail

REPO_ROOT=${REPO_ROOT:-/projects/myyyx1/data-process-test}
OUTPUT_DIR=${OUTPUT_DIR:-data/raw_pdfs}
ARXIV_ID=${ARXIV_ID:-}
MAX_REFERENCES=${MAX_REFERENCES:-200}
MIN_CITATIONS=${MIN_CITATIONS:-0}
CONDA_ENV=${CONDA_ENV:-minerU}

while [[ $# -gt 0 ]]; do
    case $1 in
        --arxiv-id)
            ARXIV_ID="$2"
            shift 2
            ;;
        --max-references)
            MAX_REFERENCES="$2"
            shift 2
            ;;
        --min-citations)
            MIN_CITATIONS="$2"
            shift 2
            ;;
        --output-dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --repo-root)
            REPO_ROOT="$2"
            shift 2
            ;;
        --conda-env)
            CONDA_ENV="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

cd "$REPO_ROOT"
mkdir -p logs

if [[ -z "$ARXIV_ID" ]]; then
    echo "Error: must provide --arxiv-id"
    exit 1
fi

echo "=========================================="
echo "PDF Download Workflow"
echo "Repo root: $REPO_ROOT"
echo "arXiv ID: $ARXIV_ID"
echo "Output dir: $OUTPUT_DIR"
echo "=========================================="

JOB1=$(sbatch --parsable \
    --export=ALL,REPO_ROOT="$REPO_ROOT",CONDA_ENV="$CONDA_ENV",ARXIV_ID="$ARXIV_ID",OUTPUT_DIR="$OUTPUT_DIR",MAX_REFERENCES="$MAX_REFERENCES",MIN_CITATIONS="$MIN_CITATIONS" \
    slurm_scripts/01_fetch_references.sh)
echo "Submitted fetch_references: Job $JOB1"

# Note: MinerU parsing step (02_parse_pdfs.sh) was removed (referenced non-existent parse_only.py).
# Run MinerU parsing manually after download completes.
# The active query generation pipeline is scripts/generate_multihop_l1_queries.py
# which should be run separately after MinerU parsing + graph construction.

echo ""
echo "=========================================="
echo "Job submitted. Check status with:"
echo "  squeue -u \$USER"
echo ""
echo "View logs:"
echo "  tail -f logs/fetch_refs_*.out"
echo "=========================================="
