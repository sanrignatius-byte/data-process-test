#!/bin/bash
# ============================================================
# M4 Workflow - 提交所有任务（带依赖）
#
# 用法:
#   ./slurm_scripts/submit_all.sh --arxiv-id 2501.09959
#   ./slurm_scripts/submit_all.sh --arxiv-id 2501.09959 --max-references 300
#   ./slurm_scripts/submit_all.sh --skip-download
#   ./slurm_scripts/submit_all.sh --parse-only
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

SKIP_DOWNLOAD=false
PARSE_ONLY=false

while [[ $# -gt 0 ]]; do
    case $1 in
        --skip-download)
            SKIP_DOWNLOAD=true
            shift
            ;;
        --parse-only)
            PARSE_ONLY=true
            shift
            ;;
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

if [[ "$SKIP_DOWNLOAD" == false && -z "$ARXIV_ID" ]]; then
    echo "Error: must provide --arxiv-id when download step is enabled"
    exit 1
fi

echo "=========================================="
echo "M4 Workflow Submission"
echo "Repo root: $REPO_ROOT"
echo "Skip download: $SKIP_DOWNLOAD"
echo "Parse only: $PARSE_ONLY"
echo "arXiv ID: ${ARXIV_ID:-N/A}"
echo "Output dir: $OUTPUT_DIR"
echo "=========================================="

if [[ "$SKIP_DOWNLOAD" == false ]]; then
    JOB1=$(sbatch --parsable \
        --export=ALL,REPO_ROOT="$REPO_ROOT",CONDA_ENV="$CONDA_ENV",ARXIV_ID="$ARXIV_ID",OUTPUT_DIR="$OUTPUT_DIR",MAX_REFERENCES="$MAX_REFERENCES",MIN_CITATIONS="$MIN_CITATIONS" \
        slurm_scripts/01_fetch_references.sh)
    echo "Submitted fetch_references: Job $JOB1"

    JOB2=$(sbatch --parsable --dependency=afterok:$JOB1 \
        --export=ALL,REPO_ROOT="$REPO_ROOT",CONDA_ENV="$CONDA_ENV",INPUT_PDF_DIR="$OUTPUT_DIR",MINERU_OUTPUT_DIR="data/mineru_output" \
        slurm_scripts/02_parse_pdfs.sh)
    echo "Submitted parse_pdfs: Job $JOB2 (depends on $JOB1)"
else
    JOB2=$(sbatch --parsable \
        --export=ALL,REPO_ROOT="$REPO_ROOT",CONDA_ENV="$CONDA_ENV",INPUT_PDF_DIR="$OUTPUT_DIR",MINERU_OUTPUT_DIR="data/mineru_output" \
        slurm_scripts/02_parse_pdfs.sh)
    echo "Submitted parse_pdfs: Job $JOB2"
fi

if [[ "$PARSE_ONLY" == false ]]; then
    JOB3=$(sbatch --parsable --kill-on-invalid-dep=yes --dependency=afterok:$JOB2 \
        --export=ALL,REPO_ROOT="$REPO_ROOT",CONDA_ENV="$CONDA_ENV",MINERU_INPUT_DIR="data/mineru_output",QUERY_OUTPUT_DIR="data/queries_output/m4_queries" \
        slurm_scripts/03_generate_m4_queries.sh)
    echo "Submitted m4_query_gen: Job $JOB3 (depends on $JOB2)"
fi

echo ""
echo "=========================================="
echo "All jobs submitted. Check status with:"
echo "  squeue -u \$USER"
echo ""
echo "View logs:"
echo "  tail -f logs/fetch_refs_*.out"
echo "  tail -f logs/parse_*.out"
echo "  tail -f logs/m4_query_*.out"
echo ""
echo "If a dependency fails, inspect with:"
echo "  sacct -j <jobid> --format=JobID,State,ExitCode,Elapsed,NodeList"
echo "=========================================="
