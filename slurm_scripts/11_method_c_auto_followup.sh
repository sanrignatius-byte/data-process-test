#!/bin/bash
#SBATCH -p cluster02
#SBATCH --job-name=method_c_auto
#SBATCH --output=logs/method_c_auto_%j.out
#SBATCH --error=logs/method_c_auto_%j.err
#SBATCH --time=2:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G

set -euo pipefail

module load Miniforge3 || true

REPO_ROOT=${REPO_ROOT:-/projects/myyyx1/data-process-test}
CONDA_ENV=${CONDA_ENV:-/projects/myyyx1/envs/minerU}
SOURCE_JOB_ID=${SOURCE_JOB_ID:?SOURCE_JOB_ID must be set}
SAMPLE_LIMIT=${SAMPLE_LIMIT:-48}

cd "$REPO_ROOT"
mkdir -p logs data/03_queries

source /cluster/apps/software/Miniforge3/24.11.3-1/etc/profile.d/conda.sh
conda activate "$CONDA_ENV"

if [[ -f .env ]]; then
    set -a
    # shellcheck disable=SC1091
    source .env
    set +a
fi

export PYTHONUNBUFFERED=1

echo "=========================================="
echo "Method C auto follow-up controller"
echo "Start time:    $(date)"
echo "Source job:    ${SOURCE_JOB_ID}"
echo "Sample limit:  ${SAMPLE_LIMIT}"
echo "=========================================="

python3 -u scripts/method_c_auto_followup.py \
    --source-job-id "$SOURCE_JOB_ID" \
    --sample-limit "$SAMPLE_LIMIT"

echo "=========================================="
echo "Method C auto follow-up controller done: $(date)"
echo "Report JSON: data/03_queries/method_c_auto_followup_latest.json"
echo "Report MD:   data/03_queries/method_c_auto_followup_latest.md"
echo "=========================================="
