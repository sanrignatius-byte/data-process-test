#!/bin/bash
#SBATCH -p cluster02
#SBATCH --job-name=l1_mh_v2
#SBATCH --output=logs/l1_mh_v2_%j.out
#SBATCH --error=logs/l1_mh_v2_%j.err
#SBATCH --time=8:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G

# ============================================================
# Step 7: Generate L1 multi-hop cross-modal dual-evidence queries (v2 prompt)
# Supports env var overrides for quick iteration:
#   LIMIT, DELAY, OUTPUT, MODEL, CANDIDATES, NO_IMAGES, DRY_RUN
# ============================================================

set -euo pipefail

module load Miniforge3

REPO_ROOT=${REPO_ROOT:-/projects/myyyx1/data-process-test}
CONDA_ENV=${CONDA_ENV:-minerU}
CANDIDATES=${CANDIDATES:-data/multihop_l1_candidates.json}
OUTPUT=${OUTPUT:-data/l1_multihop_queries_v2.jsonl}
MODEL=${MODEL:-claude-sonnet-4-5-20250929}
LIMIT=${LIMIT:-150}
DELAY=${DELAY:-0.5}
NO_IMAGES=${NO_IMAGES:-0}
DRY_RUN=${DRY_RUN:-0}

cd "$REPO_ROOT"
mkdir -p logs "$(dirname "$OUTPUT")"

if [[ -f .env ]]; then
    # shellcheck disable=SC2046
    export $(grep -v '^#' .env | xargs)
    echo "Loaded API keys from .env"
fi

if [[ -z "${ANTHROPIC_API_KEY:-}" ]]; then
    echo "ERROR: ANTHROPIC_API_KEY is not set."
    exit 1
fi

if [[ ! -f "$CANDIDATES" ]]; then
    echo "ERROR: candidates file not found: $CANDIDATES"
    exit 1
fi

echo "=========================================="
echo "Starting L1 Multi-hop v2 generation"
echo "Start time: $(date)"
echo "Repo root:   $REPO_ROOT"
echo "Candidates:  $CANDIDATES"
echo "Output:      $OUTPUT"
echo "Model:       $MODEL"
echo "Limit:       $LIMIT"
echo "Delay:       $DELAY"
echo "No images:   $NO_IMAGES"
echo "Dry run:     $DRY_RUN"
echo "=========================================="

# Always activate conda env before running generation.
# Avoid `conda info --base` because plugin discovery can fail/hang on some clusters.
export CONDA_NO_PLUGINS=true
CONDA_BASE=${CONDA_BASE:-/cluster/apps/software/Miniforge3/24.11.3-1}
if [[ ! -f "$CONDA_BASE/etc/profile.d/conda.sh" ]]; then
    echo "ERROR: conda init script not found: $CONDA_BASE/etc/profile.d/conda.sh"
    exit 1
fi
source "$CONDA_BASE/etc/profile.d/conda.sh"
conda activate "$CONDA_ENV"
echo "Using conda env: ${CONDA_DEFAULT_ENV:-unknown}"
echo "Python: $(which python)"

CMD=(
    python -u scripts/generate_multihop_l1_queries.py
    --candidates "$CANDIDATES"
    --output "$OUTPUT"
    --model "$MODEL"
    --limit "$LIMIT"
    --delay "$DELAY"
)

if [[ "$NO_IMAGES" == "1" ]]; then
    CMD+=(--no-images)
fi

if [[ "$DRY_RUN" == "1" ]]; then
    CMD+=(--dry-run)
fi

"${CMD[@]}"

echo "=========================================="
echo "Done: $(date)"
echo "Output file size:"
ls -lh "$OUTPUT" || true
echo "Output line count:"
wc -l "$OUTPUT" || true
echo "=========================================="
