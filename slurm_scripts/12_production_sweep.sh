#!/bin/bash
#SBATCH -p cluster02
#SBATCH --job-name=prod_sweep
#SBATCH --output=logs/prod_sweep_%A_%a.out
#SBATCH --error=logs/prod_sweep_%A_%a.err
#SBATCH --time=24:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --array=0-5

# ============================================================
# Production sweep: 6 configs × intra-doc-clean candidates
#
# Array index mapping:
#   0  L3 academic            (88 pairs)
#   1  L3 academic + persona  (88 pairs)
#   2  L3 mixed               (88 pairs)
#   3  L3 mixed + persona     (88 pairs)
#   4  M2 academic            (108 pairs)
#   5  M2 mixed + persona     (108 pairs)
#
# All inputs are strict intra-doc filtered.
# Each config writes to its own output file; --skip-done enables resume.
# --pass-only writes {stem}_pass.jsonl automatically.
#
# Expected yield (conservative 25% pass rate):
#   L3: 88 × 4 configs × 0.25 ≈ 88 new pass
#   M2: 108 × 2 configs × 0.25 ≈ 54 new pass
#   Total new ≈ 142  (+187 existing = ~330)
#
# Optimistic (30%):
#   L3: 88×4×0.30 ≈ 106,  M2: 108×2×0.30 ≈ 65  → 171 (+187 = ~358)
#
# To reach 500 we may need a second round with different seeds
# or additional candidate pools.  See 13_production_sweep_r2.sh.
#
# Usage:
#   sbatch slurm_scripts/12_production_sweep.sh
#   sbatch --array=0,2 slurm_scripts/12_production_sweep.sh   # subset
# ============================================================

set -euo pipefail

module load Miniforge3 || true

REPO_ROOT=${REPO_ROOT:-/projects/myyyx1/data-process-test}
CONDA_ENV=${CONDA_ENV:-/projects/myyyx1/envs/minerU}

PROVIDER=${PROVIDER:-company}
MODEL=${MODEL:-gpt-5.4}
REFERENCE_GRAPH=${REFERENCE_GRAPH:-data/01_graphs/latex_reference_graph.json}
TOPOLOGY_CANDIDATES=${TOPOLOGY_CANDIDATES:-data/01_graphs/latex_hub_multihop_candidates.json}

# Candidate files (strict intra-doc)
L3_CANDIDATES="data/03_queries/l3_candidates_v4_intra_doc.json"
M2_CANDIDATES="data/02_enriched/m2_diverse_candidates_intra_doc.json"

# Output directory
OUT_DIR="data/03_queries/sweep_2026-04-12"

cd "$REPO_ROOT"
mkdir -p logs "$OUT_DIR"

source /cluster/apps/software/Miniforge3/24.11.3-1/etc/profile.d/conda.sh
conda activate "$CONDA_ENV"

if [[ -f .env ]]; then
    set -a
    source .env
    set +a
fi

if [[ -z "${COMPANY_API_KEY:-}" ]]; then
    echo "ERROR: COMPANY_API_KEY not set"
    exit 1
fi

export PYTHONUNBUFFERED=1

# ── Config array ──────────────────────────────────────────────────────────────
IDX=${SLURM_ARRAY_TASK_ID:-0}

case $IDX in
    0)
        CAND="$L3_CANDIDATES"
        STYLE="academic"
        PERSONA=""
        TAG="l3_academic"
        ;;
    1)
        CAND="$L3_CANDIDATES"
        STYLE="academic"
        PERSONA="--use-persona"
        TAG="l3_academic_persona"
        ;;
    2)
        CAND="$L3_CANDIDATES"
        STYLE="mixed"
        PERSONA=""
        TAG="l3_mixed"
        ;;
    3)
        CAND="$L3_CANDIDATES"
        STYLE="mixed"
        PERSONA="--use-persona"
        TAG="l3_mixed_persona"
        ;;
    4)
        CAND="$M2_CANDIDATES"
        STYLE="academic"
        PERSONA=""
        TAG="m2_academic"
        ;;
    5)
        CAND="$M2_CANDIDATES"
        STYLE="mixed"
        PERSONA="--use-persona"
        TAG="m2_mixed_persona"
        ;;
    *)
        echo "ERROR: unknown array index $IDX"
        exit 1
        ;;
esac

OUTPUT="${OUT_DIR}/${TAG}.jsonl"

echo "=========================================="
echo "Production sweep – config $IDX ($TAG)"
echo "Start time:    $(date)"
echo "Candidates:    $CAND"
echo "Style:         $STYLE"
echo "Persona:       ${PERSONA:-disabled}"
echo "Output:        $OUTPUT"
echo "Provider:      $PROVIDER"
echo "Model:         $MODEL"
echo "=========================================="

GEN_CMD=(
    python scripts/generate_multihop_l1_queries.py
    --candidates "$CAND"
    --output "$OUTPUT"
    --pass-only
    --provider "$PROVIDER"
    --model "$MODEL"
    --query-style "$STYLE"
    --reference-graph "$REFERENCE_GRAPH"
    --topology-candidates "$TOPOLOGY_CANDIDATES"
    --skip-done "$OUTPUT"
    --shuffle
    --delay 0.3
)

if [[ -n "$PERSONA" ]]; then
    GEN_CMD+=($PERSONA)
fi

"${GEN_CMD[@]}"

# ── Summary ───────────────────────────────────────────────────────────────────
PASS_FILE="${OUTPUT%.jsonl}_pass.jsonl"
TOTAL=$(wc -l < "$OUTPUT" 2>/dev/null || echo 0)
PASS=$(wc -l < "$PASS_FILE" 2>/dev/null || echo 0)

echo "=========================================="
echo "Sweep config $IDX ($TAG) complete: $(date)"
echo "  Total: $TOTAL"
echo "  Pass:  $PASS"
echo "=========================================="
