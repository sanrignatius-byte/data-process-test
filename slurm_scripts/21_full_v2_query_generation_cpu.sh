#!/bin/bash
#SBATCH --job-name=full_v2_qgen_prod
#SBATCH --partition=cluster02
#SBATCH --qos=msc
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=24:00:00
#SBATCH --output=logs/full_v2_qgen_%A_%a.out
#SBATCH --error=logs/full_v2_qgen_%A_%a.err
#SBATCH --array=0-5

set -euo pipefail

REPO_ROOT=${REPO_ROOT:-/projects/myyyx1/data-process-test}
CONDA_ENV=${CONDA_ENV:-/projects/myyyx1/envs/minerU}

PROVIDER=${PROVIDER:-company}
MODEL=${MODEL:-gpt-5.4}
MAX_PER_DOC=${MAX_PER_DOC:-4}
PAIR_STRATEGY=${PAIR_STRATEGY:-all}
LIMIT=${LIMIT:-0}
DELAY=${DELAY:-0.3}
ENABLE_LLM_QC=${ENABLE_LLM_QC:-1}

ELEMENTS_V2="data/01_graphs/multimodal_elements_v2.json"
PRUNED_GRAPH_V2="data/01_graphs/pruned_graph_v2.json"
REFERENCE_GRAPH_V2="data/01_graphs/latex_reference_graph_v2.json"
TOPOLOGY_CANDIDATES_V2="data/01_graphs/latex_hub_multihop_candidates_v2.json"
ELEMENTS_V2_PROD="data/02_enriched/multimodal_elements_v2_production_partial.json"
ELEMENTS_V2_REPORT="data/02_enriched/multimodal_elements_v2_production_partial_report.json"
RAW_PAIR_FILE="data/05_eval/m2/full_doc_pairs_v2_raw_mpd${MAX_PER_DOC}.json"
PAIR_FILE="data/05_eval/m2/full_doc_pairs_v2_prod_mpd${MAX_PER_DOC}.json"
PAIR_LOCK="${PAIR_FILE}.lock"
OUT_DIR="data/03_queries/full_doc_v2_prod_1040_2026-04-18"

mkdir -p "$OUT_DIR" logs
cd "$REPO_ROOT"

source /cluster/apps/software/Miniforge3/24.11.3-1/etc/profile.d/conda.sh
conda activate "$CONDA_ENV"

if [[ -f .env ]]; then
    set -a
    source .env
    set +a
fi

if [[ "$PROVIDER" == "company" && -z "${COMPANY_API_KEY:-}" ]]; then
    echo "ERROR: COMPANY_API_KEY not set"
    exit 1
fi

export PYTHONUNBUFFERED=1

# Build the production element asset + shared pair pool once. Other array tasks wait.
if [[ ! -f "$PAIR_FILE" ]]; then
    if mkdir "$PAIR_LOCK" 2>/dev/null; then
        echo "[$(date)] Preparing production element asset: $ELEMENTS_V2_PROD"
        python scripts/prepare_v2_production_elements.py \
            --base-elements "$ELEMENTS_V2" \
            --pruned-graph "$PRUNED_GRAPH_V2" \
            --output "$ELEMENTS_V2_PROD" \
            --report "$ELEMENTS_V2_REPORT"

        echo "[$(date)] Building shared raw pair file: $RAW_PAIR_FILE"
        python scripts/select_intra_doc_pairs.py \
            --elements "$ELEMENTS_V2_PROD" \
            --output "$RAW_PAIR_FILE" \
            --strategy "$PAIR_STRATEGY" \
            --max-per-doc "$MAX_PER_DOC"

        echo "[$(date)] Injecting partial enrichments into pair file: $PAIR_FILE"
        python scripts/inject_pair_enrichments.py \
            --pairs "$RAW_PAIR_FILE" \
            --elements "$ELEMENTS_V2_PROD" \
            --output "$PAIR_FILE"
        rmdir "$PAIR_LOCK"
    else
        echo "[$(date)] Waiting for pair file to be created by another array task..."
        while [[ ! -f "$PAIR_FILE" ]]; do
            sleep 30
        done
    fi
fi

IDX=${SLURM_ARRAY_TASK_ID:-0}

case $IDX in
    0)
        STYLE="academic"
        PERSONA=""
        TAG="academic"
        ;;
    1)
        STYLE="academic"
        PERSONA="--use-persona"
        TAG="academic_persona"
        ;;
    2)
        STYLE="mixed"
        PERSONA=""
        TAG="mixed"
        ;;
    3)
        STYLE="mixed"
        PERSONA="--use-persona"
        TAG="mixed_persona"
        ;;
    4)
        STYLE="real_user"
        PERSONA=""
        TAG="real_user"
        ;;
    5)
        STYLE="real_user"
        PERSONA="--use-persona"
        TAG="real_user_persona"
        ;;
    *)
        echo "ERROR: unknown array index $IDX"
        exit 1
        ;;
esac

OUTPUT="${OUT_DIR}/full_v2_${TAG}.jsonl"

echo "=========================================="
echo "Full-doc v2 query generation – config $IDX ($TAG)"
echo "Start time:    $(date)"
echo "Pair file:     $PAIR_FILE"
echo "Elements:      $ELEMENTS_V2_PROD"
echo "Report:        $ELEMENTS_V2_REPORT"
echo "Graph:         $REFERENCE_GRAPH_V2"
echo "Topology:      $TOPOLOGY_CANDIDATES_V2"
echo "Style:         $STYLE"
echo "Persona:       ${PERSONA:-disabled}"
echo "Max per doc:   $MAX_PER_DOC"
echo "Provider:      $PROVIDER"
echo "Model:         $MODEL"
echo "Output:        $OUTPUT"
echo "LLM QC:        $([[ "$ENABLE_LLM_QC" == "1" ]] && echo enabled || echo skipped)"
echo "=========================================="

GEN_CMD=(
    python scripts/generate_multihop_l1_queries.py
    --candidates "$PAIR_FILE"
    --output "$OUTPUT"
    --pass-only
    --provider "$PROVIDER"
    --model "$MODEL"
    --query-style "$STYLE"
    --reference-graph "$REFERENCE_GRAPH_V2"
    --topology-candidates "$TOPOLOGY_CANDIDATES_V2"
    --skip-done "$OUTPUT"
    --shuffle
    --delay "$DELAY"
)

if [[ "$ENABLE_LLM_QC" != "1" ]]; then
    GEN_CMD+=(--skip-llm-qc)
fi

if [[ "$LIMIT" != "0" ]]; then
    GEN_CMD+=(--limit "$LIMIT")
fi

if [[ -n "$PERSONA" ]]; then
    GEN_CMD+=($PERSONA)
fi

"${GEN_CMD[@]}"

PASS_FILE="${OUTPUT%.jsonl}_pass.jsonl"
TOTAL=$(wc -l < "$OUTPUT" 2>/dev/null || echo 0)
PASS=$(wc -l < "$PASS_FILE" 2>/dev/null || echo 0)

echo "=========================================="
echo "Full-doc v2 config $IDX ($TAG) complete: $(date)"
echo "  Total: $TOTAL"
echo "  Pass:  $PASS"
echo "=========================================="
