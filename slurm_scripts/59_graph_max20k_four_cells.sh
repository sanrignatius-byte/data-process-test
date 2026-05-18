#!/bin/bash
#SBATCH --job-name=graph20k4
#SBATCH --partition=cluster02
#SBATCH --qos=msc
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:a6000:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=3-00:00:00
#SBATCH --output=logs/graph20k4_%j.out
#SBATCH --error=logs/graph20k4_%j.err

# Single-node 4-way concurrent production, following the delivery sweep pattern:
# keep all workers on one compute node so /projects/_hdd data access is consistent.

set -uo pipefail

REPO_ROOT=${REPO_ROOT:-/projects/myyyx1/data-process-test}
CONDA_ENV=${CONDA_ENV:-/projects/myyyx1/envs/minerU}

PROVIDER=${PROVIDER:-company}
MODEL=${MODEL:-gpt-5.4}
DELAY=${DELAY:-0.3}
LIMIT=${LIMIT:-0}
CHECKPOINT_EVERY=${CHECKPOINT_EVERY:-1}

PREPARED_CANDIDATES=${PREPARED_CANDIDATES:-data/tmp/hub_pairs_graph_max20000_production_full_endpoint_unused_all_20260512.json}
OUT_DIR=${OUT_DIR:-data/03_queries/graph_max20k_four_cells_prod_20260512}
REFERENCE_GRAPH=${REFERENCE_GRAPH:-data/01_graphs/latex_reference_graph_v2.json}
TOPOLOGY_CANDIDATES=${TOPOLOGY_CANDIDATES:-data/tmp/latex_hub_multihop_candidates_v2_max20000_20260512.json}
SECTION_ENRICH=${SECTION_ENRICH:-data/05_eval/m2/section_nodes_enriched_2026-03-26.json}

cd "$REPO_ROOT"
mkdir -p logs "$OUT_DIR"

source /cluster/apps/software/Miniforge3/24.11.3-1/etc/profile.d/conda.sh
conda activate "$CONDA_ENV"

if [[ -f .env ]]; then
    set -a
    source .env
    set +a
fi

if [[ "$PROVIDER" != "company" ]]; then
    echo "ERROR: production must use PROVIDER=company so local_api_logger wraps every request"
    exit 1
fi
if [[ -z "${COMPANY_API_URL:-}" || -z "${COMPANY_API_KEY:-}" ]]; then
    echo "ERROR: COMPANY_API_URL / COMPANY_API_KEY not set"
    exit 1
fi
if [[ ! -f "$PREPARED_CANDIDATES" ]]; then
    echo "ERROR: prepared candidates not found: $PREPARED_CANDIDATES"
    exit 1
fi

export PYTHONPATH="$REPO_ROOT:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1

echo "=========================================="
echo "Graph max20k four-cell production"
echo "Start time:        $(date)"
echo "Host:              $(hostname)"
echo "Repo:              $REPO_ROOT"
echo "Candidates:        $PREPARED_CANDIDATES"
echo "Output dir:        $OUT_DIR"
echo "Provider/model:    $PROVIDER / $MODEL"
echo "Delay:             $DELAY"
echo "Limit per cell:    $LIMIT"
echo "Checkpoint every:  $CHECKPOINT_EVERY row(s)"
echo "Cells:             academic, academic+persona, real_user, real_user+persona"
echo "QC:                strict rule QC + LLM necessity + LLM grounding"
echo "=========================================="

python3 - <<'PY'
import inspect
from pathlib import Path

import local_api_logger
import src.api.llm as llm

required = Path("/projects/myyyx1/data-process-test/local_api_logger/__init__.py")
actual = Path(local_api_logger.__file__)
print(f"local_api_logger actual:   {actual}")
print(f"local_api_logger required: {required}")
if not actual.resolve().samefile(required):
    raise SystemExit(f"ERROR: imported local_api_logger is not the required logger: {actual}")
src = inspect.getsource(llm.call_llm)
if "from local_api_logger import wrap_requests_call" not in src or "wrap_requests_call(" not in src:
    raise SystemExit("ERROR: company call_llm path does not use local_api_logger.wrap_requests_call")
print("API logger guard: PASS; company provider requests are wrapped by local_api_logger")
PY

echo "[$(date)] Candidate summary:"
python3 -c "import json; d=json.load(open('$PREPARED_CANDIDATES')); print(json.dumps(d.get('summary', {}), ensure_ascii=False, indent=2)); print(json.dumps(d.get('metadata', {}).get('endpoint_pair_filter', {}), ensure_ascii=False, indent=2))"

echo "[$(date)] API logger stats before four-cell run:"
python3 -c "from pathlib import Path; users=('pipeline','llm_qc_necessity','llm_qc_grounding'); base=Path('api_logs/stats/$MODEL'); [print(f'{u}: '+str(sum(1 for _ in (base/f'{u}_2026-05.jsonl').open()) if (base/f'{u}_2026-05.jsonl').exists() else 0)) for u in users]"

declare -a CELL_CONFIG=(
    "0|graph_academic|academic|0"
    "1|graph_academic_persona|academic|1"
    "2|graph_real_user|real_user|0"
    "3|graph_real_user_persona|real_user|1"
)

run_cell() {
    local IDX=$1
    local TAG=$2
    local STYLE=$3
    local USE_PERSONA=$4
    local CELL_LOG="logs/graph20k4_cell${IDX}_${TAG}.log"
    local OUTPUT="$OUT_DIR/${TAG}.jsonl"

    {
        echo "=========================================="
        echo "Cell $IDX → $TAG"
        echo "Start:       $(date)"
        echo "Host:        $(hostname)"
        echo "Style:       $STYLE"
        echo "Persona:     $USE_PERSONA"
        echo "Output:      $OUTPUT"
        echo "Delay:       $DELAY"
        echo "Limit:       $LIMIT"
        echo "Checkpoint:  $CHECKPOINT_EVERY row(s)"
        echo "=========================================="
    } > "$CELL_LOG" 2>&1

    TAG="$TAG" \
    STYLE="$STYLE" \
    USE_PERSONA="$USE_PERSONA" \
    DELAY="$DELAY" \
    LIMIT="$LIMIT" \
    CHECKPOINT_EVERY="$CHECKPOINT_EVERY" \
    PROVIDER="$PROVIDER" \
    MODEL="$MODEL" \
    PREPARED_CANDIDATES="$PREPARED_CANDIDATES" \
    OUT_DIR="$OUT_DIR" \
    OUTPUT="$OUTPUT" \
    REFERENCE_GRAPH="$REFERENCE_GRAPH" \
    TOPOLOGY_CANDIDATES="$TOPOLOGY_CANDIDATES" \
    SECTION_ENRICH="$SECTION_ENRICH" \
    REPO_ROOT="$REPO_ROOT" \
    CONDA_ENV="$CONDA_ENV" \
        bash slurm_scripts/58_graph_max20k_reasoning_arm.sh >> "$CELL_LOG" 2>&1
    local RC=$?

    local PASS_FILE="${OUTPUT%.jsonl}_pass.jsonl"
    local TOTAL=$(wc -l < "$OUTPUT" 2>/dev/null || echo 0)
    local PASS=$(wc -l < "$PASS_FILE" 2>/dev/null || echo 0)
    {
        echo "=========================================="
        echo "Cell $IDX complete: $(date) rc=$RC"
        echo "  Tag:        $TAG"
        echo "  Total rows: $TOTAL"
        echo "  QC pass:    $PASS"
        echo "  Output:     $OUTPUT"
        echo "  Pass file:  $PASS_FILE"
        echo "=========================================="
    } >> "$CELL_LOG" 2>&1
    return $RC
}

echo "[$(date)] Launching 4 cells on host $(hostname)"
declare -a PIDS
declare -a TAGS
for SPEC in "${CELL_CONFIG[@]}"; do
    IFS='|' read -r IDX TAG STYLE USE_PERSONA <<< "$SPEC"
    TAGS[$IDX]="$TAG"
    run_cell "$IDX" "$TAG" "$STYLE" "$USE_PERSONA" &
    PIDS[$IDX]=$!
    echo "  cell $IDX started, tag=$TAG, pid=${PIDS[$IDX]}"
done

declare -a RCS
for SPEC in "${CELL_CONFIG[@]}"; do
    IFS='|' read -r IDX TAG STYLE USE_PERSONA <<< "$SPEC"
    wait "${PIDS[$IDX]}"
    RCS[$IDX]=$?
    echo "[$(date)] cell $IDX ($TAG) finished, rc=${RCS[$IDX]}"
done

echo "[$(date)] API logger stats after four-cell run:"
python3 -c "from pathlib import Path; users=('pipeline','llm_qc_necessity','llm_qc_grounding'); base=Path('api_logs/stats/$MODEL'); [print(f'{u}: '+str(sum(1 for _ in (base/f'{u}_2026-05.jsonl').open()) if (base/f'{u}_2026-05.jsonl').exists() else 0)) for u in users]"

echo ""
echo "=========================================="
echo "FOUR-CELL ROLL-UP — $(date)"
echo "=========================================="
TOTAL_PASS=0
TOTAL_ROWS=0
FAILS=0
for SPEC in "${CELL_CONFIG[@]}"; do
    IFS='|' read -r IDX TAG STYLE USE_PERSONA <<< "$SPEC"
    OUTPUT="$OUT_DIR/${TAG}.jsonl"
    PASS_FILE="${OUTPUT%.jsonl}_pass.jsonl"
    ROWS=$(wc -l < "$OUTPUT" 2>/dev/null || echo 0)
    PASS=$(wc -l < "$PASS_FILE" 2>/dev/null || echo 0)
    TOTAL_ROWS=$((TOTAL_ROWS + ROWS))
    TOTAL_PASS=$((TOTAL_PASS + PASS))
    if [[ "${RCS[$IDX]}" != "0" ]]; then
        FAILS=$((FAILS + 1))
    fi
    printf "  cell %d  %-28s  rc=%d  rows=%d  pass=%d\n" "$IDX" "$TAG" "${RCS[$IDX]}" "$ROWS" "$PASS"
done
echo "------------------------------------------"
echo "  Total rows: $TOTAL_ROWS"
echo "  Total pass: $TOTAL_PASS"
echo "  Failed cells: $FAILS"
echo "  Output dir: $OUT_DIR"
echo "=========================================="

if [[ "$FAILS" -gt 0 ]]; then
    exit 1
fi
