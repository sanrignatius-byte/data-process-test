#!/bin/bash
#SBATCH --job-name=m2_m15_reason_prod
#SBATCH --partition=cluster02
#SBATCH --qos=msc
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=24:00:00
#SBATCH --output=logs/m2_m15_reason_prod_%j.out
#SBATCH --error=logs/m2_m15_reason_prod_%j.err

set -euo pipefail

REPO_ROOT=${REPO_ROOT:-/projects/_hdd/myyyx1/data-process-test}
CONDA_ENV=${CONDA_ENV:-/projects/myyyx1/envs/minerU}

PROVIDER=${PROVIDER:-company}
MODEL=${MODEL:-gpt-5.4}
# Defaults changed 2026-05-12: mixed + persona shown to give +18~20pt pass rate
# in the 2026-04-12 sweep (CLAUDE.md §"Phase A.1": l3_academic 26.1% →
# l3_mixed 56.2% / l3_mixed_persona 44.6%; m2_academic 39.4% → m2_mixed_persona 46.7%).
# Override with STYLE=academic USE_PERSONA=0 to reproduce the academic-only baseline.
STYLE=${STYLE:-mixed}
USE_PERSONA=${USE_PERSONA:-1}
DELAY=${DELAY:-0.5}
LIMIT=${LIMIT:-0}
FORCE_REASONING=${FORCE_REASONING:-1}

INPUT_CANDIDATES=${INPUT_CANDIDATES:-data/02_enriched/hub_candidates_v2_combined.json}
PREPARED_CANDIDATES=${PREPARED_CANDIDATES:-data/05_eval/m2/m2_m15_reasoning_path_enriched_intradoc_20260511.json}
OUT_DIR=${OUT_DIR:-data/03_queries/m2_m15_reasoning_path_prod_20260511}
OUTPUT=${OUTPUT:-${OUT_DIR}/m2_m15_reasoning_path.jsonl}

REFERENCE_GRAPH=${REFERENCE_GRAPH:-data/01_graphs/latex_reference_graph_v2.json}
TOPOLOGY_CANDIDATES=${TOPOLOGY_CANDIDATES:-data/01_graphs/latex_hub_multihop_candidates_v2.json}

cd "$REPO_ROOT"
mkdir -p logs "$OUT_DIR" "$(dirname "$PREPARED_CANDIDATES")"

source /cluster/apps/software/Miniforge3/24.11.3-1/etc/profile.d/conda.sh
conda activate "$CONDA_ENV"

if [[ -f .env ]]; then
    set -a
    source .env
    set +a
fi

if [[ "$PROVIDER" == "company" ]]; then
    if [[ -z "${COMPANY_API_URL:-}" ]]; then
        echo "ERROR: COMPANY_API_URL not set"
        exit 1
    fi
    if [[ -z "${COMPANY_API_KEY:-}" ]]; then
        echo "ERROR: COMPANY_API_KEY not set"
        exit 1
    fi
fi

if [[ "$FORCE_REASONING" == "1" && "$STYLE" == "real_user" ]]; then
    echo "ERROR: FORCE_REASONING=1 is incompatible with STYLE=real_user"
    exit 1
fi

export PYTHONPATH="$REPO_ROOT:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1

echo "=========================================="
echo "M2/M1.5 reasoning-path production"
echo "Start time:        $(date)"
echo "Host:              $(hostname)"
echo "Repo:              $REPO_ROOT"
echo "Input candidates:  $INPUT_CANDIDATES"
echo "Prepared file:     $PREPARED_CANDIDATES"
echo "Output:            $OUTPUT"
echo "Reference graph:   $REFERENCE_GRAPH"
echo "Topology cands:    $TOPOLOGY_CANDIDATES"
echo "Provider/model:    $PROVIDER / $MODEL"
echo "Style:             $STYLE"
echo "Use persona:       $USE_PERSONA"
echo "Force reasoning:   $FORCE_REASONING"
echo "Delay:             $DELAY"
echo "Limit:             $LIMIT"
echo "Expected API log users: pipeline, llm_qc_necessity, llm_qc_grounding"
echo "=========================================="

python3 -c "import inspect; import local_api_logger; import src.api.llm as llm; src=inspect.getsource(llm.call_llm); assert 'wrap_requests_call' in src; print('API logger guard: company path uses local_api_logger.wrap_requests_call')"

echo "[$(date)] Preparing strict enriched intra-doc 2/3-hop candidates..."

FILTER_CMD=(
    python scripts/filter_enriched_pair_candidates.py
    --input "$INPUT_CANDIDATES"
    --output "$PREPARED_CANDIDATES"
    --multimodal-counts 2,3
    --require-both-endpoints
    --require-all-multimodal-elements
    --require-candidate-bridge-text
    --shuffle
    --exclude-query-jsonl data/03_queries/l1_dual_evidence_queries_v3_pass.jsonl
    --exclude-query-jsonl data/03_queries/l1_dual_evidence_queries_v4_4_run1_pass.jsonl
    --exclude-query-jsonl data/05_eval/m2/level2_dual_evidence.jsonl
    --exclude-query-jsonl data/05_eval/m2/level3_reasoning_chain.jsonl
    --exclude-query-jsonl data/05_eval/m2/l2_new_batch.jsonl
    --exclude-query-jsonl data/05_eval/m2/l3_new_batch.jsonl
    --exclude-query-jsonl data/05_eval/m2/l3_reasoning_chain_queries.jsonl
)

if [[ "$FORCE_REASONING" == "1" ]]; then
    FILTER_CMD+=(--force-reasoning-chain-target)
fi
if [[ "$LIMIT" != "0" ]]; then
    FILTER_CMD+=(--limit "$LIMIT")
fi

"${FILTER_CMD[@]}"

echo "[$(date)] Prepared candidate summary:"
python3 -c "import json; d=json.load(open('$PREPARED_CANDIDATES')); print(json.dumps(d.get('summary', {}), ensure_ascii=False, indent=2)); print(json.dumps(d.get('metadata', {}).get('enriched_pair_filter', {}), ensure_ascii=False, indent=2))"

echo "[$(date)] API logger stats before generation:"
python3 -c "from pathlib import Path; users=('pipeline','llm_qc_necessity','llm_qc_grounding'); base=Path('api_logs/stats/$MODEL'); [print(f'{u}: '+str(sum(1 for _ in (base/f'{u}_2026-05.jsonl').open()) if (base/f'{u}_2026-05.jsonl').exists() else 0)) for u in users]"

GEN_CMD=(
    python scripts/generate_multihop_l1_queries.py
    --candidates "$PREPARED_CANDIDATES"
    --output "$OUTPUT"
    --pass-only
    --provider "$PROVIDER"
    --model "$MODEL"
    --query-style "$STYLE"
    --reference-graph "$REFERENCE_GRAPH"
    --topology-candidates "$TOPOLOGY_CANDIDATES"
    --skip-done "$OUTPUT"
    --shuffle
    --delay "$DELAY"
)
if [[ "$USE_PERSONA" == "1" ]]; then
    GEN_CMD+=(--use-persona)
fi

echo "[$(date)] Launching generation..."
"${GEN_CMD[@]}"

PASS_FILE="${OUTPUT%.jsonl}_pass.jsonl"
TOTAL=$(wc -l < "$OUTPUT" 2>/dev/null || echo 0)
PASS=$(wc -l < "$PASS_FILE" 2>/dev/null || echo 0)

echo "[$(date)] API logger stats after generation:"
python3 -c "from pathlib import Path; users=('pipeline','llm_qc_necessity','llm_qc_grounding'); base=Path('api_logs/stats/$MODEL'); [print(f'{u}: '+str(sum(1 for _ in (base/f'{u}_2026-05.jsonl').open()) if (base/f'{u}_2026-05.jsonl').exists() else 0)) for u in users]"

echo "=========================================="
echo "M2/M1.5 reasoning-path production complete: $(date)"
echo "  Total written: $TOTAL"
echo "  QC pass:       $PASS"
echo "  Full output:   $OUTPUT"
echo "  Pass output:   $PASS_FILE"
echo "  Candidates:    $PREPARED_CANDIDATES"
echo "=========================================="
