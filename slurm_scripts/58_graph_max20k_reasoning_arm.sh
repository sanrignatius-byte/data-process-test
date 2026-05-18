#!/bin/bash
#SBATCH --job-name=graph20k_arm
#SBATCH --partition=cluster02
#SBATCH --qos=msc
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:a6000:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=3-00:00:00
#SBATCH --output=logs/graph20k_arm_%j.out
#SBATCH --error=logs/graph20k_arm_%j.err

set -euo pipefail

REPO_ROOT=${REPO_ROOT:-/projects/myyyx1/data-process-test}
CONDA_ENV=${CONDA_ENV:-/projects/myyyx1/envs/minerU}

PROVIDER=${PROVIDER:-company}
MODEL=${MODEL:-gpt-5.4}
TAG=${TAG:?Set TAG, e.g. graph_academic}
STYLE=${STYLE:?Set STYLE: academic, real_user, or mixed}
USE_PERSONA=${USE_PERSONA:-0}
DELAY=${DELAY:-0.5}
LIMIT=${LIMIT:-0}
CHECKPOINT_EVERY=${CHECKPOINT_EVERY:-1}

PREPARED_CANDIDATES=${PREPARED_CANDIDATES:-data/tmp/hub_pairs_graph_max20000_production_full_endpoint_unused_all_20260512.json}
OUT_DIR=${OUT_DIR:-data/03_queries/graph_max20k_allstyles_prod_20260512}
OUTPUT=${OUTPUT:-${OUT_DIR}/${TAG}.jsonl}

REFERENCE_GRAPH=${REFERENCE_GRAPH:-data/01_graphs/latex_reference_graph_v2.json}
TOPOLOGY_CANDIDATES=${TOPOLOGY_CANDIDATES:-data/tmp/latex_hub_multihop_candidates_v2_max20000_20260512.json}
SECTION_ENRICH=${SECTION_ENRICH:-data/05_eval/m2/section_nodes_enriched_2026-03-26.json}
REQUIRED_LOGGER=${REQUIRED_LOGGER:-/projects/myyyx1/data-process-test/local_api_logger/__init__.py}

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
    echo "ERROR: production API calls must use PROVIDER=company so every request goes through local_api_logger.wrap_requests_call"
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
echo "Graph-derived A+bridge+B reasoning arm"
echo "Start time:        $(date)"
echo "Host:              $(hostname)"
echo "Repo:              $REPO_ROOT"
echo "Candidates:        $PREPARED_CANDIDATES"
echo "Output:            $OUTPUT"
echo "Reference graph:   $REFERENCE_GRAPH"
echo "Topology cands:    $TOPOLOGY_CANDIDATES"
echo "Section enrich:    $SECTION_ENRICH"
echo "Provider/model:    $PROVIDER / $MODEL"
echo "TAG/style/persona: $TAG / $STYLE / $USE_PERSONA"
echo "Delay:             $DELAY"
echo "Limit:             $LIMIT"
echo "Checkpoint every:  $CHECKPOINT_EVERY row(s)"
echo "Required logger:   $REQUIRED_LOGGER"
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

echo "[$(date)] API logger stats before arm:"
python3 -c "from pathlib import Path; users=('pipeline','llm_qc_necessity','llm_qc_grounding'); base=Path('api_logs/stats/$MODEL'); [print(f'{u}: '+str(sum(1 for _ in (base/f'{u}_2026-05.jsonl').open()) if (base/f'{u}_2026-05.jsonl').exists() else 0)) for u in users]"

GEN_CMD=(
    python3 scripts/generate_multihop_l1_queries.py
    --candidates "$PREPARED_CANDIDATES"
    --output "$OUTPUT"
    --pass-only
    --provider "$PROVIDER"
    --model "$MODEL"
    --query-style "$STYLE"
    --reference-graph "$REFERENCE_GRAPH"
    --topology-candidates "$TOPOLOGY_CANDIDATES"
    --section-enrich "$SECTION_ENRICH"
    --skip-done "$OUTPUT"
    --checkpoint-every "$CHECKPOINT_EVERY"
    --shuffle
    --delay "$DELAY"
)
if [[ "$LIMIT" != "0" ]]; then
    GEN_CMD+=(--limit "$LIMIT")
fi
if [[ "$USE_PERSONA" == "1" ]]; then
    GEN_CMD+=(--use-persona)
fi

echo "[$(date)] Launching generation arm..."
"${GEN_CMD[@]}"

PASS_FILE="${OUTPUT%.jsonl}_pass.jsonl"
TOTAL=$(wc -l < "$OUTPUT" 2>/dev/null || echo 0)
PASS=$(wc -l < "$PASS_FILE" 2>/dev/null || echo 0)

echo "[$(date)] API logger stats after arm:"
python3 -c "from pathlib import Path; users=('pipeline','llm_qc_necessity','llm_qc_grounding'); base=Path('api_logs/stats/$MODEL'); [print(f'{u}: '+str(sum(1 for _ in (base/f'{u}_2026-05.jsonl').open()) if (base/f'{u}_2026-05.jsonl').exists() else 0)) for u in users]"

echo "[$(date)] Arm QC summary:"
python3 -c "import json, collections, sys; p=sys.argv[1]; rows=[json.loads(l) for l in open(p) if l.strip()] if __import__('pathlib').Path(p).exists() else []; print('qc_rate', (sum(bool(r.get('qc_pass')) for r in rows), '/', len(rows))); print('issues', collections.Counter(i for r in rows for i in (r.get('qc_issues') or [])).most_common(20)); print('by_type', {k:(sum(bool(r.get('qc_pass')) for r in v), len(v)) for k,v in ((k,[r for r in rows if r.get('pair_type')==k]) for k in sorted({r.get('pair_type') for r in rows}))})" "$OUTPUT"

echo "=========================================="
echo "Graph reasoning arm complete: $(date)"
echo "  TAG:         $TAG"
echo "  Total rows:  $TOTAL"
echo "  QC pass:     $PASS"
echo "  Full output: $OUTPUT"
echo "  Pass output: $PASS_FILE"
echo "=========================================="
