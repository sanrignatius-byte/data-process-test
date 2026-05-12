#!/bin/bash
#SBATCH --job-name=graph20k_reason
#SBATCH --partition=cluster02
#SBATCH --qos=msc
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=3-00:00:00
#SBATCH --output=logs/graph20k_reason_%j.out
#SBATCH --error=logs/graph20k_reason_%j.err

set -euo pipefail

REPO_ROOT=${REPO_ROOT:-/projects/_hdd/myyyx1/data-process-test}
CONDA_ENV=${CONDA_ENV:-/projects/myyyx1/envs/minerU}

PROVIDER=${PROVIDER:-company}
MODEL=${MODEL:-gpt-5.4}
DELAY=${DELAY:-0.3}
LIMIT=${LIMIT:-0}

PREPARED_CANDIDATES=${PREPARED_CANDIDATES:-data/tmp/hub_pairs_graph_max20000_production_full_strict_unused_pairid_20260512.json}
OUT_DIR=${OUT_DIR:-data/03_queries/graph_max20k_reasoning_prod_20260512}

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

if [[ ! -f "$PREPARED_CANDIDATES" ]]; then
    echo "ERROR: prepared candidates not found: $PREPARED_CANDIDATES"
    exit 1
fi

export PYTHONPATH="$REPO_ROOT:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1

echo "=========================================="
echo "Graph-derived A+bridge+B reasoning production"
echo "Start time:        $(date)"
echo "Host:              $(hostname)"
echo "Repo:              $REPO_ROOT"
echo "Candidates:        $PREPARED_CANDIDATES"
echo "Output dir:        $OUT_DIR"
echo "Reference graph:   $REFERENCE_GRAPH"
echo "Topology cands:    $TOPOLOGY_CANDIDATES"
echo "Section enrich:    $SECTION_ENRICH"
echo "Provider/model:    $PROVIDER / $MODEL"
echo "Delay:             $DELAY"
echo "Limit per arm:     $LIMIT"
echo "Arms:              academic, academic+persona, mixed, mixed+persona"
echo "=========================================="

python3 -c "import inspect; import local_api_logger; import src.api.llm as llm; src=inspect.getsource(llm.call_llm); assert 'wrap_requests_call' in src; print('API logger guard: company path uses local_api_logger.wrap_requests_call')"

echo "[$(date)] Prepared candidate summary:"
python3 -c "import json; d=json.load(open('$PREPARED_CANDIDATES')); print(json.dumps(d.get('summary', {}), ensure_ascii=False, indent=2)); print(json.dumps(d.get('metadata', {}).get('enriched_pair_filter', {}), ensure_ascii=False, indent=2))"

echo "[$(date)] API logger stats before generation:"
python3 -c "from pathlib import Path; users=('pipeline','llm_qc_necessity','llm_qc_grounding'); base=Path('api_logs/stats/$MODEL'); [print(f'{u}: '+str(sum(1 for _ in (base/f'{u}_2026-05.jsonl').open()) if (base/f'{u}_2026-05.jsonl').exists() else 0)) for u in users]"

ARMS=(
    "graph_academic:academic:0"
    "graph_academic_persona:academic:1"
    "graph_mixed:mixed:0"
    "graph_mixed_persona:mixed:1"
)

for SPEC in "${ARMS[@]}"; do
    IFS=: read -r TAG STYLE USE_PERSONA <<< "$SPEC"
    OUTPUT="$OUT_DIR/${TAG}.jsonl"
    PASS_FILE="${OUTPUT%.jsonl}_pass.jsonl"

    echo "------------------------------------------"
    echo "[$(date)] Production arm: $TAG"
    echo "Style: $STYLE"
    echo "Use persona: $USE_PERSONA"
    echo "Output: $OUTPUT"
    echo "------------------------------------------"

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
        --shuffle
        --delay "$DELAY"
    )
    if [[ "$LIMIT" != "0" ]]; then
        GEN_CMD+=(--limit "$LIMIT")
    fi
    if [[ "$USE_PERSONA" == "1" ]]; then
        GEN_CMD+=(--use-persona)
    fi

    "${GEN_CMD[@]}"

    TOTAL=$(wc -l < "$OUTPUT" 2>/dev/null || echo 0)
    PASS=$(wc -l < "$PASS_FILE" 2>/dev/null || echo 0)
    echo "[$(date)] Arm complete: $TAG total=$TOTAL pass=$PASS"
    python3 -c "import json, collections, sys; p=sys.argv[1]; rows=[json.loads(l) for l in open(p) if l.strip()]; print('qc_rate', (sum(bool(r.get('qc_pass')) for r in rows), '/', len(rows))); print('issues', collections.Counter(i for r in rows for i in (r.get('qc_issues') or [])).most_common(12)); print('by_type', {k:(sum(bool(r.get('qc_pass')) for r in v), len(v)) for k,v in ((k,[r for r in rows if r.get('pair_type')==k]) for k in sorted({r.get('pair_type') for r in rows}))})" "$OUTPUT"
done

echo "[$(date)] API logger stats after generation:"
python3 -c "from pathlib import Path; users=('pipeline','llm_qc_necessity','llm_qc_grounding'); base=Path('api_logs/stats/$MODEL'); [print(f'{u}: '+str(sum(1 for _ in (base/f'{u}_2026-05.jsonl').open()) if (base/f'{u}_2026-05.jsonl').exists() else 0)) for u in users]"

echo "[$(date)] Aggregate pass counts:"
python3 -c "from pathlib import Path; import json; out=Path('$OUT_DIR'); rows=[]; [rows.extend(json.loads(l) for l in p.open() if l.strip()) for p in sorted(out.glob('*_pass.jsonl'))]; print('pass_files', [p.name for p in sorted(out.glob('*_pass.jsonl'))]); print('total_pass_rows', len(rows)); print('unique_queries', len({r.get('query','') for r in rows})); print('unique_pair_style', len({(r.get('pair_id'), r.get('query_style'), r.get('persona_id')) for r in rows}))"

echo "=========================================="
echo "Graph reasoning production complete: $(date)"
echo "Outputs: $OUT_DIR"
echo "Candidates: $PREPARED_CANDIDATES"
echo "=========================================="
