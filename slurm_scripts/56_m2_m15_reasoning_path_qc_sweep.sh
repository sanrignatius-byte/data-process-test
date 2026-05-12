#!/bin/bash
#SBATCH --job-name=m2_m15_sweep_full
#SBATCH --partition=cluster02
#SBATCH --qos=msc
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=1-00:00:00
#SBATCH --output=logs/m2_m15_sweep_full_%j.out
#SBATCH --error=logs/m2_m15_sweep_full_%j.err

set -euo pipefail

REPO_ROOT=${REPO_ROOT:-/projects/_hdd/myyyx1/data-process-test}
CONDA_ENV=${CONDA_ENV:-/projects/myyyx1/envs/minerU}

PROVIDER=${PROVIDER:-company}
MODEL=${MODEL:-gpt-5.4}
DELAY=${DELAY:-0.5}
SWEEP_LIMIT=${SWEEP_LIMIT:-120}
RUN_FULL_ENRICH_AFTER=${RUN_FULL_ENRICH_AFTER:-1}

INPUT_CANDIDATES=${INPUT_CANDIDATES:-data/02_enriched/hub_candidates_v2_combined.json}
PREPARED_CANDIDATES=${PREPARED_CANDIDATES:-data/05_eval/m2/m2_m15_reasoning_path_enriched_intradoc_20260511_sweep.json}
OUT_DIR=${OUT_DIR:-data/03_queries/m2_m15_reasoning_path_sweep_20260512}
FULL_ENRICH_OUTPUT=${FULL_ENRICH_OUTPUT:-data/02_enriched/hub_candidates_v2_all_enriched_20260511.json}

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

export PYTHONPATH="$REPO_ROOT:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1

echo "=========================================="
echo "M2/M1.5 reasoning-path production sweep"
echo "Start time:        $(date)"
echo "Host:              $(hostname)"
echo "Repo:              $REPO_ROOT"
echo "Input candidates:  $INPUT_CANDIDATES"
echo "Prepared file:     $PREPARED_CANDIDATES"
echo "Output dir:        $OUT_DIR"
echo "Reference graph:   $REFERENCE_GRAPH"
echo "Topology cands:    $TOPOLOGY_CANDIDATES"
echo "Provider/model:    $PROVIDER / $MODEL"
echo "Sweep limit/arm:   $SWEEP_LIMIT"
echo "Delay:             $DELAY"
echo "Arms:              academic, academic+persona, mixed, mixed+persona, real_user, real_user+persona"
echo "Run full enrich:   $RUN_FULL_ENRICH_AFTER"
echo "Full enrich out:   $FULL_ENRICH_OUTPUT"
echo "Expected API log users: pipeline, llm_qc_necessity, llm_qc_grounding"
echo "=========================================="

python3 -c "import inspect; import local_api_logger; import src.api.llm as llm; src=inspect.getsource(llm.call_llm); assert 'wrap_requests_call' in src; print('API logger guard: company path uses local_api_logger.wrap_requests_call')"

echo "[$(date)] Preparing strict enriched intra-doc 2/3-hop candidates for sweep..."
python scripts/filter_enriched_pair_candidates.py \
    --input "$INPUT_CANDIDATES" \
    --output "$PREPARED_CANDIDATES" \
    --multimodal-counts 2,3 \
    --require-both-endpoints \
    --require-all-multimodal-elements \
    --require-candidate-bridge-text \
    --force-reasoning-chain-target \
    --shuffle \
    --exclude-query-jsonl data/03_queries/l1_dual_evidence_queries_v3_pass.jsonl \
    --exclude-query-jsonl data/03_queries/l1_dual_evidence_queries_v4_4_run1_pass.jsonl \
    --exclude-query-jsonl data/05_eval/m2/level2_dual_evidence.jsonl \
    --exclude-query-jsonl data/05_eval/m2/level3_reasoning_chain.jsonl \
    --exclude-query-jsonl data/05_eval/m2/l2_new_batch.jsonl \
    --exclude-query-jsonl data/05_eval/m2/l3_new_batch.jsonl \
    --exclude-query-jsonl data/05_eval/m2/l3_reasoning_chain_queries.jsonl

echo "[$(date)] Prepared candidate summary:"
python3 -c "import json; d=json.load(open('$PREPARED_CANDIDATES')); print(json.dumps(d.get('summary', {}), ensure_ascii=False, indent=2)); print(json.dumps(d.get('metadata', {}).get('enriched_pair_filter', {}), ensure_ascii=False, indent=2))"

echo "[$(date)] API logger stats before sweep:"
python3 -c "from pathlib import Path; users=('pipeline','llm_qc_necessity','llm_qc_grounding'); base=Path('api_logs/stats/$MODEL'); [print(f'{u}: '+str(sum(1 for _ in (base/f'{u}_2026-05.jsonl').open()) if (base/f'{u}_2026-05.jsonl').exists() else 0)) for u in users]"

ARMS=(
    "reason_academic:academic:0"
    "reason_academic_persona:academic:1"
    "reason_mixed:mixed:0"
    "reason_mixed_persona:mixed:1"
    "reason_real_user:real_user:0"
    "reason_real_user_persona:real_user:1"
)

for SPEC in "${ARMS[@]}"; do
    IFS=: read -r TAG STYLE USE_PERSONA <<< "$SPEC"
    OUTPUT="$OUT_DIR/${TAG}.jsonl"
    PASS_FILE="${OUTPUT%.jsonl}_pass.jsonl"

    echo "------------------------------------------"
    echo "[$(date)] Sweep arm: $TAG"
    echo "Style: $STYLE"
    echo "Use persona: $USE_PERSONA"
    echo "Output: $OUTPUT"
    echo "------------------------------------------"

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
        --limit "$SWEEP_LIMIT"
        --delay "$DELAY"
    )
    if [[ "$USE_PERSONA" == "1" ]]; then
        GEN_CMD+=(--use-persona)
    fi

    "${GEN_CMD[@]}"

    TOTAL=$(wc -l < "$OUTPUT" 2>/dev/null || echo 0)
    PASS=$(wc -l < "$PASS_FILE" 2>/dev/null || echo 0)
    echo "[$(date)] Arm complete: $TAG total=$TOTAL pass=$PASS"
    python3 -c "import json, collections, sys; p=sys.argv[1]; rows=[json.loads(l) for l in open(p) if l.strip()]; print('qc_rate', (sum(bool(r.get('qc_pass')) for r in rows), '/', len(rows))); print('issues', collections.Counter(i for r in rows for i in (r.get('qc_issues') or [])).most_common(12)); print('by_type', {k:(sum(bool(r.get('qc_pass')) for r in v), len(v)) for k,v in ((k,[r for r in rows if r.get('pair_type')==k]) for k in sorted({r.get('pair_type') for r in rows}))})" "$OUTPUT"
done

echo "[$(date)] API logger stats after sweep:"
python3 -c "from pathlib import Path; users=('pipeline','llm_qc_necessity','llm_qc_grounding'); base=Path('api_logs/stats/$MODEL'); [print(f'{u}: '+str(sum(1 for _ in (base/f'{u}_2026-05.jsonl').open()) if (base/f'{u}_2026-05.jsonl').exists() else 0)) for u in users]"

echo "=========================================="
echo "M2/M1.5 reasoning-path production sweep complete: $(date)"
echo "Outputs: $OUT_DIR"
echo "Prepared candidates: $PREPARED_CANDIDATES"
echo "=========================================="

if [[ "$RUN_FULL_ENRICH_AFTER" == "1" ]]; then
    echo "=========================================="
    echo "Starting full hub-candidate enrichment after sweep: $(date)"
    echo "Full enrich output: $FULL_ENRICH_OUTPUT"
    echo "=========================================="
    OUTPUT="$FULL_ENRICH_OUTPUT" bash slurm_scripts/55_full_hub_enrich_after_mixed.sh
fi
