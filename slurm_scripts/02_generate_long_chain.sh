#!/bin/bash
#SBATCH -p cluster02
#SBATCH --job-name=long_chain_gen
#SBATCH --output=logs/long_chain_gen_%j.out
#SBATCH --error=logs/long_chain_gen_%j.err
#SBATCH --time=8:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G

# ============================================================
# Long-chain multi-hop query generation (iterative bridge-step)
#
# Two stages:
#   1. select_intra_doc_pairs.py --strategy chain → candidate pairs
#   2. generate_long_chain_iterative_queries.py    → JSONL queries
#
# Resume-safe: candidate selection is idempotent; the generation
# script supports --skip-done so a re-submitted job picks up where
# it left off.
#
# Common overrides (env vars):
#   REPO_ROOT        repo path (default /projects/myyyx1/data-process-test)
#   CONDA_ENV        conda env name (default minerU)
#   CANDIDATES       candidate pairs JSON path
#   OUTPUT           full output JSONL path
#   PASS_OUTPUT      pass-only JSONL path
#   PROVIDER         LLM provider (default company)
#   MODEL            generation model (default gpt-5.4)
#   JUDGE_MODEL      judge model (default = MODEL)
#   MIN_CHAIN_HOPS   minimum chain hops (default 3)
#   MAX_PER_DOC      max chains kept per document (default 5)
#   QUERY_STYLE      academic|real_user|mixed (default mixed)
#   USE_PERSONA      "1" to enable PersonaHub injection (default 1)
#   REFERENCE_GRAPH  latex_reference_graph.json path
#   LIMIT            cap pair count (default 0 = no cap)
#   SKIP_SELECT      "1" to reuse existing candidate file
# ============================================================

set -euo pipefail

module load Miniforge3 || true

REPO_ROOT=${REPO_ROOT:-/projects/myyyx1/data-process-test}
CONDA_ENV=${CONDA_ENV:-/projects/myyyx1/envs/minerU}

CANDIDATES=${CANDIDATES:-data/03_queries/long_chain_candidates.json}
OUTPUT=${OUTPUT:-data/03_queries/long_chain_iterative.jsonl}
PASS_OUTPUT=${PASS_OUTPUT:-data/03_queries/long_chain_iterative_pass.jsonl}

PROVIDER=${PROVIDER:-company}
MODEL=${MODEL:-gpt-5.4}
JUDGE_MODEL=${JUDGE_MODEL:-$MODEL}
MIN_CHAIN_HOPS=${MIN_CHAIN_HOPS:-3}
MAX_PER_DOC=${MAX_PER_DOC:-5}
QUERY_STYLE=${QUERY_STYLE:-mixed}
USE_PERSONA=${USE_PERSONA:-1}
REFERENCE_GRAPH=${REFERENCE_GRAPH:-data/01_graphs/latex_reference_graph.json}
TOPOLOGY_CANDIDATES=${TOPOLOGY_CANDIDATES:-data/01_graphs/latex_hub_multihop_candidates.json}
LIMIT=${LIMIT:-0}
SKIP_SELECT=${SKIP_SELECT:-0}
MAX_QUERY_HOPS=${MAX_QUERY_HOPS:-5}

cd "$REPO_ROOT"
mkdir -p logs "$(dirname "$CANDIDATES")" "$(dirname "$OUTPUT")"

source /cluster/apps/software/Miniforge3/24.11.3-1/etc/profile.d/conda.sh
conda activate "$CONDA_ENV"

# Load secrets (.env contains COMPANY_API_URL / COMPANY_API_KEY etc.)
if [[ -f .env ]]; then
    set -a
    # shellcheck disable=SC1091
    source .env
    set +a
fi

echo "=========================================="
echo "Long-chain query generation"
echo "Start time:    $(date)"
echo "Repo root:     ${REPO_ROOT}"
echo "Provider:      ${PROVIDER}"
echo "Model:         ${MODEL}"
echo "Judge model:   ${JUDGE_MODEL}"
echo "Candidates:    ${CANDIDATES}"
echo "Output:        ${OUTPUT}"
echo "Pass output:   ${PASS_OUTPUT}"
echo "Min hops:      ${MIN_CHAIN_HOPS}"
echo "Max per doc:   ${MAX_PER_DOC}"
echo "Query style:   ${QUERY_STYLE}"
echo "Use persona:   ${USE_PERSONA}"
echo "Limit:         ${LIMIT}"
echo "=========================================="

# ───────────────────────────────────────────────
# Stage 1: Select chain candidates (intra-doc only)
# ───────────────────────────────────────────────
if [[ "$SKIP_SELECT" != "1" || ! -s "$CANDIDATES" ]]; then
    echo "[Stage 1] Selecting chain candidates..."
    python scripts/select_intra_doc_pairs.py \
        --strategy chain \
        --min-chain-hops "$MIN_CHAIN_HOPS" \
        --max-per-doc "$MAX_PER_DOC" \
        --output "$CANDIDATES"
else
    echo "[Stage 1] SKIPPED (reusing $CANDIDATES)"
fi

# ───────────────────────────────────────────────
# Stage 2: Iterative long-chain generation
# ───────────────────────────────────────────────
echo "[Stage 2] Generating long-chain queries..."

GEN_CMD=(
    python scripts/generate_long_chain_iterative_queries.py
    --candidates "$CANDIDATES"
    --output "$OUTPUT"
    --pass-output "$PASS_OUTPUT"
    --provider "$PROVIDER"
    --model "$MODEL"
    --judge-model "$JUDGE_MODEL"
    --query-style "$QUERY_STYLE"
    --reference-graph "$REFERENCE_GRAPH"
    --topology-candidates "$TOPOLOGY_CANDIDATES"
    --skip-done "$OUTPUT"
)

if [[ "$USE_PERSONA" == "1" ]]; then
    GEN_CMD+=(--use-persona)
fi

if [[ "$LIMIT" -gt 0 ]]; then
    GEN_CMD+=(--limit "$LIMIT")
fi

if [[ "$MAX_QUERY_HOPS" -gt 0 ]]; then
    GEN_CMD+=(--max-query-hops "$MAX_QUERY_HOPS")
fi

"${GEN_CMD[@]}"

echo "=========================================="
echo "Long-chain generation complete: $(date)"
echo "Full output: $OUTPUT"
if [[ -s "$PASS_OUTPUT" ]]; then
    echo "Pass-only:   $PASS_OUTPUT ($(wc -l <"$PASS_OUTPUT") lines)"
fi
echo "=========================================="
