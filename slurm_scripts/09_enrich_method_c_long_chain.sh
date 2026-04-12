#!/bin/bash
#SBATCH -p cluster02
#SBATCH --job-name=method_c_enrich
#SBATCH --output=logs/method_c_enrich_%j.out
#SBATCH --error=logs/method_c_enrich_%j.err
#SBATCH --time=24:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G

# ============================================================
# Method C long-chain bundle enrichment
#
# Stages:
#   1. Build long-chain bundle targets (elements + bridges)
#   2. Enrich multimodal elements with MoDora-style [T]/[M]/[C]
#   3. Enrich bridge nodes with bridge-specific summaries
#
# Resume-safe:
#   - target building is cheap/idempotent
#   - element enrichment supports --incremental + --flush-every
#   - bridge enrichment supports --incremental + --flush-every
#
# Common overrides:
#   REPO_ROOT, CONDA_ENV, PROVIDER
#   MIN_HOPS, MAX_CANDIDATES, SKIP_BUILD
#   ELEMENT_MODEL, BRIDGE_MODEL
#   ELEMENT_LIMIT, BRIDGE_LIMIT
#   ELEMENT_DELAY, BRIDGE_DELAY
#   ELEMENT_FLUSH_EVERY, BRIDGE_FLUSH_EVERY
#   NO_IMAGES=1 to disable figure image inputs
# ============================================================

set -euo pipefail

module load Miniforge3 || true

REPO_ROOT=${REPO_ROOT:-/projects/myyyx1/data-process-test}
CONDA_ENV=${CONDA_ENV:-/projects/myyyx1/envs/minerU}
PROVIDER=${PROVIDER:-company}

MIN_HOPS=${MIN_HOPS:-4}
MAX_CANDIDATES=${MAX_CANDIDATES:-0}
SKIP_BUILD=${SKIP_BUILD:-0}

ELEMENT_MODEL=${ELEMENT_MODEL:-gpt-5.4}
BRIDGE_MODEL=${BRIDGE_MODEL:-$ELEMENT_MODEL}
ELEMENT_LIMIT=${ELEMENT_LIMIT:-0}
BRIDGE_LIMIT=${BRIDGE_LIMIT:-0}
ELEMENT_DELAY=${ELEMENT_DELAY:-0.05}
BRIDGE_DELAY=${BRIDGE_DELAY:-0.05}
ELEMENT_FLUSH_EVERY=${ELEMENT_FLUSH_EVERY:-50}
BRIDGE_FLUSH_EVERY=${BRIDGE_FLUSH_EVERY:-20}
NO_IMAGES=${NO_IMAGES:-0}

LONG_CHAIN_CANDIDATES=${LONG_CHAIN_CANDIDATES:-data/01_graphs/long_chain_candidates_v2.json}
REFERENCE_GRAPH=${REFERENCE_GRAPH:-data/01_graphs/latex_reference_graph_v2.json}
MULTIMODAL_ELEMENTS=${MULTIMODAL_ELEMENTS:-data/01_graphs/multimodal_elements_v2.json}
MINERU_OUTPUT_DIR=${MINERU_OUTPUT_DIR:-data/00_raw/mineru_output}

ELEMENT_SUBSET=${ELEMENT_SUBSET:-data/02_enriched/method_c_long_chain_elements_subset_v1.json}
BRIDGE_TARGETS=${BRIDGE_TARGETS:-data/02_enriched/method_c_long_chain_bridge_targets_v1.json}
MANIFEST=${MANIFEST:-data/02_enriched/method_c_long_chain_enrich_manifest_v1.json}
ELEMENT_OUTPUT=${ELEMENT_OUTPUT:-data/02_enriched/method_c_long_chain_elements_enriched_v1.json}
BRIDGE_OUTPUT=${BRIDGE_OUTPUT:-data/02_enriched/method_c_long_chain_bridges_enriched_v1.json}

cd "$REPO_ROOT"
mkdir -p logs "$(dirname "$ELEMENT_SUBSET")" "$(dirname "$ELEMENT_OUTPUT")"

source /cluster/apps/software/Miniforge3/24.11.3-1/etc/profile.d/conda.sh
conda activate "$CONDA_ENV"

if [[ -f .env ]]; then
    set -a
    # shellcheck disable=SC1091
    source .env
    set +a
fi

if [[ "$PROVIDER" == "company" ]]; then
    if [[ -z "${COMPANY_API_KEY:-}" || -z "${COMPANY_API_URL:-}" ]]; then
        echo "ERROR: COMPANY_API_KEY / COMPANY_API_URL must be set for provider=company"
        exit 1
    fi
fi

export PYTHONUNBUFFERED=1

echo "=========================================="
echo "Method C long-chain enrichment"
echo "Start time:           $(date)"
echo "Repo root:            ${REPO_ROOT}"
echo "Provider:             ${PROVIDER}"
echo "Element model:        ${ELEMENT_MODEL}"
echo "Bridge model:         ${BRIDGE_MODEL}"
echo "Min hops:             ${MIN_HOPS}"
echo "Max candidates:       ${MAX_CANDIDATES}"
echo "Element delay:        ${ELEMENT_DELAY}"
echo "Bridge delay:         ${BRIDGE_DELAY}"
echo "Element flush every:  ${ELEMENT_FLUSH_EVERY}"
echo "Bridge flush every:   ${BRIDGE_FLUSH_EVERY}"
echo "No images:            ${NO_IMAGES}"
echo "=========================================="

if [[ "$SKIP_BUILD" != "1" || ! -s "$MANIFEST" ]]; then
    echo "[Stage 1] Building Method C long-chain targets..."
    BUILD_CMD=(
        python3 -u scripts/build_method_c_long_chain_enrich_targets.py
        --min-hops "$MIN_HOPS"
        --long-chain-candidates "$LONG_CHAIN_CANDIDATES"
        --reference-graph "$REFERENCE_GRAPH"
        --multimodal-elements "$MULTIMODAL_ELEMENTS"
        --mineru-output-dir "$MINERU_OUTPUT_DIR"
        --element-subset-output "$ELEMENT_SUBSET"
        --bridge-target-output "$BRIDGE_TARGETS"
        --manifest-output "$MANIFEST"
    )
    if [[ "$MAX_CANDIDATES" -gt 0 ]]; then
        BUILD_CMD+=(--max-candidates "$MAX_CANDIDATES")
    fi
    "${BUILD_CMD[@]}"
else
    echo "[Stage 1] SKIPPED (reusing $MANIFEST)"
fi

echo "[Stage 1.5] Manifest summary:"
MANIFEST_PATH="$MANIFEST" python3 - <<'PY'
import json
import os
from pathlib import Path
path = Path(os.environ["MANIFEST_PATH"])
data = json.loads(path.read_text(encoding="utf-8"))
summary = data.get("summary", {})
meta = data.get("metadata", {})
print(f"  selected_candidates: {meta.get('selected_candidates')}")
print(f"  docs_covered:        {meta.get('docs_covered')}")
print(f"  element_targets:     {summary.get('element_targets')}")
print(f"  bridge_targets:      {summary.get('bridge_targets')}")
print(f"  unmapped_modal:      {summary.get('unmapped_modal_labels')}")
PY

echo "[Stage 2] Enriching multimodal elements..."
ELEMENT_CMD=(
    python3 -u scripts/enrich_elements_modora.py
    --input "$ELEMENT_SUBSET"
    --output "$ELEMENT_OUTPUT"
    --provider "$PROVIDER"
    --model "$ELEMENT_MODEL"
    --delay "$ELEMENT_DELAY"
    --incremental
    --flush-every "$ELEMENT_FLUSH_EVERY"
)
if [[ "$ELEMENT_LIMIT" -gt 0 ]]; then
    ELEMENT_CMD+=(--limit "$ELEMENT_LIMIT")
fi
if [[ "$NO_IMAGES" == "1" ]]; then
    ELEMENT_CMD+=(--no-images)
fi
"${ELEMENT_CMD[@]}"

echo "[Stage 3] Enriching bridge nodes..."
BRIDGE_CMD=(
    python3 -u scripts/enrich_method_c_bridge_nodes.py
    --input "$BRIDGE_TARGETS"
    --output "$BRIDGE_OUTPUT"
    --provider "$PROVIDER"
    --model "$BRIDGE_MODEL"
    --delay "$BRIDGE_DELAY"
    --incremental
    --flush-every "$BRIDGE_FLUSH_EVERY"
)
if [[ "$BRIDGE_LIMIT" -gt 0 ]]; then
    BRIDGE_CMD+=(--limit "$BRIDGE_LIMIT")
fi
"${BRIDGE_CMD[@]}"

echo "=========================================="
echo "Method C enrichment complete: $(date)"
echo "Element output: $ELEMENT_OUTPUT"
echo "Bridge output:  $BRIDGE_OUTPUT"
ls -lh "$ELEMENT_OUTPUT" "$BRIDGE_OUTPUT" 2>/dev/null || true
echo "=========================================="
