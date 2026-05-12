#!/bin/bash
#SBATCH --job-name=graph_bridge_enrich
#SBATCH --partition=cluster02
#SBATCH --qos=msc
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:pro6000:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=2-00:00:00
#SBATCH --output=logs/graph_bridge_enrich_%j.out
#SBATCH --error=logs/graph_bridge_enrich_%j.err

set -euo pipefail

REPO_ROOT=${REPO_ROOT:-/projects/myyyx1/data-process-test}
CONDA_ENV=${CONDA_ENV:-/projects/myyyx1/envs/minerU}

PROVIDER=${PROVIDER:-company}
MODEL=${MODEL:-gpt-5.4}
DELAY=${DELAY:-0.25}
DIRECT_FRAC=${DIRECT_FRAC:-0.05}
NEIGHBOR_CAP=${NEIGHBOR_CAP:-48}
PER_START_CAP=${PER_START_CAP:-24}
MAX_EDGES=${MAX_EDGES:-6}
FLUSH_EVERY=${FLUSH_EVERY:-50}
MAX_RETRIES=${MAX_RETRIES:-3}

LATEX_GRAPH=${LATEX_GRAPH:-data/01_graphs/latex_reference_graph_v2.json}
ELEMENTS=${ELEMENTS:-data/01_graphs/multimodal_elements_v2.json}
CITATION_GRAPH=${CITATION_GRAPH:-data/01_graphs/citation_graph.json}
MINERU_OUTPUT=${MINERU_OUTPUT:-data/00_raw/mineru_output}
EXISTING_ENRICH=${EXISTING_ENRICH:-data/02_enriched/multimodal_elements_v2_production_full.json}

BASE_PAIRS=${BASE_PAIRS:-data/02_enriched/graph_bridge_element_candidates_top_20260512_base.json}
MISSING_SUBSET=${MISSING_SUBSET:-data/02_enriched/graph_bridge_missing_elements_top_20260512.json}
MISSING_ENRICH=${MISSING_ENRICH:-data/02_enriched/graph_bridge_missing_elements_top_enriched_20260512.json}
MERGED_ENRICH=${MERGED_ENRICH:-data/02_enriched/multimodal_elements_v2_graph_bridge_enriched_20260512.json}
FINAL_PAIRS=${FINAL_PAIRS:-data/02_enriched/graph_bridge_element_candidates_top_enriched_20260512.json}
REQUIRED_LOGGER=${REQUIRED_LOGGER:-/projects/myyyx1/data-process-test/local_api_logger/__init__.py}

cd "$REPO_ROOT"
mkdir -p logs data/02_enriched

source /cluster/apps/software/Miniforge3/24.11.3-1/etc/profile.d/conda.sh
conda activate "$CONDA_ENV"

if [[ -f .env ]]; then
    set -a
    source .env
    set +a
fi

if [[ "$PROVIDER" != "company" ]]; then
    echo "ERROR: graph bridge enrichment must use PROVIDER=company for local_api_logger accounting"
    exit 1
fi
if [[ -z "${COMPANY_API_URL:-}" || -z "${COMPANY_API_KEY:-}" ]]; then
    echo "ERROR: COMPANY_API_URL / COMPANY_API_KEY not set"
    exit 1
fi

export PYTHONPATH="$REPO_ROOT:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1

echo "=========================================="
echo "Graph bridge pair + missing element enrichment"
echo "Start time:      $(date)"
echo "Host:            $(hostname)"
echo "Repo:            $REPO_ROOT"
echo "Model/provider:  $MODEL / $PROVIDER"
echo "Existing enrich: $EXISTING_ENRICH"
echo "Base pairs:      $BASE_PAIRS"
echo "Missing subset:  $MISSING_SUBSET"
echo "Missing enrich:  $MISSING_ENRICH"
echo "Merged enrich:   $MERGED_ENRICH"
echo "Final pairs:     $FINAL_PAIRS"
echo "Selection:       2elem top25%, 3elem top20%, direct bonus ${DIRECT_FRAC}"
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
    raise SystemExit(f"ERROR: imported local_api_logger is not required logger: {actual}")
src = inspect.getsource(llm.call_llm)
if "from local_api_logger import wrap_requests_call" not in src or "wrap_requests_call(" not in src:
    raise SystemExit("ERROR: company call_llm path does not use local_api_logger.wrap_requests_call")
print("API logger guard: PASS")
PY

echo "[$(date)] Step 1/4: build graph element+bridge candidates"
python3 scripts/build_graph_element_bridge_candidates.py \
    --latex-graph "$LATEX_GRAPH" \
    --elements "$ELEMENTS" \
    --citation-graph "$CITATION_GRAPH" \
    --mineru-output "$MINERU_OUTPUT" \
    --enriched-elements "$EXISTING_ENRICH" \
    --output "$BASE_PAIRS" \
    --missing-elements-output "$MISSING_SUBSET" \
    --two-frac 0.25 \
    --three-frac 0.20 \
    --direct-frac "$DIRECT_FRAC" \
    --max-edges "$MAX_EDGES" \
    --neighbor-cap "$NEIGHBOR_CAP" \
    --per-start-cap "$PER_START_CAP"

MISSING_COUNT=$(python3 -c "import json; d=json.load(open('$MISSING_SUBSET')); print(d.get('metadata',{}).get('total_elements',0))")
echo "[$(date)] Missing element enrich count: $MISSING_COUNT"

if [[ "$MISSING_COUNT" -gt 0 ]]; then
    echo "[$(date)] Step 2/4: enrich missing elements through local_api_logger"
    python3 scripts/enrich_elements_modora.py \
        --input "$MISSING_SUBSET" \
        --output "$MISSING_ENRICH" \
        --provider "$PROVIDER" \
        --model "$MODEL" \
        --delay "$DELAY" \
        --max-retries "$MAX_RETRIES" \
        --flush-every "$FLUSH_EVERY" \
        --incremental \
        --company-api-key "$COMPANY_API_KEY" \
        --company-api-url "$COMPANY_API_URL"
else
    echo "[$(date)] Step 2/4: no missing elements; reusing existing enrichment"
    cp "$MISSING_SUBSET" "$MISSING_ENRICH"
fi

echo "[$(date)] Step 3/4: merge existing + missing enrichment overlays"
python3 scripts/merge_enriched_overlays.py \
    --base "$ELEMENTS" \
    --overlay "$EXISTING_ENRICH" \
    --overlay "$MISSING_ENRICH" \
    --output "$MERGED_ENRICH"

echo "[$(date)] Step 4/4: inject enrichments into graph bridge pairs"
python3 scripts/inject_pair_enrichments.py \
    --pairs "$BASE_PAIRS" \
    --elements "$MERGED_ENRICH" \
    --output "$FINAL_PAIRS"

echo "[$(date)] Final summary:"
python3 - <<PY
import json, collections
pairs = json.load(open("$FINAL_PAIRS"))
rows = pairs.get("pairs", [])
print("pairs", len(rows))
print("summary", json.dumps(pairs.get("summary", {}), ensure_ascii=False, indent=2)[:4000])
print("by_kind", collections.Counter(r.get("candidate_kind") for r in rows))
print("by_bridge_count", collections.Counter(str(r.get("bridge_count")) for r in rows))
print("by_node_group_size", collections.Counter(len(r.get("node_group") or []) for r in rows))
print("enriched_node_group_items", sum(1 for r in rows for e in (r.get("node_group") or []) if e.get("enriched_content")))
print("total_node_group_items", sum(len(r.get("node_group") or []) for r in rows))
PY

echo "=========================================="
echo "DONE: $(date)"
echo "=========================================="
