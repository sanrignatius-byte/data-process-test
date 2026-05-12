#!/bin/bash
#SBATCH --job-name=full_hub_enrich
#SBATCH --partition=cluster02
#SBATCH --qos=msc
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=logs/full_hub_enrich_%j.out
#SBATCH --error=logs/full_hub_enrich_%j.err

set -euo pipefail

REPO_ROOT=${REPO_ROOT:-/projects/_hdd/myyyx1/data-process-test}
CONDA_ENV=${CONDA_ENV:-/projects/myyyx1/envs/minerU}

HUB_CANDIDATES=${HUB_CANDIDATES:-data/01_graphs/latex_hub_multihop_candidates_v2.json}
ELEMENTS=${ELEMENTS:-data/01_graphs/multimodal_elements_v2.json}
LATEX_GRAPH=${LATEX_GRAPH:-data/01_graphs/latex_reference_graph_v2.json}
ENRICHED_ELEMENTS=${ENRICHED_ELEMENTS:-data/02_enriched/multimodal_elements_v2_production_full.json}
HUBS_TOPO=${HUBS_TOPO:-data/01_graphs/latex_graph_hubs_v2.json}
SECTION_ENRICH=${SECTION_ENRICH:-data/05_eval/m2/section_nodes_enriched_2026-03-26.json}
OUTPUT=${OUTPUT:-data/02_enriched/hub_candidates_v2_all_enriched_20260511.json}

cd "$REPO_ROOT"
mkdir -p logs "$(dirname "$OUTPUT")"

source /cluster/apps/software/Miniforge3/24.11.3-1/etc/profile.d/conda.sh
conda activate "$CONDA_ENV"

export PYTHONPATH="$REPO_ROOT:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1

echo "=========================================="
echo "Full hub-candidate enrichment"
echo "Start time:      $(date)"
echo "Host:            $(hostname)"
echo "Hub candidates:  $HUB_CANDIDATES"
echo "Elements:        $ELEMENTS"
echo "Latex graph:     $LATEX_GRAPH"
echo "Enriched elems:  $ENRICHED_ELEMENTS"
echo "Hubs topology:   $HUBS_TOPO"
echo "Section enrich:  $SECTION_ENRICH"
echo "Output:          $OUTPUT"
echo "=========================================="

python scripts/enrich_hub_candidates.py \
    --hub-candidates "$HUB_CANDIDATES" \
    --elements "$ELEMENTS" \
    --latex-graph "$LATEX_GRAPH" \
    --output "$OUTPUT" \
    --enriched-elements "$ENRICHED_ELEMENTS" \
    --hubs "$HUBS_TOPO" \
    --section-enrich "$SECTION_ENRICH"

echo "=========================================="
echo "Full hub-candidate enrichment complete: $(date)"
echo "Output: $OUTPUT"
echo "=========================================="
