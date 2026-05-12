#!/bin/bash
#SBATCH -p cluster02
#SBATCH --qos=msc
#SBATCH --job-name=enrich800
#SBATCH --output=logs/enrich800_%j.out
#SBATCH --error=logs/enrich800_%j.err
#SBATCH --time=12:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G

# ============================================================
# Round 2 弹药库:  enrich 800 pairs from latex_hub_multihop_candidates_v2
# (5000 raw intra-doc pairs available; 800 is enough for delivery buffer).
#
# This runs IN PARALLEL with slurm 14. Output feeds slurm 16
# (the second-round generation cell, to be created Wed if needed).
#
# Cost estimate: 800 pairs × ~$0.03/pair enrichment = ~$24
# Wall time: ~4-6h with --delay 0.2
#
# Usage:
#   sbatch slurm_scripts/15_enrich_v2_raw_800.sh
# ============================================================

set -euo pipefail

REPO_ROOT=${REPO_ROOT:-/projects/myyyx1/data-process-test}
CONDA_ENV=${CONDA_ENV:-/projects/myyyx1/envs/minerU}

LIMIT=${LIMIT:-800}
INPUT_V2="data/01_graphs/latex_hub_multihop_candidates_v2.json"
ELEMENTS="data/01_graphs/multimodal_elements_v2.json"
LATEX_GRAPH="data/01_graphs/latex_reference_graph_v2.json"
ENRICHED_ELEMENTS="data/02_enriched/multimodal_elements_enriched.json"
OUTPUT="data/02_enriched/v2_raw_enriched_800_delivery.json"

cd "$REPO_ROOT"
mkdir -p logs "$(dirname "$OUTPUT")"

source /cluster/apps/software/Miniforge3/24.11.3-1/etc/profile.d/conda.sh
conda activate "$CONDA_ENV"

if [[ -f .env ]]; then
    set -a
    source .env
    set +a
fi

export PYTHONPATH="$REPO_ROOT:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1

echo "=========================================="
echo "v2 raw enrichment → 800 pairs for Round 2"
echo "Start:    $(date)"
echo "Input:    $INPUT_V2"
echo "Output:   $OUTPUT"
echo "Limit:    $LIMIT"
echo "=========================================="

# Use elements_v2 if present, fallback to v1
if [[ ! -f "$ELEMENTS" ]]; then
    echo "WARN: $ELEMENTS missing, falling back to data/01_graphs/multimodal_elements.json"
    ELEMENTS="data/01_graphs/multimodal_elements.json"
fi

ENR_FLAG=()
if [[ -f "$ENRICHED_ELEMENTS" ]]; then
    ENR_FLAG=(--enriched-elements "$ENRICHED_ELEMENTS")
fi

python scripts/enrich_hub_candidates.py \
    --hub-candidates "$INPUT_V2" \
    --elements "$ELEMENTS" \
    --latex-graph "$LATEX_GRAPH" \
    --output "$OUTPUT" \
    --limit "$LIMIT" \
    "${ENR_FLAG[@]}"

n_pairs=$(python3 -c "import json; d=json.load(open('$OUTPUT')); print(len(d.get('pairs', d) if isinstance(d,dict) else d))")
echo "=========================================="
echo "Enrichment complete: $(date)"
echo "  Enriched pairs: $n_pairs"
echo "  Output: $OUTPUT"
echo "  → Use as input to slurm 16 (Round 2 generation)"
echo "=========================================="
