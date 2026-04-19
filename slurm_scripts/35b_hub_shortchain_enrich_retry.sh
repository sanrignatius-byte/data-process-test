#!/bin/bash
#SBATCH --job-name=35b_hub_sc_retry
#SBATCH --output=logs/35b_hub_shortchain_retry_%j.out
#SBATCH --error=logs/35b_hub_shortchain_retry_%j.err
#SBATCH --partition=cluster02
#SBATCH --qos=msc
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=04:00:00

# Retry Step 2+3 for hub short-chain enrichment.
# Step 1 already succeeded (hub_candidates_v2_top25_base.json + subset JSON exist).
# Step 2 failed (all 403) because job 35 ran concurrently with job 34 (gap227 enrich).
# This job should run AFTER job 61526 (34_enrich_gap227) completes.

set -e
cd /projects/myyyx1/data-process-test
source /cluster/apps/software/Miniforge3/24.11.3-1/etc/profile.d/conda.sh
conda activate /projects/myyyx1/envs/minerU
export PYTHONPATH=/projects/myyyx1/data-process-test:$PYTHONPATH
export $(grep -v '^#' .env | xargs)

SUBSET_OUT="data/02_enriched/hub_shortchain_elements_subset.json"
ENRICH_OUT="data/02_enriched/hub_shortchain_elements_enriched.json"
FINAL_OUT="data/02_enriched/hub_candidates_v2_top25.json"

HUB_CANDS="data/01_graphs/latex_hub_multihop_candidates_v2.json"
ELEMENTS="data/02_enriched/multimodal_elements_v2_production_partial.json"
LATEX_GRAPH="data/01_graphs/latex_reference_graph_v2.json"
HUB_SCORES="data/01_graphs/hub_scores_v2.json"
HUBS_TOPO="data/01_graphs/latex_graph_hubs_v2.json"

mkdir -p logs

echo "=== Step 2 (retry): LLM enrich 759 element subset ==="
date

python scripts/enrich_elements_modora.py \
    --input    "$SUBSET_OUT" \
    --output   "$ENRICH_OUT" \
    --provider company \
    --model    gpt-5.4 \
    --delay    0.3 \
    --flush-every 100 \
    --incremental \
    --company-api-key "$COMPANY_API_KEY" \
    --company-api-url "$COMPANY_API_URL"

echo "  Element enrichment done: $(date)"

echo ""
echo "=== Step 3: enrich_hub_candidates (final, with T/M/C) ==="
date

python scripts/enrich_hub_candidates.py \
    --hub-candidates   "$HUB_CANDS" \
    --elements         "$ELEMENTS" \
    --latex-graph      "$LATEX_GRAPH" \
    --hubs             "$HUBS_TOPO" \
    --hub-scores       "$HUB_SCORES" \
    --top-ratio        0.25 \
    --enriched-elements "$ENRICH_OUT" \
    --output           "$FINAL_OUT"

echo ""
echo "=== ALL DONE: $(date) ==="
echo ""
echo "== Final hub candidate summary =="
python3 -c "
import json
d = json.load(open('$FINAL_OUT'))
pairs = d.get('pairs', [])
summ = d.get('summary', {})
print(f'Pairs: {len(pairs)}')
print(f'By type: {dict(summ.get(\"by_type\", {}))}')
print(f'By hop:  {dict(summ.get(\"by_hop\", {}))}')
print(f'Docs covered: {summ.get(\"docs_covered\")}')
enriched_pairs = sum(1 for p in pairs if p.get('element_a', {}).get('enriched_title'))
print(f'Pairs with enriched element_a: {enriched_pairs}/{len(pairs)}')
"
