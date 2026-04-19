#!/bin/bash
#SBATCH --job-name=35_hub_sc_enrich
#SBATCH --output=logs/35_hub_shortchain_enrich_%j.out
#SBATCH --error=logs/35_hub_shortchain_enrich_%j.err
#SBATCH --partition=cluster02
#SBATCH --qos=msc
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=08:00:00

# Hub short-chain enrichment pipeline for 1040-doc corpus (2-3 hop, top 25%)
#
# Step 1: enrich_hub_candidates (no enriched-elements) → map IDs, see coverage
# Step 2: extract elements needing enrich, run enrich_elements_modora
# Step 3: enrich_hub_candidates again with --enriched-elements → final output
#
# Inputs:
#   data/01_graphs/latex_hub_multihop_candidates_v2.json  (5000 candidates)
#   data/02_enriched/multimodal_elements_v2_production_partial.json (1040 docs)
#   data/01_graphs/latex_reference_graph_v2.json
#   data/01_graphs/hub_scores_v2.json
#   data/01_graphs/latex_graph_hubs_v2.json
# Outputs:
#   data/02_enriched/hub_candidates_v2_top25_base.json    (step 1 — no T/M/C)
#   data/02_enriched/hub_shortchain_elements_subset.json  (element subset for enrich)
#   data/02_enriched/hub_shortchain_elements_enriched.json (step 2 — T/M/C added)
#   data/02_enriched/hub_candidates_v2_top25.json         (step 3 — final)

set -e
cd /projects/myyyx1/data-process-test
source /cluster/apps/software/Miniforge3/24.11.3-1/etc/profile.d/conda.sh
conda activate /projects/myyyx1/envs/minerU
export PYTHONPATH=/projects/myyyx1/data-process-test:$PYTHONPATH
export $(grep -v '^#' .env | xargs)

HUB_CANDS="data/01_graphs/latex_hub_multihop_candidates_v2.json"
ELEMENTS="data/02_enriched/multimodal_elements_v2_production_partial.json"
LATEX_GRAPH="data/01_graphs/latex_reference_graph_v2.json"
HUB_SCORES="data/01_graphs/hub_scores_v2.json"
HUBS_TOPO="data/01_graphs/latex_graph_hubs_v2.json"

BASE_OUT="data/02_enriched/hub_candidates_v2_top25_base.json"
SUBSET_OUT="data/02_enriched/hub_shortchain_elements_subset.json"
ENRICH_OUT="data/02_enriched/hub_shortchain_elements_enriched.json"
FINAL_OUT="data/02_enriched/hub_candidates_v2_top25.json"

mkdir -p logs

# ============================================================
# STEP 1: Map IDs + attach element details (no T/M/C yet)
# ============================================================
echo "=== Step 1: enrich_hub_candidates (base, top 25%) ==="
date

python scripts/enrich_hub_candidates.py \
    --hub-candidates "$HUB_CANDS" \
    --elements       "$ELEMENTS" \
    --latex-graph    "$LATEX_GRAPH" \
    --hubs           "$HUBS_TOPO" \
    --hub-scores     "$HUB_SCORES" \
    --top-ratio      0.25 \
    --output         "$BASE_OUT"

echo "  Step 1 done: $(date)"

# ============================================================
# STEP 2: Extract elements needing T/M/C enrichment, enrich them
# ============================================================
echo ""
echo "=== Step 2: Extract element subset + LLM enrich ==="
date

# Extract unique element_ids from the mapped pairs, skipping already-enriched
python3 -c "
import json

base = json.load(open('$BASE_OUT'))
prod = json.load(open('$ELEMENTS'))

# Build set of already-enriched element_ids
already_enriched = {
    eid
    for doc in prod['documents'].values()
    for eid, e in doc.get('elements', {}).items()
    if 'enriched_title' in e
}
print(f'Already enriched: {len(already_enriched)}')

# Collect element_ids from pairs
pairs = base.get('pairs', [])
needed_ids = set()
for p in pairs:
    for side in ('element_a', 'element_b'):
        e = p.get(side, {})
        eid = e.get('element_id', '')
        if eid and eid not in already_enriched:
            needed_ids.add(eid)

print(f'Pairs: {len(pairs)}, new element_ids needed: {len(needed_ids)}')

# Build subset file in enrich_elements_modora format
out_docs = {}
for doc_id, doc in prod['documents'].items():
    elems = {eid: e for eid, e in doc.get('elements', {}).items() if eid in needed_ids}
    if not elems:
        continue
    out_docs[doc_id] = {
        'doc_id': doc_id,
        'num_elements': len(elems),
        'num_edges': 0,
        'elements': elems,
        'edges': [],
        'multimodal_pairs': [],
    }

subset = {
    'metadata': {
        'source': 'hub_shortchain_top25_element_subset',
        'total_docs': len(out_docs),
        'total_elements': sum(len(d['elements']) for d in out_docs.values()),
        'purpose': '2-3hop top25 hub pair element enrichment',
    },
    'documents': out_docs,
}
json.dump(subset, open('$SUBSET_OUT', 'w'), indent=2, ensure_ascii=False)
print(f'Subset written: {len(out_docs)} docs, {subset[\"metadata\"][\"total_elements\"]} elements')
"

echo "  Subset built: $(date)"

# Run LLM enrichment on the subset
python scripts/enrich_elements_modora.py \
    --input    "$SUBSET_OUT" \
    --output   "$ENRICH_OUT" \
    --provider company \
    --model    gpt-5.4 \
    --delay    0.5 \
    --max-retries 3 \
    --flush-every 200 \
    --incremental \
    --company-api-key "$COMPANY_API_KEY" \
    --company-api-url "$COMPANY_API_URL"

echo "  Element enrichment done: $(date)"

# ============================================================
# STEP 3: Re-run enrich_hub_candidates with enriched elements
# ============================================================
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
