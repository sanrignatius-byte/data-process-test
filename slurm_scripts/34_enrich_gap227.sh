#!/bin/bash
#SBATCH --job-name=34_enrich_gap227
#SBATCH --output=logs/34_enrich_gap227_%j.out
#SBATCH --error=logs/34_enrich_gap227_%j.err
#SBATCH --partition=cluster02
#SBATCH --qos=msc
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=06:00:00

# Enrich gap227 docs with LLM [T]/[M]/[C] descriptions (enrich_elements_modora.py)
# Input:  data/02_enriched/multimodal_elements_v2_gap227.json (227 docs, 4301 elements)
# Output: data/02_enriched/multimodal_elements_v2_gap227_enriched.json
# Final:  merge enriched fields back into multimodal_elements_v2_production_partial.json

set -e
cd /projects/myyyx1/data-process-test
source /cluster/apps/software/Miniforge3/24.11.3-1/etc/profile.d/conda.sh
conda activate /projects/myyyx1/envs/minerU
export PYTHONPATH=/projects/myyyx1/data-process-test:$PYTHONPATH
export $(grep -v '^#' .env | xargs)

GAP_INPUT="data/02_enriched/multimodal_elements_v2_gap227.json"
GAP_ENRICHED="data/02_enriched/multimodal_elements_v2_gap227_enriched.json"
PROD_PARTIAL="data/02_enriched/multimodal_elements_v2_production_partial.json"
PROD_MERGED="data/02_enriched/multimodal_elements_v2_production_full.json"

echo "=== Step 1: LLM enrich gap227 elements ==="
date

python scripts/enrich_elements_modora.py \
    --input "$GAP_INPUT" \
    --output "$GAP_ENRICHED" \
    --provider company \
    --model gpt-5.4 \
    --delay 0.2 \
    --flush-every 200 \
    --incremental \
    --company-api-key "$COMPANY_API_KEY" \
    --company-api-url "$COMPANY_API_URL"

echo ""
echo "=== Step 2: Merge enriched gap227 into production_partial ==="
date

python3 -c "
import json, sys

gap_path = '$GAP_ENRICHED'
prod_path = '$PROD_PARTIAL'
out_path = '$PROD_MERGED'

print('Loading files...')
gap = json.load(open(gap_path))
prod = json.load(open(prod_path))

gap_docs = gap['documents']
prod_docs = prod['documents']

merged_docs = 0
merged_elems = 0
missing_docs = 0

for doc_id, gap_doc in gap_docs.items():
    if doc_id not in prod_docs:
        missing_docs += 1
        continue
    prod_elems = prod_docs[doc_id].get('elements', {})
    gap_elems = gap_doc.get('elements', {})
    merged_docs += 1
    for eid, gap_elem in gap_elems.items():
        if eid not in prod_elems:
            continue
        for field in ('enriched_title', 'enriched_metadata', 'enriched_content', 'enrichment_issues'):
            if field in gap_elem:
                prod_elems[eid][field] = gap_elem[field]
                if field == 'enriched_title':
                    merged_elems += 1

print(f'Merged {merged_elems} enriched elements across {merged_docs} docs (missing: {missing_docs})')

# Update metadata
prod.setdefault('metadata', {})['enrichment_gap227'] = {
    'merged_docs': merged_docs,
    'merged_elements': merged_elems,
    'source': gap_path,
}

# Count final coverage
total_elems = 0
fully_enriched = 0
for doc in prod_docs.values():
    for e in doc.get('elements', {}).values():
        total_elems += 1
        if 'enriched_title' in e:
            fully_enriched += 1

print(f'Final coverage: {fully_enriched}/{total_elems} elements enriched ({100*fully_enriched/total_elems:.1f}%)')

with open(out_path, 'w') as f:
    json.dump(prod, f, indent=2, ensure_ascii=False)
print(f'Saved to {out_path}')
"

echo ""
echo "=== ALL DONE: $(date) ==="
echo ""
echo "== Enrichment coverage summary =="
python3 -c "
import json
data = json.load(open('$PROD_MERGED'))
docs = data['documents']
total = enriched = 0
type_enriched = {}
type_total = {}
for doc in docs.values():
    for e in doc.get('elements', {}).values():
        t = e.get('element_type','?')
        type_total[t] = type_total.get(t,0) + 1
        total += 1
        if 'enriched_title' in e:
            enriched += 1
            type_enriched[t] = type_enriched.get(t,0) + 1
print(f'Total: {enriched}/{total} ({100*enriched/total:.1f}%) enriched')
for t in sorted(type_total):
    n = type_enriched.get(t,0)
    d = type_total[t]
    print(f'  {t}: {n}/{d} ({100*n/d:.1f}%)')
"
