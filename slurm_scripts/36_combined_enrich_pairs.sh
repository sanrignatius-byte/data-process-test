#!/bin/bash
#SBATCH --job-name=36_enrich_pairs
#SBATCH --output=logs/36_combined_enrich_pairs_%j.out
#SBATCH --error=logs/36_combined_enrich_pairs_%j.err
#SBATCH --partition=cluster02
#SBATCH --qos=msc
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=12:00:00

# Combined pipeline (replaces 35b + new gap227 pairs):
#
# Step 1: Enrich hub_shortchain 759 elements (retry, sequential)
# Step 2: Enrich gap227 4301 elements (retry, sequential after Step 1)
# Step 3: Merge gap227 enriched -> production_full.json
# Step 4: enrich_hub_candidates for hub_shortchain top-25% -> hub_candidates_v2_top25.json
# Step 5: Pre-filter gap227 candidates (467) -> gap227_cands_raw.json
# Step 6: enrich_hub_candidates for gap227 (no top-ratio) -> hub_candidates_v2_gap227.json
# Step 7: Merge both pair sets -> hub_candidates_v2_combined.json
#
# Sequential API calls to avoid 403 concurrent rate limit.

set -e
cd /projects/myyyx1/data-process-test
source /cluster/apps/software/Miniforge3/24.11.3-1/etc/profile.d/conda.sh
conda activate /projects/myyyx1/envs/minerU
export PYTHONPATH=/projects/myyyx1/data-process-test:$PYTHONPATH
export $(grep -v '^#' .env | xargs)

# ── Paths ────────────────────────────────────────────────────────────────────
HUB_CANDS_V2="data/01_graphs/latex_hub_multihop_candidates_v2.json"
HUB_SCORES="data/01_graphs/hub_scores_v2.json"
HUBS_TOPO="data/01_graphs/latex_graph_hubs_v2.json"
LATEX_GRAPH="data/01_graphs/latex_reference_graph_v2.json"

# hub_shortchain
SC_SUBSET="data/02_enriched/hub_shortchain_elements_subset.json"
SC_ENRICHED="data/02_enriched/hub_shortchain_elements_enriched.json"

# gap227
GAP_RAW="data/02_enriched/multimodal_elements_v2_gap227.json"
GAP_ENRICHED="data/02_enriched/multimodal_elements_v2_gap227_enriched.json"
GAP_CANDS_RAW="data/02_enriched/hub_candidates_v2_gap227_raw.json"

# production elements
PROD_PARTIAL="data/02_enriched/multimodal_elements_v2_production_partial.json"
PROD_FULL="data/02_enriched/multimodal_elements_v2_production_full.json"

# outputs
OUT_TOP25="data/02_enriched/hub_candidates_v2_top25.json"
OUT_GAP227="data/02_enriched/hub_candidates_v2_gap227.json"
OUT_COMBINED="data/02_enriched/hub_candidates_v2_combined.json"

mkdir -p logs

# ── Step 1: Enrich hub_shortchain elements ───────────────────────────────────
echo "=== Step 1: LLM enrich hub_shortchain 759 elements ==="
date

python scripts/enrich_elements_modora.py \
    --input    "$SC_SUBSET" \
    --output   "$SC_ENRICHED" \
    --provider company \
    --model    gpt-5.4 \
    --delay    0.3 \
    --flush-every 50 \
    --incremental \
    --company-api-key "$COMPANY_API_KEY" \
    --company-api-url "$COMPANY_API_URL"

echo "  Step 1 done: $(date)"

# ── Step 2: Enrich gap227 elements ────────────────────────────────────────────
echo ""
echo "=== Step 2: LLM enrich gap227 4301 elements ==="
date

python scripts/enrich_elements_modora.py \
    --input    "$GAP_RAW" \
    --output   "$GAP_ENRICHED" \
    --provider company \
    --model    gpt-5.4 \
    --delay    0.2 \
    --flush-every 200 \
    --incremental \
    --company-api-key "$COMPANY_API_KEY" \
    --company-api-url "$COMPANY_API_URL"

echo "  Step 2 done: $(date)"

# ── Step 3: Merge gap227 enriched -> production_full ─────────────────────────
echo ""
echo "=== Step 3: Merge gap227 enriched into production_full ==="
date

python3 -c "
import json

gap = json.load(open('$GAP_ENRICHED'))
prod = json.load(open('$PROD_PARTIAL'))

gap_docs = gap['documents']
prod_docs = prod['documents']

merged_docs = merged_elems = missing_docs = 0
for doc_id, gap_doc in gap_docs.items():
    if doc_id not in prod_docs:
        missing_docs += 1
        continue
    prod_elems = prod_docs[doc_id].get('elements', {})
    merged_docs += 1
    for eid, gap_elem in gap_doc.get('elements', {}).items():
        if eid not in prod_elems:
            continue
        for field in ('enriched_title', 'enriched_metadata', 'enriched_content', 'enrichment_issues'):
            if field in gap_elem:
                prod_elems[eid][field] = gap_elem[field]
                if field == 'enriched_title':
                    merged_elems += 1

print(f'Merged {merged_elems} enriched elements across {merged_docs} docs (missing: {missing_docs})')

total_elems = fully_enriched = 0
for doc in prod_docs.values():
    for e in doc.get('elements', {}).values():
        total_elems += 1
        if 'enriched_title' in e:
            fully_enriched += 1

print(f'Final coverage: {fully_enriched}/{total_elems} ({100*fully_enriched/total_elems:.1f}%) enriched')
prod.setdefault('metadata', {})['enrichment_gap227'] = {
    'merged_docs': merged_docs, 'merged_elements': merged_elems, 'source': '$GAP_ENRICHED'
}
with open('$PROD_FULL', 'w') as f:
    json.dump(prod, f, ensure_ascii=False)
print(f'Saved to $PROD_FULL')
"

echo "  Step 3 done: $(date)"

# ── Step 4: hub_shortchain pairs (top-25%, production_full elements) ──────────
echo ""
echo "=== Step 4: enrich_hub_candidates -> hub_candidates_v2_top25.json ==="
date

python scripts/enrich_hub_candidates.py \
    --hub-candidates    "$HUB_CANDS_V2" \
    --elements          "$PROD_FULL" \
    --latex-graph       "$LATEX_GRAPH" \
    --hubs              "$HUBS_TOPO" \
    --hub-scores        "$HUB_SCORES" \
    --top-ratio         0.25 \
    --enriched-elements "$SC_ENRICHED" \
    --output            "$OUT_TOP25"

echo "  Step 4 done: $(date)"

# ── Step 5: Pre-filter gap227 candidates ─────────────────────────────────────
echo ""
echo "=== Step 5: Pre-filter gap227 candidates ==="
date

python3 -c "
import json

gap_meta = json.load(open('$GAP_RAW'))
gap_docs = set(gap_meta['documents'].keys())

cands_raw = json.load(open('$HUB_CANDS_V2'))
items = cands_raw if isinstance(cands_raw, list) else cands_raw.get('candidates', cands_raw.get('pairs', []))

gap_items = [c for c in items if c.get('doc_id') in gap_docs]
print(f'Gap227 candidates: {len(gap_items)} / {len(items)} total')

# Wrap in same format as source
out = {'candidates': gap_items, 'metadata': {'source': 'gap227_filter', 'num_docs': len(gap_docs)}}
with open('$GAP_CANDS_RAW', 'w') as f:
    json.dump(out, f, ensure_ascii=False)
print(f'Saved to $GAP_CANDS_RAW')
"

echo "  Step 5 done: $(date)"

# ── Step 6: gap227 pairs (all candidates, no top-ratio) ─────────────────────
echo ""
echo "=== Step 6: enrich_hub_candidates for gap227 -> hub_candidates_v2_gap227.json ==="
date

python scripts/enrich_hub_candidates.py \
    --hub-candidates    "$GAP_CANDS_RAW" \
    --elements          "$PROD_FULL" \
    --latex-graph       "$LATEX_GRAPH" \
    --hubs              "$HUBS_TOPO" \
    --enriched-elements "$GAP_ENRICHED" \
    --output            "$OUT_GAP227"

echo "  Step 6 done: $(date)"

# ── Step 7: Merge hub_top25 + gap227 -> combined ─────────────────────────────
echo ""
echo "=== Step 7: Merge pairs -> hub_candidates_v2_combined.json ==="
date

python3 -c "
import json

top25 = json.load(open('$OUT_TOP25'))
gap227 = json.load(open('$OUT_GAP227'))

pairs_a = top25.get('pairs', [])
pairs_b = gap227.get('pairs', [])

# Deduplicate by pair_id
seen = set()
merged_pairs = []
for p in pairs_a + pairs_b:
    pid = p.get('pair_id', '')
    if pid and pid in seen:
        continue
    seen.add(pid)
    merged_pairs.append(p)

by_type = {}
by_hop = {}
docs = set()
for p in merged_pairs:
    pt = p.get('pair_type', '?')
    by_type[pt] = by_type.get(pt, 0) + 1
    h = str(p.get('hop_distance', '?'))
    by_hop[h] = by_hop.get(h, 0) + 1
    docs.add(p.get('doc_id', ''))

out = {
    'metadata': {'source': 'hub_top25 + gap227', 'total_pairs': len(merged_pairs)},
    'summary': {'by_type': by_type, 'by_hop': by_hop, 'docs_covered': len(docs)},
    'pairs': merged_pairs,
}
with open('$OUT_COMBINED', 'w') as f:
    json.dump(out, f, ensure_ascii=False)

enriched = sum(1 for p in merged_pairs if p.get('element_a', {}).get('enriched_title'))
print(f'Combined pairs: {len(merged_pairs)} (from {len(pairs_a)} + {len(pairs_b)}, dedup by pair_id)')
print(f'By type: {by_type}')
print(f'By hop:  {by_hop}')
print(f'Docs covered: {len(docs)}')
print(f'Pairs with enriched element_a: {enriched}/{len(merged_pairs)}')
print(f'Saved to $OUT_COMBINED')
"

echo ""
echo "=== ALL DONE: $(date) ==="
