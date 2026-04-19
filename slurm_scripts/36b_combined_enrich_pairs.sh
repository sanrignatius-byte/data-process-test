#!/bin/bash
#SBATCH --job-name=36b_enrich_pairs
#SBATCH --output=logs/36b_combined_enrich_pairs_%j.out
#SBATCH --error=logs/36b_combined_enrich_pairs_%j.err
#SBATCH --partition=cluster02
#SBATCH --qos=msc
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=12:00:00

# Fixed version of 36 (three bugs fixed vs original):
#   FIX-1: Coverage gate after Step 1 + Step 2 — exit 1 if enriched count below threshold.
#           Prevents "script succeeds but enriched=0" false positive when all API calls 403.
#   FIX-2: Step 7 dedup by (doc_id, sorted endpoint pair) instead of pair_id.
#           pair_id is a doc_id+counter that resets per run — not a stable semantic key.
#   FIX-3: Step 7 preserves adjacent_bridge_elements + adjacent_bridge_adjacency from both
#           source files, so the combined pack is a valid drop-in for graph rerank consumers.
#
# Sequential API calls (Step 1 then Step 2) to avoid 403 concurrent rate limit.

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

SC_SUBSET="data/02_enriched/hub_shortchain_elements_subset.json"
SC_ENRICHED="data/02_enriched/hub_shortchain_elements_enriched.json"

GAP_RAW="data/02_enriched/multimodal_elements_v2_gap227.json"
GAP_ENRICHED="data/02_enriched/multimodal_elements_v2_gap227_enriched.json"
GAP_CANDS_RAW="data/02_enriched/hub_candidates_v2_gap227_raw.json"

PROD_PARTIAL="data/02_enriched/multimodal_elements_v2_production_partial.json"
PROD_FULL="data/02_enriched/multimodal_elements_v2_production_full.json"

OUT_TOP25="data/02_enriched/hub_candidates_v2_top25.json"
OUT_GAP227="data/02_enriched/hub_candidates_v2_gap227.json"
OUT_COMBINED="data/02_enriched/hub_candidates_v2_combined.json"

# ── Coverage gate thresholds (catch "all-403" failure mode) ──────────────────
SC_MIN_ENRICHED=100     # out of 759  (~13%)
GAP_MIN_ENRICHED=500    # out of 4301 (~12%)

mkdir -p logs

# ── Helper: count enriched_title in a flat or nested elements JSON ────────────
count_enriched() {
    python3 -c "
import json, sys
data = json.load(open(sys.argv[1]))
docs = data.get('documents', {})
n = sum(1 for doc in docs.values() for e in doc.get('elements', {}).values() if 'enriched_title' in e)
print(n)
" "$1"
}

# ── Step 1: Enrich hub_shortchain elements ────────────────────────────────────
echo "=== Step 1: LLM enrich hub_shortchain 759 elements ==="
date

python scripts/enrich_elements_modora.py \
    --input    "$SC_SUBSET" \
    --output   "$SC_ENRICHED" \
    --provider company \
    --model    gpt-5.4 \
    --delay    0.5 \
    --max-retries 3 \
    --flush-every 50 \
    --incremental \
    --company-api-key "$COMPANY_API_KEY" \
    --company-api-url "$COMPANY_API_URL"

# FIX-1: coverage gate
SC_COUNT=$(count_enriched "$SC_ENRICHED")
echo "  hub_shortchain enriched: ${SC_COUNT}/759"
if [ "$SC_COUNT" -lt "$SC_MIN_ENRICHED" ]; then
    echo "ERROR: hub_shortchain enriched ${SC_COUNT} < threshold ${SC_MIN_ENRICHED}. API likely 403. Aborting." >&2
    exit 1
fi
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
    --delay    0.5 \
    --max-retries 3 \
    --flush-every 200 \
    --incremental \
    --company-api-key "$COMPANY_API_KEY" \
    --company-api-url "$COMPANY_API_URL"

# FIX-1: coverage gate
GAP_COUNT=$(count_enriched "$GAP_ENRICHED")
echo "  gap227 enriched: ${GAP_COUNT}/4301"
if [ "$GAP_COUNT" -lt "$GAP_MIN_ENRICHED" ]; then
    echo "ERROR: gap227 enriched ${GAP_COUNT} < threshold ${GAP_MIN_ENRICHED}. API likely 403. Aborting." >&2
    exit 1
fi
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

total_elems = fully_enriched = 0
for doc in prod_docs.values():
    for e in doc.get('elements', {}).values():
        total_elems += 1
        if 'enriched_title' in e:
            fully_enriched += 1

print(f'Merged {merged_elems} enriched elements across {merged_docs} docs (missing: {missing_docs})')
print(f'Final coverage: {fully_enriched}/{total_elems} ({100*fully_enriched/total_elems:.1f}%) enriched')

prod.setdefault('metadata', {})['enrichment_gap227'] = {
    'merged_docs': merged_docs, 'merged_elements': merged_elems, 'source': '$GAP_ENRICHED'
}
with open('$PROD_FULL', 'w') as f:
    json.dump(prod, f, ensure_ascii=False)
print('Saved to $PROD_FULL')
"

echo "  Step 3 done: $(date)"

# ── Step 4: hub_shortchain pairs (top-25%) ────────────────────────────────────
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
items = cands_raw.get('candidates', [])

gap_items = [c for c in items if c.get('doc_id') in gap_docs]
print(f'Gap227 candidates: {len(gap_items)} / {len(items)} total')

out = {'candidates': gap_items, 'metadata': {'source': 'gap227_filter', 'num_docs': len(gap_docs)}}
with open('$GAP_CANDS_RAW', 'w') as f:
    json.dump(out, f, ensure_ascii=False)
print('Saved to $GAP_CANDS_RAW')
"

echo "  Step 5 done: $(date)"

# ── Step 6: gap227 pairs (all candidates, no top-ratio) ──────────────────────
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

# ── Step 7: Merge top25 + gap227 -> combined ─────────────────────────────────
echo ""
echo "=== Step 7: Merge pairs -> hub_candidates_v2_combined.json ==="
date

python3 -c "
import json

top25  = json.load(open('$OUT_TOP25'))
gap227 = json.load(open('$OUT_GAP227'))

pairs_a = top25.get('pairs', [])
pairs_b = gap227.get('pairs', [])

# FIX-2: dedup by (doc_id, frozenset of endpoint ids) — stable across runs.
# frozenset so order of a/b doesn't matter (pair AB == pair BA).
seen = set()
merged_pairs = []
for p in pairs_a + pairs_b:
    key = (
        p.get('doc_id', ''),
        frozenset([p.get('element_a_id', ''), p.get('element_b_id', '')]),
    )
    if key in seen:
        continue
    seen.add(key)
    merged_pairs.append(p)

by_type = {}
by_hop  = {}
docs    = set()
for p in merged_pairs:
    by_type[p.get('pair_type', '?')] = by_type.get(p.get('pair_type', '?'), 0) + 1
    h = str(p.get('hop_distance', '?'))
    by_hop[h] = by_hop.get(h, 0) + 1
    docs.add(p.get('doc_id', ''))

# FIX-3: preserve adjacent_bridge_elements (dict merge) and
# adjacent_bridge_adjacency (list concat, dedup by element_id key).
def merge_bridge_elements(a, b):
    merged = dict(a)
    merged.update(b)
    return merged

def merge_bridge_adjacency(a, b):
    # adjacency is a list of dicts; dedup by element_id
    seen_ids = set()
    out = []
    for item in a + b:
        eid = item.get('element_id', id(item))
        if eid not in seen_ids:
            seen_ids.add(eid)
            out.append(item)
    return out

adj_elems = merge_bridge_elements(
    top25.get('adjacent_bridge_elements', {}),
    gap227.get('adjacent_bridge_elements', {}),
)
adj_adj = merge_bridge_adjacency(
    top25.get('adjacent_bridge_adjacency', []),
    gap227.get('adjacent_bridge_adjacency', []),
)

out = {
    'metadata': {
        'source': 'hub_top25 + gap227',
        'total_pairs': len(merged_pairs),
        'pairs_from_top25': len(pairs_a),
        'pairs_from_gap227': len(pairs_b),
    },
    'summary': {'by_type': by_type, 'by_hop': by_hop, 'docs_covered': len(docs)},
    'pairs': merged_pairs,
    'adjacent_bridge_elements': adj_elems,
    'adjacent_bridge_adjacency': adj_adj,
}
with open('$OUT_COMBINED', 'w') as f:
    json.dump(out, f, ensure_ascii=False)

enriched = sum(1 for p in merged_pairs if p.get('element_a', {}).get('enriched_title'))
dup_dropped = len(pairs_a) + len(pairs_b) - len(merged_pairs)
print(f'Combined pairs: {len(merged_pairs)} ({dup_dropped} duplicates dropped)')
print(f'By type: {by_type}')
print(f'By hop:  {by_hop}')
print(f'Docs covered: {len(docs)}')
print(f'Pairs with enriched element_a: {enriched}/{len(merged_pairs)}')
print(f'adjacent_bridge_elements: {len(adj_elems)}')
print(f'adjacent_bridge_adjacency: {len(adj_adj)}')
print('Saved to $OUT_COMBINED')
"

echo ""
echo "=== ALL DONE: $(date) ==="
