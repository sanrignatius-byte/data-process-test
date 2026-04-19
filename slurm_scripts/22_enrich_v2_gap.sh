#!/bin/bash
#SBATCH --job-name=enrich_v2_gap
#SBATCH --partition=cluster02
#SBATCH --qos=msc
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=08:00:00
#SBATCH --output=logs/enrich_v2_gap_%j.out
#SBATCH --error=logs/enrich_v2_gap_%j.err

echo "============================================"
echo "Job: Enrich 227 gap docs (v2 elements)"
echo "============================================"
echo "Job ID:  $SLURM_JOB_ID"
echo "Node:    $(hostname)"
echo "Start:   $(date)"
echo "============================================"

cd /projects/myyyx1/data-process-test
source /cluster/apps/software/Miniforge3/24.11.3-1/etc/profile.d/conda.sh
conda activate /projects/myyyx1/envs/minerU

mkdir -p logs data/02_enriched

GAP_INPUT="data/02_enriched/multimodal_elements_v2_gap227.json"
GAP_OUTPUT="data/02_enriched/method_c_long_chain_elements_enriched_v1_gap227.json"

# ── Step 1: 导出 227 篇缺口文档的 v2 elements ──────────────────────────────────
echo ""
echo "=== Step 1: Export 227 gap docs from v2 elements ==="
python3 -c "
import json

v2 = json.load(open('data/01_graphs/multimodal_elements_v2.json'))
mc = json.load(open('data/02_enriched/method_c_long_chain_elements_enriched_v1.json'))
pg = json.load(open('data/01_graphs/pruned_graph_v2.json'))
old = json.load(open('data/01_graphs/multimodal_elements.json'))

prod_docs = set(pg['documents'].keys())
mc_docs = set(mc['documents'].keys())
old_docs = set(old.get('documents',{}).keys())
need_enrich = prod_docs - mc_docs - old_docs
print(f'[gap] docs needing enrich: {len(need_enrich)}')

v2_docs = v2['documents']
gap_documents = {did: v2_docs[did] for did in need_enrich if did in v2_docs}

# count elements
non_text = {'figure','table','formula','equation'}
total = sum(
    1 for doc in gap_documents.values()
    for e in doc.get('elements',{}).values()
    if e.get('element_type','') in non_text
)
print(f'[gap] non-text elements to enrich: {total}')

out = {
    'metadata': {**v2.get('metadata',{}), 'gap_docs': len(gap_documents), 'purpose': 'enrich_gap_227'},
    'documents': gap_documents,
    'stats': {'num_documents': len(gap_documents), 'num_nontext_elements': total}
}
with open('$GAP_INPUT', 'w') as f:
    json.dump(out, f, ensure_ascii=False)
print(f'[gap] saved to $GAP_INPUT')
"
echo "  export done: $(date)"

# ── Step 2: 加载 .env ──────────────────────────────────────────────────────────
if [ -f .env ]; then
    export $(grep -v '^#' .env | xargs)
fi
echo "  API URL: $COMPANY_API_URL"

# ── Step 3: Wait for API to be available ──────────────────────────────────────
echo ""
echo "=== Step 2: Waiting for company API ==="
MAX_WAIT=120   # max 120 minutes
WAITED=0
while true; do
    HTTP=$(curl -s -o /dev/null -w "%{http_code}" --max-time 10 \
        -X POST "${COMPANY_API_URL}" \
        -H "Authorization: Bearer ${COMPANY_API_KEY}" \
        -H "Content-Type: application/json" \
        -d '{"model":"claude-sonnet-4-20250514","messages":[{"role":"user","content":"ping"}],"max_tokens":3}' 2>/dev/null)
    if [[ "$HTTP" == "200" ]]; then
        echo "  API healthy (HTTP $HTTP) at $(date)"
        break
    fi
    if [[ $WAITED -ge $MAX_WAIT ]]; then
        echo "  API still unavailable after ${MAX_WAIT} min, aborting."
        exit 1
    fi
    echo "  API returned $HTTP, waiting 2 min... (${WAITED}/${MAX_WAIT} min elapsed)"
    sleep 120
    WAITED=$((WAITED + 2))
done

# ── Step 4: Enrich (incremental, flush every 50) ───────────────────────────────
echo ""
echo "=== Step 3: Enrich gap docs ==="
python scripts/enrich_elements_modora.py \
    --input "$GAP_INPUT" \
    --output "$GAP_OUTPUT" \
    --provider company \
    --delay 0.3 \
    --incremental \
    --flush-every 50
echo "  enrich done: $(date)"

# ── Step 5: 验证产物 ────────────────────────────────────────────────────────────
echo ""
echo "=== Step 4: Verify output ==="
python3 -c "
import json, os
if not os.path.exists('$GAP_OUTPUT'):
    print('[error] output not found')
    exit(1)
d = json.load(open('$GAP_OUTPUT'))
docs = d.get('documents',{})
total_enriched = sum(
    1 for doc in docs.values()
    for e in doc.get('elements',{}).values()
    if 'enriched_title' in e
)
print(f'[verify] docs: {len(docs)}, enriched elements: {total_enriched}')
"

echo ""
echo "============================================"
echo "Enrich gap done: $(date)"
echo "Output: $GAP_OUTPUT"
echo "============================================"
