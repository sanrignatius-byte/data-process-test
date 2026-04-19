#!/bin/bash
#SBATCH --job-name=v2_query_pipeline
#SBATCH --partition=cluster02
#SBATCH --qos=msc
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=logs/v2_query_pipeline_%j.out
#SBATCH --error=logs/v2_query_pipeline_%j.err
#SBATCH --dependency=afterok:61167

# ============================================================
# V2 Query Generation Full Pipeline
# 依赖: Job 61167 (enrich gap 227 docs) 完成后自动触发
#
# 步骤:
#   Step 1 — 合并 3 路 enriched → elements_enriched_full_1040.json
#   Step 2 — build feasibility candidates (全量 1040 docs)
#   Step 3 — generate queries (LLM, academic + persona)
#   Step 4 — QC pass 筛选 + dedup → delivery_v2.jsonl
#
# API 日志: api_logs/calls/gpt-5.4/2026-04/  (user_tag=v2_query_gen)
# ============================================================

set -euo pipefail

echo "============================================"
echo "Job: V2 Query Generation Full Pipeline"
echo "============================================"
echo "Job ID:  $SLURM_JOB_ID"
echo "Node:    $(hostname)"
echo "Start:   $(date)"
echo "============================================"

cd /projects/myyyx1/data-process-test
source /cluster/apps/software/Miniforge3/24.11.3-1/etc/profile.d/conda.sh
conda activate /projects/myyyx1/envs/minerU

mkdir -p logs data/02_enriched data/03_queries api_logs/calls api_logs/stats

# ── Load API credentials ─────────────────────────────────────────────────────
if [ -f .env ]; then
    export $(grep -v '^#' .env | xargs)
fi

if [ -z "${COMPANY_API_KEY:-}" ]; then
    echo "[ERROR] COMPANY_API_KEY not set. Abort." >&2
    exit 1
fi
if [ -z "${COMPANY_API_URL:-}" ]; then
    export COMPANY_API_URL="https://yunwu.ai/v1/chat/completions"
fi
echo "  API URL:  $COMPANY_API_URL"
echo "  API Key:  ${COMPANY_API_KEY:0:8}..."

# ── Paths ────────────────────────────────────────────────────────────────────
MERGED_ENRICHED="data/02_enriched/elements_enriched_full_1040.json"
FEASIBILITY_OUT="data/03_queries/v2_feasibility_candidates_1040.json"
GEN_ACADEMIC_OUT="data/03_queries/v2_gen_academic_pass.jsonl"
GEN_PERSONA_OUT="data/03_queries/v2_gen_mixed_persona_pass.jsonl"
DELIVERY_OUT="data/03_queries/delivery_v2_$(date +%Y-%m-%d).jsonl"
STATS_OUT="data/03_queries/delivery_v2_$(date +%Y-%m-%d)_stats.json"

# ============================================================
# STEP 1 — Merge 3 enriched sources → full 1040
# ============================================================
echo ""
echo "=== Step 1: Merge enriched sources (mc + old + gap227) ==="

python3 -c "
import json, sys

mc_path   = 'data/02_enriched/method_c_long_chain_elements_enriched_v1.json'
old_path  = 'data/01_graphs/multimodal_elements.json'
gap_path  = 'data/02_enriched/method_c_long_chain_elements_enriched_v1_gap227.json'
out_path  = '$MERGED_ENRICHED'
pg_path   = 'data/01_graphs/pruned_graph_v2.json'

print('  Loading sources...')
mc   = json.load(open(mc_path))
old  = json.load(open(old_path))
gap  = json.load(open(gap_path))
prod_docs = set(json.load(open(pg_path))['documents'].keys())

# mc: {documents: {doc_id: {elements: {eid: {...}}}}}
merged = {}
def add_source(src, label):
    docs = src.get('documents', {})
    count = 0
    for did, doc in docs.items():
        if did not in prod_docs:
            continue
        if did not in merged:
            merged[did] = doc
            count += 1
        else:
            # Merge elements: prefer existing enriched_title, fill missing
            existing_elems = merged[did].get('elements', {})
            for eid, elem in doc.get('elements', {}).items():
                if eid not in existing_elems:
                    existing_elems[eid] = elem
                elif 'enriched_title' not in existing_elems[eid] and 'enriched_title' in elem:
                    existing_elems[eid] = elem
            merged[did]['elements'] = existing_elems
    print(f'  [{label}] docs added/merged: {count}')

add_source(mc,  'mc_813')
add_source(gap, 'gap_227')

# old has different structure - may have documents key or be flat
if 'documents' in old:
    add_source(old, 'old_76')
else:
    print(f'  [old_76] skipping - no documents key')

# Stats
total_enriched = sum(
    1 for doc in merged.values()
    for e in doc.get('elements', {}).values()
    if 'enriched_title' in e
)
total_elements = sum(len(doc.get('elements', {})) for doc in merged.values())

print(f'  Merged: {len(merged)} docs, {total_elements} elements, {total_enriched} enriched')
missing = prod_docs - set(merged.keys())
print(f'  Missing prod docs: {len(missing)}')
if missing:
    print(f'  Missing: {list(missing)[:5]}')
    sys.exit(1)

out = {
    'metadata': {
        'sources': ['mc_813', 'gap_227', 'old_76'],
        'total_docs': len(merged),
        'total_elements': total_elements,
        'total_enriched': total_enriched,
        'generated_at': __import__('datetime').datetime.utcnow().isoformat() + 'Z',
    },
    'documents': merged,
}
with open(out_path, 'w') as f:
    json.dump(out, f, ensure_ascii=False)
print(f'  Saved: {out_path}')
"
echo "  Step 1 done: $(date)"

# Verify merge
MERGE_DOCS=$(python3 -c "import json; d=json.load(open('$MERGED_ENRICHED')); print(len(d['documents']))")
echo "  Merged doc count: $MERGE_DOCS"
if [ "$MERGE_DOCS" -lt 1000 ]; then
    echo "[ERROR] Merge produced < 1000 docs ($MERGE_DOCS). Abort." >&2
    exit 1
fi

# ============================================================
# STEP 2 — Build feasibility candidates (full 1040)
# ============================================================
echo ""
echo "=== Step 2: Build feasibility candidates (1040 docs, hop 3-5) ==="

python scripts/build_method_c_feasibility_candidates.py \
    --long-chain-candidates data/01_graphs/long_chain_candidates_v2.json \
    --reference-graph data/01_graphs/latex_reference_graph_v2.json \
    --multimodal-elements "$MERGED_ENRICHED" \
    --min-hop 3 \
    --max-hop 5 \
    --require-enriched-endpoints \
    --output "$FEASIBILITY_OUT"

echo "  Step 2 done: $(date)"

# Count feasibility candidates
N_CANDS=$(python3 -c "import json; d=json.load(open('$FEASIBILITY_OUT')); print(len(d.get('pairs', [])))")
echo "  Feasibility candidates: $N_CANDS"
if [ "$N_CANDS" -lt 100 ]; then
    echo "[ERROR] Too few feasibility candidates ($N_CANDS). Abort." >&2
    exit 1
fi

# ============================================================
# STEP 3a — Generate queries (academic style)
# ============================================================
echo ""
echo "=== Step 3a: Generate academic-style queries ==="

python scripts/generate_long_chain_iterative_queries.py \
    --candidates "$FEASIBILITY_OUT" \
    --output "$GEN_ACADEMIC_OUT" \
    --provider company \
    --model gpt-5.4 \
    --style academic \
    --company-api-url "$COMPANY_API_URL" \
    --company-api-key "$COMPANY_API_KEY" \
    --delay 0.4 \
    --repair-attempts 2 \
    --pass-only

echo "  Step 3a done: $(date)"
N_ACADEMIC=$(wc -l < "$GEN_ACADEMIC_OUT" 2>/dev/null || echo 0)
echo "  Academic pass lines: $N_ACADEMIC"

# ============================================================
# STEP 3b — Generate queries (real-user persona style)
# ============================================================
echo ""
echo "=== Step 3b: Generate persona-style queries ==="

python scripts/generate_long_chain_iterative_queries.py \
    --candidates "$FEASIBILITY_OUT" \
    --output "$GEN_PERSONA_OUT" \
    --provider company \
    --model gpt-5.4 \
    --style real_user \
    --company-api-url "$COMPANY_API_URL" \
    --company-api-key "$COMPANY_API_KEY" \
    --delay 0.4 \
    --repair-attempts 2 \
    --pass-only

echo "  Step 3b done: $(date)"
N_PERSONA=$(wc -l < "$GEN_PERSONA_OUT" 2>/dev/null || echo 0)
echo "  Persona pass lines: $N_PERSONA"

# ============================================================
# STEP 4 — Merge, dedup, stats → delivery_v2
# ============================================================
echo ""
echo "=== Step 4: Merge + dedup → delivery_v2 ==="

python3 -c "
import json, hashlib, sys
from pathlib import Path
from datetime import datetime

academic_path = '$GEN_ACADEMIC_OUT'
persona_path  = '$GEN_PERSONA_OUT'
out_path      = '$DELIVERY_OUT'
stats_path    = '$STATS_OUT'

def load_jsonl(p):
    items = []
    if not Path(p).exists():
        print(f'  [warn] {p} not found, skipping')
        return items
    with open(p) as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    items.append(json.loads(line))
                except Exception:
                    pass
    return items

academic = load_jsonl(academic_path)
persona  = load_jsonl(persona_path)
print(f'  Loaded: {len(academic)} academic, {len(persona)} persona')

all_items = academic + persona

# Dedup by query text hash
seen = set()
deduped = []
dupes = 0
for item in all_items:
    q = item.get('query', '') or item.get('question', '')
    h = hashlib.md5(q.lower().strip().encode()).hexdigest()
    if h in seen:
        dupes += 1
        continue
    seen.add(h)
    deduped.append(item)

print(f'  After dedup: {len(deduped)} (removed {dupes} dupes)')

# Stats
from collections import Counter
docs = set()
hops = Counter()
styles = Counter()
for item in deduped:
    doc_id = item.get('doc_id', '') or item.get('arxiv_id', '')
    if doc_id:
        docs.add(doc_id)
    hop = item.get('hop_distance', item.get('hops', 0))
    hops[hop] += 1
    style = item.get('style', item.get('query_style', 'unknown'))
    styles[style] += 1

stats = {
    'total': len(deduped),
    'unique_docs': len(docs),
    'hop_distribution': dict(hops),
    'style_distribution': dict(styles),
    'raw_before_dedup': len(all_items),
    'dupes_removed': dupes,
    'generated_at': datetime.utcnow().isoformat() + 'Z',
    'source_files': [academic_path, persona_path],
}
print(f'  Stats: {stats[\"total\"]} queries, {stats[\"unique_docs\"]} docs')
print(f'  Hop dist: {dict(hops)}')

with open(out_path, 'w') as f:
    for item in deduped:
        f.write(json.dumps(item, ensure_ascii=False) + '\n')
with open(stats_path, 'w') as f:
    json.dump(stats, f, indent=2, ensure_ascii=False)

print(f'  Saved: {out_path}')
print(f'  Stats: {stats_path}')
"

echo "  Step 4 done: $(date)"

# ============================================================
# STEP 5 — API log summary
# ============================================================
echo ""
echo "=== Step 5: API usage summary ==="

python3 -c "
import sys
sys.path.insert(0, '.')
try:
    from local_api_logger import print_stats_summary
    print_stats_summary()
except Exception as e:
    print(f'[warn] could not print stats: {e}')
"

# ============================================================
echo ""
echo "============================================"
echo "V2 Query Pipeline COMPLETE: $(date)"
echo "  Delivery: $DELIVERY_OUT"
echo "  Stats:    $STATS_OUT"
echo "  API logs: api_logs/calls/"
echo "============================================"
