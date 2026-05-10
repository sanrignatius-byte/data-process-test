#!/bin/bash
#SBATCH --job-name=chunk_consistent
#SBATCH --partition=cluster02
#SBATCH --qos=msc
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:a6000:1
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --output=logs/37f_chunk_consistent_%j.out
#SBATCH --error=logs/37f_chunk_consistent_%j.err

# ============================================================
# 37f_chunk_corpus_eval_consistent.sh
#
# P1 fix: 确保 build_chunk_corpus (eval qrels) 和
# eval_chunk_graph_rerank (chunk_contains_element edges)
# 使用完全同一次 chunk build，消除 chunk-id 命名空间不一致。
#
# 根因：之前 eval qrels 用 paragraph_chunks_n400.json (v1 build,
# 无 element_ids)，rerank 用独立重建的 trial_full.json (另一次
# build, 有不同的 paragraph_indices → 不同的 element→chunk 映射)
#
# 流程：
#   Step 0: 从 chunk_virtual_nodes_v2.json 重建 chunk 图
#           （带 element_ids 和 chunk_contains_element 边）
#   Step 1: 用重建的 chunk 图构建 eval corpus + qrels
#   Step 2: Dense retrieval eval
#   Step 3: Graph rerank（用同一个 chunk 图）
#
# 输出：data/05_eval/chunk_corpus_n400_consistent/
# ============================================================

set -euo pipefail

echo "============================================"
echo "Job: chunk_corpus_eval_consistent (P1 fix)"
echo "Job ID:  ${SLURM_JOB_ID}"
echo "Node:    $(hostname)"
echo "Start:   $(date)"
echo "============================================"

cd /projects/myyyx1/data-process-test
source /cluster/apps/software/Miniforge3/24.11.3-1/etc/profile.d/conda.sh
conda activate /projects/myyyx1/envs/minerU

TRIAL_M4_DIR="data/03_queries/M4query_v1"
TRIAL_GRAPH_ELEMENTS="$TRIAL_M4_DIR/graphs/multimodal_elements.json"
TRIAL_HUB_CANDS="data/02_enriched/hub_candidates_enriched_v3.json"
MODEL_NAME="Qwen/Qwen3-Embedding-0.6B"
DOWNLOAD_ROOT="models"

# Output paths — 使用新目录，不覆盖旧结果
CHUNK_GRAPH="data/01_graphs/paragraph_chunks_n400_consistent.json"
OUT_DIR="data/05_eval/chunk_corpus_n400_consistent"
GRAPH_RERANK_DIR="$OUT_DIR/graph_rerank"

mkdir -p logs "$OUT_DIR" "$GRAPH_RERANK_DIR"

# ============================================================
# Step 0: 一次性构建 chunk 图（带 element_ids + chunk_contains_element 边）
# ============================================================
echo ""
echo "=== Step 0: Build consistent chunk graph with element injection ==="

python scripts/build_paragraph_chunks.py \
    --chunk-size 400 \
    --elements "$TRIAL_GRAPH_ELEMENTS" \
    --output "$CHUNK_GRAPH"

echo "  Step 0 done: $(date)"

# 验证 chunk_contains_element 边存在
python3 -c "
import json
d = json.load(open('$CHUNK_GRAPH'))
doc = d['documents'].get('1104.3913', {})
edges = doc.get('edges', [])
from collections import Counter
ec = Counter(e['relation'] for e in edges)
print('  1104.3913 edge types:', dict(ec))
assert 'chunk_contains_element' in ec, 'FATAL: chunk_contains_element edges missing!'
print('  [OK] chunk_contains_element edges present')
print('  Stats:', json.dumps(d['stats'], indent=2))
"

# ============================================================
# Step 1: 用同一个 chunk 图构建 eval corpus + qrels
# ============================================================
echo ""
echo "=== Step 1: Build eval corpus from consistent chunk graph ==="

python scripts/build_chunk_corpus.py \
    --chunks "$CHUNK_GRAPH" \
    --graph-elements "$TRIAL_GRAPH_ELEMENTS" \
    --element-text-mode graph_only \
    --qrels "$TRIAL_M4_DIR/qrels.jsonl" \
    --out-dir "$OUT_DIR"

echo "  Step 1 done: $(date)"

# 验证 qrels 的 source_element_id 和 chunk 图的 chunk_contains_element 一致
echo ""
echo "=== Consistency check: qrels source_element_id vs graph chunk_contains_element ==="
python3 -c "
import json
from collections import defaultdict

# Load qrels
qrels = []
with open('$OUT_DIR/qrels.jsonl') as f:
    for line in f:
        qrels.append(json.loads(line))

# Build qrels map: (doc, element_id) → chunk_id
qrels_map = {}
for r in qrels:
    eid = r.get('source_element_id', '')
    cid = r['passage_id']
    qrels_map[eid] = cid

# Load graph edges
graph = json.load(open('$CHUNK_GRAPH'))
edge_map = {}  # element_id → chunk_id
for doc_id, doc in graph['documents'].items():
    for edge in doc.get('edges', []):
        if edge.get('relation') == 'chunk_contains_element':
            edge_map[edge['target']] = edge['source']

# Compare
common = set(qrels_map.keys()) & set(edge_map.keys())
match = sum(1 for eid in common if qrels_map[eid] == edge_map[eid])
total = len(common)
rate = match / total if total > 0 else 0
print(f'  Common elements: {total}')
print(f'  Matching chunk assignments: {match} ({rate:.1%})')
if rate < 0.99:
    print(f'  [FATAL] Consistency {rate:.1%} < 99% — {total - match} mismatches!')
    print(f'  This means chunk_contains_element edges do NOT match eval qrels.')
    print(f'  Aborting to prevent graph rerank on inconsistent data.')
    # Show first few mismatches for debugging
    mismatches = [(eid, qrels_map[eid], edge_map[eid]) for eid in common if qrels_map[eid] != edge_map[eid]]
    for eid, qc, gc in mismatches[:10]:
        print(f'    {eid}: qrels→{qc}  graph→{gc}')
    exit(1)
else:
    print('  [OK] Consistency >= 99%')
"

# ============================================================
# Step 2: Dense retrieval eval
# ============================================================
echo ""
echo "=== Step 2: Dense retrieval eval ==="

python scripts/eval_dense_retrieval.py \
    --data-dir "$TRIAL_M4_DIR" \
    --queries "$TRIAL_M4_DIR/queries.jsonl" \
    --corpus  "$OUT_DIR/corpus.jsonl" \
    --qrels   "$OUT_DIR/qrels.jsonl" \
    --model-name "$MODEL_NAME" \
    --download-root "$DOWNLOAD_ROOT" \
    --output "$OUT_DIR/eval_report.json" \
    --save-ranking "$OUT_DIR/ranking.jsonl" \
    --batch-size 16 \
    --max-length 1024 \
    --top-k 100

echo "  Step 2 done: $(date)"

# ============================================================
# Step 3: Graph rerank（用同一个 chunk 图，不重建）
# ============================================================
echo ""
echo "=== Step 3: Graph rerank (using consistent chunk graph) ==="

python scripts/eval_chunk_graph_rerank.py \
    --ranking        "$OUT_DIR/ranking.jsonl" \
    --corpus         "$OUT_DIR/corpus.jsonl" \
    --qrels          "$OUT_DIR/qrels.jsonl" \
    --chunk-graph    "$CHUNK_GRAPH" \
    --hub-candidates "$TRIAL_HUB_CANDS" \
    --output-dir     "$GRAPH_RERANK_DIR" \
    --static-weight  0.15 \
    --neighbor-decay 0.20 \
    --top-k          100

echo "  Step 3 done: $(date)"

# ============================================================
# Step 4: 汇总对照
# ============================================================
echo ""
echo "=== Step 4: Summary ==="
python3 - <<'PYEOF'
import json
from pathlib import Path

print()
print(f"{'Config':<45} {'R@1':>8} {'R@5':>8} {'R@10':>8} {'R@100':>8} {'MRR':>8}")
print("-" * 95)

# Old baseline for reference
for label, path in [
    ("[old] v1_enriched 0.6B (element)", "data/05_eval/dense_retrieval/stacking_06b/explicit_only/metrics_baseline.json"),
    ("[old] chunk_n400_fair (dense)", "data/05_eval/chunk_corpus_n400_fair/eval_report.json"),
]:
    p = Path(path)
    if p.exists():
        r = json.load(open(p))
        m = r.get("metrics", r)
        print(f"{label:<45} {m.get('recall@1',0):>8.4f} {m.get('recall@5',0):>8.4f} {m.get('recall@10',0):>8.4f} {m.get('recall@100',0):>8.4f} {m.get('mrr',0):>8.4f}")

# New consistent results
for label, path in [
    ("[new] chunk_n400_consistent (dense)", "data/05_eval/chunk_corpus_n400_consistent/eval_report.json"),
]:
    p = Path(path)
    if p.exists():
        r = json.load(open(p))
        m = r.get("metrics", r)
        print(f"{label:<45} {m.get('recall@1',0):>8.4f} {m.get('recall@5',0):>8.4f} {m.get('recall@10',0):>8.4f} {m.get('recall@100',0):>8.4f} {m.get('mrr',0):>8.4f}")

# Rerank results
rp = Path("data/05_eval/chunk_corpus_n400_consistent/graph_rerank/eval_report.json")
if rp.exists():
    r = json.load(open(rp))
    for config_name, res in r.get("results", {}).items():
        m = res.get("metrics", {})
        label = f"  + {config_name}"
        print(f"{label:<45} {m.get('recall@1',0):>8.4f} {m.get('recall@5',0):>8.4f} {m.get('recall@10',0):>8.4f} {m.get('recall@100',0):>8.4f} {m.get('mrr',0):>8.4f}")

print()
print("NOTE: Old fair numbers may shift slightly due to chunk rebuild from v2 paragraph nodes.")
print("The key fix is that chunk_contains_element edges now match eval qrels source_element_id.")
PYEOF

echo ""
echo "============================================"
echo "chunk_corpus_eval_consistent COMPLETE: $(date)"
echo "Output: $OUT_DIR/"
echo "Chunk graph: $CHUNK_GRAPH"
echo "============================================"
