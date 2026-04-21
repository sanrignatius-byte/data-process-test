#!/bin/bash
#SBATCH --job-name=chunk_graph_rerank
#SBATCH --partition=cluster02
#SBATCH --qos=msc
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=logs/chunk_graph_rerank_%j.out
#SBATCH --error=logs/chunk_graph_rerank_%j.err
#SBATCH --dependency=afterok:62353

# ============================================================
# 38_chunk_graph_rerank.sh
# 依赖：Job 62353（chunk_corpus_eval）完成后执行
#
# 目标：验证图结构能增强 fair chunk 检索
#
# 步骤：
#   Step 1: 重建 paragraph_chunks_n400_trial_full.json（带完整的 element_ids
#           和 chunk_contains_element 边，用于 explicit 图投影）
#   Step 2: 用 fair chunk dense ranking 做图 rerank
#           - Layer 1: chunk_sequence（顺序邻近）
#           - Layer 2: explicit hub pairs 投影到 chunk 空间
# 
# 注意：
#   - 这是 57 gold docs 的 old trial lane
#   - 不要把 1040 production 的 partial enrich / production_full 文件混入这里
# ============================================================

set -euo pipefail

echo "============================================"
echo "Job: chunk_graph_rerank"
echo "Job ID:  ${SLURM_JOB_ID}"
echo "Node:    $(hostname)"
echo "Start:   $(date)"
echo "============================================"

cd /projects/myyyx1/data-process-test
source /cluster/apps/software/Miniforge3/24.11.3-1/etc/profile.d/conda.sh
conda activate /projects/myyyx1/envs/minerU

TRIAL_M4_DIR="data/03_queries/M4query_v1"
TRIAL_GRAPH_ELEMENTS="$TRIAL_M4_DIR/graphs/multimodal_elements.json"
CHUNKS_V2="data/01_graphs/paragraph_chunks_n400_trial_full.json"
TRIAL_HUB_CANDS="data/02_enriched/hub_candidates_enriched_v3.json"

# ============================================================
# STEP 1 — 重建 chunk 图（带 element_ids 和 chunk→element 边）
# ============================================================
echo ""
echo "=== Step 1: Rebuild paragraph_chunks_n400 with element injection ==="

python scripts/build_paragraph_chunks.py \
    --chunk-size 400 \
    --elements "$TRIAL_GRAPH_ELEMENTS" \
    --output "$CHUNKS_V2"

echo "  Step 1 done: $(date)"

python3 -c "
import json
d = json.load(open('$CHUNKS_V2'))
print('stats:', d['stats'])
# 验证 element_ids 和 chunk_contains_element 边
doc = d['documents'].get('1104.3913', {})
edges = doc.get('edges', [])
from collections import Counter
ec = Counter(e['relation'] for e in edges)
print('1104.3913 edge types:', dict(ec))
node0 = list(doc.get('nodes', {}).values())[0]
print('chunk_0 element_ids:', node0.get('element_ids'))
"

# ============================================================
# STEP 2 — 图 rerank（n400 chunk corpus）
# ============================================================
echo ""
echo "=== Step 2: Graph rerank on chunk_n400 dense ranking ==="

N400_DIR="data/05_eval/chunk_corpus_n400_fair"
RANKING="$N400_DIR/ranking.jsonl"

if [ ! -f "$RANKING" ]; then
    echo "[error] ranking file not found: $RANKING"
    echo "  Job 62353 may not have completed successfully."
    exit 1
fi

python scripts/eval_chunk_graph_rerank.py \
    --ranking        "$RANKING" \
    --corpus         "$N400_DIR/corpus.jsonl" \
    --qrels          "$N400_DIR/qrels.jsonl" \
    --chunk-graph    "$CHUNKS_V2" \
    --hub-candidates "$TRIAL_HUB_CANDS" \
    --output-dir     "$N400_DIR/graph_rerank" \
    --static-weight  0.15 \
    --neighbor-decay 0.20 \
    --top-k          100

echo "  Step 2 done: $(date)"

# ============================================================
# STEP 3 — 汇总对照表
# ============================================================
echo ""
echo "=== Step 3: Final comparison ==="
python3 - <<'PYEOF'
import json
from pathlib import Path

print()
print(f"{'Config':<40} {'R@1':>8} {'R@5':>8} {'R@10':>8} {'MRR':>8}")
print("-" * 70)

# v1_enriched 0.6B baseline（已有）
baseline = json.load(open("data/05_eval/dense_retrieval/stacking_06b/explicit_only/metrics_baseline.json"))
m = baseline
print(f"{'[ref] v1_enriched 0.6B (element)':<40} {m.get('recall@1',0):>8.4f} {m.get('recall@5',0):>8.4f} {m.get('recall@10',0):>8.4f} {m.get('mrr',0):>8.4f}")

# chunk_n400 dense baseline（fair lane）
cp = Path("data/05_eval/chunk_corpus_n400_fair/eval_report.json")
if cp.exists():
    r = json.load(open(cp))
    m = r.get("metrics", {})
    print(f"{'chunk_n400 dense':<40} {m.get('recall@1',0):>8.4f} {m.get('recall@5',0):>8.4f} {m.get('recall@10',0):>8.4f} {m.get('mrr',0):>8.4f}")

# chunk_n500 dense baseline（Job 62353 结果）
cp = Path("data/05_eval/chunk_corpus_n500_fair/eval_report.json")
if cp.exists():
    r = json.load(open(cp))
    m = r.get("metrics", {})
    print(f"{'chunk_n500 dense':<40} {m.get('recall@1',0):>8.4f} {m.get('recall@5',0):>8.4f} {m.get('recall@10',0):>8.4f} {m.get('mrr',0):>8.4f}")

# chunk_n400 + graph rerank（本 job 结果）
rp = Path("data/05_eval/chunk_corpus_n400_fair/graph_rerank/eval_report.json")
if rp.exists():
    r = json.load(open(rp))
    for config_name, res in r.get("results", {}).items():
        m = res.get("metrics", {})
        short = config_name[:40]
        print(f"  {short:<38} {m.get('recall@1',0):>8.4f} {m.get('recall@5',0):>8.4f} {m.get('recall@10',0):>8.4f} {m.get('mrr',0):>8.4f}")
print()
PYEOF

echo ""
echo "============================================"
echo "chunk_graph_rerank COMPLETE: $(date)"
echo "Results: data/05_eval/chunk_corpus_n400_fair/graph_rerank/"
echo "============================================"
