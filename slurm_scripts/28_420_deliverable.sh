#!/bin/bash
#SBATCH --job-name=420_deliverable
#SBATCH --partition=cluster02
#SBATCH --qos=msc
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:a6000:1
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=logs/420_deliverable_%j.out
#SBATCH --error=logs/420_deliverable_%j.err

# ============================================================
# 4.20 Deliverable: Complete evaluation on new chunk graph v2
#
# Part A: Dense eval of v2_chunks corpus on 0.6B (missing from Job 61297)
# Part B: Graph rerank (explicit_only + static_prior) on 0.6B rankings
#         using NEW chunk_virtual_nodes_v2.json
# Part C: C-Pool retrieval (78 universal queries) on 0.6B
# ============================================================

set -euo pipefail

echo "============================================"
echo "Job: 4.20 deliverable"
echo "Job ID:  ${SLURM_JOB_ID}"
echo "Node:    $(hostname)"
echo "Start:   $(date)"
echo "============================================"

cd /projects/myyyx1/data-process-test
source /cluster/apps/software/Miniforge3/24.11.3-1/etc/profile.d/conda.sh
conda activate /projects/myyyx1/envs/minerU

mkdir -p logs

# ── Paths ────────────────────────────────────────────────────────────────────
AUG_V2="data/05_eval/dense_retrieval/augmented_v2"
REBUILT="data/05_eval/dense_retrieval/rebuilt_20260417"
CHUNK_V2="data/01_graphs/chunk_virtual_nodes_v2.json"
CHUNK_V1="data/01_graphs/chunk_virtual_nodes.json"
HUB_CANDS="data/02_enriched/hub_candidates_enriched_v3.json"
MODEL="Qwen/Qwen3-Embedding-0.6B"
MODEL_ROOT="models"

# ensure nltk
python -c "
import nltk
for pkg in ['punkt', 'punkt_tab']:
    try: nltk.data.find(f'tokenizers/{pkg}')
    except LookupError: nltk.download(pkg, quiet=True)
"

# ============================================================
# PART A — Dense retrieval: v2_chunks corpus × 0.6B
# ============================================================
echo ""
echo "=== Part A: Dense eval v2_chunks × Qwen3-Embedding-0.6B ==="

EVAL_V2="${AUG_V2}/eval_v2_chunks_qwen06b.json"
RANK_V2="${AUG_V2}/ranking_v2_chunks_qwen06b.jsonl"

if [ -f "$EVAL_V2" ] && python -c "import json; r=json.load(open('$EVAL_V2')); assert r.get('recall@1',0)>0"; then
    echo "  Already exists, skipping."
else
    python scripts/eval_dense_retrieval.py \
        --data-dir "$AUG_V2" \
        --queries "$AUG_V2/queries.jsonl" \
        --corpus "$AUG_V2/corpus_v2_chunks.jsonl" \
        --qrels "$AUG_V2/qrels_v2_chunks.jsonl" \
        --model-name "$MODEL" \
        --download-root "$MODEL_ROOT" \
        --output "$EVAL_V2" \
        --save-ranking "$RANK_V2" \
        --batch-size 16 \
        --max-length 512 \
        --top-k 100
    echo "  Part A done: $(date)"
fi

python -c "
import json
r = json.load(open('$EVAL_V2'))
print(f'  v2_chunks 0.6B: R@1={r[\"recall@1\"]:.4f}  R@5={r[\"recall@5\"]:.4f}  R@10={r[\"recall@10\"]:.4f}  R@100={r[\"recall@100\"]:.4f}  MRR={r[\"mrr\"]:.4f}')
"

# ============================================================
# PART B — Graph rerank: explicit_only + static_prior on 0.6B
#          using NEW chunk_virtual_nodes_v2.json
# ============================================================
echo ""
echo "=== Part B: Graph rerank (explicit_only + static_prior) on 0.6B ==="

# B1: Rerank v1_enriched 0.6B ranking
OUTDIR_B1="${REBUILT}/graph_06b_v1_explicit_v2chunk"
echo "  B1: v1_enriched ranking + explicit graph (v2 chunk graph)"
python scripts/eval_graph_topk_rerank.py \
    --ranking "$AUG_V2/ranking_v1_enriched_qwen06b.jsonl" \
    --qrels "$AUG_V2/qrels_v1.jsonl" \
    --corpus "$AUG_V2/corpus_v1_enriched.jsonl" \
    --chunk-graph "$CHUNK_V2" \
    --hub-candidates "$HUB_CANDS" \
    --graph-sources explicit \
    --output-dir "$OUTDIR_B1" \
    --static-weight 0.15 \
    --neighbor-decay 0.20
echo "  B1 done: $(date)"

# B2: Rerank v2_chunks 0.6B ranking (if ranking exists from Part A)
if [ -f "$RANK_V2" ]; then
    OUTDIR_B2="${REBUILT}/graph_06b_v2chunks_explicit_v2chunk"
    echo "  B2: v2_chunks ranking + explicit graph (v2 chunk graph)"
    python scripts/eval_graph_topk_rerank.py \
        --ranking "$RANK_V2" \
        --qrels "$AUG_V2/qrels_v2_chunks.jsonl" \
        --corpus "$AUG_V2/corpus_v2_chunks.jsonl" \
        --chunk-graph "$CHUNK_V2" \
        --hub-candidates "$HUB_CANDS" \
        --graph-sources explicit \
        --output-dir "$OUTDIR_B2" \
        --static-weight 0.15 \
        --neighbor-decay 0.20
    echo "  B2 done: $(date)"
fi

# B3: For comparison — also rerank v1 with OLD chunk graph (should reproduce known winner)
OUTDIR_B3="${REBUILT}/graph_06b_v1_explicit_v1chunk"
echo "  B3: v1_enriched ranking + explicit graph (v1 OLD chunk graph) — comparison"
python scripts/eval_graph_topk_rerank.py \
    --ranking "$AUG_V2/ranking_v1_enriched_qwen06b.jsonl" \
    --qrels "$AUG_V2/qrels_v1.jsonl" \
    --corpus "$AUG_V2/corpus_v1_enriched.jsonl" \
    --chunk-graph "$CHUNK_V1" \
    --hub-candidates "$HUB_CANDS" \
    --graph-sources explicit \
    --output-dir "$OUTDIR_B3" \
    --static-weight 0.15 \
    --neighbor-decay 0.20
echo "  B3 done: $(date)"

# ============================================================
# PART C — C-Pool: universal queries retrieval on 0.6B
# ============================================================
echo ""
echo "=== Part C: C-Pool retrieval (78 universal queries) ==="

CPOOL_DIR="data/05_eval/cpool_retrieval"
mkdir -p "$CPOOL_DIR"

# Convert c_pool queries to queries.jsonl format
python3 << 'PYEOF'
import json
cp = json.load(open("data/03_queries/c_pool_universal_queries.json"))
queries = cp["queries"]
# C-Pool queries are doc-agnostic templates. We need to run them against each doc.
# For now, use them as-is for retrieval on the v1_enriched corpus.
with open("data/05_eval/cpool_retrieval/queries.jsonl", "w") as f:
    for q in queries:
        f.write(json.dumps({"query_id": q["id"], "query": q["query"]}) + "\n")
print(f"Wrote {len(queries)} C-Pool queries")
PYEOF

# Dense retrieval on v1_enriched corpus
python scripts/eval_dense_retrieval.py \
    --data-dir "$CPOOL_DIR" \
    --queries "$CPOOL_DIR/queries.jsonl" \
    --corpus "$AUG_V2/corpus_v1_enriched.jsonl" \
    --model-name "$MODEL" \
    --download-root "$MODEL_ROOT" \
    --output "$CPOOL_DIR/eval_cpool_v1_enriched_qwen06b.json" \
    --save-ranking "$CPOOL_DIR/ranking_cpool_v1_enriched_qwen06b.jsonl" \
    --batch-size 16 \
    --max-length 512 \
    --top-k 100

echo "  C-Pool retrieval done: $(date)"

# Export top-5 results per query for manual inspection
python3 << 'PYEOF'
import json
from pathlib import Path

corpus = {}
for line in open("data/05_eval/dense_retrieval/augmented_v2/corpus_v1_enriched.jsonl"):
    d = json.loads(line)
    corpus[d["pid"]] = d.get("text", d.get("contents", ""))[:200]

queries = {}
for line in open("data/05_eval/cpool_retrieval/queries.jsonl"):
    d = json.loads(line)
    queries[d["query_id"]] = d["query"]

ranking_file = "data/05_eval/cpool_retrieval/ranking_cpool_v1_enriched_qwen06b.jsonl"
out = []
for line in open(ranking_file):
    r = json.loads(line)
    qid = r["query_id"]
    top5 = r["top_k"][:5] if isinstance(r["top_k"][0], str) else [x[0] for x in r["top_k"][:5]]
    entry = {
        "query_id": qid,
        "query": queries.get(qid, ""),
        "top5_pids": top5,
        "top5_snippets": [corpus.get(pid, "???")[:120] for pid in top5]
    }
    out.append(entry)

with open("data/05_eval/cpool_retrieval/cpool_top5_preview.json", "w") as f:
    json.dump(out, f, indent=2, ensure_ascii=False)
print(f"Exported {len(out)} C-Pool retrieval previews")
PYEOF

# ============================================================
# SUMMARY TABLE
# ============================================================
echo ""
echo "=== Final Comparison Table ==="

python3 << 'PYEOF'
import json
from pathlib import Path

results = []

def load_metric(path, label):
    if not Path(path).exists():
        return
    r = json.load(open(path))
    results.append((label, r.get("recall@1",0), r.get("recall@5",0), r.get("recall@10",0), r.get("recall@100",0), r.get("mrr",0)))

def load_rerank_metric(dir_path, metric_file, label):
    p = Path(dir_path) / metric_file
    if not p.exists():
        return
    r = json.load(open(p))
    m = r.get("metrics", r)
    results.append((label, m.get("recall@1",0), m.get("recall@5",0), m.get("recall@10",0), m.get("recall@100",0), m.get("mrr",0)))

# Dense baselines
load_metric("data/05_eval/dense_retrieval/augmented_v2/eval_v1_enriched_qwen06b.json", "0.6B dense (v1_enriched)")
load_metric("data/05_eval/dense_retrieval/augmented_v2/eval_v2_chunks_qwen06b.json", "0.6B dense (v2_chunks)")

# Graph rerank
rb = "data/05_eval/dense_retrieval/rebuilt_20260417"
load_rerank_metric(f"{rb}/graph_06b_v1_explicit_v1chunk", "metrics_graph_static_prior.json", "0.6B + explicit rerank (OLD chunk)")
load_rerank_metric(f"{rb}/graph_06b_v1_explicit_v2chunk", "metrics_graph_static_prior.json", "0.6B + explicit rerank (NEW chunk v2)")
load_rerank_metric(f"{rb}/graph_06b_v2chunks_explicit_v2chunk", "metrics_graph_static_prior.json", "0.6B v2corpus + explicit rerank (NEW chunk v2)")

# 4B reference
load_rerank_metric(f"{rb}/graph_explicit_only_fixed", "metrics_graph_static_prior.json", "[ref] 4B + explicit rerank (OLD chunk)")

print()
print(f"{'Configuration':<50} {'R@1':>7} {'R@5':>7} {'R@10':>7} {'R@100':>7} {'MRR':>7}")
print("=" * 92)
for label, r1, r5, r10, r100, mrr in results:
    print(f"{label:<50} {r1:>7.4f} {r5:>7.4f} {r10:>7.4f} {r100:>7.4f} {mrr:>7.4f}")
print()
PYEOF

echo ""
echo "============================================"
echo "4.20 deliverable COMPLETE: $(date)"
echo "============================================"
