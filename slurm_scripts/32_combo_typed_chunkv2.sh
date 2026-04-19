#!/bin/bash
#SBATCH --job-name=32_combo_typed_chunkv2
#SBATCH --output=logs/32_combo_typed_chunkv2_%j.out
#SBATCH --error=logs/32_combo_typed_chunkv2_%j.err
#SBATCH --partition=cluster02
#SBATCH --qos=msc
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=00:30:00

# Block 3 (R100-R103): explicit + typed_crossdoc + chunk_v2 combo, static_prior weight sweep
# Tests claim C_NEW: can one config achieve R@1 >= 0.25 AND R@10 >= 0.62?
# Baseline refs:
#   REF-A: explicit_only v1_enriched static_prior -> R@1=0.2357, R@10=0.6258
#   REF-C: explicit+typed(w=0.2) static_plus_neighbor -> R@1=0.1818, R@10=0.6406

set -e
cd /projects/myyyx1/data-process-test
source /cluster/apps/software/Miniforge3/24.11.3-1/etc/profile.d/conda.sh
conda activate /projects/myyyx1/envs/minerU
export PYTHONPATH=/projects/myyyx1/data-process-test:$PYTHONPATH

AUG_V2="data/05_eval/dense_retrieval/augmented_v2"
RANKING="$AUG_V2/ranking_v1_enriched_qwen06b.jsonl"
QRELS="$AUG_V2/qrels_v1.jsonl"
CORPUS="$AUG_V2/corpus_v1_enriched.jsonl"
CHUNK_V2="data/01_graphs/chunk_virtual_nodes_v2.json"
HUB_CANDS="data/02_enriched/hub_candidates_enriched_v3.json"
TYPED="data/01_graphs/typed_crossdoc_edges.json"
OUTBASE="data/05_eval/dense_retrieval/combo_typed_chunkv2"

mkdir -p "$OUTBASE" logs

run_rerank() {
    local name="$1"
    shift
    local outdir="$OUTBASE/$name"
    mkdir -p "$outdir"
    echo ""
    echo "=== $name ==="
    python scripts/eval_graph_topk_rerank.py \
        --ranking "$RANKING" \
        --qrels "$QRELS" \
        --corpus "$CORPUS" \
        --chunk-graph "$CHUNK_V2" \
        --hub-candidates "$HUB_CANDS" \
        --typed-crossdoc-edges "$TYPED" \
        --output-dir "$outdir" \
        --top-k 100 \
        --prior-mode weighted \
        "$@"
}

echo "=== Block 3: combo typed_crossdoc + chunk_v2, static_prior sweep ==="
echo "Date: $(date)"

# R100: w=0.1
run_rerank "explicit_typed_w01_v2chunk" \
    --graph-sources explicit typed_crossdoc \
    --typed-crossdoc-weight 0.1

# R101: w=0.2 — main hypothesis
run_rerank "explicit_typed_w02_v2chunk" \
    --graph-sources explicit typed_crossdoc \
    --typed-crossdoc-weight 0.2

# R102: w=0.3 — upper boundary
run_rerank "explicit_typed_w03_v2chunk" \
    --graph-sources explicit typed_crossdoc \
    --typed-crossdoc-weight 0.3

# R103a: w=0.2, boosted (test citation boost at this weight)
run_rerank "explicit_typed_w02_boosted_v2chunk" \
    --graph-sources explicit typed_crossdoc \
    --typed-crossdoc-weight 0.2 \
    --typed-crossdoc-use-boost

# R103b: explicit_only baseline on this ranking (reference)
run_rerank "explicit_only_v1enriched_ref" \
    --graph-sources explicit

echo ""
echo "=== ALL DONE: $(date) ==="
echo ""

# Summary table
printf "\n== static_prior results ==\n"
printf "%-48s %8s %8s %8s %8s\n" "config" "R@1" "R@5" "R@10" "MRR"
echo "--------------------------------------------------------------------------------------------"
for d in "$OUTBASE"/*/; do
    cfg=$(basename "$d")
    fp="$d/metrics_graph_static_prior.json"
    [ -f "$fp" ] || continue
    python3 -c "
import json
m=json.load(open('$fp'))
print(f\"{'$cfg':<48} {m.get('recall@1',0):8.4f} {m.get('recall@5',0):8.4f} {m.get('recall@10',0):8.4f} {m.get('mrr',0):8.4f}\")"
done

printf "\n== static_plus_neighbor results ==\n"
printf "%-48s %8s %8s %8s %8s\n" "config" "R@1" "R@5" "R@10" "MRR"
echo "--------------------------------------------------------------------------------------------"
for d in "$OUTBASE"/*/; do
    cfg=$(basename "$d")
    fp="$d/metrics_graph_static_plus_neighbor.json"
    [ -f "$fp" ] || continue
    python3 -c "
import json
m=json.load(open('$fp'))
print(f\"{'$cfg':<48} {m.get('recall@1',0):8.4f} {m.get('recall@5',0):8.4f} {m.get('recall@10',0):8.4f} {m.get('mrr',0):8.4f}\")"
done
