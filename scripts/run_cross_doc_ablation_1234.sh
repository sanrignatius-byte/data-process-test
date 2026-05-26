#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

RUN_LOG="logs/cross_doc_ablation_1234_$(date -u +%Y%m%dT%H%M%SZ).log"
mkdir -p logs data/05_eval/cross_doc_ablation_1234
exec > >(tee -a "$RUN_LOG") 2>&1

echo "[start] $(date -u --iso-8601=seconds)"
echo "[root] $ROOT"
echo "[log] $RUN_LOG"

set -a
source .env
set +a

echo "[step] build candidates"
python3 experiments/build_cross_doc_ablation_1234.py

run_judge() {
  local input="$1"
  local out="$2"
  local label="$3"
  echo
  echo "[step] judge $label"
  python3 experiments/judge_cross_doc_chain.py \
    --input "$input" \
    --output-dir "$out" \
    --model gpt-5.4 \
    --sleep-between 3 \
    --rate-limit-sleep 90 \
    --resume
}

# Strategy 1 baseline was already judged in:
# data/05_eval/cross_doc_chain_judge_fixed/summary.json
run_judge \
  data/05_eval/cross_doc_ablation_1234/entity_cluster_chains.json \
  data/05_eval/cross_doc_ablation_1234/judge_entity_cluster \
  "2_entity_cluster"

run_judge \
  data/05_eval/cross_doc_ablation_1234/gated_path_chains.json \
  data/05_eval/cross_doc_ablation_1234/judge_gated_path \
  "3_gated_path"

run_judge \
  data/05_eval/cross_doc_ablation_1234/entity_cluster_enriched_chains.json \
  data/05_eval/cross_doc_ablation_1234/judge_entity_cluster_enriched \
  "4_entity_cluster_enriched"

echo
echo "[step] compare"
python3 experiments/compare_cross_doc_ablation_1234.py

echo "[done] $(date -u --iso-8601=seconds)"
