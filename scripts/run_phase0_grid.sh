#!/usr/bin/env bash
set -euo pipefail

# Grid search helper for Phase-0 A/B retrieval evaluation.
# Uses current safer graph-rerank implementation in run_phase0_eval_ab.py.

OUT_DIR=${1:-logs/phase0_grid}
mkdir -p "$OUT_DIR"

echo "[phase0-grid] output dir: $OUT_DIR"

for a in 0.0 0.1 0.2 0.3 0.4; do
  for n in 20 50 100 200; do
    out="$OUT_DIR/report_a${a}_n${n}.json"
    echo "[phase0-grid] alpha=$a topn=$n -> $out"
    python scripts/run_phase0_eval_ab.py \
      --q1 data/l1_dual_evidence_queries_v4_4_run1_pass.jsonl \
      --q2 data/l1_dual_evidence_queries_v3_pass.jsonl \
      --q3 data111/l1_img_run_20.jsonl \
      --elements data111/multimodal_elements_enriched.json \
      --hubs "data111/latex_graph_hubs (1).json" \
      --hub-candidates data111/hub_candidates_enriched_v2.json \
      --output "$out" \
      --top-k 10 \
      --overlap-threshold 0.5 \
      --graph-alpha "$a" \
      --graph-rerank-topn "$n"
  done
done

echo "[phase0-grid] done."
python scripts/summarize_phase0_grid.py --dir "$OUT_DIR"
