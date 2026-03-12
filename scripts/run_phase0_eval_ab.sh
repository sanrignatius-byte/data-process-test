#!/usr/bin/env bash
set -euo pipefail

python scripts/run_phase0_eval_ab.py \
  --q1 data/l1_dual_evidence_queries_v4_4_run1_pass.jsonl \
  --q2 data/l1_dual_evidence_queries_v3_pass.jsonl \
  --q3 "data111/l1_img_run_20.jsonl" \
  --elements "data111/multimodal_elements_enriched.json" \
  --hubs "data111/latex_graph_hubs (1).json" \
  --hub-candidates "data111/hub_candidates_enriched_v2.json" \
  --output data/phase0_eval_report.json \
  --top-k 10 \
  --overlap-threshold 0.5 \
  --graph-alpha 0.6
