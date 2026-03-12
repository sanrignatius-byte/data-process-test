#!/usr/bin/env bash
set -euo pipefail

python scripts/run_phase0_eval_ab.py \
  --q1 data/l1_dual_evidence_queries_v4_4_run1_pass.jsonl \
  --q2 data/l1_dual_evidence_queries_v3_pass.jsonl \
  --elements data/multimodal_elements.json \
  --hubs data/latex_graph_hubs.json \
  --output data/phase0_eval_report.json \
  --top-k 10 \
  --overlap-threshold 0.5 \
  --graph-alpha 0.6
