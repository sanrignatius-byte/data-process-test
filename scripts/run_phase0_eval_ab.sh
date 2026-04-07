#!/usr/bin/env bash
set -euo pipefail

python scripts/run_phase0_eval_ab.py \
  --q1 data/03_queries/l1_dual_evidence_queries_v4_4_run1_pass.jsonl \
  --q2 data/03_queries/l1_dual_evidence_queries_v3_pass.jsonl \
  --q3 data/03_queries/l1_img_run_20.jsonl \
  --elements data/02_enriched/multimodal_elements_enriched.json \
  --hubs data/01_graphs/latex_graph_hubs.json \
  --hub-candidates data/02_enriched/hub_candidates_enriched_v2.json \
  --output data/05_eval/phase0_eval_report.json \
  --top-k 10 \
  --overlap-threshold 0.5 \
  --graph-alpha 0.6
