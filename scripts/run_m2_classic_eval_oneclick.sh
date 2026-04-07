#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   bash scripts/run_m2_classic_eval_oneclick.sh [ELEMENTS_JSON]
#
# Example:
#   bash scripts/run_m2_classic_eval_oneclick.sh data/02_enriched/multimodal_elements_enriched.json

ELEMENTS_PATH="${1:-data/02_enriched/multimodal_elements_enriched.json}"
if [[ ! -f "$ELEMENTS_PATH" ]]; then
  echo "[error] elements file not found: $ELEMENTS_PATH"
  echo "        pass the path explicitly, e.g.:"
  echo "        bash scripts/run_m2_classic_eval_oneclick.sh /path/to/multimodal_elements_enriched.json"
  exit 1
fi

OUT="data/05_eval/m2/phase0_classic_eval_m2_oneclick.json"

python3 scripts/run_phase0_eval_ab.py \
  --q1 data/05_eval/m2/l2_production_2026-03-26_section_enriched_pass.jsonl \
  --q2 data/05_eval/m2/l3_production_2026-03-26_section_enriched_pass.jsonl \
  --q3 data/05_eval/m2/level1_single_element.jsonl \
  --elements "$ELEMENTS_PATH" \
  --hub-candidates data/05_eval/m2/hub_candidates_enriched_keyword_boost_full_2026-03-24.json \
  --citation-graph data/01_graphs/citation_graph.json \
  --top-k 20 \
  --hub-weight 0.15 \
  --nprop-weight 1.0 \
  --cite-weight 0 \
  --neighbor-decay 0.5 \
  --graph-rerank-topn 100 \
  --output "$OUT"

echo "[done] report written to: $OUT"
