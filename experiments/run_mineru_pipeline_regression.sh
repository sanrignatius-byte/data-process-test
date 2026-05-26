#!/usr/bin/env bash
# Full MinerU cross-doc pipeline regression on the current 53-doc corpus.
#
# Order matters: VL edges (math_similarity formulas) -> text rerank (tiers)
# -> cross-doc bridges (consume tiers, drop visual_only_risky by default)
# -> hub candidates. Each step updates its own *_latest symlink, so later
# steps automatically pick up the fresh upstream output.
#
# Usage: bash experiments/run_mineru_pipeline_regression.sh [--rebuild-vl]
#   --rebuild-vl  also recompute VL edges (open_clip + math_similarity, ~slow on CPU)
set -euo pipefail

PY=/projects/_hdd/myyyx1/envs/glm46v_py310/bin/python
cd "$(dirname "$0")/.."
ROOT=$(pwd)
EXP="$ROOT/experiments"

if [[ "${1:-}" == "--rebuild-vl" ]]; then
  echo "== [1/4] VL edges (open_clip visual/text + math_similarity formulas) =="
  $PY "$EXP/build_mineru_vl_edges.py" --backend open_clip --formula-backend math_similarity
else
  echo "== [1/4] VL edges: skipped (using mineru_vl_edges_v1_latest) =="
fi

echo "== [2/4] cross-doc text rerank (caption/context/enriched tiers) =="
$PY "$EXP/rerank_mineru_crossdoc_vl_edges.py"

echo "== [3/4] cross-doc bridges (tiered; visual_only_risky dropped by default) =="
$PY "$EXP/build_mineru_crossdoc_bridges.py"

echo "== [4/4] hub candidates =="
$PY "$EXP/build_mineru_hub_candidates.py"

echo "== snapshot =="
$PY - <<'PYEOF'
import json
from pathlib import Path
root = Path("/projects/myyyx1/data-process-test/data/05_eval")
def load(p):
    return json.loads((root / p / "summary.json").read_text())
vl = load("mineru_vl_edges_v1_latest")
rr = load("mineru_crossdoc_text_rerank_v1_latest")
br = load("mineru_crossdoc_bridges_v1_latest")
hub = load("mineru_hub_candidates_v1_latest")
print("VL backend:", vl["backend"], "| formula backend:", vl.get("formula_backend"))
print("VL edge counts:", vl["edge_type_counts"])
print("rerank tiers:", rr["tier_counts"])
print("rerank generic-both top100:", rr["generic_caption_both_top100"])
print("bridges crossdoc edges:", br["total_crossdoc_edges"], "| dropped:", br.get("dropped_by_tier"))
print("bridges crossdoc tiers:", br.get("crossdoc_tier_counts"))
print("bridges sentence/vl_align/orphan:", br["sentence_bridge_count"], br["vl_alignment_count"], br["orphan_visual_count"])
print("hub candidates:", hub.get("candidate_count") or hub.get("num_candidates") or hub.get("total_candidates"))
PYEOF
echo "== regression complete =="
