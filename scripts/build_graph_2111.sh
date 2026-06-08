#!/bin/bash
# 2111 篇全语料建图(intra-doc)：multimodal → latex reference graph(+merge v2) → topology
set -uo pipefail
cd /root/dataDisk/kuicai/m2query/data-process-test
PY=/root/dataDisk/kuicai/m2query/.venv/bin/python3
G=data/01_graphs
MINERU=data/00_raw/noncs1000_mineru_output
EXTRACTED=data/00_raw/noncs1000_latex_sources/extracted
IDS=$G/noncs2000_docids_2111.txt

echo "[$(date)] STEP1 multimodal relationships (2111)"
$PY scripts/build_multimodal_relationships.py \
  --mineru-dir "$MINERU" \
  --output $G/noncs2000_multimodal_elements_2111.json \
  --report $G/noncs2000_multimodal_report_2111.json \
  --max-hops 3 --context-window 3 || exit 1

echo "[$(date)] STEP2 latex reference graph + merge v2 (2111, doc-ids 过滤)"
$PY scripts/build_latex_reference_graph.py \
  --source-dir "$EXTRACTED" \
  --output $G/noncs2000_latex_reference_graph_2111.json \
  --report $G/noncs2000_latex_reference_report_2111.json \
  --max-hops 3 \
  --merge-with $G/noncs2000_multimodal_elements_2111.json \
  --merged-output $G/noncs2000_multimodal_elements_v2_2111.json || exit 1

echo "[$(date)] STEP3 topology (intra-doc, top-k-hubs 5000 / max-candidates 20000)"
$PY scripts/analyze_latex_graph_topology.py \
  --latex-graph $G/noncs2000_latex_reference_graph_2111.json \
  --elements $G/noncs2000_multimodal_elements_v2_2111.json \
  --mineru-output "$MINERU" \
  --single-doc-only --top-k-hubs 5000 --max-candidates 20000 \
  --output-report $G/noncs2000_latex_graph_topology_report_2111.json \
  --output-hubs $G/noncs2000_latex_graph_hubs_2111.json \
  --output-candidates $G/noncs2000_latex_hub_multihop_candidates_2111.json || exit 1

echo "[$(date)] DONE 2111 graph build"
$PY - <<EOF
import json
e=json.load(open("$G/noncs2000_multimodal_elements_v2_2111.json"))
c=json.load(open("$G/noncs2000_latex_hub_multihop_candidates_2111.json"))
print("v2 docs:",len(e["documents"]))
cands=c["candidates"]; docs=set(x["doc_id"] for x in cands)
print("候选:",len(cands),"覆盖文档:",len(docs))
EOF
