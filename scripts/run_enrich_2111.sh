#!/bin/bash
# 2111 全量 enrich，串行两阶段(各 14-shard，避免超并发触发 403):
#   Phase A: section-enrich (35591 sections)
#   Phase B: 新增元素 enrich (106667, 排除已做的 68610) → merge 成 2111 全量 enriched
set -uo pipefail
cd /root/dataDisk/kuicai/m2query/data-process-test
PY=/root/dataDisk/kuicai/m2query/.venv/bin/python3
set -a; source .env; set +a
export API_LOG_DIR=$PWD/api_logs PYTHONPATH=$PWD PYTHONUNBUFFERED=1
G=data/01_graphs; E=data/02_enriched
N=14

run_shards () {  # $1=label  $2=cmd-template(用 {I} 占位)
  local label=$1; shift; local tmpl="$*"
  local pids=()
  for ((i=0;i<N;i++)); do
    eval "${tmpl//\{I\}/$i}" > "logs/${label}_shard_${i}.log" 2>&1 &
    pids+=($!)
  done
  local fail=0
  for ((i=0;i<N;i++)); do wait "${pids[$i]}" || fail=$((fail+1)); done
  echo "[$(date)] $label 全部 shard 结束，失败 $fail"
}

echo "[$(date)] ===== 并行启动 Phase A(section) + Phase B(elements)，各 14-shard = 28 并发 ====="
mkdir -p $E/section_shards_2111 $E/elem_shards_2111

# Phase A 与 Phase B 同时跑(各自 run_shards 后台化)
( run_shards secA "$PY scripts/enrich_section_nodes.py \
  --reference-graph $G/noncs2000_latex_reference_graph_2111.json \
  --output $E/section_shards_2111/shard_{I}.json \
  --provider company --model gpt-5.4 --delay 0.3 --incremental --flush-every 20 \
  --num-shards $N --shard-index {I}" ) &
PA=$!
( run_shards secB "$PY scripts/enrich_elements_modora.py \
  --input $E/noncs2000_new_elements_subset.json \
  --output $E/elem_shards_2111/shard_{I}.json \
  --provider company --model gpt-5.4 --delay 0.3 --max-retries 3 --flush-every 20 --incremental \
  --num-shards $N --shard-index {I}" ) &
PB=$!
wait $PA; echo "[$(date)] Phase A done"
wait $PB; echo "[$(date)] Phase B done"

echo "[$(date)] merge section shards..."
$PY - <<EOF
import json,glob
secs={}
for f in sorted(glob.glob("$E/section_shards_2111/shard_*.json")):
    for s in json.load(open(f)).get("sections",[]):
        sid=s.get("section_id")
        if sid and sid not in secs: secs[sid]=s
json.dump({"sections":list(secs.values())},open("$E/noncs2000_section_nodes_enriched_2111.json","w"),ensure_ascii=False)
print("merged sections:",len(secs))
EOF

echo "[$(date)] merge 元素: base=v2_2111 + 972已enrich + 14新shard"
OVL=(--overlay $E/noncs1000_elements_enriched_972.json)
for ((i=0;i<N;i++)); do OVL+=(--overlay $E/elem_shards_2111/shard_${i}.json); done
$PY scripts/merge_enriched_overlays.py \
  --base $G/noncs2000_multimodal_elements_v2_2111.json "${OVL[@]}" \
  --output $E/noncs2000_elements_enriched_2111.json

echo "[$(date)] ===== DONE 2111 enrich ====="
$PY -c "
import json
d=json.load(open('$E/noncs2000_elements_enriched_2111.json'))
n=sum(1 for doc in d['documents'].values() for e in doc['elements'].values() if e.get('enriched_title') or e.get('enriched_content'))
print('2111 enriched 元素:',n)
"
