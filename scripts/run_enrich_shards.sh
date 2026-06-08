#!/bin/bash
# 14-cell 进程级并行 enrich（沿用 four_cells 的 &+wait 模式）。
# 每个 shard 起一个 enrich_elements_modora.py，--num-shards/--shard-index 取模分片，
# 各自 incremental+flush 落到 shard 文件，全部 wait 后 merge_enriched_overlays 合并。
set -uo pipefail
cd /root/dataDisk/kuicai/m2query/data-process-test
PY=/root/dataDisk/kuicai/m2query/.venv/bin/python3
set -a; source .env; set +a
export API_LOG_DIR=$PWD/api_logs
export PYTHONUNBUFFERED=1

N=${N:-14}
INPUT=${INPUT:-data/01_graphs/noncs1000_multimodal_elements_v2_972.json}
SHARD_DIR=${SHARD_DIR:-data/02_enriched/enrich_shards_972}
MERGED=${MERGED:-data/02_enriched/noncs1000_elements_enriched_972.json}
LIMIT=${LIMIT:-0}          # >0 时每片只跑这么多（冒烟用）
DELAY=${DELAY:-0.3}
MODEL=${MODEL:-gpt-5.4}
mkdir -p "$SHARD_DIR" logs

echo "[$(date)] 启动 $N 片 enrich | input=$INPUT | limit/片=$LIMIT"
declare -a PIDS
for ((i=0; i<N; i++)); do
    OUT="$SHARD_DIR/shard_${i}.json"
    LOG="logs/enrich_shard_${i}.log"
    $PY scripts/enrich_elements_modora.py \
        --input "$INPUT" --output "$OUT" \
        --provider company --model "$MODEL" \
        --delay "$DELAY" --max-retries 3 --flush-every 20 --incremental \
        --num-shards "$N" --shard-index "$i" --limit "$LIMIT" \
        > "$LOG" 2>&1 &
    PIDS[$i]=$!
    echo "  shard $i started pid=${PIDS[$i]} → $OUT"
done

FAILS=0
for ((i=0; i<N; i++)); do
    wait "${PIDS[$i]}"; rc=$?
    echo "[$(date)] shard $i 完成 rc=$rc"
    [[ $rc -ne 0 ]] && FAILS=$((FAILS+1))
done

echo "[$(date)] 全部 shard 结束，失败 $FAILS 片。开始 merge..."
OVERLAYS=()
for ((i=0; i<N; i++)); do OVERLAYS+=(--overlay "$SHARD_DIR/shard_${i}.json"); done
$PY scripts/merge_enriched_overlays.py \
    --base "$INPUT" "${OVERLAYS[@]}" --output "$MERGED"

echo "[$(date)] DONE → $MERGED"
$PY - <<EOF
import json
d=json.load(open("$MERGED"))
n=sum(1 for doc in d["documents"].values() for e in doc["elements"].values() if "enriched_title" in e)
print("merged enriched 元素:",n)
EOF
