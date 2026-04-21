#!/bin/bash
#SBATCH --job-name=trial57_backfill
#SBATCH --output=logs/37b_trial57_backfill_%j.out
#SBATCH --error=logs/37b_trial57_backfill_%j.err
#SBATCH --partition=cluster02
#SBATCH --qos=msc
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=12:00:00

# ============================================================
# 37b_trial57_backfill_enrich.sh
#
# 目标：
#   为 57 gold docs 的 old-trial retrieval-eval lane 补齐 enrichable
#   figure/table/formula，修正 selective enrich 带来的检索公平性问题。
#
# 关键约束：
#   - 只处理 57 gold docs（由 trial qrels 定义）
#   - 不接入 1040 production lane
#   - 输出独立 overlay：multimodal_elements_trial57_enriched_full.json
# ============================================================

set -euo pipefail

echo "============================================"
echo "Job: trial57_backfill_enrich"
echo "Job ID:  ${SLURM_JOB_ID}"
echo "Node:    $(hostname)"
echo "Start:   $(date)"
echo "============================================"

cd /projects/myyyx1/data-process-test
source /cluster/apps/software/Miniforge3/24.11.3-1/etc/profile.d/conda.sh
conda activate /projects/myyyx1/envs/minerU
export PYTHONPATH=/projects/myyyx1/data-process-test:${PYTHONPATH:-}
export $(grep -v '^#' .env | xargs)

TRIAL_GRAPH="data/03_queries/M4query_v1/graphs/multimodal_elements.json"
TRIAL_QRELS="data/03_queries/M4query_v1/qrels.jsonl"
TRIAL_OLD_OVERLAY="data/02_enriched/multimodal_elements_enriched.json"

TRIAL_SUBSET="data/02_enriched/trial57_missing_enrich_subset.json"
TRIAL_REPORT="data/02_enriched/trial57_missing_enrich_report.json"
TRIAL_BACKFILL="data/02_enriched/trial57_backfill_enriched.json"
TRIAL_FULL_OVERLAY="data/02_enriched/multimodal_elements_trial57_enriched_full.json"
TRIAL_POST_SUBSET="data/02_enriched/trial57_missing_enrich_postmerge_subset.json"
TRIAL_POST_REPORT="data/02_enriched/trial57_missing_enrich_postmerge_report.json"

mkdir -p logs data/02_enriched

echo ""
echo "=== Step 1: Build missing-enrich subset for trial57 ==="
python scripts/build_trial57_enrich_subset.py \
    --graph-elements "$TRIAL_GRAPH" \
    --existing-enriched "$TRIAL_OLD_OVERLAY" \
    --qrels "$TRIAL_QRELS" \
    --output-subset "$TRIAL_SUBSET" \
    --output-report "$TRIAL_REPORT"

MISSING_COUNT=$(python - <<'PY'
import json
rep = json.load(open("data/02_enriched/trial57_missing_enrich_report.json"))
print(rep["summary"]["missing"])
PY
)

echo "  Missing enrichable elements in trial57: ${MISSING_COUNT}"

if [ "$MISSING_COUNT" -gt 0 ]; then
    echo ""
    echo "=== Step 2: Backfill missing enrichable elements ==="
    echo "  Note: token usage is recorded by scripts/enrich_elements_modora.py -> logs/token_usage.db"
    python scripts/enrich_elements_modora.py \
        --input "$TRIAL_SUBSET" \
        --output "$TRIAL_BACKFILL" \
        --provider company \
        --model gpt-5.4 \
        --delay 0.3 \
        --flush-every 100 \
        --incremental \
        --company-api-key "$COMPANY_API_KEY" \
        --company-api-url "$COMPANY_API_URL"
else
    echo ""
    echo "=== Step 2: Skip API enrich (subset already complete) ==="
fi

echo ""
echo "=== Step 3: Merge old trial overlay + backfill overlay ==="
python scripts/merge_enriched_overlays.py \
    --base "$TRIAL_GRAPH" \
    --overlay "$TRIAL_OLD_OVERLAY" \
    --overlay "$TRIAL_BACKFILL" \
    --output "$TRIAL_FULL_OVERLAY"

echo ""
echo "=== Step 4: Post-merge coverage check ==="
python scripts/build_trial57_enrich_subset.py \
    --graph-elements "$TRIAL_GRAPH" \
    --existing-enriched "$TRIAL_FULL_OVERLAY" \
    --qrels "$TRIAL_QRELS" \
    --output-subset "$TRIAL_POST_SUBSET" \
    --output-report "$TRIAL_POST_REPORT"

python - <<'PY'
import json
rep = json.load(open("data/02_enriched/trial57_missing_enrich_postmerge_report.json"))
summ = rep["summary"]
print("")
print("== trial57 post-merge coverage ==")
print(f"Coverage: {summ['already_enriched']}/{summ['total_enrichable']} ({summ['coverage']:.1%})")
print(f"Gold qrel enrichable coverage: {summ['gold_qrel_enrichable_covered']}/{summ['gold_qrel_enrichable']} ({summ['gold_qrel_enrichable_coverage']:.1%})")
print(f"Missing by type: {summ['missing_by_type']}")
print(f"Remaining missing elements: {summ['missing']}")
PY

echo ""
echo "============================================"
echo "trial57_backfill_enrich COMPLETE: $(date)"
echo "Merged overlay: $TRIAL_FULL_OVERLAY"
echo "============================================"
