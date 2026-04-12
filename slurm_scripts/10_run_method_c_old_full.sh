#!/bin/bash
#SBATCH -p cluster02
#SBATCH --job-name=method_c_old
#SBATCH --output=logs/method_c_old_%j.out
#SBATCH --error=logs/method_c_old_%j.err
#SBATCH --time=12:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G

set -euo pipefail

module load Miniforge3 || true

REPO_ROOT=${REPO_ROOT:-/projects/myyyx1/data-process-test}
CONDA_ENV=${CONDA_ENV:-/projects/myyyx1/envs/minerU}

API_URL=${API_URL:-https://az.gptplus5.com/v1/chat/completions}
MODEL=${MODEL:-gpt-4.1}
JUDGE_MODEL=${JUDGE_MODEL:-gpt-4.1}
OLD_ENRICHED=${OLD_ENRICHED:-data/02_enriched/hub_candidates_enriched_v4.json}
MIN_HOP=${MIN_HOP:-2}
MAX_HOP=${MAX_HOP:-3}
NUM_SAMPLES=${NUM_SAMPLES:-0}
PER_DOC_CAP=${PER_DOC_CAP:-0}
SEED=${SEED:-42}
SKIP_B1=${SKIP_B1:-0}
SKIP_B2=${SKIP_B2:-0}

OUT_B1=${OUT_B1:-data/03_queries/pilot_method_c_v3_old_full_bridge1.json}
OUT_B2=${OUT_B2:-data/03_queries/pilot_method_c_v3_old_full_bridge2.json}

cd "$REPO_ROOT"
mkdir -p logs "$(dirname "$OUT_B1")"

source /cluster/apps/software/Miniforge3/24.11.3-1/etc/profile.d/conda.sh
conda activate "$CONDA_ENV"

if [[ -f .env ]]; then
    set -a
    # shellcheck disable=SC1091
    source .env
    set +a
fi

if [[ -z "${COMPANY_API_KEY:-}" ]]; then
    echo "ERROR: COMPANY_API_KEY not set"
    exit 1
fi

export PYTHONUNBUFFERED=1

echo "=========================================="
echo "Method C feasibility on old enriched data"
echo "Start time:   $(date)"
echo "Model:        ${MODEL}"
echo "Judge model:  ${JUDGE_MODEL}"
echo "Input:        ${OLD_ENRICHED}"
echo "Hop filter:   ${MIN_HOP}-${MAX_HOP}"
echo "Num samples:  ${NUM_SAMPLES} (0 = all filtered)"
echo "Per-doc cap:  ${PER_DOC_CAP} (0 = disabled)"
echo "=========================================="

COMMON_ARGS=(
    --api-url "$API_URL"
    --api-key "$COMPANY_API_KEY"
    --model "$MODEL"
    --judge-model "$JUDGE_MODEL"
    --enriched-path "$OLD_ENRICHED"
    --min-hop "$MIN_HOP"
    --max-hop "$MAX_HOP"
    --num-samples "$NUM_SAMPLES"
    --per-doc-cap "$PER_DOC_CAP"
    --seed "$SEED"
)

if [[ "$SKIP_B1" != "1" ]]; then
    echo "[Run A] max_bridge_nodes=1"
    python3 -u scripts/pilot_method_c.py \
        "${COMMON_ARGS[@]}" \
        --max-bridge-nodes 1 \
        --output "$OUT_B1"
else
    echo "[Run A] SKIPPED"
fi

if [[ "$SKIP_B2" != "1" ]]; then
    echo "[Run B] max_bridge_nodes=2"
    python3 -u scripts/pilot_method_c.py \
        "${COMMON_ARGS[@]}" \
        --max-bridge-nodes 2 \
        --output "$OUT_B2"
else
    echo "[Run B] SKIPPED"
fi

echo "[Summary] Comparing outputs..."
python3 - <<'PY'
import json
from collections import Counter
from pathlib import Path

paths = [
    ("bridge1", Path("data/03_queries/pilot_method_c_v3_old_full_bridge1.json")),
    ("bridge2", Path("data/03_queries/pilot_method_c_v3_old_full_bridge2.json")),
]
for label, path in paths:
    if not path.exists():
        print(f"{label}: missing ({path})")
        continue
    data = json.loads(path.read_text(encoding="utf-8"))
    rows = [r for r in data.get("results", []) if "qc" in r]
    passed = sum(1 for r in rows if r["qc"].get("passed"))
    issues = Counter()
    for row in rows:
        for issue in row["qc"].get("all_issues", []):
            issues[issue] += 1
    print(f"\n[{label}] total={len(rows)} pass={passed} fail={len(rows)-passed}")
    print(f"  metadata={data.get('metadata', {})}")
    print(f"  top_issues={issues.most_common(6)}")
PY

echo "=========================================="
echo "Method C old-data feasibility finished: $(date)"
echo "Output A: $OUT_B1"
echo "Output B: $OUT_B2"
echo "=========================================="
