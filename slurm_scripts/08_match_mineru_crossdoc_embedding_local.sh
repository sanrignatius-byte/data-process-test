#!/bin/bash
#SBATCH -p cluster02
#SBATCH -C gpu
#SBATCH --job-name=mineru_emb_match
#SBATCH --output=logs/mineru_emb_match_%j.out
#SBATCH --error=logs/mineru_emb_match_%j.err
#SBATCH --time=12:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gpus=1

# ============================================================
# Step 8: Local embedding-based cross-document MinerU matching
#
# Defaults: high-performance Qwen3-Embedding-8B
# Env overrides:
#   MODEL_ID=Qwen/Qwen3-Embedding-4B
#   MODEL_DIR=/projects/myyyx1/data-process-test/models/Qwen3-Embedding-4B
#   TOPK=20 BATCH_SIZE=8 MAX_LEN=512 MAX_ELEMENTS=0
#   QUERIES=data/l1_dual_evidence_queries_slurm_img150_tuned_v4_official.jsonl
# ============================================================

set -euo pipefail

module load Miniforge3

REPO_ROOT=${REPO_ROOT:-/projects/myyyx1/data-process-test}
CONDA_ENV=${CONDA_ENV:-/projects/myyyx1/envs/minerU}
MODEL_ID=${MODEL_ID:-Qwen/Qwen3-Embedding-8B}
MODEL_TAG=${MODEL_TAG:-$(basename "$MODEL_ID")}
MODEL_DIR=${MODEL_DIR:-${REPO_ROOT}/models/${MODEL_TAG}}

TOPK=${TOPK:-20}
BATCH_SIZE=${BATCH_SIZE:-8}
MAX_LEN=${MAX_LEN:-512}
MAX_ELEMENTS=${MAX_ELEMENTS:-0}
QUERIES=${QUERIES:-data/l1_dual_evidence_queries_slurm_img150_tuned_v4_official.jsonl}
ELEMENTS=${ELEMENTS:-data/multimodal_elements.json}
OUTPUT=${OUTPUT:-data/mineru_crossdoc_embedding_matches_${MODEL_TAG}.jsonl}
REPORT=${REPORT:-data/mineru_crossdoc_embedding_matches_${MODEL_TAG}_report.json}
DOWNLOAD_RETRIES=${DOWNLOAD_RETRIES:-5}
HF_HUB_ETAG_TIMEOUT=${HF_HUB_ETAG_TIMEOUT:-120}
HF_HUB_DOWNLOAD_TIMEOUT=${HF_HUB_DOWNLOAD_TIMEOUT:-120}

mkdir -p "${REPO_ROOT}/logs" "${REPO_ROOT}/models" "${REPO_ROOT}/data"
cd "$REPO_ROOT"

echo "=========================================="
echo "Start: $(date)"
echo "Node:  $(hostname)"
echo "Model: ${MODEL_ID}"
echo "Path:  ${MODEL_DIR}"
echo "=========================================="

export CONDA_NO_PLUGINS=true
CONDA_BASE=${CONDA_BASE:-/cluster/apps/software/Miniforge3/24.11.3-1}
source "$CONDA_BASE/etc/profile.d/conda.sh"
conda activate "$CONDA_ENV"
export HF_HUB_ETAG_TIMEOUT
export HF_HUB_DOWNLOAD_TIMEOUT

echo "Python: $(which python)"
python - <<'PY'
import importlib
mods=['torch','transformers','huggingface_hub']
for m in mods:
    try:
        importlib.import_module(m)
        print('OK',m)
    except Exception as e:
        print('MISSING',m,str(e))
        raise
PY

echo ""
echo "[1/2] Download/check embedding model..."
python - <<PY
from pathlib import Path
import time
from huggingface_hub import snapshot_download

model_id = "${MODEL_ID}"
model_dir = Path("${MODEL_DIR}")
model_dir.mkdir(parents=True, exist_ok=True)
max_retries = int("${DOWNLOAD_RETRIES}")

def has_weights(p: Path) -> bool:
    if (p / "model.safetensors").exists():
        return True
    if list(p.glob("model-*-of-*.safetensors")):
        return True
    return False

if has_weights(model_dir):
    print(f"Model weights already present: {model_dir}")
else:
    print(f"Downloading {model_id} -> {model_dir}")
    last_err = None
    for i in range(1, max_retries + 1):
        try:
            print(f"Attempt {i}/{max_retries}")
            snapshot_download(
                repo_id=model_id,
                local_dir=str(model_dir),
                local_dir_use_symlinks=False,
                resume_download=True,
                max_workers=2,
            )
            if has_weights(model_dir):
                print("Download done.")
                last_err = None
                break
        except Exception as e:
            last_err = e
            print(f"Attempt {i} failed: {e}")
            time.sleep(min(30, 5 * i))
    if last_err is not None and not has_weights(model_dir):
        raise SystemExit(f"Failed to download model after {max_retries} attempts: {last_err}")
PY

echo ""
echo "[2/2] Run cross-document embedding matching..."
python scripts/match_mineru_crossdoc_with_embeddings.py \
  --backend local \
  --local-model-path "${MODEL_DIR}" \
  --model "${MODEL_ID}" \
  --elements "${ELEMENTS}" \
  --queries "${QUERIES}" \
  --output "${OUTPUT}" \
  --report "${REPORT}" \
  --top-k "${TOPK}" \
  --batch-size "${BATCH_SIZE}" \
  --local-max-length "${MAX_LEN}" \
  --max-elements "${MAX_ELEMENTS}" \
  --modalities figure table formula

echo ""
echo "Done: $(date)"
echo "Output: ${OUTPUT}"
echo "Report: ${REPORT}"
echo "=========================================="
