#!/bin/bash
#SBATCH --job-name=noncs400_graph
#SBATCH --partition=cluster02
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:pro6000:4
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=2-00:00:00
#SBATCH --output=logs/noncs400_mineru_graph_%j.out
#SBATCH --error=logs/noncs400_mineru_graph_%j.err

set -euo pipefail

echo "=========================================="
echo "noncs400 MinerU + graph started: $(date)"
echo "Job ID: ${SLURM_JOBID:-manual}"
echo "Node:   $(hostname)"
echo "=========================================="

PROJECT_DIR="/projects/myyyx1/data-process-test"
cd "$PROJECT_DIR"
mkdir -p logs

module load Miniforge3 2>/dev/null || true
source /cluster/apps/software/Miniforge3/24.11.3-1/etc/profile.d/conda.sh 2>/dev/null || true
conda activate /projects/myyyx1/envs/minerU

BASE="data/00_raw"
PDF_DIR="${BASE}/noncs400_pdfs"
LATEX_DIR="${BASE}/noncs400_latex_sources"
SUCCESS_IDS="${BASE}/noncs400_success_ids.txt"
MINERU_OUT="${BASE}/noncs400_mineru_output"
LOCKDIR="logs/.noncs400_mineru_locks"
MINERU_BIN="/projects/myyyx1/envs/minerU/bin/mineru"

GRAPH_DIR="data/01_graphs"
MM_ELEMENTS="${GRAPH_DIR}/noncs400_multimodal_elements.json"
MM_REPORT="${GRAPH_DIR}/noncs400_multimodal_report.json"
LATEX_GRAPH="${GRAPH_DIR}/noncs400_latex_reference_graph.json"
LATEX_REPORT="${GRAPH_DIR}/noncs400_latex_reference_report.json"
MM_ELEMENTS_V2="${GRAPH_DIR}/noncs400_multimodal_elements_v2.json"
CITATION_GRAPH="${GRAPH_DIR}/noncs400_citation_graph.json"
TOPO_REPORT="${GRAPH_DIR}/noncs400_latex_graph_topology_report.json"
HUBS="${GRAPH_DIR}/noncs400_latex_graph_hubs.json"
CANDIDATES="${GRAPH_DIR}/noncs400_latex_hub_multihop_candidates.json"

if [ ! -s "$SUCCESS_IDS" ]; then
  echo "ERROR: success id list is missing or empty: $SUCCESS_IDS"
  exit 1
fi

NUM_GPUS=$(nvidia-smi -L 2>/dev/null | wc -l)
if [ "$NUM_GPUS" -eq 0 ]; then
  echo "ERROR: No GPUs detected"
  exit 1
fi
echo "GPUs: $NUM_GPUS"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader || true

mkdir -p "$MINERU_OUT" "$LOCKDIR" "$GRAPH_DIR"

WORKLIST=$(mktemp /tmp/noncs400_mineru_worklist_XXXX.txt)
SKIP=0
MISSING=0
while IFS= read -r aid; do
  [ -n "$aid" ] || continue
  safe="${aid//\//_}"
  pdf="${PDF_DIR}/${safe}.pdf"
  if [ ! -s "$pdf" ]; then
    echo "[WARN] missing PDF for $aid: $pdf"
    MISSING=$((MISSING + 1))
    continue
  fi
  if find "${MINERU_OUT}/${safe}" -name "*.md" -type f 2>/dev/null | head -1 | grep -q .; then
    SKIP=$((SKIP + 1))
    continue
  fi
  echo "$pdf" >> "$WORKLIST"
done < "$SUCCESS_IDS"

TOTAL=$(wc -l < "$WORKLIST")
echo "PDFs to parse: $TOTAL  skipped: $SKIP  missing: $MISSING"

parse_one_pdf() {
  local pdf="$1"
  local gpu_id="$2"
  local bname lockfile doc_output t_start elapsed rc
  bname=$(basename "$pdf" .pdf)
  lockfile="${LOCKDIR}/${bname}.lock"
  doc_output="${MINERU_OUT}/${bname}"

  if ! mkdir "$lockfile" 2>/dev/null; then
    echo "[GPU${gpu_id}] SKIP ${bname} (locked)"
    return 0
  fi
  trap "rmdir '$lockfile' 2>/dev/null" RETURN

  if find "$doc_output" -name "*.md" -type f 2>/dev/null | head -1 | grep -q .; then
    echo "[GPU${gpu_id}] SKIP ${bname} (already parsed)"
    return 0
  fi

  echo "[GPU${gpu_id}] START ${bname}"
  t_start=$SECONDS
  CUDA_VISIBLE_DEVICES="$gpu_id" "$MINERU_BIN" \
    -p "$pdf" \
    -o "$doc_output" \
    -m auto \
    -b pipeline \
    -l en \
    -f True \
    -t True \
    -d cuda \
    2>&1 | sed "s/^/[GPU${gpu_id}] /"
  rc=${PIPESTATUS[0]}
  elapsed=$((SECONDS - t_start))
  if [ "$rc" -eq 0 ]; then
    echo "[GPU${gpu_id}] DONE ${bname} (${elapsed}s)"
  else
    echo "[GPU${gpu_id}] FAIL ${bname} exit=${rc} (${elapsed}s)"
  fi
  return "$rc"
}
export -f parse_one_pdf
export MINERU_BIN MINERU_OUT LOCKDIR

if [ "$TOTAL" -gt 0 ]; then
  if command -v parallel &>/dev/null; then
    parallel --will-cite -j "$NUM_GPUS" --linebuffer --halt soon,fail=20% \
      'parse_one_pdf {1} $(( ({%} - 1) ))' \
      :::: "$WORKLIST" || true
  else
    declare -A slot_pids
    while IFS= read -r pdf; do
      while [ "${#slot_pids[@]}" -ge "$NUM_GPUS" ]; do
        for g in "${!slot_pids[@]}"; do
          if ! kill -0 "${slot_pids[$g]}" 2>/dev/null; then
            wait "${slot_pids[$g]}" 2>/dev/null || true
            unset "slot_pids[$g]"
          fi
        done
        [ "${#slot_pids[@]}" -ge "$NUM_GPUS" ] && sleep 2
      done
      local_gpu=0
      for ((g=0; g<NUM_GPUS; g++)); do
        if [ -z "${slot_pids[$g]+x}" ]; then
          local_gpu=$g
          break
        fi
      done
      parse_one_pdf "$pdf" "$local_gpu" &
      slot_pids[$local_gpu]=$!
    done < "$WORKLIST"
    for g in "${!slot_pids[@]}"; do
      wait "${slot_pids[$g]}" 2>/dev/null || true
    done
  fi
fi
rm -f "$WORKLIST"

PARSED_DOCS=$(find "$MINERU_OUT" -mindepth 1 -maxdepth 1 -type d | wc -l)
echo "Parsed doc directories: $PARSED_DOCS"

echo "=========================================="
echo "Step 2: build MinerU multimodal graph"
echo "=========================================="
python scripts/build_multimodal_relationships.py \
  --mineru-dir "$MINERU_OUT" \
  --output "$MM_ELEMENTS" \
  --report "$MM_REPORT" \
  --max-hops 3 \
  --context-window 3

echo "=========================================="
echo "Step 3: build LaTeX reference graph and merge"
echo "=========================================="
python scripts/build_latex_reference_graph.py \
  --source-dir "${LATEX_DIR}/extracted" \
  --output "$LATEX_GRAPH" \
  --report "$LATEX_REPORT" \
  --max-hops 3 \
  --merge-with "$MM_ELEMENTS" \
  --merged-output "$MM_ELEMENTS_V2"

echo "=========================================="
echo "Step 4: build citation graph"
echo "=========================================="
python scripts/build_citation_graph.py \
  --input "$LATEX_GRAPH" \
  --output "$CITATION_GRAPH"

echo "=========================================="
echo "Step 5: analyze unified topology"
echo "=========================================="
python scripts/analyze_latex_graph_topology.py \
  --latex-graph "$LATEX_GRAPH" \
  --elements "$MM_ELEMENTS_V2" \
  --citation-graph "$CITATION_GRAPH" \
  --mineru-output "$MINERU_OUT" \
  --output-report "$TOPO_REPORT" \
  --output-hubs "$HUBS" \
  --output-candidates "$CANDIDATES" \
  --top-k-hubs 500 \
  --max-candidates 5000

python - <<'PY'
import json
from pathlib import Path
paths = {
    "elements": Path("data/01_graphs/noncs400_multimodal_elements_v2.json"),
    "latex_graph": Path("data/01_graphs/noncs400_latex_reference_graph.json"),
    "citation": Path("data/01_graphs/noncs400_citation_graph.json"),
    "topology": Path("data/01_graphs/noncs400_latex_graph_topology_report.json"),
    "candidates": Path("data/01_graphs/noncs400_latex_hub_multihop_candidates.json"),
}
for name, path in paths.items():
    print(f"{name}: {path} exists={path.exists()} bytes={path.stat().st_size if path.exists() else 0}")
if paths["topology"].exists():
    report = json.loads(paths["topology"].read_text(encoding="utf-8"))
    print("topology nodes:", report.get("density", {}).get("global", {}).get("nodes"))
    print("topology edges:", report.get("density", {}).get("global", {}).get("edges"))
    print("bridge hubs:", report.get("hubs_summary", {}).get("bridge_hubs_count"))
    print("candidates:", report.get("hub_multihop_summary", {}).get("candidate_count"))
PY

echo "=========================================="
echo "noncs400 MinerU + graph finished: $(date)"
echo "=========================================="
