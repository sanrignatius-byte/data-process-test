#!/bin/bash
#SBATCH -p cluster02
#SBATCH --qos=msc
#SBATCH --job-name=deliv2000
#SBATCH --output=logs/deliv2000_%A_%a.out
#SBATCH --error=logs/deliv2000_%A_%a.err
#SBATCH --time=24:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --array=0-6

# ============================================================
# Delivery sweep 2026-05-12 → 2000 queries by Thu morning
#
# 7-cell array on 1077 unconsumed enriched intra-doc candidates.
# M2 (dual-evidence, 2 q/pair, 39-47% pass) is the workhorse;
# L3 only on long-seed cell to harvest reasoning-chain diversity.
#
# Per-cell expected yield (M2 mixed_persona baseline 46.7% × 2 q/pair):
#   Cell 0  method_c_true2[0:205]      M2 mixed_persona  → ~190 pass
#   Cell 1  method_c_true2[205:410]    M2 mixed_persona  → ~190 pass
#   Cell 2  method_c_true2[410:615]    M2 mixed          → ~185 pass
#   Cell 3  method_c_true2[615:817]    M2 academic       → ~160 pass
#   Cell 4  hub_v4_intra_doc (96)      M2 mixed_persona  → ~88  pass
#   Cell 5  m2_diverse_intra (108)     M2 mixed_persona  → ~100 pass
#   Cell 6  hub_v4_long_seed (88)      L3 mixed          → ~50  pass
#   ─────────────────────────────────────────────────────────────
#   Round 1 total expected:                              ~963 pass
#   + existing 563 unique pass                          = ~1526
#
#   Gap to 2000 covered by slurm 15 (v2 enrich) + Round 2.
#
# Concurrency knobs (vs slurm 53):
#   - DELAY=0.1 (was 0.5)         → 3× faster per cell
#   - 7 cells × 1 API key         → 7× parallel fan-out
#   - Each cell uses --skip-done  → safe to re-submit if interrupted
#
# Prerequisites:
#   python3 scripts/prep_delivery_chunks.py   # creates 7 chunk files
#   COMPANY_API_KEY + COMPANY_API_URL in .env
#
# Usage:
#   sbatch slurm_scripts/14_delivery_sweep_2000.sh
#   sbatch --array=0,4 slurm_scripts/14_delivery_sweep_2000.sh   # subset
#
# Monitoring (DO NOT enter the running job's terminal):
#   tail -50 logs/deliv2000_*.out
#   wc -l data/03_queries/delivery_sweep_2000/cell_*_pass.jsonl
# ============================================================

set -euo pipefail

REPO_ROOT=${REPO_ROOT:-/projects/myyyx1/data-process-test}
CONDA_ENV=${CONDA_ENV:-/projects/myyyx1/envs/minerU}

PROVIDER=${PROVIDER:-company}
MODEL=${MODEL:-gpt-5.4}
DELAY=${DELAY:-0.1}

CHUNK_DIR="data/02_enriched/delivery_chunks_20260512"
OUT_DIR="data/03_queries/delivery_sweep_2000"
REFERENCE_GRAPH=${REFERENCE_GRAPH:-data/01_graphs/latex_reference_graph_v2.json}
TOPOLOGY_CANDIDATES=${TOPOLOGY_CANDIDATES:-data/01_graphs/latex_hub_multihop_candidates_v2.json}

cd "$REPO_ROOT"
mkdir -p logs "$OUT_DIR"

source /cluster/apps/software/Miniforge3/24.11.3-1/etc/profile.d/conda.sh
conda activate "$CONDA_ENV"

if [[ -f .env ]]; then
    set -a
    source .env
    set +a
fi

if [[ -z "${COMPANY_API_KEY:-}" || -z "${COMPANY_API_URL:-}" ]]; then
    echo "ERROR: COMPANY_API_KEY / COMPANY_API_URL not set"
    exit 1
fi

export PYTHONPATH="$REPO_ROOT:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1

# Auto-prep chunks if missing
if [[ ! -d "$CHUNK_DIR" || -z "$(ls -A "$CHUNK_DIR" 2>/dev/null)" ]]; then
    echo "[$(date)] Running prep_delivery_chunks.py..."
    python3 scripts/prep_delivery_chunks.py
fi

# ── Config array ─────────────────────────────────────────────────────────────
IDX=${SLURM_ARRAY_TASK_ID:-0}

# (chunk_file, style, persona-flag, force-L3-flag, tag)
case $IDX in
    0) CAND_RAW="$CHUNK_DIR/cell_0_methodc_a.json"  STYLE="mixed"    PERSONA="--use-persona"   FORCE_L3=0 TAG="cell0_methodc_a_m2_mp" ;;
    1) CAND_RAW="$CHUNK_DIR/cell_1_methodc_b.json"  STYLE="mixed"    PERSONA="--use-persona"   FORCE_L3=0 TAG="cell1_methodc_b_m2_mp" ;;
    2) CAND_RAW="$CHUNK_DIR/cell_2_methodc_c.json"  STYLE="mixed"    PERSONA=""                FORCE_L3=0 TAG="cell2_methodc_c_m2_mixed" ;;
    3) CAND_RAW="$CHUNK_DIR/cell_3_methodc_d.json"  STYLE="academic" PERSONA=""                FORCE_L3=0 TAG="cell3_methodc_d_m2_academic" ;;
    4) CAND_RAW="$CHUNK_DIR/cell_4_hub_v4.json"     STYLE="mixed"    PERSONA="--use-persona"   FORCE_L3=0 TAG="cell4_hubv4_m2_mp" ;;
    5) CAND_RAW="$CHUNK_DIR/cell_5_m2_diverse.json" STYLE="mixed"    PERSONA="--use-persona"   FORCE_L3=0 TAG="cell5_m2div_m2_mp" ;;
    6) CAND_RAW="$CHUNK_DIR/cell_6_long_seed.json"  STYLE="mixed"    PERSONA=""                FORCE_L3=1 TAG="cell6_longseed_l3_mixed" ;;
    *) echo "ERROR: unknown array index $IDX"; exit 1 ;;
esac

CAND_FILTERED="${CAND_RAW%.json}_filtered_${TAG}.json"
OUTPUT="$OUT_DIR/${TAG}.jsonl"

echo "=========================================="
echo "Delivery sweep cell $IDX → $TAG"
echo "Start:           $(date)"
echo "Host:            $(hostname)"
echo "Candidate raw:   $CAND_RAW"
echo "Candidate filt:  $CAND_FILTERED"
echo "Output:          $OUTPUT"
echo "Style:           $STYLE"
echo "Use persona:     ${PERSONA:-(off)}"
echo "Force L3:        $FORCE_L3"
echo "Delay:           $DELAY"
echo "=========================================="

# ── Filter (dedup against previously-consumed pair_ids) ──────────────────────
FILTER_CMD=(
    python scripts/filter_enriched_pair_candidates.py
    --input "$CAND_RAW"
    --output "$CAND_FILTERED"
    --multimodal-counts 2,3
    --require-both-endpoints
    --require-all-multimodal-elements
    --require-candidate-bridge-text
    --shuffle
    --exclude-query-jsonl data/03_queries/delivery_v1_2026-04-13_intra_doc.jsonl
    --exclude-query-jsonl data/03_queries/m2_m15_reasoning_path_prod_20260511/m2_m15_reasoning_path_pass.jsonl
    --exclude-query-jsonl data/03_queries/m2_diverse_v1_hub_kb.jsonl
    --exclude-query-jsonl data/03_queries/long_chain_iterative_pass.jsonl
)

if [[ "$FORCE_L3" == "1" ]]; then
    FILTER_CMD+=(--force-reasoning-chain-target)
fi

echo "[$(date)] Filtering candidates..."
"${FILTER_CMD[@]}"

# ── Generate ─────────────────────────────────────────────────────────────────
GEN_CMD=(
    python scripts/generate_multihop_l1_queries.py
    --candidates "$CAND_FILTERED"
    --output "$OUTPUT"
    --pass-only
    --provider "$PROVIDER"
    --model "$MODEL"
    --query-style "$STYLE"
    --reference-graph "$REFERENCE_GRAPH"
    --topology-candidates "$TOPOLOGY_CANDIDATES"
    --skip-done "$OUTPUT"
    --shuffle
    --delay "$DELAY"
)
if [[ -n "$PERSONA" ]]; then
    GEN_CMD+=($PERSONA)
fi

echo "[$(date)] Generation start..."
"${GEN_CMD[@]}"

PASS_FILE="${OUTPUT%.jsonl}_pass.jsonl"
TOTAL=$(wc -l < "$OUTPUT" 2>/dev/null || echo 0)
PASS=$(wc -l < "$PASS_FILE" 2>/dev/null || echo 0)

echo "=========================================="
echo "Cell $IDX ($TAG) complete: $(date)"
echo "  Total written: $TOTAL"
echo "  QC pass:       $PASS"
echo "  Output:        $OUTPUT"
echo "  Pass file:     $PASS_FILE"
echo "=========================================="
