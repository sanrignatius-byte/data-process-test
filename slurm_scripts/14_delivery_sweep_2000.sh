#!/bin/bash
#SBATCH -p cluster02
#SBATCH --qos=msc
#SBATCH --job-name=deliv2000
#SBATCH --output=logs/deliv2000_%j.out
#SBATCH --error=logs/deliv2000_%j.err
#SBATCH --time=24:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=14
#SBATCH --mem=64G

# ============================================================
# Delivery sweep 2026-05-12 → 2000 queries by Thu morning
#
# HOTFIX (2026-05-12 vs initial PR #175):
#   1. Removed --array=0-6. Now single sbatch job with 7-way bash & fork
#      because the project data lives on /projects/_hdd/... which (per
#      mentor) is not uniformly mountable across all compute nodes. A
#      single job pins all 7 workers to one node = guaranteed HDD mount.
#   2. Fixed REPO_ROOT to /projects/_hdd/myyyx1/data-process-test
#      (matches slurm 53/55/56; original PR #175 used non-HDD prefix
#      and would have failed to find data files).
#   3. Bumped cpus 4→14 and mem 16G→64G to accommodate 7 parallel Python
#      processes each loading multimodal_elements_v2 + reference_graph_v2.
#
# 7-cell concurrent on 1077 unconsumed enriched intra-doc candidates.
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
#   - DELAY=0.1 (was 0.5)              → 3× faster per cell
#   - 7 bash-forked cells × 1 API key  → 7× parallel fan-out (same node)
#   - Each cell uses --skip-done       → safe to re-submit if interrupted
#
# Prerequisites:
#   python3 scripts/prep_delivery_chunks.py   # creates 7 chunk files
#   COMPANY_API_KEY + COMPANY_API_URL in .env
#
# Usage:
#   sbatch slurm_scripts/14_delivery_sweep_2000.sh
#
# Per-cell logs go to logs/deliv2000_cell${IDX}.log so the main
# stdout/stderr stays as a roll-up; tail individual cells via:
#   tail -50 logs/deliv2000_cell0.log
#
# Monitoring (DO NOT enter the running job's terminal):
#   tail -50 logs/deliv2000_cell*.log
#   wc -l data/03_queries/delivery_sweep_2000/*_pass.jsonl
# ============================================================

set -uo pipefail
# Note: NOT using -e because we want all 7 cells to run independently
# even if one fails; we collect exit codes at the end instead.

REPO_ROOT=${REPO_ROOT:-/projects/_hdd/myyyx1/data-process-test}
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

# ── Cell config table ─────────────────────────────────────────────────────────
# Indexed by cell number. Format: CAND_RAW|STYLE|PERSONA|FORCE_L3|TAG
declare -a CELL_CONFIG=(
    "$CHUNK_DIR/cell_0_methodc_a.json|mixed|--use-persona|0|cell0_methodc_a_m2_mp"
    "$CHUNK_DIR/cell_1_methodc_b.json|mixed|--use-persona|0|cell1_methodc_b_m2_mp"
    "$CHUNK_DIR/cell_2_methodc_c.json|mixed||0|cell2_methodc_c_m2_mixed"
    "$CHUNK_DIR/cell_3_methodc_d.json|academic||0|cell3_methodc_d_m2_academic"
    "$CHUNK_DIR/cell_4_hub_v4.json|mixed|--use-persona|0|cell4_hubv4_m2_mp"
    "$CHUNK_DIR/cell_5_m2_diverse.json|mixed|--use-persona|0|cell5_m2div_m2_mp"
    "$CHUNK_DIR/cell_6_long_seed.json|mixed||1|cell6_longseed_l3_mixed"
)

# ── Cell worker function ──────────────────────────────────────────────────────
run_cell() {
    local IDX=$1
    IFS='|' read -r CAND_RAW STYLE PERSONA FORCE_L3 TAG <<< "${CELL_CONFIG[$IDX]}"

    local CAND_FILTERED="${CAND_RAW%.json}_filtered_${TAG}.json"
    local OUTPUT="$OUT_DIR/${TAG}.jsonl"
    local CELL_LOG="logs/deliv2000_cell${IDX}.log"

    {
        echo "=========================================="
        echo "Cell $IDX → $TAG"
        echo "Start:           $(date)"
        echo "Host:            $(hostname)"
        echo "Pid:             $$"
        echo "Candidate raw:   $CAND_RAW"
        echo "Candidate filt:  $CAND_FILTERED"
        echo "Output:          $OUTPUT"
        echo "Style:           $STYLE"
        echo "Use persona:     ${PERSONA:-(off)}"
        echo "Force L3:        $FORCE_L3"
        echo "Delay:           $DELAY"
        echo "=========================================="
    } > "$CELL_LOG" 2>&1

    # Filter
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

    echo "[$(date)] Cell $IDX: filtering candidates..." >> "$CELL_LOG" 2>&1
    "${FILTER_CMD[@]}" >> "$CELL_LOG" 2>&1
    local FILTER_RC=$?
    if [[ $FILTER_RC -ne 0 ]]; then
        echo "[$(date)] Cell $IDX: filter FAILED (rc=$FILTER_RC)" >> "$CELL_LOG" 2>&1
        return $FILTER_RC
    fi

    # Generate
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

    echo "[$(date)] Cell $IDX: generation start..." >> "$CELL_LOG" 2>&1
    "${GEN_CMD[@]}" >> "$CELL_LOG" 2>&1
    local GEN_RC=$?

    local PASS_FILE="${OUTPUT%.jsonl}_pass.jsonl"
    local TOTAL=$(wc -l < "$OUTPUT" 2>/dev/null || echo 0)
    local PASS=$(wc -l < "$PASS_FILE" 2>/dev/null || echo 0)
    {
        echo "=========================================="
        echo "Cell $IDX ($TAG) complete: $(date)  rc=$GEN_RC"
        echo "  Total written: $TOTAL"
        echo "  QC pass:       $PASS"
        echo "  Output:        $OUTPUT"
        echo "  Pass file:     $PASS_FILE"
        echo "=========================================="
    } >> "$CELL_LOG" 2>&1

    return $GEN_RC
}

# ── Launch 7 cells in parallel ────────────────────────────────────────────────
echo "[$(date)] Launching 7-way concurrent on host $(hostname)"
echo "[$(date)] Cell logs: logs/deliv2000_cell{0..6}.log"

declare -a PIDS
for IDX in 0 1 2 3 4 5 6; do
    run_cell $IDX &
    PIDS[$IDX]=$!
    echo "  cell $IDX started, pid=${PIDS[$IDX]}"
done

# Wait for all cells; collect exit codes
echo "[$(date)] All 7 cells running. Waiting for completion..."
declare -a RCS
for IDX in 0 1 2 3 4 5 6; do
    wait ${PIDS[$IDX]}
    RCS[$IDX]=$?
    echo "[$(date)] cell $IDX finished, rc=${RCS[$IDX]}"
done

# ── Roll-up summary ───────────────────────────────────────────────────────────
echo ""
echo "=========================================="
echo "DELIVERY SWEEP ROLL-UP — $(date)"
echo "=========================================="
TOTAL_PASS=0
for IDX in 0 1 2 3 4 5 6; do
    IFS='|' read -r _ _ _ _ TAG <<< "${CELL_CONFIG[$IDX]}"
    PASS_FILE="$OUT_DIR/${TAG}_pass.jsonl"
    N=$(wc -l < "$PASS_FILE" 2>/dev/null || echo 0)
    TOTAL_PASS=$((TOTAL_PASS + N))
    printf "  cell %d  %-35s  rc=%d  pass=%d\n" "$IDX" "$TAG" "${RCS[$IDX]}" "$N"
done
echo "------------------------------------------"
echo "  Total new pass this run: $TOTAL_PASS"
echo "=========================================="
