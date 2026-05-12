#!/bin/bash
#SBATCH --job-name=m2_m15_reason_mix
#SBATCH --partition=cluster02
#SBATCH --qos=msc
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=24:00:00
#SBATCH --output=logs/m2_m15_reason_mix_%j.out
#SBATCH --error=logs/m2_m15_reason_mix_%j.err

set -euo pipefail

export STYLE=${STYLE:-mixed}
export USE_PERSONA=${USE_PERSONA:-1}
export FORCE_REASONING=${FORCE_REASONING:-1}
export PREPARED_CANDIDATES=${PREPARED_CANDIDATES:-data/05_eval/m2/m2_m15_reasoning_path_enriched_intradoc_20260511_mixed.json}
export OUT_DIR=${OUT_DIR:-data/03_queries/m2_m15_reasoning_path_mixed_20260511}
export OUTPUT=${OUTPUT:-${OUT_DIR}/m2_m15_reasoning_path.jsonl}

bash slurm_scripts/53_m2_m15_reasoning_path_production.sh
