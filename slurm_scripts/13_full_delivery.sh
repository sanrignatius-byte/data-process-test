#!/bin/bash
#SBATCH --job-name=m4q_delivery
#SBATCH --partition=cluster02
#SBATCH --qos=msc
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=logs/delivery_%j.out
#SBATCH --error=logs/delivery_%j.err

echo "============================================"
echo "M4query Full Delivery Pipeline"
echo "Job ID: $SLURM_JOB_ID"
echo "Node:   $(hostname)"
echo "Start:  $(date)"
echo "============================================"

cd /projects/myyyx1/data-process-test
source /cluster/apps/software/Miniforge3/24.11.3-1/etc/profile.d/conda.sh
conda activate /projects/myyyx1/envs/minerU

mkdir -p logs

# Load API keys if needed
if [ -f .env ]; then
    source .env
fi

python scripts/build_full_delivery.py

echo "============================================"
echo "Done: $(date)"
echo "============================================"
