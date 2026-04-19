#!/bin/bash
#SBATCH --job-name=ms_sync
#SBATCH --partition=cluster02
#SBATCH --qos=msc
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=logs/modelscope_sync_%j.out
#SBATCH --error=logs/modelscope_sync_%j.err

echo "============================================"
echo "Job: ModelScope Delivery Sync"
echo "============================================"
echo "Job ID:  $SLURM_JOB_ID"
echo "Node:    $(hostname)"
echo "Start:   $(date)"
echo "============================================"

cd /projects/myyyx1/data-process-test
source /cluster/apps/software/Miniforge3/24.11.3-1/etc/profile.d/conda.sh
conda activate /projects/myyyx1/envs/minerU

mkdir -p logs

python scripts/sync_modelscope_dataset.py \
  --threshold 450 \
  --commit-timeout 900 \
  --commit-retries 5

echo "============================================"
echo "Job Complete: $(date)"
echo "============================================"
