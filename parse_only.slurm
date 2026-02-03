#!/bin/bash
#SBATCH -p cluster02
#SBATCH -C gpu
#SBATCH --job-name=mineru_parse
#SBATCH --output=logs/parse_only_%j.out
#SBATCH --error=logs/parse_only_%j.err
#SBATCH --time=24:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --gpus=4

module load Miniforge3
conda run -n minerU python -c "import sys; print(sys.executable); import aiohttp; print(aiohttp.__version__)"

cd /projects/myyyx1/data-process-test

conda run -n minerU python scripts/parse_only.py --input data/raw_pdfs_remaining --output data/mineru_output --workers 4 --devices cuda:0 cuda:1 cuda:2 cuda:3 --timeout 900
