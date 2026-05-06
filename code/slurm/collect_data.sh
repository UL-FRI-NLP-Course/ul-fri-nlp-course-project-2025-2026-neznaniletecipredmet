#!/bin/bash
#SBATCH --job-name=fri-rag-collect-data
#SBATCH --partition=all
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=06:00:00
#SBATCH --output=logs/collect_data_%j.out
#SBATCH --error=logs/collect_data_%j.err

set -euo pipefail

# Resolve working directory assuming sbatch is launched from the code/slurm/ directory
# or code/ directory.
if [[ "$SLURM_SUBMIT_DIR" == *"/slurm"* ]]; then
    cd "$SLURM_SUBMIT_DIR/.."
else
    cd "$SLURM_SUBMIT_DIR"
fi
mkdir -p logs

module load Python/3.11

if [ ! -d ".venv" ]; then
    python -m venv .venv
fi

source .venv/bin/activate

# Install (or reuse) dependencies for the crawler.
# This repo uses a single requirements.txt for the whole pipeline.
python -m pip install --upgrade pip
pip install -r requirements.txt

RUN_NAME="${RUN_NAME:-jtdh_3_1_0}" 
MODE="${MODE:-update}"            # new | update
DEPTH_FRI="${DEPTH_FRI:-2}"
DEPTH_UL="${DEPTH_UL:-1}"
DEPTH_V="${DEPTH_V:-0}"

python scripts/collect_data.py \
  --run "$RUN_NAME" \
  --mode "$MODE" \
  --depth-fri "$DEPTH_FRI" \
  --depth-ul "$DEPTH_UL" \
  --depth-v "$DEPTH_V"
