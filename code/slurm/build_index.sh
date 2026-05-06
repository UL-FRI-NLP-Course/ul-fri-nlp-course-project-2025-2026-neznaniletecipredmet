#!/bin/bash
#SBATCH --job-name=fri-rag-build-index
#SBATCH --partition=gpu
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --output=logs/build_index_%j.out
#SBATCH --error=logs/build_index_%j.err

set -euo pipefail

# Resolve working directory assuming sbatch is launched from the code/slurm/ directory
# or code/ directory.
if [[ "$SLURM_SUBMIT_DIR" == *"/slurm"* ]]; then
    cd "$SLURM_SUBMIT_DIR/.."
else
    cd "$SLURM_SUBMIT_DIR"
fi
mkdir -p logs
export HF_HOME="$PWD/.hf_cache"
mkdir -p "$HF_HOME"

module load CUDA/12.2.0
module load Python/3.11

if [ ! -d ".venv" ]; then
    python -m venv .venv
fi

source .venv/bin/activate

# install torch from the official website to avoid errors
pip install -r requirements.txt
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121 --upgrade

RUN_NAME="${RUN_NAME:-jtdh_3_1_0}" 
MODE="${MODE:-new}"            # new | update

python scripts/build_index.py \
  --run "$RUN_NAME" \
  --mode "$MODE"
