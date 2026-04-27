#!/bin/bash
#SBATCH --job-name=fri-rag-build-index
#SBATCH --partition=gpu
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=logs/build_index_%j.out
#SBATCH --error=logs/build_index_%j.err

export HF_HOME=FILL_IN_SHARED_CACHE_DIR
module load Python/3.11

if [ ! -d ".venv" ]; then
    python -m venv .venv
fi

source .venv/bin/activate
pip install -r ../requirements.txt

cd "$SLURM_SUBMIT_DIR/.."

python scripts/build_index.py
