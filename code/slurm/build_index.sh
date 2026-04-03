#!/bin/bash
#SBATCH --job-name=fri-rag-build-index
#SBATCH --partition=FILL_IN
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=logs/build_index_%j.out
#SBATCH --error=logs/build_index_%j.err

export HF_HOME=FILL_IN_SHARED_CACHE_DIR

source FILL_IN_VENV_ACTIVATE_PATH

cd "$(dirname "$0")/.."

python scripts/build_index.py
