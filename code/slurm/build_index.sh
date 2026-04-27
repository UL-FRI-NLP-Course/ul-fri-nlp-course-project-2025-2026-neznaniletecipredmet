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

cd "$SLURM_SUBMIT_DIR/.."
export HF_HOME=FILL_IN_SHARED_CACHE_DIR


module load CUDA/12.2.0
module load Python/3.11

if [ ! -d ".venv" ]; then
    python -m venv .venv
fi

ls -al .venv/bin
source .venv/bin/activate

# install torch from the official website to avoid errors
pip install -r requirements.txt
pip install torch==2.1.2 --index-url https://download.pytorch.org/whl/cu118

python scripts/build_index.py
