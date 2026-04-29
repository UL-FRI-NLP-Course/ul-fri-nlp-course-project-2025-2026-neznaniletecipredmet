#!/bin/bash
#SBATCH --job-name=fri-rag-evaluate
#SBATCH --partition=gpu
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=40G
#SBATCH --time=24:00:00
#SBATCH --output=logs/evaluate_%j.out
#SBATCH --error=logs/evaluate_%j.err

cd "$SLURM_SUBMIT_DIR/.."
export HF_HOME=/d/hpc/projects/onj_fri/neznani-leteci-predmet/cache

module load CUDA/12.2.0
module load Python/3.11

if [ ! -d ".venv" ]; then
    python -m venv .venv
fi

source .venv/bin/activate

# install torch from the official website to avoid errors
pip install -r requirements.txt
pip install torch==2.1.2 --index-url https://download.pytorch.org/whl/cu118

MODEL_NAME="${MODEL_NAME:-cjvt/GaMS3-12B-Instruct}"
RUN_NAME="${RUN_NAME:-default}"

python scripts/evaluate.py --run "$RUN_NAME" --model "$MODEL_NAME"
