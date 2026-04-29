#!/bin/bash
#SBATCH --job-name=fri-rag-evaluate
#SBATCH --partition=FILL_IN
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=40G
#SBATCH --time=04:00:00
#SBATCH --output=logs/evaluate_%j.out
#SBATCH --error=logs/evaluate_%j.err

export HF_HOME=/d/hpc/projects/onj_fri/neznani-leteci-predmet/cache

source FILL_IN_VENV_ACTIVATE_PATH

cd "$(dirname "$0")/.."

MODEL_NAME="${MODEL_NAME:-cjvt/GaMS3-12B-Instruct}"
RUN_NAME="${RUN_NAME:-default}"

python scripts/evaluate.py --run "$RUN_NAME" --model "$MODEL_NAME"
