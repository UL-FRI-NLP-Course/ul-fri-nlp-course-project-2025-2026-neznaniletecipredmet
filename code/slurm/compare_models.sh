#!/bin/bash
#SBATCH --job-name=fri-rag-compare
#SBATCH --partition=gpu
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=40G
#SBATCH --time=12:00:00
#SBATCH --output=logs/compare_models_%j.out
#SBATCH --error=logs/compare_models_%j.err

cd "$SLURM_SUBMIT_DIR/.."
export HF_HOME=/d/hpc/projects/onj_fri/neznani-leteci-predmet/cache

module load CUDA/12.2.0
module load Python/3.11

if [ ! -d ".venv" ]; then
    python -m venv .venv
fi

source .venv/bin/activate

cd "$(dirname "$0")/.."

RUN_NAME="${RUN_NAME:-default}"

for MODEL in "cjvt/GaMS3-12B-Instruct" "meta-llama/Llama-3.1-8B-Instruct" "mistralai/Mistral-7B-Instruct-v0.3"; do
    echo "=== Evaluating: $MODEL ==="
    python scripts/evaluate.py --run "$RUN_NAME" --model "$MODEL"
done
