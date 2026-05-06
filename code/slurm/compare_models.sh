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
pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cu121

RUN_NAME="${RUN_NAME:-Test}"

for MODEL in "Qwen/Qwen2.5-1.5B-Instruct" "cjvt/GaMS3-12B-Instruct" "meta-llama/Llama-3.1-8B-Instruct" "mistralai/Mistral-7B-Instruct-v0.3"; do
    echo "=== Evaluating: $MODEL ==="
    python scripts/evaluate.py --run "$RUN_NAME" --model "$MODEL"
done
