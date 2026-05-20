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
pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cu121

RUN_NAME="${RUN_NAME:-Test}"
TOP_K="${TOP_K:-4}"
RERANK_CANDIDATE_K="${RERANK_CANDIDATE_K:-20}"

run_retrieval_eval() {
    local label="$1"
    shift

    echo "=== Evaluating retrieval setting: $label ==="
    python scripts/evaluate.py --run "$RUN_NAME" --top-k "$TOP_K" --retrieval-only "$@"

    local results_file="eval/results_retrieval_only.jsonl"
    local target_file="eval/results_${label}.jsonl"
    mv -f "$results_file" "$target_file"
}

run_retrieval_eval dense
run_retrieval_eval hybrid --hybrid
run_retrieval_eval dense_rerank --rerank --rerank-candidate-k "$RERANK_CANDIDATE_K"
run_retrieval_eval hybrid_rerank --hybrid --rerank --rerank-candidate-k "$RERANK_CANDIDATE_K"
