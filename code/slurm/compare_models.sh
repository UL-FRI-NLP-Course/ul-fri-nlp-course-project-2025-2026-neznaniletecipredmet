#!/bin/bash
#SBATCH --job-name=fri-rag-compare
#SBATCH --partition=gpu
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=logs/compare_models_%j.out
#SBATCH --error=logs/compare_models_%j.err

set -euo pipefail

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

pip install -r requirements.txt
TORCH_INDEX_URL="${TORCH_INDEX_URL:-https://download.pytorch.org/whl/cu124}"
TORCH_VERSION="${TORCH_VERSION:-2.6.0}"
pip install --index-url "$TORCH_INDEX_URL" --upgrade \
    torch=="$TORCH_VERSION" torchvision==0.21.0 torchaudio==2.6.0

# Winning RAG configuration from the local sweeps (see eval/SUMMARY.md):
#   embedder       = intfloat/multilingual-e5-base
#   retrieval      = hybrid (BM25 + dense) + cross-encoder rerank
#   top-k          = 4
#   chunk size     = 400 (default)
RUN_NAME="${RUN_NAME:-2026-05-20__full__fri4_ul2__v1__e5_base}"
TOP_K="${TOP_K:-4}"
RERANK_CANDIDATE_K="${RERANK_CANDIDATE_K:-20}"
USE_HYBRID="${USE_HYBRID:-1}"
USE_RERANK="${USE_RERANK:-1}"
LIMIT="${LIMIT:-}"

EXTRA_FLAGS=()
if [[ "$USE_HYBRID" == "1" ]]; then EXTRA_FLAGS+=("--hybrid"); fi
if [[ "$USE_RERANK" == "1" ]]; then EXTRA_FLAGS+=("--rerank" "--rerank-candidate-k" "$RERANK_CANDIDATE_K"); fi
if [[ -n "$LIMIT" ]]; then EXTRA_FLAGS+=("--limit" "$LIMIT"); fi

# Optional but recommended on first run: HF_TOKEN for gated models
# (Llama-3.1-8B-Instruct and some Mistral checkpoints require it).
if [[ -n "${HF_TOKEN:-}" ]]; then
    export HUGGING_FACE_HUB_TOKEN="$HF_TOKEN"
    echo "HF token is set."
fi

MODELS=(
    "Qwen/Qwen2.5-1.5B-Instruct"
    "cjvt/GaMS3-12B-Instruct"
    "meta-llama/Llama-3.1-8B-Instruct"
    "mistralai/Mistral-7B-Instruct-v0.3"
)
if [[ -n "${MODELS_OVERRIDE:-}" ]]; then
    IFS=',' read -r -a MODELS <<< "$MODELS_OVERRIDE"
fi

echo "=== Run:    $RUN_NAME"
echo "=== Top-k:  $TOP_K"
echo "=== Flags:  ${EXTRA_FLAGS[*]}"
echo "=== Models: ${MODELS[*]}"

for MODEL in "${MODELS[@]}"; do
    echo
    echo "============================================================"
    echo "=== Evaluating: $MODEL"
    echo "============================================================"
    python scripts/evaluate.py \
        --run "$RUN_NAME" \
        --model "$MODEL" \
        --top-k "$TOP_K" \
        "${EXTRA_FLAGS[@]}"
done

echo
echo "Done. Per-model outputs are in:"
echo "  data/runs/$RUN_NAME/eval/results_<model_slug>.jsonl"
echo
echo "Next: rsync those back and run scripts/compare_generators.py to LLM-judge them."
