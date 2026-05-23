#!/bin/bash
#SBATCH --job-name=fri-rag-ask
#SBATCH --partition=gpu
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=logs/ask_%j.out
#SBATCH --error=logs/ask_%j.err

set -euo pipefail

WORK_DIR="${SLURM_SUBMIT_DIR:-$PWD}"
if [[ -f "$WORK_DIR/requirements.txt" ]]; then
    cd "$WORK_DIR"
elif [[ -f "$WORK_DIR/code/requirements.txt" ]]; then
    cd "$WORK_DIR/code"
elif [[ "$WORK_DIR" == *"/slurm" ]]; then
    cd "$WORK_DIR/.."
else
    cd "$WORK_DIR"
fi

mkdir -p logs
export HF_HOME="$PWD/.hf_cache"
mkdir -p "$HF_HOME"
export HF_HUB_DISABLE_PROGRESS_BARS=1
export TRANSFORMERS_VERBOSITY=error
export TRANSFORMERS_NO_ADVISORY_WARNINGS=1
export PIP_DISABLE_PIP_VERSION_CHECK=1

module load CUDA/12.2.0
module load Python/3.11

if [[ ! -d ".venv" ]]; then
    python -m venv .venv
fi

source .venv/bin/activate

PIP_LOG="${PIP_LOG:-$PWD/logs/pip_install.log}"
if ! pip install -q -r requirements.txt >"$PIP_LOG" 2>&1; then
    cat "$PIP_LOG"
    exit 1
fi

# If a model is requested, ensure torch with CUDA is installed
if [[ -n "${MODEL_NAME:-}" ]]; then
    TORCH_INDEX_URL="${TORCH_INDEX_URL:-https://download.pytorch.org/whl/cu124}"
    TORCH_VERSION="${TORCH_VERSION:-2.6.0}"
    if ! pip install -q --index-url "$TORCH_INDEX_URL" --upgrade \
        torch=="$TORCH_VERSION" torchvision==0.21.0 torchaudio==2.6.0 >"$PIP_LOG" 2>&1; then
        cat "$PIP_LOG"
        exit 1
    fi
fi

# Inputs / environment knobs
INDEX_RUN="${INDEX_RUN:-2026-05-20__full__fri4_ul2__v1}"
# Default index path is constructed relative to the original submit dir (WORK_DIR)
INDEX_DIR="${INDEX_DIR:-$WORK_DIR/code/data/runs/$INDEX_RUN/index}"
QUESTION="${QUESTION:-}"
MODEL_NAME="${MODEL_NAME:-}"
TOP_K="${TOP_K:-5}"
RERANK_CANDIDATE_K="${RERANK_CANDIDATE_K:-30}"
NO_RERANK="${NO_RERANK:-0}"
NO_HYBRID="${NO_HYBRID:-0}"

if [[ -z "$QUESTION" ]]; then
    echo "ERROR: set QUESTION environment variable to the question text (or edit the script)."
    exit 2
fi

HF_HUB_TOKEN_VALUE="${HF_TOKEN:-${HUGGING_FACE_HUB_TOKEN:-}}"
if [[ -n "$HF_HUB_TOKEN_VALUE" ]]; then
    export HF_TOKEN="$HF_HUB_TOKEN_VALUE"
    export HUGGING_FACE_HUB_TOKEN="$HF_HUB_TOKEN_VALUE"
    export HF_HUB_TOKEN="$HF_HUB_TOKEN_VALUE"
    export HUGGINGFACE_HUB_TOKEN="$HF_HUB_TOKEN_VALUE"
    echo "Using Hugging Face token from environment."
fi

echo
echo "===================================================================="
echo "Ask job"
echo "===================================================================="
echo "Index dir:     $INDEX_DIR"
echo "Question:      $QUESTION"
if [[ -n "$MODEL_NAME" ]]; then
    echo "Model:         $MODEL_NAME"
else
    echo "Model:         (none — retrieval only)"
fi
echo "Top-k:         $TOP_K"
echo "Rerank:        $( [[ "$NO_RERANK" == "1" ]] && echo "disabled" || echo "enabled" )"
echo "Hybrid:        $( [[ "$NO_HYBRID" == "1" ]] && echo "dense-only" || echo "BM25+dense" )"
echo "===================================================================="
echo

CMD=(python "$WORK_DIR/code/scripts/ask.py" --index-dir "$INDEX_DIR" --question "$QUESTION" --top-k "$TOP_K" --rerank-candidate-k "$RERANK_CANDIDATE_K")
if [[ "$NO_RERANK" == "1" ]]; then
    CMD+=(--no-rerank)
fi
if [[ "$NO_HYBRID" == "1" ]]; then
    CMD+=(--no-hybrid)
fi
if [[ -n "$MODEL_NAME" ]]; then
    CMD+=(--model "$MODEL_NAME")
fi

"${CMD[@]}"

echo
echo "Done."
