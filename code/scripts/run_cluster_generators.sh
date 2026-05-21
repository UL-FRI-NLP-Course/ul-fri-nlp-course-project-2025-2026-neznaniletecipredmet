#!/usr/bin/env bash
# Submit the remaining two RAG-generator evaluations to the cluster.
#
# This wraps slurm/compare_models.sh with the right env vars so a teammate
# only has to:
#   1) clone the eval-pipeline branch,
#   2) request access to the gated models on Hugging Face,
#   3) export HF_TOKEN,
#   4) run this script from inside `code/`.
#
# What it does:
#   * makes sure the cleaned eval set is materialised at
#     data/runs/<RUN>/eval/questions.jsonl (regenerates if missing).
#   * sbatches slurm/compare_models.sh with MODELS_OVERRIDE set to
#     `meta-llama/Llama-3.1-8B-Instruct,mistralai/Mistral-7B-Instruct-v0.3`.
#   * keeps the same retrieval config used in the report
#     (k=4, BM25+dense hybrid, cross-encoder rerank).
#
# Models tested by default (override with MODELS env var):
#   meta-llama/Llama-3.1-8B-Instruct
#   mistralai/Mistral-7B-Instruct-v0.3
#
# Both are GATED on Hugging Face. Before running:
#   1) Sign in at https://huggingface.co
#   2) Visit each model page and click "Agree and access repository":
#        https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct
#        https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.3
#      (Both are usually approved within a few minutes.)
#   3) Generate a token at https://huggingface.co/settings/tokens
#      (Read scope is enough). Export it as HF_TOKEN.
#
# Usage on the cluster:
#   git clone -b eval-pipeline https://github.com/UL-FRI-NLP-Course/ul-fri-nlp-course-project-2025-2026-neznaniletecipredmet.git fri-rag
#   cd fri-rag/code
#   export HF_TOKEN=hf_xxxxxxxxxxxxxxxxxxxxxx
#   ./scripts/run_cluster_generators.sh
#
# Optional env overrides:
#   RUN_NAME   defaults to 2026-05-20__full__fri4_ul2__v1
#   MODELS     comma-separated override list of HF model ids
#   LIMIT      cap questions for a smoke test (e.g. LIMIT=3)
#   TOP_K      retrieved chunks per question (default 4)
#
# After the SLURM job finishes the per-model results land in:
#   data/runs/<RUN>/eval/results_meta-llama_Llama-3.1-8B-Instruct.jsonl
#   data/runs/<RUN>/eval/results_mistralai_Mistral-7B-Instruct-v0.3.jsonl
#
# Send those two files back to Luka (or rsync to your laptop), then the
# LLM-as-judge step on the laptop is:
#   python scripts/compare_generators.py \
#       --run <RUN> \
#       --reference-file questions_full_v2.json \
#       --judge-model claude-haiku-4-5

set -euo pipefail

DEFAULT_MODELS="meta-llama/Llama-3.1-8B-Instruct,mistralai/Mistral-7B-Instruct-v0.3"
RUN="${RUN_NAME:-2026-05-20__full__fri4_ul2__v1}"
MODELS="${MODELS:-$DEFAULT_MODELS}"
LIMIT_ARG=""
TOP_K_ARG=""

if [[ -n "${LIMIT:-}" ]]; then
    LIMIT_ARG="LIMIT=$LIMIT"
fi
if [[ -n "${TOP_K:-}" ]]; then
    TOP_K_ARG="TOP_K=$TOP_K"
fi

if [[ -z "${HF_TOKEN:-}" ]]; then
    cat >&2 <<EOF
ERROR: HF_TOKEN is not set.

Both Llama-3.1-8B-Instruct and Mistral-7B-Instruct-v0.3 are gated on
Hugging Face. To run them you need a token from a Hugging Face account
that already has access:

  1. Sign in at  https://huggingface.co
  2. Click 'Agree and access repository' on each model page:
       https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct
       https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.3
  3. Generate a token (Read scope) at:
       https://huggingface.co/settings/tokens
  4. Export it and rerun:
       export HF_TOKEN=hf_xxxxxxxxxxxxxxxxxxxxxx
       ./scripts/run_cluster_generators.sh
EOF
    exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CODE_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$CODE_DIR"

QUESTIONS="data/runs/$RUN/eval/questions.jsonl"
if [[ ! -f "$QUESTIONS" ]]; then
    echo ">>> questions.jsonl not found at $QUESTIONS"
    echo ">>> generating it from questions_full_v2.json"

    if ! command -v python >/dev/null 2>&1; then
        echo "ERROR: 'python' not on PATH. Load Python first, e.g." >&2
        echo "  module load Python/3.11" >&2
        exit 2
    fi

    python scripts/prepare_cluster_questions.py \
        --questions questions_full_v2.json \
        --run "$RUN"
fi

if [[ ! -f slurm/compare_models.sh ]]; then
    echo "ERROR: slurm/compare_models.sh not found (expected to be run from code/)" >&2
    exit 3
fi

echo
echo "===================================================================="
echo "Submitting cluster generator evaluation"
echo "===================================================================="
echo "Run name:          $RUN"
echo "Models:            $MODELS"
echo "HF token:          set ($((${#HF_TOKEN}/8))xN chars)"
echo "Retrieval config:  k=${TOP_K:-4}, hybrid + cross-encoder rerank"
echo "Eval questions:    $QUESTIONS ($(wc -l < "$QUESTIONS") rows)"
echo "===================================================================="
echo

cd slurm
mkdir -p logs

env_vars=(
    "RUN_NAME=$RUN"
    "MODELS_OVERRIDE=$MODELS"
    "HF_TOKEN=$HF_TOKEN"
)
[[ -n "$LIMIT_ARG" ]] && env_vars+=("$LIMIT_ARG")
[[ -n "$TOP_K_ARG" ]] && env_vars+=("$TOP_K_ARG")

JOBID="$(env "${env_vars[@]}" sbatch --parsable compare_models.sh)"

echo "Submitted job: $JOBID"
echo
echo "Watch progress:"
echo "  squeue -u \$USER"
echo "  tail -f $(pwd)/logs/compare_models_${JOBID}.out"
echo
echo "When the job finishes, the result files will be at:"
for m in $(echo "$MODELS" | tr ',' ' '); do
    safe="$(echo "$m" | tr '/:' '__')"
    echo "  data/runs/$RUN/eval/results_${safe}.jsonl"
done
echo
echo "Send those .jsonl files back to Luka (or rsync to your laptop),"
echo "then the LLM-as-judge step is:"
echo "  python scripts/compare_generators.py \\"
echo "      --run $RUN \\"
echo "      --reference-file questions_full_v2.json \\"
echo "      --judge-model claude-haiku-4-5"
