#!/usr/bin/env bash
# Local smoke test for the cluster-bound generator-eval pipeline.
#
# Runs the smallest model (Qwen2.5-1.5B-Instruct, ~3 GB in fp16) on 3 questions
# of the cleaned eval set, against the winning RAG retrieval configuration, then
# (optionally) judges the output with Claude Haiku to verify the end-to-end
# wiring (evaluate.py -> results_*.jsonl -> compare_generators.py).
#
# This proves all the seams before you sbatch the real 4-model job on the
# cluster. It does NOT validate cluster-only models (GaMS, Llama, Mistral) nor
# the slurm script itself - just that the local pipeline produces files in the
# right shape.
#
# Cost: ~$0.03 if --skip-judge is not passed. Time: 5-15 min (mostly Qwen
# download on first run + model load + 3 generations on MPS/CPU).
#
# Usage:
#   ./scripts/local_smoke_test.sh                # full smoke (eval + judge)
#   ./scripts/local_smoke_test.sh --skip-judge   # eval only (no API cost)
#   MODEL=... LIMIT=5 ./scripts/local_smoke_test.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CODE_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
REPO_ROOT="$(cd "$CODE_DIR/.." && pwd)"

PYTHON_BIN="${PYTHON_BIN:-/opt/anaconda3/envs/fri-rag/bin/python}"
RUN_NAME="${RUN_NAME:-2026-05-20__full__fri4_ul2__v1__e5_base}"
MODEL="${MODEL:-Qwen/Qwen2.5-1.5B-Instruct}"
LIMIT="${LIMIT:-3}"
TOP_K="${TOP_K:-4}"
RERANK_CANDIDATE_K="${RERANK_CANDIDATE_K:-20}"
JUDGE_MODEL="${JUDGE_MODEL:-claude-haiku-4-5}"

SKIP_JUDGE=0
for arg in "$@"; do
    case "$arg" in
        --skip-judge) SKIP_JUDGE=1 ;;
        -h|--help)
            grep -E '^# ' "$0" | sed 's/^# //'
            exit 0
            ;;
    esac
done

echo "=== Local smoke test ==="
echo "  Run:        $RUN_NAME"
echo "  Model:      $MODEL"
echo "  Limit:      $LIMIT questions"
echo "  Retrieval:  top-k=$TOP_K, hybrid + rerank (candidate-k=$RERANK_CANDIDATE_K)"
echo "  Python:     $PYTHON_BIN"
[[ "$SKIP_JUDGE" -eq 1 ]] && echo "  Judge:      (skipped)" || echo "  Judge:      $JUDGE_MODEL"
echo

if [[ ! -x "$PYTHON_BIN" ]]; then
    echo "ERROR: PYTHON_BIN does not exist or is not executable: $PYTHON_BIN" >&2
    echo "Set PYTHON_BIN to your env's python path (e.g. /opt/anaconda3/envs/fri-rag/bin/python)" >&2
    exit 2
fi

cd "$CODE_DIR"

INDEX_FILE="data/runs/$RUN_NAME/index/index.faiss"
QUESTIONS_FILE="data/runs/$RUN_NAME/eval/questions.jsonl"

if [[ ! -f "$INDEX_FILE" ]]; then
    echo "ERROR: missing index at $INDEX_FILE" >&2
    echo "Build it first or pick a different RUN_NAME." >&2
    exit 3
fi

if [[ ! -f "$QUESTIONS_FILE" ]]; then
    echo "Creating $QUESTIONS_FILE from questions_full_v2.json..."
    "$PYTHON_BIN" scripts/prepare_cluster_questions.py \
        --questions questions_full_v2.json \
        --run "$RUN_NAME"
fi

echo
echo ">>> [1/2] Running evaluate.py on $MODEL (limit=$LIMIT)"
echo "    The first run will download the model into ./.hf_cache (one-time, ~3 GB)."
echo

"$PYTHON_BIN" scripts/evaluate.py \
    --run "$RUN_NAME" \
    --model "$MODEL" \
    --top-k "$TOP_K" \
    --hybrid \
    --rerank \
    --rerank-candidate-k "$RERANK_CANDIDATE_K" \
    --limit "$LIMIT"

MODEL_SLUG="$(echo "$MODEL" | sed 's|/|_|g')"
RESULTS_FILE="data/runs/$RUN_NAME/eval/results_${MODEL_SLUG}.jsonl"

if [[ ! -f "$RESULTS_FILE" ]]; then
    echo "ERROR: evaluate.py finished but $RESULTS_FILE was not written." >&2
    exit 4
fi

ROWS="$(wc -l < "$RESULTS_FILE" | tr -d ' ')"
echo
echo "    -> Wrote $ROWS rows to $RESULTS_FILE"

if [[ "$SKIP_JUDGE" -eq 1 ]]; then
    echo
    echo "Done (eval only). Re-run without --skip-judge to also test the LLM-judge step."
    exit 0
fi

echo
echo ">>> [2/2] Running compare_generators.py to LLM-judge the smoke-test output"
echo

if [[ -z "${ANTHROPIC_API_KEY:-}" ]]; then
    if [[ -f "$REPO_ROOT/.env" ]]; then
        echo "    Sourcing $REPO_ROOT/.env for ANTHROPIC_API_KEY..."
        set -a
        # shellcheck disable=SC1090
        source "$REPO_ROOT/.env"
        set +a
    fi
fi

if [[ -z "${ANTHROPIC_API_KEY:-}" ]]; then
    echo "ERROR: ANTHROPIC_API_KEY is not set (and no .env in repo root)." >&2
    echo "Set it, or pass --skip-judge to skip this step." >&2
    exit 5
fi

OUT_MD="data/runs/$RUN_NAME/eval/model_comparison_smoke.md"

"$PYTHON_BIN" scripts/compare_generators.py \
    --run "$RUN_NAME" \
    --reference-file questions_full_v2.json \
    --judge-model "$JUDGE_MODEL" \
    --include-pattern "results_${MODEL_SLUG}.jsonl" \
    --out-md "$OUT_MD" \
    --limit "$LIMIT"

echo
echo "Done. Smoke-test summary at: $OUT_MD"
echo
echo "If this completed without errors, the cluster-side flow should work too."
echo "Next: rsync to cluster and submit the full job."
echo "  CLUSTER_HOST=... CLUSTER_PATH=... ./scripts/sync_to_cluster.sh"
