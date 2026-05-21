#!/usr/bin/env bash
# Pull cluster-side evaluation outputs back to the local Mac.
#
# Required env:
#   CLUSTER_HOST   e.g. user@hpc.fri.uni-lj.si
#   CLUSTER_PATH   absolute path on cluster (same as in sync_to_cluster.sh)
#
# Optional env:
#   RUN_NAME       which derived run to fetch (default: 2026-05-20__full__fri4_ul2__v1__e5_base)
#
# Pulls back:
#   1. <RUN_NAME>/eval/results_*.jsonl  (per-model generation outputs)
#   2. <RUN_NAME>/eval/results_*.txt    (any auxiliary outputs, if produced)
#   3. slurm/logs/compare_models_*.out  (latest job logs for inspection)
#
# Usage:
#   CLUSTER_HOST=user@hpc.fri.uni-lj.si \
#   CLUSTER_PATH=/d/hpc/users/$USER/fri-rag \
#   ./scripts/sync_from_cluster.sh

set -euo pipefail

if [[ -z "${CLUSTER_HOST:-}" || -z "${CLUSTER_PATH:-}" ]]; then
    echo "ERROR: CLUSTER_HOST and CLUSTER_PATH must be set." >&2
    exit 2
fi

RUN_NAME="${RUN_NAME:-2026-05-20__full__fri4_ul2__v1__e5_base}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CODE_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
LOCAL_RUN_DIR="$CODE_DIR/data/runs/$RUN_NAME"
mkdir -p "$LOCAL_RUN_DIR/eval"
mkdir -p "$CODE_DIR/slurm/logs"

echo ">>> Fetching results_*.jsonl"
rsync -avh --progress \
    --include 'results_*.jsonl' --exclude '*' \
    "$CLUSTER_HOST:$CLUSTER_PATH/code/data/runs/$RUN_NAME/eval/" \
    "$LOCAL_RUN_DIR/eval/"

echo
echo ">>> Fetching latest compare_models_*.out (for sanity)"
rsync -avh \
    --include 'compare_models_*.out' --include 'compare_models_*.err' \
    --exclude '*' \
    "$CLUSTER_HOST:$CLUSTER_PATH/code/slurm/logs/" \
    "$CODE_DIR/slurm/logs/"

echo
echo "Local results_*.jsonl files:"
ls -la "$LOCAL_RUN_DIR/eval/" | grep '^-' | grep 'results_' || echo "  (none yet)"
echo
echo "Next:"
echo "  python scripts/compare_generators.py --run $RUN_NAME --reference-file questions_full_v2.json"
