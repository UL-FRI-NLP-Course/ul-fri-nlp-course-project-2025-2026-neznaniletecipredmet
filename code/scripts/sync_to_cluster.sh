#!/usr/bin/env bash
# Sync the winning index + cleaned eval set + updated scripts to a remote cluster.
#
# Required env:
#   CLUSTER_HOST   e.g. user@hpc.fri.uni-lj.si
#   CLUSTER_PATH   absolute path on cluster, e.g. /d/hpc/users/<you>/fri-rag
#
# Optional env:
#   RUN_NAME       which derived run to sync (default: 2026-05-20__full__fri4_ul2__v1__e5_base)
#   ALSO_RAW       1 = also rsync the raw/ folder (~25 MB; only needed if you plan
#                  to rebuild the index on the cluster). Default: 0.
#   ALSO_HF_CACHE  1 = also rsync .hf_cache/ (skip download time on cluster). Default: 0.
#
# Pulls back-to-back:
#   1. Updated python scripts and slurm scripts
#   2. <RUN_NAME>/index/        (FAISS + metadata)
#   3. <RUN_NAME>/eval/questions.jsonl  (cleaned eval set)
#   4. questions_full_v2.json (for the local --reference-file judge step later)
#
# Usage:
#   CLUSTER_HOST=user@hpc.fri.uni-lj.si \
#   CLUSTER_PATH=/d/hpc/users/$USER/fri-rag \
#   ./scripts/sync_to_cluster.sh

set -euo pipefail

if [[ -z "${CLUSTER_HOST:-}" || -z "${CLUSTER_PATH:-}" ]]; then
    echo "ERROR: CLUSTER_HOST and CLUSTER_PATH must be set." >&2
    echo "Example:" >&2
    echo "  CLUSTER_HOST=user@hpc.fri.uni-lj.si CLUSTER_PATH=/d/hpc/users/\$USER/fri-rag $0" >&2
    exit 2
fi

RUN_NAME="${RUN_NAME:-2026-05-20__full__fri4_ul2__v1__e5_base}"
ALSO_RAW="${ALSO_RAW:-0}"
ALSO_HF_CACHE="${ALSO_HF_CACHE:-0}"

# Resolve the local code/ root regardless of where the script is invoked from.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CODE_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
REPO_ROOT="$(cd "$CODE_DIR/.." && pwd)"
LOCAL_RUN_DIR="$CODE_DIR/data/runs/$RUN_NAME"

if [[ ! -d "$LOCAL_RUN_DIR" ]]; then
    echo "ERROR: local run dir not found: $LOCAL_RUN_DIR" >&2
    exit 3
fi

echo "Local code dir:  $CODE_DIR"
echo "Local run dir:   $LOCAL_RUN_DIR"
echo "Cluster target:  $CLUSTER_HOST:$CLUSTER_PATH/code/"
echo

ssh "$CLUSTER_HOST" "mkdir -p '$CLUSTER_PATH/code/data/runs/$RUN_NAME/index' '$CLUSTER_PATH/code/data/runs/$RUN_NAME/eval' '$CLUSTER_PATH/code/scripts' '$CLUSTER_PATH/code/slurm' '$CLUSTER_PATH/code/src'"

# 1. Python source code (excluding heavy stuff). Use --update to avoid clobbering
#    cluster-side iteration.
echo ">>> Syncing scripts/, src/, slurm/, top-level files"
rsync -avh --update \
    --exclude '__pycache__' --exclude '*.pyc' \
    "$CODE_DIR/scripts/" "$CLUSTER_HOST:$CLUSTER_PATH/code/scripts/"
rsync -avh --update \
    --exclude '__pycache__' --exclude '*.pyc' \
    "$CODE_DIR/src/" "$CLUSTER_HOST:$CLUSTER_PATH/code/src/"
rsync -avh --update \
    "$CODE_DIR/slurm/" "$CLUSTER_HOST:$CLUSTER_PATH/code/slurm/" \
    --exclude 'logs/' --exclude '*.out' --exclude '*.err'
for f in config.py requirements.txt QUICKSTART.md questions_full_v2.json questions_full.json questions_audited.json; do
    if [[ -f "$CODE_DIR/$f" ]]; then
        rsync -avh --update "$CODE_DIR/$f" "$CLUSTER_HOST:$CLUSTER_PATH/code/$f"
    fi
done

# 2. Index (FAISS + metadata + embedding_info.json). This is the only really
#    chunky thing - typically ~180 MB total. Required for evaluate.py to load.
echo
echo ">>> Syncing index/ (FAISS + metadata)"
rsync -avh --progress \
    "$LOCAL_RUN_DIR/index/" \
    "$CLUSTER_HOST:$CLUSTER_PATH/code/data/runs/$RUN_NAME/index/"

# 3. Cleaned eval set (questions.jsonl).
echo
echo ">>> Syncing eval/questions.jsonl"
rsync -avh --update \
    "$LOCAL_RUN_DIR/eval/questions.jsonl" \
    "$CLUSTER_HOST:$CLUSTER_PATH/code/data/runs/$RUN_NAME/eval/questions.jsonl"

# 4. Optional: raw data (only if you plan to rebuild on cluster).
if [[ "$ALSO_RAW" == "1" ]]; then
    echo
    echo ">>> Syncing raw/ (~25 MB compressed)"
    rsync -avh --progress \
        "$LOCAL_RUN_DIR/raw/" \
        "$CLUSTER_HOST:$CLUSTER_PATH/code/data/runs/$RUN_NAME/raw/"
fi

# 5. Optional: prebuilt HF cache to skip model downloads on cluster (~5-30 GB).
if [[ "$ALSO_HF_CACHE" == "1" && -d "$CODE_DIR/.hf_cache" ]]; then
    echo
    echo ">>> Syncing .hf_cache/ (large; only if you've cached models locally)"
    rsync -avh --progress \
        "$CODE_DIR/.hf_cache/" \
        "$CLUSTER_HOST:$CLUSTER_PATH/code/.hf_cache/"
fi

echo
echo "Sync done."
echo
echo "Next steps on the cluster:"
echo "  ssh $CLUSTER_HOST"
echo "  cd $CLUSTER_PATH/code/slurm"
echo "  RUN_NAME=$RUN_NAME sbatch compare_models.sh"
echo
echo "Then on the local Mac, after sbatch finishes:"
echo "  ./scripts/sync_from_cluster.sh   # fetches results_*.jsonl"
echo "  python scripts/compare_generators.py --run $RUN_NAME --reference-file questions_full_v2.json"
