#!/usr/bin/env bash

set -euo pipefail

# Score every checkpoint of a 1002 run on the held-out split — merged episodes
# 61/66/71/76 (one per box_pose_case 1-4), renumbered 0..3 here.
#
# Full trajectories (--steps 1000 is capped by episode length), horizon 40,
# one plot per trajectory. Compare checkpoints episode by episode, not by the
# mean: the same checkpoint moves about ±15% between runs. With only 4
# episodes, 4/4 wins still happens by chance 1 time in 16.
#
#   RUN_DIR=... OUT_DIR=... TRAJ_IDS="0 1 2 3" bash eval_holdout_1002.sh

REPO_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

export PATH="/NHNHOME/doyoung/bin:$PATH"
export HF_HOME="${HF_HOME:-/NHNHOME/doyoung/.cache/huggingface}"
export UV_PYTHON_INSTALL_DIR="/NHNHOME/doyoung/.uv/python"
export UV_CACHE_DIR="/NHNHOME/doyoung/.cache/uv"

RUN_DIR="${RUN_DIR:-$REPO_DIR/ckpt/gr00t_n17_ffw_sh5_1002_b200_relarm/gr00t_n17_ffw_sh5_1002_b200_relarm}"
DATASET_PATH="${DATASET_PATH:-$REPO_DIR/dataset/rlwrld_demo_merged_1002_holdout}"
OUT_DIR="${OUT_DIR:-$REPO_DIR/outputs/eval/holdout_1002_relarm}"
TRAJ_IDS="${TRAJ_IDS:-0 1 2 3}"

mkdir -p "$OUT_DIR"
cd "$REPO_DIR"

for ckpt in "$RUN_DIR"/checkpoint-*; do
    step="${ckpt##*checkpoint-}"
    log="$OUT_DIR/ckpt-$step/eval.log"
    mkdir -p "$OUT_DIR/ckpt-$step"
    if grep -q "MSE across single traj" "$log" 2>/dev/null; then
        echo "[i] checkpoint-$step already scored, skipping"
        continue
    fi
    echo "[i] scoring checkpoint-$step"
    # shellcheck disable=SC2086
    uv run python open_loop_eval_verbose.py \
        --model_path "$ckpt" \
        --dataset_path "$DATASET_PATH" \
        --embodiment_tag new_embodiment \
        --action_horizon 40 \
        --traj_ids $TRAJ_IDS \
        --steps 1000 \
        --save_plot_path "$OUT_DIR/ckpt-$step/plot.jpeg" \
        > "$log" 2>&1 || echo "[!] checkpoint-$step FAILED (see $log)"
done

echo "=== per-episode unnormalized MSE (traj $TRAJ_IDS) ==="
for log in "$OUT_DIR"/ckpt-*/eval.log; do
    step="$(basename "$(dirname "$log")")"
    mses=$(grep -a "Unnormalized Action MSE across single traj" "$log" | grep -aoE "[0-9.]+$" | tr '\n' ' ')
    echo "$step: $mses"
done
echo ALL_EVAL_DONE
