#!/usr/bin/env bash

set -euo pipefail

# Score every checkpoint of the b200 run against the held-out episodes —
# the same 9 episodes RLDX-1's checkpoints were scored on.
#
# Per-episode MSE lines are what matters: compare checkpoints by pairwise
# per-episode wins, not by the mean (run-to-run sampling noise is ~±15%).

REPO_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

export PATH="/NHNHOME/doyoung/bin:$PATH"
export HF_HOME="${HF_HOME:-/NHNHOME/doyoung/.cache/huggingface}"
export UV_PYTHON_INSTALL_DIR="/NHNHOME/doyoung/.uv/python"
export UV_CACHE_DIR="/NHNHOME/doyoung/.cache/uv"

RUN_DIR="${RUN_DIR:-$REPO_DIR/ckpt/gr00t_n17_ffw_sh5_0826_b200/gr00t_n17_ffw_sh5_0826_b200}"
DATASET_PATH="${DATASET_PATH:-$REPO_DIR/dataset/rlwrld_demo_0826_holdout}"
OUT_DIR="${OUT_DIR:-$REPO_DIR/outputs/eval/holdout_b200}"

mkdir -p "$OUT_DIR"
cd "$REPO_DIR"

for ckpt in "$RUN_DIR"/checkpoint-*; do
    step="${ckpt##*checkpoint-}"
    log="$OUT_DIR/ckpt-$step.log"
    if grep -q "MSE across single traj" "$log" 2>/dev/null; then
        echo "[i] checkpoint-$step already scored, skipping"
        continue
    fi
    echo "[i] scoring checkpoint-$step"
    uv run python open_loop_eval_verbose.py \
        --model_path "$ckpt" \
        --dataset_path "$DATASET_PATH" \
        --embodiment_tag new_embodiment \
        --action_horizon 40 \
        --traj_ids 0 1 2 3 4 5 6 7 8 \
        --steps 1000 \
        --save_plot_path "$OUT_DIR/plots-$step/traj.jpeg" \
        > "$log" 2>&1 || echo "[!] checkpoint-$step FAILED (see $log)"
done

echo "=== per-episode unnormalized MSE ==="
for log in "$OUT_DIR"/ckpt-*.log; do
    step="${log##*ckpt-}"; step="${step%.log}"
    mses=$(grep -a "Unnormalized Action MSE across single traj" "$log" | grep -aoE "[0-9.]+$" | tr '\n' ' ')
    echo "checkpoint-$step: $mses"
done
echo ALL_EVAL_DONE
