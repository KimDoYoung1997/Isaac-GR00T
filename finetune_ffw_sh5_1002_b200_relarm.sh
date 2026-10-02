#!/usr/bin/env bash

set -euo pipefail

# Fine-tune GR00T-N1.7-3B on the 1002 merge (rlwrld_demo_merged_1002) with the
# relarm representation adopted on 0826: arms RELATIVE, fingers ABSOLUTE. That
# closed the +24% gap to RLDX-1 on 0826 (N1.7 was pretrained on delta-style
# actions). Mirrors RLDX-1's rldx1_ft_ffw_sh5_1002 run on the same split.
#
#   data      dataset/rlwrld_demo_merged_1002_train   166 ep / 85,494 frames
#             (held-out: dataset/rlwrld_demo_merged_1002_holdout, merged ep
#             61/66/71/76 = one per box_pose_case 1-4)
#   action    26-dim, arms RELATIVE + fingers ABSOLUTE, chunk 40 = the head's
#             native horizon. The data is 20 fps, so 40 steps = 2.0 s.
#   steps     50k default, save every 10k — on 0826 relarm kept improving to
#             50k instead of overfitting after 10k. Judge by held-out scoring.
#             Re-running with the same OUTPUT_DIR auto-resumes.
#
# B200 note: this box needs sm_100 CUBINs. torch 2.7.1+cu128 ships them, but
# the pinned flash-attn wheel stops at sm_90 — it must NOT be installed
# (qwen3_backbone falls back to sdpa on ImportError). setup leaves it out.

REPO_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

export PATH="/NHNHOME/doyoung/bin:$PATH"
export HF_HOME="${HF_HOME:-/NHNHOME/doyoung/.cache/huggingface}"
export UV_PYTHON_INSTALL_DIR="/NHNHOME/doyoung/.uv/python"
export UV_CACHE_DIR="/NHNHOME/doyoung/.cache/uv"

BASE_MODEL_PATH="${BASE_MODEL_PATH:-nvidia/GR00T-N1.7-3B}"
DATASET_PATH="${DATASET_PATH:-$REPO_DIR/dataset/rlwrld_demo_merged_1002_train}"
MODALITY_CONFIG_PATH="${MODALITY_CONFIG_PATH:-$REPO_DIR/examples/FFW_SH5/ffw_sh5_0826_used26_h40_relarm_config.py}"

export NUM_GPUS="${NUM_GPUS:-4}"
export GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-256}"
export MAX_STEPS="${MAX_STEPS:-50000}"
export SAVE_STEPS="${SAVE_STEPS:-10000}"
export DATALOADER_NUM_WORKERS="${DATALOADER_NUM_WORKERS:-8}"
export USE_WANDB="${USE_WANDB:-0}"
export MASTER_PORT="${MASTER_PORT:-$(shuf -i 20000-30000 -n 1)}"

SAVE_TOTAL_LIMIT="${SAVE_TOTAL_LIMIT:-10}"
STATE_DROPOUT_PROB="${STATE_DROPOUT_PROB:-0.2}"

CKPT_NAME="${CKPT_NAME:-gr00t_n17_ffw_sh5_1002_b200_relarm}"

# SMOKE=1 proves the pipeline assembles (stats generation, loader keys,
# 26-dim action, VRAM) without touching the real run's output directory —
# auto-resume would otherwise inherit the smoke step counter.
if [[ "${SMOKE:-0}" == "1" ]]; then
    export MAX_STEPS=20 SAVE_STEPS=20 GLOBAL_BATCH_SIZE=16
    CKPT_NAME="${CKPT_NAME}_smoke"
fi

EXPERIMENT_NAME="${EXPERIMENT_NAME:-$CKPT_NAME}"
OUTPUT_DIR="${OUTPUT_DIR:-/NHNHOME/doyoung/Isaac-GR00T/ckpt/$CKPT_NAME}"

if [[ ! -f "$MODALITY_CONFIG_PATH" ]]; then
    echo "Modality config is missing: $MODALITY_CONFIG_PATH" >&2
    exit 1
fi
if [[ ! -f "$DATASET_PATH/meta/info.json" || ! -f "$DATASET_PATH/meta/modality.json" ]]; then
    echo "Dataset metadata is missing: $DATASET_PATH/meta" >&2
    exit 1
fi

# Resuming is implicit (experiment.py passes resume_from_checkpoint=True) but
# should never be a surprise.
if compgen -G "$OUTPUT_DIR/checkpoint-*" >/dev/null; then
    _last=$(ls -d "$OUTPUT_DIR"/checkpoint-* | sed 's/.*checkpoint-//' | sort -n | tail -1)
    echo "[!] $OUTPUT_DIR already holds checkpoints (latest: $_last)."
    echo "[!] This run will RESUME from it, not start fresh."
    echo "[!] For a fresh run: CKPT_NAME=<something-new> bash $0"
fi

echo "[i] dataset    : $DATASET_PATH"
echo "[i] modality   : $MODALITY_CONFIG_PATH  (chunk 40, 26-dim action)"
echo "[i] output     : $OUTPUT_DIR"
echo "[i] batch/steps: $GLOBAL_BATCH_SIZE x $MAX_STEPS (save every $SAVE_STEPS, keep $SAVE_TOTAL_LIMIT)"

cd "$REPO_DIR"

# save_total_limit rides after `--`: the wrapper hardcodes 5 and tyro lets
# the last occurrence win.
exec uv run bash examples/finetune.sh \
    --base-model-path "$BASE_MODEL_PATH" \
    --dataset-path "$DATASET_PATH" \
    --modality-config-path "$MODALITY_CONFIG_PATH" \
    --embodiment-tag NEW_EMBODIMENT \
    --output-dir "$OUTPUT_DIR" \
    --experiment-name "$EXPERIMENT_NAME" \
    --state-dropout-prob "$STATE_DROPOUT_PROB" \
    -- --save_total_limit "$SAVE_TOTAL_LIMIT"
