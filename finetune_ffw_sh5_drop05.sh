#!/usr/bin/env bash

set -euo pipefail

REPO_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

BASE_MODEL_PATH="${BASE_MODEL_PATH:-nvidia/GR00T-N1.7-3B}"
DATASET_PATH="${DATASET_PATH:-/workspace/dataset/merge_ffw_sh5_rev1_260627_psc_left}"
MODALITY_CONFIG_PATH="${MODALITY_CONFIG_PATH:-$REPO_DIR/examples/FFW_SH5/ffw_sh5_config.py}"
OUTPUT_DIR="${OUTPUT_DIR:-$REPO_DIR/ffw_sh5_n17-checkpoint1_drop05}"
EXPERIMENT_NAME="${EXPERIMENT_NAME:-ffw_sh5_n17_psc_left_drop05}"
WANDB_PROJECT="${WANDB_PROJECT:-finetune-gr00t-n1d7-ffw-sh5-drop05}"

# Same as finetune_ffw_sh5.sh, but state_dropout_prob=0.5 on GPU 3.
# Resume from checkpoint-100000 when extending to 150k (same OUTPUT_DIR).
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-3}"
export NUM_GPUS="${NUM_GPUS:-1}"
export GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-64}"
export MAX_STEPS="${MAX_STEPS:-150000}"
export SAVE_STEPS="${SAVE_STEPS:-50000}"
export DATALOADER_NUM_WORKERS="${DATALOADER_NUM_WORKERS:-4}"
export USE_WANDB="${USE_WANDB:-1}"

LFS_WHEEL="$REPO_DIR/scripts/deployment/dgpu/wheels/flash_attn-2.7.4.post1-cp310-cp310-linux_aarch64.whl"
if [[ -f "$LFS_WHEEL" ]]; then
    IFS= read -r lfs_header < "$LFS_WHEEL"
    if [[ "$lfs_header" == "version https://git-lfs.github.com/spec/v1" ]]; then
        echo "Git LFS files have not been downloaded in: $REPO_DIR" >&2
        echo "Install git-lfs, then run: git lfs install && git lfs pull" >&2
        exit 1
    fi
fi

if [[ ! -f "$DATASET_PATH/meta/info.json" || ! -f "$DATASET_PATH/meta/modality.json" ]]; then
    echo "Dataset metadata is missing: $DATASET_PATH/meta" >&2
    exit 1
fi

if ! compgen -G "$DATASET_PATH/data/chunk-*/episode_*.parquet" >/dev/null; then
    echo "Dataset parquet files are missing under: $DATASET_PATH/data" >&2
    echo "Download the complete dataset before starting fine-tuning." >&2
    exit 1
fi

if ! compgen -G "$DATASET_PATH/videos/chunk-*/*/episode_*.mp4" >/dev/null; then
    echo "Dataset video files are missing under: $DATASET_PATH/videos" >&2
    echo "Download the complete dataset before starting fine-tuning." >&2
    exit 1
fi

if [[ "${PREFLIGHT_ONLY:-0}" == "1" ]]; then
    echo "FFW SH5 N1.7 preflight passed (state_dropout_prob=0.5, GPU 3, max_steps=150000)."
    echo "Dataset: $DATASET_PATH"
    echo "Modality config: $MODALITY_CONFIG_PATH"
    echo "Output: $OUTPUT_DIR"
    exit 0
fi

cd "$REPO_DIR"

exec uv run bash examples/finetune.sh \
    --base-model-path "$BASE_MODEL_PATH" \
    --dataset-path "$DATASET_PATH" \
    --modality-config-path "$MODALITY_CONFIG_PATH" \
    --embodiment-tag NEW_EMBODIMENT \
    --output-dir "$OUTPUT_DIR" \
    --experiment-name "$EXPERIMENT_NAME" \
    --wandb-project "$WANDB_PROJECT" \
    --state-dropout-prob 0.5
