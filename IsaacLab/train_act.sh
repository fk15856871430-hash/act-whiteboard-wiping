#!/usr/bin/env bash
set -euo pipefail

ISAACLAB_ROOT="${ISAACLAB_ROOT:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)}"
LEROBOT_PYTHON="${LEROBOT_PYTHON:-${HOME}/miniconda3/envs/lerobot/bin/python}"
DATASET_ROOT="${DATASET_ROOT:-/mnt/bigdrive/datasets/wipe_board_act_lerobot_v3}"
OUTPUT_DIR="${OUTPUT_DIR:-/mnt/bigdrive/output_models/wipe_board_act}"
LAST_CONFIG="${OUTPUT_DIR}/checkpoints/last/pretrained_model/train_config.json"

if [[ ! -f "${DATASET_ROOT}/meta/info.json" ]]; then
  echo "ACT dataset not found: ${DATASET_ROOT}" >&2
  exit 1
fi

ACT_CACHE_ROOT="${ACT_CACHE_ROOT:-/mnt/bigdrive/hf_cache}"
export HF_HOME="${ACT_CACHE_ROOT}"
export HF_LEROBOT_HOME="${ACT_CACHE_ROOT}/lerobot"
export HF_DATASETS_CACHE="${ACT_CACHE_ROOT}/datasets"
export TORCH_HOME="${TORCH_HOME:-/mnt/bigdrive/torch_cache}"
export PYTHONPATH="${ISAACLAB_ROOT}"

cd "${ISAACLAB_ROOT}"
if [[ -f "${LAST_CONFIG}" ]]; then
  echo "Resuming ACT training from ${LAST_CONFIG}"
  exec "${LEROBOT_PYTHON}" -m lerobot.scripts.lerobot_train \
    --config_path="${LAST_CONFIG}" \
    --resume=true \
    --dataset.root="${DATASET_ROOT}" \
    --output_dir="${OUTPUT_DIR}" \
    "$@"
fi

exec "${LEROBOT_PYTHON}" -m lerobot.scripts.lerobot_train \
  --dataset.repo_id=wipe_board_act_lerobot_v3 \
  --dataset.root="${DATASET_ROOT}" \
  --policy.type=act \
  --policy.device=cuda \
  --policy.chunk_size=50 \
  --policy.n_action_steps=50 \
  --batch_size=8 \
  --steps=100000 \
  --save_freq=10000 \
  --output_dir="${OUTPUT_DIR}" \
  --job_name=wipe_board_act \
  --policy.push_to_hub=false \
  --wandb.enable=false \
  "$@"
