#!/usr/bin/env bash
set -euo pipefail

ISAACLAB_ROOT="${ISAACLAB_ROOT:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)}"
LEROBOT_PYTHON="${LEROBOT_PYTHON:-${HOME}/miniconda3/envs/lerobot/bin/python}"
DATASET_ROOT="${DATASET_ROOT:-/mnt/bigdrive/datasets/wipe_board_act_lerobot_v3}"
MODEL_PATH="${MODEL_PATH:-/mnt/bigdrive/output_models/wipe_board_act/checkpoints/last/pretrained_model}"

if [[ ! -d "${MODEL_PATH}" ]]; then
  echo "ACT checkpoint not found: ${MODEL_PATH}" >&2
  echo "Run ${ISAACLAB_ROOT}/train_act.sh first." >&2
  exit 1
fi

# The dataset/checkpoint may live on /mnt/bigdrive, but that mount is
# read-only on some machines.  Hugging Face and torch still need a writable
# lock/cache directory while loading metadata.
if [[ -n "${ACT_CACHE_ROOT:-}" ]]; then
  :
elif [[ -w "/mnt/bigdrive" ]]; then
  ACT_CACHE_ROOT="/mnt/bigdrive/hf_cache"
else
  ACT_CACHE_ROOT="/tmp/isaaclab_act_cache"
fi
mkdir -p "${ACT_CACHE_ROOT}/datasets" "${ACT_CACHE_ROOT}/lerobot" "${ACT_CACHE_ROOT}/torch"
export HF_HOME="${ACT_CACHE_ROOT}"
export HF_LEROBOT_HOME="${ACT_CACHE_ROOT}/lerobot"
export HF_DATASETS_CACHE="${ACT_CACHE_ROOT}/datasets"
export TORCH_HOME="${ACT_CACHE_ROOT}/torch"
# This server runs in a separate Conda environment. Inheriting an activated
# Isaac Lab PYTHONPATH can make it import Isaac Sim's bundled torch instead.
export PYTHONPATH="${ISAACLAB_ROOT}"

cd "${ISAACLAB_ROOT}"
exec "${LEROBOT_PYTHON}" -m policy_servers.lerobot_policy_server \
  --model_path "${MODEL_PATH}" \
  --dataset_root "${DATASET_ROOT}" \
  --repo_id wipe_board_act_lerobot_v3 \
  --device cuda \
  --port 5556 \
  "$@"
