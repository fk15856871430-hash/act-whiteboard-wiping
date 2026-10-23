#!/usr/bin/env bash
# Optional overrides live in local.env (copy local.env.example first).
WB_KIT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
export ISAACLAB_ROOT="$(cd -- "${WB_KIT_DIR}/.." && pwd)"
WB_BUNDLE_ROOT="$(cd -- "${ISAACLAB_ROOT}/.." && pwd)"
if [[ -f "${WB_KIT_DIR}/local.env" ]]; then source "${WB_KIT_DIR}/local.env"; fi
if [[ -n "${WHITEBOARD_ROOM_USD:-}" ]]; then export WHITEBOARD_ROOM_USD; fi
if [[ -d "${WB_BUNDLE_ROOT}/data/wipe_board_act_lerobot_v3" ]]; then
  WB_DATA_HOME="${WB_BUNDLE_ROOT}/data"
  WB_MODEL_HOME="${WB_BUNDLE_ROOT}/models"
else
  WB_DATA_HOME="/mnt/bigdrive/datasets"
  WB_MODEL_HOME="/mnt/bigdrive/output_models"
fi
export WB_CONDA_ROOT="${WB_CONDA_ROOT:-${HOME}/miniconda3}"
export LEROBOT_PYTHON="${LEROBOT_PYTHON:-${WB_CONDA_ROOT}/envs/lerobot/bin/python}"
export WB_SIM_PYTHON="${WB_SIM_PYTHON:-${WB_CONDA_ROOT}/envs/env_isaaclab/bin/python}"
export DATASET_ROOT="${DATASET_ROOT:-${WB_DATA_HOME}/wipe_board_act_lerobot_v3}"
export MODEL_PATH="${MODEL_PATH:-${WB_MODEL_HOME}/wipe_board_act/checkpoints/last/pretrained_model}"
# New training and collection never use the exported originals as output.
export OUTPUT_DIR="${OUTPUT_DIR:-${WB_BUNDLE_ROOT}/runs/wipe_board_act_new}"
export WHITEBOARD_RAW_DATA="${WHITEBOARD_RAW_DATA:-${WB_BUNDLE_ROOT}/runs/new_demonstrations}"
export WB_ORIGINAL_RAW="${WB_ORIGINAL_RAW:-${WB_BUNDLE_ROOT}/data/raw_hdf5}"
[[ -d "${WB_ORIGINAL_RAW}" ]] || WB_ORIGINAL_RAW="${ISAACLAB_ROOT}/vla_dataset"
export ACT_CACHE_ROOT="${ACT_CACHE_ROOT:-${WB_BUNDLE_ROOT}/cache}"
export TORCH_HOME="${TORCH_HOME:-${ACT_CACHE_ROOT}/torch}"
