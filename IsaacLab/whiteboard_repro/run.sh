#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/config.sh"
mode="${1:-help}"
if [[ $# -gt 0 ]]; then shift; fi
cd "${ISAACLAB_ROOT}"
sim() {
  # isaaclab.sh selects CONDA_PREFIX/bin/python, regardless of this shell's env.
  [[ -x "${WB_SIM_PYTHON}" ]] || { echo "Missing WB_SIM_PYTHON: ${WB_SIM_PYTHON}" >&2; exit 1; }
  env -u PYTHONPATH CONDA_PREFIX="$(dirname -- "$(dirname -- "${WB_SIM_PYTHON}")")" \
    bash -c 'source "$1/_isaac_sim/setup_conda_env.sh"; cd "$1"; exec bash "$1/isaaclab.sh" -p "${@:2}"' \
    whiteboard-sim "${ISAACLAB_ROOT}" "$@"
}
policy() { env -u PYTHONPATH "${LEROBOT_PYTHON}" "$@"; }
case "${mode}" in
  server) exec bash ./deploy_act.sh "$@" ;;
  sim) sim scripts/my_projects/dagger728.py --policy_state_dim 20 --enable_cameras "$@" ;;
  record) sim scripts/my_projects/dagger728.py --policy_state_dim 20 --enable_cameras \
      --record_trajectory trajectory_results/act_wipe_episode.json "$@" ;;
  collect-env) mkdir -p "${WHITEBOARD_RAW_DATA}"; sim scripts/my_projects/wipe728.py --enable_cameras "$@" ;;
  collect-planner) env -u PYTHONPATH "${WB_SIM_PYTHON}" scripts/my_projects/planner728.py "$@" ;;
  convert) policy whiteboard_repro/convert_frozen.py --input-dir "${WB_ORIGINAL_RAW}" \
      --output-path "${WB_BUNDLE_ROOT}/data/wipe_board_act_reconverted" "$@" ;;
  train) exec bash ./train_act.sh "$@" ;;
  resume)
    resume_model="${MODEL_PATH}"
    [[ -d "${resume_model}/../training_state" ]] || { echo "Missing training_state beside ${resume_model}" >&2; exit 1; }
    export HF_HOME="${ACT_CACHE_ROOT}" HF_LEROBOT_HOME="${ACT_CACHE_ROOT}/lerobot" HF_DATASETS_CACHE="${ACT_CACHE_ROOT}/datasets"
    policy -m lerobot.scripts.lerobot_train --config_path="${resume_model}/train_config.json" \
      --resume=true --dataset.root="${DATASET_ROOT}" \
      --output_dir="${OUTPUT_DIR}" "$@" ;;
  csv) env -u PYTHONPATH "${WB_SIM_PYTHON}" scripts/export_act_wipe_csv.py "$@" ;;
  plot) env -u PYTHONPATH "${WB_SIM_PYTHON}" scripts/plot_act_wipe_trajectory.py "$@" ;;
  check) policy whiteboard_repro/check.py --role policy --model "${MODEL_PATH}" --dataset "${DATASET_ROOT}" "$@"
    env -u PYTHONPATH "${WB_SIM_PYTHON}" whiteboard_repro/check.py --role sim "$@" ;;
  *) echo "Usage: bash whiteboard_repro/run.sh {server|sim|record|collect-env|collect-planner|convert|train|resume|csv|plot|check} [arguments]" ;;
esac
