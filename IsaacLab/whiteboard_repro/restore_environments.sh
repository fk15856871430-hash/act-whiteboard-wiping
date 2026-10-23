#!/usr/bin/env bash
# Run on the NEW computer only, after unpacking / copying the bundle.
set -euo pipefail
kit="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo="$(cd -- "${kit}/.." && pwd)"
bundle="$(cd -- "${repo}/.." && pwd)"
env_root="${1:?Usage: bash restore_environments.sh /absolute/new/environment/root}"
[[ "${env_root}" = /* ]] || { echo "Use an absolute target path" >&2; exit 1; }
for name in env_isaaclab lerobot; do
  [[ -f "${bundle}/environments/${name}.tar.gz" ]] || { echo "Missing archive: ${name}" >&2; exit 1; }
  [[ ! -e "${env_root}/${name}" ]] || { echo "Target exists; refusing to overwrite: ${env_root}/${name}" >&2; exit 1; }
done
for name in env_isaaclab lerobot; do
  mkdir -p "${env_root}/${name}"
  tar -xzf "${bundle}/environments/${name}.tar.gz" -C "${env_root}/${name}"
  # No --dest-prefix was used while packing: conda-unpack relocates the environment here.
  env -u PYTHONPATH "${env_root}/${name}/bin/python" "${env_root}/${name}/bin/conda-unpack"
done
# Packed editable .pth files still refer to the original PC. Reinstall from bundled sources.
env -u PYTHONPATH "${env_root}/lerobot/bin/python" -m pip install --no-deps --no-build-isolation -e "${bundle}/lerobot"
for module in isaaclab isaaclab_assets isaaclab_tasks isaaclab_rl isaaclab_mimic isaaclab_contrib; do
  env -u PYTHONPATH "${env_root}/env_isaaclab/bin/python" -m pip install --no-deps --no-build-isolation -e "${repo}/source/${module}"
done
env -u PYTHONPATH "${env_root}/env_isaaclab/bin/python" "${kit}/relocate_hook.py" \
  "${env_root}/env_isaaclab" "${repo}"
echo "Set WB_SIM_PYTHON=${env_root}/env_isaaclab/bin/python and LEROBOT_PYTHON=${env_root}/lerobot/bin/python in ${kit}/local.env"
