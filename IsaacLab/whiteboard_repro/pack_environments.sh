#!/usr/bin/env bash
# Run on the ORIGINAL computer. Produces portable Linux x86_64 Conda archives.
set -euo pipefail
bundle="${1:?Usage: bash pack_environments.sh /absolute/path/to/bundle}"
[[ -f "${bundle}/build_options.json" ]] || { echo "Not a bundle: ${bundle}" >&2; exit 1; }
WB_PACK_CONDA_ROOT="${WB_PACK_CONDA_ROOT:-${HOME}/miniconda3}"
WB_PACK_TOOLS="$(mktemp -d -t whiteboard-pack-tools-XXXXXXXX)"
env -u PYTHONPATH "${WB_PACK_CONDA_ROOT}/bin/python" -m pip install --target "${WB_PACK_TOOLS}" 'conda-pack==0.8.1' 'setuptools<81'
mkdir -p "${bundle}/environments"
for name in env_isaaclab lerobot; do
  env PYTHONPATH="${WB_PACK_TOOLS}" TMPDIR="${bundle}/environments" \
    "${WB_PACK_CONDA_ROOT}/bin/python" "$(dirname -- "${BASH_SOURCE[0]}")/pack_runtime.py" \
    --prefix "${WB_PACK_CONDA_ROOT}/envs/${name}" --output "${bundle}/environments/${name}.tar.gz"
done
echo "Environment archives created. Refresh SHA256SUMS with build_bundle.py --rehash."
echo "Temporary conda-pack tool directory: ${WB_PACK_TOOLS}"
