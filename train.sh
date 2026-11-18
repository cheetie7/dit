#!/usr/bin/env bash
set -euo pipefail

repo="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$repo"

if [[ "${1:-}" == "--inside" ]]; then
  config="${2:-configs/server_train.json}"
  unset HTTP_PROXY HTTPS_PROXY ALL_PROXY http_proxy https_proxy all_proxy
  export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl
  export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
  export PYTHONPATH="$repo/src${PYTHONPATH:+:$PYTHONPATH}"
  python_path="$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["runtime"]["python"])' "$config")"
  exec "$python_path" tools/server_train.py --config "$config"
fi

config="${1:-configs/server_train.json}"
if [[ "$config" = /* || "$config" == *".."* ]]; then
  echo "Use a config path relative to the DiT repository, e.g. configs/server_train.json" >&2
  exit 1
fi
image="$(python3 -c 'import json,os,sys; print(os.path.expanduser(json.load(open(sys.argv[1]))["runtime"]["image"]))' "$config")"
container_repo="$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["runtime"]["repo"])' "$config")"
if [[ ! -f "$image" ]]; then
  echo "Saved container image not found: $image" >&2
  exit 1
fi
exec srun --container-image="$image" --container-remap-root --container-mount-home \
  bash "$container_repo/train.sh" --inside "$config"
