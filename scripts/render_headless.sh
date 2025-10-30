#!/usr/bin/env bash
# Headless batch rendering (no UI). Use absolute paths for robustness.
set -euo pipefail

BLENDER_BIN=${BLENDER_BIN:-blender}

# Project root = parent of this script directory
PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"

# Let Python find `src` as a first-party package without sys.path hacks
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"

# Default config path if not provided as $1
CFG_PATH="${1:-${PROJECT_ROOT}/config/scene_example.toml}"

# Optional: sanity checks
if [[ ! -f "${CFG_PATH}" ]]; then
  echo "Config not found: ${CFG_PATH}" >&2
  exit 1
fi

# Run Blender with your entry script; pass args after '--'
"${BLENDER_BIN}" --background \
  --python-use-system-env \
  --python-expr "import sys; sys.path.insert(0, r'${PROJECT_ROOT}')" \
  --python "${PROJECT_ROOT}/src/blender_rgbd_render_seq.py" -- \
  --config "${CFG_PATH}"
