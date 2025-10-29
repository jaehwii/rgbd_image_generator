#!/usr/bin/env bash
# Headless batch rendering (no UI). Use absolute paths for robustness.
set -euo pipefail

BLENDER_BIN=${BLENDER_BIN:-blender}
PROJECT_ROOT=$(cd "$(dirname "$0")/.." && pwd)
CFG_PATH="${1:-${PROJECT_ROOT}/config/scene_example.toml}"

"${BLENDER_BIN}" --background \
  --python "${PROJECT_ROOT}/src/blender_rgbd_render_seq.py" -- \
  --config "${CFG_PATH}"
