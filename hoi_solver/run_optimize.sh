#!/usr/bin/env bash
set -euo pipefail

# Minimal one-click run: just change the variables below, then `bash run_optimize.sh`

# Session directory (should contain: kp_record_merged.json / motion/result.pt / obj_init.obj, etc.)
SESSION_DIR="./demo"

# Debug: only optimize the first N frames (leave empty for all frames)
MAX_FRAMES="2"   # e.g. "2"; set to "" to run all frames

# Specify frame range for optimization (leave empty to use merged.start_frame_index..last)
START_FRAME="120"            # e.g. "120"
END_FRAME_EXCLUSIVE="121"    # e.g. "140" (exclusive)

# Whether to save meshes after least-squares (0=off, 1=on)
SAVE_LS_MESHES=1
LS_MESH_DIR="debug_ls_meshes"

cd "$(dirname "$0")"

cmd="python optimize.py --data_dir \"$SESSION_DIR\""

if [[ -n "$MAX_FRAMES" ]]; then
  cmd+=" --max_frames \"$MAX_FRAMES\""
fi

if [[ -n "$START_FRAME" ]]; then
  cmd+=" --start_frame \"$START_FRAME\""
fi

if [[ -n "$END_FRAME_EXCLUSIVE" ]]; then
  cmd+=" --end_frame_exclusive \"$END_FRAME_EXCLUSIVE\""
fi

if [[ "$SAVE_LS_MESHES" == "1" ]]; then
  cmd+=" --save_ls_meshes --ls_mesh_dir \"$LS_MESH_DIR\""
fi

echo "$cmd"
eval "$cmd"
