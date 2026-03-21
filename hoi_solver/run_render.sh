#!/usr/bin/env bash
set -euo pipefail

# Minimal render script: change the variables below, then run `bash run_render.sh`

SESSION_DIR="./demo"
OUT_MP4="output_render.mp4"
FPS=30

# Ground alignment: default is none. Change to miny if you want to force ground contact.
GROUND_ALIGN="none"

# Optional: explicitly specify the parameter file to render (avoids selecting the wrong one when multiple all_parameters_*.json exist)
# e.g.: "./demo/final_optimized_parameters/all_parameters_20260129_042706.json"
PARAMS_JSON=""

# Optional: only render a specific frame range (leave empty to use frame_range from all_parameters_*.json)
START_FRAME=""
END_FRAME_EXCLUSIVE=""

cd "$(dirname "$0")"

cmd="python render.py --data_dir \"$SESSION_DIR\" --out \"$OUT_MP4\" --fps $FPS"

if [[ -n "$GROUND_ALIGN" ]]; then
  cmd+=" --ground_align \"$GROUND_ALIGN\""
fi

if [[ -n "$PARAMS_JSON" ]]; then
  cmd+=" --params_json \"$PARAMS_JSON\""
fi

if [[ -n "$START_FRAME" ]]; then
  cmd+=" --start_frame \"$START_FRAME\""
fi

if [[ -n "$END_FRAME_EXCLUSIVE" ]]; then
  cmd+=" --end_frame_exclusive \"$END_FRAME_EXCLUSIVE\""
fi

echo "$cmd"
eval "$cmd"
