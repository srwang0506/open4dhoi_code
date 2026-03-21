#!/usr/bin/env bash
# ==============================================================================
# Step 4: Assemble HOI scene (compute object scale/pose)
# Conda environment: 4dhoi_pipeline
#
# Usage:
#   bash step4_hoi.sh <video_dir> [--render]
# ==============================================================================
set -eo pipefail

PIPELINE_DIR="$(cd "$(dirname "$0")" && pwd)"
source "${PIPELINE_DIR}/config.sh"

VIDEO_DIR="${1:?Usage: $0 <video_dir> [--render]}"
VIDEO_DIR="$(cd "$VIDEO_DIR" && pwd)"
shift

echo "============================================"
echo "Step 4: HOI Assembly"
echo "Session: ${VIDEO_DIR}"
echo "Conda env: ${ENV_PIPELINE}"
echo "============================================"

eval "$(conda shell.bash hook)"
set +u
conda activate "${ENV_PIPELINE}"
set -u

echo ""
echo "[1/1] Assembling HOI scene..."
if [ -f "${VIDEO_DIR}/output/obj_poses.json" ]; then
    echo "  obj_poses.json already exists, skipping."
else
    python "${SCRIPT_DIR}/make_hoi.py" \
        --video_dir "${VIDEO_DIR}" \
        "$@"
    echo "  HOI scene assembled."
fi

echo ""
echo "Step 4 complete."
