#!/usr/bin/env bash
# ==============================================================================
# Step 2: Human motion estimation (GVHMR) + Depth estimation
# Conda environment: 4dhoi_pipeline
#
# Usage:
#   bash step2_motion_depth.sh <video_dir>
# ==============================================================================
set -eo pipefail

PIPELINE_DIR="$(cd "$(dirname "$0")" && pwd)"
source "${PIPELINE_DIR}/config.sh"

VIDEO_DIR="${1:?Usage: $0 <video_dir>}"
VIDEO_DIR="$(cd "$VIDEO_DIR" && pwd)"

echo "============================================"
echo "Step 2: Motion Estimation + Depth"
echo "Session: ${VIDEO_DIR}"
echo "Conda env: ${ENV_PIPELINE}"
echo "============================================"

eval "$(conda shell.bash hook)"
set +u
conda activate "${ENV_PIPELINE}"
set -u

# --- 2.1 Human motion estimation (GVHMR) ---
echo ""
echo "[1/2] Estimating human motion (GVHMR)..."
if [ -f "${VIDEO_DIR}/motion/result.pt" ]; then
    echo "  Motion data already exists, skipping."
else
    CUDA_VISIBLE_DEVICES="${CUDA_MOTION}" \
    python "${SCRIPT_DIR}/make_motion.py" \
        --video_dir "${VIDEO_DIR}"
    echo "  Motion estimated."
fi

# --- 2.2 Depth estimation ---
echo ""
echo "[2/2] Estimating depth (Depth-Anything-V2)..."
if [ -f "${VIDEO_DIR}/depth.npy" ]; then
    echo "  Depth maps already exist, skipping."
else
    CUDA_VISIBLE_DEVICES="${CUDA_DEPTH}" \
    python "${SCRIPT_DIR}/make_depth.py" \
        --video_dir "${VIDEO_DIR}"
    echo "  Depth estimated."
fi

echo ""
echo "Step 2 complete."
