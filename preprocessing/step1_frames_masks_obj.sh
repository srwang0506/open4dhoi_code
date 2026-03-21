#!/usr/bin/env bash
# ==============================================================================
# Step 1: Extract frames, generate masks, reconstruct object mesh
# Conda environment: sam3d_obj_4d
#
# Usage:
#   bash step1_frames_masks_obj.sh <video_dir>
#
# Example:
#   bash step1_frames_masks_obj.sh /path/to/session_folder
# ==============================================================================
set -eo pipefail

PIPELINE_DIR="$(cd "$(dirname "$0")" && pwd)"
source "${PIPELINE_DIR}/config.sh"

VIDEO_DIR="${1:?Usage: $0 <video_dir>}"
VIDEO_DIR="$(cd "$VIDEO_DIR" && pwd)"

echo "============================================"
echo "Step 1: Frames + Masks + Object Mesh"
echo "Session: ${VIDEO_DIR}"
echo "Conda env: ${ENV_SAM3D_OBJ}"
echo "============================================"

# Activate conda environment
eval "$(conda shell.bash hook)"
set +u
conda activate "${ENV_SAM3D_OBJ}"
set -u

# --- 1.1 Extract frames ---
echo ""
echo "[1/3] Extracting frames..."
if [ -d "${VIDEO_DIR}/frames" ] && [ "$(ls -A "${VIDEO_DIR}/frames" 2>/dev/null)" ]; then
    echo "  Frames already exist, skipping."
else
    python "${SCRIPT_DIR}/make_extract_frames.py" \
        --video_dir "${VIDEO_DIR}" \
        --skip_existing
    echo "  Frames extracted."
fi

# --- 1.2 Generate masks ---
echo ""
echo "[2/3] Generating masks (SAM2)..."
if [ -d "${VIDEO_DIR}/mask_dir" ] && [ -d "${VIDEO_DIR}/human_mask_dir" ]; then
    echo "  Masks already exist, skipping."
else
    CUDA_VISIBLE_DEVICES="${CUDA_MASKS}" \
    python "${SCRIPT_DIR}/make_masks.py" \
        --video_dir "${VIDEO_DIR}" \
        --skip_existing
    echo "  Masks generated."
fi

# --- 1.3 Reconstruct object mesh ---
echo ""
echo "[3/3] Reconstructing object mesh (SAM-3D-Objects)..."
if [ -f "${VIDEO_DIR}/obj_org.obj" ]; then
    echo "  Object mesh already exists, skipping."
else
    CUDA_VISIBLE_DEVICES="${CUDA_OBJ_ORG}" \
    python "${SCRIPT_DIR}/make_obj_org.py" \
        --video_dir "${VIDEO_DIR}"
    echo "  Object mesh reconstructed."
fi

echo ""
echo "Step 1 complete."
