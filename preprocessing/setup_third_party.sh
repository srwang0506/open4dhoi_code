#!/usr/bin/env bash
# ==============================================================================
# Setup third-party dependencies for the preprocessing pipeline.
#
# Clones all required external repositories into preprocessing/third_party/.
# Run this once before using the preprocessing pipeline.
#
# Usage:
#   bash setup_third_party.sh
# ==============================================================================
set -eo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
TP_DIR="${SCRIPT_DIR}/third_party"
mkdir -p "$TP_DIR"

echo "============================================"
echo "  Setting up third-party dependencies"
echo "  Target: ${TP_DIR}"
echo "============================================"
echo ""

# ---------- SAM2 (Segment Anything 2) ----------
if [ -d "$TP_DIR/sam2" ]; then
    echo "[SAM2] Already exists, skipping."
else
    echo "[SAM2] Cloning..."
    git clone https://github.com/facebookresearch/sam2.git "$TP_DIR/sam2"
    echo "[SAM2] Installing..."
    cd "$TP_DIR/sam2" && pip install -e . && cd "$SCRIPT_DIR"
fi
echo "  Checkpoint needed: sam2.1_hiera_large.pt"
echo "  Download from: https://github.com/facebookresearch/sam2#download-checkpoints"
echo "  Place at: ${TP_DIR}/sam2/checkpoints/sam2.1_hiera_large.pt"
echo ""

# ---------- SAM-3D-Objects ----------
if [ -d "$TP_DIR/sam-3d-objects" ]; then
    echo "[SAM-3D-Objects] Already exists, skipping."
else
    echo "[SAM-3D-Objects] Cloning..."
    git clone https://github.com/prs-eth/SAM-3D-Objects.git "$TP_DIR/sam-3d-objects"
fi
echo "  Checkpoint needed: see SAM-3D-Objects README for download instructions."
echo ""

# ---------- GVHMR ----------
if [ -d "$TP_DIR/GVHMR" ]; then
    echo "[GVHMR] Already exists, skipping."
else
    echo "[GVHMR] Cloning..."
    git clone https://github.com/zju3dv/GVHMR.git "$TP_DIR/GVHMR"
fi
echo "  Checkpoints needed: see GVHMR README for download instructions."
echo "  Place body models at: ${TP_DIR}/GVHMR/inputs/checkpoints/body_models/"
echo ""

# ---------- Depth-Anything-V2 ----------
if [ -d "$TP_DIR/Depth-Anything-V2" ]; then
    echo "[Depth-Anything-V2] Already exists, skipping."
else
    echo "[Depth-Anything-V2] Cloning..."
    git clone https://github.com/DepthAnything/Depth-Anything-V2.git "$TP_DIR/Depth-Anything-V2"
fi
echo "  Checkpoint needed: depth_anything_v2_vitl.pth"
echo "  Download from: https://github.com/DepthAnything/Depth-Anything-V2#pre-trained-models"
echo "  Place at: ${TP_DIR}/Depth-Anything-V2/checkpoints/depth_anything_v2_vitl.pth"
echo ""

# ---------- SAM-3D-Body ----------
if [ -d "$TP_DIR/sam-3d-body" ]; then
    echo "[SAM-3D-Body] Already exists, skipping."
else
    echo "[SAM-3D-Body] Cloning..."
    git clone https://github.com/prs-eth/SAM-3D-Body.git "$TP_DIR/sam-3d-body"
fi
echo "  Checkpoint needed: sam-3d-body-dinov3/model.ckpt"
echo "  Download from: see SAM-3D-Body README for download instructions."
echo "  Place at: ${TP_DIR}/sam-3d-body/checkpoints/sam-3d-body-dinov3/model.ckpt"
echo ""

echo "============================================"
echo "  Clone complete!"
echo "============================================"
echo ""
echo "Next steps:"
echo "  1. Download checkpoints for each model (see messages above)"
echo "  2. Review preprocessing/config.sh (paths should auto-resolve)"
echo "  3. Run: bash preprocessing/run_pipeline.sh <session_dir>"
