#!/usr/bin/env bash
# ==============================================================================
# 4DHOI Preprocessing Pipeline - Full Pipeline Runner
#
# Runs all 4 steps sequentially for a given session directory.
# Each step checks for existing outputs and skips if already done.
#
# Usage:
#   bash run_pipeline.sh <video_dir> [options]
#
# Options:
#   --force          Re-run all steps even if outputs exist
#   --render         Render HOI visualization in step 4
#   --retarget       Enable arm retargeting in step 3
#   --smooth 0.3     Hand smoothing cutoff for step 3
#
# Example:
#   bash run_pipeline.sh ./data/my_session
#   bash run_pipeline.sh ./data/my_session --retarget --smooth 0.3 --render
#
# Prerequisites:
#   Session directory must contain:
#     - video.mp4              (input video)
#     - select_id.json         (frame selection: {"select_id": 0, "start_id": 0})
#     - points.json            (click prompts: {"human_points": [...], "object_points": [...]})
#
# Conda environments required:
#   - sam3d_obj_4d    (step 1: frames, masks, object mesh)
#   - 4dhoi_pipeline  (step 2: motion, depth; step 4: HOI assembly)
#   - mhr             (step 3: hand pose refinement)
# ==============================================================================
set -euo pipefail

PIPELINE_DIR="$(cd "$(dirname "$0")" && pwd)"

# Parse arguments
VIDEO_DIR=""
FORCE=false
RENDER=""
HAND_ARGS=()

while [[ $# -gt 0 ]]; do
    case $1 in
        --force)
            FORCE=true
            shift
            ;;
        --render)
            RENDER="--render"
            shift
            ;;
        --retarget)
            HAND_ARGS+=("--retarget")
            shift
            ;;
        --smooth)
            HAND_ARGS+=("--smooth_cutoff" "$2")
            shift 2
            ;;
        *)
            if [ -z "$VIDEO_DIR" ]; then
                VIDEO_DIR="$1"
            fi
            shift
            ;;
    esac
done

if [ -z "$VIDEO_DIR" ]; then
    echo "Usage: $0 <video_dir> [--force] [--render] [--retarget] [--smooth N]"
    exit 1
fi

VIDEO_DIR="$(cd "$VIDEO_DIR" && pwd)"

echo "======================================================"
echo "  4DHOI Preprocessing Pipeline"
echo "======================================================"
echo "  Session:   ${VIDEO_DIR}"
echo "  Force:     ${FORCE}"
echo "  Render:    ${RENDER:-no}"
echo "  Hand args: ${HAND_ARGS[*]:-none}"
echo "======================================================"
echo ""

# Validate prerequisites
for f in video.mp4 select_id.json points.json; do
    if [ ! -f "${VIDEO_DIR}/${f}" ]; then
        echo "ERROR: Required file not found: ${VIDEO_DIR}/${f}"
        exit 1
    fi
done

# Remove outputs if --force is set
if [ "$FORCE" = true ]; then
    echo "Force mode: removing existing outputs..."
    rm -rf "${VIDEO_DIR}/frames"
    rm -rf "${VIDEO_DIR}/mask_dir"
    rm -rf "${VIDEO_DIR}/human_mask_dir"
    rm -f  "${VIDEO_DIR}/obj_org.obj"
    rm -rf "${VIDEO_DIR}/motion"
    rm -f  "${VIDEO_DIR}/depth.npy"
    rm -rf "${VIDEO_DIR}/output"
    echo ""
fi

# Run steps
echo ">>> Step 1: Frames + Masks + Object Mesh (env: sam3d_obj_4d)"
bash "${PIPELINE_DIR}/step1_frames_masks_obj.sh" "${VIDEO_DIR}"
echo ""

echo ">>> Step 2: Motion + Depth (env: 4dhoi_pipeline)"
bash "${PIPELINE_DIR}/step2_motion_depth.sh" "${VIDEO_DIR}"
echo ""

echo ">>> Step 3: Hand Pose Refinement (env: mhr)"
bash "${PIPELINE_DIR}/step3_hand.sh" "${VIDEO_DIR}" "${HAND_ARGS[@]}"
echo ""

echo ">>> Step 4: HOI Assembly (env: 4dhoi_pipeline)"
bash "${PIPELINE_DIR}/step4_hoi.sh" "${VIDEO_DIR}" ${RENDER}
echo ""

echo "======================================================"
echo "  Pipeline complete!"
echo "======================================================"
echo ""
echo "Output files:"
echo "  ${VIDEO_DIR}/frames/             - Extracted video frames"
echo "  ${VIDEO_DIR}/mask_dir/           - Object segmentation masks"
echo "  ${VIDEO_DIR}/human_mask_dir/     - Human segmentation masks"
echo "  ${VIDEO_DIR}/obj_org.obj         - 3D object mesh"
echo "  ${VIDEO_DIR}/motion/result.pt    - SMPL-X motion parameters"
echo "  ${VIDEO_DIR}/motion/result_hand.pt - With refined hand pose"
echo "  ${VIDEO_DIR}/depth.npy           - Depth maps"
echo "  ${VIDEO_DIR}/output/obj_poses.json - Object scale and position"
echo ""
echo "Ready for annotation with 4dhoi_annotator."
