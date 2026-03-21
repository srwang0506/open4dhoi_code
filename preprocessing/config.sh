#!/usr/bin/env bash
# ==============================================================================
# 4DHOI Preprocessing Pipeline - Configuration
#
# Paths default to preprocessing/third_party/ (populated by setup_third_party.sh).
# Override any variable by exporting it before sourcing this file.
# ==============================================================================

# ---------- Resolve project root ----------
PIPELINE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
_TP="${PIPELINE_ROOT}/preprocessing/third_party"

# ---------- Conda environments ----------
export ENV_SAM3D_OBJ="${ENV_SAM3D_OBJ:-sam3d_obj_4d}"
export ENV_PIPELINE="${ENV_PIPELINE:-4dhoi_pipeline}"
export ENV_MHR="${ENV_MHR:-mhr}"

# ---------- CUDA device assignment ----------
export CUDA_EXTRACT_FRAMES="${CUDA_EXTRACT_FRAMES:-}"
export CUDA_MASKS="${CUDA_MASKS:-0}"
export CUDA_OBJ_ORG="${CUDA_OBJ_ORG:-0}"
export CUDA_MOTION="${CUDA_MOTION:-0}"
export CUDA_DEPTH="${CUDA_DEPTH:-0}"
export CUDA_HAND="${CUDA_HAND:-0}"
export CUDA_HOI="${CUDA_HOI:-}"

# ---------- Third-party model paths ----------
# Each defaults to third_party/<name>. Override with env vars if installed elsewhere.

# SAM2 (mask generation)
export SAM2_ROOT="${SAM2_ROOT:-${_TP}/sam2}"
export SAM2_CHECKPOINT="${SAM2_CHECKPOINT:-${SAM2_ROOT}/checkpoints/sam2.1_hiera_large.pt}"

# SAM-3D-Objects (object mesh reconstruction)
export SAM3D_OBJ_ROOT="${SAM3D_OBJ_ROOT:-${_TP}/sam-3d-objects}"

# GVHMR (human motion estimation)
export GVHMR_ROOT="${GVHMR_ROOT:-${_TP}/GVHMR}"

# Depth-Anything-V2 (depth estimation)
export DEPTH_ANYTHING_ROOT="${DEPTH_ANYTHING_ROOT:-${_TP}/Depth-Anything-V2}"
export DEPTH_ANYTHING_CHECKPOINT="${DEPTH_ANYTHING_CHECKPOINT:-${DEPTH_ANYTHING_ROOT}/checkpoints/depth_anything_v2_vitl.pth}"

# SAM-3D-Body (hand pose refinement)
export SAM3D_BODY_ROOT="${SAM3D_BODY_ROOT:-${_TP}/sam-3d-body}"
export SAM3D_BODY_CHECKPOINT="${SAM3D_BODY_CHECKPOINT:-${SAM3D_BODY_ROOT}/checkpoints/sam-3d-body-dinov3/model.ckpt}"
export SAM3D_BODY_MHR="${SAM3D_BODY_MHR:-${SAM3D_BODY_ROOT}/checkpoints/sam-3d-body-dinov3/assets/mhr_model.pt}"

# ---------- Shared model files ----------
export SMPLX_MODEL="${SMPLX_MODEL:-${PIPELINE_ROOT}/shared_data/SMPLX_NEUTRAL.npz}"

# ---------- Script directory ----------
export SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/scripts" && pwd)"
