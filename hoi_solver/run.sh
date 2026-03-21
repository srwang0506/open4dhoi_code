#!/usr/bin/env bash
# ==============================================================================
# 4DHOI Solver - Run optimization pipeline for a single session
#
# The optimization automatically converts annotations from decimated mesh
# to original mesh indices (kp_record_merged.json → kp_record_new.json)
# before running the solver.
#
# Usage:
#   bash run.sh <data_dir> [options]
#
# Options:
#   --render                    Also render the result video after optimization
#   --use_least_squares_only    Skip Adam optimization (faster, less refined)
#   --start_frame N             Optimize from frame N
#   --end_frame N               Optimize until frame N (inclusive)
#   --best_frame N              Specify best frame for static objects
#   --gpu N                     CUDA device (default: 0)
#
# Example:
#   bash run.sh /path/to/session
#   bash run.sh /path/to/session --render --use_least_squares_only
# ==============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

# Parse arguments
DATA_DIR=""
RENDER=false
GPU="0"
OPT_ARGS=()
RENDER_ARGS=()

while [[ $# -gt 0 ]]; do
    case $1 in
        --render)
            RENDER=true
            shift
            ;;
        --gpu)
            GPU="$2"
            shift 2
            ;;
        --start_frame|--end_frame|--end_frame_exclusive|--frame|--best_frame|--max_frames)
            OPT_ARGS+=("$1" "$2")
            RENDER_ARGS+=("$1" "$2")
            shift 2
            ;;
        --use_least_squares_only|--save_ls_meshes)
            OPT_ARGS+=("$1")
            shift
            ;;
        *)
            if [ -z "$DATA_DIR" ]; then
                DATA_DIR="$1"
            fi
            shift
            ;;
    esac
done

if [ -z "$DATA_DIR" ]; then
    echo "Usage: $0 <data_dir> [--render] [--gpu N] [--use_least_squares_only] [...]"
    exit 1
fi

DATA_DIR="$(cd "$DATA_DIR" && pwd)"
export CUDA_VISIBLE_DEVICES="$GPU"

echo "============================================"
echo "  4DHOI Solver"
echo "  Session: ${DATA_DIR}"
echo "  GPU: ${GPU}"
echo "============================================"

# Run optimization (annotation conversion is built-in)
echo ""
echo "Running HOI optimization..."
echo "(Annotations will be auto-converted from decimated to original mesh if needed)"
python "${SCRIPT_DIR}/optimize.py" \
    --data_dir "${DATA_DIR}" \
    "${OPT_ARGS[@]}"

# Optional: Render
if [ "$RENDER" = true ]; then
    echo ""
    echo "Generating visualization..."
    python "${SCRIPT_DIR}/render.py" \
        --data_dir "${DATA_DIR}" \
        --save_transformed_params \
        "${RENDER_ARGS[@]}"
fi

echo ""
echo "============================================"
echo "  Optimization complete!"
echo "============================================"
echo "Output: ${DATA_DIR}/final_optimized_parameters/"
