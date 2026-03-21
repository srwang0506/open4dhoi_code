#!/usr/bin/env bash
# Evaluate InterPoint model on 4D-HOI test set
#
# Usage:
#   bash evaluate.sh <checkpoint_path> [extra_args...]
#
# Example:
#   bash evaluate.sh checkpoints/4dhoi/epoch_080.pth
#   bash evaluate.sh checkpoints/4dhoi/epoch_080.pth --threshold_human 0.3

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
CHECKPOINT="${1:?Usage: $0 <checkpoint_path> [extra_args...]}"
shift

python "${SCRIPT_DIR}/scripts/evaluate_4dhoi_new.py" \
    --checkpoint "$CHECKPOINT" \
    "$@"
