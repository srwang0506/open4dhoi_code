#!/usr/bin/env bash
# Train InterPoint model on 4D-HOI dataset
#
# Usage:
#   bash train.sh [--checkpoint /path/to/pretrained.pth]
#
# The model freezes all layers except the transformer and prediction head,
# then fine-tunes on the 4D-HOI dataset.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
CHECKPOINT="${1:---checkpoint}"

# Default: train from scratch (no pretrained checkpoint)
# To fine-tune from InterCap pretrained weights, pass the checkpoint path:
#   bash train.sh --checkpoint checkpoints/pretrained/epoch_078.pth

python "${SCRIPT_DIR}/scripts/train_4dhoi.py" \
    --batch_size 24 \
    --epochs 80 \
    --lr 3e-5 \
    --frame_interval 10 \
    --test_split_ratio 0.2 \
    --project interpoint-4dhoi \
    --run_name 4dhoi-train \
    --save_dir checkpoints/4dhoi \
    "$@"
