# InterPoint

Interaction point prediction model for 4D Human-Object Interaction. Given an RGB image and a 3D object point cloud, the model predicts which human body joints are in contact with the object and where on the object surface those contacts occur.

## Architecture

```
RGB Image (224x224)          Object Point Cloud (1024x3)
       |                              |
  [VLM Encoder]               [PointNet++ Encoder]
       |                              |
       +----------+-------------------+
                  |
      [Interaction Transformer]
         Cross-modal attention
                  |
          +-------+-------+
          |               |
   [Human Head]    [Object Head]
   87-way binary   87x3 coordinates
   classification  on object surface
```

- **VLM Encoder**: LLaVA-based vision-language model for semantic features
- **PointNet++ Encoder**: 3D point cloud feature extraction
- **Interaction Transformer**: Cross-attention fusion of image and geometry features
- **Human Head**: Predicts contact probability for 87 SMPL-X body keypoints
- **Object Head**: Predicts 3D coordinates on the object surface for each contact

## Project Structure

```
interpoint/
├── models/                    # Model architecture
│   ├── ivd_model.py           # Main model (IVDModel)
│   ├── vlm_module.py          # VLM encoder (LLaVA)
│   ├── interaction_transformer.py  # Cross-modal transformer
│   ├── pointnet2_encoder.py   # PointNet++ encoder
│   ├── pointnet2_decoder.py   # Feature decoder
│   └── losses.py              # Loss functions
│
├── data/                      # Dataset and transforms
│   ├── dataset_4dhoi.py       # 4D-HOI dataset loader
│   ├── transforms.py          # Image preprocessing
│   └── part_kp.json           # 87 SMPL-X keypoint definitions
│
├── utils/                     # Utilities
│   ├── keypoints.py           # Keypoint manager
│   └── metrics.py             # Evaluation metrics
│
├── configs/
│   └── default.yaml           # Model configuration
│
├── scripts/
│   ├── train_4dhoi.py         # Training script
│   ├── evaluate_4dhoi_new.py  # Evaluation script
│   └── fixed_split_utils.py   # Train/test split utilities
│
├── train.sh                   # Training launcher
├── evaluate.sh                # Evaluation launcher
└── requirements.txt           # Dependencies
```

## Quick Start

### Install

```bash
pip install -r requirements.txt
# pytorch3d must be built from source, see https://github.com/facebookresearch/pytorch3d
```

### Download Pretrained Checkpoints

Download from Google Drive and place in `checkpoints/`:

| Checkpoint | Description | Link |
|------------|-------------|------|
| `4dhoi/epoch_080.pth` | Trained on 4D-HOI dataset | [Google Drive](TODO) |

```bash
mkdir -p checkpoints/4dhoi
# Download and place: checkpoints/4dhoi/epoch_080.pth
```

### Train

```bash
# Train from scratch on 4D-HOI
bash train.sh

# Fine-tune from pretrained checkpoint
bash train.sh --checkpoint /path/to/pretrained.pth
```

Training parameters (configurable in `train.sh`):
- Batch size: 24
- Epochs: 80
- Learning rate: 3e-5
- Frame interval: 10 (sample every 10th frame)
- Test split: 20%

### Evaluate

```bash
bash evaluate.sh checkpoints/4dhoi/epoch_080.pth
```

Evaluation metrics:
- **Human**: Precision, Recall, F1 for 87-way binary classification
- **Object**: Point-coverage recall (predicted points near GT on object surface)
- **Interaction**: Pair recall (both human joint and object point correct)

## Data Format

The 4D-HOI dataset expects session directories with:

```
session_folder/
├── video.mp4                    # or frames/ directory
├── obj_init.obj                 # Object mesh (or obj_org.obj)
├── kp_record_new.json           # Keypoint annotations
│   {
│     "00000": {                 # Frame index
│       "joint_name": vertex_idx,  # Body joint → object vertex
│       "2D_keypoint": [...]
│     }
│   }
└── motion/
    └── result_hand.pt           # SMPL-X parameters (for body keypoint positions)
```

## Configuration

Model hyperparameters in `configs/default.yaml`:
- `d_tr`: Transformer dimension (256)
- `num_body_points`: Number of human keypoints (87)
- `num_object_queries`: Number of object surface queries (87)

## Integration with 4DHOI Annotator

This model is used by the [4dhoi_annotator](../4dhoi_annotator/) for automatic interaction point prediction. The annotator's `ivd_predictor.py` wraps this model and expects it at `../interpoint/` relative path.

## License

[TODO: Add license]

## Citation

[TODO: Add citation]
