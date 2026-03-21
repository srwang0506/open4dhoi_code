# 4DHOI Annotator

Interactive web-based annotation tool for 4D Human-Object Interaction (HOI). Annotate contact points between human body joints and object surfaces, then optimize body poses to satisfy physical constraints.

## Features

- **Web-based 3D visualization** using Plotly.js with interactive camera control
- **SMPL-X body model** rendering with hand pose support
- **Per-frame annotation** of 3D contact points (body joint to object vertex)
- **2D point tracking** via CoTracker for temporal propagation
- **Optional auto-prediction** of interaction points using the [InterPoint](../interpoint/) model
- **Contact-based pose optimization** via the built-in HOI solver
- **Object scale adjustment** for correcting mesh size mismatches

## Project Structure

```
4dhoi_annotator/
├── app.py                 # Flask application (main entry point)
├── ivd_predictor.py       # InterPoint model wrapper (optional)
├── config.yaml            # Configuration file
├── geometry.py            # Rotation/quaternion utilities
├── requirements.txt       # Python dependencies
│
├── solver/                # Pose optimization
│   ├── hoi_solver.py      # HOI solver implementation
│   ├── kp_use_new.py      # Keypoint optimization pipeline
│   ├── config.py          # Optimizer configuration
│   ├── optimizer_part.py  # Optimization logic
│   └── data/              # Joint mappings (part_kp.json, etc.)
│
├── co-tracker/            # CoTracker for 2D point tracking
│
├── static/js/main.js      # Frontend JavaScript
├── templates/index.html    # Web UI template
│
├── asset/data/             # SMPL-X model (SMPLX_NEUTRAL.npz)
└── demo/                   # Example session data
```

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt

# Clone and install CoTracker
git clone https://github.com/facebookresearch/co-tracker.git co-tracker
cd co-tracker && pip install -e . && cd ..
```

Download the CoTracker checkpoint:
```bash
mkdir -p co-tracker/checkpoints
# Download scaled_online.pth from https://github.com/facebookresearch/co-tracker#download
# Place at: co-tracker/checkpoints/scaled_online.pth
```

### 2. Prepare Data

Each annotation session requires the following files in a directory:

```
session_folder/
├── video.mp4                    # Input video
├── obj_org.obj                  # Object mesh (with optional vertex colors)
├── motion/
│   └── result_hand.pt           # SMPL-X motion parameters (or result.pt)
└── output/                      # (or align/)
    └── obj_poses.json           # Per-frame object pose (scale, rotation, translation)
```

Place session folders under `demo/` or any data directory.

### 3. Download Model Weights

- **SMPL-X model**: Download `SMPLX_NEUTRAL.npz` and place it in `asset/data/`
- **CoTracker**: Download `scaled_online.pth` and place it in `co-tracker/checkpoints/`
- **InterPoint model** (optional): Set up the [interpoint](../interpoint/) sibling directory with checkpoints

### 4. Run

```bash
# Load all sessions from demo/ directory
python app.py --data_dir demo

# Or load a specific session directly
python app.py --video_dir demo/spear_example

# Custom host/port
python app.py --data_dir demo --host 0.0.0.0 --port 8080
```

Then open `http://localhost:5027` in your browser.

## Configuration

Edit `config.yaml` to customize behavior:

```yaml
# Disable auto-prediction if InterPoint model is not available
ivd_model:
  enabled: false

# Adjust mesh decimation for performance
mesh:
  obj_decimation_target_faces: 30000

# Change server settings
server:
  host: "0.0.0.0"
  port: 5027
```

### InterPoint Auto-Prediction

The tool optionally integrates with the InterPoint model for automatic interaction point prediction. To use this feature:

1. Set up the `interpoint/` directory as a sibling to this project (or set the `INTERPOINT_DIR` environment variable)
2. Ensure `ivd_model.enabled: true` in `config.yaml`
3. The model will be loaded at startup and available via the "Auto Predict" button in the UI

To run without auto-prediction, set `ivd_model.enabled: false`.

## Data Format

### Input: Object Poses (`obj_poses.json`)

```json
{
  "scale": 1.0,
  "rotation": [[[r00, r01, r02], [r10, r11, r12], [r20, r21, r22]], ...],
  "center": [[cx, cy, cz], ...]
}
```

### Output: Merged Annotations (`kp_record_merged.json`)

```json
{
  "00000": {
    "joint_name": vertex_index,
    "2D_keypoint": [[obj_vertex_index, [pixel_x, pixel_y]], ...]
  },
  "00001": { ... },
  "object_scale": 1.0,
  "is_static_object": false,
  "start_frame_index": 0
}
```

## Annotation Workflow

1. **Check Scale**: Verify and adjust the object mesh scale relative to the human body
2. **Select Contact Points**: Click on the 3D object mesh to place contact points, then assign body joints
3. **Track 2D Points**: Use CoTracker to propagate 2D reference points across frames
4. **Auto-Predict** (optional): Use the InterPoint model to suggest contact points automatically
5. **Save & Optimize**: Merge annotations and run the HOI solver to optimize body poses

## License

[TODO: Add license]

## Citation

[TODO: Add citation]
