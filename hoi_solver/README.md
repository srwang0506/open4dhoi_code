# 4DHOI Solver

Optimization solver for 4D Human-Object Interaction. Given annotated contact points between human body joints and object surfaces, optimizes body poses and object transforms to satisfy physical contact constraints.

## Pipeline

```
kp_record_merged.json (from annotator, decimated mesh indices)
    |
    v  (automatic, built into optimize.py)
kp_record_new.json (original mesh indices)
    |
    v
[optimize.py]  Per-frame least-squares + optional Adam refinement
    |
    v
final_optimized_parameters/all_parameters_latest.json
    |
    v
[render.py]  (optional) Render global-view visualization video
```

The annotation conversion (decimated mesh → original mesh indices) is built into
`optimize.py` and runs automatically when `kp_record_new.json` doesn't exist yet.
You can also run it standalone via `convert_annotations.py`.

## Quick Start

```bash
# Run full pipeline (convert + optimize)
bash run.sh /path/to/session

# With rendering
bash run.sh /path/to/session --render

# Least-squares only (faster, no Adam refinement)
bash run.sh /path/to/session --use_least_squares_only

# Specific frame range
bash run.sh /path/to/session --start_frame 0 --end_frame 50
```

## Input Requirements

The session directory must contain:
```
session_folder/
├── video.mp4                    # Input video
├── obj_org.obj                  # Original object mesh
├── kp_record_merged.json        # Annotations (from 4dhoi_annotator)
├── motion/
│   ├── result_hand.pt           # SMPL-X params with hands (preferred)
│   └── result.pt                # SMPL-X params (fallback)
└── align/ or output/
    └── obj_poses.json           # Object scale and position
```

## Output

```
session_folder/
├── kp_record_new.json                          # Annotations mapped to original mesh
├── obj_init.obj                                # Preprocessed object mesh
├── final_optimized_parameters/
│   ├── all_parameters_latest.json              # Optimized human + object params
│   └── transformed_parameters_final.json       # Global-space transforms (if rendered)
├── output_render.mp4                           # Visualization (if --render)
└── optimize_summary.json                       # Summary metadata
```

## Optimization Details

### Phase 1: Least-Squares (per frame)
- Builds SMPL-X human mesh from motion parameters
- Establishes 3D-3D correspondences (object vertex to body vertex)
- Establishes 3D-2D correspondences (object vertex to image keypoint)
- Solves weighted least-squares for object rotation R and translation t
- 3D weight: 900.0, 2D weight: 2.0 (heavily favors 3D constraints)

### Phase 2: Adam Refinement (optional, per frame)
- Refines body pose, hand pose, and object transform jointly
- Contact loss: penalizes non-contact between paired body/object points
- Collision loss: penalizes mesh interpenetration
- Mask loss: penalizes misalignment with 2D segmentation masks

### Configuration
Edit `video_optimizer/configs/optimizer.yaml`:
```yaml
optimize:
  steps: 100        # Adam iterations per frame
loss_weights:
  contact: 50.0     # Contact constraint weight
  collision: 8.0    # Collision avoidance weight
  mask: 0.05        # 2D mask alignment weight
```

## Scripts

| Script | Purpose |
|--------|---------|
| `run.sh` | One-command pipeline: optimize + optional render |
| `optimize.py` | Core optimization (auto-converts annotations, least-squares + Adam) |
| `render.py` | Render optimized result as global-view video |
| `convert_annotations.py` | Standalone annotation index conversion (also built into optimize.py) |
