# 4DHOI Preprocessing Pipeline

Automated pipeline for preprocessing video data into 4D Human-Object Interaction scenes. Extracts frames, generates segmentation masks, reconstructs 3D object meshes, estimates human motion and depth, refines hand poses, and assembles the final HOI scene.

## Pipeline Overview

```
video.mp4 + select_id.json + points.json
    |
    v
[Step 1] Extract Frames → Generate Masks → Reconstruct Object  (env: sam3d_obj_4d)
    |
    v
[Step 2] Estimate Motion (GVHMR) + Estimate Depth              (env: 4dhoi_pipeline)
    |
    v
[Step 3] Refine Hand Pose (SAM-3D-Body)                        (env: mhr)
    |
    v
[Step 4] Assemble HOI Scene (scale + position)                  (env: 4dhoi_pipeline)
    |
    v
Ready for annotation with 4dhoi_annotator
```

## Quick Start

### 1. Setup Third-Party Dependencies

```bash
# Clone all required external repos into preprocessing/third_party/
bash setup_third_party.sh
```

This clones:

| Repository | Purpose | Checkpoints Required |
|------------|---------|---------------------|
| [SAM2](https://github.com/facebookresearch/sam2) | Video segmentation (step 1) | `sam2.1_hiera_large.pt` |
| [SAM-3D-Objects](https://github.com/prs-eth/SAM-3D-Objects) | Single-image 3D reconstruction (step 1) | See repo README |
| [GVHMR](https://github.com/zju3dv/GVHMR) | Human motion estimation (step 2) | See repo README |
| [Depth-Anything-V2](https://github.com/DepthAnything/Depth-Anything-V2) | Monocular depth estimation (step 2) | `depth_anything_v2_vitl.pth` |
| [SAM-3D-Body](https://github.com/prs-eth/SAM-3D-Body) | Hand pose refinement (step 3) | `sam-3d-body-dinov3/model.ckpt` |

After cloning, download the required checkpoints for each model following their respective READMEs. The `config.sh` defaults automatically point to `third_party/`.

### 2. Configure (Optional)

`config.sh` defaults work out of the box after `setup_third_party.sh`. Override if needed:

### 2. Prepare Input Data

Each session directory must contain:

```
session_folder/
├── video.mp4          # Input video
├── select_id.json     # Frame selection metadata
└── points.json        # Click prompts for mask generation
```

**select_id.json** format:
```json
{
  "select_id": 0,
  "start_id": 0,
  "object_name": "spear"
}
```
- `select_id`: Frame index used for 3D object reconstruction
- `start_id`: Starting frame offset (0 = no trimming)

**points.json** format:
```json
{
  "human_points": [[x1, y1], [x2, y2]],
  "object_points": [[x1, y1], [x2, y2]]
}
```
- Pixel coordinates of click prompts for SAM2 segmentation (on the first frame)

### 3. Run

```bash
# Full pipeline
bash run_pipeline.sh /path/to/session_folder

# With hand retargeting and rendering
bash run_pipeline.sh /path/to/session_folder --retarget --smooth 0.3 --render

# Force re-run (delete existing outputs first)
bash run_pipeline.sh /path/to/session_folder --force

# Run individual steps
bash step1_frames_masks_obj.sh /path/to/session_folder
bash step2_motion_depth.sh /path/to/session_folder
bash step3_hand.sh /path/to/session_folder --retarget
bash step4_hoi.sh /path/to/session_folder --render
```

## Steps in Detail

### Step 1: Frames + Masks + Object Mesh
**Conda env:** `sam3d_obj_4d`

| Sub-step | Script | Input | Output |
|----------|--------|-------|--------|
| Extract frames | `make_extract_frames.py` | `video.mp4`, `select_id.json` | `frames/*.jpg` |
| Generate masks | `make_masks.py` | `frames/`, `points.json` | `mask_dir/*.png`, `human_mask_dir/*.png` |
| Reconstruct object | `make_obj_org.py` | `frames/`, `mask_dir/`, `select_id.json` | `obj_org.obj` |

### Step 2: Motion + Depth
**Conda env:** `4dhoi_pipeline`

| Sub-step | Script | Input | Output |
|----------|--------|-------|--------|
| Human motion (GVHMR) | `make_motion.py` | `video.mp4` | `motion/result.pt` |
| Depth estimation | `make_depth.py` | `frames/` | `depth.npy` |

### Step 3: Hand Pose Refinement
**Conda env:** `mhr`

| Sub-step | Script | Input | Output |
|----------|--------|-------|--------|
| SAM-3D-Body | `make_hand_sam3d.py` | `frames/`, `motion/result.pt` | `motion/result_hand.pt` |

Options:
- `--retarget`: Align SAM3D arm pose to GVHMR skeleton
- `--smooth_cutoff 0.3`: Apply Gaussian smoothing to hand joints

### Step 4: HOI Assembly
**Conda env:** `4dhoi_pipeline`

| Sub-step | Script | Input | Output |
|----------|--------|-------|--------|
| Assemble HOI | `make_hoi.py` | `motion/result_hand.pt`, `depth.npy`, `obj_org.obj`, `mask_dir/` | `output/obj_poses.json` |

Options:
- `--render`: Generate `output/global.mp4` visualization

## Output Structure

After the full pipeline, the session directory will contain:

```
session_folder/
├── video.mp4                # Input (may be trimmed)
├── select_id.json           # Input
├── points.json              # Input
├── frames/                  # Extracted frames
│   ├── 00000.jpg
│   └── ...
├── mask_dir/                # Object segmentation masks
│   ├── 00000.png
│   └── ...
├── human_mask_dir/          # Human segmentation masks
│   ├── 00000.png
│   └── ...
├── obj_org.obj              # 3D object mesh (with vertex colors)
├── depth.npy                # Depth maps (T, H, W)
├── motion/
│   ├── result.pt            # SMPL-X parameters (body only)
│   ├── result_hand.pt       # SMPL-X parameters (with refined hands)
│   ├── sam3d_params.pt      # Raw SAM-3D-Body output
│   └── global.mp4           # Motion visualization
└── output/
    └── obj_poses.json       # Object scale and position
```

## External Dependencies

The pipeline requires these pre-trained models:

| Model | Used By | Purpose |
|-------|---------|---------|
| [SAM2](https://github.com/facebookresearch/sam2) | Step 1 | Video segmentation |
| [SAM-3D-Objects](https://github.com/prs-eth/sam-3d-objects) | Step 1 | Single-image 3D reconstruction |
| [GVHMR](https://github.com/zju3dv/GVHMR) | Step 2 | Human motion estimation |
| [Depth-Anything-V2](https://github.com/DepthAnything/Depth-Anything-V2) | Step 2 | Monocular depth estimation |
| [SAM-3D-Body](https://github.com/prs-eth/sam-3d-body) | Step 3 | Hand pose refinement |
| SMPL-X | Steps 3-4 | Parametric body model |
