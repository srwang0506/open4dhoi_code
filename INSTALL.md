# Installation Guide

This guide covers environment setup for the full 4DHOI pipeline: preprocessing, annotation, and optimization.

## Overview

The pipeline uses multiple conda environments:

| Environment | Python | Used For |
|-------------|--------|----------|
| `sam3d_obj_4d` | 3.11 | Preprocessing Step 1 (frames, masks, object mesh) |
| `4dhoi_pipeline` | 3.10 | Preprocessing Steps 2 & 4, annotator, optimization |
| `mhr` | 3.12 | Preprocessing Step 3 (hand pose refinement) |

---

## 1. Preprocessing: Third-Party Repositories

Clone all required external repositories:

```bash
bash preprocessing/setup_third_party.sh
```

This clones into `preprocessing/third_party/`:
- [SAM2](https://github.com/facebookresearch/sam2)
- [SAM-3D-Objects](https://github.com/facebookresearch/sam-3d-objects)
- [GVHMR](https://github.com/zju3dv/GVHMR)
- [Depth-Anything-V2](https://github.com/DepthAnything/Depth-Anything-V2)
- [SAM-3D-Body](https://github.com/facebookresearch/sam-3d-body)

Then download the required checkpoints for each model following their respective READMEs. See `preprocessing/README.md` for checkpoint locations.

---

## 2. Environment: `sam3d_obj_4d` (Preprocessing Step 1)

```bash
conda create -n sam3d_obj_4d python=3.11 -y
conda activate sam3d_obj_4d
```

Install PyTorch (CUDA 12.8):

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
```

Install SAM2:

```bash
cd preprocessing/third_party/sam2
pip install -e .
cd ../../..
```

Install remaining dependencies:

```bash
pip install open3d opencv-python trimesh scipy numpy pytorch3d
```

---

## 3. Environment: `4dhoi_pipeline` (Main Environment)

This is the primary environment used for preprocessing steps 2 & 4, the annotator app, and the HOI solver/optimization.

```bash
conda create -n 4dhoi_pipeline python=3.10 -y
conda activate 4dhoi_pipeline
```

Install PyTorch (CUDA 12.x):

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

Install core dependencies:

```bash
pip install open3d opencv-python trimesh scipy numpy smplx einops roma \
    hydra-core omegaconf pytorch-lightning lightning chumpy \
    open-clip-torch torch_scatter flask PyYAML
```

Install pytorch3d (follow [official instructions](https://github.com/facebookresearch/pytorch3d/blob/main/INSTALL.md) for your CUDA version).

### GVHMR

```bash
cd preprocessing/third_party/GVHMR
pip install -e .
cd ../../..
```

### Optimization Dependencies (HOI Solver)

The `multiperson` and `neural_renderer` packages are included in `hoi_solver/`. Install them:

```bash
cd hoi_solver/multiperson/sdf
pip install -e . --no-build-isolation
cd ../../..

cd hoi_solver/neural_renderer
pip install -e . --no-build-isolation
cd ..
```

---

## 4. Environment: `mhr` (Preprocessing Step 3 — Hand Pose)

Requires [MHR](https://github.com/facebookresearch/MHR) and [SAM-3D-Body](https://github.com/facebookresearch/sam-3d-body). SAM-3D-Body is already cloned into `preprocessing/third_party/sam-3d-body` via `setup_third_party.sh`.

**Step 1: Create conda environment**

```bash
conda create -n mhr python=3.12
conda activate mhr
conda install -c conda-forge pymomentum-cpu
```

**Step 2: Install dependencies**

```bash
cd preprocessing/third_party/sam-3d-body
pip install pytorch-lightning pyrender opencv-python yacs scikit-image einops timm \
    dill pandas rich hydra-core hydra-submitit-launcher hydra-colorlog pyrootutils \
    webdataset networkx==3.2.1 roma joblib seaborn wandb appdirs appnope ffmpeg \
    cython jsonlines pytest xtcocotools loguru optree fvcore black pycocotools \
    tensorboard huggingface_hub
cd ../../..
```

---

## 5. Body Models (SMPL-X)

The SMPL-X parametric body model is required for Steps 3–4 and the HOI solver. Download from the [SMPL-X website](https://smpl-x.is.tue.mpg.de/) and configure the path in `preprocessing/config.sh`:

```bash
export SMPLX_MODEL=/path/to/smplx/models
```

---

## 6. Verification

After setup, verify the pipeline runs end-to-end:

```bash
bash preprocessing/run_pipeline.sh /path/to/test_session
```

Individual steps can also be tested in isolation — see `preprocessing/README.md`.
