# Ant Detection Training Guide (Co-DINO)

This repository contains the setup for training a **Co-DINO (Swin-L)** model on a custom "Ant" dataset using MMDetection v3.0.

## 🚀 Quick Start (Training Environment)

If you are starting from a fresh environment with **Python 3.12** and **CUDA 12.8 / PyTorch 2.9+**, follow these steps exactly.

### 1. Build Dependencies
First, install the core build tools and dependencies.

```bash
# Upgrade build tools for Python 3.12 compatibility
pip install -U setuptools setuptools-scm

# Install project dependencies
pip install -r requirements/albu.txt
pip install fairscale

# Install OpenMMLab manager
pip install -U openmim
```

### 2. Build MMCV from Source
Since your environment is cutting-edge (CUDA 12.8), you must compile MMCV with CUDA operations manually.

```bash
cd /tmp
git clone https://github.com/open-mmlab/mmcv.git
cd mmcv
git checkout v2.1.0  # Or checkout 'main' for latest compatibility

# IMPORTANT: Pin setuptools to avoid Python 3.12 pkgutil error
pip install "setuptools<81"

MMCV_WITH_OPS=1 FORCE_CUDA=1 pip install . --no-build-isolation
cd /workspace/mmdetection
```

### 3. Initialize MMDetection
Install the toolbox and link the configuration folders.

```bash
# Install MMDetection in editable mode
pip install -e . --no-build-isolation

# Initialize internal package links (fixes FileNotFoundError)
python mim_init.py
```

### 4. Prepare Dataset
Convert your YOLO-formatted dataset to the COCO format required by Co-DINO.

```bash
# Convert YOLO to COCO
python yolo_to_coco.py --data_dir ./dataset

# IMPORTANT: Fix Windows backslashes if you converted on Windows but train on Linux
python fix_json_paths.py
```

### 5. Start Training
Use the custom configuration tailored for the single-class "Ant" task.

```bash
PYTHONPATH=. python tools/train.py projects/CO-DETR/configs/codino/co_dino_5scale_swin_l_ant.py
```

## 📂 Key Files
- `projects/CO-DETR/configs/codino/co_dino_5scale_swin_l_ant.py`: Your custom training configuration.
- `yolo_to_coco.py`: Dataset conversion script.
- `mim_init.py`: Internal link initializer (vital for fixing path errors).

## 🛠️ Common Fixes
- **AttributeError: 'pkgutil' has no attribute 'ImpImporter'**: Run `pip install "setuptools<81"` before building MMCV from source.
- **FileNotFoundError: .../train\\images\\...**: This happens if the COCO JSON contains Windows-style backslashes. Run `python fix_json_paths.py` on the training machine to fix them.
- **ModuleNotFoundError: No module named 'mmcv._ext'**: This happens if MMCV was installed without CUDA. Re-run Step 2.
- **FileNotFoundError: .../model-index.yml**: This happens if the internal links are broken. Re-run Step 3 (`python mim_init.py`).
- **ModuleNotFoundError: No module named 'projects'**: Always ensure you are in the root directory and use `PYTHONPATH=.`.

---

## 🧠 Alternative: ViT Backbone (DINOv3)

If you want to use a ViT (DINOv3 Base) backbone instead of Swin-L:

### Prerequisites
```bash
pip install einops xformers  # Optional but recommended
```

### Download DINOv3 Checkpoint
Download your DINOv3-B checkpoint (from HuggingFace) and place it in `models/dinov3_vitb14_pretrain.pth`.

### Start Training
```bash
PYTHONPATH=. python tools/train.py projects/CO-DETR/configs/codino/co_dino_5scale_vit_b_ant.py
```

> [!WARNING]
> If you encounter OOM errors, reduce `batch_size` to 1.
