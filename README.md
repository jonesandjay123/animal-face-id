# Animal Face Identification POC

## Overview
- Proof-of-concept pipeline for animal face identification with closed-set baseline, open-set rejection, and future enrollment workflows.
- Designed for local experimentation on a Windows workstation with NVIDIA RTX 5080; code targets PyTorch 2.x.
- Builds on cropped animal face datasets before adding detection/tracking.

## Repository Layout
- `docs/` – context (`context-raw.md`), project plan, plus `notes/` and `research/` placeholders for experiments and literature.
- `src/config/` – YAML-driven config schema (see `configs/train_closed_set.yaml` for defaults).
- `src/datasets/` – Animal face dataset wrapper, transform builder, and dataloader registry.
- `src/models/` – ResNet18 embedding backbone scaffold.
- `src/training/` – Closed-set training loop skeleton (`train.py`) and evaluation placeholder.
- `src/inference/` – k-NN gallery helper plus `predict.py` CLI scaffold.
- `data/` – (gitignored) location for raw/processed datasets, gallery embeddings, and unknown pools.
- `configs/` – Run configurations; currently only `train_closed_set.yaml`.

## Current Status (Phase 1 Skeleton)
- ResNet18 embedding wrapper implemented with L2-normalized outputs; classifier head defined inside training loop.
- Animal face dataset loader expects `data/processed/animal_faces/<split>/<individual_id>/*.jpg|png` layout.
- Training loop and validation pass are stubbed with TODOs; checkpoints saved to `artifacts/closed_set_skeleton.pt`.
- k-NN inference pipeline loads gallery index (`artifacts/gallery_index.pkl`) built outside this scaffold; prediction CLI prints nearest IDs.
- Config loader now produces a `TrainingConfig` dataclass for reuse across training/inference scripts.

## Getting Started

### Platform-Specific Setup

<details>
<summary><b>🐧 Windows with WSL2 + Ubuntu (Recommended for NVIDIA GPU Training)</b></summary>

If you're on Windows with an NVIDIA GPU (e.g., RTX 5080), using **WSL2 with Ubuntu** provides the best PyTorch + CUDA performance and compatibility.

#### Prerequisites
1. **Install WSL2** (if not already installed):
   ```powershell
   # In PowerShell (Administrator)
   wsl --install
   # Restart your computer after installation
   ```

2. **Verify NVIDIA Driver** (in Windows PowerShell):
   ```powershell
   nvidia-smi
   ```
   You should see your GPU listed (e.g., RTX 5080).

3. **Verify GPU Access in WSL2** (in Ubuntu terminal):
   ```bash
   nvidia-smi
   ```
   If you see the same GPU table, WSL2 can access your GPU ✅

#### Environment Setup in WSL2 Ubuntu

1. **Navigate to Project Directory**:
   ```bash
   # If your repo is at C:\Users\jones\Downloads\animal-face-id
   cd /mnt/c/Users/jones/Downloads/animal-face-id
   ```

2. **Install Python and venv** (if needed):
   ```bash
   sudo apt update
   sudo apt install -y python3 python3-venv python3-pip
   python3 --version  # Should be 3.10+
   ```

3. **Create Virtual Environment**:
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   ```
   You should see `(.venv)` prefix in your prompt.

4. **Upgrade pip**:
   ```bash
   pip install --upgrade pip
   ```

5. **Install PyTorch with CUDA Support**:
   ```bash
   # For CUDA 12.1 (check https://pytorch.org/get-started/locally/ for latest)
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
   ```

6. **Install Project Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

7. **Verify GPU Detection**:
   ```bash
   python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}'); print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\"}')"
   ```
   Expected output:
   ```
   PyTorch: 2.x.x
   CUDA available: True
   GPU: NVIDIA GeForce RTX 5080
   ```

#### Daily Workflow
```bash
# 1. Open WSL2 Ubuntu terminal (in Cursor or Windows Terminal)
# 2. Navigate to project
cd /mnt/c/Users/jones/Downloads/animal-face-id

# 3. Activate venv
source .venv/bin/activate

# 4. Run training
python -m src.training.train --config configs/train_closed_set.yaml
```

</details>

<details>
<summary><b>🪟 Windows (Native, without WSL)</b></summary>

1. **Python Version**: Python 3.11 (recommended) or 3.10+

2. **Create Virtual Environment**:
   ```powershell
   # In PowerShell or CMD
   python -m venv venv
   
   # Activate (PowerShell)
   .\venv\Scripts\Activate.ps1
   # Or CMD
   .\venv\Scripts\activate.bat
   ```

3. **Install PyTorch with CUDA**:
   ```bash
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
   ```

4. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

5. **Verify GPU**:
   ```bash
   python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
   ```

</details>

<details>
<summary><b>🍎 macOS (Apple Silicon)</b></summary>

1. **Create Virtual Environment**:
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```

2. **Install PyTorch with MPS Support**:
   ```bash
   pip install torch torchvision torchaudio
   ```

3. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Verify MPS**:
   ```bash
   python -c "import torch; print(f'MPS available: {torch.backends.mps.is_available()}')"
   ```

</details>

<details>
<summary><b>🍎 macOS (Intel) or Linux CPU-only</b></summary>

1. **Create Virtual Environment**:
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```

2. **Install PyTorch (CPU)**:
   ```bash
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
   ```

3. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

</details>
2. **Datasets**
   - Download animal face crop datasets.
   - Arrange into `data/processed/animal_faces/{train,val,test}/{individual_id}/image.jpg`.
   - Update `data.num_classes` in `configs/train_closed_set.yaml` to match the individual count.
3. **Configuration**
   - Copy `configs/train_closed_set.yaml` and adjust paths, batch sizes, embedding dimensions, etc., as needed.

## Usage
1. **Training Skeleton**
   ```bash
   python -m src.training.train --config configs/train_closed_set.yaml
   ```
   - Script currently iterates over loaders without real forward/backward logic. Fill in TODO sections before running actual training.
2. **Building the Gallery Index**
   - After training, write a small helper (future task) that exports embeddings for gallery splits and saves `EmbeddingIndex` via `src/inference/knn.py`.
   - Expected output: `artifacts/gallery_index.pkl`.
3. **Prediction CLI**
   ```bash
    python -m src.inference.predict --image path/to/cropped_face.jpg --config configs/train_closed_set.yaml
   ```
   - Requires existing gallery index; raises if missing.

## GroVE Integration Outlook
- **What it is**: cvjena/GroVE converts deterministic vision-language embeddings into probabilistic embeddings (mean + covariance), providing uncertainty estimates for each sample.
- **Why it matters**: uncertainty-aware distances (e.g., KL divergence) dramatically improve open-set rejection and prevent low-quality crops from spawning new IDs during enrollment.
- **When to use**:
  1. *Phase 1–2*: optional; baseline can stay deterministic.
  2. *Phase 3*: highly valuable—filter unknown pools by uncertainty before clustering.
  3. *Future wildlife deployments*: near-essential for noisy side-profile footage.
- **Integration plan**: keep the existing ResNet/ViT backbone, add a GroVE head to output μ/Σ, and bolt on KL-divergence scoring + uncertainty gating without discarding current weights.

## Recommended Next Steps
1. Flesh out `train_closed_set` forward/backward pass, metric logging, and checkpointing.
2. Add an embedding export script to populate `artifacts/gallery_index.pkl`.
3. Extend evaluation to compute Top-1/Top-5 accuracy and confusion matrices.
4. Once closed-set baseline stabilizes, start Phase 2 work (open-set threshold calibration, unknown buffering).
5. Prototype GroVE probabilistic head as a drop-in module to compare deterministic vs probabilistic open-set metrics.
