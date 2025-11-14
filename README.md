# Gorilla Face Identification POC

## Overview
- Proof-of-concept pipeline for gorilla face identification with closed-set baseline, open-set rejection, and future enrollment workflows.
- Designed for local experimentation on a Windows workstation with NVIDIA RTX 5080; code targets PyTorch 2.x.
- Builds on cropped gorilla face datasets (Kaggle `smiles28/gorillas`, BristolGorillas2020) before adding detection/tracking.

## Repository Layout
- `docs/` – context (`context-raw.md`), project plan, plus `notes/` and `research/` placeholders for experiments and literature.
- `src/config/` – YAML-driven config schema (see `configs/train_closed_set.yaml` for defaults).
- `src/datasets/` – Gorilla face dataset wrapper, transform builder, and dataloader registry.
- `src/models/` – ResNet18 embedding backbone scaffold.
- `src/training/` – Closed-set training loop skeleton (`train.py`) and evaluation placeholder.
- `src/inference/` – k-NN gallery helper plus `predict.py` CLI scaffold.
- `data/` – (gitignored) location for raw/processed datasets, gallery embeddings, and unknown pools.
- `configs/` – Run configurations; currently only `train_closed_set.yaml`.

## Current Status (Phase 1 Skeleton)
- ResNet18 embedding wrapper implemented with L2-normalized outputs; classifier head defined inside training loop.
- Gorilla face dataset loader expects `data/processed/gorilla_faces/<split>/<gorilla_id>/*.jpg|png` layout.
- Training loop and validation pass are stubbed with TODOs; checkpoints saved to `artifacts/closed_set_skeleton.pt`.
- k-NN inference pipeline loads gallery index (`artifacts/gallery_index.pkl`) built outside this scaffold; prediction CLI prints nearest IDs.
- Config loader now produces a `TrainingConfig` dataclass for reuse across training/inference scripts.

## Getting Started
1. **Environment Setup**
   - **Python Version**: Python 3.11 (recommended) or 3.10+
   - **Create Virtual Environment** (strongly recommended):
     ```bash
     # Create venv
     python -m venv venv
     
     # Activate venv
     # Windows PowerShell:
     .\venv\Scripts\Activate.ps1
     # Windows CMD:
     .\venv\Scripts\activate.bat
     # macOS / Linux:
     source venv/bin/activate
     ```
   - **Install PyTorch**:
     - **Windows with NVIDIA GPU (e.g., RTX 5080)**:
       ```bash
       # Install PyTorch with CUDA 12.x support
       pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
       ```
     - **macOS (Apple Silicon M1/M2/M3+)**:
       ```bash
       # Install PyTorch with MPS (Metal Performance Shaders) support
       pip install torch torchvision torchaudio
       ```
     - **macOS (Intel) or CPU-only**:
       ```bash
       pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
       ```
   - **Install Other Dependencies**:
     ```bash
     pip install -r requirements.txt
     ```
   - **Verify GPU/Acceleration Availability**:
     ```bash
     # Check CUDA (Windows/Linux) or MPS (macOS Apple Silicon)
     python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'MPS available: {torch.backends.mps.is_available()}'); print(f'Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"MPS\" if torch.backends.mps.is_available() else \"CPU\"}')"
     ```
2. **Datasets**
   - Download gorilla face crops (e.g., Kaggle `smiles28/gorillas`).
   - Arrange into `data/processed/gorilla_faces/{train,val,test}/{gorilla_id}/image.jpg`.
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
