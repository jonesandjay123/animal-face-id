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
1. **Environment**
   - Python 3.11 with PyTorch 2.x + CUDA 12 toolchain.
   - Install dependencies (example): `pip install torch torchvision scikit-learn pillow pyyaml joblib`.
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

## Recommended Next Steps
1. Flesh out `train_closed_set` forward/backward pass, metric logging, and checkpointing.
2. Add an embedding export script to populate `artifacts/gallery_index.pkl`.
3. Extend evaluation to compute Top-1/Top-5 accuracy and confusion matrices.
4. Once closed-set baseline stabilizes, start Phase 2 work (open-set threshold calibration, unknown buffering).
