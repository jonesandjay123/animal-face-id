# Animal Face Identification

This repository contains a proof-of-concept pipeline for animal face identification using PyTorch. It is designed for closed-set identification (recognizing known individuals) and is built with components to support future open-set and enrollment workflows.

The current implementation focuses on identifying individual chimpanzees using the [Chimpanzee Faces](https://github.com/cvjena/chimpanzee_faces) dataset.

## Features
- **End-to-End Workflow**: Covers the full pipeline from data preparation to training, evaluation, and inference.
- **Configuration-Driven**: All experiments are controlled via simple YAML configuration files.
- **High-Performance Models**: Includes configurations for ResNet backbones with ArcFace loss, a standard for face recognition tasks.
- **Reproducibility**: Provides scripts and fixed seeds to ensure that data splits and training runs are reproducible.

---

## Project Status: High-Performance Model Trained

**As of November 2025, a high-performance chimpanzee recognition model has been successfully trained.**

- **Model:** `ResNet50` backbone with an `ArcFace` head.
- **Training:** Full run on the `min10` dataset (200 epochs, config `configs/train_chimp_min10_resnet50_arc_full.yaml`).
- **Result:** Best checkpoint: `artifacts/chimp-min10-resnet50-arcface-full_best.pt`. Ready for evaluation and inference.

---

## Conceptual Overview

This project involves several key deep learning concepts. For a detailed explanation of the project's architecture and answers to common questions, please read our new guide:

**➡️ [Conceptual Overview & FAQ](./docs/CONCEPTS.md)**

This guide answers questions such as:
- How does the model "remember" new faces without full retraining?
- What is the difference between the GPU's role in training vs. the CPU's role in inference?
- Why is this model a "chimpanzee expert" and what are its limitations?

---

## Documentation

This project is organized into a series of detailed guides. Start with setting up your environment and follow the steps in order.

| # | Guide | Description |
|---|---|---|
| 1 | **[Environment Setup](./docs/SETUP.md)** | How to configure your Python environment on Windows, WSL, or macOS. |
| 2 | **[Data Preparation](./docs/DATA_PREPARATION.md)** | How to download, validate, and prepare the dataset for training. |
| 3 | **[Model Training](./docs/TRAINING.md)** | How to run the training script and understand the outputs. |
| 4 | **[Evaluation and Inference](./docs/EVALUATION_AND_INFERENCE.md)** | How to evaluate your trained model and predict new images. |

---

## How to Run: The Full Workflow

Here is the complete sequence of commands to go from a fresh clone to making a prediction.

### 1. Setup and Data Prep
*Ensure you have completed the steps in the [Environment Setup](./docs/SETUP.md) and [Data Preparation](./docs/DATA_PREPARATION.md) guides first.*

```bash
# Activate your virtual environment (e.g., on Linux/WSL/macOS)
source .venv/bin/activate

# 1. Validate that your dataset is structured correctly
python validate_dataset.py

# 2. Create the train/validation/test split file (only needs to be run once)
python scripts/prepare_chimpanzee_splits.py
```

### 2. Train the Model
*For details, see the [Model Training](./docs/TRAINING.md) guide.*

```bash
# Run a full training using the high-performance configuration
python -m src.training.train --config configs/train_chimp_min10_resnet50_arc_full.yaml
```

### 3. Build Gallery and Predict
*For details, see the [Evaluation and Inference](./docs/EVALUATION_AND_INFERENCE.md) guide.*

```bash
# 1. Build the k-NN gallery index from your trained model
python -m src.inference.build_gallery --config configs/train_chimp_min10_resnet50_arc_full.yaml --device cuda

# 2. Predict the ID of a new image
python -m src.inference.predict --image /path/to/your/chimp_face.png --config configs/train_chimp_min10_resnet50_arc_full.yaml --device cpu
```

### 4. Final Evaluation on Test Split
```bash
python tools/run_final_eval.py \
  --config configs/train_chimp_min10_resnet50_arc_full.yaml \
  --ckpt artifacts/chimp-min10-resnet50-arcface-full_best.pt \
  --device cuda
```
Outputs go to `artifacts/final_eval/` and `FINAL_EVAL_REPORT.md`.

---

## Repository Structure

```
.
├── artifacts/              # Output folder for models (.pt) and logs (.csv)
├── configs/                # YAML configuration files for training runs
├── data/
│   ├── chimpanzee_faces/
│   │   ├── annotations/    # Annotation files and generated splits.json
│   │   └── raw/            # Location for the downloaded image dataset (ignored by Git)
├── docs/                   # Detailed documentation guides
├── scripts/                # Helper scripts (e.g., for preparing data splits)
├── src/                    # Main source code
│   ├── datasets/           # Dataloaders
│   ├── inference/          # Inference scripts (prediction, gallery building)
│   ├── models/             # Model definitions (backbones, heads, losses)
│   └── training/           # Training and evaluation logic
├── tools/                  # Standalone tools (e.g., final evaluation script)
└── README.md               # This file
```
