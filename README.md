# Animal Face Identification

This repository contains a proof-of-concept pipeline for animal face identification using PyTorch. It is designed for closed-set identification (recognizing known individuals) and is built with components to support future open-set and enrollment workflows.

The current implementation focuses on identifying individual chimpanzees using the [Chimpanzee Faces](https://github.com/cvjena/chimpanzee_faces) dataset.

## Features
- **End-to-End Workflow**: Covers the full pipeline from data preparation to training, evaluation, and inference.
- **Configuration-Driven**: All experiments are controlled via simple YAML configuration files.
- **High-Performance Models**: Includes configurations for ResNet backbones with ArcFace loss, a standard for face recognition tasks.
- **Reproducibility**: Provides scripts and fixed seeds to ensure that data splits and training runs are reproducible.

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