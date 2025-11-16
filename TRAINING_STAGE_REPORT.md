# Stage Report – Chimpanzee Faces (min10) Training Pipeline

## What’s new this stage
- **Stratified split generator**: `scripts/prepare_chimpanzee_splits.py` creates `splits_min10.json` (70/15/15, each ID appears in val/test). Validates raw paths before saving.
- **Dataset loader**: `ChimpanzeeFacesDataset` (annotations + split manifest, no file copies) registered as `dataset_name: chimpanzee_faces`.
- **Training loop upgrade**: `src/training/train.py` now has full forward/backward, AMP, grad clip, AdamW, cosine/step schedulers, ArcFace or linear head, metrics (top-1/top-5), CSV logging, best/last checkpoints.
- **Loss/head**: ArcFace head + builder in `src/models/losses.py`; ResNet18/50 backbones supported.
- **Config**: `configs/train_chimp_min10.yaml` wired for 87 classes, ArcFace head, cosine scheduler, AMP, checkpoint/log paths under `artifacts/`.
- **Inference utilities**:
  - `src/inference/build_gallery.py` exports embeddings for a split and writes `artifacts/gallery_index.pkl`.
  - `src/inference/predict.py` now loads trained checkpoints before querying the gallery.

## How to run (manual steps)
1) **Generate splits** (once per seed):
```bash
python scripts/prepare_chimpanzee_splits.py
```
Expected: `data/chimpanzee_faces/annotations/splits_min10.json` plus summary line counts.

2) **Train (short smoke or full run)**:
```bash
python -m src.training.train --config configs/train_chimp_min10.yaml
```
Outputs (under `artifacts/`):
- `chimp-min10-resnet18_best.pt`, `chimp-min10-resnet18_last.pt`
- `chimp-min10-resnet18_metrics.csv`

3) **Build gallery index** (after training):
```bash
python -m src.inference.build_gallery --config configs/train_chimp_min10.yaml --device cuda
```
Outputs: `artifacts/{split}_embeddings.npz`, `artifacts/gallery_index.pkl`

4) **Single-image predict** (requires gallery + checkpoint):
```bash
python -m src.inference.predict \
  --image path/to/cropped_face.png \
  --config configs/train_chimp_min10.yaml \
  --checkpoint artifacts/chimp-min10-resnet18_best.pt \
  --device cpu   # or cuda
```

## Notes / known gaps
- No training run executed yet—checkpoints will appear only after you run Phase 3/4.
- Dataloader smoke script is not included; if desired, add a tiny loader test to print one batch.
- Gallery builder uses the chosen split (default train); adjust `--split` if you want a dedicated gallery.
- If you prefer a quick baseline, switch `model.head` to `linear` in the config; ArcFace is the default.
- Increase `trainer.max_epochs` (e.g., 50–100) for the real run; current default is 10.

## New config: RTX 5080 + ResNet50 + ArcFace (trial)
- File: `configs/train_chimp_min10_resnet50_arc.yaml`
- Rationale:
  - Backbone: `resnet50` to improve capacity over ResNet18.
  - Head/loss: ArcFace (margin=0.5, scale=30) to learn stronger embeddings for gallery/enrollment.
  - Batch: 128 with AMP sized for ~16GB VRAM; workers 8 to keep the GPU fed.
  - LR: 5e-4 AdamW with cosine decay, weight_decay 0.01; 30 epochs as a trial before longer runs.
  - Checkpoint path: `artifacts/chimp-min10-resnet50-arcface_best.pt`.

## Full run config (extended training)
- File: `configs/train_chimp_min10_resnet50_arc_full.yaml`
- Trial summary: 30-epoch ResNet50+ArcFace reached ~0.73 val top-1, stable training (100% GPU, no OOM).
- Key hyperparameters:
  - Backbone/head/loss: ResNet50 + ArcFace (margin 0.5, scale 30), cross-entropy on logits.
  - Batch/loader: batch_size 128, num_workers 8, AMP enabled.
  - Optimizer/scheduler: AdamW (lr 4e-4, weight_decay 0.01) with cosine schedule; grad_clip 1.0.
  - Epochs: 120 for a full run to allow continued improvement beyond the 30-epoch trial.
  - Outputs: checkpoints to `artifacts/chimp-min10-resnet50-arcface-full_best.pt` and `_full_last.pt`; metrics to `artifacts/chimp-min10-resnet50-arcface_full_metrics.csv`.
- Why longer (120): the 30-epoch curve was still improving without overfitting signs, so extending to 120 epochs should let the cosine schedule anneal further and squeeze extra accuracy before saturation.
- Command to launch (from repo root):
```bash
python -m src.training.train --config configs/train_chimp_min10_resnet50_arc_full.yaml
```

## Step A: Final Evaluation (test split, trained model)
- Script: `tools/run_final_eval.py`
- Purpose: run the trained checkpoint on the **test split**, produce metrics/plots/CSV/Markdown in `artifacts/final_eval/` and update `FINAL_EVAL_REPORT.md`.
- Example:
```bash
python tools/run_final_eval.py \
  --config configs/train_chimp_min10_resnet50_arc_full.yaml \
  --ckpt artifacts/chimp-min10-resnet50-arcface_full_best.pt \
  --device cuda
```
