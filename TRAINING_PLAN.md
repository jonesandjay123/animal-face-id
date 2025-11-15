# Chimpanzee Faces Min10 Training Plan

## Project Snapshot (from current repo)
- Dataset: `data/chimpanzee_faces/raw/datasets_cropped_chimpanzee_faces/` with 7,150 images in the min10 subset covering 87 individuals (annotations under `data/chimpanzee_faces/annotations/`; processed folder empty).
- Validation: `validate_dataset.py` plus `DATASET_AUDIT_REPORT.md` confirm paths, IDs, and counts are clean; `validation_results.json` exists.
- Dataloader: `AnimalFaceDataset` expects an image-per-class folder layout under `data/processed/animal_faces/{train,val,test}/{id}/*.jpg|png`; no loader yet for annotation TXT files.
- Training: `src/training/train.py` is a stub (no forward/backward logic, no metrics, fixed classifier head, single checkpoint path). `configs/train_closed_set.yaml` uses placeholder values (`num_classes: 8`, root `data/processed/animal_faces`).
- Inference: k-NN wrapper present; `predict.py` loads a backbone with random weights and assumes an existing gallery index.

## Goal
Train a closed-set face identification model on the min10 subset (87 IDs) with stratified splits and reproducible outputs (checkpoints, metrics, embeddings).

## Required Additions/Adjustments
- Data prep:
  - Add a split/prepare script (e.g., `scripts/prepare_chimp_splits.py`) that reads `annotations_merged_min10.txt`, performs stratified 70/15/15 (or 70/15/15 with min 1 per held-out split), and materializes `data/processed/chimpanzee_faces/{train,val,test}/{id}/*.png`.
  - Optionally add an annotation-driven dataset class to avoid materializing copies (reads paths/labels directly from TXT and applies transforms).
- Configs:
  - Create a dedicated config (e.g., `configs/train_chimp_min10.yaml`) pointing to the processed split root, `dataset_name: chimpanzee_faces` (or keep `animal_faces` but update root), and `num_classes: 87`.
  - Expose seed, scheduler choice, and log/artifact paths.
- Dataloaders:
  - Update `dataset_registry` to register the chimp dataset/split builder (either folder-based or annotation-based).
  - Ensure deterministic shuffling per epoch and class-to-index mapping saved for reproducibility.
- Training loop:
  - Implement forward/backward, loss logging, accuracy tracking, and checkpointing (best and last). Save class index mapping.
  - Add mixed precision and gradient clipping hooks.
- Evaluation and exports:
  - Add val/test evaluation computing Top-1/Top-5 accuracy, per-class accuracy, and confusion matrix export.
  - Add an embedding export script (e.g., `scripts/export_embeddings.py`) to write gallery embeddings (`*.npz`) and build `artifacts/gallery_index.pkl`.
- Inference cleanup:
  - Make `predict.py` load model weights from checkpoint and reuse the same transforms; provide CLI flags for checkpoint and index paths.

## Backbone Recommendation
- Baseline: `resnet18` with ImageNet weights (already scaffolded) and embedding_dim 256.
- If baseline lags: swap to `resnet50` or a lightweight ViT-S/16 after baseline stabilizes.
- Loss: start with cross-entropy on classifier head; later consider ArcFace/AM-Softmax or cosine classifier for better separation.

## Split Strategy (min10 subset)
- Stratified by individual ID: target 70/15/15 split (approx. train 5,000; val 1,070; test 1,070). For IDs with exactly 10 images, enforce at least 1 sample in val and 1 in test (ceil-based allocation).
- Single deterministic seed; write the split manifest to `data/chimpanzee_faces/annotations/splits_min10.json` for reproducibility.
- Keep CTai/CZoo mixed unless a site-balanced experiment is desired later; log site distribution per split.

## Evaluation and Expected Outputs
- Metrics: Top-1 and Top-5 accuracy on val/test; per-class accuracy; confusion matrix; overall precision/recall/F1 (macro); optional verification-style ROC using cosine distances on embeddings.
- Outputs:
  - Checkpoints: `artifacts/chimp_min10_resnet18_best.pt` and `..._last.pt`.
  - Logs: TensorBoard or CSV under `artifacts/logs/chimp_min10/`.
  - Embeddings: `{split}_embeddings.npz` with features and labels plus `artifacts/gallery_index.pkl`.
  - Reports: JSON/Markdown summary of final metrics (e.g., `artifacts/reports/chimp_min10_metrics.json`).

## Potential Issues / Cleanup Items (priority order)
1) Dataset-path mismatch: loader expects pre-split class folders, but current data is only in raw + annotations; need split materialization or annotation-based loader before training.
2) Config placeholders: `num_classes: 8` and `data.root` are incorrect for min10 (should be 87 and chimp path).
3) Training stub: no optimization/metrics; checkpoints not tied to accuracy; no logging or early stopping.
4) Inference uses random weights: `predict.py` does not load trained checkpoint; needs fix after training.
5) Utilities are stubs (`losses.py`, `evaluate.py`, `utils/io.py`, `utils/logging.py`); fill minimal functionality to support training/eval outputs.

## Phased Execution (each ~1–2 hours)
- Phase 1: Data prep — implement split script, generate train/val/test folders (or manifest), verify counts vs annotations, and rerun `validate_dataset.py` or a quick sanity check.
- Phase 2: Config and loader — add `train_chimp_min10.yaml`, wire dataset registry (folder or annotation-based), and smoke-test dataloaders for all splits.
- Phase 3: Training loop — implement forward/backward, metrics, checkpointing, and logging; run a short sanity training (few batches) to confirm loss decreases.
- Phase 4: Evaluation and exports — run full train for baseline, compute val/test metrics, export embeddings, and build `gallery_index.pkl`; update `predict.py` to consume checkpoint and index.
- Phase 5: Enhancements — experiment with improved loss/backbone, add class imbalance handling and richer eval (verification ROC), and document results.
