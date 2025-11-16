# Evaluation and Inference Guide

Once you have a trained model checkpoint (`.pt` file), you can use it for two main purposes:
1.  **Inference**: Predict the identity of a new, unseen chimpanzee face.
2.  **Evaluation**: Measure the model's performance on the test dataset to see how well it performs.

---

## The Workflow: Gallery, then Predict

The core of the identification process is a **gallery**. This is an indexed collection of "faceprints" (embeddings) from all the known individuals in our training set.

The workflow is always:
1.  **Build the Gallery**: Use a trained model to create a gallery file. This only needs to be done once per trained model.
2.  **Run Inference/Evaluation**: Use the trained model and the gallery to perform predictions.

---

## Step 1: Build the Gallery Index

The gallery index is a file (`gallery_index.pkl`) that acts as a searchable database of facial features from the individuals the model was trained on.

**Command:**
Run the `build_gallery.py` script, pointing it to the configuration file that corresponds to your trained model.

```bash
python -m src.inference.build_gallery --config <path_to_your_config.yaml> --device cuda
```

**Example:**
If you trained using `configs/train_chimp_min10_resnet50_arc_full.yaml`, you would run:

```bash
python -m src.inference.build_gallery --config configs/train_chimp_min10_resnet50_arc_full.yaml --device cuda
```

This script will:
1.  Load the best checkpoint specified in the config file (`inference.checkpoint`).
2.  Process all images in the **training set**.
3.  Save the computed embeddings and a k-NN index to files in the `artifacts/` directory.

**Outputs:**
-   `artifacts/gallery_index.pkl`: The crucial k-NN index for fast similarity search.
-   `artifacts/train_embeddings.npz`: The raw embeddings and labels (optional, for analysis).

---

## Step 2 (Option A): Predict a Single Image

With the gallery built, you can now identify a single image.

**Command:**
Use the `predict.py` script, providing the path to an image and the corresponding config file.

```bash
python -m src.inference.predict --image <path_to_image.png> --config <path_to_your_config.yaml> --device cpu
```

**Example:**
To predict the identity of a chimp in an image named `some_chimp_face.png`:

```bash
python -m src.inference.predict --image /path/to/some_chimp_face.png --config configs/train_chimp_min10_resnet50_arc_full.yaml --device cpu
```

**Output:**
The script will print a dictionary containing the prediction:

```json
{
  "image": "/path/to/some_chimp_face.png",
  "predicted_id": "Frodo",
  "distance": 0.2345,
  "neighbors": [
    ["Frodo", 0.2345],
    ["Gertrudia", 0.5678],
    ...
  ],
  ...
}
```
-   `predicted_id`: The model's best guess for the individual's name.
-   `distance`: A measure of similarity (lower is better). A very high distance might indicate an unknown individual.

---

## Step 2 (Option B): Run Full Evaluation on the Test Set

To get an objective measure of your model's performance, run the final evaluation script on the **test set**. This set contains images the model has never seen during training.

**Command:**
Use the `run_final_eval.py` script. You must provide the config file and the path to the specific checkpoint you want to evaluate.

```bash
python tools/run_final_eval.py --config <path_to_config.yaml> --ckpt <path_to_checkpoint.pt> --device cuda
```

**Example:**
To evaluate the best model from the full ResNet50 run:

```bash
python tools/run_final_eval.py \
  --config configs/train_chimp_min10_resnet50_arc_full.yaml \
  --ckpt artifacts/chimp-min10-resnet50-arcface-full_best.pt \
  --device cuda
```

**Outputs:**
This script generates a comprehensive set of reports in the `artifacts/final_eval/` directory and updates the main evaluation report.

-   **`FINAL_EVAL_REPORT.md`**: The main, human-readable summary report is automatically updated with the latest results.
-   **`artifacts/final_eval/`**:
    -   **Confusion Matrix (`..._confusion_matrix.png`)**: Shows which individuals are often confused with each other.
    -   **Per-Class Metrics (`..._per_class_metrics.csv`)**: Details the accuracy for each individual chimpanzee.
    -   **Embeddings Plot (`..._embeddings_tsne.png`)**: A 2D visualization of how the model groups different individuals in its feature space.
    -   **Metrics Summary (`..._metrics_summary.json`)**: A JSON file with all the key performance indicators (Top-1, Top-5, F1-score, etc.).

By analyzing these outputs, you can gain a deep understanding of your model's strengths and weaknesses.
