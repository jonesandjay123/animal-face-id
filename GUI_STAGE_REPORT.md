# GUI & Inference MVP (Step B)

## Files added
- `src/inference/inference_core.py` — load model+config+head, single-image inference API (returns embedding + top-k logits).
- `src/inference/index_store.py` — lightweight cosine index (save/load/query/add/build from folder).
- `tools/chimp_index_cli.py` — CLI to build/load/update index (`build`, `add`, `stats`).
- `tools/chimp_gui_app.py` — Gradio MVP (Identify / Enroll tabs).
- `requirements.txt` — added `gradio`.

## How to run (from repo root)
1) (Optional) Build or preload an index:
```bash
python tools/chimp_index_cli.py build \
  --folder /path/to/gallery_root \
  --config configs/train_chimp_min10_resnet50_arc_full.yaml \
  --ckpt artifacts/chimp-min10-resnet50-arcface-full_best.pt \
  --prefix artifacts/index/chimp_index
```
2) Launch GUI:
```bash
python tools/chimp_gui_app.py
```

## GUI usage
- **Identify tab:** upload a cropped face (png/jpg), choose top-k. Shows model top-k logits and gallery top-k (if index loaded). If no index, message prompts to enroll/build.
- **Enroll tab:** enter new individual name + upload one or more cropped faces. Updates index and saves to `artifacts/index/chimp_index_*`.

## Notes
- Assumes images are pre-cropped chimp faces (no detector in GUI).
- Default config/ckpt: `configs/train_chimp_min10_resnet50_arc_full.yaml` + `artifacts/chimp-min10-resnet50-arcface-full_best.pt`.
- Index defaults to `artifacts/index/chimp_index`; swap via CLI or by editing defaults in the GUI script.

## Common commands (with defaults)
- Build index (png):
```bash
python tools/chimp_index_cli.py \
  --prefix artifacts/index/chimp_index \
  --config configs/train_chimp_min10_resnet50_arc_full.yaml \
  --ckpt artifacts/chimp-min10-resnet50-arcface-full_best.pt \
  build \
  --folder /path/to/gallery_root \
  --pattern "**/*.png"
```
- Build index (jpg):
```bash
python tools/chimp_index_cli.py --prefix artifacts/index/chimp_index --config configs/train_chimp_min10_resnet50_arc_full.yaml --ckpt artifacts/chimp-min10-resnet50-arcface-full_best.pt build --folder /path/to/gallery_root --pattern "**/*.jpg"
```
- Add a new individual to an existing index:
```bash
python tools/chimp_index_cli.py \
  --prefix artifacts/index/chimp_index \
  --config configs/train_chimp_min10_resnet50_arc_full.yaml \
  --ckpt artifacts/chimp-min10-resnet50-arcface-full_best.pt \
  add \
  --name NewChimp \
  --images /path/to/img1.png /path/to/img2.png
```
- Show index stats:
```bash
python tools/chimp_index_cli.py --prefix artifacts/index/chimp_index stats
```
- Launch GUI (uses defaults above):
```bash
python tools/chimp_gui_app.py
```
