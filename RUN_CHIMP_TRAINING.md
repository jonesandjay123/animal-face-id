# Chimpanzee Faces – Training Runbook (Manual-First)

> 假設環境：WSL Ubuntu，專案路徑 `/mnt/c/Users/jones/Downloads/animal-face-id`
>
> 原則：你是司機，agent 只做導航；需要改碼時再呼叫 agent（Cursor / Codex），指令照抄即可。

## 用前備忘
- **venv 已建？** 沒有就建 `.venv` 並 `source .venv/bin/activate`
- **資料完整？** `data/chimpanzee_faces/raw/datasets_cropped_chimpanzee_faces/` 已在位；`annotations/*.txt` 存在。
- **GPU 可用？** 可跑一次簡短 `torch.cuda.is_available()` 檢查。

---

## Phase 0 – 環境 & 資料 sanity check（建議跑）
目的：確認 Python/套件/GPU/資料都 OK。

```bash
cd /mnt/c/Users/jones/Downloads/animal-face-id
python -m venv .venv        # 若已存在可跳過
source .venv/bin/activate
pip install -r requirements.txt
python -c "import torch; print('PyTorch:', torch.__version__, 'CUDA:', torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'No GPU')"
python validate_dataset.py   # 期待 7,187 / 7,150 / 87，無 missing files
```

---

## Phase 1 – 產生 min10 的 train/val/test split
目的：用 `annotations_merged_min10.txt` 產生 70/15/15 分割並寫入 JSON。

需要 agent 幫忙的檔案：`scripts/prepare_chimpanzee_splits.py`
- 功能：讀 `data/chimpanzee_faces/annotations/annotations_merged_min10.txt`，做 stratified 70/15/15，確保每 ID 在 val/test 至少 1 張。
- 輸出：`data/chimpanzee_faces/annotations/splits_min10.json`，格式 `{ "train": [...], "val": [...], "test": [...] }`，含相對路徑與 ID。

指令（等 agent 寫好後再跑）：
```bash
cd /mnt/c/Users/jones/Downloads/animal-face-id
source .venv/bin/activate
python scripts/prepare_chimpanzee_splits.py
ls data/chimpanzee_faces/annotations | grep splits
head data/chimpanzee_faces/annotations/splits_min10.json  # 可用 jq 看格式
```

---

## Phase 2 – Dataset Loader smoke test
目的：確認 dataloader 能讀 min10 split，batch 能跑。

需要 agent 幫忙的檔案：
- `src/datasets/chimpanzee_faces.py`：實作 `ChimpanzeeFacesDataset(split=...)` 讀 annotations + splits JSON（或 materialized folders）。
- dataset registry：能用 config key 叫到這個 dataset。
- 小型 debug script（建議 `src/datasets/debug_chimpanzee_loader.py`）印 dataset 長度與 1–2 個 batch shape。

指令（agent 完成後）：
```bash
python -m src.datasets.debug_chimpanzee_loader
```
預期：train/val/test 長度吻合統計，batch `images.shape` / `labels.shape` 正常，無檔案找不到錯誤。

---

## Phase 3 – 短程訓練 smoketest
目的：讓訓練 loop 跑完 1–2 個 epoch，產生 checkpoint。

需要 agent 幫忙：
- 新 config：`configs/train_chimp_min10.yaml`（或你喜歡的檔名）。關鍵：`num_classes: 87`、正確 dataset root、batch size（例如 64）、epoch 先設 2、log/checkpoint 路徑。
- 確認 `src/training/train.py` 補完 forward/backward、loss、metrics、checkpoint 保存。

指令：
```bash
cd /mnt/c/Users/jones/Downloads/animal-face-id
source .venv/bin/activate
python -m src.training.train --config configs/train_chimp_min10.yaml
```
預期：跑完 1–2 epoch，不崩潰，loss/accuracy 有輸出，`artifacts/` 下有 checkpoint。

---

## Phase 4 – 正式訓練
目的：跑完整訓練（如 50–100 epoch），得到 baseline。

在 config 裡調參（交給 agent 改 YAML）：epoch、lr/scheduler、batch size、log/checkpoint 路徑。

指令（與 Phase 3 相同）：
```bash
python -m src.training.train --config configs/train_chimp_min10.yaml
```
建議：用 `tmux` / `screen` 保持 session；記下最佳 checkpoint 路徑與最佳 val accuracy。

---

## Phase 5 – 建 gallery & 測 predict CLI
目的：用訓練好的 checkpoint 產生 `gallery_index` 並測單張推論。

需要 agent 幫忙：
- 實作 gallery 建立腳本（例如 `src/inference/build_gallery.py` 或在 `knn.py` 增 CLI）。
  - 載入 checkpoint
  - 對指定 split（通常 train/gallery）抽 embedding，存 `.npz` + `artifacts/gallery_index.pkl`
- 更新 `predict.py` 讓它載 checkpoint、共用 transforms。

指令（依實作微調）：
```bash
python -m src.inference.build_gallery --config configs/train_chimp_min10.yaml
python -m src.inference.predict --image path/to/cropped_face.png --config configs/train_chimp_min10.yaml
```
預期：build 不報「gallery missing」，predict 印出 top-1/ top-k ID＋距離。

---

## 一頁速查（流程記憶用）
```bash
cd /mnt/c/Users/jones/Downloads/animal-face-id
source .venv/bin/activate
python validate_dataset.py
python scripts/prepare_chimpanzee_splits.py              # 首次或換 split 才跑
python -m src.datasets.debug_chimpanzee_loader           # smoke test
python -m src.training.train --config configs/train_chimp_min10.yaml
python -m src.inference.build_gallery --config configs/train_chimp_min10.yaml
python -m src.inference.predict --image some_face.png --config configs/train_chimp_min10.yaml
```

---

## 小建議 / 注意事項
- split 工具完成後，版本鎖定：保留 `splits_min10.json` 供重跑；改 seed 時另存檔名。
- 跑長訓練前先 smoke（Phase 3），避免 GPU 時間浪費。
- 如要節省磁碟，可考慮 annotations-based Dataset（不複製檔案）；若已 materialize 分割，記得檢查 `.gitignore`。
- 訓練中途重啟，可從 checkpoint 繼續；確保 config 記錄 class_to_idx 以便推論一致。
