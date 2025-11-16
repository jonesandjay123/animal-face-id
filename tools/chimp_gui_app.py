#!/usr/bin/env python3
"""Gradio MVP for chimpanzee face identification and enrollment."""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any, Dict, List

import gradio as gr
import numpy as np

# Ensure project root on sys.path
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.inference.inference_core import infer_single_image, load_model_from_config
from src.inference.index_store import SimpleIndex, add_individual


DEFAULT_CONFIG = "configs/train_chimp_min10_resnet50_arc_full.yaml"
DEFAULT_CKPT = "artifacts/chimp-min10-resnet50-arcface-full_best.pt"
DEFAULT_INDEX_CANDIDATES = [
    "artifacts/index/chimp_min10_auto",
    "artifacts/index/chimp_index",
]


class AppState:
    def __init__(self) -> None:
        self.model_bundle: Dict[str, Any] | None = None
        self.index: SimpleIndex | None = None


STATE = AppState()


def _find_existing_index() -> str | None:
    """Return a prefix for an existing index if found, checking common candidates."""
    for cand in DEFAULT_INDEX_CANDIDATES:
        emb_path = Path(cand + "_embeddings.npy")
        meta_path = Path(cand + "_meta.json")
        if emb_path.exists() and meta_path.exists():
            return cand
    # Fallback: first matching chimp_min10_auto*.npy
    auto = list(Path("artifacts/index").glob("chimp_*_auto*_embeddings.npy"))
    if auto:
        prefix = str(auto[0]).replace("_embeddings.npy", "")
        meta = Path(prefix + "_meta.json")
        if meta.exists():
            return prefix
    return None


def _load_index(prefix: str | None) -> SimpleIndex | None:
    if prefix is None:
        return None
    emb_path = Path(prefix + "_embeddings.npy")
    meta_path = Path(prefix + "_meta.json")
    if emb_path.exists() and meta_path.exists():
        try:
            return SimpleIndex.load(prefix)
        except Exception:
            return None
    return None


def init_model(config: str = DEFAULT_CONFIG, ckpt: str = DEFAULT_CKPT, device: str = "cuda") -> str:
    STATE.model_bundle = load_model_from_config(config, ckpt, device=device)
    prefix = _find_existing_index()
    STATE.index = _load_index(prefix)
    msg = f"Model loaded from {ckpt} on {STATE.model_bundle['device']}. "
    if STATE.index and STATE.index.size > 0:
        unique_ids = len(set(STATE.index.labels))
        msg += f"Index loaded ({STATE.index.size} embeddings, {unique_ids} IDs) from prefix: {prefix}."
    else:
        msg += "No index found; build or enroll to add a gallery."
    return msg


def identify(image, topk: int):
    """Identify a single uploaded face.

    Returns:
    - status string
    - model_topk: list of dicts with rank, id, prob (model classifier top-k on logits/probs)
    - gallery_topk: list of dicts with rank, id, similarity (gallery index cosine results)
    """
    if STATE.model_bundle is None:
        return "Model not loaded yet.", [], []
    if image is None:
        return "Please upload an image.", [], []

    tmp_path = Path(".__tmp_upload.png")
    image.save(tmp_path)
    try:
        result = infer_single_image(STATE.model_bundle, str(tmp_path), topk=topk)
    finally:
        if tmp_path.exists():
            tmp_path.unlink()

    # Model classifier top-k (classes/IDs + probabilities from logits)
    model_topk = [
        [rank, label, float(score)]
        for rank, (label, score) in enumerate(zip(result["topk_labels"], result["topk_scores"]), start=1)
    ]

    # Gallery index top-k (cosine similarity)
    if STATE.index is None or STATE.index.size == 0:
        gallery_topk = [["-", "No index loaded", "-"]]
        status = "No index loaded; showing classifier top-k only."
    else:
        matches = STATE.index.query(result["embedding"], topk=topk)
        gallery_topk = [[i + 1, m["label"], float(m["score"])] for i, m in enumerate(matches)]
        status = "Done"

    return status, model_topk, gallery_topk


def enroll(name: str, files: List[Any], aggregate: bool):
    if STATE.model_bundle is None:
        return "Model not loaded yet.", 0
    if not name:
        return "Please provide a name/ID.", 0
    if not files:
        return "Please upload at least one image.", 0

    if STATE.index is None:
        STATE.index = SimpleIndex()

    paths = []
    for f in files:
        if isinstance(f, str):
            paths.append(f)
        else:
            # gradio returns tempfile objects with .name
            paths.append(getattr(f, "name", None))
    paths = [p for p in paths if p]

    add_individual(STATE.model_bundle, STATE.index, name, paths, aggregate=aggregate)
    STATE.index.save(DEFAULT_INDEX_PREFIX)
    return f"Added {len(paths)} images under '{name}'. Index size: {STATE.index.size}", len(paths)


def build_interface():
    with gr.Blocks(title="Chimpanzee Face ID") as demo:
        status = gr.State("")

        gr.Markdown("# Chimpanzee Face Identification (MVP)\nUpload cropped face images to identify or enroll.")
        with gr.Row():
            cfg = gr.Textbox(label="Config path", value=DEFAULT_CONFIG)
            ckpt = gr.Textbox(label="Checkpoint path", value=DEFAULT_CKPT)
            device = gr.Radio(choices=["cuda", "cpu"], value="cuda", label="Device")
            init_btn = gr.Button("Load model")
        init_out = gr.Textbox(label="Init status")
        init_btn.click(init_model, inputs=[cfg, ckpt, device], outputs=init_out)

        with gr.Tab("Identify"):
            img = gr.Image(type="pil", label="Cropped chimp face")
            topk_in = gr.Slider(1, 5, value=5, step=1, label="Top-k")
            identify_btn = gr.Button("Identify")
            msg = gr.Textbox(label="Status")
            clf_table = gr.Dataframe(headers=["rank", "id", "prob"], label="Model top-k", interactive=False, datatype=["number", "str", "number"])
            gallery_table = gr.Dataframe(headers=["rank", "id", "similarity"], label="Gallery (index) top-k", interactive=False, datatype=["number", "str", "number"])
            identify_btn.click(identify, inputs=[img, topk_in], outputs=[msg, clf_table, gallery_table])

        with gr.Tab("Enroll"):
            name_in = gr.Textbox(label="New individual name/ID")
            files_in = gr.File(file_count="multiple", type="filepath", label="Images (cropped faces)")
            aggregate = gr.Checkbox(value=False, label="Average embeddings into one vector")
            enroll_btn = gr.Button("Add to index")
            enroll_msg = gr.Textbox(label="Enroll status")
            enroll_count = gr.Number(label="Images added", precision=0)
            enroll_btn.click(enroll, inputs=[name_in, files_in, aggregate], outputs=[enroll_msg, enroll_count])

    return demo


def main():
    # Preload model/index on startup
    init_model()
    demo = build_interface()
    demo.launch(server_name=os.environ.get("GRADIO_HOST", "0.0.0.0"), server_port=int(os.environ.get("GRADIO_PORT", 7860)))


if __name__ == "__main__":
    main()
