# Deepfake Detection via LoRA Fine-Tuned ViT

Binary classifier distinguishing real portrait photos from AI-generated faces.
Fine-tunes a pre-trained ViT-B/16 using LoRA adapters (PEFT), keeping 99%+ of
the backbone frozen while adapting only the attention projections.

**Primary dataset:** [140K Real and Fake Faces](https://www.kaggle.com/datasets/xhlulu/140k-real-and-fake-faces) —
140 000 images, perfectly balanced, predefined train/valid/test split. Real faces from Flickr,
fake faces generated with StyleGAN2.

**Tech stack:** PyTorch · PyTorch Lightning · HuggingFace Transformers · PEFT · Hydra · MLFlow · uv

---

## Setup

**Requirements:** Python 3.11+, [uv](https://github.com/astral-sh/uv)

```bash
git clone https://github.com/chris-beischl/deepfake-detection
cd deepfake_detection
uv sync
```

For development tools (ruff, mypy, pytest, jupyter):

```bash
uv sync              # dev dependencies are included by default
uv run pre-commit install   # optional: enable pre-commit hooks
```

---

## Data

**140K Real and Fake Faces (primary):** Download from [Kaggle](https://www.kaggle.com/datasets/xhlulu/140k-real-and-fake-faces)
and place at `data/140k-real-and-fake-faces/`. No further setup required — images are
pre-organised into `{train,valid,test}/{real,fake}/` directories.

**Deepfake Detection Dataset 2026 (auxiliary):** Download the CSV from
[Kaggle](https://www.kaggle.com/datasets/chuneeb/deepfake-detection-dataset-2026), then fetch images:

```bash
uv run python scripts/download_data.py --csv data/deepfake_detection_dataset.csv
```

Images are saved to `data/images/{train,val,test}/{image_id}.jpg`.
All datasets and images are excluded from version control.

---

## Training

```bash
# 140k dataset (primary)
uv run python train.py data=faces140k loss=bce_balanced

# Deepfake Detection Dataset 2026
uv run python train.py data=deepfake loss=bce
```

Config values can be overridden via Hydra at the command line:

```bash
uv run python train.py data=faces140k loss=bce_balanced model.peft_cfg.r=32 trainer.max_epochs=20
```

Monitor runs in the MLFlow UI:

```bash
uv run mlflow ui --backend-store-uri sqlite:///mlflow.db
```

Checkpoints are saved to `checkpoints/{experiment}/{model}/{data}/{run_id}/`.

---

## Evaluation

Run on the test set from a saved checkpoint:

```bash
uv run python eval.py --checkpoint checkpoints/<path>/<checkpoint_name>.ckpt
```

Results are logged to the original MLFlow run via the `run_id` stored in the checkpoint.

---

## Export & Quantization

Export a trained checkpoint to ONNX and optionally quantize:

```bash
# FP32 ONNX + INT8 quantization (default)
uv run python export.py --checkpoint checkpoints/<path>/<checkpoint_name>.ckpt

# Skip quantization
uv run python export.py --checkpoint checkpoints/<path>/<checkpoint_name>.ckpt --quant-type none

# Use a different quantization type
uv run python export.py --checkpoint checkpoints/<path>/<checkpoint_name>.ckpt --quant-type QUInt8

# Custom output directory
uv run python export.py --checkpoint checkpoints/<path>/<checkpoint_name>.ckpt --output-dir exports/
```

LoRA adapters are merged into the backbone before export (`merge_and_unload()`), producing a
standard ViT-B/16 with no PEFT dependency at inference time. Outputs are saved to `--output-dir`:
- `model.onnx` — FP32 model
- `model_qint8.onnx` — dynamically quantized INT8 model

**Quantization results (dynamic INT8, CPU, 100 inference runs):**

| Metric           |   FP32  | INT8   | Improvement |
|------------------|---------|--------|-------------|
| Size (MB)        | 327.5   | 82.9   | −74.7%      |
| Latency mean (ms)| 136.3   | 46.9   | −65.6%      |
| Latency std (ms) | 36.8    | 10.2   | −72.2%      |

---

## Results

**Dataset:** 140K Real and Fake Faces — 100k train / 20k val / 20k test, perfectly balanced.
**Model:** ViT-B/16 + LoRA (r=16, target: query + value projections)
**Training:** 10 epochs, AdamW, cosine LR with warmup, batch size 128

| Metric   | Test score |
|----------|-----------|
| Accuracy | 99.29%    |
| AUROC    | 99.98%    |
| F1       | 99.28%    |
| Loss     | 0.0186    |

The model converges rapidly — 96.8% accuracy is already reached after epoch 2, with diminishing
gains thereafter. LoRA keeps 99%+ of backbone parameters frozen throughout, training only
~0.5M adapter parameters on top of the 86M ViT-B/16 backbone.

**Note on dataset scope:** Results reflect detection of StyleGAN2-generated faces specifically.
Generalisation to other generation methods (diffusion models, face-swap) is not evaluated here.

---

## Project Structure

```
deepfake_detection/
├── train.py                  # Training entry point
├── eval.py                   # Evaluation entry point
├── export.py                 # ONNX export + quantization + benchmarking
├── scripts/
│   └── download_data.py      # Dataset download utility
├── configs/                  # Hydra configuration files
├── deepfakedet/
│   ├── data/                 # DataModule + Dataset
│   ├── models/               # ViT + LoRA wrapper
│   └── training/             # LightningModule, callbacks, utils
└── data/                     # Downloaded images (not committed)
```
