# Deepfake Detection via LoRA Fine-Tuned ViT

Binary classifier distinguishing real portrait photos from AI-generated faces.
Fine-tunes a pre-trained ViT-B/16 using LoRA adapters (PEFT), keeping 99%+ of
the backbone frozen while adapting only the attention projections.

**Dataset:** [Deepfake Detection Dataset 2026](https://www.kaggle.com/datasets/chuneeb/deepfake-detection-dataset-2026) —
6557 images (57% fake, 43% real) with a predefined train/val/test split.

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

Download and cache all images locally (~6 500 images, ~1–2 GB):

```bash
uv run python scripts/download_data.py --csv data/deepfake_detection_dataset.csv
```

Images are saved to `data/images/{train,val,test}/{image_id}.jpg`.
The CSV and downloaded images are excluded from version control.

---

## Training

```bash
uv run python train.py
```

Config values can be overridden via Hydra at the command line:

```bash
uv run python train.py model=vit_base_lora model.peft_cfg.r=32 trainer.max_epochs=20
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
uv run python eval.py --checkpoint checkpoints/<path>/best.ckpt
```

Results are logged to the original MLFlow run via the `run_id` stored in the checkpoint.

---

## Results

*To be updated after training on a suitable dataset.*

**Note on current dataset:** The [Deepfake Detection Dataset 2026](https://www.kaggle.com/datasets/chuneeb/deepfake-detection-dataset-2026)
yields inflated metrics (~98% test accuracy after 2 epochs) because real and fake images
originate from visually distinct sources (Unsplash photography vs. synthetic face generators).
The model learns source-domain features rather than manipulation artifacts.
A dataset with consistent source distribution across classes is required for meaningful evaluation.

---

## Project Structure

```
deepfake_detection/
├── train.py                  # Training entry point
├── eval.py                   # Evaluation entry point
├── scripts/
│   └── download_data.py      # Dataset download utility
├── configs/                  # Hydra configuration files
├── deepfakedet/
│   ├── data/                 # DataModule + Dataset
│   ├── models/               # ViT + LoRA wrapper
│   └── training/             # LightningModule, callbacks, utils
└── data/                     # Downloaded images (not committed)
```
