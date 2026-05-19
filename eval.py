# Adapted from https://github.com/chris-beischl/liteViT

from argparse import ArgumentParser
from typing import Any

import hydra
import lightning as L
import numpy as np
import onnxruntime as ort
import torch
from omegaconf import OmegaConf
from torchmetrics import MetricCollection
from tqdm import tqdm

from deepfakedet.data import BaseDataModule
from deepfakedet.training import ClassificationModule


def eval(ckpt: dict[str, Any], cfg: Any) -> dict[str, Any]:
    run_id = ckpt.get("run_id")

    L.seed_everything(cfg.seed, workers=True)

    model = hydra.utils.instantiate(cfg.model)

    pos_weight = (
        torch.tensor(cfg.loss.pos_weight) if cfg.loss.get("pos_weight") else None
    )
    loss = hydra.utils.instantiate(cfg.loss, pos_weight=pos_weight)

    optimizer_cfg = cfg.optimizer
    scheduler_cfg = cfg.scheduler
    metrics = (
        MetricCollection(
            {
                name: hydra.utils.instantiate(metric_cfg)
                for name, metric_cfg in cfg.metrics.items()
            }
        )
        if cfg.metrics is not None
        else None
    )

    module = ClassificationModule(
        model=model,
        loss=loss,
        optimizer_cfg=optimizer_cfg,
        scheduler_cfg=scheduler_cfg,
        metrics=metrics,
    )

    module.load_state_dict(ckpt["state_dict"])

    data = hydra.utils.instantiate(cfg.data)
    logger = (
        hydra.utils.instantiate(cfg.logger, run_id=run_id)
        if cfg.logger is not None
        else None
    )

    trainer: L.Trainer = hydra.utils.instantiate(cfg.trainer, logger=logger)
    trainer.test(module, datamodule=data)
    return trainer.callback_metrics


def eval_onnx(onnx_path: str, cfg: Any, provider: str) -> None:
    L.seed_everything(cfg.seed, workers=True)
    session = ort.InferenceSession(str(onnx_path), providers=[provider])

    metrics = (
        MetricCollection(
            {
                name: hydra.utils.instantiate(metric_cfg)
                for name, metric_cfg in cfg.metrics.items()
            }
        )
        if cfg.metrics is not None
        else None
    )

    data: BaseDataModule = hydra.utils.instantiate(cfg.data)
    data.prepare_data()
    data.setup(stage="test")

    labels = []
    preds = []
    dataloader = data.test_dataloader()

    for x, y in tqdm(iter(dataloader), total=len(dataloader)):
        labels.append(y)
        pred = session.run(None, {"input": x.cpu().numpy()})
        preds.append(pred[0].squeeze(1))

    all_logits = np.concatenate(preds)
    all_labels = np.concatenate([label.numpy() for label in labels])

    logits_tensor = torch.from_numpy(all_logits)
    labels_tensor = torch.from_numpy(all_labels)

    if metrics is not None:
        metrics.update(logits_tensor, labels_tensor)
        results = metrics.compute()
        for k, v in results.items():
            print(f"{k}: {v:.4f}")


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--checkpoint", "-c", type=str)
    parser.add_argument("--onnx", type=str, default=None)
    parser.add_argument("--provider", type=str, default="CPUExecutionProvider")
    parser.add_argument("--num-workers", type=int, default=4)

    args = parser.parse_args()
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    cfg = OmegaConf.create(ckpt.get("cfg"))
    cfg.data.num_workers = args.num_workers

    if args.onnx is not None:
        eval_onnx(args.onnx, cfg, args.provider)
    else:
        eval(ckpt, cfg)
