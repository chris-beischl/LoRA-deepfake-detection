# Adapted from https://github.com/chris-beischl/liteViT

from typing import Any

import hydra
import lightning as L
import torch
from hydra.core.hydra_config import HydraConfig
from lightning.pytorch.callbacks import ModelCheckpoint
from omegaconf import OmegaConf
from torch import set_float32_matmul_precision as torch_set_float32_matmul_precision
from torchmetrics import MetricCollection

from deepfakedet.conf import register_configs
from deepfakedet.training import ClassificationModule
from deepfakedet.training.callbacks import CheckpointMetadataCallback

register_configs()


@hydra.main(version_base=None, config_path="configs", config_name="train")
def main(cfg: dict[str, Any]) -> None:
    run_training(cfg)


def run_training(cfg: Any) -> float:
    # resolve then re-wrap so interpolations are gone before passing to Lightning
    cfg = OmegaConf.to_container(cfg, resolve=True)
    cfg = OmegaConf.create(cfg)

    L.seed_everything(cfg.seed, workers=True)
    torch_set_float32_matmul_precision("medium")

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

    data = hydra.utils.instantiate(cfg.data)
    logger = hydra.utils.instantiate(cfg.logger) if cfg.logger is not None else None

    # HydraConfig is unavailable when called via hydra.compose (sweeps)
    try:
        model_cfg = HydraConfig.get().runtime.choices["model"]
        data_cfg = HydraConfig.get().runtime.choices["data"]
    except ValueError:
        model_cfg = "unknown"
        data_cfg = "unknown"

    experiment_name = logger._experiment_name if logger is not None else "default"
    run_id = logger.run_id if logger is not None else None

    callbacks = [hydra.utils.instantiate(cb) for cb in cfg.callbacks]
    checkpoint_metadata_callback = CheckpointMetadataCallback(cfg, run_id)
    callbacks.append(checkpoint_metadata_callback)

    if logger is not None and hasattr(logger, "log_hyperparams"):
        checkpoint_path = (
            f"checkpoints/{experiment_name}/{model_cfg}/{data_cfg}/{run_id}"
        )

        # organise checkpoints under the MLflow experiment/model/data/run tree
        for cb in callbacks:
            if isinstance(cb, ModelCheckpoint):
                cb.dirpath = checkpoint_path

        logger.log_hyperparams(OmegaConf.to_container(cfg, resolve=True))
        # log config group names separately so MLflow runs are filterable by model/data
        logger.log_hyperparams(
            {
                "model_cfg": model_cfg,
                "data_cfg": data_cfg,
                "checkpoint_path": (
                    checkpoint_path
                    if any([isinstance(cb, ModelCheckpoint) for cb in callbacks])
                    else None
                ),
            }
        )

    trainer: L.Trainer = hydra.utils.instantiate(
        cfg.trainer, callbacks=callbacks, logger=logger
    )
    trainer.fit(module, datamodule=data)

    # return validation accuracy for optuna to use for hparam selection
    return trainer.callback_metrics.get("val_accuracy", torch.tensor(0.0)).item()


if __name__ == "__main__":
    main()
