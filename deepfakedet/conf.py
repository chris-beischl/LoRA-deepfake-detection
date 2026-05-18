# Adapted from https://github.com/chris-beischl/liteViT

from dataclasses import dataclass
from typing import Any

from hydra.core.config_store import ConfigStore
from omegaconf import MISSING


@dataclass
class TrainConfig:
    model: Any = MISSING
    data: Any = MISSING
    optimizer: Any = MISSING
    scheduler: Any = None
    metrics: Any = None
    callbacks: Any = None
    trainer: Any = MISSING
    logger: Any = None
    loss: Any = MISSING
    seed: int = 0


def register_configs() -> None:
    cs = ConfigStore.instance()
    cs.store(name="train_config", node=TrainConfig)
