# Copied from https://github.com/chris-beischl/liteViT
from typing import Any

import lightning as L
from omegaconf import DictConfig, OmegaConf


class CheckpointMetadataCallback(L.Callback):
    def __init__(self, cfg: DictConfig, run_id: str | None) -> None:
        super().__init__()
        self.cfg = cfg
        self.run_id = run_id

    def on_save_checkpoint(
        self,
        trainer: L.Trainer,
        pl_module: L.LightningModule,
        checkpoint: dict[str, Any],
    ) -> None:
        checkpoint["cfg"] = OmegaConf.to_container(self.cfg, resolve=True)
        checkpoint["run_id"] = self.run_id
