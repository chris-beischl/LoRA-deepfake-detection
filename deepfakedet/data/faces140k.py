import warnings
from pathlib import Path
from typing import Any

from torchvision.datasets import ImageFolder
from torchvision.transforms import Compose

from .base import BaseDataModule

_SPLITS = ("train", "valid", "test")


class Faces140kDataModule(BaseDataModule):
    def __init__(
        self,
        root_dir: str | Path,
        train_transform: Compose | None = None,
        eval_transform: Compose | None = None,
        num_workers: int = 0,
        batch_size: int = 32,
        pin_memory: bool = False,
        persistent_workers: bool = False,
        prefetch_factor: int = 2,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            train_transform,
            eval_transform,
            num_workers,
            batch_size,
            pin_memory,
            persistent_workers,
            prefetch_factor,
            **kwargs,
        )
        self.root_dir = Path(root_dir)

    @property
    def num_classes(self) -> int:
        return 2

    @property
    def num_channels(self) -> int:
        return 3

    @property
    def img_size(self) -> tuple[int, int]:
        return (224, 224)

    def prepare_data(self) -> None:
        if not self.root_dir.exists():
            raise FileNotFoundError(f"Dataset root not found: {self.root_dir}")
        for split in _SPLITS:
            split_dir = self.root_dir / split
            if not split_dir.exists():
                warnings.warn(f"Split directory missing: {split_dir}")

    def setup(self, stage: str | None = None) -> None:
        if stage == "fit" or stage is None:
            self._train_dataset = ImageFolder(
                root=str(self.root_dir / "train"),
                transform=self.train_transform,
            )
            self._val_dataset = ImageFolder(
                root=str(self.root_dir / "valid"),
                transform=self.eval_transform,
            )
        if stage == "test" or stage is None:
            self._test_dataset = ImageFolder(
                root=str(self.root_dir / "test"),
                transform=self.eval_transform,
            )
