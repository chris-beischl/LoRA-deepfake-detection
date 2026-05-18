# Adapted from https://github.com/chris-beischl/liteViT

from abc import ABC, abstractmethod
from typing import Any

import torch.nn as nn
from lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset


class BaseDataModule(LightningDataModule, ABC):
    def __init__(
        self,
        train_transform: nn.Module | None = None,
        eval_transform: nn.Module | None = None,
        num_workers: int = 0,
        batch_size: int = 32,
        pin_memory: bool = False,
        persistent_workers: bool = False,
        prefetch_factor: int = 2,
        seed: int = 0,
        **kwargs: Any,
    ) -> None:
        super().__init__()
        self.train_transform = train_transform or nn.Identity()
        self.eval_transform = eval_transform or nn.Identity()
        self.num_workers = num_workers
        self.batch_size = batch_size
        self.pin_memory = pin_memory
        self.persistent_workers = persistent_workers
        self.prefetch_factor = prefetch_factor

        self._train_dataset: Dataset[tuple[Any, Any]] | None = None
        self._val_dataset: Dataset[tuple[Any, Any]] | None = None
        self._test_dataset: Dataset[tuple[Any, Any]] | None = None

    @property
    @abstractmethod
    def num_classes(self) -> int: ...

    @property
    @abstractmethod
    def num_channels(self) -> int: ...

    @property
    @abstractmethod
    def img_size(self) -> tuple[int, int]: ...

    @abstractmethod
    def prepare_data(self) -> None: ...

    @abstractmethod
    def setup(self, stage: str | None = None) -> None: ...

    def _make_dataloader(
        self, dataset: Dataset[tuple[Any, Any]], shuffle: bool
    ) -> DataLoader[Any]:
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.persistent_workers,
            prefetch_factor=self.prefetch_factor if self.num_workers > 0 else None,
            shuffle=shuffle,
        )

    def train_dataloader(self) -> DataLoader[Any]:
        if self._train_dataset is None:
            raise RuntimeError(
                "train_dataloader() called but _train_dataset is None. "
                "Ensure setup() was called with the correct stage or stage=None."
            )
        return self._make_dataloader(self._train_dataset, shuffle=True)

    def val_dataloader(self) -> DataLoader[Any]:
        if self._val_dataset is None:
            raise RuntimeError(
                "val_dataloader() called but _val_dataset is None. "
                "Ensure setup() was called with the correct stage or stage=None."
            )
        return self._make_dataloader(self._val_dataset, shuffle=False)

    def test_dataloader(self) -> DataLoader[Any]:
        if self._test_dataset is None:
            raise RuntimeError(
                "test_dataloader() called but _test_dataset is None. "
                "Ensure setup() was called with the correct stage or stage=None."
            )
        return self._make_dataloader(self._test_dataset, shuffle=False)
