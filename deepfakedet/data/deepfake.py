import warnings
from pathlib import Path
from typing import Any

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import Compose

from .base import BaseDataModule


class DeepfakeDataset(Dataset[tuple[torch.Tensor, int]]):
    def __init__(self, df: pd.DataFrame, img_dir: Path, transform: Compose) -> None:
        self.df = df.reset_index(drop=True)
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        row = self.df.iloc[idx]
        img = Image.open(
            self.img_dir / row.dataset_split / f"{row.image_id}.jpg"
        ).convert("RGB")
        return self.transform(img), int(row.label_numeric)


class DeepfakeDataModule(BaseDataModule):
    def __init__(
        self,
        csv_path: Path | str,
        img_dir: Path | str,
        train_transform: Compose | None = None,
        eval_transform: Compose | None = None,
        num_workers: int = 0,
        batch_size: int = 32,
        pin_memory: bool = False,
        persistent_workers: bool = False,
        prefetch_factor: int = 2,
        **kwargs: Any,
    ):
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

        self.csv_path = Path(csv_path)
        self.img_dir = Path(img_dir)

        self.df = pd.read_csv(csv_path)

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
        # Check if csv file and image dir exist:
        if not self.csv_path.exists():
            raise FileNotFoundError(f"{self.csv_path} doesn't exist!")
        if not self.img_dir.exists():
            raise FileNotFoundError(f"Image directory {self.img_dir} doesn't exist!")

        for split in self.df["dataset_split"].unique():
            split_dir = self.img_dir / split
            if not split_dir.exists():
                warnings.warn(f"Split directory {split_dir} does not exist.")
                continue
            expected = len(self.df[self.df["dataset_split"] == split])
            actual = len(list(split_dir.glob("*.jpg")))
            if actual != expected:
                warnings.warn(
                    f"Split '{split}': expected {expected} images, found {actual}. "
                    "Run scripts/download_data.py to complete the download."
                )

    def setup(self, stage: str | None = None) -> None:
        if stage == "fit" or stage is None:
            train_df = self._filter_existing(
                self.df[self.df["dataset_split"] == "train"], "train"
            )
            val_df = self._filter_existing(
                self.df[self.df["dataset_split"] == "val"], "val"
            )
            self._train_dataset = DeepfakeDataset(
                train_df, self.img_dir, transform=self.train_transform
            )
            self._val_dataset = DeepfakeDataset(
                val_df, self.img_dir, transform=self.eval_transform
            )
        if stage == "test" or stage is None:
            test_df = self._filter_existing(
                self.df[self.df["dataset_split"] == "test"], "test"
            )
            self._test_dataset = DeepfakeDataset(
                test_df, self.img_dir, transform=self.eval_transform
            )

    def _filter_existing(self, df: pd.DataFrame, split: str) -> pd.DataFrame:
        mask = df["image_id"].apply(
            lambda img_id: (self.img_dir / split / f"{img_id}.jpg").exists()
        )
        missing = int((~mask).sum())
        if missing > 0:
            warnings.warn(f"Split '{split}': skipping {missing} missing images.")
        return df[mask]
