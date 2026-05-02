from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

from uno_vision.config import CONDITION_TO_INDEX, PLAYER_TO_CONDITION, PLAYER_TO_INDEX
from uno_vision.labels import LabelEncoder


ImageSize = int | str


def normalize_image_size(image_size: ImageSize) -> int | None:
    if isinstance(image_size, str):
        if image_size.lower() == "original":
            return None
        return int(image_size)
    return image_size


def image_transform(image_size: ImageSize, train: bool) -> transforms.Compose:
    normalized_size = normalize_image_size(image_size)
    steps = []
    if normalized_size is not None:
        steps.append(transforms.Resize((normalized_size, normalized_size)))
    if train:
        steps.extend(
            [
                transforms.ColorJitter(brightness=0.12, contrast=0.12, saturation=0.10, hue=0.02),
            ]
        )
        if normalized_size is not None:
            steps.append(transforms.RandomAffine(degrees=4, translate=(0.02, 0.02), scale=(0.96, 1.04)))
    steps.extend(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ]
    )
    return transforms.Compose(steps)


def read_image(path: Path) -> Image.Image:
    with Image.open(path) as image:
        return image.convert("RGB")


def train_val_image_ids(train_csv: Path, val_fraction: float, seed: int) -> tuple[set[str], set[str]]:
    df = pd.read_csv(train_csv)
    image_ids = sorted(df["image_id"].tolist())
    rng = random.Random(seed)
    rng.shuffle(image_ids)
    val_count = max(1, int(round(len(image_ids) * val_fraction)))
    val_ids = set(image_ids[:val_count])
    train_ids = set(image_ids[val_count:])
    return train_ids, val_ids


@dataclass(frozen=True)
class ConditionalCardSample:
    image_id: str
    condition: str
    target_column: str
    is_center: bool


class ConditionalCardDataset(Dataset):
    def __init__(
        self,
        train_csv: Path,
        image_dir: Path,
        encoder: LabelEncoder,
        image_ids: set[str] | None = None,
        image_size: ImageSize = 384,
        train: bool = True,
    ) -> None:
        self.df = pd.read_csv(train_csv)
        if image_ids is not None:
            self.df = self.df[self.df["image_id"].isin(image_ids)].reset_index(drop=True)
        self.image_dir = image_dir
        self.encoder = encoder
        self.transform = image_transform(image_size, train=train)
        self.samples = self._build_samples()

    def _build_samples(self) -> list[ConditionalCardSample]:
        samples: list[ConditionalCardSample] = []
        for row in self.df.itertuples(index=False):
            image_id = str(row.image_id)
            samples.append(ConditionalCardSample(image_id, "center", "center_card", True))
            for player_index, player in enumerate(["p1", "p2", "p3", "p4"], start=1):
                target_column = f"player_{player_index}_cards"
                samples.append(
                    ConditionalCardSample(
                        image_id=image_id,
                        condition=PLAYER_TO_CONDITION[player],
                        target_column=target_column,
                        is_center=False,
                    )
                )
        return samples

    def __len__(self) -> int:
        return len(self.samples)

    def label_statistics(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        card_positive = torch.zeros(len(self.encoder.cards), dtype=torch.float32)
        card_total = 0
        empty_positive = torch.zeros(1, dtype=torch.float32)
        empty_total = torch.zeros(1, dtype=torch.float32)
        for sample in self.samples:
            row = self.df[self.df["image_id"] == sample.image_id].iloc[0]
            card_positive += torch.tensor(self.encoder.encode_cards(row[sample.target_column]), dtype=torch.float32)
            card_total += 1
            if not sample.is_center:
                empty_positive += 1.0 if row[sample.target_column] == "EMPTY" else 0.0
                empty_total += 1.0
        card_total_tensor = torch.full_like(card_positive, float(card_total))
        return card_positive, card_total_tensor, torch.cat([empty_positive, empty_total])

    def __getitem__(self, index: int):
        sample = self.samples[index]
        row = self.df[self.df["image_id"] == sample.image_id].iloc[0]
        image = self.transform(read_image(self.image_dir / f"{sample.image_id}.jpg"))
        condition = torch.tensor(CONDITION_TO_INDEX[sample.condition], dtype=torch.long)
        target = torch.tensor(self.encoder.encode_cards(row[sample.target_column]), dtype=torch.float32)
        empty_target = torch.tensor([1.0 if row[sample.target_column] == "EMPTY" else 0.0], dtype=torch.float32)
        empty_mask = torch.tensor([0.0 if sample.is_center else 1.0], dtype=torch.float32)
        return image, condition, target, empty_target, empty_mask


class ActivePlayerDataset(Dataset):
    def __init__(
        self,
        train_csv: Path,
        image_dir: Path,
        image_ids: set[str] | None = None,
        image_size: ImageSize = 384,
        train: bool = True,
    ) -> None:
        self.df = pd.read_csv(train_csv)
        if image_ids is not None:
            self.df = self.df[self.df["image_id"].isin(image_ids)].reset_index(drop=True)
        self.image_dir = image_dir
        self.transform = image_transform(image_size, train=train)

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, index: int):
        row = self.df.iloc[index]
        image = self.transform(read_image(self.image_dir / f"{row['image_id']}.jpg"))
        target = torch.tensor(PLAYER_TO_INDEX[row["active_player"]], dtype=torch.long)
        return image, target


class TestImageDataset(Dataset):
    def __init__(self, image_dir: Path, image_ids: list[str], image_size: ImageSize = 384) -> None:
        self.image_dir = image_dir
        self.image_ids = image_ids
        self.transform = image_transform(image_size, train=False)

    def __len__(self) -> int:
        return len(self.image_ids)

    def __getitem__(self, index: int):
        image_id = self.image_ids[index]
        image = self.transform(read_image(self.image_dir / f"{image_id}.jpg"))
        return image_id, image
