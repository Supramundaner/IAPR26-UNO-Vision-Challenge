from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import pandas as pd


CARD_COLUMNS = [
    "center_card",
    "player_1_cards",
    "player_2_cards",
    "player_3_cards",
    "player_4_cards",
]


def split_cards(value: str) -> list[str]:
    if pd.isna(value) or value == "" or value == "EMPTY":
        return []
    return [card.strip() for card in str(value).split(";") if card.strip() and card.strip() != "EMPTY"]


def build_card_vocabulary(train_csv: Path) -> list[str]:
    df = pd.read_csv(train_csv)
    cards: set[str] = set()
    for column in CARD_COLUMNS:
        for value in df[column]:
            cards.update(split_cards(value))
    return sorted(cards)


@dataclass(frozen=True)
class LabelEncoder:
    cards: list[str]

    @property
    def card_to_index(self) -> dict[str, int]:
        return {card: idx for idx, card in enumerate(self.cards)}

    def encode_cards(self, value: str) -> list[float]:
        vector = [0.0] * len(self.cards)
        mapping = self.card_to_index
        for card in split_cards(value):
            vector[mapping[card]] = 1.0
        return vector

    def encode_center(self, value: str) -> int:
        cards = split_cards(value)
        if len(cards) != 1:
            raise ValueError(f"Expected exactly one center card, got {value!r}")
        return self.card_to_index[cards[0]]

    def decode_hand(self, probabilities, threshold: float = 0.5) -> str:
        selected = [card for card, prob in zip(self.cards, probabilities) if float(prob) >= threshold]
        return ";".join(selected) if selected else "EMPTY"

    def decode_center(self, probabilities) -> str:
        index = int(max(range(len(self.cards)), key=lambda idx: float(probabilities[idx])))
        return self.cards[index]

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"cards": self.cards}, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> "LabelEncoder":
        payload = json.loads(path.read_text(encoding="utf-8"))
        return cls(cards=list(payload["cards"]))


def load_or_create_encoder(train_csv: Path, output_path: Path) -> LabelEncoder:
    if output_path.exists():
        return LabelEncoder.load(output_path)
    encoder = LabelEncoder(build_card_vocabulary(train_csv))
    encoder.save(output_path)
    return encoder
