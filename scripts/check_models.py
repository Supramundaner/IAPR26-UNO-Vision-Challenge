from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

sys.path.append(str(Path(__file__).resolve().parents[1]))

from uno_vision.config import ARTIFACT_DIR, PARAMETER_LIMIT, RAW_DATA_DIR
from uno_vision.labels import load_or_create_encoder
from uno_vision.models import ActivePlayerCNN, ConditionalCardCNN, count_parameters


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--image-size", type=int, default=384)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    encoder = load_or_create_encoder(RAW_DATA_DIR / "train.csv", ARTIFACT_DIR / "card_vocab.json")
    card_model = ConditionalCardCNN(num_cards=len(encoder.cards))
    active_model = ActivePlayerCNN()

    card_params = count_parameters(card_model)
    active_params = count_parameters(active_model)
    print(f"Card vocabulary size: {len(encoder.cards)}")
    print(f"ConditionalCardCNN parameters: {card_params:,}")
    print(f"ActivePlayerCNN parameters: {active_params:,}")

    if card_params >= PARAMETER_LIMIT:
        raise SystemExit(f"ConditionalCardCNN exceeds {PARAMETER_LIMIT:,} parameters")
    if active_params >= PARAMETER_LIMIT:
        raise SystemExit(f"ActivePlayerCNN exceeds {PARAMETER_LIMIT:,} parameters")

    batch = torch.randn(2, 3, args.image_size, args.image_size)
    conditions = torch.tensor([0, 1], dtype=torch.long)
    with torch.no_grad():
        card_logits, empty_logits = card_model(batch, conditions)
        active_logits = active_model(batch)
    print(f"Card logits shape: {tuple(card_logits.shape)}")
    print(f"Active logits shape: {tuple(active_logits.shape)}")
    print(f"Empty logits shape: {tuple(empty_logits.shape)}")


if __name__ == "__main__":
    main()
