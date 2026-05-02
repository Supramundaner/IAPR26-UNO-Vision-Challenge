from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import DataLoader

sys.path.append(str(Path(__file__).resolve().parents[1]))

from uno_vision.config import (
    ARTIFACT_DIR,
    CONDITION_TO_INDEX,
    PLAYER_TO_CONDITION,
    PLAYERS,
    RAW_DATA_DIR,
    SUBMISSION_DIR,
)
from uno_vision.data import TestImageDataset
from uno_vision.labels import LabelEncoder
from uno_vision.models import ActivePlayerCNN, ConditionalCardCNN


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--card-model", type=Path, default=ARTIFACT_DIR / "card_model.pt")
    parser.add_argument("--active-model", type=Path, default=ARTIFACT_DIR / "active_player_model.pt")
    parser.add_argument("--vocab", type=Path, default=ARTIFACT_DIR / "card_vocab.json")
    parser.add_argument("--sample-submission", type=Path, default=RAW_DATA_DIR / "sample_submission.csv")
    parser.add_argument("--image-dir", type=Path, default=RAW_DATA_DIR / "test_images")
    parser.add_argument("--output", type=Path, default=SUBMISSION_DIR / "submission.csv")
    parser.add_argument("--image-size", default="384", help="Square resize size, or 'original'")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--empty-threshold", type=float, default=0.5)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--crop-by-token", action="store_true", help="Use token-specific ROI crops for card inference")
    parser.add_argument("--allow-random", action="store_true")
    return parser.parse_args()


def load_card_model(path: Path, num_cards: int, device: torch.device, allow_random: bool):
    model = ConditionalCardCNN(num_cards=num_cards).to(device)
    if path.exists():
        checkpoint = torch.load(path, map_location=device)
        model.load_state_dict(checkpoint["model_state"])
    elif not allow_random:
        raise FileNotFoundError(f"Missing card model checkpoint: {path}")
    return model.eval()


def load_active_model(path: Path, device: torch.device, allow_random: bool):
    model = ActivePlayerCNN().to(device)
    if path.exists():
        checkpoint = torch.load(path, map_location=device)
        model.load_state_dict(checkpoint["model_state"])
    elif not allow_random:
        raise FileNotFoundError(f"Missing active-player checkpoint: {path}")
    return model.eval()


def main() -> None:
    args = parse_args()
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        print(f"Requested {args.device}, but CUDA is unavailable. Falling back to CPU.")
        device = torch.device("cpu")
    else:
        device = torch.device(args.device)
    encoder = LabelEncoder.load(args.vocab)
    sample_df = pd.read_csv(args.sample_submission)
    image_ids = sample_df["image_id"].tolist()
    existing_image_ids = [image_id for image_id in image_ids if (args.image_dir / f"{image_id}.jpg").exists()]
    missing_image_ids = [image_id for image_id in image_ids if image_id not in set(existing_image_ids)]
    if missing_image_ids:
        print(f"Warning: {len(missing_image_ids)} sample rows have no image file: {missing_image_ids}")
    card_model = load_card_model(args.card_model, len(encoder.cards), device, args.allow_random)
    active_model = load_active_model(args.active_model, device, args.allow_random)

    predictions_by_id = {
        image_id: {
            "image_id": image_id,
            "center_card": encoder.cards[0],
            "active_player": "p1",
            "player_1_cards": "EMPTY",
            "player_2_cards": "EMPTY",
            "player_3_cards": "EMPTY",
            "player_4_cards": "EMPTY",
        }
        for image_id in existing_image_ids
    }
    with torch.no_grad():
        active_loader = DataLoader(
            TestImageDataset(args.image_dir, existing_image_ids, image_size=args.image_size),
            batch_size=args.batch_size,
            shuffle=False,
        )
        for batch_image_ids, images in active_loader:
            images = images.to(device)
            active_logits = active_model(images)
            active_indices = active_logits.argmax(dim=1).cpu().tolist()
            for row_index, image_id in enumerate(batch_image_ids):
                predictions_by_id[image_id]["active_player"] = PLAYERS[active_indices[row_index]]

        center_loader = DataLoader(
            TestImageDataset(
                args.image_dir,
                existing_image_ids,
                image_size=args.image_size,
                condition="center" if args.crop_by_token else None,
            ),
            batch_size=args.batch_size,
            shuffle=False,
        )
        for batch_image_ids, images in center_loader:
            images = images.to(device)
            center_condition = torch.full(
                (images.size(0),), CONDITION_TO_INDEX["center"], dtype=torch.long, device=device
            )
            center_logits, _ = card_model(images, center_condition)
            center_probs = torch.sigmoid(center_logits).cpu()
            for row_index, image_id in enumerate(batch_image_ids):
                predictions_by_id[image_id]["center_card"] = encoder.decode_center(center_probs[row_index])

        hand_probabilities = {}
        empty_probabilities = {}
        for player in PLAYERS:
            condition_name = PLAYER_TO_CONDITION[player]
            condition_loader = DataLoader(
                TestImageDataset(
                    args.image_dir,
                    existing_image_ids,
                    image_size=args.image_size,
                    condition=condition_name if args.crop_by_token else None,
                ),
                batch_size=args.batch_size,
                shuffle=False,
            )
            for batch_image_ids, images in condition_loader:
                images = images.to(device)
                condition = torch.full(
                    (images.size(0),), CONDITION_TO_INDEX[condition_name], dtype=torch.long, device=device
                )
                hand_logits, empty_logits = card_model(images, condition)
                hand_probs = torch.sigmoid(hand_logits).cpu()
                empty_probs = torch.sigmoid(empty_logits).cpu()
                for row_index, image_id in enumerate(batch_image_ids):
                    hand_probabilities[(image_id, player)] = hand_probs[row_index]
                    empty_probabilities[(image_id, player)] = empty_probs[row_index]

        for image_id in existing_image_ids:
            for player_number, player in enumerate(PLAYERS, start=1):
                if float(empty_probabilities[(image_id, player)].item()) >= args.empty_threshold:
                    predictions_by_id[image_id][f"player_{player_number}_cards"] = "EMPTY"
                else:
                    predictions_by_id[image_id][f"player_{player_number}_cards"] = encoder.decode_hand(
                        hand_probabilities[(image_id, player)], threshold=args.threshold
                    )

    fallback_center = encoder.cards[0]
    for image_id in missing_image_ids:
        predictions_by_id[image_id] = {
            "image_id": image_id,
            "center_card": fallback_center,
            "active_player": "p1",
            "player_1_cards": "EMPTY",
            "player_2_cards": "EMPTY",
            "player_3_cards": "EMPTY",
            "player_4_cards": "EMPTY",
        }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    rows = [predictions_by_id[image_id] for image_id in image_ids]
    pd.DataFrame(rows).to_csv(args.output, index=False)
    print(f"Wrote {args.output} ({len(rows)} rows)")


if __name__ == "__main__":
    main()
