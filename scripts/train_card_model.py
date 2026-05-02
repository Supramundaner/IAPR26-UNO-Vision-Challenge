from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader

sys.path.append(str(Path(__file__).resolve().parents[1]))

from uno_vision.config import ARTIFACT_DIR, PARAMETER_LIMIT, RAW_DATA_DIR
from uno_vision.data import ConditionalCardDataset, train_val_image_ids
from uno_vision.labels import load_or_create_encoder
from uno_vision.models import ConditionalCardCNN, count_parameters


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--image-size", type=int, default=384)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--output", type=Path, default=ARTIFACT_DIR / "card_model.pt")
    return parser.parse_args()


def run_epoch(model, loader, criterion, optimizer, device: torch.device, train: bool) -> float:
    model.train(train)
    total_loss = 0.0
    total_count = 0
    for images, conditions, targets in loader:
        images = images.to(device)
        conditions = conditions.to(device)
        targets = targets.to(device)
        with torch.set_grad_enabled(train):
            logits = model(images, conditions)
            loss = criterion(logits, targets)
            if train:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
        total_loss += float(loss.item()) * images.size(0)
        total_count += images.size(0)
    return total_loss / max(1, total_count)


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    encoder = load_or_create_encoder(RAW_DATA_DIR / "train.csv", ARTIFACT_DIR / "card_vocab.json")
    train_ids, val_ids = train_val_image_ids(RAW_DATA_DIR / "train.csv", args.val_fraction, args.seed)
    train_dataset = ConditionalCardDataset(
        RAW_DATA_DIR / "train.csv",
        RAW_DATA_DIR / "train_images",
        encoder,
        image_ids=train_ids,
        image_size=args.image_size,
        train=True,
    )
    val_dataset = ConditionalCardDataset(
        RAW_DATA_DIR / "train.csv",
        RAW_DATA_DIR / "train_images",
        encoder,
        image_ids=val_ids,
        image_size=args.image_size,
        train=False,
    )
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    model = ConditionalCardCNN(num_cards=len(encoder.cards)).to(device)
    params = count_parameters(model)
    print(f"ConditionalCardCNN parameters: {params:,}")
    if params >= PARAMETER_LIMIT:
        raise SystemExit(f"Model exceeds {PARAMETER_LIMIT:,} parameters")

    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    best_val = float("inf")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    for epoch in range(1, args.epochs + 1):
        train_loss = run_epoch(model, train_loader, criterion, optimizer, device, train=True)
        val_loss = run_epoch(model, val_loader, criterion, optimizer, device, train=False)
        print(f"epoch={epoch:03d} train_loss={train_loss:.5f} val_loss={val_loss:.5f}")
        if val_loss < best_val:
            best_val = val_loss
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "num_cards": len(encoder.cards),
                    "image_size": args.image_size,
                    "card_vocab_path": str(ARTIFACT_DIR / "card_vocab.json"),
                },
                args.output,
            )
            print(f"saved {args.output}")


if __name__ == "__main__":
    main()
