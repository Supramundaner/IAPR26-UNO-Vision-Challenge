from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

sys.path.append(str(Path(__file__).resolve().parents[1]))

from uno_vision.config import ARTIFACT_DIR, PARAMETER_LIMIT, RAW_DATA_DIR
from uno_vision.data import ConditionalCardDataset, train_val_image_ids
from uno_vision.labels import load_or_create_encoder
from uno_vision.models import ConditionalCardCNN, count_parameters
from uno_vision.training import add_common_training_args, maybe_init_wandb, maybe_log_wandb, resolve_device


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--image-size", default="384", help="Square resize size, or 'original'")
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--center-loss-weight", type=float, default=1.0)
    parser.add_argument("--hand-loss-weight", type=float, default=1.0)
    parser.add_argument("--empty-loss-weight", type=float, default=0.25)
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--output", type=Path, default=ARTIFACT_DIR / "card_model.pt")
    add_common_training_args(parser)
    return parser.parse_args()


def run_epoch(
    model,
    loader,
    center_criterion,
    hand_criterion,
    empty_criterion,
    optimizer,
    device: torch.device,
    train: bool,
    center_loss_weight: float,
    hand_loss_weight: float,
    empty_loss_weight: float,
    desc: str,
) -> tuple[float, float, float, float]:
    model.train(train)
    total_loss = 0.0
    total_center_loss = 0.0
    total_hand_loss = 0.0
    total_empty_loss = 0.0
    total_count = 0
    progress = tqdm(loader, desc=desc, leave=False, dynamic_ncols=True)
    for images, conditions, card_targets, center_targets, empty_targets, center_masks, hand_card_masks, empty_masks in progress:
        images = images.to(device)
        conditions = conditions.to(device)
        card_targets = card_targets.to(device)
        center_targets = center_targets.to(device)
        empty_targets = empty_targets.to(device)
        center_masks = center_masks.to(device)
        hand_card_masks = hand_card_masks.to(device)
        empty_masks = empty_masks.to(device)
        with torch.set_grad_enabled(train):
            card_logits, empty_logits = model(images, conditions)
            raw_center_loss = center_criterion(card_logits, center_targets)
            center_loss = (raw_center_loss * center_masks.flatten()).sum() / center_masks.sum().clamp_min(1.0)
            raw_hand_loss = hand_criterion(card_logits, card_targets).mean(dim=1, keepdim=True)
            hand_loss = (raw_hand_loss * hand_card_masks).sum() / hand_card_masks.sum().clamp_min(1.0)
            raw_empty_loss = empty_criterion(empty_logits, empty_targets)
            empty_loss = (raw_empty_loss * empty_masks).sum() / empty_masks.sum().clamp_min(1.0)
            loss = (
                center_loss_weight * center_loss
                + hand_loss_weight * hand_loss
                + empty_loss_weight * empty_loss
            )
            if train:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
        total_loss += float(loss.item()) * images.size(0)
        total_center_loss += float(center_loss.item()) * images.size(0)
        total_hand_loss += float(hand_loss.item()) * images.size(0)
        total_empty_loss += float(empty_loss.item()) * images.size(0)
        total_count += images.size(0)
        progress.set_postfix(
            loss=f"{float(loss.item()):.4f}",
            center=f"{float(center_loss.item()):.4f}",
            hand=f"{float(hand_loss.item()):.4f}",
            empty=f"{float(empty_loss.item()):.4f}",
        )
    denominator = max(1, total_count)
    return total_loss / denominator, total_center_loss / denominator, total_hand_loss / denominator, total_empty_loss / denominator


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    device = resolve_device(args.device)

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
    print(f"Device: {device}")
    print(f"ConditionalCardCNN parameters: {params:,}")
    if params >= PARAMETER_LIMIT:
        raise SystemExit(f"Model exceeds {PARAMETER_LIMIT:,} parameters")

    center_criterion = nn.CrossEntropyLoss(reduction="none")
    hand_criterion = nn.BCEWithLogitsLoss(reduction="none")
    empty_criterion = nn.BCEWithLogitsLoss(reduction="none")
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    run = maybe_init_wandb(
        args,
        "train-card-model",
        {
            **vars(args),
            "parameters": params,
            "num_cards": len(encoder.cards),
        },
    )

    best_val = float("inf")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    for epoch in range(1, args.epochs + 1):
        train_loss, train_center_loss, train_hand_loss, train_empty_loss = run_epoch(
            model,
            train_loader,
            center_criterion,
            hand_criterion,
            empty_criterion,
            optimizer,
            device,
            True,
            args.center_loss_weight,
            args.hand_loss_weight,
            args.empty_loss_weight,
            f"epoch {epoch:03d}/{args.epochs} train",
        )
        val_loss, val_center_loss, val_hand_loss, val_empty_loss = run_epoch(
            model,
            val_loader,
            center_criterion,
            hand_criterion,
            empty_criterion,
            optimizer,
            device,
            False,
            args.center_loss_weight,
            args.hand_loss_weight,
            args.empty_loss_weight,
            f"epoch {epoch:03d}/{args.epochs} val",
        )
        print(
            f"epoch={epoch:03d} train_loss={train_loss:.5f} train_center={train_center_loss:.5f} "
            f"train_hand={train_hand_loss:.5f} "
            f"train_empty={train_empty_loss:.5f} val_loss={val_loss:.5f} "
            f"val_center={val_center_loss:.5f} val_hand={val_hand_loss:.5f} val_empty={val_empty_loss:.5f}"
        )
        maybe_log_wandb(
            run,
            {
                "epoch": epoch,
                "train/loss": train_loss,
                "train/center_loss": train_center_loss,
                "train/hand_loss": train_hand_loss,
                "train/empty_loss": train_empty_loss,
                "val/loss": val_loss,
                "val/center_loss": val_center_loss,
                "val/hand_loss": val_hand_loss,
                "val/empty_loss": val_empty_loss,
            },
        )
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
    if run is not None:
        run.finish()


if __name__ == "__main__":
    main()
