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
from uno_vision.data import ActivePlayerDataset, train_val_image_ids
from uno_vision.models import ActivePlayerCNN, count_parameters
from uno_vision.training import add_common_training_args, maybe_init_wandb, maybe_log_wandb, resolve_device


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--image-size", default="384", help="Square resize size, or 'original'")
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--output", type=Path, default=ARTIFACT_DIR / "active_player_model.pt")
    add_common_training_args(parser)
    return parser.parse_args()


def run_epoch(model, loader, criterion, optimizer, device: torch.device, train: bool, desc: str) -> tuple[float, float]:
    model.train(train)
    total_loss = 0.0
    total_correct = 0
    total_count = 0
    progress = tqdm(loader, desc=desc, leave=False, dynamic_ncols=True)
    for images, targets in progress:
        images = images.to(device)
        targets = targets.to(device)
        with torch.set_grad_enabled(train):
            logits = model(images)
            loss = criterion(logits, targets)
            if train:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
        total_loss += float(loss.item()) * images.size(0)
        total_correct += int((logits.argmax(dim=1) == targets).sum().item())
        total_count += images.size(0)
        batch_acc = float((logits.argmax(dim=1) == targets).float().mean().item())
        progress.set_postfix(loss=f"{float(loss.item()):.4f}", acc=f"{batch_acc:.3f}")
    return total_loss / max(1, total_count), total_correct / max(1, total_count)


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    device = resolve_device(args.device)
    train_ids, val_ids = train_val_image_ids(RAW_DATA_DIR / "train.csv", args.val_fraction, args.seed)
    train_dataset = ActivePlayerDataset(
        RAW_DATA_DIR / "train.csv",
        RAW_DATA_DIR / "train_images",
        image_ids=train_ids,
        image_size=args.image_size,
        train=True,
    )
    val_dataset = ActivePlayerDataset(
        RAW_DATA_DIR / "train.csv",
        RAW_DATA_DIR / "train_images",
        image_ids=val_ids,
        image_size=args.image_size,
        train=False,
    )
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    model = ActivePlayerCNN().to(device)
    params = count_parameters(model)
    print(f"Device: {device}")
    print(f"ActivePlayerCNN parameters: {params:,}")
    if params >= PARAMETER_LIMIT:
        raise SystemExit(f"Model exceeds {PARAMETER_LIMIT:,} parameters")

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    run = maybe_init_wandb(args, "train-active-player", {**vars(args), "parameters": params})
    best_val = float("inf")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = run_epoch(
            model, train_loader, criterion, optimizer, device, True, f"epoch {epoch:03d}/{args.epochs} train"
        )
        val_loss, val_acc = run_epoch(
            model, val_loader, criterion, optimizer, device, False, f"epoch {epoch:03d}/{args.epochs} val"
        )
        print(
            f"epoch={epoch:03d} train_loss={train_loss:.5f} train_acc={train_acc:.3f} "
            f"val_loss={val_loss:.5f} val_acc={val_acc:.3f}"
        )
        maybe_log_wandb(
            run,
            {
                "epoch": epoch,
                "train/loss": train_loss,
                "train/accuracy": train_acc,
                "val/loss": val_loss,
                "val/accuracy": val_acc,
            },
        )
        if val_loss < best_val:
            best_val = val_loss
            torch.save({"model_state": model.state_dict(), "image_size": args.image_size}, args.output)
            print(f"saved {args.output}")
    if run is not None:
        run.finish()


if __name__ == "__main__":
    main()
