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
from uno_vision.training import AsymmetricLoss, add_common_training_args, maybe_init_wandb, maybe_log_wandb, resolve_device


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--image-size", default="384", help="Square resize size, or 'original'")
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--center-loss-weight", type=float, default=0.1)
    parser.add_argument("--hand-loss-weight", type=float, default=2.0)
    parser.add_argument("--empty-loss-weight", type=float, default=0.5)
    parser.add_argument("--hand-loss-type", choices=["asl", "bce"], default="asl")
    parser.add_argument("--asl-gamma-neg", type=float, default=2.0)
    parser.add_argument("--asl-gamma-pos", type=float, default=0.0)
    parser.add_argument("--asl-clip", type=float, default=0.05)
    parser.add_argument("--metric-threshold", type=float, default=0.5)
    parser.add_argument("--metric-empty-threshold", type=float, default=0.5)
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--output", type=Path, default=ARTIFACT_DIR / "card_model.pt")
    parser.add_argument("--crop-by-token", action="store_true", help="Enable token-specific ROI crops")
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
    center_count = 0.0
    hand_count = 0.0
    empty_count = 0.0
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
        batch_center_count = float(center_masks.sum().item())
        batch_hand_count = float(hand_card_masks.sum().item())
        batch_empty_count = float(empty_masks.sum().item())
        total_center_loss += float(center_loss.item()) * batch_center_count
        total_hand_loss += float(hand_loss.item()) * batch_hand_count
        total_empty_loss += float(empty_loss.item()) * batch_empty_count
        center_count += batch_center_count
        hand_count += batch_hand_count
        empty_count += batch_empty_count
        total_count += images.size(0)
        progress.set_postfix(
            loss=f"{float(loss.item()):.4f}",
            center=f"{float(center_loss.item()):.4f}",
            hand=f"{float(hand_loss.item()):.4f}",
            empty=f"{float(empty_loss.item()):.4f}",
        )
    denominator = max(1, total_count)
    return (
        total_loss / denominator,
        total_center_loss / max(1.0, center_count),
        total_hand_loss / max(1.0, hand_count),
        total_empty_loss / max(1.0, empty_count),
    )


def multiset_f1_from_counts(pred_count: int, target_count: int, true_positive: int) -> float:
    false_positive = pred_count - true_positive
    false_negative = target_count - true_positive
    denominator = 2 * true_positive + false_positive + false_negative
    if denominator == 0:
        return 1.0
    return 2 * true_positive / denominator


def evaluate_validation_metrics(
    model,
    dataset,
    loader,
    device: torch.device,
    threshold: float,
    empty_threshold: float,
) -> dict[str, float]:
    model.eval()
    sample_index = 0
    center_correct = 0
    center_total = 0
    empty_correct = 0
    empty_total = 0
    hand_f1_sum = 0.0
    hand_total = 0
    predicted_cards_sum = 0
    target_cards_sum = 0

    with torch.no_grad():
        for images, conditions, card_targets, center_targets, empty_targets, _, _, _ in loader:
            batch_size = images.size(0)
            images = images.to(device)
            conditions = conditions.to(device)
            card_logits, empty_logits = model(images, conditions)
            card_probs = torch.sigmoid(card_logits).cpu()
            empty_probs = torch.sigmoid(empty_logits).cpu()

            for row in range(batch_size):
                sample = dataset.samples[sample_index]
                sample_index += 1
                if sample.is_center:
                    center_total += 1
                    center_correct += int(int(card_logits[row].argmax().item()) == int(center_targets[row].item()))
                    continue

                predicted_empty = float(empty_probs[row].item()) >= empty_threshold
                target_empty = bool(empty_targets[row].item() >= 0.5)
                empty_total += 1
                empty_correct += int(predicted_empty == target_empty)

                if predicted_empty:
                    pred_indices: set[int] = set()
                else:
                    pred_indices = {idx for idx, prob in enumerate(card_probs[row].tolist()) if prob >= threshold}
                target_indices = {idx for idx, value in enumerate(card_targets[row].tolist()) if value >= 0.5}
                true_positive = len(pred_indices & target_indices)
                hand_f1_sum += multiset_f1_from_counts(len(pred_indices), len(target_indices), true_positive)
                hand_total += 1
                predicted_cards_sum += len(pred_indices)
                target_cards_sum += len(target_indices)

    return {
        "val/center_accuracy": center_correct / max(1, center_total),
        "val/empty_accuracy": empty_correct / max(1, empty_total),
        "val/hand_f1": hand_f1_sum / max(1, hand_total),
        "val/avg_predicted_cards_per_hand": predicted_cards_sum / max(1, hand_total),
        "val/avg_target_cards_per_hand": target_cards_sum / max(1, hand_total),
    }


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
        crop_by_token=args.crop_by_token,
    )
    val_dataset = ConditionalCardDataset(
        RAW_DATA_DIR / "train.csv",
        RAW_DATA_DIR / "train_images",
        encoder,
        image_ids=val_ids,
        image_size=args.image_size,
        train=False,
        crop_by_token=args.crop_by_token,
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
    if args.hand_loss_type == "asl":
        hand_criterion = AsymmetricLoss(gamma_neg=args.asl_gamma_neg, gamma_pos=args.asl_gamma_pos, clip=args.asl_clip)
    else:
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
        metrics = evaluate_validation_metrics(
            model,
            val_dataset,
            val_loader,
            device,
            threshold=args.metric_threshold,
            empty_threshold=args.metric_empty_threshold,
        )
        print(
            "val_metrics "
            f"center_acc={metrics['val/center_accuracy']:.3f} "
            f"empty_acc={metrics['val/empty_accuracy']:.3f} "
            f"hand_f1={metrics['val/hand_f1']:.3f} "
            f"avg_pred_cards={metrics['val/avg_predicted_cards_per_hand']:.2f} "
            f"avg_target_cards={metrics['val/avg_target_cards_per_hand']:.2f}"
        )
        maybe_log_wandb(run, {"epoch": epoch, **metrics})
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
