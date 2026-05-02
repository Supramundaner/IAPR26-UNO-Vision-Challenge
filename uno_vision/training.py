from __future__ import annotations

import argparse
from typing import Any

import torch
from torch import nn


def add_common_training_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--device", default="cuda:0", help="Torch device, for example cuda:0 or cpu")
    parser.add_argument("--wandb", action="store_true", help="Enable Weights & Biases logging")
    parser.add_argument("--wandb-project", default="iapr26-uno-vision", help="W&B project name")
    parser.add_argument("--wandb-run-name", default=None, help="Optional W&B run name")


def resolve_device(device_name: str) -> torch.device:
    if device_name.startswith("cuda") and not torch.cuda.is_available():
        print(f"Requested {device_name}, but CUDA is unavailable. Falling back to CPU.")
        return torch.device("cpu")
    return torch.device(device_name)


def maybe_init_wandb(args: argparse.Namespace, job_type: str, config: dict[str, Any]):
    if not args.wandb:
        return None
    import wandb

    safe_config = {key: str(value) if not isinstance(value, (int, float, str, bool, type(None))) else value for key, value in config.items()}
    return wandb.init(
        project=args.wandb_project,
        name=args.wandb_run_name,
        job_type=job_type,
        config=safe_config,
    )


def maybe_log_wandb(run, metrics: dict[str, Any]) -> None:
    if run is not None:
        run.log(metrics)


class AsymmetricLoss(nn.Module):
    """Asymmetric loss for sparse multi-label classification.

    This is a common replacement for BCE in multi-label settings where easy
    negative labels dominate the gradient.
    """

    def __init__(self, gamma_neg: float = 2.0, gamma_pos: float = 0.0, clip: float = 0.05, eps: float = 1e-8) -> None:
        super().__init__()
        self.gamma_neg = gamma_neg
        self.gamma_pos = gamma_pos
        self.clip = clip
        self.eps = eps

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        prob_pos = torch.sigmoid(logits)
        prob_neg = 1.0 - prob_pos
        if self.clip > 0:
            prob_neg = (prob_neg + self.clip).clamp(max=1.0)

        loss_pos = targets * torch.log(prob_pos.clamp(min=self.eps))
        loss_neg = (1.0 - targets) * torch.log(prob_neg.clamp(min=self.eps))

        if self.gamma_neg > 0 or self.gamma_pos > 0:
            pt = prob_pos * targets + prob_neg * (1.0 - targets)
            gamma = self.gamma_pos * targets + self.gamma_neg * (1.0 - targets)
            weight = torch.pow(1.0 - pt, gamma)
            loss_pos = loss_pos * weight
            loss_neg = loss_neg * weight

        return -(loss_pos + loss_neg)
