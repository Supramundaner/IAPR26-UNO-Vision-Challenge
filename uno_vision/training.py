from __future__ import annotations

import argparse
from typing import Any

import torch


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
