from __future__ import annotations

import torch
from torch import nn


class ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.SiLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class TinyBackbone(nn.Module):
    def __init__(self, channels: tuple[int, ...] = (32, 64, 128, 192, 256)) -> None:
        super().__init__()
        layers = []
        in_channels = 3
        for idx, out_channels in enumerate(channels):
            layers.append(ConvBlock(in_channels, out_channels, stride=2 if idx > 0 else 1))
            in_channels = out_channels
        self.features = nn.Sequential(*layers)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.out_channels = channels[-1]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.pool(x)
        return torch.flatten(x, 1)


class ConditionalCardCNN(nn.Module):
    def __init__(self, num_cards: int, num_conditions: int = 5, condition_dim: int = 32) -> None:
        super().__init__()
        self.backbone = TinyBackbone()
        self.condition_embedding = nn.Embedding(num_conditions, condition_dim)
        self.fusion = nn.Sequential(
            nn.Linear(self.backbone.out_channels + condition_dim, 256),
            nn.SiLU(inplace=True),
            nn.Dropout(0.20),
        )
        self.card_head = nn.Linear(256, num_cards)
        self.empty_head = nn.Linear(256, 1)

    def forward(self, image: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        image_features = self.backbone(image)
        condition_features = self.condition_embedding(condition)
        features = torch.cat([image_features, condition_features], dim=1)
        fused = self.fusion(features)
        return self.card_head(fused), self.empty_head(fused)


class ActivePlayerCNN(nn.Module):
    def __init__(self, num_players: int = 4) -> None:
        super().__init__()
        self.backbone = TinyBackbone(channels=(24, 48, 96, 128))
        self.head = nn.Sequential(
            nn.Linear(self.backbone.out_channels, 128),
            nn.SiLU(inplace=True),
            nn.Dropout(0.15),
            nn.Linear(128, num_players),
        )

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        return self.head(self.backbone(image))


def count_parameters(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
