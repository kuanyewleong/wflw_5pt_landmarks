from __future__ import annotations

import torch
import torch.nn as nn


class ConvBNAct(nn.Module):
    def __init__(self, c1: int, c2: int, k: int = 3, s: int = 1, p: int | None = None):
        super().__init__()
        if p is None:
            p = k // 2
        self.block = nn.Sequential(
            nn.Conv2d(c1, c2, kernel_size=k, stride=s, padding=p, bias=False),
            nn.BatchNorm2d(c2),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class LandmarkNet5(nn.Module):
    def __init__(self, num_points: int = 5):
        super().__init__()
        self.features = nn.Sequential(
            ConvBNAct(3, 32, 3, 2),
            ConvBNAct(32, 32, 3, 1),
            ConvBNAct(32, 64, 3, 2),
            ConvBNAct(64, 64, 3, 1),
            ConvBNAct(64, 128, 3, 2),
            ConvBNAct(128, 128, 3, 1),
            ConvBNAct(128, 192, 3, 2),
            ConvBNAct(192, 192, 3, 1),
            ConvBNAct(192, 256, 3, 2),
            ConvBNAct(256, 256, 3, 1),
        )
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(256, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(256, num_points * 2),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.features(x))