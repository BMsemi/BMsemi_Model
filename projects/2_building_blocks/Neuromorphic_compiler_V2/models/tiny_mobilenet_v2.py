from __future__ import annotations

import torch
import torch.nn as nn


class ConvBNAct(nn.Sequential):
    def __init__(self, inp: int, oup: int, kernel: int = 3, stride: int = 1, groups: int = 1) -> None:
        padding = (kernel - 1) // 2
        super().__init__(
            nn.Conv2d(inp, oup, kernel, stride, padding, groups=groups, bias=False),
            nn.BatchNorm2d(oup),
            nn.ReLU6(inplace=True),
        )


class InvertedResidual(nn.Module):
    def __init__(self, inp: int, oup: int, stride: int, expand_ratio: int) -> None:
        super().__init__()
        hidden_dim = int(round(inp * expand_ratio))
        self.use_res_connect = stride == 1 and inp == oup

        layers = []
        if expand_ratio != 1:
            layers.append(ConvBNAct(inp, hidden_dim, kernel=1))
        layers.extend(
            [
                ConvBNAct(hidden_dim, hidden_dim, stride=stride, groups=hidden_dim),
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
            ]
        )
        self.conv = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.conv(x)
        if self.use_res_connect:
            return x + out
        return out


class TinyMobileNetV2(nn.Module):
    """A compact MobileNetV2-style model for small grayscale images."""

    def __init__(self, num_classes: int = 10, width_mult: float = 0.5, in_channels: int = 1) -> None:
        super().__init__()

        def adjust(channels: int) -> int:
            return max(8, int(channels * width_mult))

        self.features = nn.Sequential(
            ConvBNAct(in_channels, adjust(16), stride=2),
            InvertedResidual(adjust(16), adjust(16), 1, 1),
            InvertedResidual(adjust(16), adjust(24), 2, 2),
            InvertedResidual(adjust(24), adjust(24), 1, 2),
            InvertedResidual(adjust(24), adjust(32), 2, 2),
            InvertedResidual(adjust(32), adjust(32), 1, 2),
            ConvBNAct(adjust(32), adjust(64), kernel=1),
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(adjust(64), num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.pool(x).flatten(1)
        return self.classifier(x)
