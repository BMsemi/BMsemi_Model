from __future__ import annotations

import ast
import json
from pathlib import Path
from typing import Any, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


def _load_structured_text(path: Path) -> Any:
    """Load a Python/JSON literal stored in a text file.

    The YOLO one-layer example stores weights, bias, and padding as plain-text
    nested lists. This helper accepts either strict JSON or Python-literal style
    formatting and returns the parsed object.
    """
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        raise ValueError(f"File is empty: {path}")
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return ast.literal_eval(text)



def load_layer0_weights(source_dir: Path) -> torch.Tensor:
    """Load YOLO layer-0 weights from ``layer0_weights.txt``.

    Expected shape is ``(16, 3, 3, 3)`` corresponding to
    ``(out_channels, in_channels, kernel_h, kernel_w)``.
    """
    path = source_dir / "layer0_weights.txt"
    weights = torch.tensor(_load_structured_text(path), dtype=torch.float32)
    if tuple(weights.shape) != (16, 3, 3, 3):
        raise ValueError(
            f"Unexpected YOLO layer-0 weight shape {tuple(weights.shape)}. "
            "Expected (16, 3, 3, 3)."
        )
    return weights



def load_layer0_bias(source_dir: Path) -> torch.Tensor:
    """Load YOLO layer-0 bias from ``bias.txt``."""
    path = source_dir / "bias.txt"
    bias = torch.tensor(_load_structured_text(path), dtype=torch.float32)
    if tuple(bias.shape) != (16,):
        raise ValueError(f"Unexpected YOLO layer-0 bias shape {tuple(bias.shape)}. Expected (16,).")
    return bias



def load_layer0_padding(source_dir: Path) -> Tuple[int, int, int, int]:
    """Load explicit padding from ``Layer0_padding.txt``.

    The original YOLO example uses a 4x2 list:
      [[n0,n1], [top,bottom], [left,right], [c0,c1]]

    Only the spatial padding terms are needed here.
    """
    path = source_dir / "Layer0_padding.txt"
    raw = _load_structured_text(path)
    if not isinstance(raw, list) or len(raw) != 4:
        raise ValueError(f"Unexpected padding format in {path}. Expected a 4-entry list.")
    top, bottom = raw[1]
    left, right = raw[2]
    return int(top), int(bottom), int(left), int(right)


class YoloLayer0(nn.Module):
    """PyTorch wrapper for the YOLO first convolution layer.

    This is a minimal model-layer description that fits directly into the
    existing Neuromorphic Compiler V2 infrastructure.  It keeps the original
    weights, bias, stride, and explicit padding from the YOLO one-layer example
    while exposing a standard ``nn.Module`` interface to the rest of the X1
    pipeline.
    """

    def __init__(self, source_dir: str | Path) -> None:
        super().__init__()
        self.source_dir = Path(source_dir)
        if not self.source_dir.exists():
            raise FileNotFoundError(f"YOLO source directory not found: {self.source_dir}")

        weights = load_layer0_weights(self.source_dir)
        bias = load_layer0_bias(self.source_dir)
        self.padding = load_layer0_padding(self.source_dir)

        self.conv = nn.Conv2d(
            in_channels=3,
            out_channels=16,
            kernel_size=(3, 3),
            stride=(2, 2),
            padding=(0, 0),
            bias=True,
        )
        with torch.no_grad():
            self.conv.weight.copy_(weights)
            self.conv.bias.copy_(bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        top, bottom, left, right = self.padding
        x = F.pad(x, (left, right, top, bottom))
        return self.conv(x)
