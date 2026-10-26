from __future__ import annotations

from collections import defaultdict
from typing import Any, Dict

import torch
import torch.nn as nn

from .onnx_ir import ONNXIR


def quantize(ir: ONNXIR, max_batches: int = 4) -> ONNXIR:
    stats: Dict[str, Dict[str, float]] = defaultdict(lambda: {"min": float("inf"), "max": float("-inf")})
    hooks = []

    def make_hook(name: str):
        def hook(_module: nn.Module, _inputs: Any, output: Any) -> None:
            tensor = output[0] if isinstance(output, (tuple, list)) else output
            if not torch.is_tensor(tensor):
                return
            tensor = tensor.detach().cpu().float()
            stats[name]["min"] = min(stats[name]["min"], float(tensor.min().item()))
            stats[name]["max"] = max(stats[name]["max"], float(tensor.max().item()))
        return hook

    for name, module in ir.model.named_modules():
        if isinstance(module, (nn.Conv2d, nn.Linear, nn.ReLU6, nn.BatchNorm2d)):
            hooks.append(module.register_forward_hook(make_hook(name)))

    ir.model.eval()
    with torch.no_grad():
        for batch_idx, (inputs, _) in enumerate(ir.calibration_loader):
            if batch_idx >= max_batches:
                break
            ir.model(inputs)

    for hook in hooks:
        hook.remove()

    activation_quant: Dict[str, Dict[str, float]] = {}
    for name, values in sorted(stats.items()):
        max_abs = max(abs(values["min"]), abs(values["max"]))
        scale = max(max_abs / 127.0, 1e-8)
        activation_quant[name] = {"min": values["min"], "max": values["max"], "scale": scale}
        if ir.logger is not None:
            ir.logger.check(
                name=f"activation_quant_valid::{name}",
                condition=values["min"] <= values["max"] and scale > 0,
                details=f"Collected activation range for {name}.",
                payload=activation_quant[name],
            )
    ir.activation_stats = dict(stats)
    ir.activation_quant = activation_quant
    return ir
