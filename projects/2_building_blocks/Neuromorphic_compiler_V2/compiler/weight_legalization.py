from __future__ import annotations

from typing import Dict

import torch

from .onnx_ir import ONNXIR


def legalize(ir: ONNXIR) -> ONNXIR:
    stats: Dict[str, Dict[str, float]] = {}
    for name, module in ir.iter_compute_modules():
        weight = module.weight.detach().cpu()
        is_finite = bool(torch.isfinite(weight).all().item())
        stats[name] = {
            "min": float(weight.min().item()),
            "max": float(weight.max().item()),
            "mean_abs": float(weight.abs().mean().item()),
        }
        if ir.logger is not None:
            ir.logger.check(
                name=f"weights_finite::{name}",
                condition=is_finite,
                details=f"Weight tensor is finite for module {name}.",
                payload={"shape": list(weight.shape), **stats[name]},
            )
    ir.weight_stats = stats
    return ir
