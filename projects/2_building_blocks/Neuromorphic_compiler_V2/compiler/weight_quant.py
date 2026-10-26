from __future__ import annotations

from typing import Dict

import torch

from .onnx_ir import ONNXIR


def quantize(ir: ONNXIR) -> ONNXIR:
    quantized: Dict[str, Dict[str, float]] = {}
    for name, module in ir.iter_compute_modules():
        weight = module.weight.detach().cpu().float()
        max_abs = float(weight.abs().max().item())
        scale = max(max_abs / 127.0, 1e-8)
        q = torch.clamp(torch.round(weight / scale), -127, 127).to(torch.int8)
        dequant = q.float() * scale
        mse = float(torch.mean((weight - dequant) ** 2).item())
        quantized[name] = {
            "scale": scale,
            "mse": mse,
            "max_abs": max_abs,
            "min_int8": int(q.min().item()),
            "max_int8": int(q.max().item()),
        }
        if ir.logger is not None:
            ir.logger.check(
                name=f"weight_quant_valid::{name}",
                condition=scale > 0 and quantized[name]["min_int8"] >= -127 and quantized[name]["max_int8"] <= 127,
                details=f"Symmetric int8 quantization computed for {name}.",
                payload=quantized[name],
            )
    ir.quantized_weights = quantized
    return ir
