from __future__ import annotations

from typing import Any, Dict, List

import torch.nn as nn

from .onnx_ir import ONNXIR


def optimize(ir: ONNXIR) -> ONNXIR:
    module_lookup = ir.named_modules
    call_modules = [node for node in ir.nodes if node["op"] == "call_module"]
    compute_count = 0
    fuse_triplets = 0
    for index, node in enumerate(call_modules):
        module = module_lookup.get(node["target"])
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            compute_count += 1
        if index + 2 < len(call_modules):
            m0 = module_lookup.get(call_modules[index]["target"])
            m1 = module_lookup.get(call_modules[index + 1]["target"])
            m2 = module_lookup.get(call_modules[index + 2]["target"])
            if isinstance(m0, nn.Conv2d) and isinstance(m1, nn.BatchNorm2d) and isinstance(m2, nn.ReLU6):
                fuse_triplets += 1

    ir.optimization_report = {
        "call_module_nodes": len(call_modules),
        "compute_nodes": compute_count,
        "conv_bn_relu_triplets": fuse_triplets,
        "note": "Analysis-only optimization report; no graph rewrite applied.",
    }
    if ir.logger is not None:
        ir.logger.check(
            name="graph_optimizer_compute_nodes_present",
            condition=compute_count > 0,
            details="Graph contains real compute nodes ready for downstream passes.",
            payload=ir.optimization_report,
        )
    return ir
