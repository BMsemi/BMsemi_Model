from __future__ import annotations

from collections import Counter
from typing import Any, Dict, Optional

import torch
import torch.nn as nn
from torch.fx import symbolic_trace

from utils.checkpoints import CheckpointLogger


def get_spec(
    model: nn.Module,
    example_input: torch.Tensor,
    logger: Optional[CheckpointLogger] = None,
) -> Dict[str, Any]:
    """Create a lightweight ExecuTorch-style compile spec from a real model."""
    traced = symbolic_trace(model.eval())
    node_ops = Counter(node.op for node in traced.graph.nodes)
    module_types = Counter(type(module).__name__ for _, module in model.named_modules())
    num_parameters = int(sum(param.numel() for param in model.parameters()))
    spec: Dict[str, Any] = {
        "model_type": model.__class__.__name__,
        "input_shape": list(example_input.shape),
        "num_parameters": num_parameters,
        "fx_node_count": sum(1 for _ in traced.graph.nodes),
        "node_ops": dict(node_ops),
        "module_types": dict(module_types),
        "precision": "fp32",
        "target_backend": "neuromorphic_x1",
    }
    if logger is not None:
        logger.check(
            name="executorch_spec_has_nodes",
            condition=spec["fx_node_count"] > 0,
            details="Generated a compile spec from traced graph.",
            payload={"fx_node_count": spec["fx_node_count"], "num_parameters": num_parameters},
        )
    return spec
