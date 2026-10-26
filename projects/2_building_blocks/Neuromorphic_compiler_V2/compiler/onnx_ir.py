from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Tuple

import torch
import torch.nn as nn

from utils.checkpoints import CheckpointLogger


@dataclass
class ONNXIR:
    """FX-backed IR container used across compiler passes."""

    bundle: Dict[str, Any]
    model: nn.Module
    example_input: torch.Tensor
    logger: Optional[CheckpointLogger] = None
    calibration_loader: Any = None
    nodes: List[Dict[str, Any]] = field(init=False)
    graph_module: Any = field(init=False)
    named_modules: Dict[str, nn.Module] = field(init=False)
    optimization_report: Dict[str, Any] = field(default_factory=dict)
    weight_stats: Dict[str, Any] = field(default_factory=dict)
    weight_clusters: Dict[str, Any] = field(default_factory=dict)
    quantized_weights: Dict[str, Any] = field(default_factory=dict)
    activation_stats: Dict[str, Any] = field(default_factory=dict)
    activation_quant: Dict[str, Any] = field(default_factory=dict)
    tiles: List[Dict[str, Any]] = field(default_factory=list)
    mapping: List[Dict[str, Any]] = field(default_factory=list)
    maestro_dataflows: List[Dict[str, Any]] = field(default_factory=list)
    maestro_status: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.nodes = self.bundle["nodes"]
        self.graph_module = self.bundle["graph_module"]
        self.named_modules = dict(self.model.named_modules())
        if self.logger is not None:
            self.logger.info(
                name="ir_initialized",
                details="Initialized FX-backed compiler IR.",
                payload={
                    "num_nodes": len(self.nodes),
                    "num_named_modules": len(self.named_modules),
                },
            )

    @property
    def path(self) -> str:
        return self.bundle.get("export_path", "")

    def iter_compute_modules(self) -> Iterable[Tuple[str, nn.Module]]:
        for name, module in self.named_modules.items():
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                yield name, module

    def module_output_shape(self, module_name: str) -> Optional[List[int]]:
        for node in self.nodes:
            if node["op"] == "call_module" and node["target"] == module_name:
                return node.get("shape")
        return None
