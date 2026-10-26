from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn
from torch.fx import symbolic_trace
from torch.fx.passes.shape_prop import ShapeProp

from utils.checkpoints import CheckpointLogger


def _node_shape(node: Any) -> Optional[List[int]]:
    tensor_meta = node.meta.get("tensor_meta")
    if tensor_meta is None:
        return None
    return list(tensor_meta.shape)


def _try_export_onnx(model: nn.Module, example_input: torch.Tensor, onnx_path: Path) -> Dict[str, Any]:
    status: Dict[str, Any] = {
        "requested_path": str(onnx_path),
        "status": "not_generated",
        "reason": "unknown",
    }
    try:
        # The modern exporter may require onnx/onnxscript. We attempt the export
        # and surface the exact reason on failure.
        torch.onnx.export(
            model.eval(),
            example_input,
            str(onnx_path),
            input_names=["input"],
            output_names=["output"],
            opset_version=17,
        )
        status.update({"status": "generated", "reason": "export_ok"})
    except Exception as exc:  # pragma: no cover - depends on environment
        status.update({"status": "not_generated", "reason": str(exc)})
    return status


def simplify(
    model: nn.Module,
    example_input: torch.Tensor,
    export_path: str | Path,
    logger: Optional[CheckpointLogger] = None,
) -> Dict[str, Any]:
    """Capture a compiler graph and export JSON, with optional ONNX export.

    The function name remains `simplify` to preserve the requested compiler lane
    naming. In this runtime the executable path is `torch.fx` + JSON, while ONNX
    export is attempted opportunistically and its status is written to disk.
    """
    traced = symbolic_trace(model.eval())
    ShapeProp(traced).propagate(example_input)

    nodes: List[Dict[str, Any]] = []
    for node in traced.graph.nodes:
        nodes.append(
            {
                "name": node.name,
                "op": node.op,
                "target": str(node.target),
                "shape": _node_shape(node),
                "users": sorted(user.name for user in node.users),
            }
        )

    export_path = Path(export_path)
    export_path.parent.mkdir(parents=True, exist_ok=True)
    export_path.write_text(json.dumps({"nodes": nodes}, indent=2))

    # Also emit a generic compiler JSON graph name for repo clarity.
    compiler_graph_path = export_path.parent / "compiler_graph.json"
    compiler_graph_path.write_text(json.dumps({"nodes": nodes}, indent=2))

    onnx_path = export_path.parent / "model.onnx"
    onnx_status = _try_export_onnx(model, example_input, onnx_path)
    onnx_status_path = export_path.parent / "onnx_export_status.json"
    onnx_status_path.write_text(json.dumps(onnx_status, indent=2, sort_keys=True))

    bundle = {
        "graph_module": traced,
        "nodes": nodes,
        "export_path": str(export_path),
        "compiler_graph_path": str(compiler_graph_path),
        "onnx_path": str(onnx_path) if onnx_status.get("status") == "generated" else None,
        "onnx_status_path": str(onnx_status_path),
        "onnx_status": onnx_status,
    }
    if logger is not None:
        logger.check(
            name="fx_capture_nonempty",
            condition=len(nodes) > 0,
            details="Captured a real compiler graph from the trained model.",
            payload={"num_nodes": len(nodes), "json_export_path": str(export_path)},
        )
        logger.info(
            name="onnx_export_attempted",
            details="Attempted ONNX export and wrote status manifest.",
            payload={
                "status": onnx_status.get("status"),
                "onnx_status_path": str(onnx_status_path),
                "requested_path": str(onnx_path),
            },
        )
    return bundle
