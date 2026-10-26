from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional

import torch
import torch.nn as nn

from utils.checkpoints import CheckpointLogger


def export_quantized_tflite(
    model: nn.Module,
    example_input: torch.Tensor,
    export_dir: str | Path,
    logger: Optional[CheckpointLogger] = None,
) -> Dict[str, Any]:
    """Attempt to prepare a TFLite deployment artifact.

    This repository does not include a full PyTorch->TFLite conversion bridge.
    The function therefore emits a machine-readable status manifest that makes
    the current limitation explicit. If a suitable converter is available in a
    downstream environment, this function is the right integration point.
    """
    export_dir = Path(export_dir)
    export_dir.mkdir(parents=True, exist_ok=True)
    status_path = export_dir / "tflite_export_status.json"
    tflite_path = export_dir / "model_int8.tflite"

    status: Dict[str, Any] = {
        "requested_output": str(tflite_path),
        "status": "not_generated",
        "reason": "No supported PyTorch->TFLite converter is installed in this runtime.",
        "input_shape": list(example_input.shape),
        "model_name": model.__class__.__name__,
        "suggested_paths": {
            "option_1": "Export ONNX first, then convert in a TensorFlow/TFLite environment.",
            "option_2": "Rebuild the model in TensorFlow and run full integer quantization there.",
            "option_3": "Use a dedicated bridge such as ai-edge/onnx-tf in a prepared environment.",
        },
    }

    try:
        import tensorflow  # type: ignore  # noqa: F401

        status["tensorflow_present"] = True
        status["reason"] = (
            "TensorFlow is installed, but this repo does not implement an automatic "
            "PyTorch->TFLite bridge. Provide ONNX or a TF graph in a downstream environment."
        )
    except Exception as exc:  # pragma: no cover - depends on environment
        status["tensorflow_present"] = False
        status["import_error"] = str(exc)

    status_path.write_text(json.dumps(status, indent=2, sort_keys=True))
    if logger is not None:
        logger.info(
            name="tflite_export_status_written",
            details="Wrote TFLite deployment status manifest.",
            payload={"status_path": str(status_path), "status": status["status"]},
        )
    return status
