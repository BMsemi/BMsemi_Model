"""Compiler prototype for the ZCU106 X2 INT8 MVM accelerator."""

from .compiler import X2Compiler
from .hybrid_runtime import HybridTFLiteExecutor, HybridResult
from .ir import X2Program, X2Target, X2WeightTile
from .planner import plan_tflite_model
from .postprocess import Detection, decode_yolo
from .runtime import (
    CrossbarX2Backend,
    PEArrayX2Backend,
    PynqX2Backend,
    create_accelerator_backend,
    execute_conv2d,
    execute_matrix,
    public_backend_name,
)
from .tflite_frontend import TFLiteX2Frontend

__all__ = [
    "CrossbarX2Backend",
    "PEArrayX2Backend",
    "PynqX2Backend",
    "create_accelerator_backend",
    "HybridResult",
    "HybridTFLiteExecutor",
    "Detection",
    "TFLiteX2Frontend",
    "X2Compiler",
    "X2Program",
    "X2Target",
    "X2WeightTile",
    "decode_yolo",
    "execute_conv2d",
    "execute_matrix",
    "public_backend_name",
    "plan_tflite_model",
]
