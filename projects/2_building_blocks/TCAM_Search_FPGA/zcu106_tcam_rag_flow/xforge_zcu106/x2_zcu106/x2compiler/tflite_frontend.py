from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from .compiler import X2Compiler
from .ir import X2Program


def _tensor_metadata(detail: dict[str, Any]) -> dict[str, Any]:
    quant = detail.get("quantization_parameters", {})
    return {
        "index": int(detail["index"]),
        "name": str(detail["name"]),
        "shape": [int(value) for value in detail["shape"]],
        "dtype": str(np.dtype(detail["dtype"])),
        "scale": float((detail.get("quantization") or (0.0, 0))[0]),
        "zero_point": int((detail.get("quantization") or (0.0, 0))[1]),
        "scales": np.asarray(quant.get("scales", []), dtype=np.float32).tolist(),
        "zero_points": np.asarray(quant.get("zero_points", []), dtype=np.int64).tolist(),
        "quantized_dimension": int(quant.get("quantized_dimension", 0)),
    }


def _infer_conv_geometry(
    input_shape: tuple[int, int, int, int],
    weight_shape: tuple[int, int, int, int],
    output_shape: tuple[int, int, int, int],
) -> tuple[tuple[int, int], str]:
    _, input_h, input_w, _ = input_shape
    _, kernel_h, kernel_w, _ = weight_shape
    _, output_h, output_w, _ = output_shape
    candidates: list[tuple[tuple[int, int], str]] = []
    for stride_y in range(1, 9):
        for stride_x in range(1, 9):
            same_h = (input_h + stride_y - 1) // stride_y
            same_w = (input_w + stride_x - 1) // stride_x
            if (same_h, same_w) == (output_h, output_w):
                candidates.append(((stride_y, stride_x), "same"))
            if input_h >= kernel_h and input_w >= kernel_w:
                valid_h = (input_h - kernel_h) // stride_y + 1
                valid_w = (input_w - kernel_w) // stride_x + 1
                if (valid_h, valid_w) == (output_h, output_w):
                    candidates.append(((stride_y, stride_x), "valid"))
    if not candidates:
        raise ValueError(
            "Could not infer Conv2D stride/padding from "
            f"input={input_shape}, weights={weight_shape}, output={output_shape}."
        )
    # SAME and VALID are equivalent for 1x1 kernels with no border effect.
    # Prefer VALID when both match because it avoids introducing pad work.
    candidates.sort(key=lambda item: (item[1] != "valid", item[0]))
    return candidates[0]


class TFLiteX2Frontend:
    """Import TFLite Conv2D constants and geometry into X2 programs."""

    def __init__(self, model_path: str | Path, *, compiler: X2Compiler | None = None) -> None:
        import ai_edge_litert.interpreter as litert

        self.model_path = Path(model_path).resolve()
        self.compiler = compiler or X2Compiler()
        kwargs: dict[str, Any] = {
            "model_path": str(self.model_path),
            "experimental_preserve_all_tensors": True,
        }
        resolver = getattr(litert, "OpResolverType", None)
        if resolver is not None:
            kwargs["experimental_op_resolver_type"] = resolver.BUILTIN_WITHOUT_DEFAULT_DELEGATES
        self.interpreter = litert.Interpreter(**kwargs)
        self.interpreter.allocate_tensors()
        self.ops = list(self.interpreter._get_ops_details())  # noqa: SLF001
        self.tensors = {
            int(detail["index"]): detail
            for detail in self.interpreter.get_tensor_details()
        }

    def operator_report(self) -> list[dict[str, Any]]:
        report: list[dict[str, Any]] = []
        for op in self.ops:
            op_name = str(op["op_name"])
            inputs = [int(index) for index in op.get("inputs", [])]
            outputs = [int(index) for index in op.get("outputs", [])]
            entry: dict[str, Any] = {
                "index": int(op["index"]),
                "op_name": op_name,
                "inputs": inputs,
                "outputs": outputs,
                "status": "cpu_fallback",
                "reason": "The first compiler slice targets Conv2D only.",
            }
            if op_name == "CONV_2D":
                try:
                    self._validate_conv(op)
                    entry["status"] = "x2_prototype"
                    input_dtype = np.dtype(self.tensors[inputs[0]]["dtype"])
                    entry["reason"] = (
                        "Native signed INT8 activation."
                        if input_dtype == np.dtype(np.int8)
                        else "Float activation requires the signed INT8 runtime bridge."
                    )
                except Exception as exc:
                    entry["reason"] = str(exc)
            report.append(entry)
        return report

    def _find_op(self, op_index: int) -> dict[str, Any]:
        for op in self.ops:
            if int(op["index"]) == int(op_index):
                return op
        raise KeyError(f"TFLite operator {op_index} was not found.")

    def _validate_conv(self, op: dict[str, Any]) -> None:
        if str(op["op_name"]) != "CONV_2D":
            raise ValueError(f"Operator {op['index']} is {op['op_name']}, not CONV_2D.")
        inputs = [int(index) for index in op.get("inputs", [])]
        outputs = [int(index) for index in op.get("outputs", [])]
        if len(inputs) < 2 or len(outputs) != 1:
            raise ValueError("Conv2D must expose activation, weight, and one output tensor.")
        input_detail = self.tensors[inputs[0]]
        weight_detail = self.tensors[inputs[1]]
        output_detail = self.tensors[outputs[0]]
        if len(input_detail["shape"]) != 4 or len(output_detail["shape"]) != 4:
            raise ValueError("Only rank-four NHWC Conv2D tensors are supported.")
        if np.dtype(weight_detail["dtype"]) != np.dtype(np.int8):
            raise ValueError(f"X2 requires int8 weights, got {weight_detail['dtype']}.")
        if len(weight_detail["shape"]) != 4:
            raise ValueError("TFLite Conv2D weights must be OHWI.")
        input_dtype = np.dtype(input_detail["dtype"])
        if input_dtype not in {np.dtype(np.float32), np.dtype(np.int8)}:
            raise ValueError(f"Prototype activation dtype is unsupported: {input_dtype}.")
        weight_quant = weight_detail.get("quantization_parameters", {})
        zero_points = np.asarray(weight_quant.get("zero_points", []), dtype=np.int64)
        if zero_points.size and np.any(zero_points != 0):
            raise ValueError("X2 prototype requires symmetric int8 weights with zero point 0.")
        scales = np.asarray(weight_quant.get("scales", []), dtype=np.float32)
        if scales.size == 0 or np.any(scales <= 0):
            raise ValueError("Conv2D weight scales are missing or invalid.")
        _infer_conv_geometry(
            tuple(int(value) for value in input_detail["shape"]),
            tuple(int(value) for value in weight_detail["shape"]),
            tuple(int(value) for value in output_detail["shape"]),
        )

    def compile_conv(
        self,
        op_index: int,
        *,
        activation_scale: float | None = None,
        allow_saturation: bool = False,
    ) -> X2Program:
        op = self._find_op(op_index)
        self._validate_conv(op)
        inputs = [int(index) for index in op.get("inputs", [])]
        output_index = int(op.get("outputs", [])[0])
        input_detail = self.tensors[inputs[0]]
        weight_detail = self.tensors[inputs[1]]
        output_detail = self.tensors[output_index]
        bias_detail = self.tensors[inputs[2]] if len(inputs) > 2 and inputs[2] >= 0 else None

        weights = np.array(self.interpreter.get_tensor(inputs[1]), copy=True)
        bias = (
            np.array(self.interpreter.get_tensor(inputs[2]), copy=True)
            if bias_detail is not None
            else None
        )
        weight_quant = weight_detail.get("quantization_parameters", {})
        weight_scales = np.asarray(weight_quant.get("scales", []), dtype=np.float32)
        input_shape = tuple(int(value) for value in input_detail["shape"])
        output_shape = tuple(int(value) for value in output_detail["shape"])
        weight_shape = tuple(int(value) for value in weight_detail["shape"])
        stride, padding = _infer_conv_geometry(input_shape, weight_shape, output_shape)

        input_dtype = np.dtype(input_detail["dtype"])
        input_scale, input_zero_point = input_detail.get("quantization") or (0.0, 0)
        if input_dtype == np.dtype(np.float32):
            quant_mode = "static_symmetric" if activation_scale else "dynamic_symmetric"
            scale = float(activation_scale or 0.0)
            zero_point = 0
        else:
            quant_mode = "tflite_integer"
            scale = float(input_scale)
            zero_point = int(input_zero_point)
            if scale <= 0:
                raise ValueError("Integer activation tensor is missing its scale.")

        output_scale, output_zero_point = output_detail.get("quantization") or (0.0, 0)
        metadata = {
            "source": {
                "format": "tflite",
                "model": str(self.model_path),
                "op_index": int(op_index),
                "op_name": "CONV_2D",
                "activation_tensor": _tensor_metadata(input_detail),
                "weight_tensor": _tensor_metadata(weight_detail),
                "bias_tensor": _tensor_metadata(bias_detail) if bias_detail is not None else None,
                "output_tensor": _tensor_metadata(output_detail),
            },
            "activation_quantization": {
                "mode": quant_mode,
                "scale": scale,
                "zero_point": zero_point,
                "allow_saturation": bool(allow_saturation),
            },
        }
        output_dtype = np.dtype(output_detail["dtype"])
        if np.issubdtype(output_dtype, np.integer) and float(output_scale) > 0:
            metadata["output_quantization"] = {
                "scale": float(output_scale),
                "zero_point": int(output_zero_point),
                "dtype": str(output_dtype),
            }

        bias_i32 = bias if bias is not None and bias.dtype == np.dtype(np.int32) else None
        bias_float = bias if bias is not None and np.issubdtype(bias.dtype, np.floating) else None
        return self.compiler.compile_conv2d(
            weights,
            input_shape=input_shape,
            output_shape=output_shape,
            stride=stride,
            padding=padding,
            name=f"tflite_conv2d_op_{int(op_index)}",
            bias_i32=bias_i32,
            bias_float=bias_float,
            weight_scales=weight_scales,
            metadata=metadata,
        )

    def compile_model(
        self,
        output_directory: str | Path,
        *,
        activation_scale_overrides: dict[int, float] | None = None,
        allow_saturation: bool = False,
    ) -> dict[str, Any]:
        output = Path(output_directory)
        output.mkdir(parents=True, exist_ok=True)
        overrides = {int(key): float(value) for key, value in (activation_scale_overrides or {}).items()}
        operator_report = self.operator_report()
        compiled: list[dict[str, Any]] = []
        errors: list[dict[str, Any]] = []
        for entry in operator_report:
            if entry["status"] != "x2_prototype":
                continue
            op_index = int(entry["index"])
            try:
                program = self.compile_conv(
                    op_index,
                    activation_scale=overrides.get(op_index),
                    allow_saturation=allow_saturation,
                )
                layer_dir = output / f"op_{op_index:04d}_conv2d"
                program.save(layer_dir)
                compiled.append(
                    {
                        "op_index": op_index,
                        "program": str(layer_dir.relative_to(output)),
                        "tiles": len(program.tiles),
                        "logical_shape": [program.m_dim, program.k_dim, program.n_dim],
                    }
                )
            except Exception as exc:
                errors.append(
                    {
                        "op_index": op_index,
                        "error_type": type(exc).__name__,
                        "error": str(exc),
                    }
                )

        model_manifest = {
            "schema_version": 1,
            "source_model": str(self.model_path),
            "compiled_conv2d_count": len(compiled),
            "compile_error_count": len(errors),
            "compiled_programs": compiled,
            "compile_errors": errors,
            "operators": operator_report,
            "fallback_policy": (
                "All non-Conv2D operators and any rejected Conv2D remain on the CPU. "
                "Float Conv2D activations use the explicit signed INT8 bridge."
            ),
        }
        (output / "model_manifest.json").write_text(
            json.dumps(model_manifest, indent=2),
            encoding="utf-8",
        )
        return model_manifest


TFLiteFrontend = TFLiteX2Frontend
