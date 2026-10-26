from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass
import json
from pathlib import Path
import time
from typing import Any, Callable, Iterable

import numpy as np

from .ir import X2Program
from .runtime import ExecutionTrace, X2Backend, execute_conv2d
from .tflite_ops import (
    TensorMeta,
    execute_conv2d_cpu,
    execute_depthwise_conv2d,
    execute_digital_op,
    is_quantized,
)


@dataclass(frozen=True)
class OperatorTrace:
    op_index: int
    op_name: str
    backend: str
    input_indices: list[int]
    output_indices: list[int]
    duration_ms: float
    accelerator_calls: int = 0
    accelerator_input_words: int = 0
    accelerator_output_words: int = 0
    logical_macs: int = 0
    activation_scale: float = 0.0
    activation_max_abs: float = 0.0
    saturation_count: int = 0


@dataclass(frozen=True)
class HybridResult:
    output: np.ndarray
    traces: list[OperatorTrace]
    elapsed_ms: float
    peak_cached_bytes: int

    def summary(self) -> dict[str, Any]:
        counts = Counter(trace.backend for trace in self.traces)
        return {
            "operator_count": len(self.traces),
            "backend_counts": dict(sorted(counts.items())),
            "elapsed_ms": self.elapsed_ms,
            "peak_cached_bytes": self.peak_cached_bytes,
            "accelerator_calls": sum(
                trace.accelerator_calls for trace in self.traces
            ),
            "accelerator_input_words": sum(
                trace.accelerator_input_words for trace in self.traces
            ),
            "accelerator_output_words": sum(
                trace.accelerator_output_words for trace in self.traces
            ),
            "logical_macs": sum(trace.logical_macs for trace in self.traces),
            "saturation_count": sum(
                trace.saturation_count for trace in self.traces
            ),
        }


def _interpreter(model_path: Path):
    try:
        import ai_edge_litert.interpreter as litert
    except ImportError:
        try:
            import tflite_runtime.interpreter as litert
        except ImportError:
            from tensorflow.lite.python import interpreter as litert

    kwargs: dict[str, Any] = {
        "model_path": str(model_path),
        "experimental_preserve_all_tensors": True,
    }
    resolver = getattr(litert, "OpResolverType", None)
    if resolver is not None:
        kwargs["experimental_op_resolver_type"] = (
            resolver.BUILTIN_WITHOUT_DEFAULT_DELEGATES
        )
    result = litert.Interpreter(**kwargs)
    result.allocate_tensors()
    return result


def _float_constant(value: np.ndarray, meta: TensorMeta) -> np.ndarray:
    source = np.asarray(value)
    if not np.issubdtype(source.dtype, np.integer):
        return source.astype(np.float32)
    scales = np.asarray(meta.scales, dtype=np.float32).reshape(-1)
    zero_points = np.asarray(meta.zero_points, dtype=np.float32).reshape(-1)
    if scales.size == 0:
        scales = np.asarray([meta.scale], dtype=np.float32)
    if zero_points.size == 0:
        zero_points = np.asarray([meta.zero_point], dtype=np.float32)
    if scales.size == 1:
        return (
            source.astype(np.float32) - zero_points[0]
        ) * scales[0]
    shape = [1] * source.ndim
    shape[meta.quantized_dimension] = scales.size
    return (
        source.astype(np.float32) - zero_points.reshape(shape)
    ) * scales.reshape(shape)


class HybridTFLiteExecutor:
    """
    Strict YOLO TFLite executor with compiled Conv2D dispatch.

    LiteRT is used only to parse the FlatBuffer and read immutable constants.
    Operator outputs are always computed locally; there is no reference-output
    fallback in this class.
    """

    def __init__(
        self,
        model_path: str | Path,
        artifact_directory: str | Path,
        backend: X2Backend,
        *,
        offload_conv_indices: Iterable[int] | None = None,
    ) -> None:
        self.model_path = Path(model_path).resolve()
        self.artifact_directory = Path(artifact_directory).resolve()
        self.backend = backend
        self.interpreter = _interpreter(self.model_path)
        self.ops = list(self.interpreter._get_ops_details())  # noqa: SLF001
        details = self.interpreter.get_tensor_details()
        self.tensor_meta = {
            int(detail["index"]): TensorMeta.from_detail(detail)
            for detail in details
        }
        self.input_details = list(self.interpreter.get_input_details())
        self.output_details = list(self.interpreter.get_output_details())
        if len(self.input_details) != 1 or len(self.output_details) != 1:
            raise ValueError("Prototype runtime expects one model input and output.")

        produced = {
            int(index)
            for op in self.ops
            for index in op.get("outputs", [])
            if int(index) >= 0
        }
        model_inputs = {
            int(detail["index"]) for detail in self.input_details
        }
        self.constant_indices = set(self.tensor_meta) - produced - model_inputs
        self.output_indices = {
            int(detail["index"]) for detail in self.output_details
        }
        self._constant_cache: dict[int, np.ndarray] = {}
        self._program_cache: dict[int, X2Program] = {}
        self._constant_archive: dict[str, np.ndarray] | None = None

        model_manifest = json.loads(
            (self.artifact_directory / "model_manifest.json").read_text(
                encoding="utf-8"
            )
        )
        self.program_paths = {
            int(item["op_index"]): self.artifact_directory / item["program"]
            for item in model_manifest["compiled_programs"]
        }
        if offload_conv_indices is None:
            self.offload_conv_indices = set(self.program_paths)
        else:
            self.offload_conv_indices = {
                int(index) for index in offload_conv_indices
            }
        unknown = self.offload_conv_indices - set(self.program_paths)
        if unknown:
            raise ValueError(
                f"No compiled X2 program for operator(s) {sorted(unknown)}."
            )

        self.last_use: dict[int, int] = {}
        for ordinal, op in enumerate(self.ops):
            for index in op.get("inputs", []):
                if int(index) >= 0:
                    self.last_use[int(index)] = ordinal
        for index in self.output_indices:
            self.last_use[index] = len(self.ops)

    @property
    def input_detail(self) -> dict[str, Any]:
        return self.input_details[0]

    @property
    def output_detail(self) -> dict[str, Any]:
        return self.output_details[0]

    def _constant(self, index: int) -> np.ndarray:
        if index not in self.constant_indices:
            raise KeyError(f"Tensor {index} is not an immutable graph constant.")
        if index not in self._constant_cache:
            try:
                value = self.interpreter.get_tensor(index)
            except ValueError as exc:
                if "allocate_tensors" not in str(exc):
                    raise
                self.interpreter.allocate_tensors()
                try:
                    value = self.interpreter.get_tensor(index)
                except ValueError:
                    value = self._archived_constant(index)
            self._constant_cache[index] = np.array(
                value,
                copy=True,
            )
        return self._constant_cache[index]

    def _archived_constant(self, index: int) -> np.ndarray:
        if self._constant_archive is None:
            archive_path = self.artifact_directory / "model_constants.npz"
            if not archive_path.exists():
                raise FileNotFoundError(
                    f"Missing constant archive: {archive_path}"
                )
            with np.load(archive_path, allow_pickle=False) as archive:
                self._constant_archive = {
                    name: np.array(archive[name], copy=True)
                    for name in archive.files
                }
        key = f"tensor_{int(index)}"
        if key not in self._constant_archive:
            raise KeyError(
                f"Tensor {index} is not present in model_constants.npz."
            )
        return np.array(self._constant_archive[key], copy=True)

    def _program(self, op_index: int) -> X2Program:
        if op_index not in self._program_cache:
            self._program_cache[op_index] = X2Program.load(
                self.program_paths[op_index]
            )
        return self._program_cache[op_index]

    def _value(
        self,
        cache: dict[int, np.ndarray],
        index: int,
    ) -> np.ndarray:
        if index in cache:
            return cache[index]
        return self._constant(index)

    def preprocess_image(
        self,
        image_path: str | Path,
        *,
        mode: str = "resize",
    ) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
        import cv2

        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image is None:
            raise FileNotFoundError(f"Could not load image: {image_path}")
        shape = tuple(int(value) for value in self.input_detail["shape"])
        if len(shape) != 4:
            raise ValueError(f"Expected rank-four input, got {shape}.")
        if shape[1] == 3:
            layout, height, width = "NCHW", shape[2], shape[3]
        elif shape[-1] == 3:
            layout, height, width = "NHWC", shape[1], shape[2]
        else:
            raise ValueError(f"Could not infer RGB layout from {shape}.")

        original_h, original_w = image.shape[:2]
        if mode == "resize":
            prepared = cv2.resize(
                image,
                (width, height),
                interpolation=cv2.INTER_LINEAR,
            )
            metadata: dict[str, Any] = {
                "mode": "resize",
                "original_shape": [original_h, original_w],
                "new_shape": [height, width],
                "gain": None,
                "pad_x": 0.0,
                "pad_y": 0.0,
            }
        elif mode == "letterbox":
            gain = min(width / original_w, height / original_h)
            resized_w = int(round(original_w * gain))
            resized_h = int(round(original_h * gain))
            resized = cv2.resize(
                image,
                (resized_w, resized_h),
                interpolation=cv2.INTER_LINEAR,
            )
            left = int(round((width - resized_w) / 2.0 - 0.1))
            top = int(round((height - resized_h) / 2.0 - 0.1))
            prepared = np.full((height, width, 3), 114, dtype=np.uint8)
            prepared[top : top + resized_h, left : left + resized_w] = resized
            metadata = {
                "mode": "letterbox",
                "original_shape": [original_h, original_w],
                "new_shape": [height, width],
                "gain": gain,
                "pad_x": float(left),
                "pad_y": float(top),
            }
        else:
            raise ValueError("Preprocessing mode must be resize or letterbox.")

        rgb = cv2.cvtColor(prepared, cv2.COLOR_BGR2RGB)
        real = rgb.astype(np.float32) / 255.0
        if layout == "NCHW":
            real = np.transpose(real[None, ...], (0, 3, 1, 2))
        else:
            real = real[None, ...]
        dtype = np.dtype(self.input_detail["dtype"])
        scale, zero_point = self.input_detail.get("quantization", (0.0, 0))
        if np.issubdtype(dtype, np.integer) and float(scale) > 0:
            limits = np.iinfo(dtype)
            real = np.clip(
                np.rint(real / float(scale) + int(zero_point)),
                limits.min,
                limits.max,
            )
        return (
            np.ascontiguousarray(real, dtype=dtype),
            image,
            metadata,
        )

    def run(
        self,
        input_tensor: np.ndarray,
        *,
        capture_indices: Iterable[int] = (),
        progress: Callable[[int, int, OperatorTrace], None] | None = None,
    ) -> HybridResult:
        input_index = int(self.input_detail["index"])
        expected_shape = tuple(int(value) for value in self.input_detail["shape"])
        value = np.asarray(input_tensor, dtype=self.input_detail["dtype"])
        if value.shape != expected_shape:
            raise ValueError(
                f"Model input expects {expected_shape}, got {value.shape}."
            )
        captured = {int(index) for index in capture_indices}
        cache: dict[int, np.ndarray] = {input_index: np.array(value, copy=True)}
        peak_cached_bytes = int(cache[input_index].nbytes)
        traces: list[OperatorTrace] = []
        run_start = time.perf_counter()

        for ordinal, op in enumerate(self.ops):
            op_index = int(op["index"])
            op_name = str(op["op_name"])
            input_indices = [
                int(index) for index in op.get("inputs", []) if int(index) >= 0
            ]
            output_indices = [
                int(index) for index in op.get("outputs", []) if int(index) >= 0
            ]
            inputs = [self._value(cache, index) for index in input_indices]
            input_meta = [self.tensor_meta[index] for index in input_indices]
            output_meta = [self.tensor_meta[index] for index in output_indices]
            start = time.perf_counter()
            accelerator_trace = ExecutionTrace(0, 0, 0, 0)
            activation_scale = 0.0
            activation_max_abs = 0.0
            saturation_count = 0

            if op_name == "CONV_2D":
                weights = inputs[1]
                bias = (
                    _float_constant(inputs[2], input_meta[2])
                    if len(inputs) > 2
                    else None
                )
                if op_index in self.offload_conv_indices:
                    program = self._program(op_index)
                    result = execute_conv2d(program, inputs[0], self.backend)
                    output = (
                        result.output_quantized
                        if is_quantized(output_meta[0])
                        else result.output_float
                    )
                    if output is None:
                        raise RuntimeError(
                            f"Compiled program for op {op_index} did not emit "
                            "the model output representation."
                        )
                    outputs = [
                        np.asarray(output, dtype=output_meta[0].dtype)
                    ]
                    accelerator_trace = result.trace
                    activation_scale = result.activation_scale
                    activation_max_abs = float(
                        np.max(np.abs(np.asarray(inputs[0], dtype=np.float32)))
                    )
                    centered = np.rint(
                        np.asarray(inputs[0], dtype=np.float32)
                        / max(activation_scale, 1e-30)
                    )
                    saturation_count = int(
                        np.count_nonzero((centered < -128) | (centered > 127))
                    )
                    backend_name = str(
                        getattr(self.backend, "report_name", "accelerator")
                    )
                else:
                    outputs = [
                        execute_conv2d_cpu(
                            inputs[0],
                            weights,
                            input_meta[1],
                            bias,
                            output_meta[0],
                        )
                    ]
                    backend_name = "cpu_conv2d"
            elif op_name == "DEPTHWISE_CONV_2D":
                bias = (
                    _float_constant(inputs[2], input_meta[2])
                    if len(inputs) > 2
                    else None
                )
                outputs = [
                    execute_depthwise_conv2d(
                        inputs[0],
                        inputs[1],
                        input_meta[1],
                        bias,
                        output_meta[0],
                    )
                ]
                backend_name = "cpu_depthwise"
            else:
                outputs = execute_digital_op(
                    op_name,
                    inputs,
                    input_meta,
                    output_meta,
                )
                backend_name = "cpu_digital"

            duration_ms = (time.perf_counter() - start) * 1000.0
            if len(outputs) != len(output_indices):
                raise RuntimeError(
                    f"Op {op_index} {op_name} produced {len(outputs)} outputs; "
                    f"expected {len(output_indices)}."
                )
            for index, output in zip(output_indices, outputs):
                expected = self.tensor_meta[index]
                resolved = np.asarray(output, dtype=expected.dtype)
                if resolved.shape != expected.shape:
                    raise RuntimeError(
                        f"Op {op_index} {op_name} output {index} has "
                        f"{resolved.shape}; expected {expected.shape}."
                    )
                cache[index] = resolved

            for index in input_indices:
                if (
                    index in cache
                    and self.last_use.get(index) == ordinal
                    and index not in self.output_indices
                    and index not in captured
                ):
                    del cache[index]
            peak_cached_bytes = max(
                peak_cached_bytes,
                sum(array.nbytes for array in cache.values()),
            )
            trace = OperatorTrace(
                op_index=op_index,
                op_name=op_name,
                backend=backend_name,
                input_indices=input_indices,
                output_indices=output_indices,
                duration_ms=duration_ms,
                accelerator_calls=accelerator_trace.backend_calls,
                accelerator_input_words=accelerator_trace.input_words,
                accelerator_output_words=accelerator_trace.output_words,
                logical_macs=accelerator_trace.logical_macs,
                activation_scale=activation_scale,
                activation_max_abs=activation_max_abs,
                saturation_count=saturation_count,
            )
            traces.append(trace)
            if progress is not None:
                progress(ordinal + 1, len(self.ops), trace)

        output_index = int(self.output_detail["index"])
        return HybridResult(
            output=np.array(cache[output_index], copy=True),
            traces=traces,
            elapsed_ms=(time.perf_counter() - run_start) * 1000.0,
            peak_cached_bytes=peak_cached_bytes,
        )

    def reference_output(self, input_tensor: np.ndarray) -> np.ndarray:
        """Run LiteRT explicitly for validation; never used by ``run``."""

        reference = _interpreter(self.model_path)
        reference.set_tensor(
            int(self.input_detail["index"]),
            np.asarray(input_tensor, dtype=self.input_detail["dtype"]),
        )
        reference.invoke()
        return np.array(
            reference.get_tensor(int(self.output_detail["index"])),
            copy=True,
        )


def compare_outputs(
    candidate: np.ndarray,
    reference: np.ndarray,
) -> dict[str, Any]:
    lhs = np.asarray(candidate, dtype=np.float32)
    rhs = np.asarray(reference, dtype=np.float32)
    if lhs.shape != rhs.shape:
        raise ValueError(f"Output shapes differ: {lhs.shape} vs {rhs.shape}.")
    difference = lhs - rhs
    return {
        "shape": list(lhs.shape),
        "finite": bool(np.all(np.isfinite(lhs))),
        "mae": float(np.mean(np.abs(difference))),
        "rmse": float(np.sqrt(np.mean(np.square(difference)))),
        "max_abs_error": float(np.max(np.abs(difference))),
        "reference_abs_mean": float(np.mean(np.abs(rhs))),
        "cosine_similarity": float(
            np.dot(lhs.reshape(-1), rhs.reshape(-1))
            / max(
                np.linalg.norm(lhs.reshape(-1))
                * np.linalg.norm(rhs.reshape(-1)),
                1e-30,
            )
        ),
    }


def save_trace(path: str | Path, result: HybridResult) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(
            {
                "summary": result.summary(),
                "operators": [asdict(trace) for trace in result.traces],
            },
            indent=2,
        ),
        encoding="utf-8",
    )
