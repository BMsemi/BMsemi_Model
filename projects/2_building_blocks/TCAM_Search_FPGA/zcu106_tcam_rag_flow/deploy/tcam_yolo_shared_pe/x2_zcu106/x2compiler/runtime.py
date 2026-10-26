from __future__ import annotations

import contextlib
from dataclasses import dataclass
import importlib.util
import io
from pathlib import Path
import sys
from typing import Any, Protocol

import numpy as np
from numpy.typing import NDArray

from .ir import X2Program


_CROSSBAR_ARRAY_CLASS = None


def _project_crossbar_array_class():
    """Load the project's programmed-cell crossbar engine without a path import."""

    global _CROSSBAR_ARRAY_CLASS
    if _CROSSBAR_ARRAY_CLASS is not None:
        return _CROSSBAR_ARRAY_CLASS
    try:
        from .engine_core import CrossbarArray
    except ImportError:
        pass
    else:
        _CROSSBAR_ARRAY_CLASS = CrossbarArray
        return _CROSSBAR_ARRAY_CLASS
    source = (
        Path(__file__).resolve().parents[2]
        / "YOLO26n-TFLite-FPGA-Partner-Project"
        / "core"
        / "Crossbar_array.py"
    )
    if not source.is_file():
        raise RuntimeError(
            "The local engine source was not found. Expected "
            f"{source}."
        )
    module_name = "_project_local_engine_core"
    spec = importlib.util.spec_from_file_location(module_name, source)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load the local engine from {source}.")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    _CROSSBAR_ARRAY_CLASS = module.CrossbarArray
    return _CROSSBAR_ARRAY_CLASS


class X2Backend(Protocol):
    max_batch: int

    def run_batch(
        self,
        vectors: NDArray[np.integer],
        weights: NDArray[np.integer],
    ) -> NDArray[np.int32]:
        ...


class _BufferedProcessingElement:
    """Single software PE with explicit local buffers for MLIR MVM tiles."""

    def __init__(
        self,
        pe_id: int,
        *,
        max_rows: int = 32,
        max_cols: int = 32,
        max_batch: int = 32,
    ) -> None:
        self.pe_id = int(pe_id)
        self.max_rows = int(max_rows)
        self.max_cols = int(max_cols)
        self.max_batch = int(max_batch)
        self.weight_buffer = np.zeros(
            (self.max_rows, self.max_cols),
            dtype=np.int8,
        )
        self.input_buffer = np.zeros(
            (self.max_batch, self.max_rows),
            dtype=np.int8,
        )
        self.partial_sum_buffer = np.zeros(
            (self.max_batch, self.max_cols),
            dtype=np.int32,
        )
        self.programmed_shape = (0, 0)
        self.programming_events = 0
        self.input_buffer_loads = 0
        self.output_buffer_writes = 0
        self.computed_vectors = 0

    def program_weights(self, weights: NDArray[np.integer]) -> None:
        matrix = np.asarray(weights, dtype=np.int8)
        if matrix.ndim != 2:
            raise ValueError(f"PE weights must be rank two, got {matrix.shape}.")
        rows, cols = (int(value) for value in matrix.shape)
        if not 1 <= rows <= self.max_rows or not 1 <= cols <= self.max_cols:
            raise ValueError(
                f"PE tile {matrix.shape} exceeds "
                f"{self.max_rows}x{self.max_cols}."
            )
        self.weight_buffer.fill(0)
        self.weight_buffer[:rows, :cols] = matrix
        self.programmed_shape = (rows, cols)
        self.programming_events += 1

    def compute_batch(self, vectors: NDArray[np.integer]) -> NDArray[np.int32]:
        batch = np.asarray(vectors, dtype=np.int8)
        if batch.ndim != 2:
            raise ValueError(f"PE input must be [B,K], got {batch.shape}.")
        rows, cols = self.programmed_shape
        if rows == 0 or cols == 0:
            raise RuntimeError("PE weights must be programmed before compute.")
        if batch.shape[1] != rows:
            raise ValueError(
                f"PE input K={batch.shape[1]} does not match programmed rows={rows}."
            )

        outputs: list[np.ndarray] = []
        for start in range(0, batch.shape[0], self.max_batch):
            chunk = batch[start : start + self.max_batch]
            chunk_rows = int(chunk.shape[0])
            self.input_buffer.fill(0)
            self.partial_sum_buffer.fill(0)
            self.input_buffer[:chunk_rows, :rows] = chunk
            partial = (
                self.input_buffer[:chunk_rows, :rows].astype(np.int32)
                @ self.weight_buffer[:rows, :cols].astype(np.int32)
            )
            self.partial_sum_buffer[:chunk_rows, :cols] = partial
            outputs.append(self.partial_sum_buffer[:chunk_rows, :cols].copy())
            self.input_buffer_loads += 1
            self.output_buffer_writes += 1
            self.computed_vectors += chunk_rows
        if not outputs:
            return np.zeros((0, cols), dtype=np.int32)
        return np.vstack(outputs).astype(np.int32, copy=False)


class PEArrayX2Backend:
    """
    Execute MLIR ``x2.mvm_tile`` programs on a buffered PE-array model.

    This backend keeps the compiler's existing tiled MLIR contract, but replaces
    the X1/X2/crossbar execution path with PE-local weight, activation, and
    partial-sum buffers followed by signed INT8xINT8 INT32 MAC.
    """

    max_batch = 32
    report_name = "accelerator"
    diagnostics_key = "accelerator"

    def __init__(
        self,
        *,
        pe_count: int = 8,
        max_rows: int = 32,
        max_cols: int = 32,
        max_batch: int = 32,
    ) -> None:
        if int(pe_count) < 1:
            raise ValueError("pe_count must be positive.")
        if int(max_rows) != 32 or int(max_cols) != 32 or int(max_batch) != 32:
            raise ValueError(
                "The current MLIR target is fixed at 32 rows, 32 columns, "
                "and batch 32."
            )
        self.pe_count = int(pe_count)
        self.max_rows = int(max_rows)
        self.max_cols = int(max_cols)
        self.max_batch = int(max_batch)
        self._pes = [
            _BufferedProcessingElement(
                index,
                max_rows=self.max_rows,
                max_cols=self.max_cols,
                max_batch=self.max_batch,
            )
            for index in range(self.pe_count)
        ]
        self._next_pe = 0
        self.programming_events = 0
        self.programmed_logical_weights = 0
        self.input_buffer_loads = 0
        self.output_buffer_writes = 0
        self.computed_vectors = 0
        self.tile_dispatches = 0
        self.accumulator_updates = 0
        self._used_pe_ids: set[int] = set()
        self._instantiated_shapes: set[tuple[int, int]] = set()

    @staticmethod
    def _validate(
        vectors: NDArray[np.integer],
        weights: NDArray[np.integer],
        *,
        allow_large_batch: bool = False,
    ) -> tuple[NDArray[np.int8], NDArray[np.int8]]:
        batch = np.asarray(vectors)
        matrix = np.asarray(weights)
        if batch.ndim != 2 or matrix.ndim != 2:
            raise ValueError("PE backend expects vectors [B,K] and weights [K,N].")
        if batch.shape[1] != matrix.shape[0]:
            raise ValueError(f"PE K mismatch: {batch.shape} vs {matrix.shape}.")
        if not 1 <= batch.shape[0] or (
            not allow_large_batch
            and batch.shape[0] > PEArrayX2Backend.max_batch
        ):
            raise ValueError(
                f"PE batch {batch.shape[0]} is outside "
                f"1..{PEArrayX2Backend.max_batch}."
            )
        if not 1 <= matrix.shape[0] <= 32 or not 1 <= matrix.shape[1] <= 32:
            raise ValueError(
                f"PE tile exceeds the implemented 32x32 contract: {matrix.shape}."
            )
        if not np.issubdtype(batch.dtype, np.integer):
            raise TypeError(f"PE activations must be integer, got {batch.dtype}.")
        if not np.issubdtype(matrix.dtype, np.integer):
            raise TypeError(f"PE weights must be integer, got {matrix.dtype}.")
        if batch.size and (int(batch.min()) < -128 or int(batch.max()) > 127):
            raise ValueError("PE activations are outside signed INT8.")
        if matrix.size and (int(matrix.min()) < -128 or int(matrix.max()) > 127):
            raise ValueError("PE weights are outside signed INT8.")
        return (
            np.ascontiguousarray(batch, dtype=np.int8),
            np.ascontiguousarray(matrix, dtype=np.int8),
        )

    def _allocate_pe(self, tile_id: int | None = None) -> _BufferedProcessingElement:
        if tile_id is None:
            pe_index = self._next_pe
            self._next_pe = (self._next_pe + 1) % self.pe_count
        else:
            pe_index = int(tile_id) % self.pe_count
        self._used_pe_ids.add(pe_index)
        return self._pes[pe_index]

    def _program_and_compute(
        self,
        vectors: NDArray[np.integer],
        weights: NDArray[np.integer],
        *,
        allow_large_batch: bool = False,
        tile_id: int | None = None,
    ) -> NDArray[np.int32]:
        batch, matrix = self._validate(
            vectors,
            weights,
            allow_large_batch=allow_large_batch,
        )
        pe = self._allocate_pe(tile_id)
        before_input_loads = pe.input_buffer_loads
        before_output_writes = pe.output_buffer_writes
        pe.program_weights(matrix)
        output = pe.compute_batch(batch)

        rows, cols = matrix.shape
        self.programming_events += 1
        self.programmed_logical_weights += int(rows * cols)
        self.input_buffer_loads += int(pe.input_buffer_loads - before_input_loads)
        self.output_buffer_writes += int(
            pe.output_buffer_writes - before_output_writes
        )
        self.computed_vectors += int(batch.shape[0])
        self.tile_dispatches += 1
        self._instantiated_shapes.add((int(rows), int(cols)))
        return output.astype(np.int32, copy=False)

    def run_batch(
        self,
        vectors: NDArray[np.integer],
        weights: NDArray[np.integer],
    ) -> NDArray[np.int32]:
        return self._program_and_compute(vectors, weights)

    def run_program(
        self,
        program: X2Program,
        inputs: NDArray[np.integer],
    ) -> NDArray[np.int32]:
        vectors = np.asarray(inputs, dtype=np.int8)
        output = np.zeros((program.m_dim, program.n_dim), dtype=np.int32)
        for tile in program.tiles:
            weights = np.asarray(
                program.constants[tile.weight_key],
                dtype=np.int8,
            )
            lhs = vectors[:, tile.k_start : tile.k_start + tile.rows]
            partial = self._program_and_compute(
                lhs,
                weights,
                allow_large_batch=True,
                tile_id=tile.tile_id,
            )
            output[
                :,
                tile.n_start : tile.n_start + tile.cols,
            ] += partial
            self.accumulator_updates += 1
        return output

    def diagnostics(self) -> dict[str, Any]:
        active_pes = [self._pes[index] for index in sorted(self._used_pe_ids)]
        return {
            "engine": "LocalAcceleratorEngine",
            "mode": "INT8_COMPUTE",
            "compute_unit_count": self.pe_count,
            "compute_unit_shape": {
                "rows": self.max_rows,
                "cols": self.max_cols,
                "max_batch": self.max_batch,
            },
            "weight_storage": "local signed INT8 weight buffer",
            "activation_storage": "local signed INT8 input buffer",
            "partial_sum_storage": "local INT32 output buffer",
            "programming_events": self.programming_events,
            "programmed_logical_weights": self.programmed_logical_weights,
            "input_buffer_loads": self.input_buffer_loads,
            "output_buffer_writes": self.output_buffer_writes,
            "computed_vectors": self.computed_vectors,
            "tile_dispatches": self.tile_dispatches,
            "accumulator_updates": self.accumulator_updates,
            "active_unit_ids": [pe.pe_id for pe in active_pes],
            "instantiated_shapes": [
                {"rows": rows, "cols": cols}
                for rows, cols in sorted(self._instantiated_shapes)
            ],
            "compiled_mapping": "compiled_tile -> program_weights -> input buffer -> INT8 compute -> output buffer",
            "signed_activation_contract": "direct signed INT8 activations and signed INT8 weights",
            "execution_boundary": "local execution only; no external runtime access",
        }


class CrossbarX2Backend:
    """
    Program and execute the project's differential INT8 crossbar engine.

    X2 programs use signed INT8 activations while the PWM crossbar consumes
    unsigned drive counts.  The backend applies the exact identity

        signed(x) @ W = (signed(x) + 128) @ W - 128 * sum_rows(W)

    after programming each tile into the four 4-bit differential cell banks.
    """

    # Preserve the compiler/RTL launch contract even though the local engine
    # can vectorize a complete program internally.
    max_batch = 32
    diagnostics_key = "crossbar"

    def __init__(
        self,
        *,
        crossbar_features: dict[str, Any] | None = None,
        quiet: bool = True,
    ) -> None:
        self.crossbar_features = crossbar_features
        self.quiet = bool(quiet)
        self._engines: dict[tuple[int, int], Any] = {}
        self.programming_events = 0
        self.programmed_logical_weights = 0
        self.programmed_physical_cells = 0
        self.computed_vectors = 0

    def _silence(self):
        return (
            contextlib.redirect_stdout(io.StringIO())
            if self.quiet
            else contextlib.nullcontext()
        )

    def _engine(self, rows: int, cols: int):
        key = (int(rows), int(cols))
        if key not in self._engines:
            crossbar_class = _project_crossbar_array_class()
            with self._silence():
                self._engines[key] = crossbar_class(
                    input_size=key[0],
                    output_size=key[1] * 2,
                    crossbar_features=self.crossbar_features,
                )
        return self._engines[key]

    @staticmethod
    def _validate(
        vectors: NDArray[np.integer],
        weights: NDArray[np.integer],
        *,
        allow_large_batch: bool = False,
    ) -> tuple[NDArray[np.int8], NDArray[np.int8]]:
        batch = np.asarray(vectors)
        matrix = np.asarray(weights)
        if batch.ndim != 2 or matrix.ndim != 2:
            raise ValueError(
                "Crossbar backend expects vectors [B,K] and weights [K,N]."
            )
        if batch.shape[1] != matrix.shape[0]:
            raise ValueError(
                f"Crossbar K mismatch: {batch.shape} vs {matrix.shape}."
            )
        if not 1 <= batch.shape[0] or (
            not allow_large_batch
            and batch.shape[0] > CrossbarX2Backend.max_batch
        ):
            raise ValueError(
                f"Crossbar batch {batch.shape[0]} is outside "
                f"1..{CrossbarX2Backend.max_batch}."
            )
        if not 1 <= matrix.shape[0] <= 32 or not 1 <= matrix.shape[1] <= 32:
            raise ValueError(
                f"Crossbar tile exceeds the implemented 32x32 contract: "
                f"{matrix.shape}."
            )
        if not np.issubdtype(batch.dtype, np.integer):
            raise TypeError(f"Crossbar activations must be integer, got {batch.dtype}.")
        if not np.issubdtype(matrix.dtype, np.integer):
            raise TypeError(f"Crossbar weights must be integer, got {matrix.dtype}.")
        if batch.size and (int(batch.min()) < -128 or int(batch.max()) > 127):
            raise ValueError("Crossbar activations are outside signed INT8.")
        if matrix.size and (int(matrix.min()) < -128 or int(matrix.max()) > 127):
            raise ValueError("Crossbar weights are outside signed INT8.")
        return (
            np.ascontiguousarray(batch, dtype=np.int8),
            np.ascontiguousarray(matrix, dtype=np.int8),
        )

    def _program_and_compute(
        self,
        vectors: NDArray[np.integer],
        weights: NDArray[np.integer],
        *,
        allow_large_batch: bool = False,
    ) -> NDArray[np.int32]:
        batch, matrix = self._validate(
            vectors,
            weights,
            allow_large_batch=allow_large_batch,
        )
        rows, cols = matrix.shape
        engine = self._engine(rows, cols)
        with self._silence():
            engine.program_weights(matrix)
            unsigned_drive = (
                batch.astype(np.int16, copy=False) + 128
            ).astype(np.uint8)
            raw = np.asarray(
                engine.compute_batch(unsigned_drive),
                dtype=np.int32,
            )[:, :cols]

        correction = (
            128
            * matrix.astype(np.int32, copy=False).sum(
                axis=0,
                dtype=np.int64,
            )
        )
        corrected = raw.astype(np.int64, copy=False) - correction.reshape(1, -1)
        self.programming_events += 1
        self.programmed_logical_weights += int(rows * cols)
        self.programmed_physical_cells += int(rows * cols * 4)
        self.computed_vectors += int(batch.shape[0])
        return corrected.astype(np.int32)

    def run_batch(
        self,
        vectors: NDArray[np.integer],
        weights: NDArray[np.integer],
    ) -> NDArray[np.int32]:
        return self._program_and_compute(vectors, weights)

    def run_program(
        self,
        program: X2Program,
        inputs: NDArray[np.integer],
    ) -> NDArray[np.int32]:
        vectors = np.asarray(inputs, dtype=np.int8)
        output = np.zeros((program.m_dim, program.n_dim), dtype=np.int32)
        for tile in program.tiles:
            weights = np.asarray(
                program.constants[tile.weight_key],
                dtype=np.int8,
            )
            lhs = vectors[:, tile.k_start : tile.k_start + tile.rows]
            partial = self._program_and_compute(
                lhs,
                weights,
                allow_large_batch=True,
            )
            output[
                :,
                tile.n_start : tile.n_start + tile.cols,
            ] += partial
        return output

    def diagnostics(self) -> dict[str, Any]:
        return {
            "engine": "CrossbarArray",
            "mode": "PWM_INT8_ROW_SERIAL",
            "weight_storage": "four 4-bit differential cell banks",
            "programming_events": self.programming_events,
            "programmed_logical_weights": self.programmed_logical_weights,
            "programmed_physical_cells": self.programmed_physical_cells,
            "computed_vectors": self.computed_vectors,
            "instantiated_shapes": [
                {"rows": rows, "cols": cols}
                for rows, cols in sorted(self._engines)
            ],
            "signed_activation_bridge": "unsigned_drive=x+128; subtract 128*sum(weights)",
            "nonideal_features": self.crossbar_features or {},
        }


class PynqX2Backend:
    max_batch = 64

    def __init__(self, accelerator) -> None:
        self.accelerator = accelerator

    @classmethod
    def from_bitstream(cls, bit_path: str) -> "PynqX2Backend":
        try:
            from .external_runtime import DualAxiX2MvmPynq
        except ImportError:
            from x2_zcu106.runtime.axi_x2_mvm_driver import DualAxiX2MvmPynq

        return cls(DualAxiX2MvmPynq(bit_path))

    def run_batch(
        self,
        vectors: NDArray[np.integer],
        weights: NDArray[np.integer],
    ) -> NDArray[np.int32]:
        return self.accelerator.run_mvm_batch(vectors, weights)


def public_backend_name(name: str | None) -> str:
    normalized = (name or "local").strip().lower().replace("_", "-")
    if normalized in {"local", "accelerator", "default", "pe"}:
        return "local"
    if normalized in {"external", "remote", "board", "pynq"}:
        return "external"
    if normalized in {"comparison", "compare", "crossbar"}:
        return "comparison"
    raise ValueError("backend must be 'local' or 'external'.")


def create_accelerator_backend(
    name: str | None = "local",
    *,
    runtime_package: str | Path | None = None,
):
    selected = public_backend_name(name)
    if selected == "local":
        return PEArrayX2Backend()
    if selected == "comparison":
        return CrossbarX2Backend()
    if runtime_package is None:
        raise ValueError("An external runtime package is required.")
    return PynqX2Backend.from_bitstream(str(Path(runtime_package).resolve()))


@dataclass(frozen=True)
class ExecutionTrace:
    backend_calls: int
    input_words: int
    output_words: int
    logical_macs: int


@dataclass(frozen=True)
class MatrixResult:
    accumulators: np.ndarray
    trace: ExecutionTrace


@dataclass(frozen=True)
class Conv2DResult:
    quantized_input: np.ndarray
    accumulators: np.ndarray
    output_float: np.ndarray | None
    output_quantized: np.ndarray | None
    activation_scale: float
    trace: ExecutionTrace


def execute_matrix(
    program: X2Program,
    inputs_i8: NDArray[np.integer],
    backend: X2Backend,
) -> MatrixResult:
    program.validate()
    inputs = np.asarray(inputs_i8)
    if inputs.shape != (program.m_dim, program.k_dim):
        raise ValueError(
            f"Program expects input {(program.m_dim, program.k_dim)}, got {inputs.shape}."
        )
    if not np.issubdtype(inputs.dtype, np.integer):
        raise TypeError(f"X2 inputs must be integer, got {inputs.dtype}.")
    if inputs.size and (int(inputs.min()) < -128 or int(inputs.max()) > 127):
        raise ValueError("X2 inputs contain values outside signed INT8.")
    inputs = inputs.astype(np.int8, copy=False)

    backend_calls = 0
    input_words = 0
    output_words = 0
    for tile in program.tiles:
        for batch_start in range(0, program.m_dim, int(backend.max_batch)):
            batch_size = min(int(backend.max_batch), program.m_dim - batch_start)
            backend_calls += 1
            input_words += (
                (tile.rows * tile.cols + 3) // 4
                + batch_size * ((tile.rows + 3) // 4)
            )
            output_words += batch_size * tile.cols

    run_program = getattr(backend, "run_program", None)
    if callable(run_program):
        output = np.asarray(run_program(program, inputs), dtype=np.int32)
        if output.shape != (program.m_dim, program.n_dim):
            raise RuntimeError(
                f"Backend returned {output.shape}; expected "
                f"{(program.m_dim, program.n_dim)}."
            )
    else:
        output = np.zeros((program.m_dim, program.n_dim), dtype=np.int32)
        for tile in program.tiles:
            weights = np.asarray(program.constants[tile.weight_key], dtype=np.int8)
            for batch_start in range(0, program.m_dim, int(backend.max_batch)):
                batch_size = min(int(backend.max_batch), program.m_dim - batch_start)
                vectors = inputs[
                    batch_start : batch_start + batch_size,
                    tile.k_start : tile.k_start + tile.rows,
                ]
                partial = np.asarray(backend.run_batch(vectors, weights), dtype=np.int32)
                if partial.shape != (batch_size, tile.cols):
                    raise RuntimeError(
                        f"Backend returned {partial.shape}; "
                        f"expected {(batch_size, tile.cols)}."
                    )
                output[
                    batch_start : batch_start + batch_size,
                    tile.n_start : tile.n_start + tile.cols,
                ] += partial

    bias_key = program.metadata.get("bias_i32_key")
    if bias_key:
        output += np.asarray(program.constants[str(bias_key)], dtype=np.int32).reshape(1, -1)
    return MatrixResult(
        accumulators=output,
        trace=ExecutionTrace(
            backend_calls=backend_calls,
            input_words=input_words,
            output_words=output_words,
            logical_macs=program.m_dim * program.k_dim * program.n_dim,
        ),
    )


def _im2col_nhwc(
    input_i8: np.ndarray,
    *,
    kernel: tuple[int, int],
    stride: tuple[int, int],
    padding: str,
) -> tuple[np.ndarray, tuple[int, int]]:
    value = np.asarray(input_i8, dtype=np.int8)
    if value.ndim != 4:
        raise ValueError(f"Expected NHWC input, got {value.shape}.")
    batch, input_h, input_w, channels = value.shape
    kernel_h, kernel_w = kernel
    stride_y, stride_x = stride
    if padding == "same":
        output_h = (input_h + stride_y - 1) // stride_y
        output_w = (input_w + stride_x - 1) // stride_x
        total_pad_h = max((output_h - 1) * stride_y + kernel_h - input_h, 0)
        total_pad_w = max((output_w - 1) * stride_x + kernel_w - input_w, 0)
        top = total_pad_h // 2
        bottom = total_pad_h - top
        left = total_pad_w // 2
        right = total_pad_w - left
        padded = np.pad(
            value,
            ((0, 0), (top, bottom), (left, right), (0, 0)),
            mode="constant",
            constant_values=0,
        )
    elif padding == "valid":
        output_h = (input_h - kernel_h) // stride_y + 1
        output_w = (input_w - kernel_w) // stride_x + 1
        padded = value
    else:
        raise ValueError(f"Unsupported padding: {padding}")

    windows = np.lib.stride_tricks.sliding_window_view(
        padded,
        (kernel_h, kernel_w),
        axis=(1, 2),
    )
    selected = windows[:, ::stride_y, ::stride_x, :, :, :]
    selected = selected[:, :output_h, :output_w, :, :, :]
    columns = np.ascontiguousarray(
        selected.transpose(0, 1, 2, 4, 5, 3).reshape(
            batch * output_h * output_w,
            kernel_h * kernel_w * channels,
        ),
        dtype=np.int8,
    )
    return columns, (output_h, output_w)


def execute_conv2d(
    program: X2Program,
    input_nhwc: NDArray[np.number],
    backend: X2Backend,
) -> Conv2DResult:
    program.validate()
    if program.kind != "conv2d":
        raise ValueError(f"Expected a Conv2D program, got {program.kind}.")
    config = dict(program.metadata["conv2d"])
    expected_shape = tuple(int(value) for value in config["input_shape_nhwc"])
    input_value = np.asarray(input_nhwc)
    if input_value.shape != expected_shape:
        raise ValueError(f"Conv2D expects input {expected_shape}, got {input_value.shape}.")

    quant = dict(program.metadata.get("activation_quantization", {}))
    input_zero_point = int(quant.get("zero_point", 0))
    configured_scale = float(quant.get("scale", 0.0))
    if np.issubdtype(input_value.dtype, np.floating):
        max_abs = float(np.max(np.abs(input_value))) if input_value.size else 0.0
        activation_scale = configured_scale if configured_scale > 0 else max(max_abs / 127.0, 1e-12)
        centered = np.rint(input_value.astype(np.float32) / activation_scale)
    elif np.issubdtype(input_value.dtype, np.integer):
        activation_scale = configured_scale
        if activation_scale <= 0:
            raise ValueError("Integer activation input requires a positive quantization scale.")
        centered = input_value.astype(np.int32) - input_zero_point
    else:
        raise TypeError(f"Unsupported Conv2D activation dtype: {input_value.dtype}")

    if centered.size and (int(centered.min()) < -128 or int(centered.max()) > 127):
        if bool(quant.get("allow_saturation", False)):
            centered = np.clip(centered, -128, 127)
        else:
            raise ValueError(
                "Centered activations exceed signed INT8. Recalibrate the activation "
                "scale or enable explicit prototype saturation."
            )
    input_i8 = centered.astype(np.int8)

    kernel = tuple(int(value) for value in config["kernel"])
    stride = tuple(int(value) for value in config["stride"])
    columns, output_hw = _im2col_nhwc(
        input_i8,
        kernel=kernel,
        stride=stride,
        padding=str(config["padding"]),
    )
    matrix_result = execute_matrix(program, columns, backend)
    expected_output = tuple(int(value) for value in config["output_shape_nhwc"])
    batch = input_i8.shape[0]
    output_h, output_w = output_hw
    if (batch, output_h, output_w, program.n_dim) != expected_output:
        raise RuntimeError(
            "Runtime Conv2D geometry differs from the compiled output shape: "
            f"{(batch, output_h, output_w, program.n_dim)} vs {expected_output}."
        )
    accumulators = matrix_result.accumulators.reshape(expected_output)

    output_float: np.ndarray | None = None
    scale_key = program.metadata.get("weight_scales_key")
    if scale_key:
        weight_scales = np.asarray(
            program.constants[str(scale_key)],
            dtype=np.float32,
        ).reshape(-1)
        if weight_scales.size == 1:
            weight_scales = np.repeat(weight_scales, program.n_dim)
        output_float = (
            accumulators.astype(np.float32)
            * np.float32(activation_scale)
            * weight_scales.reshape(1, 1, 1, -1)
        )
        bias_float_key = program.metadata.get("bias_float_key")
        if bias_float_key:
            output_float += np.asarray(
                program.constants[str(bias_float_key)],
                dtype=np.float32,
            ).reshape(1, 1, 1, -1)

    output_quantized: np.ndarray | None = None
    output_quant = dict(program.metadata.get("output_quantization", {}))
    output_scale = float(output_quant.get("scale", 0.0))
    if output_scale > 0:
        if output_float is None:
            raise ValueError("Output requantization requires weight scales.")
        output_zero_point = int(output_quant.get("zero_point", 0))
        dtype = np.dtype(str(output_quant.get("dtype", "int8")))
        q = np.rint(output_float / output_scale + output_zero_point)
        limits = np.iinfo(dtype)
        output_quantized = np.clip(q, limits.min, limits.max).astype(dtype)

    return Conv2DResult(
        quantized_input=input_i8,
        accumulators=accumulators,
        output_float=output_float,
        output_quantized=output_quantized,
        activation_scale=activation_scale,
        trace=matrix_result.trace,
    )
