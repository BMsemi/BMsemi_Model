from __future__ import annotations

import contextlib
import io
from pathlib import Path
import sys
from typing import Any

import numpy as np
from numpy.typing import NDArray


class LegacyNeuromorphicEmulatorBackend:
    """
    Execute remaining MLIR Conv2D MVM tiles through the legacy emulator model.

    The visible dataflow is:

        input_buffer -> compute_buffer -> CrossbarArray -> output_buffer -> L2_Buffer

    ``L2_Buffer`` stores the output channel tensors produced by each layer. The
    XForge executor still owns graph-level tensor scheduling, while this backend
    records the same layer output in the legacy L2 structure so the next layer
    can be understood as reading the current tensor from L2/activation storage.
    """

    max_batch = 32
    report_name = "neuromorphic_emulator"
    diagnostics_key = "legacy_neuromorphic_emulator"

    def __init__(
        self,
        *,
        input_size: int = 64,
        output_size: int = 64,
        l2_capacity: int = 1024,
        quiet: bool = True,
    ) -> None:
        if int(output_size) % 2:
            raise ValueError("Legacy differential output_size must be even.")
        self.input_size = int(input_size)
        self.output_size = int(output_size)
        self.logical_outputs = self.output_size // 2
        self.quiet = bool(quiet)

        legacy_root = Path(__file__).resolve().parents[2]
        if str(legacy_root) not in sys.path:
            sys.path.insert(0, str(legacy_root))

        from legacy_neuromorphic_core.Crossbar_array import CrossbarArray
        from legacy_neuromorphic_core.L2_buffer import L2_Buffer

        with contextlib.redirect_stdout(io.StringIO()):
            self.crossbar_array = CrossbarArray(
                input_size=self.input_size,
                output_size=self.output_size,
            )
        self.l2_buffer = L2_Buffer(capacity=int(l2_capacity))

        self.input_buffer = np.zeros((0, self.input_size), dtype=np.uint8)
        self.compute_buffer = np.zeros((0, self.input_size), dtype=np.uint8)
        self.output_buffer = np.zeros((0, self.logical_outputs), dtype=np.int32)
        self.layer_outputs: dict[int, np.ndarray] = {}
        self.dataflow_events: list[dict[str, Any]] = []
        self.programming_events = 0
        self.crossbar_compute_events = 0
        self.l2_writes = 0
        self.l2_reads = 0
        self.last_input_source = "runtime activation cache"

    def _silence(self):
        return (
            contextlib.redirect_stdout(io.StringIO())
            if self.quiet
            else contextlib.nullcontext()
        )

    def _prepare_weights(self, weights: NDArray[np.integer]) -> np.ndarray:
        matrix = np.asarray(weights)
        if matrix.ndim != 2:
            raise ValueError(f"Legacy emulator weights must be [K,N], got {matrix.shape}.")
        rows, cols = (int(value) for value in matrix.shape)
        if not 1 <= rows <= self.input_size:
            raise ValueError(f"Weight rows {rows} exceed legacy input_size={self.input_size}.")
        if not 1 <= cols <= self.logical_outputs:
            raise ValueError(
                f"Logical output columns {cols} exceed legacy capacity={self.logical_outputs}."
            )
        if not np.issubdtype(matrix.dtype, np.integer):
            raise TypeError(f"Legacy emulator weights must be integer, got {matrix.dtype}.")
        if matrix.size and (int(matrix.min()) < -128 or int(matrix.max()) > 127):
            raise ValueError("Legacy emulator weights must fit signed INT8.")

        padded = np.zeros((self.input_size, self.logical_outputs), dtype=np.int8)
        padded[:rows, :cols] = matrix.astype(np.int8, copy=False)
        return padded

    def _prepare_inputs(self, vectors: NDArray[np.integer], rows: int) -> np.ndarray:
        batch = np.asarray(vectors)
        if batch.ndim != 2:
            raise ValueError(f"Legacy emulator input must be [B,K], got {batch.shape}.")
        if int(batch.shape[1]) != int(rows):
            raise ValueError(f"Input K={batch.shape[1]} does not match tile rows={rows}.")
        if not 1 <= int(batch.shape[0]) <= self.max_batch:
            raise ValueError(f"Batch size must be 1..{self.max_batch}, got {batch.shape[0]}.")
        if not np.issubdtype(batch.dtype, np.integer):
            raise TypeError(f"Legacy emulator activations must be integer, got {batch.dtype}.")
        if batch.size and (int(batch.min()) < -128 or int(batch.max()) > 127):
            raise ValueError("Legacy emulator activations must fit signed INT8.")

        padded = np.zeros((int(batch.shape[0]), self.input_size), dtype=np.uint8)
        signed = batch.astype(np.int16, copy=False)
        padded[:, :rows] = np.clip(signed + 128, 0, 255).astype(np.uint8)
        return padded

    def _program_and_compute(
        self,
        vectors: NDArray[np.integer],
        weights: NDArray[np.integer],
    ) -> np.ndarray:
        signed_weights = np.asarray(weights, dtype=np.int8)
        rows, cols = (int(value) for value in signed_weights.shape)
        programmed = self._prepare_weights(signed_weights)
        self.input_buffer = self._prepare_inputs(vectors, rows)
        self.compute_buffer = self.input_buffer.copy()

        with self._silence():
            self.crossbar_array.program_weights(programmed)
            raw = np.asarray(
                self.crossbar_array.compute_batch(self.compute_buffer),
                dtype=np.int32,
            )

        correction = 128 * signed_weights.astype(np.int32).sum(axis=0).reshape(1, -1)
        partial = raw[:, :cols].astype(np.int32, copy=False) - correction
        self.output_buffer = np.zeros((partial.shape[0], self.logical_outputs), dtype=np.int32)
        self.output_buffer[:, :cols] = partial
        self.programming_events += 1
        self.crossbar_compute_events += 1
        return partial

    @staticmethod
    def _op_index(program: Any) -> int:
        source = dict(getattr(program, "metadata", {}).get("source", {}))
        return int(source.get("op_index", -1))

    def write_layer_tensor(
        self,
        layer_id: int,
        tensor: np.ndarray,
        *,
        source: str = "neuromorphic_emulator",
    ) -> None:
        tensor_i32 = np.asarray(tensor, dtype=np.int32)
        self.layer_outputs[int(layer_id)] = tensor_i32.copy()
        for channel_id in range(tensor_i32.shape[1]):
            self.l2_buffer.write_output_channel(
                int(layer_id),
                int(channel_id),
                tensor_i32[:, channel_id],
            )
            self.l2_writes += 1
        self.dataflow_events.append(
            {
                "layer_id": int(layer_id),
                "source": str(source),
                "shape": list(tensor_i32.shape),
                "dataflow": (
                    "input_buffer -> compute_buffer -> crossbar_array -> "
                    "output_buffer -> L2_Buffer"
                    if source == "neuromorphic_emulator"
                    else "FPGA PE output -> DDR/L2_Buffer"
                ),
            }
        )

    def _write_layer_to_l2(self, layer_id: int, tensor: np.ndarray) -> None:
        self.write_layer_tensor(
            layer_id,
            tensor,
            source="neuromorphic_emulator",
        )

    def read_layer_tensor(self, layer_id: int) -> np.ndarray | None:
        tensor = self.layer_outputs.get(int(layer_id))
        if tensor is None:
            return None
        self.l2_reads += int(tensor.shape[1])
        self.last_input_source = f"L2_Buffer layer {int(layer_id)}"
        return np.array(tensor, copy=True)

    def run_batch(
        self,
        vectors: NDArray[np.integer],
        weights: NDArray[np.integer],
    ) -> NDArray[np.int32]:
        return self._program_and_compute(vectors, weights)

    def run_program(
        self,
        program: Any,
        inputs: NDArray[np.integer],
    ) -> NDArray[np.int32]:
        vectors = np.asarray(inputs, dtype=np.int8)
        output = np.zeros((int(program.m_dim), int(program.n_dim)), dtype=np.int32)
        layer_id = self._op_index(program)
        for tile in program.tiles:
            weights = np.asarray(program.constants[tile.weight_key], dtype=np.int8)
            for start in range(0, int(program.m_dim), self.max_batch):
                stop = min(start + self.max_batch, int(program.m_dim))
                partial = self._program_and_compute(
                    vectors[start:stop, tile.k_start : tile.k_start + tile.rows],
                    weights,
                )
                output[start:stop, tile.n_start : tile.n_start + tile.cols] += partial
                self.dataflow_events.append(
                    {
                        "layer_id": layer_id,
                        "tile_id": int(tile.tile_id),
                        "k_start": int(tile.k_start),
                        "n_start": int(tile.n_start),
                        "rows": int(tile.rows),
                        "cols": int(tile.cols),
                        "dataflow": (
                            "input_buffer -> compute_buffer -> crossbar_array -> "
                            "output_buffer -> L2_Buffer"
                        ),
                    }
                )
        self._write_layer_to_l2(layer_id, output)
        return output

    def diagnostics(self) -> dict[str, Any]:
        return {
            "engine": "LegacyNeuromorphicCore",
            "mode": "remaining_yolo_layers",
            "files": {
                "crossbar": "python/legacy_neuromorphic_core/Crossbar_array.py",
                "l2_buffer": "python/legacy_neuromorphic_core/L2_buffer.py",
            },
            "dataflow": [
                "input_buffer",
                "compute_buffer",
                "crossbar_array",
                "output_buffer",
                "L2_Buffer",
            ],
            "layer_to_layer_contract": (
                "L2_Buffer stores each layer output channel; the current layer tensor "
                "is the activation source for the next layer."
            ),
            "input_buffer_shape": list(self.input_buffer.shape),
            "compute_buffer_shape": list(self.compute_buffer.shape),
            "output_buffer_shape": list(self.output_buffer.shape),
            "l2_stats": self.l2_buffer.get_stats(),
            "layer_outputs": {
                str(layer_id): list(value.shape)
                for layer_id, value in sorted(self.layer_outputs.items())
            },
            "last_input_source": self.last_input_source,
            "programming_events": self.programming_events,
            "crossbar_compute_events": self.crossbar_compute_events,
            "l2_writes": self.l2_writes,
            "l2_reads": self.l2_reads,
            "dataflow_events": list(self.dataflow_events[-16:]),
        }
