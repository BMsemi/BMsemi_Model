"""Small TFLite FlatBuffer reader for constant tensor buffers.

The ZCU106 image's tflite_runtime can allocate the graph but does not expose
many constant tensors through get_tensor().  For the PE runtime we only need
immutable buffers such as Conv weights, biases, and shape constants, so this
reader extracts those directly from the .tflite FlatBuffer without depending on
TensorFlow or the flatbuffers Python package.
"""

from __future__ import annotations

import math
import struct
from dataclasses import dataclass
from pathlib import Path

import numpy as np


_TENSOR_TYPE_TO_DTYPE = {
    0: np.float32,  # FLOAT32
    1: np.float16,  # FLOAT16
    2: np.int32,    # INT32
    3: np.uint8,    # UINT8
    4: np.int64,    # INT64
    6: np.bool_,    # BOOL
    7: np.int16,    # INT16
    9: np.int8,     # INT8
    10: np.float64, # FLOAT64
    12: np.uint64,  # UINT64
    15: np.uint32,  # UINT32
    16: np.uint16,  # UINT16
}


class _FlatBuffer:
    def __init__(self, data: bytes) -> None:
        self.data = data

    def _u8(self, pos: int) -> int:
        return self.data[pos]

    def _u16(self, pos: int) -> int:
        return struct.unpack_from("<H", self.data, pos)[0]

    def _u32(self, pos: int) -> int:
        return struct.unpack_from("<I", self.data, pos)[0]

    def _u64(self, pos: int) -> int:
        return struct.unpack_from("<Q", self.data, pos)[0]

    def _i32(self, pos: int) -> int:
        return struct.unpack_from("<i", self.data, pos)[0]

    def root_table(self) -> int:
        return self._u32(0)

    def field_loc(self, table: int, field_index: int) -> int | None:
        vtable = table - self._i32(table)
        vtable_len = self._u16(vtable)
        slot = 4 + field_index * 2
        if slot + 2 > vtable_len:
            return None
        rel = self._u16(vtable + slot)
        if rel == 0:
            return None
        return table + rel

    def indirect(self, loc: int) -> int:
        return loc + self._u32(loc)

    def table_vector(self, table: int, field_index: int) -> list[int]:
        loc = self.field_loc(table, field_index)
        if loc is None:
            return []
        vec = self.indirect(loc)
        length = self._u32(vec)
        start = vec + 4
        return [self.indirect(start + 4 * i) for i in range(length)]

    def int32_vector(self, table: int, field_index: int) -> list[int]:
        loc = self.field_loc(table, field_index)
        if loc is None:
            return []
        vec = self.indirect(loc)
        length = self._u32(vec)
        start = vec + 4
        return [self._i32(start + 4 * i) for i in range(length)]

    def byte_vector(self, table: int, field_index: int) -> bytes:
        loc = self.field_loc(table, field_index)
        if loc is None:
            return b""
        vec = self.indirect(loc)
        length = self._u32(vec)
        start = vec + 4
        return bytes(self.data[start : start + length])

    def buffer_bytes(self, buffer_table: int) -> bytes:
        inline = self.byte_vector(buffer_table, 0)
        if inline:
            return inline

        offset_loc = self.field_loc(buffer_table, 1)
        size_loc = self.field_loc(buffer_table, 2)
        if offset_loc is None or size_loc is None:
            return b""

        offset = int(self._u64(offset_loc))
        size = int(self._u64(size_loc))
        if size <= 0 or offset < 0 or offset + size > len(self.data):
            return b""
        return bytes(self.data[offset : offset + size])


@dataclass(frozen=True)
class _TensorInfo:
    shape: tuple[int, ...]
    tensor_type: int
    buffer_index: int


class TFLiteConstantReader:
    """Read constant tensor buffers from a TFLite model."""

    def __init__(self, model_path: str | Path) -> None:
        self.model_path = Path(model_path)
        self._fb = _FlatBuffer(self.model_path.read_bytes())

        model = self._fb.root_table()
        subgraphs = self._fb.table_vector(model, 2)
        if not subgraphs:
            raise ValueError(f"No subgraphs found in {self.model_path}")

        self._buffers = self._fb.table_vector(model, 4)
        self._tensors = self._read_tensors(subgraphs[0])

    def _read_tensors(self, subgraph: int) -> list[_TensorInfo]:
        tensors: list[_TensorInfo] = []
        for tensor in self._fb.table_vector(subgraph, 0):
            shape = tuple(int(v) for v in self._fb.int32_vector(tensor, 0))

            type_loc = self._fb.field_loc(tensor, 1)
            tensor_type = self._fb._u8(type_loc) if type_loc is not None else 0

            buffer_loc = self._fb.field_loc(tensor, 2)
            buffer_index = self._fb._u32(buffer_loc) if buffer_loc is not None else 0

            tensors.append(_TensorInfo(shape=shape, tensor_type=tensor_type, buffer_index=buffer_index))
        return tensors

    def get_tensor(self, tensor_index: int) -> np.ndarray | None:
        tensor_index = int(tensor_index)
        if tensor_index < 0 or tensor_index >= len(self._tensors):
            return None

        info = self._tensors[tensor_index]
        if info.buffer_index < 0 or info.buffer_index >= len(self._buffers):
            return None

        raw = self._fb.buffer_bytes(self._buffers[info.buffer_index])
        if not raw:
            return None

        dtype = _TENSOR_TYPE_TO_DTYPE.get(info.tensor_type)
        if dtype is None:
            return None

        arr = np.frombuffer(raw, dtype=np.dtype(dtype)).copy()
        expected = math.prod(info.shape) if info.shape else arr.size
        if expected == arr.size:
            return arr.reshape(info.shape)
        return arr
