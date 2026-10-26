from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import numpy as np
from numpy.typing import NDArray


HEADER_TAG = 0x0DED0000
HEADER_MASK = 0xFFFF0000
EMPTY_TOKEN = 0xDEADC0DE

N_X2 = 8
X2_ROWS = 32
X2_COLS = 32

OP_READ = 0b01 << 30
OP_COMPUTE = 0b10 << 30
OP_WRITE = 0b11 << 30


def fmt_rd_cmd(row: int, col: int) -> np.uint32:
    return np.uint32(OP_READ | ((row & 0x1F) << 25) | ((col & 0x1F) << 20))


def fmt_wr_cmd(row: int, col: int, value: int) -> np.uint32:
    return np.uint32(OP_WRITE | ((row & 0x1F) << 25) | ((col & 0x1F) << 20) | (value & 0xFF))


def fmt_comp_cmd(row: int, value: int) -> np.uint32:
    return np.uint32(OP_COMPUTE | ((row & 0x1F) << 25) | (value & 0xFF))


def is_header(word: int) -> bool:
    return (int(word) & HEADER_MASK) == HEADER_TAG


def _require_32_vector(vector: NDArray[np.integer]) -> NDArray[np.uint8]:
    arr = np.asarray(vector)
    if arr.shape != (X2_ROWS,):
        raise ValueError(f"Expected a 32-element vector, got {arr.shape}")
    return np.clip(arr, 0, 255).astype(np.uint8, copy=False)


def _require_32_matrix(matrix: NDArray[np.integer | np.bool_]) -> NDArray[np.uint8]:
    arr = np.asarray(matrix)
    if arr.shape != (X2_ROWS, X2_COLS):
        raise ValueError(f"Expected a 32x32 matrix, got {arr.shape}")
    return np.where(arr != 0, 0xFF, 0x00).astype(np.uint8, copy=False)


def make_wr_matrix_packet(matrix: NDArray[np.integer | np.bool_], inst_id: int = 0) -> NDArray[np.uint32]:
    mat = _require_32_matrix(matrix)
    header = np.uint32(HEADER_TAG | (1 << int(inst_id)))
    packet = np.empty((X2_ROWS, X2_COLS + 1), dtype=np.uint32)
    packet[:, 0] = header
    for row in range(X2_ROWS):
        for col in range(X2_COLS):
            packet[row, col + 1] = fmt_wr_cmd(row, col, int(mat[row, col]))
    return packet.reshape(-1)


def make_vmm_packet(vector: NDArray[np.integer], inst_id: int = 0) -> NDArray[np.uint32]:
    vec = _require_32_vector(vector)
    packet = np.empty(X2_ROWS + 1, dtype=np.uint32)
    packet[0] = np.uint32(HEADER_TAG | (1 << int(inst_id)))
    for row in range(X2_ROWS):
        packet[row + 1] = fmt_comp_cmd(row, int(vec[row]))
    return packet


def extract_vmm_payload(words: NDArray[np.integer], limit: int = X2_COLS) -> NDArray[np.uint16]:
    payload: list[int] = []
    for raw in np.asarray(words, dtype=np.uint32):
        word = int(raw)
        if word == EMPTY_TOKEN or is_header(word):
            continue
        payload.append(word & 0xFFFF)
        if len(payload) == limit:
            break
    return np.asarray(payload, dtype=np.uint16)


class BinaryX2Backend(Protocol):
    """Smallest hardware interface needed by the bit-plane scheduler."""

    num_devices: int

    def reset_all(self) -> None:
        ...

    def program_binary_matrix(self, inst_id: int, matrix: NDArray[np.integer | np.bool_]) -> None:
        ...

    def run_binary_vmm(self, inst_id: int, vector: NDArray[np.integer]) -> NDArray[np.uint16]:
        ...


class NumpyBinaryX2Backend:
    """Exact software model of the current 32x32 binary X2 RTL behavior."""

    def __init__(self, num_devices: int = N_X2) -> None:
        self.num_devices = max(1, min(int(num_devices), N_X2))
        self._matrices = np.zeros((self.num_devices, X2_ROWS, X2_COLS), dtype=np.uint8)

    def reset_all(self) -> None:
        self._matrices.fill(0)

    def program_binary_matrix(self, inst_id: int, matrix: NDArray[np.integer | np.bool_]) -> None:
        self._check_inst(inst_id)
        self._matrices[int(inst_id)] = (_require_32_matrix(matrix) > 0x7F).astype(np.uint8)

    def run_binary_vmm(self, inst_id: int, vector: NDArray[np.integer]) -> NDArray[np.uint16]:
        self._check_inst(inst_id)
        vec = _require_32_vector(vector).astype(np.uint16)
        out = vec.astype(np.uint32) @ self._matrices[int(inst_id)].astype(np.uint32)
        return out.astype(np.uint16)

    def _check_inst(self, inst_id: int) -> None:
        if not 0 <= int(inst_id) < self.num_devices:
            raise ValueError(f"X2 instance {inst_id} is outside 0..{self.num_devices - 1}")


class PynqDmaBinaryX2Backend:
    """
    PYNQ DMA backend for the generated ZCU106 bitstream.

    This class must run on the ZCU106 Linux/PYNQ image. It lazy-imports PYNQ so
    the same module can be tested on a normal workstation with the NumPy backend.
    """

    def __init__(
        self,
        bit_path: str | Path,
        *,
        dma_name: str = "axi_dma_x2_simd",
        num_devices: int = N_X2,
        rx_words: int = 1056,
    ) -> None:
        from pynq import Overlay, allocate  # type: ignore

        self._allocate = allocate
        self.num_devices = max(1, min(int(num_devices), N_X2))
        self.rx_words = int(rx_words)
        self.overlay = Overlay(str(Path(bit_path)))
        self.dma = getattr(self.overlay, dma_name)

    def reset_all(self) -> None:
        zero = np.zeros((X2_ROWS, X2_COLS), dtype=np.uint8)
        for inst_id in range(self.num_devices):
            self.program_binary_matrix(inst_id, zero)

    def program_binary_matrix(self, inst_id: int, matrix: NDArray[np.integer | np.bool_]) -> None:
        self._check_inst(inst_id)
        self._dma_transfer(make_wr_matrix_packet(matrix, inst_id=inst_id), rx_words=self.rx_words)

    def run_binary_vmm(self, inst_id: int, vector: NDArray[np.integer]) -> NDArray[np.uint16]:
        self._check_inst(inst_id)
        rx = self._dma_transfer(make_vmm_packet(vector, inst_id=inst_id), rx_words=max(64, X2_COLS + 2))
        payload = extract_vmm_payload(rx)
        if payload.shape != (X2_COLS,):
            raise RuntimeError(f"Expected 32 VMM outputs from X2 instance {inst_id}, got {payload.shape[0]}")
        return payload

    def _dma_transfer(self, tx_words: NDArray[np.uint32], *, rx_words: int) -> NDArray[np.uint32]:
        tx = self._allocate(shape=(len(tx_words),), dtype=np.uint32)
        rx = self._allocate(shape=(int(rx_words),), dtype=np.uint32)
        try:
            tx[:] = np.asarray(tx_words, dtype=np.uint32)
            rx[:] = np.uint32(EMPTY_TOKEN)
            rx.flush()
            tx.flush()

            self.dma.recvchannel.transfer(rx)
            self.dma.sendchannel.transfer(tx)
            self.dma.sendchannel.wait()
            self.dma.recvchannel.wait()
            rx.invalidate()
            return np.asarray(rx, dtype=np.uint32).copy()
        finally:
            tx.freebuffer()
            rx.freebuffer()

    def _check_inst(self, inst_id: int) -> None:
        if not 0 <= int(inst_id) < self.num_devices:
            raise ValueError(f"X2 instance {inst_id} is outside 0..{self.num_devices - 1}")


@dataclass(frozen=True)
class WorkEstimate:
    input_rows: int
    input_cols: int
    output_cols: int
    k_tiles: int
    output_tile_groups: int
    weight_programs: int
    vmm_calls: int


class Int8BitPlaneScheduler:
    """
    Map signed INT8 linear layers onto the binary 32x32 X2 primitive.

    The current RTL stores 1 bit per weight. Signed INT8 weights are represented
    as 8 two's-complement bit planes. The sign plane contributes -128, while
    lower planes contribute +1, +2, ..., +64.
    """

    def __init__(self, backend: BinaryX2Backend, *, cache_planes: bool = True) -> None:
        self.backend = backend
        self.cache_planes = bool(cache_planes)
        self._plane_cache: dict[int, bytes] = {}

    def reset(self) -> None:
        self.backend.reset_all()
        self._plane_cache.clear()

    def estimate_matmul(self, inputs: NDArray[np.integer], weights: NDArray[np.integer], *, signed_inputs: bool) -> WorkEstimate:
        x = _as_2d(inputs, "inputs")
        w = _as_2d(weights, "weights")
        if x.shape[1] != w.shape[0]:
            raise ValueError(f"Dimension mismatch: inputs {x.shape}, weights {w.shape}")
        batch, k_dim = x.shape
        out_cols = w.shape[1]
        k_tiles = (k_dim + X2_ROWS - 1) // X2_ROWS
        group_width = X2_COLS * int(self.backend.num_devices)
        output_tile_groups = (out_cols + group_width - 1) // group_width
        weight_programs = 0
        for col_start in range(0, out_cols, group_width):
            active = min(int(self.backend.num_devices), (out_cols - col_start + X2_COLS - 1) // X2_COLS)
            weight_programs += k_tiles * 8 * active
        input_passes = 2 if signed_inputs else 1
        return WorkEstimate(
            input_rows=batch,
            input_cols=k_dim,
            output_cols=out_cols,
            k_tiles=k_tiles,
            output_tile_groups=output_tile_groups,
            weight_programs=weight_programs,
            vmm_calls=weight_programs * batch * input_passes,
        )

    def matmul_u8_i8(self, inputs_u8: NDArray[np.integer], weights_i8: NDArray[np.integer]) -> NDArray[np.int32]:
        x = _as_2d(inputs_u8, "inputs_u8")
        x_pos = np.clip(x, 0, 255).astype(np.uint8)
        return self._matmul_bitplanes(x_pos, None, weights_i8)

    def matmul_i8_i8(self, inputs_i8: NDArray[np.integer], weights_i8: NDArray[np.integer]) -> NDArray[np.int32]:
        x = _as_2d(inputs_i8, "inputs_i8").astype(np.int16)
        x_pos = np.maximum(x, 0).astype(np.uint8)
        x_neg = np.maximum(-x, 0).astype(np.uint8)
        return self._matmul_bitplanes(x_pos, x_neg, weights_i8)

    def conv2d_tflite_u8_i8(
        self,
        input_nhwc_u8: NDArray[np.integer],
        weights_ohwi_i8: NDArray[np.integer],
        *,
        bias_i32: NDArray[np.integer] | None = None,
        stride: tuple[int, int] = (1, 1),
        padding: str = "same",
        pad_value: int = 0,
    ) -> NDArray[np.int32]:
        x = _as_nhwc(input_nhwc_u8, "input_nhwc_u8")
        w = np.asarray(weights_ohwi_i8, dtype=np.int8)
        if w.ndim != 4:
            raise ValueError(f"Expected TFLite conv weights [O,H,W,I], got {w.shape}")
        out_ch, kh, kw, in_ch = w.shape
        if x.shape[-1] != in_ch:
            raise ValueError(f"Input channels {x.shape[-1]} do not match weight channels {in_ch}")

        cols, out_hw = im2col_nhwc_u8(x, (kh, kw), stride=stride, padding=padding, pad_value=pad_value)
        weight_matrix = np.transpose(w, (1, 2, 3, 0)).reshape(kh * kw * in_ch, out_ch)
        out = self.matmul_u8_i8(cols, weight_matrix)
        if bias_i32 is not None:
            out += np.asarray(bias_i32, dtype=np.int32).reshape(1, -1)
        return out.reshape(x.shape[0], out_hw[0], out_hw[1], out_ch)

    def depthwise_conv2d_tflite_u8_i8(
        self,
        input_nhwc_u8: NDArray[np.integer],
        weights_1hwim_i8: NDArray[np.integer],
        *,
        bias_i32: NDArray[np.integer] | None = None,
        stride: tuple[int, int] = (1, 1),
        padding: str = "same",
        pad_value: int = 0,
    ) -> NDArray[np.int32]:
        x = _as_nhwc(input_nhwc_u8, "input_nhwc_u8")
        w = np.asarray(weights_1hwim_i8, dtype=np.int8)
        if w.ndim != 4 or w.shape[0] != 1:
            raise ValueError(f"Expected TFLite depthwise weights [1,H,W,I*M], got {w.shape}")
        _, kh, kw, in_mul = w.shape
        in_ch = x.shape[-1]
        if in_mul % in_ch != 0:
            raise ValueError(f"Depthwise weight channel dimension {in_mul} is not a multiple of input channels {in_ch}")
        depth_multiplier = in_mul // in_ch
        out_ch = in_ch * depth_multiplier

        cols, out_hw = im2col_nhwc_u8(x, (kh, kw), stride=stride, padding=padding, pad_value=pad_value)
        weight_matrix = np.zeros((kh * kw * in_ch, out_ch), dtype=np.int8)
        w_hwcm = w.reshape(kh, kw, in_ch, depth_multiplier)
        for ky in range(kh):
            for kx in range(kw):
                for c in range(in_ch):
                    row = ((ky * kw) + kx) * in_ch + c
                    for m in range(depth_multiplier):
                        weight_matrix[row, c * depth_multiplier + m] = w_hwcm[ky, kx, c, m]

        out = self.matmul_u8_i8(cols, weight_matrix)
        if bias_i32 is not None:
            out += np.asarray(bias_i32, dtype=np.int32).reshape(1, -1)
        return out.reshape(x.shape[0], out_hw[0], out_hw[1], out_ch)

    def _matmul_bitplanes(
        self,
        x_pos_u8: NDArray[np.uint8],
        x_neg_u8: NDArray[np.uint8] | None,
        weights_i8: NDArray[np.integer],
    ) -> NDArray[np.int32]:
        x_pos = _as_2d(x_pos_u8, "x_pos_u8").astype(np.uint8, copy=False)
        x_neg = None if x_neg_u8 is None else _as_2d(x_neg_u8, "x_neg_u8").astype(np.uint8, copy=False)
        weights = _as_2d(weights_i8, "weights_i8").astype(np.int8, copy=False)
        if x_pos.shape[1] != weights.shape[0]:
            raise ValueError(f"Dimension mismatch: inputs {x_pos.shape}, weights {weights.shape}")
        if x_neg is not None and x_neg.shape != x_pos.shape:
            raise ValueError(f"x_neg shape {x_neg.shape} does not match x_pos shape {x_pos.shape}")

        batch, k_dim = x_pos.shape
        out_cols = weights.shape[1]
        out = np.zeros((batch, out_cols), dtype=np.int32)
        devices = int(self.backend.num_devices)
        group_width = X2_COLS * devices

        for col_start in range(0, out_cols, group_width):
            active_count = min(devices, (out_cols - col_start + X2_COLS - 1) // X2_COLS)
            for k_start in range(0, k_dim, X2_ROWS):
                x_pos_tile = _pad_vector_batch(x_pos[:, k_start : k_start + X2_ROWS])
                x_neg_tile = None if x_neg is None else _pad_vector_batch(x_neg[:, k_start : k_start + X2_ROWS])

                for bit in range(8):
                    coeff = -128 if bit == 7 else (1 << bit)
                    any_plane = False
                    logical_cols: list[slice] = []

                    for dev in range(active_count):
                        dev_col_start = col_start + dev * X2_COLS
                        dev_col_end = min(dev_col_start + X2_COLS, out_cols)
                        logical_cols.append(slice(dev_col_start, dev_col_end))
                        tile = np.zeros((X2_ROWS, X2_COLS), dtype=np.int8)
                        weight_chunk = weights[k_start : k_start + X2_ROWS, dev_col_start:dev_col_end]
                        tile[: weight_chunk.shape[0], : weight_chunk.shape[1]] = weight_chunk
                        plane = ((tile.astype(np.uint8) >> bit) & 1).astype(np.uint8)
                        if np.any(plane):
                            any_plane = True
                        self._program_plane(dev, plane)

                    if not any_plane:
                        continue

                    for row in range(batch):
                        pos_out = self._run_active_devices(active_count, x_pos_tile[row])
                        if x_neg_tile is None:
                            delta = pos_out
                        else:
                            neg_out = self._run_active_devices(active_count, x_neg_tile[row])
                            delta = pos_out - neg_out

                        for dev, col_slice in enumerate(logical_cols):
                            width = col_slice.stop - col_slice.start
                            start = dev * X2_COLS
                            out[row, col_slice] += coeff * delta[start : start + width]

        return out

    def _program_plane(self, inst_id: int, plane: NDArray[np.uint8]) -> None:
        packed = np.packbits(plane.astype(np.uint8).reshape(-1)).tobytes()
        if self.cache_planes and self._plane_cache.get(int(inst_id)) == packed:
            return
        self.backend.program_binary_matrix(inst_id, plane)
        self._plane_cache[int(inst_id)] = packed

    def _run_active_devices(self, active_count: int, vector32: NDArray[np.uint8]) -> NDArray[np.int32]:
        parts = [self.backend.run_binary_vmm(dev, vector32).astype(np.int32) for dev in range(active_count)]
        return np.concatenate(parts, axis=0) if parts else np.zeros((0,), dtype=np.int32)


def _as_2d(arr: NDArray[np.integer], name: str) -> NDArray[np.integer]:
    out = np.asarray(arr)
    if out.ndim == 1:
        return out.reshape(1, -1)
    if out.ndim != 2:
        raise ValueError(f"Expected {name} to be 1D or 2D, got {out.shape}")
    return out


def _as_nhwc(arr: NDArray[np.integer], name: str) -> NDArray[np.integer]:
    out = np.asarray(arr)
    if out.ndim != 4:
        raise ValueError(f"Expected {name} to be NHWC, got {out.shape}")
    return out


def _pad_vector_batch(batch: NDArray[np.integer]) -> NDArray[np.uint8]:
    out = np.zeros((batch.shape[0], X2_ROWS), dtype=np.uint8)
    out[:, : batch.shape[1]] = np.clip(batch, 0, 255).astype(np.uint8)
    return out


def im2col_nhwc_u8(
    input_nhwc_u8: NDArray[np.integer],
    kernel_hw: tuple[int, int],
    *,
    stride: tuple[int, int] = (1, 1),
    padding: str = "same",
    pad_value: int = 0,
) -> tuple[NDArray[np.uint8], tuple[int, int]]:
    x = _as_nhwc(input_nhwc_u8, "input_nhwc_u8")
    x_u8 = np.clip(x, 0, 255).astype(np.uint8)
    batch, height, width, channels = x_u8.shape
    kh, kw = int(kernel_hw[0]), int(kernel_hw[1])
    sy, sx = int(stride[0]), int(stride[1])
    if kh <= 0 or kw <= 0 or sy <= 0 or sx <= 0:
        raise ValueError("Kernel and stride dimensions must be positive")

    mode = padding.lower()
    if mode == "same":
        out_h = int(np.ceil(height / sy))
        out_w = int(np.ceil(width / sx))
        pad_h = max((out_h - 1) * sy + kh - height, 0)
        pad_w = max((out_w - 1) * sx + kw - width, 0)
        pad_top = pad_h // 2
        pad_bottom = pad_h - pad_top
        pad_left = pad_w // 2
        pad_right = pad_w - pad_left
    elif mode == "valid":
        out_h = (height - kh + sy) // sy
        out_w = (width - kw + sx) // sx
        pad_top = pad_bottom = pad_left = pad_right = 0
    else:
        raise ValueError("padding must be 'same' or 'valid'")

    if out_h <= 0 or out_w <= 0:
        raise ValueError(f"Convolution output would be empty: {(out_h, out_w)}")

    padded = np.pad(
        x_u8,
        ((0, 0), (pad_top, pad_bottom), (pad_left, pad_right), (0, 0)),
        mode="constant",
        constant_values=int(np.clip(pad_value, 0, 255)),
    )

    windows = np.lib.stride_tricks.sliding_window_view(padded, (kh, kw), axis=(1, 2))
    y_idx = np.arange(out_h, dtype=np.int64) * sy
    x_idx = np.arange(out_w, dtype=np.int64) * sx
    selected = windows[:, y_idx, :, :, :, :][:, :, x_idx, :, :, :]
    cols = selected.transpose(0, 1, 2, 4, 5, 3).reshape(batch * out_h * out_w, kh * kw * channels)
    return np.ascontiguousarray(cols, dtype=np.uint8), (out_h, out_w)
