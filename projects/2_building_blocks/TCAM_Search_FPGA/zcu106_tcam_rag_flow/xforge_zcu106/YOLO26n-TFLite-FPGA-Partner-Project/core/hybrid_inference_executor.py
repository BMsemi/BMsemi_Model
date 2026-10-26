from __future__ import annotations

import argparse
import contextlib
import csv
import io
import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
try:
    import ai_edge_litert.interpreter as Interpreter
except ImportError:
    try:
        import tflite_runtime.interpreter as Interpreter
    except ImportError:
        from tensorflow.lite.python import interpreter as Interpreter

from .Mapper import MaestroClusterMapper, PE_OUT_COLS
from .Multiple_PE_core import MultiPENeuromorphicCore
from .Scheduler import MaestroClusterScheduler
from .tflite_constants import TFLiteConstantReader
from user import (
    _extract_patch_from_padded_2d,
    _extract_patch_same_2d,
    load_rgb_image,
    make_weight_provider,
)


def _safe_print(*args, **kwargs) -> None:
    try:
        print(*args, **kwargs)
    except BrokenPipeError:
        return


@dataclass(frozen=True)
class TensorMeta:
    index: int
    name: str
    shape: tuple[int, ...]
    dtype: Any
    scale: float
    zero_point: int
    scales: np.ndarray
    zero_points: np.ndarray
    quantized_dimension: int


@dataclass
class OpRecord:
    index: int
    op_name: str
    backend: str
    layer_name: str
    input_indices: list[int]
    output_indices: list[int]
    note: str = ""
    duration_ms: float = 0.0


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def _to_numpy_dtype(dtype: Any) -> np.dtype:
    return np.dtype(dtype)


def _is_quantized_int(meta: TensorMeta) -> bool:
    return np.issubdtype(_to_numpy_dtype(meta.dtype), np.integer) and meta.scale > 0.0


def _is_quantized_weight(meta: TensorMeta) -> bool:
    return np.issubdtype(_to_numpy_dtype(meta.dtype), np.integer) and (
        meta.scale > 0.0 or bool(np.any(meta.scales > 0.0))
    )


def _is_float_tensor(meta: TensorMeta) -> bool:
    return np.issubdtype(_to_numpy_dtype(meta.dtype), np.floating)


def _make_tensor_meta(detail: dict) -> TensorMeta:
    qparams = detail.get("quantization_parameters", {})
    scales = np.asarray(qparams.get("scales", []), dtype=np.float32)
    zero_points = np.asarray(qparams.get("zero_points", []), dtype=np.int32)
    scale, zero_point = detail.get("quantization", (0.0, 0))
    return TensorMeta(
        index=int(detail["index"]),
        name=str(detail["name"]),
        shape=tuple(int(v) for v in detail["shape"]),
        dtype=detail["dtype"],
        scale=float(scale),
        zero_point=int(zero_point),
        scales=scales,
        zero_points=zero_points,
        quantized_dimension=int(qparams.get("quantized_dimension", 0)),
    )


def _dequantize(arr: np.ndarray, meta: TensorMeta) -> np.ndarray:
    if not _is_quantized_int(meta):
        return np.asarray(arr, dtype=np.float32)
    return meta.scale * (np.asarray(arr, dtype=np.float32) - float(meta.zero_point))


def _requantize(real: np.ndarray, meta: TensorMeta) -> np.ndarray:
    if not _is_quantized_int(meta):
        return np.asarray(real, dtype=_to_numpy_dtype(meta.dtype))

    q = np.rint(np.asarray(real, dtype=np.float32) / meta.scale + float(meta.zero_point))
    dtype = _to_numpy_dtype(meta.dtype)
    if dtype == np.dtype(np.int8):
        q = np.clip(q, -128, 127)
    elif dtype == np.dtype(np.uint8):
        q = np.clip(q, 0, 255)
    elif dtype == np.dtype(np.int16):
        q = np.clip(q, -32768, 32767)
    return q.astype(dtype)


def _infer_concat_axis(inputs: list[np.ndarray], output_shape: tuple[int, ...]) -> int:
    rank = len(output_shape)
    candidates: list[int] = []
    for axis in range(rank):
        matches = True
        total = 0
        for arr in inputs:
            shape = arr.shape
            if len(shape) != rank:
                matches = False
                break
            for dim in range(rank):
                if dim == axis:
                    continue
                if shape[dim] != output_shape[dim]:
                    matches = False
                    break
            if not matches:
                break
            total += shape[axis]
        if matches and total == output_shape[axis]:
            candidates.append(axis)
    if len(candidates) != 1:
        raise ValueError(f"Could not infer concat axis for output shape {output_shape}")
    return candidates[0]


def _slice_tensor(data: np.ndarray, begin: np.ndarray, size: np.ndarray) -> np.ndarray:
    arr = np.asarray(data)
    begin_i = np.asarray(begin, dtype=np.int64).reshape(-1)
    size_i = np.asarray(size, dtype=np.int64).reshape(-1)
    if begin_i.size != arr.ndim or size_i.size != arr.ndim:
        raise ValueError(f"SLICE rank mismatch: input rank {arr.ndim}, begin {begin_i.shape}, size {size_i.shape}")

    slices = []
    for dim, (start, length) in enumerate(zip(begin_i, size_i)):
        dim_len = arr.shape[dim]
        if start < 0:
            start += dim_len
        stop = dim_len if length < 0 else start + length
        slices.append(slice(int(start), int(stop)))
    return arr[tuple(slices)]


def _max_pool2d_same_5x5(data: np.ndarray, output_shape: tuple[int, ...]) -> np.ndarray:
    arr = np.asarray(data)
    if arr.ndim != 4:
        raise ValueError(f"MAX_POOL_2D expects NHWC rank-4 input, got {arr.shape}")
    if len(output_shape) != 4:
        raise ValueError(f"MAX_POOL_2D expects rank-4 output, got {output_shape}")

    out_h, out_w = int(output_shape[1]), int(output_shape[2])
    stride_y = max(1, int(round(arr.shape[1] / float(out_h)))) if out_h else 1
    stride_x = max(1, int(round(arr.shape[2] / float(out_w)))) if out_w else 1
    kernel_h = 5
    kernel_w = 5
    pad_h = max((out_h - 1) * stride_y + kernel_h - arr.shape[1], 0)
    pad_w = max((out_w - 1) * stride_x + kernel_w - arr.shape[2], 0)
    pad_top = pad_h // 2
    pad_bottom = pad_h - pad_top
    pad_left = pad_w // 2
    pad_right = pad_w - pad_left

    if np.issubdtype(arr.dtype, np.floating):
        pad_value = -np.inf
    elif np.issubdtype(arr.dtype, np.integer):
        pad_value = np.iinfo(arr.dtype).min
    else:
        pad_value = 0

    padded = np.pad(
        arr,
        ((0, 0), (pad_top, pad_bottom), (pad_left, pad_right), (0, 0)),
        mode="constant",
        constant_values=pad_value,
    )
    windows = np.lib.stride_tricks.sliding_window_view(padded, (kernel_h, kernel_w), axis=(1, 2))
    y_idx = np.arange(out_h, dtype=np.int64) * stride_y
    x_idx = np.arange(out_w, dtype=np.int64) * stride_x
    selected = windows[:, y_idx, :, :, :, :][:, :, x_idx, :, :, :]
    return np.max(selected, axis=(-2, -1)).astype(arr.dtype, copy=False)


def _resize_nearest_neighbor(data: np.ndarray, size: np.ndarray, output_shape: tuple[int, ...]) -> np.ndarray:
    arr = np.asarray(data)
    target = np.asarray(size, dtype=np.int64).reshape(-1)
    if target.size != 2:
        raise ValueError(f"RESIZE_NEAREST_NEIGHBOR expects size [height, width], got {target}")
    out_h, out_w = int(target[0]), int(target[1])
    if len(output_shape) == 4:
        out_h, out_w = int(output_shape[1]), int(output_shape[2])

    if arr.ndim != 4:
        raise ValueError(f"RESIZE_NEAREST_NEIGHBOR expects rank-4 input, got {arr.shape}")

    # YOLO internal tensors are NHWC; keep a small NCHW fallback for safety.
    if len(output_shape) == 4 and output_shape[1] == out_h and output_shape[2] == out_w:
        in_h, in_w = arr.shape[1], arr.shape[2]
        y_idx = np.floor(np.arange(out_h, dtype=np.float32) * in_h / float(out_h)).astype(np.int64)
        x_idx = np.floor(np.arange(out_w, dtype=np.float32) * in_w / float(out_w)).astype(np.int64)
        return arr[:, y_idx, :, :][:, :, x_idx, :].astype(arr.dtype, copy=False)

    in_h, in_w = arr.shape[2], arr.shape[3]
    y_idx = np.floor(np.arange(out_h, dtype=np.float32) * in_h / float(out_h)).astype(np.int64)
    x_idx = np.floor(np.arange(out_w, dtype=np.float32) * in_w / float(out_w)).astype(np.int64)
    return arr[:, :, y_idx, :][:, :, :, x_idx].astype(arr.dtype, copy=False)


class CrossbarConvRunner:
    def __init__(self, mapping_path: str | Path, *, num_pes: int = 512) -> None:
        self.mapping_path = Path(mapping_path)
        self.mapper = MaestroClusterMapper()
        self.scheduler = MaestroClusterScheduler()
        with contextlib.redirect_stdout(io.StringIO()):
            self.mpe = MultiPENeuromorphicCore(num_pes=num_pes, input_size=64, output_size=64, iterations=0)
        self.pe_backend_name = os.environ.get("YOLO_PE_BACKEND", "pl_dma").strip().lower() or "pl_dma"
        self.pl_backend = None
        if self.pe_backend_name in {"pl", "pl_dma", "dma", "mmio", "axi_mvm"}:
            from .pl_dma_backend import AxiX2MvmCmaBackend, PlBackendUnavailable

            try:
                self.pl_backend = AxiX2MvmCmaBackend()
            except PlBackendUnavailable:
                raise
            except Exception as exc:
                raise PlBackendUnavailable(str(exc)) from exc
            _safe_print("[PL-DMA] Real AXI/MMIO + DMA PE backend is active.", flush=True)
        else:
            raise ValueError(
                f"Crossbar layers must run on the real PL-DMA backend; "
                f"YOLO_PE_BACKEND={self.pe_backend_name!r} is not allowed."
            )

    def record_backend(self, *, use_pl: bool = True) -> str:
        return "pl_dma" if use_pl else "software_conv"

    def _prepare_weights(
        self,
        op_name: str,
        weight_q: np.ndarray,
    ) -> np.ndarray:
        if op_name == "CONV_2D":
            # TFLite Conv: [K, R, S, C] -> [K, C, R, S]
            return np.transpose(weight_q, (0, 3, 1, 2)).astype(np.int8, copy=False)
        if op_name == "DEPTHWISE_CONV_2D":
            # TFLite Depthwise: [1, R, S, C] -> [C, 1, R, S]
            return np.transpose(weight_q, (3, 0, 1, 2)).astype(np.int8, copy=False)
        raise ValueError(f"Unsupported crossbar op: {op_name}")

    def _input_is_pre_padded(self, input_chw_u8: np.ndarray, mdims) -> bool:
        if int(mdims.R) <= 1 and int(mdims.S) <= 1:
            return False
        h, w = int(input_chw_u8.shape[1]), int(input_chw_u8.shape[2])
        return h == int(mdims.Y) + int(mdims.R) - 1 and w == int(mdims.X) + int(mdims.S) - 1

    def _extract_patch_u8(
        self,
        img2d: np.ndarray,
        y: int,
        x: int,
        r: int,
        s: int,
        *,
        pre_padded: bool,
        pad_fill: int,
    ) -> np.ndarray:
        y = int(y)
        x = int(x)
        r = int(r)
        s = int(s)
        fill = int(np.clip(pad_fill, 0, 255))

        if pre_padded:
            patch = np.full((r, s), fill, dtype=np.uint8)
            y1 = y + r
            x1 = x + s
            h, w = img2d.shape
            sy0 = max(0, y)
            sx0 = max(0, x)
            sy1 = min(h, y1)
            sx1 = min(w, x1)
            if sy1 > sy0 and sx1 > sx0:
                patch[sy0 - y : sy1 - y, sx0 - x : sx1 - x] = img2d[sy0:sy1, sx0:sx1]
            return patch.reshape(-1)

        pad_y = r // 2
        pad_x = s // 2
        padded = np.pad(img2d, ((pad_y, pad_y), (pad_x, pad_x)), mode="constant", constant_values=fill)
        yy = y + pad_y
        xx = x + pad_x
        return padded[yy - pad_y : yy - pad_y + r, xx - pad_x : xx - pad_x + s].reshape(-1)

    def _im2col_typed(
        self,
        input_chw: np.ndarray,
        mdims,
        *,
        pre_padded: bool,
        pad_fill: int | float,
        out_y: int,
        out_x: int,
        dtype: np.dtype,
    ) -> np.ndarray:
        expected_c = int(mdims.C)
        dtype = np.dtype(dtype)
        if np.issubdtype(dtype, np.integer):
            info = np.iinfo(dtype)
            fill = int(np.clip(pad_fill, info.min, info.max))
        else:
            fill = float(pad_fill)
        x = np.asarray(input_chw, dtype=dtype)
        if x.shape[0] < expected_c:
            pad_c = expected_c - x.shape[0]
            x = np.pad(x, ((0, pad_c), (0, 0), (0, 0)), mode="constant", constant_values=fill)
        elif x.shape[0] > expected_c:
            x = x[:expected_c]

        r = int(mdims.R)
        s = int(mdims.S)
        stride_y = int(mdims.stride_y)
        stride_x = int(mdims.stride_x)

        if pre_padded:
            padded = x
        else:
            padded = np.pad(x, ((0, 0), (r // 2, r // 2), (s // 2, s // 2)), mode="constant", constant_values=fill)

        needed_h = (int(out_y) - 1) * stride_y + r
        needed_w = (int(out_x) - 1) * stride_x + s
        extra_h = max(0, needed_h - padded.shape[1])
        extra_w = max(0, needed_w - padded.shape[2])
        if extra_h or extra_w:
            padded = np.pad(padded, ((0, 0), (0, extra_h), (0, extra_w)), mode="constant", constant_values=fill)

        windows = np.lib.stride_tricks.sliding_window_view(padded, (r, s), axis=(1, 2))
        y_idx = np.arange(int(out_y), dtype=np.int64) * stride_y
        x_idx = np.arange(int(out_x), dtype=np.int64) * stride_x
        selected = windows[:, y_idx, :, :, :][:, :, x_idx, :, :]
        return np.ascontiguousarray(selected.transpose(1, 2, 0, 3, 4).reshape(int(out_y) * int(out_x), -1))

    def _im2col_u8(
        self,
        input_chw_u8: np.ndarray,
        mdims,
        *,
        pre_padded: bool,
        pad_fill: int,
        out_y: int,
        out_x: int,
    ) -> np.ndarray:
        return self._im2col_typed(
            input_chw_u8,
            mdims,
            pre_padded=pre_padded,
            pad_fill=pad_fill,
            out_y=out_y,
            out_x=out_x,
            dtype=np.uint8,
        )

    def _run_crossbar_accum_vectorized(
        self,
        *,
        layer_name: str,
        input_chw_u8: np.ndarray,
        weight_provider,
        mdims,
        actual_output_channels: int,
        pad_fill: int,
        pre_padded: bool,
        signed_input: bool = False,
        use_pl: bool = True,
    ) -> np.ndarray:
        out_y = (int(mdims.Y) + int(mdims.stride_y) - 1) // int(mdims.stride_y)
        out_x = (int(mdims.X) + int(mdims.stride_x) - 1) // int(mdims.stride_x)
        input_dtype = np.int8 if signed_input else np.uint8
        cols = self._im2col_typed(
            input_chw_u8,
            mdims,
            pre_padded=pre_padded,
            pad_fill=pad_fill,
            out_y=out_y,
            out_x=out_x,
            dtype=input_dtype,
        )
        weights = np.rint(weight_provider(0, int(actual_output_channels))).astype(np.int32, copy=False)
        if cols.shape[1] != weights.shape[0]:
            raise ValueError(
                f"{layer_name} im2col width {cols.shape[1]} does not match weight rows {weights.shape[0]}"
            )

        _safe_print(
            f"[CROSSBAR] {layer_name}: {out_y * out_x:,} positions "
            f"x {int(actual_output_channels)} channels via {self.record_backend(use_pl=use_pl)}",
            flush=True,
        )
        if use_pl and self.pl_backend is None:
            raise RuntimeError("PL-DMA backend is not open; refusing PS software crossbar execution.")
        if not use_pl:
            accum_2d = cols.astype(np.int32, copy=False) @ weights.astype(np.int32, copy=False)
        elif signed_input:
            accum_2d = self.pl_backend.matmul_i8_i8(
                cols.astype(np.int8, copy=False),
                weights.astype(np.int8, copy=False),
                layer_name=layer_name,
            )
        else:
            accum_2d = self.pl_backend.matmul_u8_i8(
                cols.astype(np.uint8, copy=False),
                weights.astype(np.int8, copy=False),
                layer_name=layer_name,
            )
        return accum_2d.reshape(out_y, out_x, int(actual_output_channels)).astype(np.int32, copy=False)

    def _run_crossbar_accum(
        self,
        *,
        layer_name: str,
        op_name: str,
        input_chw_u8: np.ndarray,
        weight_q: np.ndarray,
        pad_fill: int = 0,
        signed_input: bool = False,
        use_pl: bool = True,
    ) -> tuple[np.ndarray, int, np.ndarray]:
        mdims, mmapping = self.mapper.map_layer_from_file(self.mapping_path, layer_name=layer_name)
        conv_weights = self._prepare_weights(op_name, np.asarray(weight_q))

        provider_mdims = mdims
        if op_name == "DEPTHWISE_CONV_2D":
            provider_mdims = type(mdims)(
                K=mdims.K,
                C=int(conv_weights.shape[0]),
                R=mdims.R,
                S=mdims.S,
                Y=mdims.Y,
                X=mdims.X,
                stride_x=mdims.stride_x,
                stride_y=mdims.stride_y,
            )

        with contextlib.redirect_stdout(io.StringIO()):
            weight_provider, is_depthwise, actual_output_channels, actual_dims = make_weight_provider(conv_weights, provider_mdims)

        # Patch scheduler dimensions to the real tensor dimensions for this layer.
        mdims = type(mdims)(
            K=actual_output_channels,
            C=actual_dims["C"],
            R=actual_dims["R"],
            S=actual_dims["S"],
            Y=mdims.Y,
            X=mdims.X,
            stride_x=mdims.stride_x,
            stride_y=mdims.stride_y,
        )
        mmapping.patch_len = int(actual_dims["patch_len"])
        mmapping.fold_sizes = [int(actual_dims["patch_len"])]

        rows_per_channel = max(1, int(actual_dims["R"]) * int(actual_dims["S"]))
        crossbar_rows = int(getattr(self.mpe, "input_size", 64))
        channels_per_pe = max(1, crossbar_rows // rows_per_channel)
        c_groups = int(np.ceil(int(actual_dims["C"]) / float(channels_per_pe)))
        num_clusters = max(1, int(getattr(mmapping, "num_clusters", 1)))
        required_pes = num_clusters * c_groups
        available_pes = int(getattr(self.mpe, "num_pes", required_pes))
        use_vectorized = True
        if required_pes > available_pes and not use_vectorized:
            raise ValueError(
                f"{layer_name} requires {required_pes} PEs to cover C*R*S={actual_dims['patch_len']} "
                f"with {crossbar_rows} input rows, but only {available_pes} are available."
            )

        # The PE has a fixed 64-row input vector.  Split large C*R*S patches
        # across channel groups so the scheduler never silently truncates rows.
        mmapping.cluster_size = max(int(getattr(mmapping, "cluster_size", 1)), c_groups)
        mmapping.inner_spatial_dim = "C"
        mmapping.inner_tile = channels_per_pe
        mmapping.inner_stride = channels_per_pe
        mmapping.num_pes_total = num_clusters * int(mmapping.cluster_size)
        mmapping.num_active_pes_total = required_pes

        if signed_input:
            input_chw_u8 = np.asarray(input_chw_u8, dtype=np.int8)
        else:
            input_chw_u8 = np.asarray(input_chw_u8, dtype=np.uint8)
        pre_padded = self._input_is_pre_padded(input_chw_u8, mdims)

        if use_vectorized:
            accum = self._run_crossbar_accum_vectorized(
                layer_name=layer_name,
                input_chw_u8=input_chw_u8,
                weight_provider=weight_provider,
                mdims=mdims,
                actual_output_channels=actual_output_channels,
                pad_fill=pad_fill,
                pre_padded=pre_padded,
                signed_input=signed_input,
                use_pl=use_pl,
            )
            weight_sums = conv_weights.reshape(conv_weights.shape[0], -1).sum(axis=1).astype(np.int32)
            return accum, actual_output_channels, weight_sums[:actual_output_channels]

        if signed_input:
            raise NotImplementedError("Signed activation replay requires the vectorized/PL path.")

        mops = list(self.scheduler.iter_ops(mdims, mmapping, weight_provider))
        total_compute_ops = sum(1 for op in mops if op.op == "COMPUTE_CLUSTER_C")
        progress_step = max(1, total_compute_ops // 10) if total_compute_ops >= 10_000 else 0
        next_progress = progress_step
        compute_count = 0
        out_y = (int(mdims.Y) + int(mdims.stride_y) - 1) // int(mdims.stride_y)
        out_x = (int(mdims.X) + int(mdims.stride_x) - 1) // int(mdims.stride_x)
        accum = np.zeros((out_y, out_x, actual_output_channels), dtype=np.int32)

        if total_compute_ops:
            print(f"[CROSSBAR] {layer_name}: {total_compute_ops:,} PE compute calls", flush=True)

        processed_positions: set[tuple[int, int, int, int]] = set()
        for op in mops:
            if op.op == "PROGRAM_CLUSTER":
                pe_id = int(op.info["pe_id"])
                weight_tile = np.asarray(op.info["weight_tile"])
                if weight_tile.ndim != 2:
                    continue
                tile_i8 = np.rint(weight_tile[:, :PE_OUT_COLS]).astype(np.int8, copy=False)
                if tile_i8.shape[0] < 64:
                    padded = np.zeros((64, tile_i8.shape[1]), dtype=np.int8)
                    padded[: tile_i8.shape[0], :] = tile_i8
                    tile_i8 = padded
                self.mpe.program_weights_one(pe_id, tile_i8)
                continue

            if op.op != "COMPUTE_CLUSTER_C":
                continue

            pe_id = int(op.info["pe_id"])
            c_start = int(op.info.get("c_start", 0))
            c_end = int(op.info.get("c_end", mdims.C))
            out_y_idx = int(op.info["oy"])
            out_x_idx = int(op.info["ox"])
            k_global = int(op.info["k_global"])
            k_count = int(op.info["k_count"])
            y_in = int(op.info["y_in"])
            x_in = int(op.info["x_in"])

            position_key = (out_y_idx, out_x_idx, k_global, c_start)
            if position_key in processed_positions:
                continue
            processed_positions.add(position_key)

            patches: list[np.ndarray] = []
            if int(mdims.R) == 1 and int(mdims.S) == 1:
                for c_idx in range(c_start, c_end):
                    pixel_val = int(np.clip(pad_fill, 0, 255))
                    if c_idx < input_chw_u8.shape[0] and 0 <= y_in < input_chw_u8.shape[1] and 0 <= x_in < input_chw_u8.shape[2]:
                        pixel_val = int(input_chw_u8[c_idx, y_in, x_in])
                    patches.append(np.asarray([pixel_val], dtype=np.uint8))
            else:
                for c_idx in range(c_start, c_end):
                    if c_idx >= input_chw_u8.shape[0]:
                        patch = np.full((int(mdims.R) * int(mdims.S),), int(np.clip(pad_fill, 0, 255)), dtype=np.uint8)
                    else:
                        patch = self._extract_patch_u8(
                            input_chw_u8[c_idx],
                            y_in,
                            x_in,
                            int(mdims.R),
                            int(mdims.S),
                            pre_padded=pre_padded,
                            pad_fill=pad_fill,
                        )
                    patches.append(np.asarray(np.clip(patch, 0, 255), dtype=np.uint8))

            patch_vec = np.concatenate(patches) if patches else np.zeros((0,), dtype=np.uint8)
            vec64 = np.zeros((64,), dtype=np.uint8)
            limit = min(len(patch_vec), 64)
            vec64[:limit] = patch_vec[:limit]

            out_full = np.asarray(self.mpe.compute_pe_fast(pe_id, vec64), dtype=np.int32)
            chunk_base = (k_global // PE_OUT_COLS) * PE_OUT_COLS
            local_base = k_global - chunk_base

            for k_offset in range(k_count):
                k_actual = k_global + k_offset
                if k_actual >= actual_output_channels:
                    break
                local_index = local_base + k_offset
                if local_index >= len(out_full):
                    break
                accum[out_y_idx, out_x_idx, k_actual] += int(out_full[local_index])

            compute_count += 1
            if progress_step and compute_count >= next_progress:
                pct = 100.0 * compute_count / float(total_compute_ops)
                print(f"[CROSSBAR] {layer_name}: {compute_count:,}/{total_compute_ops:,} ({pct:.0f}%)", flush=True)
                next_progress += progress_step

        if total_compute_ops:
            print(f"[CROSSBAR] {layer_name}: complete", flush=True)

        weight_sums = conv_weights.reshape(conv_weights.shape[0], -1).sum(axis=1).astype(np.int32)
        return accum, actual_output_channels, weight_sums[:actual_output_channels]

    def run_quantized_conv(
        self,
        *,
        layer_name: str,
        op_name: str,
        input_q: np.ndarray,
        input_zero_point: int,
        input_scale: float,
        weight_q: np.ndarray,
        weight_scales: np.ndarray,
        bias_q: np.ndarray | None,
        output_scale: float,
        output_zero_point: int,
        use_pl: bool = True,
    ) -> np.ndarray:
        input_arr = np.asarray(input_q)
        if np.dtype(input_arr.dtype) == np.dtype(np.int8):
            input_chw = input_arr.astype(np.int8, copy=False)[0].transpose(2, 0, 1)
            accum, actual_output_channels, weight_sums = self._run_crossbar_accum(
                layer_name=layer_name,
                op_name=op_name,
                input_chw_u8=input_chw,
                weight_q=weight_q,
                pad_fill=int(input_zero_point),
                signed_input=True,
                use_pl=use_pl,
            )
            accum = accum - (int(input_zero_point) * weight_sums.reshape(1, 1, -1))
        else:
            input_nhwc_i8 = np.asarray(input_q, dtype=np.int16)
            input_chw_u8 = np.clip(input_nhwc_i8[0] - int(input_zero_point), 0, 255).astype(np.uint8).transpose(2, 0, 1)
            accum, actual_output_channels, _ = self._run_crossbar_accum(
                layer_name=layer_name,
                op_name=op_name,
                input_chw_u8=input_chw_u8,
                weight_q=weight_q,
                pad_fill=0,
                use_pl=use_pl,
            )

        if bias_q is not None:
            accum = accum + np.asarray(bias_q, dtype=np.int32).reshape(1, 1, -1)

        scales = np.asarray(weight_scales, dtype=np.float32)
        if scales.size == 0:
            scales = np.asarray([1.0], dtype=np.float32)
        if scales.size == 1:
            scales = np.repeat(scales, actual_output_channels)

        real_multiplier = (float(input_scale) * scales[:actual_output_channels]) / float(output_scale)
        output_q = np.empty_like(accum, dtype=np.int32)
        for channel in range(actual_output_channels):
            output_q[:, :, channel] = np.rint(accum[:, :, channel].astype(np.float32) * real_multiplier[channel]) + int(output_zero_point)

        output_q = np.clip(output_q, -128, 127).astype(np.int8)
        return output_q[None, ...]

    def run_float_activation_conv(
        self,
        *,
        layer_name: str,
        op_name: str,
        input_float: np.ndarray,
        weight_q: np.ndarray,
        weight_scales: np.ndarray,
        bias_float: np.ndarray | None,
        use_pl: bool = True,
    ) -> np.ndarray:
        input_real = np.asarray(input_float, dtype=np.float32)
        if input_real.ndim != 4 or input_real.shape[0] != 1:
            raise ValueError(f"Expected NHWC input with batch 1, got {input_real.shape}")

        max_abs = float(np.max(np.abs(input_real))) if input_real.size else 0.0
        activation_scale = max(max_abs / 127.0, 1.0 / 127.0)

        # Signed INT8 activation q ~= real / scale. The PE consumes signed INT8
        # directly, so no unsigned high-byte correction pass is needed here.
        input_q = np.clip(np.rint(input_real / activation_scale), -128, 127).astype(np.int16)
        input_chw_i8 = input_q.astype(np.int8)[0].transpose(2, 0, 1)

        accum_signed, actual_output_channels, _ = self._run_crossbar_accum(
            layer_name=layer_name,
            op_name=op_name,
            input_chw_u8=input_chw_i8,
            weight_q=weight_q,
            pad_fill=0,
            signed_input=True,
            use_pl=use_pl,
        )

        scales = np.asarray(weight_scales, dtype=np.float32)
        if scales.size == 0:
            scales = np.asarray([1.0], dtype=np.float32)
        if scales.size == 1:
            scales = np.repeat(scales, actual_output_channels)

        output = np.empty(accum_signed.shape, dtype=np.float32)
        for channel in range(actual_output_channels):
            output[:, :, channel] = accum_signed[:, :, channel].astype(np.float32) * activation_scale * scales[channel]

        if bias_float is not None:
            output += np.asarray(bias_float, dtype=np.float32).reshape(1, 1, -1)

        return output[None, ...].astype(np.float32)


class TFLiteHybridExecutor:
    def __init__(
        self,
        model_path: str | Path,
        *,
        mapping_path: str | Path | None = None,
        num_pes: int = 512,
    ) -> None:
        self.model_path = Path(model_path)
        self.mapping_path = Path(mapping_path) if mapping_path is not None else None
        self.interpreter = Interpreter.Interpreter(
            model_path=str(self.model_path),
            experimental_preserve_all_tensors=True,
            experimental_op_resolver_type=Interpreter.OpResolverType.BUILTIN_WITHOUT_DEFAULT_DELEGATES,
        )
        self.interpreter.allocate_tensors()

        self.ops = list(self.interpreter._get_ops_details())
        self.tensor_meta = {
            int(detail["index"]): _make_tensor_meta(detail)
            for detail in self.interpreter.get_tensor_details()
        }
        self.input_detail = self.interpreter.get_input_details()[0]
        self.output_detail = self.interpreter.get_output_details()[0]
        self.reference_cache: dict[int, np.ndarray] = {}
        self.constant_reader = TFLiteConstantReader(self.model_path)
        self.conv_layer_names = self._build_conv_layer_names()
        self.crossbar_runner = CrossbarConvRunner(self.mapping_path, num_pes=num_pes) if self.mapping_path is not None else None

        produced = set()
        for op in self.ops:
            produced.update(int(idx) for idx in op.get("outputs", []))
        self.constant_tensor_indices = {idx for idx in self.tensor_meta if idx not in produced}

    def _build_conv_layer_names(self) -> dict[int, str]:
        names: dict[int, str] = {}
        mapped_idx = 0
        for op in self.ops:
            op_index = int(op["index"])
            op_name = str(op["op_name"])
            if op_name == "CONV_2D":
                names[op_index] = "Conv2D" if mapped_idx == 0 else f"Conv2D_{mapped_idx}"
                mapped_idx += 1
            elif op_name == "DEPTHWISE_CONV_2D":
                names[op_index] = f"DepthwiseConv2D_{mapped_idx}"
                mapped_idx += 1
        return names

    def crossbar_candidate_layers(self) -> list[str]:
        layers: list[str] = []
        for op in self.ops:
            op_name = str(op["op_name"])
            if op_name in {"CONV_2D", "DEPTHWISE_CONV_2D"}:
                layer_name = self.conv_layer_names.get(int(op["index"]), "")
                if layer_name:
                    layers.append(layer_name)
        return layers

    def crossbar_eligible_layers(self) -> tuple[set[str], dict[str, str]]:
        eligible: set[str] = set()
        ineligible: dict[str, str] = {}

        for op in self.ops:
            op_name = str(op["op_name"])
            if op_name not in {"CONV_2D", "DEPTHWISE_CONV_2D"}:
                continue

            layer_name = self.conv_layer_names.get(int(op["index"]), "")
            if not layer_name:
                continue

            if self.crossbar_runner is None:
                ineligible[layer_name] = "No mapping file was provided for PE/crossbar replay."
                continue

            input_ids = [int(idx) for idx in op.get("inputs", [])]
            output_ids = [int(idx) for idx in op.get("outputs", [])]
            if len(input_ids) < 2 or not output_ids:
                ineligible[layer_name] = "Conv/Depthwise op does not expose the expected input, weight, and output tensors."
                continue

            input_meta = self.tensor_meta[input_ids[0]]
            weight_meta = self.tensor_meta[input_ids[1]]
            output_meta = self.tensor_meta[output_ids[0]]

            input_is_supported = _is_quantized_int(input_meta) or _is_float_tensor(input_meta)
            output_is_supported = _is_quantized_int(output_meta) or _is_float_tensor(output_meta)

            if not input_is_supported:
                ineligible[layer_name] = (
                    f"Input tensor {input_meta.index} is {np.dtype(input_meta.dtype)} "
                    f"with activation scale {input_meta.scale}; expected integer activation tensor with scale > 0 "
                    "or float32 tensor for runtime INT8 bridge."
                )
                continue

            if not _is_quantized_weight(weight_meta):
                ineligible[layer_name] = (
                    f"Weight tensor {weight_meta.index} is {np.dtype(weight_meta.dtype)} "
                    "without scalar or per-channel quantization scales."
                )
                continue

            if not output_is_supported:
                ineligible[layer_name] = (
                    f"Output tensor {output_meta.index} is {np.dtype(output_meta.dtype)} "
                    f"with activation scale {output_meta.scale}; expected integer output tensor with scale > 0 "
                    "or float32 tensor for runtime INT8 bridge."
                )
                continue

            eligible.add(layer_name)

        return eligible, ineligible

    def preprocess_image(self, image_path: str | Path) -> tuple[np.ndarray, dict]:
        _, _, model_input_i8, meta = load_rgb_image(str(image_path), size=(640, 640), return_preprocessed=True)
        return np.asarray(model_input_i8, dtype=np.int8), meta

    def _get_reference_tensor(self, tensor_index: int) -> np.ndarray:
        tensor_index = int(tensor_index)
        cached = self.reference_cache.get(tensor_index)
        if cached is None:
            if tensor_index in self.constant_tensor_indices:
                constant = self.constant_reader.get_tensor(tensor_index)
                if constant is not None:
                    cached = np.array(constant, copy=True)
                    self.reference_cache[tensor_index] = cached
                    return np.array(cached, copy=True)

            try:
                cached = np.array(self.interpreter.get_tensor(tensor_index), copy=True)
            except Exception:
                constant = self.constant_reader.get_tensor(tensor_index)
                if constant is None:
                    raise
                cached = np.array(constant, copy=True)
            self.reference_cache[tensor_index] = cached
        return np.array(cached, copy=True)

    def _invoke_reference(self, input_tensor: np.ndarray) -> None:
        self.reference_cache.clear()
        self.interpreter.set_tensor(int(self.input_detail["index"]), np.asarray(input_tensor, dtype=np.int8))
        self.interpreter.invoke()

    def _reference_outputs_for_op(self, op: dict) -> list[np.ndarray]:
        return [self._get_reference_tensor(int(idx)) for idx in op.get("outputs", [])]

    def _get_tensor_value(
        self,
        tensor_cache: dict[int, np.ndarray],
        tensor_index: int,
    ) -> np.ndarray:
        tensor_index = int(tensor_index)
        if tensor_index in tensor_cache:
            return tensor_cache[tensor_index]
        if tensor_index in self.constant_tensor_indices:
            value = self._get_reference_tensor(tensor_index)
            tensor_cache[tensor_index] = value
            return value
        raise KeyError(f"Tensor {tensor_index} not available in cache")

    def _compute_digital_op(
        self,
        op: dict,
        tensor_cache: dict[int, np.ndarray],
    ) -> list[np.ndarray]:
        op_name = str(op["op_name"])
        input_ids = [int(idx) for idx in op.get("inputs", [])]
        output_ids = [int(idx) for idx in op.get("outputs", [])]
        inputs = [self._get_tensor_value(tensor_cache, idx) for idx in input_ids]
        output_meta = [self.tensor_meta[idx] for idx in output_ids]

        if op_name == "PAD":
            pad_value = output_meta[0].zero_point if _is_quantized_int(output_meta[0]) else 0
            return [np.pad(inputs[0], tuple(tuple(int(v) for v in row) for row in np.asarray(inputs[1])), mode="constant", constant_values=pad_value).astype(output_meta[0].dtype)]

        if op_name == "LOGISTIC":
            real = _sigmoid(_dequantize(inputs[0], self.tensor_meta[input_ids[0]]))
            return [_requantize(real, output_meta[0])]

        if op_name in {"MUL", "ADD", "SUB"}:
            a = _dequantize(inputs[0], self.tensor_meta[input_ids[0]])
            b = _dequantize(inputs[1], self.tensor_meta[input_ids[1]])
            if op_name == "MUL":
                real = a * b
            elif op_name == "ADD":
                real = a + b
            else:
                real = a - b
            return [_requantize(real, output_meta[0])]

        if op_name == "CONCATENATION":
            axis = _infer_concat_axis(inputs, output_meta[0].shape)
            return [np.concatenate(inputs, axis=axis).astype(output_meta[0].dtype)]

        if op_name == "SLICE":
            return [_slice_tensor(inputs[0], inputs[1], inputs[2]).astype(output_meta[0].dtype)]

        if op_name == "RESHAPE":
            target_shape = tuple(int(v) for v in np.asarray(inputs[1]).reshape(-1))
            return [np.reshape(inputs[0], target_shape).astype(output_meta[0].dtype)]

        if op_name == "TRANSPOSE":
            perm = tuple(int(v) for v in np.asarray(inputs[1]).reshape(-1))
            return [np.transpose(inputs[0], axes=perm).astype(output_meta[0].dtype)]

        if op_name == "MAX_POOL_2D":
            return [_max_pool2d_same_5x5(inputs[0], output_meta[0].shape).astype(output_meta[0].dtype)]

        if op_name == "RESIZE_NEAREST_NEIGHBOR":
            return [_resize_nearest_neighbor(inputs[0], inputs[1], output_meta[0].shape).astype(output_meta[0].dtype)]

        if op_name == "BATCH_MATMUL":
            return [np.matmul(inputs[0], inputs[1]).astype(output_meta[0].dtype)]

        if op_name == "QUANTIZE":
            src_meta = self.tensor_meta[input_ids[0]]
            real = _dequantize(inputs[0], src_meta) if _is_quantized_int(src_meta) else np.asarray(inputs[0], dtype=np.float32)
            return [_requantize(real, output_meta[0])]

        if op_name == "DEQUANTIZE":
            return [_dequantize(inputs[0], self.tensor_meta[input_ids[0]])]

        if op_name == "CAST":
            return [np.asarray(inputs[0], dtype=output_meta[0].dtype)]

        if op_name == "TILE":
            reps = tuple(int(v) for v in np.asarray(inputs[1]).reshape(-1))
            return [np.tile(inputs[0], reps).astype(output_meta[0].dtype)]

        if op_name == "LESS":
            return [np.less(inputs[0], inputs[1])]

        if op_name == "FLOOR_MOD":
            return [np.mod(inputs[0], inputs[1]).astype(output_meta[0].dtype)]

        if op_name == "GATHER":
            return [np.take(inputs[0], inputs[1], axis=0)]

        if op_name == "GATHER_ND":
            indices = np.asarray(inputs[1], dtype=np.int64)
            gathered = np.asarray([inputs[0][tuple(ind)] for ind in indices.reshape(-1, indices.shape[-1])])
            tail_shape = inputs[0].shape[indices.shape[-1] :]
            out_shape = tuple(indices.shape[:-1]) + tail_shape
            return [gathered.reshape(out_shape).astype(output_meta[0].dtype)]

        if op_name == "REDUCE_MAX":
            axis = tuple(int(v) for v in np.asarray(inputs[1]).reshape(-1))
            keepdims = len(output_meta[0].shape) == inputs[0].ndim
            return [np.max(inputs[0], axis=axis, keepdims=keepdims).astype(output_meta[0].dtype)]

        if op_name == "SOFTMAX":
            real = _dequantize(inputs[0], self.tensor_meta[input_ids[0]])
            shifted = real - np.max(real, axis=-1, keepdims=True)
            probs = np.exp(shifted)
            probs /= np.sum(probs, axis=-1, keepdims=True)
            return [_requantize(probs, output_meta[0])]

        raise NotImplementedError(op_name)

    def _compute_conv_op(
        self,
        op: dict,
        tensor_cache: dict[int, np.ndarray],
        *,
        layer_name: str,
        use_pl: bool = True,
    ) -> np.ndarray:
        if self.crossbar_runner is None:
            raise RuntimeError("Crossbar runner requested without a mapping file")

        input_ids = [int(idx) for idx in op.get("inputs", [])]
        input_q = self._get_tensor_value(tensor_cache, input_ids[0])
        weight_q = self._get_reference_tensor(input_ids[1])
        bias_q = self._get_reference_tensor(input_ids[2]) if len(input_ids) > 2 and input_ids[2] >= 0 else None

        input_meta = self.tensor_meta[input_ids[0]]
        weight_meta = self.tensor_meta[input_ids[1]]
        output_id = int(op.get("outputs", [])[0])
        output_meta = self.tensor_meta[output_id]
        return self.crossbar_runner.run_quantized_conv(
            layer_name=layer_name,
            op_name=str(op["op_name"]),
            input_q=input_q,
            input_zero_point=input_meta.zero_point,
            input_scale=input_meta.scale,
            weight_q=weight_q,
            weight_scales=weight_meta.scales,
            bias_q=bias_q,
            output_scale=output_meta.scale,
            output_zero_point=output_meta.zero_point,
            use_pl=use_pl,
        )

    def run(
        self,
        image_path: str | Path,
        *,
        crossbar_layers: set[str] | None = None,
        save_dir: str | Path = "hybrid_inference_outputs",
        capture_tensor_ids: set[int] | None = None,
        stop_after_tensor_id: int | None = None,
    ) -> tuple[np.ndarray, np.ndarray, list[OpRecord]]:
        crossbar_layers = set(crossbar_layers or set())
        capture_tensor_ids = {int(idx) for idx in (capture_tensor_ids or set())}
        target_tensor_id = int(stop_after_tensor_id) if stop_after_tensor_id is not None else None
        if target_tensor_id is not None and target_tensor_id not in self.tensor_meta:
            raise ValueError(f"Unknown stop-after tensor id: {target_tensor_id}")
        save_path = Path(save_dir)
        save_path.mkdir(exist_ok=True)

        input_i8, preprocess_meta = self.preprocess_image(image_path)
        reference_error = ""
        skip_reference_invoke = os.environ.get("YOLO_SKIP_REFERENCE_INVOKE", "").strip().lower() in {"1", "true", "yes"}
        if skip_reference_invoke:
            reference_error = "skipped by YOLO_SKIP_REFERENCE_INVOKE"
            self.reference_cache.clear()
            _safe_print(f"[REFERENCE] Full TFLite invoke skipped; using PS digital ops for fallbacks.", flush=True)
        else:
            try:
                self._invoke_reference(input_i8)
            except Exception as exc:
                reference_error = str(exc)
                self.reference_cache.clear()
                try:
                    self.interpreter.allocate_tensors()
                except Exception:
                    pass
                _safe_print(f"[REFERENCE] Full TFLite invoke unavailable; using PS digital ops for fallbacks: {reference_error}", flush=True)

        tensor_cache: dict[int, np.ndarray] = {int(self.input_detail["index"]): np.array(input_i8, copy=True)}
        records: list[OpRecord] = []

        for op in self.ops:
            op_index = int(op["index"])
            op_name = str(op["op_name"])
            layer_name = self.conv_layer_names.get(op_index, "")
            output_ids = [int(idx) for idx in op.get("outputs", [])]

            backend = "reference"
            note = ""
            op_start = time.perf_counter()
            try:
                if op_name in {"CONV_2D", "DEPTHWISE_CONV_2D"} and layer_name in crossbar_layers:
                    try:
                        outputs = [self._compute_conv_op(op, tensor_cache, layer_name=layer_name, use_pl=True)]
                    except Exception as exc:
                        raise RuntimeError(
                            f"PL-DMA crossbar layer {layer_name} failed; "
                            f"reference/PS crossbar fallback is disabled: {exc}"
                        ) from exc
                    backend = self.crossbar_runner.record_backend(use_pl=True) if self.crossbar_runner is not None else "pl_dma"
                elif op_name in {"CONV_2D", "DEPTHWISE_CONV_2D"} and self.crossbar_runner is not None and layer_name:
                    outputs = [self._compute_conv_op(op, tensor_cache, layer_name=layer_name, use_pl=False)]
                    backend = self.crossbar_runner.record_backend(use_pl=False)
                else:
                    outputs = self._compute_digital_op(op, tensor_cache)
                    backend = "digital"
            except Exception as exc:
                if op_name in {"CONV_2D", "DEPTHWISE_CONV_2D"} and layer_name in crossbar_layers:
                    raise
                try:
                    outputs = self._reference_outputs_for_op(op)
                    backend = "reference"
                    note = str(exc)
                except Exception as ref_exc:
                    ref_note = f"; reference fallback unavailable after invoke error: {reference_error}" if reference_error else ""
                    raise RuntimeError(
                        f"Op {op_index} {op_name} failed in PS runtime: {exc}; "
                        f"reference fallback also failed: {ref_exc}{ref_note}"
                    ) from exc

            duration_ms = (time.perf_counter() - op_start) * 1000.0
            for out_id, out_value in zip(output_ids, outputs):
                tensor_cache[out_id] = np.array(out_value, copy=True)
                if out_id in capture_tensor_ids:
                    np.save(save_path / f"tensor_{out_id}.npy", tensor_cache[out_id])

            records.append(
                OpRecord(
                    index=op_index,
                    op_name=op_name,
                    backend=backend,
                    layer_name=layer_name,
                    input_indices=[int(idx) for idx in op.get("inputs", [])],
                    output_indices=output_ids,
                    note=note,
                    duration_ms=duration_ms,
                )
            )
            if target_tensor_id is not None and target_tensor_id in output_ids:
                break

        output_index = target_tensor_id if target_tensor_id is not None else int(self.output_detail["index"])
        if output_index not in tensor_cache:
            raise RuntimeError(
                f"Execution stopped without producing requested tensor {output_index}."
            )
        output_q = np.array(tensor_cache[output_index], copy=True)
        output_real = _dequantize(output_q, self.tensor_meta[output_index])

        self._save_run_artifacts(
            save_path=save_path,
            preprocess_meta=preprocess_meta,
            output_q=output_q,
            output_real=output_real,
            records=records,
        )
        return output_q, output_real, records

    def _save_run_artifacts(
        self,
        *,
        save_path: Path,
        preprocess_meta: dict,
        output_q: np.ndarray,
        output_real: np.ndarray,
        records: list[OpRecord],
    ) -> None:
        (save_path / "preprocess_meta.json").write_text(json.dumps(preprocess_meta, indent=2), encoding="utf-8")
        np.save(save_path / "final_output_int8.npy", output_q)
        np.save(save_path / "final_output_float.npy", output_real)

        flat_q = output_q.reshape(-1, output_q.shape[-1])
        flat_real = output_real.reshape(-1, output_real.shape[-1])
        with open(save_path / "final_output_int8.txt", "w", encoding="utf-8") as f:
            f.write("Final output int8 values\n")
            f.write(f"Shape: {tuple(int(v) for v in output_q.shape)}\n")
            for row in flat_q:
                f.write(" ".join(str(int(v)) for v in row) + "\n")
        with open(save_path / "final_output_float.txt", "w", encoding="utf-8") as f:
            f.write("Final output dequantized values\n")
            f.write(f"Shape: {tuple(int(v) for v in output_real.shape)}\n")
            for row in flat_real:
                f.write(" ".join(f"{float(v):.6f}" for v in row) + "\n")

        with open(save_path / "execution_trace.csv", "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["op_index", "op_name", "backend", "layer_name", "duration_ms", "inputs", "outputs", "note"])
            for record in records:
                writer.writerow(
                    [
                        record.index,
                        record.op_name,
                        record.backend,
                        record.layer_name,
                        f"{record.duration_ms:.6f}",
                        " ".join(str(v) for v in record.input_indices),
                        " ".join(str(v) for v in record.output_indices),
                        record.note,
                    ]
                )

        profile = {
            "total_profiled_ms": float(sum(record.duration_ms for record in records)),
            "by_backend": {},
            "by_op_name": {},
            "top_ops": [],
        }
        for record in records:
            for bucket_name, key in (
                ("by_backend", record.backend),
                ("by_op_name", record.op_name),
            ):
                bucket = profile[bucket_name].setdefault(key, {"count": 0, "duration_ms": 0.0})
                bucket["count"] += 1
                bucket["duration_ms"] += float(record.duration_ms)
        profile["top_ops"] = [
            {
                "op_index": record.index,
                "op_name": record.op_name,
                "backend": record.backend,
                "layer_name": record.layer_name,
                "duration_ms": float(record.duration_ms),
            }
            for record in sorted(records, key=lambda item: item.duration_ms, reverse=True)[:25]
        ]
        for group_name in ("by_backend", "by_op_name"):
            for value in profile[group_name].values():
                value["duration_ms"] = float(value["duration_ms"])
        (save_path / "profile_summary.json").write_text(json.dumps(profile, indent=2), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a full TFLite inference executor with optional crossbar Conv replay.")
    parser.add_argument("--image", default="bus.jpg", help="Input image path.")
    parser.add_argument("--model", default="yolo26n_full_integer_quant.tflite", help="Quantized TFLite model path.")
    parser.add_argument("--mapping", default="Yolo26n_auto_full_mapping.m", help="MAESTRO mapping file for Conv layers.")
    parser.add_argument("--save-dir", default="hybrid_inference_outputs", help="Directory for outputs and execution trace.")
    parser.add_argument(
        "--crossbar-layers",
        default="",
        help="Comma-separated Conv/Depthwise layer names to execute on the crossbar path, e.g. Conv2D,Conv2D_1.",
    )
    args = parser.parse_args()

    crossbar_layers = {name.strip() for name in args.crossbar_layers.split(",") if name.strip()}
    executor = TFLiteHybridExecutor(args.model, mapping_path=args.mapping)
    output_q, output_real, records = executor.run(args.image, crossbar_layers=crossbar_layers, save_dir=args.save_dir)

    backend_counts: dict[str, int] = {}
    for record in records:
        backend_counts[record.backend] = backend_counts.get(record.backend, 0) + 1

    print(f"Saved outputs to {Path(args.save_dir).resolve()}")
    print(f"Final int8 output shape: {output_q.shape}")
    print(f"Final float output shape: {output_real.shape}")
    print(f"Backends used: {backend_counts}")
    if output_real.size:
        print("First 5 detections:")
        for row in output_real.reshape(-1, output_real.shape[-1])[:5]:
            print(" ".join(f"{float(v):.6f}" for v in row))


if __name__ == "__main__":
    main()
