from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import numpy as np

from .tflite_frontend import _infer_conv_geometry


@dataclass(frozen=True)
class TensorMeta:
    index: int
    name: str
    shape: tuple[int, ...]
    dtype: np.dtype
    scale: float
    zero_point: int
    scales: np.ndarray
    zero_points: np.ndarray
    quantized_dimension: int

    @property
    def nbytes(self) -> int:
        return int(np.prod(self.shape, dtype=np.int64)) * self.dtype.itemsize

    @classmethod
    def from_detail(cls, detail: dict[str, Any]) -> "TensorMeta":
        qparams = detail.get("quantization_parameters", {})
        scale, zero_point = detail.get("quantization", (0.0, 0))
        return cls(
            index=int(detail["index"]),
            name=str(detail["name"]),
            shape=tuple(int(value) for value in detail["shape"]),
            dtype=np.dtype(detail["dtype"]),
            scale=float(scale),
            zero_point=int(zero_point),
            scales=np.asarray(qparams.get("scales", []), dtype=np.float32),
            zero_points=np.asarray(qparams.get("zero_points", []), dtype=np.int64),
            quantized_dimension=int(qparams.get("quantized_dimension", 0)),
        )


def is_quantized(meta: TensorMeta) -> bool:
    return np.issubdtype(meta.dtype, np.integer) and meta.scale > 0.0


def dequantize(value: np.ndarray, meta: TensorMeta) -> np.ndarray:
    if not is_quantized(meta):
        return np.asarray(value, dtype=np.float32)
    return (
        np.asarray(value, dtype=np.float32) - np.float32(meta.zero_point)
    ) * np.float32(meta.scale)


def requantize(value: np.ndarray, meta: TensorMeta) -> np.ndarray:
    if not is_quantized(meta):
        return np.asarray(value, dtype=meta.dtype)
    quantized = np.rint(
        np.asarray(value, dtype=np.float32) / np.float32(meta.scale)
        + np.float32(meta.zero_point)
    )
    limits = np.iinfo(meta.dtype)
    return np.clip(quantized, limits.min, limits.max).astype(meta.dtype)


def sigmoid(value: np.ndarray) -> np.ndarray:
    source = np.asarray(value, dtype=np.float32)
    result = np.empty_like(source)
    positive = source >= 0
    result[positive] = 1.0 / (1.0 + np.exp(-source[positive]))
    exp_value = np.exp(source[~positive])
    result[~positive] = exp_value / (1.0 + exp_value)
    return result


def _round_half_away_from_zero(value: np.ndarray | float) -> np.ndarray:
    source = np.asarray(value)
    return np.where(
        source >= 0,
        np.floor(source + 0.5),
        np.ceil(source - 0.5),
    )


def _infer_concat_axis(
    inputs: list[np.ndarray],
    output_shape: tuple[int, ...],
) -> int:
    candidates: list[int] = []
    for axis in range(len(output_shape)):
        if all(
            value.ndim == len(output_shape)
            and all(
                dim == axis or value.shape[dim] == output_shape[dim]
                for dim in range(len(output_shape))
            )
            for value in inputs
        ) and sum(value.shape[axis] for value in inputs) == output_shape[axis]:
            candidates.append(axis)
    if len(candidates) != 1:
        raise ValueError(
            f"Could not infer one CONCATENATION axis for {output_shape}; "
            f"candidates={candidates}."
        )
    return candidates[0]


def _slice_tensor(
    data: np.ndarray,
    begin: np.ndarray,
    size: np.ndarray,
) -> np.ndarray:
    starts = np.asarray(begin, dtype=np.int64).reshape(-1)
    lengths = np.asarray(size, dtype=np.int64).reshape(-1)
    if starts.size != data.ndim or lengths.size != data.ndim:
        raise ValueError("SLICE begin/size rank does not match the data tensor.")
    slices: list[slice] = []
    for dimension, (start, length) in enumerate(zip(starts, lengths)):
        resolved_start = int(start)
        if resolved_start < 0:
            resolved_start += data.shape[dimension]
        stop = (
            data.shape[dimension]
            if int(length) < 0
            else resolved_start + int(length)
        )
        slices.append(slice(resolved_start, stop))
    return data[tuple(slices)]


def _max_pool2d_same_5x5(
    data: np.ndarray,
    output_shape: tuple[int, ...],
) -> np.ndarray:
    if data.ndim != 4 or len(output_shape) != 4:
        raise ValueError("MAX_POOL_2D expects NHWC rank-four tensors.")
    out_h, out_w = output_shape[1:3]
    stride_h = max(1, int(round(data.shape[1] / float(out_h))))
    stride_w = max(1, int(round(data.shape[2] / float(out_w))))
    kernel_h = kernel_w = 5
    pad_h = max((out_h - 1) * stride_h + kernel_h - data.shape[1], 0)
    pad_w = max((out_w - 1) * stride_w + kernel_w - data.shape[2], 0)
    if np.issubdtype(data.dtype, np.floating):
        pad_value: float | int = -np.inf
    else:
        pad_value = int(np.iinfo(data.dtype).min)
    padded = np.pad(
        data,
        (
            (0, 0),
            (pad_h // 2, pad_h - pad_h // 2),
            (pad_w // 2, pad_w - pad_w // 2),
            (0, 0),
        ),
        constant_values=pad_value,
    )
    windows = np.lib.stride_tricks.sliding_window_view(
        padded,
        (kernel_h, kernel_w),
        axis=(1, 2),
    )
    selected = windows[:, ::stride_h, ::stride_w, :, :, :]
    return np.max(
        selected[:, :out_h, :out_w, :, :, :],
        axis=(-2, -1),
    ).astype(data.dtype, copy=False)


def _resize_nearest(
    data: np.ndarray,
    size: np.ndarray,
    output_shape: tuple[int, ...],
) -> np.ndarray:
    target = np.asarray(size, dtype=np.int64).reshape(-1)
    if data.ndim != 4 or target.size != 2:
        raise ValueError("RESIZE_NEAREST_NEIGHBOR expects NHWC data and [H,W].")
    out_h, out_w = output_shape[1:3]
    in_h, in_w = data.shape[1:3]
    y_indices = np.floor(
        np.arange(out_h, dtype=np.float32) * in_h / float(out_h)
    ).astype(np.int64)
    x_indices = np.floor(
        np.arange(out_w, dtype=np.float32) * in_w / float(out_w)
    ).astype(np.int64)
    return data[:, y_indices, :, :][:, :, x_indices, :]


def execute_digital_op(
    op_name: str,
    inputs: list[np.ndarray],
    input_meta: list[TensorMeta],
    output_meta: list[TensorMeta],
) -> list[np.ndarray]:
    """Execute every non-convolution operator present in yolo26n_int8.tflite."""

    if op_name == "PAD":
        fill = output_meta[0].zero_point if is_quantized(output_meta[0]) else 0
        padding = tuple(
            tuple(int(item) for item in row)
            for row in np.asarray(inputs[1])
        )
        return [
            np.pad(inputs[0], padding, constant_values=fill).astype(
                output_meta[0].dtype
            )
        ]

    if op_name == "LOGISTIC":
        return [
            requantize(sigmoid(dequantize(inputs[0], input_meta[0])), output_meta[0])
        ]

    if op_name in {"MUL", "ADD", "SUB"}:
        lhs = dequantize(inputs[0], input_meta[0])
        rhs = dequantize(inputs[1], input_meta[1])
        operation: dict[str, Callable[[np.ndarray, np.ndarray], np.ndarray]] = {
            "MUL": np.multiply,
            "ADD": np.add,
            "SUB": np.subtract,
        }
        return [requantize(operation[op_name](lhs, rhs), output_meta[0])]

    if op_name == "CONCATENATION":
        axis = _infer_concat_axis(inputs, output_meta[0].shape)
        compatible = all(
            meta.dtype == output_meta[0].dtype
            and meta.scale == output_meta[0].scale
            and meta.zero_point == output_meta[0].zero_point
            for meta in input_meta
        )
        if compatible:
            result = np.concatenate(inputs, axis=axis)
        else:
            result = requantize(
                np.concatenate(
                    [dequantize(value, meta) for value, meta in zip(inputs, input_meta)],
                    axis=axis,
                ),
                output_meta[0],
            )
        return [result.astype(output_meta[0].dtype, copy=False)]

    if op_name == "SLICE":
        return [
            _slice_tensor(inputs[0], inputs[1], inputs[2]).astype(
                output_meta[0].dtype
            )
        ]

    if op_name == "RESHAPE":
        shape = tuple(int(value) for value in np.asarray(inputs[1]).reshape(-1))
        return [np.reshape(inputs[0], shape).astype(output_meta[0].dtype)]

    if op_name == "TRANSPOSE":
        axes = tuple(int(value) for value in np.asarray(inputs[1]).reshape(-1))
        return [np.transpose(inputs[0], axes).astype(output_meta[0].dtype)]

    if op_name == "MAX_POOL_2D":
        return [
            _max_pool2d_same_5x5(inputs[0], output_meta[0].shape).astype(
                output_meta[0].dtype
            )
        ]

    if op_name == "RESIZE_NEAREST_NEIGHBOR":
        return [
            _resize_nearest(inputs[0], inputs[1], output_meta[0].shape).astype(
                output_meta[0].dtype
            )
        ]

    if op_name == "BATCH_MATMUL":
        return [
            np.matmul(inputs[0], inputs[1]).astype(output_meta[0].dtype)
        ]

    if op_name == "SOFTMAX":
        real = dequantize(inputs[0], input_meta[0])
        shifted = real - np.max(real, axis=-1, keepdims=True)
        exponential = np.exp(shifted)
        probabilities = exponential / np.sum(
            exponential,
            axis=-1,
            keepdims=True,
        )
        return [requantize(probabilities, output_meta[0])]

    raise NotImplementedError(f"CPU implementation is missing for {op_name}.")


def execute_depthwise_conv2d(
    input_value: np.ndarray,
    weights: np.ndarray,
    weight_meta: TensorMeta,
    bias: np.ndarray | None,
    output_meta: TensorMeta,
) -> np.ndarray:
    """Vectorized float CPU fallback for TFLite OHWI depthwise convolution."""

    activation = np.asarray(input_value, dtype=np.float32)
    quantized_weights = np.asarray(weights)
    if activation.ndim != 4 or quantized_weights.ndim != 4:
        raise ValueError("Depthwise Conv2D expects rank-four NHWC/OHWI tensors.")
    _, kernel_h, kernel_w, output_channels = quantized_weights.shape
    input_channels = activation.shape[-1]
    if output_channels % input_channels:
        raise ValueError("Depthwise output channels are not a multiple of inputs.")
    multiplier = output_channels // input_channels

    scales = np.asarray(weight_meta.scales, dtype=np.float32).reshape(-1)
    if scales.size == 0:
        scales = np.asarray([weight_meta.scale], dtype=np.float32)
    if scales.size == 1:
        scales = np.repeat(scales, output_channels)
    zero_points = np.asarray(weight_meta.zero_points, dtype=np.float32).reshape(-1)
    if zero_points.size == 0:
        zero_points = np.zeros(output_channels, dtype=np.float32)
    if zero_points.size == 1:
        zero_points = np.repeat(zero_points, output_channels)
    weights_i32 = (
        quantized_weights.astype(np.int32).reshape(
            kernel_h,
            kernel_w,
            input_channels,
            multiplier,
        )
        - zero_points.astype(np.int32).reshape(
            1,
            1,
            input_channels,
            multiplier,
        )
    )

    # LiteRT's float-activation/int8-weight hybrid kernel dynamically maps the
    # full activation range to [-128, 127].  Reproducing that asymmetric bridge
    # makes the CPU depthwise fallback bit-for-bit equivalent for this model.
    activation_min = float(np.min(activation))
    activation_max = float(np.max(activation))
    activation_scale = max(
        (activation_max - activation_min) / 255.0,
        1e-12,
    )
    activation_zero_point = int(
        np.clip(
            _round_half_away_from_zero(
                -128.0 - activation_min / activation_scale
            ),
            -128,
            127,
        )
    )
    activation_i8 = np.clip(
        _round_half_away_from_zero(
            activation / activation_scale + activation_zero_point
        ),
        -128,
        127,
    ).astype(np.int8)

    expected_output = output_meta.shape
    stride, padding = _infer_conv_geometry(
        tuple(int(value) for value in activation.shape),
        tuple(int(value) for value in quantized_weights.shape),
        expected_output,
    )
    stride_h, stride_w = stride
    out_h, out_w = expected_output[1:3]
    if padding == "same":
        pad_h = max(
            (out_h - 1) * stride_h + kernel_h - activation.shape[1],
            0,
        )
        pad_w = max(
            (out_w - 1) * stride_w + kernel_w - activation.shape[2],
            0,
        )
        activation_i8 = np.pad(
            activation_i8,
            (
                (0, 0),
                (pad_h // 2, pad_h - pad_h // 2),
                (pad_w // 2, pad_w - pad_w // 2),
                (0, 0),
            ),
            constant_values=activation_zero_point,
        )

    windows = np.lib.stride_tricks.sliding_window_view(
        activation_i8,
        (kernel_h, kernel_w),
        axis=(1, 2),
    )
    selected = windows[
        :,
        ::stride_h,
        ::stride_w,
        :,
        :,
        :,
    ][:, :out_h, :out_w, :, :, :]
    patches = selected.transpose(0, 1, 2, 4, 5, 3)
    accumulators = np.einsum(
        "bhwklc,klcm->bhwcm",
        patches.astype(np.int32) - activation_zero_point,
        weights_i32,
        optimize=True,
        dtype=np.int32,
    ).reshape(expected_output)
    result = (
        accumulators.astype(np.float32)
        * np.float32(activation_scale)
        * scales.reshape(1, 1, 1, -1)
    )
    if bias is not None:
        result += np.asarray(bias, dtype=np.float32).reshape(1, 1, 1, -1)
    return requantize(result, output_meta)


def execute_conv2d_cpu(
    input_value: np.ndarray,
    weights: np.ndarray,
    weight_meta: TensorMeta,
    bias: np.ndarray | None,
    output_meta: TensorMeta,
) -> np.ndarray:
    """Vectorized float CPU implementation of TFLite OHWI Conv2D."""

    activation = np.asarray(input_value, dtype=np.float32)
    quantized_weights = np.asarray(weights)
    if activation.ndim != 4 or quantized_weights.ndim != 4:
        raise ValueError("Conv2D expects rank-four NHWC/OHWI tensors.")
    output_channels, kernel_h, kernel_w, input_channels = (
        quantized_weights.shape
    )
    if input_channels != activation.shape[-1]:
        raise ValueError(
            f"Conv2D channel mismatch: {activation.shape} vs "
            f"{quantized_weights.shape}."
        )

    scales = np.asarray(weight_meta.scales, dtype=np.float32).reshape(-1)
    if scales.size == 0:
        scales = np.asarray([weight_meta.scale], dtype=np.float32)
    if scales.size == 1:
        scales = np.repeat(scales, output_channels)
    zero_points = np.asarray(weight_meta.zero_points, dtype=np.float32).reshape(-1)
    if zero_points.size == 0:
        zero_points = np.zeros(output_channels, dtype=np.float32)
    if zero_points.size == 1:
        zero_points = np.repeat(zero_points, output_channels)
    weights_i32 = (
        quantized_weights.astype(np.int32)
        - zero_points.astype(np.int32).reshape(-1, 1, 1, 1)
    )
    activation_min = float(np.min(activation))
    activation_max = float(np.max(activation))
    activation_scale = max(
        (activation_max - activation_min) / 255.0,
        1e-12,
    )
    activation_zero_point = int(
        np.clip(
            _round_half_away_from_zero(
                -128.0 - activation_min / activation_scale
            ),
            -128,
            127,
        )
    )
    activation_i8 = np.clip(
        _round_half_away_from_zero(
            activation / activation_scale + activation_zero_point
        ),
        -128,
        127,
    ).astype(np.int8)

    expected_output = output_meta.shape
    stride, padding = _infer_conv_geometry(
        tuple(int(value) for value in activation.shape),
        tuple(int(value) for value in quantized_weights.shape),
        expected_output,
    )
    stride_h, stride_w = stride
    out_h, out_w = expected_output[1:3]
    if padding == "same":
        pad_h = max(
            (out_h - 1) * stride_h + kernel_h - activation.shape[1],
            0,
        )
        pad_w = max(
            (out_w - 1) * stride_w + kernel_w - activation.shape[2],
            0,
        )
        activation_i8 = np.pad(
            activation_i8,
            (
                (0, 0),
                (pad_h // 2, pad_h - pad_h // 2),
                (pad_w // 2, pad_w - pad_w // 2),
                (0, 0),
            ),
            constant_values=activation_zero_point,
        )

    windows = np.lib.stride_tricks.sliding_window_view(
        activation_i8,
        (kernel_h, kernel_w),
        axis=(1, 2),
    )
    selected = windows[
        :,
        ::stride_h,
        ::stride_w,
        :,
        :,
        :,
    ][:, :out_h, :out_w, :, :, :]
    patches = selected.transpose(0, 1, 2, 4, 5, 3)
    accumulators = np.einsum(
        "bhwklc,oklc->bhwo",
        patches.astype(np.int32) - activation_zero_point,
        weights_i32,
        optimize=True,
        dtype=np.int32,
    )
    result = (
        accumulators.astype(np.float32)
        * np.float32(activation_scale)
        * scales.reshape(1, 1, 1, -1)
    )
    if bias is not None:
        result += np.asarray(bias, dtype=np.float32).reshape(1, 1, 1, -1)
    return requantize(result, output_meta)
