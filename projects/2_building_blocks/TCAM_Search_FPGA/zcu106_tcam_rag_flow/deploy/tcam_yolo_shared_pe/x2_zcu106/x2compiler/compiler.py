from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray

from .ir import X2Program, X2Target, X2WeightTile


def _strict_i8(array: NDArray[np.integer], name: str) -> np.ndarray:
    value = np.asarray(array)
    if value.ndim < 1:
        raise ValueError(f"{name} must be an array.")
    if not np.issubdtype(value.dtype, np.integer):
        raise TypeError(f"{name} must contain integer values, got {value.dtype}.")
    if value.size and (int(value.min()) < -128 or int(value.max()) > 127):
        raise ValueError(f"{name} contains values outside signed INT8.")
    return value.astype(np.int8, copy=False)


class X2Compiler:
    """Lower logical INT8 matrices and Conv2D weights to the X2 tile IR."""

    def __init__(self, target: X2Target | None = None) -> None:
        self.target = target or X2Target()
        self.target.validate()

    def compile_matmul(
        self,
        weights: NDArray[np.integer],
        *,
        input_rows: int,
        name: str = "x2_matmul",
        bias_i32: NDArray[np.integer] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> X2Program:
        matrix = _strict_i8(weights, "weights")
        if matrix.ndim != 2:
            raise ValueError(f"Matmul weights must be [K,N], got {matrix.shape}.")
        m_dim = int(input_rows)
        k_dim, n_dim = (int(value) for value in matrix.shape)
        if m_dim < 1:
            raise ValueError("input_rows must be positive.")

        constants: dict[str, np.ndarray] = {}
        tiles: list[X2WeightTile] = []
        tile_id = 0
        for n_start in range(0, n_dim, self.target.max_cols):
            cols = min(self.target.max_cols, n_dim - n_start)
            for k_start in range(0, k_dim, self.target.max_rows):
                rows = min(self.target.max_rows, k_dim - k_start)
                weight_key = f"weight_k{k_start}_n{n_start}"
                constants[weight_key] = np.ascontiguousarray(
                    matrix[k_start : k_start + rows, n_start : n_start + cols],
                    dtype=np.int8,
                )
                tiles.append(
                    X2WeightTile(
                        tile_id=tile_id,
                        k_start=k_start,
                        n_start=n_start,
                        rows=rows,
                        cols=cols,
                        weight_key=weight_key,
                        accumulate=k_start != 0,
                    )
                )
                tile_id += 1

        program_metadata = dict(metadata or {})
        if bias_i32 is not None:
            bias = np.asarray(bias_i32)
            if bias.shape != (n_dim,):
                raise ValueError(f"INT32 bias must have shape {(n_dim,)}, got {bias.shape}.")
            constants["bias_i32"] = bias.astype(np.int32, copy=False)
            program_metadata["bias_i32_key"] = "bias_i32"

        program = X2Program(
            name=name,
            kind="matmul",
            m_dim=m_dim,
            k_dim=k_dim,
            n_dim=n_dim,
            tiles=tiles,
            constants=constants,
            target=self.target,
            metadata=program_metadata,
        )
        program.validate()
        return program

    def compile_conv2d(
        self,
        weights_ohwi: NDArray[np.integer],
        *,
        input_shape: tuple[int, int, int, int],
        output_shape: tuple[int, int, int, int],
        stride: tuple[int, int],
        padding: str,
        name: str = "x2_conv2d",
        bias_i32: NDArray[np.integer] | None = None,
        bias_float: NDArray[np.floating] | None = None,
        weight_scales: NDArray[np.floating] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> X2Program:
        weights = _strict_i8(weights_ohwi, "weights_ohwi")
        if weights.ndim != 4:
            raise ValueError(f"Conv2D weights must be [O,H,W,I], got {weights.shape}.")
        if len(input_shape) != 4 or len(output_shape) != 4:
            raise ValueError("Conv2D input/output shapes must be rank four NHWC.")
        batch, input_h, input_w, input_channels = (int(v) for v in input_shape)
        out_batch, output_h, output_w, output_channels = (int(v) for v in output_shape)
        out_channels, kernel_h, kernel_w, weight_channels = weights.shape
        if batch != out_batch:
            raise ValueError("Conv2D input/output batch dimensions differ.")
        if input_channels != weight_channels:
            raise ValueError(
                f"Input channels {input_channels} differ from weight channels {weight_channels}."
            )
        if output_channels != out_channels:
            raise ValueError(
                f"Output channels {output_channels} differ from weights {out_channels}."
            )
        stride_y, stride_x = (int(v) for v in stride)
        if stride_y < 1 or stride_x < 1:
            raise ValueError(f"Invalid Conv2D stride {stride}.")
        padding = str(padding).lower()
        if padding not in {"same", "valid"}:
            raise ValueError(f"Unsupported Conv2D padding: {padding}")

        weight_matrix = np.transpose(weights, (1, 2, 3, 0)).reshape(
            kernel_h * kernel_w * input_channels,
            output_channels,
        )
        conv_metadata = dict(metadata or {})
        conv_metadata["conv2d"] = {
            "input_shape_nhwc": list(input_shape),
            "output_shape_nhwc": list(output_shape),
            "kernel": [int(kernel_h), int(kernel_w)],
            "stride": [stride_y, stride_x],
            "padding": padding,
        }
        if bias_float is not None:
            bias = np.asarray(bias_float, dtype=np.float32)
            if bias.shape != (output_channels,):
                raise ValueError(
                    f"Float bias must have shape {(output_channels,)}, got {bias.shape}."
                )
            conv_metadata["bias_float_key"] = "bias_float"
        if weight_scales is not None:
            scales = np.asarray(weight_scales, dtype=np.float32).reshape(-1)
            if scales.size not in {1, output_channels}:
                raise ValueError(
                    f"Weight scales must have 1 or {output_channels} values, got {scales.size}."
                )
            conv_metadata["weight_scales_key"] = "weight_scales"

        program = self.compile_matmul(
            weight_matrix,
            input_rows=batch * output_h * output_w,
            name=name,
            bias_i32=bias_i32,
            metadata=conv_metadata,
        )
        program.kind = "conv2d"
        if bias_float is not None:
            program.constants["bias_float"] = np.asarray(bias_float, dtype=np.float32)
        if weight_scales is not None:
            program.constants["weight_scales"] = np.asarray(
                weight_scales,
                dtype=np.float32,
            ).reshape(-1)
        program.validate()
        return program
