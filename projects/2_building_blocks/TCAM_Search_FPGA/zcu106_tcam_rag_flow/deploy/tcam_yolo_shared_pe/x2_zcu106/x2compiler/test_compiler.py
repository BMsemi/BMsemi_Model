from __future__ import annotations

import shutil
import subprocess
import tempfile
from pathlib import Path

import numpy as np

from x2_zcu106.x2compiler import (
    CrossbarX2Backend,
    PEArrayX2Backend,
    PynqX2Backend,
    TFLiteX2Frontend,
    X2Compiler,
    X2Program,
    execute_conv2d,
    execute_matrix,
)


ROOT = Path(__file__).resolve().parents[2]
MODEL = ROOT / "YOLO26n-TFLite-FPGA-Partner-Project" / "yolo26n_int8.tflite"


def test_tiled_matmul_exact_and_round_trip() -> None:
    rng = np.random.default_rng(1234)
    inputs = rng.integers(-128, 128, size=(35, 65), dtype=np.int16).astype(np.int8)
    weights = rng.integers(-128, 128, size=(65, 40), dtype=np.int16).astype(np.int8)
    program = X2Compiler().compile_matmul(
        weights,
        input_rows=inputs.shape[0],
        name="tiled_matmul_test",
    )

    assert len(program.tiles) == 6
    result = execute_matrix(program, inputs, CrossbarX2Backend())
    expected = inputs.astype(np.int32) @ weights.astype(np.int32)
    np.testing.assert_array_equal(result.accumulators, expected)
    assert result.trace.backend_calls == 12

    with tempfile.TemporaryDirectory(prefix="x2_program_roundtrip_") as temp_dir:
        output = Path(temp_dir) / "program"
        program.save(output)
        loaded = X2Program.load(output)
        round_trip = execute_matrix(loaded, inputs, CrossbarX2Backend())
        np.testing.assert_array_equal(round_trip.accumulators, expected)

        iree_opt = shutil.which("iree-opt")
        if iree_opt:
            subprocess.run(
                [
                    iree_opt,
                    "--allow-unregistered-dialect",
                    str(output / "program.mlir"),
                    "-o",
                    "NUL",
                ],
                check=True,
                capture_output=True,
                text=True,
                timeout=60,
            )


def test_vectorized_crossbar_backend_matches_batch_protocol() -> None:
    class BatchOnlyBackend:
        max_batch = 32

        def run_batch(
            self,
            vectors: np.ndarray,
            weights: np.ndarray,
        ) -> np.ndarray:
            return (
                np.asarray(vectors, dtype=np.int32)
                @ np.asarray(weights, dtype=np.int32)
            )

    rng = np.random.default_rng(55)
    inputs = rng.integers(
        -128,
        128,
        size=(77, 99),
        dtype=np.int16,
    ).astype(np.int8)
    weights = rng.integers(
        -128,
        128,
        size=(99, 61),
        dtype=np.int16,
    ).astype(np.int8)
    program = X2Compiler().compile_matmul(weights, input_rows=inputs.shape[0])
    vectorized = execute_matrix(program, inputs, CrossbarX2Backend())
    batched = execute_matrix(program, inputs, BatchOnlyBackend())
    np.testing.assert_array_equal(vectorized.accumulators, batched.accumulators)
    assert vectorized.trace == batched.trace


def test_pe_backend_matches_tiled_protocol() -> None:
    rng = np.random.default_rng(77)
    inputs = rng.integers(
        -128,
        128,
        size=(73, 67),
        dtype=np.int16,
    ).astype(np.int8)
    weights = rng.integers(
        -128,
        128,
        size=(67, 41),
        dtype=np.int16,
    ).astype(np.int8)
    program = X2Compiler().compile_matmul(weights, input_rows=inputs.shape[0])
    backend = PEArrayX2Backend()
    result = execute_matrix(program, inputs, backend)
    expected = inputs.astype(np.int32) @ weights.astype(np.int32)

    np.testing.assert_array_equal(result.accumulators, expected)
    assert result.trace.backend_calls == 18
    diagnostics = backend.diagnostics()
    assert diagnostics["engine"] == "LocalAcceleratorEngine"
    assert diagnostics["programming_events"] == len(program.tiles)
    assert diagnostics["input_buffer_loads"] == 18
    assert diagnostics["output_buffer_writes"] == 18


def test_pynq_backend_schedules_dual_pe_batches() -> None:
    calls = []

    class FakeDualAccelerator:
        def run_mvm_batch(self, vectors, weights):
            calls.append(np.asarray(vectors).shape[0])
            return (
                np.asarray(vectors, dtype=np.int32)
                @ np.asarray(weights, dtype=np.int32)
            )

    inputs = np.arange(65 * 3, dtype=np.int16).reshape(65, 3).astype(np.int8)
    weights = np.arange(3 * 2, dtype=np.int8).reshape(3, 2)
    program = X2Compiler().compile_matmul(
        weights,
        input_rows=inputs.shape[0],
        name="dual_pe_batch_test",
    )

    result = execute_matrix(program, inputs, PynqX2Backend(FakeDualAccelerator()))

    np.testing.assert_array_equal(
        result.accumulators,
        inputs.astype(np.int32) @ weights.astype(np.int32),
    )
    assert calls == [64, 1]
    assert result.trace.backend_calls == 2


def _manual_same_conv(
    inputs: np.ndarray,
    weights_ohwi: np.ndarray,
    stride: tuple[int, int],
) -> np.ndarray:
    batch, input_h, input_w, _ = inputs.shape
    output_channels, kernel_h, kernel_w, _ = weights_ohwi.shape
    stride_y, stride_x = stride
    output_h = (input_h + stride_y - 1) // stride_y
    output_w = (input_w + stride_x - 1) // stride_x
    total_pad_h = max((output_h - 1) * stride_y + kernel_h - input_h, 0)
    total_pad_w = max((output_w - 1) * stride_x + kernel_w - input_w, 0)
    top = total_pad_h // 2
    left = total_pad_w // 2
    padded = np.pad(
        inputs.astype(np.int32),
        (
            (0, 0),
            (top, total_pad_h - top),
            (left, total_pad_w - left),
            (0, 0),
        ),
        mode="constant",
    )
    output = np.zeros(
        (batch, output_h, output_w, output_channels),
        dtype=np.int32,
    )
    for batch_index in range(batch):
        for output_y in range(output_h):
            for output_x in range(output_w):
                patch = padded[
                    batch_index,
                    output_y * stride_y : output_y * stride_y + kernel_h,
                    output_x * stride_x : output_x * stride_x + kernel_w,
                    :,
                ]
                for channel in range(output_channels):
                    output[batch_index, output_y, output_x, channel] = np.sum(
                        patch * weights_ohwi[channel].astype(np.int32)
                    )
    return output


def test_synthetic_conv_exact() -> None:
    rng = np.random.default_rng(5678)
    inputs = rng.integers(-8, 8, size=(1, 6, 5, 3), dtype=np.int8)
    weights = rng.integers(-8, 8, size=(7, 3, 3, 3), dtype=np.int8)
    bias = rng.normal(size=(7,)).astype(np.float32)
    stride = (2, 1)
    output_shape = (1, 3, 5, 7)
    program = X2Compiler().compile_conv2d(
        weights,
        input_shape=inputs.shape,
        output_shape=output_shape,
        stride=stride,
        padding="same",
        bias_float=bias,
        weight_scales=np.ones((7,), dtype=np.float32),
        metadata={
            "activation_quantization": {
                "mode": "tflite_integer",
                "scale": 1.0,
                "zero_point": 0,
                "allow_saturation": False,
            }
        },
    )
    result = execute_conv2d(program, inputs, PEArrayX2Backend())
    expected = _manual_same_conv(inputs, weights, stride)
    np.testing.assert_array_equal(result.accumulators, expected)
    np.testing.assert_allclose(
        result.output_float,
        expected.astype(np.float32) + bias.reshape(1, 1, 1, -1),
        rtol=0,
        atol=0,
    )


def test_actual_tflite_first_conv_import_and_zero_execution() -> None:
    frontend = TFLiteX2Frontend(MODEL)
    program = frontend.compile_conv(
        2,
        activation_scale=1.0 / 127.0,
        allow_saturation=False,
    )
    assert program.kind == "conv2d"
    assert (program.m_dim, program.k_dim, program.n_dim) == (102400, 27, 16)
    assert len(program.tiles) == 1
    assert program.metadata["conv2d"]["stride"] == [2, 2]
    assert program.metadata["conv2d"]["padding"] == "valid"

    input_shape = tuple(program.metadata["conv2d"]["input_shape_nhwc"])
    result = execute_conv2d(
        program,
        np.zeros(input_shape, dtype=np.float32),
        PEArrayX2Backend(),
    )
    bias = np.asarray(program.constants["bias_float"], dtype=np.float32)
    np.testing.assert_allclose(
        result.output_float,
        np.broadcast_to(bias, result.output_float.shape),
        rtol=0,
        atol=0,
    )
    assert result.trace.backend_calls == 3200


if __name__ == "__main__":
    test_tiled_matmul_exact_and_round_trip()
    test_vectorized_crossbar_backend_matches_batch_protocol()
    test_pe_backend_matches_tiled_protocol()
    test_pynq_backend_schedules_dual_pe_batches()
    test_synthetic_conv_exact()
    test_actual_tflite_first_conv_import_and_zero_execution()
    print("PASS")
