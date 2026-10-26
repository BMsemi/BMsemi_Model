from __future__ import annotations

import numpy as np

from x2_zcu106.x2compiler.planner import build_memory_plan
from x2_zcu106.x2compiler.postprocess import (
    Detection,
    compare_detections,
)
from x2_zcu106.x2compiler.tflite_ops import (
    TensorMeta,
    execute_digital_op,
)


def _meta(index: int, shape: tuple[int, ...], dtype=np.float32) -> TensorMeta:
    return TensorMeta(
        index=index,
        name=f"tensor_{index}",
        shape=shape,
        dtype=np.dtype(dtype),
        scale=0.0,
        zero_point=0,
        scales=np.asarray([], dtype=np.float32),
        zero_points=np.asarray([], dtype=np.int64),
        quantized_dimension=0,
    )


def test_digital_operators() -> None:
    lhs = np.asarray([[-1.0, 2.0]], dtype=np.float32)
    rhs = np.asarray([[3.0, 4.0]], dtype=np.float32)
    meta = [_meta(0, lhs.shape), _meta(1, rhs.shape)]
    output = [_meta(2, lhs.shape)]
    [added] = execute_digital_op("ADD", [lhs, rhs], meta, output)
    np.testing.assert_array_equal(added, np.asarray([[2.0, 6.0]]))
    [logistic] = execute_digital_op("LOGISTIC", [lhs], meta[:1], output)
    np.testing.assert_allclose(
        logistic,
        1.0 / (1.0 + np.exp(-lhs)),
        rtol=1e-6,
        atol=1e-7,
    )


def test_memory_plan_reuses_non_overlapping_tensors() -> None:
    tensors = {
        0: _meta(0, (16,)),
        1: _meta(1, (16,)),
        2: _meta(2, (16,)),
    }
    ops = [
        {"index": 0, "inputs": [0], "outputs": [1]},
        {"index": 1, "inputs": [1], "outputs": [2]},
    ]
    plan = build_memory_plan(ops, tensors, set(), {2}, alignment=64)
    assert plan["arena_bytes"] < plan["naive_activation_bytes"]
    assert all(
        allocation["offset"] % 64 == 0
        for allocation in plan["allocations"]
    )


def test_detection_comparison() -> None:
    reference = [
        Detection(0, 5, "bus", 0.9, [0, 0, 100, 100], [0, 0, 100, 100])
    ]
    candidate = [
        Detection(0, 5, "bus", 0.8, [2, 2, 98, 98], [2, 2, 96, 96])
    ]
    comparison = compare_detections(candidate, reference)
    assert comparison["matched_count"] == 1
    assert comparison["precision"] == 1.0
    assert comparison["recall"] == 1.0
    assert comparison["mean_iou"] > 0.9


if __name__ == "__main__":
    test_digital_operators()
    test_memory_plan_reuses_non_overlapping_tensors()
    test_detection_comparison()
    print("PASS")
