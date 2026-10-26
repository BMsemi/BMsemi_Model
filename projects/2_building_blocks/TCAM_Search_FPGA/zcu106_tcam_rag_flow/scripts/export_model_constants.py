from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export immutable TFLite tensors for board runtimes that cannot read them directly."
    )
    parser.add_argument("--xforge", type=Path, required=True)
    parser.add_argument("--model", type=Path, default=None)
    parser.add_argument("--artifacts", type=Path, default=None)
    args = parser.parse_args()

    xforge = args.xforge.resolve()
    sys.path.insert(0, str(xforge))
    from x2_zcu106.x2compiler.hybrid_runtime import _interpreter

    model = args.model or xforge / "model" / "yolo26n_int8.tflite"
    artifacts = (
        args.artifacts
        or xforge / "x2_zcu106" / "compiler_artifacts" / "yolo26n_hybrid_calibrated"
    )
    interpreter = _interpreter(model)
    ops = list(interpreter._get_ops_details())  # noqa: SLF001
    details = interpreter.get_tensor_details()
    all_indices = {int(detail["index"]) for detail in details}
    produced = {
        int(index)
        for op in ops
        for index in op.get("outputs", [])
        if int(index) >= 0
    }
    model_inputs = {
        int(detail["index"]) for detail in interpreter.get_input_details()
    }
    constant_indices = sorted(all_indices - produced - model_inputs)

    constants: dict[str, np.ndarray] = {}
    failed: list[tuple[int, str]] = []
    for index in constant_indices:
        try:
            constants[f"tensor_{index}"] = np.array(
                interpreter.get_tensor(index),
                copy=True,
            )
        except Exception as exc:  # pragma: no cover - diagnostic script
            failed.append((index, f"{type(exc).__name__}: {exc}"))

    artifacts.mkdir(parents=True, exist_ok=True)
    archive = artifacts / "model_constants.npz"
    np.savez_compressed(archive, **constants)
    print(f"MODEL_CONSTANT_ARCHIVE={archive}")
    print(f"constants={len(constants)} failed={len(failed)} bytes={archive.stat().st_size}")
    if failed:
        print("FAILED_CONSTANTS")
        for index, reason in failed[:20]:
            print(f"{index}: {reason}")


if __name__ == "__main__":
    main()
