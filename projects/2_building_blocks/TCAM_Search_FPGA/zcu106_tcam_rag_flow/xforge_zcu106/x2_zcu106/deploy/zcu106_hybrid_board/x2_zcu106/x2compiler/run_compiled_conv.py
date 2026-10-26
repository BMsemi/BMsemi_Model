from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import numpy as np

from .ir import X2Program
from .runtime import create_accelerator_backend, public_backend_name, execute_conv2d


def _image_to_first_conv_input(image_path: str | Path, expected_shape: tuple[int, ...]) -> np.ndarray:
    from PIL import Image

    if expected_shape != (1, 642, 642, 3):
        raise ValueError(
            "--image currently implements the YOLO first-layer 640x640 input "
            f"plus one-pixel padding, but this program expects {expected_shape}."
        )
    image = Image.open(image_path).convert("RGB").resize((640, 640))
    normalized = np.asarray(image, dtype=np.float32) / np.float32(255.0)
    return np.pad(
        normalized[None, ...],
        ((0, 0), (1, 1), (1, 1), (0, 0)),
        mode="constant",
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Execute one compiled Conv2D layer."
    )
    parser.add_argument(
        "--program",
        required=True,
        help="Compiled program directory.",
    )
    parser.add_argument(
        "--backend",
        choices=("local", "external"),
        default="local",
    )
    parser.add_argument("--runtime-package", help="External runtime package.")
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--input-npy", help="NHWC activation tensor file.")
    source.add_argument("--image", help="Image input for the compiled first YOLO layer.")
    parser.add_argument("--output", required=True, help="Output artifact directory.")
    args = parser.parse_args()

    program = X2Program.load(args.program)
    config = dict(program.metadata["conv2d"])
    expected_shape = tuple(int(value) for value in config["input_shape_nhwc"])
    if args.input_npy:
        input_value = np.load(args.input_npy, allow_pickle=False)
    elif args.image:
        input_value = _image_to_first_conv_input(args.image, expected_shape)
    else:
        input_value = np.zeros(expected_shape, dtype=np.float32)

    backend_name = public_backend_name(args.backend)
    if backend_name == "external" and not args.runtime_package:
        parser.error("--runtime-package is required with --backend external.")
    backend = create_accelerator_backend(
        backend_name,
        runtime_package=args.runtime_package,
    )

    start = time.perf_counter()
    result = execute_conv2d(program, input_value, backend)
    elapsed_s = time.perf_counter() - start

    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    np.save(output / "quantized_input.npy", result.quantized_input)
    np.save(output / "accumulators_i32.npy", result.accumulators)
    if result.output_float is not None:
        np.save(output / "output_float.npy", result.output_float)
    if result.output_quantized is not None:
        np.save(output / "output_quantized.npy", result.output_quantized)
    report = {
        "status": "pass",
        "backend": backend_name,
        "program": Path(args.program).name,
        "elapsed_s": elapsed_s,
        "activation_scale": result.activation_scale,
        "output_shape": list(result.accumulators.shape),
        "backend_calls": result.trace.backend_calls,
        "input_words": result.trace.input_words,
        "output_words": result.trace.output_words,
        "logical_macs": result.trace.logical_macs,
    }
    diagnostics = getattr(backend, "diagnostics", None)
    if callable(diagnostics):
        report[str(getattr(backend, "diagnostics_key", "accelerator"))] = (
            diagnostics()
        )
    (output / "run.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
