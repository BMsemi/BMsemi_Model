from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np

from .ir import X2Program
from .runtime import create_accelerator_backend, public_backend_name, execute_conv2d
from .tflite_frontend import TFLiteX2Frontend


def _metrics(candidate: np.ndarray, reference: np.ndarray) -> dict[str, float]:
    difference = np.asarray(candidate, dtype=np.float32) - np.asarray(
        reference,
        dtype=np.float32,
    )
    return {
        "mae": float(np.mean(np.abs(difference))),
        "rmse": float(np.sqrt(np.mean(np.square(difference)))),
        "max_abs": float(np.max(np.abs(difference))),
        "reference_abs_mean": float(np.mean(np.abs(reference))),
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Validate a compiled Conv2D program against LiteRT."
    )
    parser.add_argument("--model", required=True, help="Source TFLite model.")
    parser.add_argument(
        "--program",
        required=True,
        help="Compiled program directory.",
    )
    parser.add_argument("--output", required=True, help="Validation JSON path.")
    parser.add_argument(
        "--input-mode",
        choices=("random-0-1", "zeros"),
        default="random-0-1",
    )
    parser.add_argument(
        "--backend",
        choices=("local",),
        default="local",
    )
    parser.add_argument("--seed", type=int, default=20260727)
    args = parser.parse_args()

    program = X2Program.load(args.program)
    source = dict(program.metadata["source"])
    if source.get("format") != "tflite":
        raise ValueError("The program does not contain TFLite source metadata.")
    frontend = TFLiteX2Frontend(args.model)
    model_inputs = frontend.interpreter.get_input_details()
    if len(model_inputs) != 1:
        raise ValueError("Validation currently expects one model input.")
    model_input = model_inputs[0]
    input_shape = tuple(int(value) for value in model_input["shape"])
    if args.input_mode == "zeros":
        sample = np.zeros(input_shape, dtype=model_input["dtype"])
    else:
        rng = np.random.default_rng(args.seed)
        sample = rng.random(input_shape, dtype=np.float32).astype(
            model_input["dtype"],
            copy=False,
        )

    frontend.interpreter.set_tensor(int(model_input["index"]), sample)
    invoke_start = time.perf_counter()
    frontend.interpreter.invoke()
    invoke_s = time.perf_counter() - invoke_start

    activation_index = int(source["activation_tensor"]["index"])
    output_index = int(source["output_tensor"]["index"])
    layer_input = np.array(
        frontend.interpreter.get_tensor(activation_index),
        copy=True,
    )
    reference = np.array(
        frontend.interpreter.get_tensor(output_index),
        copy=True,
    )
    accelerator_start = time.perf_counter()
    backend_name = public_backend_name(args.backend)
    backend = create_accelerator_backend(backend_name)
    result = execute_conv2d(program, layer_input, backend)
    accelerator_s = time.perf_counter() - accelerator_start
    if result.output_float is None:
        raise ValueError("Compiled program does not produce a float comparison output.")

    report = {
        "status": "pass",
        "model": Path(args.model).name,
        "program": Path(args.program).name,
        "source_op_index": int(source["op_index"]),
        "backend": backend_name,
        "input_mode": args.input_mode,
        "seed": args.seed,
        "activation_scale": result.activation_scale,
        "layer_input": {
            "shape": list(layer_input.shape),
            "dtype": str(layer_input.dtype),
            "min": float(layer_input.min()),
            "max": float(layer_input.max()),
        },
        "reference_output": {
            "shape": list(reference.shape),
            "dtype": str(reference.dtype),
            "min": float(reference.min()),
            "max": float(reference.max()),
        },
        "error": _metrics(result.output_float, reference),
        "execution": {
            "litert_full_model_invoke_s": invoke_s,
            "accelerator_layer_s": accelerator_s,
            "backend_calls": result.trace.backend_calls,
            "input_words": result.trace.input_words,
            "output_words": result.trace.output_words,
            "logical_macs": result.trace.logical_macs,
            "accumulator_sha256": hashlib.sha256(
                np.ascontiguousarray(result.accumulators).tobytes()
            ).hexdigest(),
            str(getattr(backend, "diagnostics_key", "accelerator")): (
                backend.diagnostics()
            ),
        },
        "notes": [
            "The compiled integer accumulator path is covered by exact unit tests.",
            "The error reported here is the expected float-activation-to-INT8 bridge error.",
            "The measured time uses the selected local software accelerator backend.",
        ],
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({"status": "pass", "error": report["error"], "output": str(output.resolve())}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
