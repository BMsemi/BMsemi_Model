from __future__ import annotations

import argparse
import json
from pathlib import Path

from .tflite_frontend import TFLiteX2Frontend


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Freeze static X2 activation scales from a completed dynamic "
            "hybrid execution trace."
        )
    )
    parser.add_argument("model", type=Path)
    parser.add_argument("trace", type=Path)
    parser.add_argument("compile_output", type=Path)
    parser.add_argument("--headroom", type=float, default=1.10)
    parser.add_argument(
        "--calibration-output",
        type=Path,
        default=None,
    )
    args = parser.parse_args()
    if args.headroom < 1.0:
        raise ValueError("Headroom must be at least 1.0.")

    trace = json.loads(args.trace.read_text(encoding="utf-8"))
    maxima = {
        int(item["op_index"]): float(item["activation_max_abs"])
        for item in trace["operators"]
        if item["backend"] == "x2"
    }
    scales = {
        index: max(maximum * args.headroom / 127.0, 1e-12)
        for index, maximum in maxima.items()
    }
    report = {
        "schema_version": 1,
        "source": "dynamic_hybrid_execution_trace",
        "model": str(args.model.resolve()),
        "trace": str(args.trace.resolve()),
        "headroom": args.headroom,
        "conv2d_count": len(scales),
        "activation_max_abs": {
            str(index): maxima[index] for index in sorted(maxima)
        },
        "activation_scales": {
            str(index): scales[index] for index in sorted(scales)
        },
    }
    calibration_output = (
        args.calibration_output
        or args.compile_output / "activation_calibration.json"
    )
    calibration_output.parent.mkdir(parents=True, exist_ok=True)
    calibration_output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    manifest = TFLiteX2Frontend(args.model).compile_model(
        args.compile_output,
        activation_scale_overrides=scales,
        allow_saturation=True,
    )
    print(json.dumps({
        "calibration": str(calibration_output.resolve()),
        "compiled_artifacts": str(args.compile_output.resolve()),
        "compiled_conv2d_count": manifest["compiled_conv2d_count"],
        "compile_error_count": manifest["compile_error_count"],
    }, indent=2))


if __name__ == "__main__":
    main()
