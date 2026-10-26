from __future__ import annotations

import argparse
import json
from pathlib import Path

from .tflite_frontend import TFLiteX2Frontend


def main() -> int:
    parser = argparse.ArgumentParser(description="Compile TFLite Conv2D operators for X2.")
    parser.add_argument("--model", required=True, help="Input .tflite model.")
    parser.add_argument("--output", required=True, help="Output artifact directory.")
    parser.add_argument(
        "--op-index",
        type=int,
        help="Compile one TFLite operator instead of every eligible Conv2D.",
    )
    parser.add_argument(
        "--activation-max-abs",
        type=float,
        help="Static float-activation calibration range; scale=max_abs/127.",
    )
    parser.add_argument(
        "--allow-saturation",
        action="store_true",
        help="Allow explicit clipping when centered activations exceed signed INT8.",
    )
    args = parser.parse_args()

    frontend = TFLiteX2Frontend(args.model)
    output = Path(args.output)
    if args.op_index is not None:
        scale = None
        if args.activation_max_abs is not None:
            if args.activation_max_abs <= 0:
                raise ValueError("--activation-max-abs must be positive.")
            scale = float(args.activation_max_abs) / 127.0
        program = frontend.compile_conv(
            args.op_index,
            activation_scale=scale,
            allow_saturation=args.allow_saturation,
        )
        program.save(output)
        summary = {
            "status": "ok",
            "op_index": args.op_index,
            "output": str(output.resolve()),
            "tiles": len(program.tiles),
            "logical_shape": [program.m_dim, program.k_dim, program.n_dim],
        }
    else:
        manifest = frontend.compile_model(
            output,
            allow_saturation=args.allow_saturation,
        )
        summary = {
            "status": "ok" if not manifest["compile_errors"] else "partial",
            "output": str(output.resolve()),
            "compiled_conv2d_count": manifest["compiled_conv2d_count"],
            "compile_error_count": manifest["compile_error_count"],
        }
    print(json.dumps(summary, indent=2))
    return 0 if summary["status"] in {"ok", "partial"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
