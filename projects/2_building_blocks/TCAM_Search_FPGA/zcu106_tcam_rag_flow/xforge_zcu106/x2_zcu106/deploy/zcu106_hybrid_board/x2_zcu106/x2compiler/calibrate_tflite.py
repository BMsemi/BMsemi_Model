from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from .tflite_frontend import TFLiteX2Frontend


def _make_interpreter(model: Path):
    import ai_edge_litert.interpreter as litert

    kwargs: dict[str, Any] = {
        "model_path": str(model.resolve()),
        "experimental_preserve_all_tensors": True,
    }
    resolver = getattr(litert, "OpResolverType", None)
    if resolver is not None:
        kwargs["experimental_op_resolver_type"] = (
            resolver.BUILTIN_WITHOUT_DEFAULT_DELEGATES
        )
    interpreter = litert.Interpreter(**kwargs)
    interpreter.allocate_tensors()
    return interpreter


def _preprocess(
    image_path: Path,
    input_detail: dict[str, Any],
    mode: str,
) -> np.ndarray:
    import cv2

    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"Could not load calibration image {image_path}.")
    shape = tuple(int(value) for value in input_detail["shape"])
    if shape[1] == 3:
        layout, height, width = "NCHW", shape[2], shape[3]
    elif shape[-1] == 3:
        layout, height, width = "NHWC", shape[1], shape[2]
    else:
        raise ValueError(f"Could not infer RGB input layout from {shape}.")
    if mode == "resize":
        prepared = cv2.resize(image, (width, height), interpolation=cv2.INTER_LINEAR)
    elif mode == "letterbox":
        original_h, original_w = image.shape[:2]
        gain = min(width / original_w, height / original_h)
        resized_w = int(round(original_w * gain))
        resized_h = int(round(original_h * gain))
        resized = cv2.resize(
            image,
            (resized_w, resized_h),
            interpolation=cv2.INTER_LINEAR,
        )
        left = int(round((width - resized_w) / 2.0 - 0.1))
        top = int(round((height - resized_h) / 2.0 - 0.1))
        prepared = np.full((height, width, 3), 114, dtype=np.uint8)
        prepared[top : top + resized_h, left : left + resized_w] = resized
    else:
        raise ValueError("Calibration preprocessing must be resize or letterbox.")
    rgb = cv2.cvtColor(prepared, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    if layout == "NCHW":
        rgb = np.transpose(rgb[None, ...], (0, 3, 1, 2))
    else:
        rgb = rgb[None, ...]
    return np.ascontiguousarray(rgb, dtype=input_detail["dtype"])


def calibrate(
    model: Path,
    images: list[Path],
    *,
    preprocess: str,
    headroom: float,
) -> dict[str, Any]:
    if not images:
        raise ValueError("At least one calibration image is required.")
    if headroom < 1.0:
        raise ValueError("Calibration headroom must be at least 1.0.")
    interpreter = _make_interpreter(model)
    input_detail = interpreter.get_input_details()[0]
    ops = list(interpreter._get_ops_details())  # noqa: SLF001
    conv_inputs = {
        int(op["index"]): int(op["inputs"][0])
        for op in ops
        if str(op["op_name"]) == "CONV_2D"
    }
    maxima = {index: 0.0 for index in conv_inputs}
    for image in images:
        input_tensor = _preprocess(image, input_detail, preprocess)
        interpreter.set_tensor(int(input_detail["index"]), input_tensor)
        interpreter.invoke()
        for op_index, tensor_index in conv_inputs.items():
            value = interpreter.get_tensor(tensor_index)
            maximum = float(np.max(np.abs(value))) if value.size else 0.0
            maxima[op_index] = max(maxima[op_index], maximum)

    scales = {
        op_index: max(maximum * headroom / 127.0, 1e-12)
        for op_index, maximum in maxima.items()
    }
    return {
        "schema_version": 1,
        "model": str(model.resolve()),
        "images": [str(image.resolve()) for image in images],
        "preprocess": preprocess,
        "headroom": headroom,
        "conv2d_count": len(scales),
        "activation_max_abs": {
            str(index): maxima[index] for index in sorted(maxima)
        },
        "activation_scales": {
            str(index): scales[index] for index in sorted(scales)
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Calibrate float activations and optionally rebuild X2 programs."
    )
    parser.add_argument("model", type=Path)
    parser.add_argument("images", nargs="+", type=Path)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("x2_activation_calibration.json"),
    )
    parser.add_argument(
        "--compile-output",
        type=Path,
        default=None,
        help="Build statically calibrated X2 artifacts into this directory.",
    )
    parser.add_argument(
        "--preprocess",
        choices=("resize", "letterbox"),
        default="resize",
    )
    parser.add_argument("--headroom", type=float, default=1.05)
    args = parser.parse_args()

    report = calibrate(
        args.model,
        args.images,
        preprocess=args.preprocess,
        headroom=args.headroom,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    result: dict[str, Any] = {
        "calibration": str(args.output.resolve()),
        "conv2d_count": report["conv2d_count"],
    }
    if args.compile_output is not None:
        scales = {
            int(index): float(scale)
            for index, scale in report["activation_scales"].items()
        }
        manifest = TFLiteX2Frontend(args.model).compile_model(
            args.compile_output,
            activation_scale_overrides=scales,
            allow_saturation=True,
        )
        result["compiled_artifacts"] = str(args.compile_output.resolve())
        result["compiled_conv2d_count"] = manifest["compiled_conv2d_count"]
        result["compile_error_count"] = manifest["compile_error_count"]
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
