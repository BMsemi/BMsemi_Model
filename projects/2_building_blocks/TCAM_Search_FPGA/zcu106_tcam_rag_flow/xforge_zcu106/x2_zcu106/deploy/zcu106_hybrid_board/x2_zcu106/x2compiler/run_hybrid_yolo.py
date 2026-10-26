from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from .hybrid_runtime import (
    HybridTFLiteExecutor,
    OperatorTrace,
    compare_outputs,
    save_trace,
)
from .postprocess import (
    compare_detections,
    decode_yolo,
    save_detection_artifacts,
)
from .runtime import create_accelerator_backend, public_backend_name


def _artifact_name(path: str | Path) -> str:
    return Path(path).name


def _offload_indices(
    value: str,
    available: set[int],
) -> set[int]:
    normalized = value.strip().lower()
    if normalized in {"all", "all-compiled"}:
        return set(available)
    if normalized in {"none", "cpu"}:
        return set()
    result = {
        int(item.strip())
        for item in value.split(",")
        if item.strip()
    }
    unknown = result - available
    if unknown:
        raise ValueError(f"Uncompiled Conv2D operator(s): {sorted(unknown)}.")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run compiled YOLO inference on the local or external engine."
    )
    parser.add_argument("model", type=Path)
    parser.add_argument("artifacts", type=Path)
    parser.add_argument("image", type=Path)
    parser.add_argument(
        "--backend",
        choices=("local", "external"),
        default="local",
    )
    parser.add_argument(
        "--runtime-package",
        type=Path,
        default=None,
        help="External runtime package; required only with --backend external.",
    )
    parser.add_argument(
        "--offload",
        default="all",
        help="all, none, or comma-separated TFLite Conv2D operator indices.",
    )
    parser.add_argument(
        "--preprocess",
        choices=("resize", "letterbox"),
        default="resize",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("compiler_output"),
    )
    parser.add_argument("--confidence", type=float, default=0.25)
    parser.add_argument("--iou", type=float, default=0.45)
    parser.add_argument("--max-detections", type=int, default=100)
    parser.add_argument(
        "--validate-reference",
        action="store_true",
        help="Run LiteRT once after strict execution and report output error.",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress periodic operator progress.",
    )
    args = parser.parse_args()

    manifest = json.loads(
        (args.artifacts / "model_manifest.json").read_text(encoding="utf-8")
    )
    available = {
        int(item["op_index"]) for item in manifest["compiled_programs"]
    }
    selected = _offload_indices(args.offload, available)
    backend_name = public_backend_name(args.backend)
    if backend_name == "external" and args.runtime_package is None:
        parser.error("--runtime-package is required with --backend external.")
    backend = create_accelerator_backend(
        backend_name,
        runtime_package=args.runtime_package,
    )

    executor = HybridTFLiteExecutor(
        args.model,
        args.artifacts,
        backend,
        offload_conv_indices=selected,
    )
    input_tensor, original_bgr, preprocess_meta = executor.preprocess_image(
        args.image,
        mode=args.preprocess,
    )

    def progress(
        completed: int,
        total: int,
        trace: OperatorTrace,
    ) -> None:
        if not args.quiet and (
            completed == 1
            or completed == total
            or completed % 25 == 0
        ):
            print(
                f"[{completed:03d}/{total:03d}] op {trace.op_index:03d} "
                f"{trace.op_name} -> {trace.backend} "
                f"({trace.duration_ms:.1f} ms)",
                flush=True,
            )

    result = executor.run(input_tensor, progress=progress)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_path = args.output_dir / "raw_output.npy"
    np.save(output_path, result.output)
    trace_path = args.output_dir / "execution_trace.json"
    save_trace(trace_path, result)

    detections = decode_yolo(
        result.output,
        preprocess_meta,
        confidence_threshold=args.confidence,
        iou_threshold=args.iou,
        max_detections=args.max_detections,
    )
    detection_artifacts = save_detection_artifacts(
        args.output_dir,
        original_bgr,
        detections,
        metadata={"preprocess": preprocess_meta},
    )

    report: dict[str, Any] = {
        "schema_version": 1,
        "model": args.model.name,
        "compiler_artifacts": args.artifacts.name,
        "image": args.image.name,
        "backend": backend_name,
        "offloaded_conv2d_count": len(selected),
        "preprocess": preprocess_meta,
        "execution": result.summary(),
        "detection_count": len(detections),
        "artifacts": {
            "raw_output": output_path.name,
            "execution_trace": trace_path.name,
            **{
                key: _artifact_name(value)
                for key, value in detection_artifacts.items()
            },
        },
    }
    diagnostics = getattr(backend, "diagnostics", None)
    if callable(diagnostics):
        report[str(getattr(backend, "diagnostics_key", "accelerator"))] = (
            diagnostics()
        )
    if args.validate_reference:
        reference = executor.reference_output(input_tensor)
        reference_path = args.output_dir / "reference_output.npy"
        np.save(reference_path, reference)
        report["reference_validation"] = compare_outputs(
            result.output,
            reference,
        )
        reference_detections = decode_yolo(
            reference,
            preprocess_meta,
            confidence_threshold=args.confidence,
            iou_threshold=args.iou,
            max_detections=args.max_detections,
        )
        report["reference_validation"]["detections"] = compare_detections(
            detections,
            reference_detections,
        )
        report["artifacts"]["reference_output"] = reference_path.name

    report_path = args.output_dir / "run_report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    report["artifacts"]["run_report"] = report_path.name
    # Rewrite once so the self-reference is present in the persisted report.
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
