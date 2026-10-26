from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from hybrid_inference_executor import (
    TFLiteHybridExecutor,
    _dequantize,
    _is_float_tensor,
    _is_quantized_int,
    _is_quantized_weight,
)
from user import letterbox_ultralytics


class FullInferenceExecutor(TFLiteHybridExecutor):
    """
    Full-model executor entry point.

    This uses the local yolo26n_int8 TFLite model layout directly.  The current
    local model has float NCHW input, so preprocessing must not reuse the older
    full-integer NHWC path from user.load_rgb_image().
    """

    def __init__(
        self,
        model_path: str | Path = "yolo26n_int8.tflite",
        *,
        mapping_path: str | Path | None = "Yolo26n_auto_full_mapping.m",
        num_pes: int = 512,
        preprocess_mode: str = "plain-resize",
    ) -> None:
        self.preprocess_mode = preprocess_mode
        super().__init__(model_path, mapping_path=mapping_path, num_pes=num_pes)

    def preprocess_image(self, image_path: str | Path) -> tuple[np.ndarray, dict]:
        bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if bgr is None:
            raise FileNotFoundError(f"Could not load image: {image_path}")

        input_shape = tuple(int(v) for v in self.input_detail["shape"])
        if len(input_shape) != 4:
            raise ValueError(f"Expected 4D model input, got shape {input_shape}")

        if input_shape[1] == 3:
            _, _, height, width = input_shape
            channel_layout = "NCHW"
        elif input_shape[-1] == 3:
            _, height, width, _ = input_shape
            channel_layout = "NHWC"
        else:
            raise ValueError(f"Could not infer RGB channel layout from input shape {input_shape}")

        if self.preprocess_mode == "letterbox":
            bgr_prepared, meta = letterbox_ultralytics(bgr, new_shape=(height, width))
            rgb = np.ascontiguousarray(bgr_prepared[:, :, ::-1])
        elif self.preprocess_mode == "plain-resize":
            rgb = cv2.resize(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB), (width, height), interpolation=cv2.INTER_LINEAR)
            meta = {
                "mode": "plain-resize",
                "original_shape": tuple(int(v) for v in bgr.shape[:2]),
                "new_shape": (height, width),
            }
        else:
            raise ValueError(f"Unsupported preprocess mode: {self.preprocess_mode}")

        real_input = rgb.astype(np.float32) / 255.0
        if channel_layout == "NCHW":
            real_input = np.transpose(real_input[None, ...], (0, 3, 1, 2))
        else:
            real_input = real_input[None, ...]

        dtype = self.input_detail["dtype"]
        if dtype == np.float32:
            return np.ascontiguousarray(real_input, dtype=np.float32), meta

        scale, zero_point = self.input_detail.get("quantization", (0.0, 0))
        if not scale:
            return np.ascontiguousarray(real_input, dtype=dtype), meta

        q = np.rint(real_input / float(scale) + int(zero_point))
        if dtype == np.int8:
            q = np.clip(q, -128, 127)
        elif dtype == np.uint8:
            q = np.clip(q, 0, 255)
        return np.ascontiguousarray(q.astype(dtype)), meta

    def _invoke_reference(self, input_tensor: np.ndarray) -> None:
        self.reference_cache.clear()
        self.interpreter.set_tensor(
            int(self.input_detail["index"]),
            np.asarray(input_tensor, dtype=self.input_detail["dtype"]),
        )
        self.interpreter.invoke()

    def _compute_conv_op(
        self,
        op: dict,
        tensor_cache: dict[int, np.ndarray],
        *,
        layer_name: str,
    ) -> np.ndarray:
        input_ids = [int(idx) for idx in op.get("inputs", [])]
        output_ids = [int(idx) for idx in op.get("outputs", [])]
        input_meta = self.tensor_meta[input_ids[0]]
        weight_meta = self.tensor_meta[input_ids[1]]
        output_meta = self.tensor_meta[output_ids[0]]

        if not _is_quantized_weight(weight_meta):
            raise NotImplementedError("Crossbar replay requires quantized weights; using reference op.")

        if _is_quantized_int(input_meta) and _is_quantized_int(output_meta):
            return super()._compute_conv_op(op, tensor_cache, layer_name=layer_name)

        if _is_float_tensor(input_meta) and _is_float_tensor(output_meta):
            input_q_or_float = self._get_tensor_value(tensor_cache, input_ids[0])
            weight_q = self._get_reference_tensor(input_ids[1])
            bias = self._get_reference_tensor(input_ids[2]) if len(input_ids) > 2 and input_ids[2] >= 0 else None
            if bias is not None and np.issubdtype(np.dtype(bias.dtype), np.integer):
                bias = _dequantize(bias, self.tensor_meta[input_ids[2]])
            return self.crossbar_runner.run_float_activation_conv(
                layer_name=layer_name,
                op_name=str(op["op_name"]),
                input_float=input_q_or_float,
                weight_q=weight_q,
                weight_scales=weight_meta.scales,
                bias_float=bias,
            )

        raise NotImplementedError(
            "Crossbar replay requires either quantized activation/output tensors or float32 activation/output tensors "
            "for the runtime INT8 bridge; using reference op."
        )


def _backend_counts(records: list[Any]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for record in records:
        counts[record.backend] = counts.get(record.backend, 0) + 1
    return counts


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the full YOLO26n TFLite inference executor.")
    parser.add_argument("--image", default="bus.jpg", help="Input image path.")
    parser.add_argument("--model", default="yolo26n_int8.tflite", help="Local YOLO26n INT8 TFLite model path.")
    parser.add_argument("--mapping", default="Yolo26n_auto_full_mapping.m", help="Auto-generated MAESTRO mapping file.")
    parser.add_argument("--save-dir", default="full_inference_outputs", help="Directory for outputs and execution trace.")
    parser.add_argument(
        "--preprocess",
        default="plain-resize",
        choices=("plain-resize", "letterbox"),
        help="Input preprocessing mode. plain-resize matches the verified layer-0 LiteRT reference.",
    )
    parser.add_argument(
        "--crossbar-layers",
        default="all-eligible",
        help="Comma-separated Conv/Depthwise layer names to replay on crossbar, or all-eligible for every compatible layer.",
    )
    args = parser.parse_args()

    mapping_path = Path(args.mapping) if args.mapping else None
    executor = FullInferenceExecutor(
        args.model,
        mapping_path=mapping_path,
        preprocess_mode=args.preprocess,
    )

    requested_layers = args.crossbar_layers.strip()
    if requested_layers.lower() in {"all", "all-eligible", "eligible"}:
        crossbar_layers, ineligible_layers = executor.crossbar_eligible_layers()
        print(f"Conv/Depthwise candidates: {len(executor.crossbar_candidate_layers())}")
        print(f"PE/crossbar eligible layers: {len(crossbar_layers)}")
        print(f"Software/reference fallbacks: {len(ineligible_layers)}")
    else:
        crossbar_layers = {name.strip() for name in requested_layers.split(",") if name.strip()}

    output_q, output_real, records = executor.run(args.image, crossbar_layers=crossbar_layers, save_dir=args.save_dir)

    print(f"Saved outputs to {Path(args.save_dir).resolve()}")
    print(f"Final output dtype: {output_q.dtype}")
    print(f"Final output shape: {output_q.shape}")
    print(f"Final float output shape: {output_real.shape}")
    print(f"Backends used: {_backend_counts(records)}")
    if output_real.size:
        print("First 5 flattened rows:")
        for row in output_real.reshape(-1, output_real.shape[-1])[:5]:
            print(" ".join(f"{float(v):.6f}" for v in row))


if __name__ == "__main__":
    main()
