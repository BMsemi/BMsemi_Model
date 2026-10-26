from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import cv2
import numpy as np


COCO80_CLASSES = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "traffic light",
    "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
    "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
    "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard",
    "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
    "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
    "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard",
    "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase",
    "scissors", "teddy bear", "hair drier", "toothbrush",
]


def import_tflite_interpreter() -> tuple[Any, str]:
    try:
        import ai_edge_litert.interpreter as interpreter_module  # type: ignore

        return interpreter_module, "ai_edge_litert"
    except Exception:
        pass

    try:
        import tflite_runtime.interpreter as interpreter_module  # type: ignore

        return interpreter_module, "tflite_runtime"
    except Exception:
        pass

    from tensorflow.lite.python import interpreter as interpreter_module  # type: ignore

    return interpreter_module, "tensorflow.lite"


@dataclass(frozen=True)
class YoloCoreConfig:
    model_path: str | Path = "yolo26n_int8.tflite"
    labels_path: str | Path | None = None
    input_size: tuple[int, int] | None = None
    preprocess: str = "resize"
    conf_threshold: float = 0.25
    iou_threshold: float = 0.45
    max_detections: int = 100
    num_threads: int | None = None


@dataclass(frozen=True)
class Detection:
    rank: int
    class_id: int
    label: str
    confidence: float
    box_xyxy: list[int]
    box_xywh: list[int]


@dataclass(frozen=True)
class YoloCoreResult:
    detections: list[Detection]
    timings: dict[str, float]
    model: dict[str, Any]
    preprocess: dict[str, Any]
    raw_output_shapes: list[list[int]]
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "detections": [asdict(det) for det in self.detections],
            "timings": self.timings,
            "model": self.model,
            "preprocess": self.preprocess,
            "raw_output_shapes": self.raw_output_shapes,
            "warnings": list(self.warnings),
        }


def load_labels(labels_path: str | Path | None) -> list[str]:
    if labels_path is None:
        return list(COCO80_CLASSES)
    path = Path(labels_path)
    labels = [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not labels:
        raise ValueError(f"No labels found in {path}")
    return labels


def _json_ready(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, type):
        return str(value)
    if isinstance(value, dict):
        return {str(k): _json_ready(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(v) for v in value]
    return value


def _infer_layout(input_shape: tuple[int, ...]) -> tuple[str, int, int]:
    if len(input_shape) != 4:
        raise ValueError(f"Expected 4D YOLO input tensor, got shape {input_shape}")
    if input_shape[1] == 3:
        return "NCHW", int(input_shape[3]), int(input_shape[2])
    if input_shape[-1] == 3:
        return "NHWC", int(input_shape[2]), int(input_shape[1])
    raise ValueError(f"Could not infer RGB channel layout from input shape {input_shape}")


def _letterbox_bgr(image: np.ndarray, width: int, height: int) -> tuple[np.ndarray, dict[str, Any]]:
    orig_h, orig_w = image.shape[:2]
    gain = min(width / orig_w, height / orig_h)
    new_w = int(round(orig_w * gain))
    new_h = int(round(orig_h * gain))
    pad_x = (width - new_w) / 2.0
    pad_y = (height - new_h) / 2.0
    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    canvas = np.full((height, width, 3), 114, dtype=np.uint8)
    left = int(round(pad_x - 0.1))
    top = int(round(pad_y - 0.1))
    canvas[top:top + new_h, left:left + new_w] = resized
    return canvas, {
        "mode": "letterbox",
        "original_shape": [int(orig_h), int(orig_w)],
        "new_shape": [int(height), int(width)],
        "gain": float(gain),
        "pad_x": float(left),
        "pad_y": float(top),
    }


def _preprocess_bgr(
    image: np.ndarray,
    *,
    input_detail: dict[str, Any],
    input_size: tuple[int, int] | None,
    mode: str,
) -> tuple[np.ndarray, dict[str, Any]]:
    input_shape = tuple(int(v) for v in input_detail["shape"])
    layout, inferred_w, inferred_h = _infer_layout(input_shape)
    width, height = input_size if input_size is not None else (inferred_w, inferred_h)
    orig_h, orig_w = image.shape[:2]

    if mode == "resize":
        prepared = cv2.resize(image, (width, height), interpolation=cv2.INTER_LINEAR)
        meta: dict[str, Any] = {
            "mode": "resize",
            "original_shape": [int(orig_h), int(orig_w)],
            "new_shape": [int(height), int(width)],
            "gain": None,
            "pad_x": 0.0,
            "pad_y": 0.0,
        }
    elif mode == "letterbox":
        prepared, meta = _letterbox_bgr(image, width, height)
    else:
        raise ValueError("preprocess must be 'resize' or 'letterbox'")

    rgb = cv2.cvtColor(prepared, cv2.COLOR_BGR2RGB)
    real_input = rgb.astype(np.float32) / 255.0
    if layout == "NCHW":
        real_input = np.transpose(real_input[None, ...], (0, 3, 1, 2))
    else:
        real_input = real_input[None, ...]

    dtype = np.dtype(input_detail["dtype"])
    if np.issubdtype(dtype, np.floating):
        return np.ascontiguousarray(real_input, dtype=dtype), meta

    scale, zero_point = input_detail.get("quantization", (0.0, 0))
    if not scale:
        return np.ascontiguousarray(real_input.astype(dtype)), meta

    q = np.rint(real_input / float(scale) + int(zero_point))
    if dtype == np.dtype(np.int8):
        q = np.clip(q, -128, 127)
    elif dtype == np.dtype(np.uint8):
        q = np.clip(q, 0, 255)
    return np.ascontiguousarray(q.astype(dtype)), meta


def _nms_xyxy(boxes: np.ndarray, scores: np.ndarray, iou_threshold: float, max_detections: int) -> list[int]:
    if boxes.size == 0:
        return []
    x1 = boxes[:, 0].astype(np.float32)
    y1 = boxes[:, 1].astype(np.float32)
    x2 = boxes[:, 2].astype(np.float32)
    y2 = boxes[:, 3].astype(np.float32)
    areas = np.maximum(0.0, x2 - x1) * np.maximum(0.0, y2 - y1)
    order = scores.astype(np.float32).argsort()[::-1]
    keep: list[int] = []
    while order.size > 0 and len(keep) < max_detections:
        idx = int(order[0])
        keep.append(idx)
        if order.size == 1:
            break
        rest = order[1:]
        xx1 = np.maximum(x1[idx], x1[rest])
        yy1 = np.maximum(y1[idx], y1[rest])
        xx2 = np.minimum(x2[idx], x2[rest])
        yy2 = np.minimum(y2[idx], y2[rest])
        inter = np.maximum(0.0, xx2 - xx1) * np.maximum(0.0, yy2 - yy1)
        union = areas[idx] + areas[rest] - inter
        iou = np.divide(inter, np.maximum(union, 1e-9))
        order = rest[iou <= float(iou_threshold)]
    return keep


def decode_yolo_output(
    output: np.ndarray,
    *,
    labels: list[str],
    preprocess_meta: dict[str, Any],
    conf_threshold: float,
    iou_threshold: float,
    max_detections: int,
) -> tuple[list[Detection], list[str]]:
    warnings: list[str] = []
    pred = np.asarray(output, dtype=np.float32)
    if pred.ndim == 3 and pred.shape[0] == 1:
        pred = pred[0]
    if pred.ndim != 2:
        raise ValueError(f"Expected YOLO output with 2 dimensions after batch removal, got {pred.shape}")
    if pred.shape[0] in {len(labels) + 4, len(labels) + 5, 84, 85} and pred.shape[0] < pred.shape[1]:
        pred = pred.T
    if pred.shape[1] < 5:
        raise ValueError(f"Expected YOLO predictions with at least 5 values, got {pred.shape}")

    orig_h, orig_w = [int(v) for v in preprocess_meta["original_shape"]]
    new_h, new_w = [int(v) for v in preprocess_meta["new_shape"]]
    boxes_xywh = pred[:, :4].copy()
    if float(np.nanmax(np.abs(boxes_xywh))) <= 2.0:
        boxes_xywh[:, [0, 2]] *= float(new_w)
        boxes_xywh[:, [1, 3]] *= float(new_h)

    class_values = pred[:, 4:]
    if class_values.shape[1] == len(labels) + 1:
        objectness = class_values[:, 0]
        class_scores = class_values[:, 1:]
        score_multiplier = objectness
    else:
        class_scores = class_values[:, :len(labels)]
        score_multiplier = np.ones((pred.shape[0],), dtype=np.float32)
        if class_values.shape[1] != len(labels):
            warnings.append(
                f"Output has {class_values.shape[1]} score columns but {len(labels)} labels; using the first {len(labels)} columns."
            )

    class_ids = np.argmax(class_scores, axis=1)
    confidences = class_scores[np.arange(class_scores.shape[0]), class_ids] * score_multiplier
    keep = confidences >= float(conf_threshold)
    if not np.any(keep):
        return [], warnings

    selected_boxes: list[list[float]] = []
    selected_scores: list[float] = []
    selected_class_ids: list[int] = []

    gain = preprocess_meta.get("gain")
    pad_x = float(preprocess_meta.get("pad_x") or 0.0)
    pad_y = float(preprocess_meta.get("pad_y") or 0.0)
    resize_scale_x = orig_w / float(new_w)
    resize_scale_y = orig_h / float(new_h)

    for idx in np.where(keep)[0]:
        cx, cy, bw, bh = [float(v) for v in boxes_xywh[idx]]
        x1 = cx - bw / 2.0
        y1 = cy - bh / 2.0
        x2 = cx + bw / 2.0
        y2 = cy + bh / 2.0
        if gain:
            x1 = (x1 - pad_x) / float(gain)
            x2 = (x2 - pad_x) / float(gain)
            y1 = (y1 - pad_y) / float(gain)
            y2 = (y2 - pad_y) / float(gain)
        else:
            x1 *= resize_scale_x
            x2 *= resize_scale_x
            y1 *= resize_scale_y
            y2 *= resize_scale_y
        x1 = max(0.0, min(float(orig_w - 1), x1))
        y1 = max(0.0, min(float(orig_h - 1), y1))
        x2 = max(0.0, min(float(orig_w - 1), x2))
        y2 = max(0.0, min(float(orig_h - 1), y2))
        if (x2 - x1) <= 1.0 or (y2 - y1) <= 1.0:
            continue
        selected_boxes.append([x1, y1, x2, y2])
        selected_scores.append(float(confidences[idx]))
        selected_class_ids.append(int(class_ids[idx]))

    if not selected_boxes:
        return [], warnings

    boxes = np.asarray(selected_boxes, dtype=np.float32)
    scores = np.asarray(selected_scores, dtype=np.float32)
    keep_indices = _nms_xyxy(boxes, scores, iou_threshold, max_detections)
    detections: list[Detection] = []
    for rank, local_idx in enumerate(keep_indices):
        x1, y1, x2, y2 = [int(round(v)) for v in boxes[local_idx]]
        class_id = selected_class_ids[local_idx]
        label = labels[class_id] if class_id < len(labels) else f"class_{class_id}"
        detections.append(
            Detection(
                rank=rank,
                class_id=class_id,
                label=label,
                confidence=float(scores[local_idx]),
                box_xyxy=[x1, y1, x2, y2],
                box_xywh=[x1, y1, max(0, x2 - x1), max(0, y2 - y1)],
            )
        )
    return detections, warnings


def draw_detections(image_bgr: np.ndarray, detections: list[Detection]) -> np.ndarray:
    annotated = image_bgr.copy()
    for det in detections:
        x1, y1, x2, y2 = det.box_xyxy
        label = f"{det.label} {det.confidence:.2f}"
        cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 180, 255), 2)
        text_size, baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
        ty = max(0, y1 - text_size[1] - baseline - 4)
        cv2.rectangle(
            annotated,
            (x1, ty),
            (x1 + text_size[0] + 6, ty + text_size[1] + baseline + 4),
            (0, 180, 255),
            -1,
        )
        cv2.putText(
            annotated,
            label,
            (x1 + 3, ty + text_size[1] + 1),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 0, 0),
            2,
            cv2.LINE_AA,
        )
    return annotated


class YoloCoreDetector:
    def __init__(self, config: YoloCoreConfig) -> None:
        self.config = config
        self.model_path = Path(config.model_path).expanduser().resolve()
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model not found: {self.model_path}")
        self.labels = load_labels(config.labels_path)
        interpreter_module, self.runtime_name = import_tflite_interpreter()
        kwargs: dict[str, Any] = {"model_path": str(self.model_path)}
        if config.num_threads is not None:
            kwargs["num_threads"] = int(config.num_threads)
        self.interpreter = interpreter_module.Interpreter(**kwargs)
        self.interpreter.allocate_tensors()
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()
        if len(self.input_details) != 1:
            raise ValueError(f"Expected one model input, got {len(self.input_details)}")

    def model_info(self) -> dict[str, Any]:
        return {
            "model_path": str(self.model_path),
            "runtime": self.runtime_name,
            "labels": len(self.labels),
            "inputs": _json_ready(self.input_details),
            "outputs": _json_ready(self.output_details),
        }

    def predict_array(
        self,
        image_bgr: np.ndarray,
        *,
        conf_threshold: float | None = None,
        iou_threshold: float | None = None,
        max_detections: int | None = None,
    ) -> YoloCoreResult:
        timings: dict[str, float] = {}
        start = time.perf_counter()
        input_tensor, preprocess_meta = _preprocess_bgr(
            image_bgr,
            input_detail=self.input_details[0],
            input_size=self.config.input_size,
            mode=self.config.preprocess,
        )
        timings["preprocess_ms"] = (time.perf_counter() - start) * 1000.0

        start = time.perf_counter()
        self.interpreter.set_tensor(int(self.input_details[0]["index"]), input_tensor)
        self.interpreter.invoke()
        timings["inference_ms"] = (time.perf_counter() - start) * 1000.0

        start = time.perf_counter()
        outputs = [self.interpreter.get_tensor(int(detail["index"])) for detail in self.output_details]
        if not outputs:
            raise RuntimeError("Model produced no output tensors.")
        detections, warnings = decode_yolo_output(
            outputs[0],
            labels=self.labels,
            preprocess_meta=preprocess_meta,
            conf_threshold=self.config.conf_threshold if conf_threshold is None else conf_threshold,
            iou_threshold=self.config.iou_threshold if iou_threshold is None else iou_threshold,
            max_detections=self.config.max_detections if max_detections is None else max_detections,
        )
        timings["postprocess_ms"] = (time.perf_counter() - start) * 1000.0
        timings["total_ms"] = sum(timings.values())

        return YoloCoreResult(
            detections=detections,
            timings={key: round(float(value), 3) for key, value in timings.items()},
            model=self.model_info(),
            preprocess=preprocess_meta,
            raw_output_shapes=[list(np.asarray(out).shape) for out in outputs],
            warnings=warnings,
        )

    def predict_file(
        self,
        image_path: str | Path,
        *,
        conf_threshold: float | None = None,
        iou_threshold: float | None = None,
        max_detections: int | None = None,
    ) -> YoloCoreResult:
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image is None:
            raise FileNotFoundError(f"Could not load image: {image_path}")
        return self.predict_array(
            image,
            conf_threshold=conf_threshold,
            iou_threshold=iou_threshold,
            max_detections=max_detections,
        )

    def save_prediction(
        self,
        image_path: str | Path,
        output_dir: str | Path,
        *,
        conf_threshold: float | None = None,
        iou_threshold: float | None = None,
        max_detections: int | None = None,
    ) -> dict[str, Any]:
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image is None:
            raise FileNotFoundError(f"Could not load image: {image_path}")
        result = self.predict_array(
            image,
            conf_threshold=conf_threshold,
            iou_threshold=iou_threshold,
            max_detections=max_detections,
        )
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        annotated = draw_detections(image, result.detections)
        image_out = output_path / "detected.jpg"
        json_out = output_path / "detections.json"
        cv2.imwrite(str(image_out), annotated)
        payload = result.to_dict()
        payload["artifacts"] = {"annotated_image": str(image_out), "detections_json": str(json_out)}
        json_out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        return payload
