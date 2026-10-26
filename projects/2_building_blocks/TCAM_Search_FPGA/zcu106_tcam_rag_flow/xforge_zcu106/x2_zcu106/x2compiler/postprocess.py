from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Any

import numpy as np


COCO80_CLASSES = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train",
    "truck", "boat", "traffic light", "fire hydrant", "stop sign",
    "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep",
    "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
    "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard",
    "sports ball", "kite", "baseball bat", "baseball glove", "skateboard",
    "surfboard", "tennis racket", "bottle", "wine glass", "cup", "fork",
    "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange",
    "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair",
    "couch", "potted plant", "bed", "dining table", "toilet", "tv",
    "laptop", "mouse", "remote", "keyboard", "cell phone", "microwave",
    "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase",
    "scissors", "teddy bear", "hair drier", "toothbrush",
]


@dataclass(frozen=True)
class Detection:
    rank: int
    class_id: int
    label: str
    confidence: float
    box_xyxy: list[int]
    box_xywh: list[int]


def _nms(
    boxes: np.ndarray,
    scores: np.ndarray,
    threshold: float,
) -> list[int]:
    if boxes.size == 0:
        return []
    x1, y1, x2, y2 = boxes.T
    area = np.maximum(0.0, x2 - x1) * np.maximum(0.0, y2 - y1)
    order = scores.argsort()[::-1]
    keep: list[int] = []
    while order.size:
        selected = int(order[0])
        keep.append(selected)
        if order.size == 1:
            break
        remaining = order[1:]
        intersection = (
            np.maximum(
                0.0,
                np.minimum(x2[selected], x2[remaining])
                - np.maximum(x1[selected], x1[remaining]),
            )
            * np.maximum(
                0.0,
                np.minimum(y2[selected], y2[remaining])
                - np.maximum(y1[selected], y1[remaining]),
            )
        )
        union = area[selected] + area[remaining] - intersection
        iou = intersection / np.maximum(union, 1e-9)
        order = remaining[iou <= threshold]
    return keep


def decode_yolo(
    output: np.ndarray,
    preprocess_metadata: dict[str, Any],
    *,
    confidence_threshold: float = 0.25,
    iou_threshold: float = 0.45,
    max_detections: int = 100,
    labels: list[str] | None = None,
) -> list[Detection]:
    labels = labels or COCO80_CLASSES
    predictions = np.asarray(output, dtype=np.float32)
    if predictions.ndim == 3 and predictions.shape[0] == 1:
        predictions = predictions[0]
    if predictions.ndim != 2:
        raise ValueError(f"Expected rank-two YOLO predictions, got {predictions.shape}.")
    if predictions.shape[0] in {len(labels) + 4, len(labels) + 5}:
        predictions = predictions.T
    if predictions.shape[1] not in {len(labels) + 4, len(labels) + 5}:
        raise ValueError(
            f"YOLO prediction width {predictions.shape[1]} does not match "
            f"{len(labels)} classes."
        )

    boxes_xywh = predictions[:, :4].copy()
    new_h, new_w = [
        int(value) for value in preprocess_metadata["new_shape"]
    ]
    if boxes_xywh.size and float(np.max(np.abs(boxes_xywh))) <= 2.0:
        boxes_xywh[:, [0, 2]] *= new_w
        boxes_xywh[:, [1, 3]] *= new_h
    score_columns = predictions[:, 4:]
    if score_columns.shape[1] == len(labels) + 1:
        scores_by_class = score_columns[:, 1:] * score_columns[:, :1]
    else:
        scores_by_class = score_columns
    class_ids = np.argmax(scores_by_class, axis=1)
    scores = scores_by_class[np.arange(predictions.shape[0]), class_ids]
    selected = np.flatnonzero(scores >= confidence_threshold)
    if selected.size == 0:
        return []

    boxes_xywh = boxes_xywh[selected]
    scores = scores[selected]
    class_ids = class_ids[selected]
    boxes = np.column_stack(
        (
            boxes_xywh[:, 0] - boxes_xywh[:, 2] / 2.0,
            boxes_xywh[:, 1] - boxes_xywh[:, 3] / 2.0,
            boxes_xywh[:, 0] + boxes_xywh[:, 2] / 2.0,
            boxes_xywh[:, 1] + boxes_xywh[:, 3] / 2.0,
        )
    )

    original_h, original_w = [
        int(value) for value in preprocess_metadata["original_shape"]
    ]
    gain = preprocess_metadata.get("gain")
    if gain:
        boxes[:, [0, 2]] = (
            boxes[:, [0, 2]] - float(preprocess_metadata["pad_x"])
        ) / float(gain)
        boxes[:, [1, 3]] = (
            boxes[:, [1, 3]] - float(preprocess_metadata["pad_y"])
        ) / float(gain)
    else:
        boxes[:, [0, 2]] *= original_w / float(new_w)
        boxes[:, [1, 3]] *= original_h / float(new_h)
    boxes[:, [0, 2]] = np.clip(boxes[:, [0, 2]], 0, original_w - 1)
    boxes[:, [1, 3]] = np.clip(boxes[:, [1, 3]], 0, original_h - 1)

    keep: list[int] = []
    for class_id in np.unique(class_ids):
        members = np.flatnonzero(class_ids == class_id)
        keep.extend(
            int(members[index])
            for index in _nms(boxes[members], scores[members], iou_threshold)
        )
    keep.sort(key=lambda index: float(scores[index]), reverse=True)
    detections: list[Detection] = []
    for rank, index in enumerate(keep[:max_detections]):
        x1, y1, x2, y2 = [int(round(value)) for value in boxes[index]]
        if x2 <= x1 or y2 <= y1:
            continue
        class_id = int(class_ids[index])
        detections.append(
            Detection(
                rank=rank,
                class_id=class_id,
                label=(
                    labels[class_id]
                    if class_id < len(labels)
                    else f"class_{class_id}"
                ),
                confidence=float(scores[index]),
                box_xyxy=[x1, y1, x2, y2],
                box_xywh=[x1, y1, x2 - x1, y2 - y1],
            )
        )
    return detections


def compare_detections(
    candidate: list[Detection],
    reference: list[Detection],
    *,
    match_iou: float = 0.5,
) -> dict[str, Any]:
    def iou(lhs: Detection, rhs: Detection) -> float:
        lx1, ly1, lx2, ly2 = lhs.box_xyxy
        rx1, ry1, rx2, ry2 = rhs.box_xyxy
        intersection = max(0, min(lx2, rx2) - max(lx1, rx1)) * max(
            0,
            min(ly2, ry2) - max(ly1, ry1),
        )
        lhs_area = max(0, lx2 - lx1) * max(0, ly2 - ly1)
        rhs_area = max(0, rx2 - rx1) * max(0, ry2 - ry1)
        return intersection / max(lhs_area + rhs_area - intersection, 1e-9)

    candidates: list[tuple[float, int, int]] = []
    for candidate_index, candidate_detection in enumerate(candidate):
        for reference_index, reference_detection in enumerate(reference):
            if candidate_detection.class_id == reference_detection.class_id:
                candidates.append(
                    (
                        iou(candidate_detection, reference_detection),
                        candidate_index,
                        reference_index,
                    )
                )
    matched_candidate: set[int] = set()
    matched_reference: set[int] = set()
    matches: list[tuple[float, int, int]] = []
    for overlap, candidate_index, reference_index in sorted(
        candidates,
        reverse=True,
    ):
        if overlap < match_iou:
            break
        if (
            candidate_index not in matched_candidate
            and reference_index not in matched_reference
        ):
            matched_candidate.add(candidate_index)
            matched_reference.add(reference_index)
            matches.append((overlap, candidate_index, reference_index))
    confidence_errors = [
        abs(
            candidate[candidate_index].confidence
            - reference[reference_index].confidence
        )
        for _, candidate_index, reference_index in matches
    ]
    return {
        "candidate_count": len(candidate),
        "reference_count": len(reference),
        "matched_count": len(matches),
        "precision": len(matches) / max(len(candidate), 1),
        "recall": len(matches) / max(len(reference), 1),
        "mean_iou": (
            float(np.mean([item[0] for item in matches]))
            if matches
            else 0.0
        ),
        "mean_confidence_abs_error": (
            float(np.mean(confidence_errors)) if confidence_errors else 0.0
        ),
        "match_iou_threshold": match_iou,
    }


def save_detection_artifacts(
    directory: str | Path,
    original_bgr: np.ndarray,
    detections: list[Detection],
    *,
    metadata: dict[str, Any] | None = None,
) -> dict[str, str]:
    import cv2

    destination = Path(directory)
    destination.mkdir(parents=True, exist_ok=True)
    annotated = np.array(original_bgr, copy=True)
    for detection in detections:
        x1, y1, x2, y2 = detection.box_xyxy
        caption = f"{detection.label} {detection.confidence:.2f}"
        cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 180, 255), 2)
        cv2.putText(
            annotated,
            caption,
            (x1, max(y1 - 6, 16)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 180, 255),
            2,
            cv2.LINE_AA,
        )
    image_path = destination / "detections.jpg"
    json_path = destination / "detections.json"
    if not cv2.imwrite(str(image_path), annotated):
        raise RuntimeError(f"Could not write {image_path}.")
    json_path.write_text(
        json.dumps(
            {
                "detections": [asdict(item) for item in detections],
                "metadata": metadata or {},
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return {
        "annotated_image": str(image_path.resolve()),
        "detections_json": str(json_path.resolve()),
    }
