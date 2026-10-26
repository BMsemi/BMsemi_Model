"""YOLO26n INT8 detection as an operation (Yolo26_INT8_mac).

Runs yolo26n_int8.tflite on the CPU (the "reference" path of the project) with
the project's own pre/post-processing: plain-resize 640x640 RGB /255 NCHW,
decode (cx,cy,w,h + 80 class scores), scale to the original image, clamp,
cv2.dnn.NMSBoxes.  The crossbar/PE replay (hundreds of layers, minutes per
image) is NOT run, so scores differ slightly from the saved hardware-sim run
(e.g. bus.jpg: same 5 objects, scores within ~0.04, boxes within ~5 px).

Needs: ai-edge-litert (or tflite-runtime / tensorflow), opencv-python-headless.
Model: NEURO_YOLO_DIR (default projects/3_example_models/Yolo26_INT8_mac)/yolo26n_int8.tflite

params:
  image_base64     PNG/JPEG, <= 5 MB                          (required)
  conf, iou        thresholds (defaults 0.25 / 0.45)
  max_detections   1..300 (default 100)
  return_image     true -> also return the input image with labelled boxes drawn
                   (annotated_image_base64, JPEG)
Direct file upload (no base64) is POST /v1/detect/yolo26 in api/server.py.
"""
from __future__ import annotations

import base64
import binascii
import os
import threading
from pathlib import Path

from ..registry import OperationUnavailable

DEFAULT_DIR = Path(__file__).resolve().parents[2] / "projects" / "3_example_models" / "Yolo26_INT8_mac"
MAX_IMAGE_BYTES = 5_000_000
MAX_PIXELS = 40_000_000
COCO80 = [
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
_lock = threading.Lock()          # TFLite interpreters are not thread-safe
_state: dict = {}


def _runtime():
    try:
        import cv2
        import numpy as np
    except ImportError as e:
        raise OperationUnavailable(f"YOLO needs numpy and opencv-python-headless ({e})")
    if "interp" not in _state:
        Interp = None
        for mod in ("ai_edge_litert.interpreter", "tflite_runtime.interpreter", "tensorflow.lite"):
            try:
                Interp = __import__(mod, fromlist=["Interpreter"]).Interpreter
                break
            except (ImportError, AttributeError):
                continue
        if Interp is None:
            raise OperationUnavailable("YOLO needs ai-edge-litert (pip install ai-edge-litert) or tflite-runtime")
        model = DEFAULT_DIR / "yolo26n_int8.tflite"
        if not model.is_file():
            raise OperationUnavailable(f"model file not found: {model.name} (set NEURO_YOLO_DIR)")
        it = Interp(model_path=str(model))
        it.allocate_tensors()
        _state.update(interp=it, inp=it.get_input_details()[0], out=it.get_output_details()[0])
    return cv2, np


def detect(params: dict) -> dict:
    b64 = params.get("image_base64")
    if not isinstance(b64, str) or not b64:
        raise ValueError("image_base64 is required")
    try:
        raw = base64.b64decode(b64, validate=True)
    except (binascii.Error, ValueError):
        raise ValueError("image_base64 is not valid base64")
    return detect_bytes(raw, params)


_COLOURS = [(56, 56, 255), (255, 157, 151), (31, 112, 255), (29, 178, 255), (49, 210, 207), (10, 249, 72),
            (23, 204, 146), (134, 219, 61), (52, 147, 26), (187, 212, 0), (168, 153, 44), (255, 194, 0),
            (147, 69, 52), (255, 115, 100), (236, 24, 0), (255, 56, 132), (133, 0, 82), (255, 56, 203)]


def _palette(c: int) -> tuple:
    # bright, stable BGR colour per class (white label text stays readable)
    return _COLOURS[c % len(_COLOURS)]


def annotate(bgr, dets):
    """Draw square boxes + 'label 87%' tags on a copy of the BGR image."""
    import cv2
    img = bgr.copy()
    H, W = img.shape[:2]
    lw = max(2, round(max(H, W) / 400))
    scale = max(0.5, max(H, W) / 1300)
    for d in dets:
        x1, y1, x2, y2 = d["box_xyxy"]
        col = _palette(d["class_id"])
        cv2.rectangle(img, (x1, y1), (x2, y2), col, lw)
        tag = f"{d['label']} {d['confidence'] * 100:.0f}%"
        (tw, th), base = cv2.getTextSize(tag, cv2.FONT_HERSHEY_SIMPLEX, scale, max(1, lw // 2))
        ty = y1 - th - base - 4 if y1 - th - base - 4 >= 0 else y1
        cv2.rectangle(img, (x1, ty), (x1 + tw + 6, ty + th + base + 4), col, -1)
        cv2.putText(img, tag, (x1 + 3, ty + th + 2), cv2.FONT_HERSHEY_SIMPLEX, scale, (255, 255, 255),
                    max(1, lw // 2), cv2.LINE_AA)
    return img


def encode_jpeg(img, quality: int = 90) -> bytes:
    import cv2
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not ok:
        raise ValueError("could not encode output image")
    return buf.tobytes()


def detect_bytes(raw: bytes, params: dict, *, want_image_bytes: bool = False):
    """Run detection on raw PNG/JPEG bytes.  With want_image_bytes=True returns
    (result_dict, annotated_jpeg_bytes) for the upload endpoint."""
    cv2, np = _runtime()
    if not 0 < len(raw) <= MAX_IMAGE_BYTES:
        raise ValueError(f"image must be 1..{MAX_IMAGE_BYTES} bytes")
    conf, iou = float(params.get("conf", 0.25)), float(params.get("iou", 0.45))
    maxd = int(params.get("max_detections", 100))
    if not (0 < conf <= 1 and 0 < iou <= 1 and 1 <= maxd <= 300):
        raise ValueError("conf and iou must be in (0,1], max_detections in 1..300")

    bgr = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)
    if bgr is None:
        raise ValueError("could not decode image")
    H, W = bgr.shape[:2]
    if H * W > MAX_PIXELS:
        raise ValueError("image too large")

    rgb = cv2.resize(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB), (640, 640), interpolation=cv2.INTER_LINEAR)
    x = np.ascontiguousarray(np.transpose(rgb.astype(np.float32)[None] / 255.0, (0, 3, 1, 2)))
    with _lock:
        it = _state["interp"]
        it.set_tensor(_state["inp"]["index"], x)
        it.invoke()
        out = np.array(it.get_tensor(_state["out"]["index"]), dtype=np.float32)

    pred = out[0]
    if pred.shape[0] == 84:
        pred = pred.T
    scores = pred[:, 4:]
    cids = scores.argmax(1)
    confs = scores[np.arange(len(scores)), cids]
    boxes, kept = [], []
    sx, sy = W / 640.0, H / 640.0
    for k in np.where(confs >= conf)[0]:
        cx, cy, w, h = (float(v) for v in pred[k, :4])
        x1, x2 = (max(0, min(W - 1, int(round(v * sx)))) for v in (cx - w / 2, cx + w / 2))
        y1, y2 = (max(0, min(H - 1, int(round(v * sy)))) for v in (cy - h / 2, cy + h / 2))
        if x2 - x1 <= 1 or y2 - y1 <= 1:
            continue
        boxes.append([x1, y1, x2 - x1, y2 - y1])
        kept.append(int(k))
    sel = []
    if boxes:
        n = cv2.dnn.NMSBoxes(boxes, [float(confs[k]) for k in kept], conf, iou)
        sel = np.array(n).reshape(-1).tolist() if len(n) else []
    dets = []
    for rank, j in enumerate(sel[:maxd]):
        k, (bx, by, bw, bh) = kept[j], boxes[j]
        c = int(cids[k])
        dets.append({"rank": rank, "class_id": c, "label": COCO80[c] if c < 80 else f"class_{c}",
                     "confidence": float(confs[k]), "box_xyxy": [bx, by, bx + bw, by + bh]})
    result = {"image_size": [W, H], "detections": dets, "count": len(dets),
              "engine": "TFLite CPU reference (no crossbar replay)", "preprocess": "plain-resize 640x640"}
    if want_image_bytes:
        return result, encode_jpeg(annotate(bgr, dets))
    rv = params.get("return_image", False)
    if rv is True or (isinstance(rv, str) and rv.lower() in ("1", "true", "yes")):
        result["annotated_image_format"] = "jpeg"
        result["annotated_image_base64"] = base64.b64encode(encode_jpeg(annotate(bgr, dets))).decode()
    return result
