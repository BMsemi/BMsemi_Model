from __future__ import annotations

import argparse
import ast
import csv
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np
import ai_edge_litert.interpreter as Interpreter


DEFAULT_MODEL_CANDIDATES = (
    Path("yolo26n_int8.tflite"),
    Path("yolo26n_int8 .tflite"),
)


def _first_existing(paths: Iterable[Path]) -> Path:
    for path in paths:
        if path.exists():
            return path
    checked = ", ".join(str(path) for path in paths)
    raise FileNotFoundError(f"Could not find a yolo26n_int8 TFLite model. Checked: {checked}")


def _resize_rgb_float(image_path: Path, height: int, width: int) -> np.ndarray:
    bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if bgr is None:
        raise FileNotFoundError(f"Could not load image: {image_path}")
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    return cv2.resize(rgb, (width, height), interpolation=cv2.INTER_LINEAR).astype(np.float32) / 255.0


def _prepare_input(image_path: Path, input_detail: dict) -> np.ndarray:
    shape = tuple(int(v) for v in input_detail["shape"])
    if len(shape) != 4:
        raise ValueError(f"Expected 4D model input, got shape {shape}")

    if shape[1] == 3:
        _, _, height, width = shape
        rgb = _resize_rgb_float(image_path, height, width)
        model_input = np.transpose(rgb[None, ...], (0, 3, 1, 2))
    elif shape[-1] == 3:
        _, height, width, _ = shape
        rgb = _resize_rgb_float(image_path, height, width)
        model_input = rgb[None, ...]
    else:
        raise ValueError(f"Could not infer RGB channel position from input shape {shape}")

    dtype = input_detail["dtype"]
    if dtype == np.float32:
        return np.ascontiguousarray(model_input, dtype=np.float32)

    scale, zero_point = input_detail.get("quantization", (0.0, 0))
    if not scale:
        return np.ascontiguousarray(model_input, dtype=dtype)
    q = np.rint(model_input / float(scale) + int(zero_point))
    if dtype == np.int8:
        q = np.clip(q, -128, 127)
    elif dtype == np.uint8:
        q = np.clip(q, 0, 255)
    return np.ascontiguousarray(q.astype(dtype))


def _find_layer0_tensor(interpreter: Interpreter.Interpreter, requested_index: int | None) -> dict:
    tensor_details = {int(t["index"]): t for t in interpreter.get_tensor_details()}
    if requested_index is not None:
        if requested_index not in tensor_details:
            raise ValueError(f"Tensor index {requested_index} not found in model")
        return tensor_details[requested_index]

    # For the NCHW yolo26n_int8 model, this is op 4 -> tensor 248.
    for op in interpreter._get_ops_details():
        if str(op.get("op_name")) != "MUL":
            continue
        for output_index in op.get("outputs", []):
            detail = tensor_details[int(output_index)]
            if tuple(int(v) for v in detail["shape"]) == (1, 320, 320, 16):
                return detail

    for detail in tensor_details.values():
        if tuple(int(v) for v in detail["shape"]) == (1, 320, 320, 16):
            return detail
    raise ValueError("Could not find a layer-0 tensor with shape (1, 320, 320, 16)")


def _dequantize_if_needed(values: np.ndarray, detail: dict) -> np.ndarray:
    scale, zero_point = detail.get("quantization", (0.0, 0))
    if scale:
        return (values.astype(np.float32) - int(zero_point)) * float(scale)
    return values.astype(np.float32)


def _write_values_csv(path: Path, hwc: np.ndarray) -> None:
    h, w, c = hwc.shape
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["y", "x", *[f"c{i}" for i in range(c)]])
        for y in range(h):
            for x in range(w):
                writer.writerow([y, x, *[repr(float(v)) for v in hwc[y, x]]])


def _write_values_txt(path: Path, hwc: np.ndarray) -> None:
    h, w, c = hwc.shape
    with open(path, "w", encoding="utf-8") as f:
        f.write("Layer0 embedding values\n")
        f.write(f"Shape: H={h}, W={w}, C={c}\n")
        f.write("Format: y x " + " ".join(f"c{i}" for i in range(c)) + "\n")
        f.write("=" * 120 + "\n")
        for y in range(h):
            for x in range(w):
                values = " ".join(repr(float(v)) for v in hwc[y, x])
                f.write(f"{y} {x} {values}\n")


def _write_first_n(path: Path, hwc: np.ndarray, count: int = 100) -> None:
    h, w, c = hwc.shape
    with open(path, "w", encoding="utf-8") as f:
        f.write(f"First {count} layer0 embedding vectors. Format: y x " + " ".join(f"c{i}" for i in range(c)) + "\n")
        written = 0
        for y in range(h):
            for x in range(w):
                values = " ".join(repr(float(v)) for v in hwc[y, x])
                f.write(f"{y} {x} {values}\n")
                written += 1
                if written >= count:
                    return


def _write_summary(path: Path, hwc: np.ndarray) -> None:
    c = hwc.shape[-1]
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["channel", "min", "max", "mean", "std", "top_left_3x3"])
        for channel in range(c):
            ch = hwc[:, :, channel]
            top_left = " ".join(repr(float(v)) for v in ch[:3, :3].reshape(-1))
            writer.writerow(
                [
                    channel,
                    repr(float(ch.min())),
                    repr(float(ch.max())),
                    repr(float(ch.mean())),
                    repr(float(ch.std())),
                    top_left,
                ]
            )


def _write_preview(path: Path, hwc: np.ndarray) -> None:
    with open(path, "w", encoding="utf-8") as f:
        f.write(f"Layer0 embedding tensor saved as HWC float32: {tuple(int(v) for v in hwc.shape)}\n")
        f.write(f"Global min/max/mean/std: {float(hwc.min())}, {float(hwc.max())}, {float(hwc.mean())}, {float(hwc.std())}\n")
        for channel in range(min(4, hwc.shape[-1])):
            f.write(f"\nCHANNEL {channel} top-left 5x5\n")
            for row in hwc[:5, :5, channel]:
                f.write(" ".join(repr(float(v)) for v in row) + "\n")


def _load_reference_txt(path: Path, limit: int | None) -> np.ndarray:
    rows: list[list[float]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.startswith("["):
                continue
            rows.append(ast.literal_eval(line.split(":", 1)[1].strip()))
            if limit is not None and len(rows) >= limit:
                break
    if not rows:
        raise ValueError(f"No tensor rows found in reference file: {path}")
    return np.asarray(rows, dtype=np.float32)


def _compare_reference(hwc: np.ndarray, reference_path: Path, limit: int | None) -> tuple[float, float, float, int]:
    reference = _load_reference_txt(reference_path, limit)
    generated = hwc.reshape(-1, hwc.shape[-1])[: reference.shape[0]]
    diff = generated - reference
    return (
        float(np.mean(diff * diff)),
        float(np.mean(np.abs(diff))),
        float(np.max(np.abs(diff))),
        int(reference.shape[0]),
    )


def extract_layer0(model_path: Path, image_path: Path, tensor_index: int | None) -> tuple[np.ndarray, dict, dict]:
    interpreter = Interpreter.Interpreter(
        model_path=str(model_path),
        experimental_preserve_all_tensors=True,
        experimental_op_resolver_type=Interpreter.OpResolverType.BUILTIN_WITHOUT_DEFAULT_DELEGATES,
    )
    interpreter.allocate_tensors()

    input_detail = interpreter.get_input_details()[0]
    model_input = _prepare_input(image_path, input_detail)
    interpreter.set_tensor(int(input_detail["index"]), model_input)
    interpreter.invoke()

    layer_detail = _find_layer0_tensor(interpreter, tensor_index)
    raw = interpreter.get_tensor(int(layer_detail["index"]))
    layer0 = _dequantize_if_needed(raw, layer_detail)
    if layer0.shape != (1, 320, 320, 16):
        raise ValueError(f"Expected layer0 shape (1, 320, 320, 16), got {layer0.shape}")
    return layer0, input_detail, layer_detail


def save_layer0_artifacts(hwc: np.ndarray, save_dir: Path) -> None:
    save_dir.mkdir(parents=True, exist_ok=True)
    np.save(save_dir / "yolo26n_layer0_embedding.npy", hwc)
    _write_values_csv(save_dir / "yolo26n_layer0_embedding_values.csv", hwc)
    _write_values_txt(save_dir / "yolo26n_layer0_embedding_values.txt", hwc)
    _write_first_n(save_dir / "yolo26n_layer0_embedding_values_first100.txt", hwc)
    _write_summary(save_dir / "yolo26n_layer0_embedding_summary.csv", hwc)
    _write_preview(save_dir / "yolo26n_layer0_embedding_preview.txt", hwc)


def run_litert_layer0_step(
    *,
    model_path: str | Path | None = None,
    image_path: str | Path = "bus.jpg",
    save_dir: str | Path = "layer0_embeddings",
    tensor_index: int | None = 248,
) -> dict:
    selected_model = Path(model_path) if model_path is not None else _first_existing(DEFAULT_MODEL_CANDIDATES)
    layer0, input_detail, layer_detail = extract_layer0(selected_model, Path(image_path), tensor_index)
    hwc = np.ascontiguousarray(layer0[0], dtype=np.float32)
    save_layer0_artifacts(hwc, Path(save_dir))
    return {
        "model_path": selected_model,
        "input_detail": input_detail,
        "layer_detail": layer_detail,
        "layer0": layer0,
        "hwc": hwc,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract YOLO26n LiteRT layer-0 embeddings from yolo26n_int8.tflite.")
    parser.add_argument("--model", default="", help="Path to yolo26n_int8.tflite. Auto-detects the local model when omitted.")
    parser.add_argument("--image", default="bus.jpg", help="Input image path.")
    parser.add_argument("--tensor-index", type=int, default=None, help="Layer-0 tensor index. Defaults to first (1,320,320,16) MUL output.")
    parser.add_argument("--save-dir", default="layer0_embeddings", help="Output directory.")
    parser.add_argument("--reference", default="", help="Optional layer0_litert.txt reference file for comparison.")
    parser.add_argument("--compare-limit", type=int, default=100, help="Number of flattened vectors to compare; use 0 for all.")
    args = parser.parse_args()

    model_path = Path(args.model) if args.model else _first_existing(DEFAULT_MODEL_CANDIDATES)
    image_path = Path(args.image)
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    result = run_litert_layer0_step(
        model_path=model_path,
        image_path=image_path,
        save_dir=save_dir,
        tensor_index=args.tensor_index,
    )
    layer0 = result["layer0"]
    hwc = result["hwc"]
    input_detail = result["input_detail"]
    layer_detail = result["layer_detail"]

    print(f"Model: {model_path}")
    print(f"Input: name={input_detail['name']}, shape={tuple(int(v) for v in input_detail['shape'])}, dtype={input_detail['dtype']}")
    print(f"Layer0 tensor: index={layer_detail['index']}, name={layer_detail['name']}")
    print(f"Layer0 shape: {tuple(int(v) for v in layer0.shape)}")
    print(f"Layer0 min/max/mean/std: {float(hwc.min())}, {float(hwc.max())}, {float(hwc.mean())}, {float(hwc.std())}")
    print(f"Saved outputs to: {save_dir.resolve()}")

    if args.reference:
        limit = None if args.compare_limit == 0 else max(1, args.compare_limit)
        mse, mae, max_abs, compared = _compare_reference(hwc, Path(args.reference), limit)
        print(f"Reference comparison ({compared} vectors): mse={mse}, mae={mae}, max_abs={max_abs}")


if __name__ == "__main__":
    main()
