from __future__ import annotations

import argparse
import ast
import csv
import re
from pathlib import Path

import numpy as np


DEFAULT_INDEX = Path(r"C:\Users\Avadh\Downloads\qwen_extracted_layer_weights\index.csv")
DEFAULT_OUT_DIR = Path(r"C:\Users\Avadh\Downloads\qwen_weight_text")


def parse_layers(spec: str) -> set[int] | None:
    spec = spec.strip().lower()
    if spec in {"", "all"}:
        return None
    layers: set[int] = set()
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            start_s, end_s = part.split("-", 1)
            start, end = int(start_s), int(end_s)
            if end < start:
                raise ValueError(f"Invalid layer range: {part}")
            layers.update(range(start, end + 1))
        else:
            layers.add(int(part))
    return layers


def parse_roles(spec: str) -> set[str] | None:
    spec = spec.strip()
    if spec.lower() in {"", "all"}:
        return None
    return {part.strip() for part in spec.split(",") if part.strip()}


def safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "__", value).replace(".", "__")


def load_index(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def load_tensor(row: dict[str, str]) -> np.ndarray:
    path = Path(row["output_path"])
    shape = ast.literal_eval(row["shape"])
    export_format = row["export_format"]
    dtype = row["dtype"].upper()

    if export_format == "npy-fp32":
        return np.load(path).astype(np.float32, copy=False)

    if export_format == "raw-bf16":
        if dtype != "BF16":
            raise ValueError(f"raw-bf16 loader only supports BF16 rows, got {dtype} for {row['tensor_name']}")
        raw = np.fromfile(path, dtype="<u2")
        expected = int(np.prod(shape))
        if raw.size != expected:
            raise ValueError(f"{path} has {raw.size} BF16 values, but shape {shape} expects {expected}.")
        return (raw.astype(np.uint32) << 16).view("<f4").reshape(shape)

    raise ValueError(f"Unsupported export_format {export_format!r} for {row['tensor_name']}")


def maybe_crossbar_layout(tensor: np.ndarray, layout: str) -> np.ndarray:
    if layout == "hf":
        return tensor
    if layout == "crossbar" and tensor.ndim == 2:
        return tensor.T
    return tensor


def stats_text(tensor: np.ndarray) -> list[str]:
    flat = tensor.astype(np.float64, copy=False).reshape(-1)
    return [
        f"min: {float(np.min(flat)):.9g}",
        f"max: {float(np.max(flat)):.9g}",
        f"mean: {float(np.mean(flat)):.9g}",
        f"std: {float(np.std(flat)):.9g}",
    ]


def write_tensor_text(
    row: dict[str, str],
    tensor: np.ndarray,
    output_path: Path,
    *,
    layout: str,
    rows: int,
    cols: int,
    full: bool,
    precision: int,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sample = tensor
    if not full:
        if sample.ndim == 1:
            sample = sample[:cols]
        elif sample.ndim == 2:
            sample = sample[:rows, :cols]
        else:
            slices = tuple(slice(0, min(dim, rows if idx == 0 else cols)) for idx, dim in enumerate(sample.shape))
            sample = sample[slices]

    with output_path.open("w", encoding="utf-8") as handle:
        handle.write(f"tensor_name: {row['tensor_name']}\n")
        handle.write(f"source_file: {row['output_path']}\n")
        handle.write(f"role: {row['role']}\n")
        handle.write(f"layer: {row['layer']}\n")
        handle.write(f"source_shape_hf: {row['shape']}\n")
        handle.write(f"written_shape: {list(tensor.shape)}\n")
        handle.write(f"layout: {layout}\n")
        handle.write(f"dtype_loaded: fp32_from_{row['dtype'].lower()}\n")
        handle.write(f"values_written: {'full' if full else f'first {rows} rows x {cols} cols'}\n")
        handle.write("\n")
        handle.write("stats:\n")
        for line in stats_text(tensor):
            handle.write(f"  {line}\n")
        handle.write("\n")
        handle.write("values:\n")

        if full:
            data = tensor.reshape(1, -1) if tensor.ndim == 1 else tensor.reshape(tensor.shape[0], -1)
            np.savetxt(handle, data, fmt=f"%.{precision}g", delimiter=",")
        else:
            handle.write(np.array2string(sample, precision=precision, separator=", ", threshold=sample.size + 1))
            handle.write("\n")


def selected(row: dict[str, str], layers: set[int] | None, roles: set[str] | None) -> bool:
    if row.get("layer", "") == "":
        return False
    layer = int(row["layer"])
    if layers is not None and layer not in layers:
        return False
    if roles is not None and row["role"] not in roles:
        return False
    return True


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export extracted Qwen BF16/NPY weights to readable per-layer text files."
    )
    parser.add_argument("--index", type=Path, default=DEFAULT_INDEX)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--layers", default="all", help="Layer list/ranges, for example 0,1,5-7, or all.")
    parser.add_argument(
        "--roles",
        default="all",
        help="Comma-separated roles from index.csv, for example self_attn.q_proj,mlp.down_proj, or all.",
    )
    parser.add_argument(
        "--layout",
        choices=("hf", "crossbar", "both"),
        default="hf",
        help="hf keeps [out,in] checkpoint layout. crossbar transposes 2D tensors to [in,out].",
    )
    parser.add_argument("--rows", type=int, default=8, help="Rows to print per 2D tensor when not using --full.")
    parser.add_argument("--cols", type=int, default=8, help="Columns/1D values to print when not using --full.")
    parser.add_argument("--precision", type=int, default=8)
    parser.add_argument("--full", action="store_true", help="Write the full tensor values, not just a preview slice.")
    args = parser.parse_args()

    rows = load_index(args.index)
    layers = parse_layers(args.layers)
    roles = parse_roles(args.roles)
    filtered = [row for row in rows if selected(row, layers, roles)]
    if not filtered:
        raise SystemExit("No tensors matched the requested layers/roles.")

    layouts = ("hf", "crossbar") if args.layout == "both" else (args.layout,)
    written: list[Path] = []
    for row in filtered:
        tensor_hf = load_tensor(row)
        for layout in layouts:
            tensor = maybe_crossbar_layout(tensor_hf, layout)
            layer_dir = args.out_dir / f"layer_{int(row['layer']):02d}"
            suffix = f".{layout}.txt" if args.layout == "both" else ".txt"
            out_path = layer_dir / f"{safe_name(row['role'])}{suffix}"
            write_tensor_text(
                row,
                tensor,
                out_path,
                layout=layout,
                rows=max(1, args.rows),
                cols=max(1, args.cols),
                full=bool(args.full),
                precision=max(1, args.precision),
            )
            written.append(out_path)
            print(f"wrote {out_path}")

    summary_path = args.out_dir / "export_index.txt"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text("\n".join(str(path) for path in written) + "\n", encoding="utf-8")
    print(f"\nDone. Wrote {len(written)} files.")
    print(f"Index: {summary_path}")


if __name__ == "__main__":
    main()
