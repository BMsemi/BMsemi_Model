import argparse
import csv
import json
import math
import re
import shutil
import struct
from pathlib import Path
from typing import Any


DEFAULT_MODEL = Path(r"C:\Users\Avadh\Downloads\model.safetensors")
DEFAULT_OUT_DIR = Path(r"C:\Users\Avadh\Downloads\qwen_extracted_layer_weights")
CHUNK_BYTES = 64 * 1024 * 1024


PE_WEIGHT_RE = re.compile(
    r"^model\.layers\.(\d+)\."
    r"("
    r"self_attn\.(q_proj|k_proj|v_proj|o_proj)"
    r"|mlp\.(gate_proj|up_proj|down_proj)"
    r")\.weight$"
)
LAYER_WEIGHT_RE = re.compile(r"^model\.layers\.(\d+)\.(.+)\.weight$")
ANY_LAYER_RE = re.compile(r"^model\.layers\.(\d+)\.(.+)$")


DTYPE_BYTES = {
    "BF16": 2,
    "F16": 2,
    "F32": 4,
    "F64": 8,
    "I8": 1,
    "I16": 2,
    "I32": 4,
    "I64": 8,
    "U8": 1,
    "U16": 2,
    "U32": 4,
    "U64": 8,
    "BOOL": 1,
}


def read_safetensors_header(path: Path) -> tuple[dict[str, Any], int]:
    with path.open("rb") as handle:
        header_size = struct.unpack("<Q", handle.read(8))[0]
        header = json.loads(handle.read(header_size))
    return header, 8 + header_size


def product(values: list[int]) -> int:
    out = 1
    for value in values:
        out *= int(value)
    return out


def layer_and_role(name: str) -> tuple[int | None, str]:
    match = ANY_LAYER_RE.match(name)
    if not match:
        return None, name.replace("model.", "")
    return int(match.group(1)), match.group(2).replace(".weight", "")


def selected_tensor(name: str, selection: str) -> bool:
    if name == "__metadata__":
        return False
    if selection == "pe":
        return PE_WEIGHT_RE.match(name) is not None
    if selection == "all-layer":
        return LAYER_WEIGHT_RE.match(name) is not None
    if selection == "all":
        return True
    raise ValueError(f"Unknown selection {selection!r}")


def safe_file_stem(name: str) -> str:
    layer, role = layer_and_role(name)
    if layer is None:
        stem = name.replace("model.", "")
    else:
        stem = f"layer_{layer:02d}__{role}"
    return re.sub(r"[^A-Za-z0-9_.-]+", "__", stem).replace(".", "__")


def copy_raw_tensor(
    model_path: Path,
    data_start: int,
    offsets: list[int],
    output_path: Path,
) -> int:
    start, end = int(offsets[0]), int(offsets[1])
    remaining = end - start
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with model_path.open("rb") as src, output_path.open("wb") as dst:
        src.seek(data_start + start)
        while remaining > 0:
            chunk = src.read(min(CHUNK_BYTES, remaining))
            if not chunk:
                raise RuntimeError(f"Unexpected EOF while reading {model_path}")
            dst.write(chunk)
            remaining -= len(chunk)
    return end - start


def write_npy_fp32_tensor(
    model_path: Path,
    data_start: int,
    offsets: list[int],
    shape: list[int],
    dtype: str,
    output_path: Path,
) -> int:
    if dtype != "BF16":
        raise ValueError(f"npy-fp32 export currently supports BF16 tensors only, got {dtype}")

    try:
        import numpy as np
    except Exception as exc:
        raise RuntimeError("npy-fp32 export requires numpy.") from exc

    start, end = int(offsets[0]), int(offsets[1])
    nbytes = end - start
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with model_path.open("rb") as src:
        src.seek(data_start + start)
        raw = src.read(nbytes)
    if len(raw) != nbytes:
        raise RuntimeError(f"Unexpected EOF while reading {model_path}")

    bf16 = np.frombuffer(raw, dtype="<u2").astype(np.uint32)
    fp32 = (bf16 << 16).view("<f4").reshape(shape)
    np.save(output_path, fp32)
    return int(output_path.stat().st_size)


def tensor_output_path(out_dir: Path, name: str, export_format: str) -> Path:
    layer, _ = layer_and_role(name)
    parent = out_dir / ("non_layer" if layer is None else f"layer_{layer:02d}")
    suffix = ".npy" if export_format == "npy-fp32" else ".bf16.bin"
    return parent / f"{safe_file_stem(name)}{suffix}"


def write_index(out_dir: Path, rows: list[dict[str, Any]]) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "index.json"
    csv_path = out_dir / "index.csv"
    readme_path = out_dir / "README.txt"

    json_path.write_text(json.dumps(rows, indent=2) + "\n", encoding="utf-8")

    fieldnames = [
        "tensor_name",
        "layer",
        "role",
        "shape",
        "dtype",
        "parameter_count",
        "source_bytes",
        "export_format",
        "output_path",
        "output_bytes",
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})

    readme_path.write_text(
        "Qwen extracted layer weights\n"
        "============================\n\n"
        "index.csv and index.json map each tensor name to its extracted file.\n\n"
        "raw-bf16 format:\n"
        "- Exact tensor bytes copied from model.safetensors.\n"
        "- Values are BF16 stored as little-endian uint16 words.\n"
        "- Use the shape and dtype from index.csv/index.json to interpret each file.\n\n"
        "npy-fp32 format:\n"
        "- BF16 values converted to float32 and saved as NumPy .npy arrays.\n"
        "- Easier to load, but about 2x larger than raw BF16.\n",
        encoding="utf-8",
    )


def build_rows(
    header: dict[str, Any],
    selection: str,
    export_format: str,
    out_dir: Path,
) -> list[dict[str, Any]]:
    rows = []
    for name, info in header.items():
        if not selected_tensor(name, selection):
            continue
        shape = [int(value) for value in info.get("shape", [])]
        dtype = str(info.get("dtype", "UNKNOWN"))
        layer, role = layer_and_role(name)
        param_count = product(shape) if shape else 1
        dtype_bytes = DTYPE_BYTES.get(dtype)
        source_bytes = param_count * dtype_bytes if dtype_bytes is not None else None
        rows.append(
            {
                "tensor_name": name,
                "layer": "" if layer is None else layer,
                "role": role,
                "shape": shape,
                "dtype": dtype,
                "parameter_count": param_count,
                "source_bytes": source_bytes,
                "data_offsets": info.get("data_offsets"),
                "export_format": export_format,
                "output_path": str(tensor_output_path(out_dir, name, export_format)),
                "output_bytes": 0,
            }
        )
    return rows


def human_gib(num_bytes: int | None) -> str:
    if num_bytes is None:
        return "unknown"
    return f"{num_bytes / (1024.0 ** 3):.3f} GiB"


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract Qwen layer weights from a safetensors checkpoint.")
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument(
        "--selection",
        choices=("all-layer", "pe", "all"),
        default="all-layer",
        help="all-layer extracts every model.layers.* learned tensor. pe extracts only crossbar-mappable projections.",
    )
    parser.add_argument(
        "--format",
        choices=("raw-bf16", "npy-fp32"),
        default="raw-bf16",
        help="raw-bf16 is exact and compact. npy-fp32 is easier to load but about 2x larger.",
    )
    parser.add_argument("--dry-run", action="store_true", help="Only print what would be extracted.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite files in the output directory.")
    args = parser.parse_args()

    if not args.model.exists():
        raise FileNotFoundError(args.model)

    header, data_start = read_safetensors_header(args.model)
    rows = build_rows(header, args.selection, args.format, args.out_dir)
    total_source_bytes = sum(int(row["source_bytes"] or 0) for row in rows)
    expected_output_bytes = total_source_bytes if args.format == "raw-bf16" else total_source_bytes * 2

    print(f"Model: {args.model}")
    print(f"Output directory: {args.out_dir}")
    print(f"Selection: {args.selection}")
    print(f"Format: {args.format}")
    print(f"Tensors selected: {len(rows)}")
    print(f"Source tensor bytes: {human_gib(total_source_bytes)}")
    print(f"Expected output size: {human_gib(expected_output_bytes)}")

    free_bytes = shutil.disk_usage(args.out_dir.parent if args.out_dir.parent.exists() else Path.cwd()).free
    if not args.dry_run and free_bytes < expected_output_bytes:
        raise RuntimeError(
            f"Not enough free disk space. Need about {human_gib(expected_output_bytes)}, "
            f"available {human_gib(free_bytes)}."
        )

    if args.dry_run:
        for row in rows[:20]:
            print(f"{row['tensor_name']} -> {row['output_path']}")
        if len(rows) > 20:
            print(f"... {len(rows) - 20} more tensors")
        return

    args.out_dir.mkdir(parents=True, exist_ok=True)
    for idx, row in enumerate(rows, start=1):
        output_path = Path(row["output_path"])
        if output_path.exists() and not args.overwrite:
            row["output_bytes"] = output_path.stat().st_size
            print(f"[{idx}/{len(rows)}] exists: {row['tensor_name']}")
            continue

        if args.format == "raw-bf16":
            written = copy_raw_tensor(args.model, data_start, row["data_offsets"], output_path)
        else:
            written = write_npy_fp32_tensor(
                args.model,
                data_start,
                row["data_offsets"],
                row["shape"],
                row["dtype"],
                output_path,
            )
        row["output_bytes"] = written
        print(f"[{idx}/{len(rows)}] wrote {row['tensor_name']} -> {output_path}")

    write_index(args.out_dir, rows)
    print(f"\nDone. Index files:")
    print(args.out_dir / "index.csv")
    print(args.out_dir / "index.json")


if __name__ == "__main__":
    main()
