from __future__ import annotations

import argparse
import json
import math
import struct
from pathlib import Path
from typing import Any


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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="View exact BF16 Q/K/V values from a QKV_PREFILL safetensors artifact."
    )
    parser.add_argument(
        "artifact",
        type=Path,
        help="Path to QKV_PREFILL_*.safetensors.",
    )
    parser.add_argument("--list", action="store_true", help="List tensors and exit.")
    parser.add_argument("--export-all-qkv", action="store_true", help="Export all layer_XX_q/k/v BF16 values.")
    parser.add_argument("--out", type=Path, help="Write output to this text file instead of stdout.")
    parser.add_argument("--tensor", help="Tensor name, e.g. layer_00_q.")
    parser.add_argument("--layer", type=int, choices=(0, 1, 2), help="Layer index for Q/K/V tensor selection.")
    parser.add_argument("--projection", choices=("q", "k", "v"), help="Projection for Q/K/V tensor selection.")
    parser.add_argument("--batch", type=int, default=0, help="Batch index for row view.")
    parser.add_argument("--token", type=int, default=0, help="Token index for row view.")
    parser.add_argument("--channel-start", type=int, default=0, help="First channel index for row view.")
    parser.add_argument("--limit", type=int, default=16, help="Number of values to print.")
    parser.add_argument("--flat-start", type=int, default=None, help="Flat element offset to view instead of token/channel row.")
    parser.add_argument(
        "--index",
        action="append",
        default=[],
        help="Exact index as comma-separated coordinates, e.g. 0,12,31. Can be repeated.",
    )
    return parser.parse_args()


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


def bf16_to_float32(word: int) -> float:
    return struct.unpack("<f", struct.pack("<I", int(word) << 16))[0]


def unravel_index(flat_index: int, shape: list[int]) -> tuple[int, ...]:
    coords = []
    remaining = int(flat_index)
    for dim in reversed(shape):
        coords.append(remaining % dim)
        remaining //= dim
    return tuple(reversed(coords))


def ravel_index(coords: list[int], shape: list[int]) -> int:
    if len(coords) != len(shape):
        raise ValueError(f"Expected {len(shape)} coordinates for shape {shape}, got {coords}")
    flat = 0
    stride = 1
    for coord, dim in zip(reversed(coords), reversed(shape)):
        if coord < 0 or coord >= dim:
            raise ValueError(f"Coordinate {coords} is out of bounds for shape {shape}")
        flat += coord * stride
        stride *= dim
    return flat


def read_bf16_words(path: Path, data_start: int, offsets: list[int], flat_start: int, count: int) -> list[int]:
    byte_start = int(offsets[0]) + (flat_start * 2)
    byte_count = count * 2
    with path.open("rb") as handle:
        handle.seek(data_start + byte_start)
        raw = handle.read(byte_count)
    if len(raw) != byte_count:
        raise RuntimeError(f"Could not read {byte_count} bytes from {path}")
    return list(struct.unpack(f"<{count}H", raw))


def qkv_tensor_names(header: dict[str, Any]) -> list[str]:
    names = []
    for layer in range(3):
        for projection in ("q", "k", "v"):
            name = f"layer_{layer:02d}_{projection}"
            if name in header:
                names.append(name)
    return names


def tensor_name_from_args(args: argparse.Namespace) -> str | None:
    if args.tensor:
        return args.tensor
    if args.layer is not None and args.projection:
        return f"layer_{args.layer:02d}_{args.projection}"
    return None


def print_tensor_list(header: dict[str, Any]) -> None:
    for name, info in sorted(header.items()):
        if name == "__metadata__":
            continue
        print(f"{name:34s} dtype={info['dtype']:5s} shape={info['shape']}")


def print_values(path: Path, header: dict[str, Any], data_start: int, name: str, args: argparse.Namespace) -> None:
    if name not in header or name == "__metadata__":
        choices = ", ".join(sorted(key for key in header if key != "__metadata__"))
        raise SystemExit(f"Unknown tensor {name!r}. Available tensors: {choices}")

    info = header[name]
    dtype = str(info["dtype"])
    shape = [int(value) for value in info["shape"]]
    offsets = [int(value) for value in info["data_offsets"]]
    numel = product(shape)
    if dtype != "BF16":
        raise SystemExit(f"{name} has dtype={dtype}; this viewer is for exact BF16 output values.")

    requested_flat_indices: list[int] = []
    for raw_index in args.index:
        coords = [int(part.strip()) for part in raw_index.split(",") if part.strip()]
        requested_flat_indices.append(ravel_index(coords, shape))

    if requested_flat_indices:
        flat_indices = requested_flat_indices
    elif args.flat_start is not None:
        start = max(0, min(args.flat_start, numel))
        stop = min(numel, start + max(0, args.limit))
        flat_indices = list(range(start, stop))
    else:
        if len(shape) != 3:
            raise SystemExit(f"Row view expects a 3D tensor, got shape={shape}. Use --flat-start instead.")
        batch, token = args.batch, args.token
        if batch < 0 or batch >= shape[0] or token < 0 or token >= shape[1]:
            raise SystemExit(f"batch/token ({batch}, {token}) out of bounds for shape={shape}")
        channel_start = max(0, min(args.channel_start, shape[2]))
        channel_stop = min(shape[2], channel_start + max(0, args.limit))
        flat_indices = [ravel_index([batch, token, channel], shape) for channel in range(channel_start, channel_stop)]

    output = args.out.open("w", encoding="utf-8", newline="\n") if args.out else None
    stream = output if output is not None else None

    def emit(line: str = "") -> None:
        print(line, file=stream)

    try:
        emit(f"artifact: {path}")
        emit(f"tensor:   {name}")
        emit(f"dtype:    {dtype}")
        emit(f"shape:    {shape}")
        emit()
        emit("flat_index\tindex\tbf16_hex\tbf16_value_as_float32")
        words = read_bf16_words(path, data_start, offsets, flat_indices[0], len(flat_indices)) if flat_indices else []
        if flat_indices and flat_indices != list(range(flat_indices[0], flat_indices[0] + len(flat_indices))):
            words = [read_bf16_words(path, data_start, offsets, flat_index, 1)[0] for flat_index in flat_indices]
        for flat_index, word in zip(flat_indices, words):
            coords = unravel_index(flat_index, shape)
            value = bf16_to_float32(word)
            emit(f"{flat_index}\t{coords}\t0x{word:04x}\t{value:.9g}")
    finally:
        if output is not None:
            output.close()


def export_all_qkv(path: Path, header: dict[str, Any], data_start: int, out_path: Path) -> None:
    names = qkv_tensor_names(header)
    if not names:
        raise SystemExit("No layer_XX_q/k/v tensors found in artifact.")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    total_values = 0
    with out_path.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write("# QKV_PREFILL BF16 output values\n")
        handle.write(f"# artifact={path}\n")
        handle.write("# bf16_hex is the exact raw BF16 16-bit word stored in the tensor.\n")
        handle.write("# bf16_value_as_float32 is that exact BF16 value widened to float32 for readability.\n")
        handle.write("tensor\tflat_index\tindex\tbf16_hex\tbf16_value_as_float32\n")

        for name in names:
            info = header[name]
            dtype = str(info["dtype"])
            shape = [int(value) for value in info["shape"]]
            offsets = [int(value) for value in info["data_offsets"]]
            if dtype != "BF16":
                raise SystemExit(f"{name} has dtype={dtype}; expected BF16.")
            numel = product(shape)
            words = read_bf16_words(path, data_start, offsets, 0, numel)
            handle.write(f"# tensor={name} dtype={dtype} shape={shape} numel={numel}\n")
            for flat_index, word in enumerate(words):
                coords = unravel_index(flat_index, shape)
                value = bf16_to_float32(word)
                handle.write(f"{name}\t{flat_index}\t{coords}\t0x{word:04x}\t{value:.9g}\n")
            total_values += numel

        handle.write(f"# total_qkv_values={total_values}\n")


def main() -> None:
    args = parse_args()
    path = args.artifact.expanduser().resolve()
    header, data_start = read_safetensors_header(path)
    if args.list:
        print_tensor_list(header)
        return

    if args.export_all_qkv:
        out_path = args.out or path.with_name(path.stem + "_bf16_values.txt")
        export_all_qkv(path, header, data_start, out_path.expanduser().resolve())
        print(f"wrote {out_path.expanduser().resolve()}")
        return

    name = tensor_name_from_args(args)
    if not name:
        print_tensor_list(header)
        raise SystemExit("\nChoose a tensor with --tensor layer_00_q or --layer 0 --projection q.")
    print_values(path, header, data_start, name, args)


if __name__ == "__main__":
    main()
