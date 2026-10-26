import argparse
import ast
import csv
from pathlib import Path

import numpy as np


DEFAULT_INDEX = Path(r"C:\Users\Avadh\Downloads\qwen_extracted_layer_weights\index.csv")


def bf16_file_to_fp32(path: Path, shape: list[int]) -> np.ndarray:
    raw = np.fromfile(path, dtype="<u2")
    expected = int(np.prod(shape))
    if raw.size != expected:
        raise ValueError(f"{path} has {raw.size} BF16 values, but shape {shape} expects {expected}.")
    return (raw.astype(np.uint32) << 16).view("<f4").reshape(shape)


def load_index(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def find_row(rows: list[dict[str, str]], tensor: str | None, layer: int | None, role: str | None) -> dict[str, str]:
    if tensor:
        matches = [row for row in rows if row["tensor_name"] == tensor]
    else:
        if layer is None or role is None:
            raise ValueError("Use either --tensor or both --layer and --role.")
        matches = [row for row in rows if row["layer"] == str(layer) and row["role"] == role]

    if not matches:
        raise ValueError("No matching tensor found in index.csv.")
    if len(matches) > 1:
        names = "\n".join(row["tensor_name"] for row in matches[:10])
        raise ValueError(f"More than one match found:\n{names}")
    return matches[0]


def main() -> None:
    parser = argparse.ArgumentParser(description="View extracted Qwen raw BF16 weight files.")
    parser.add_argument("--index", type=Path, default=DEFAULT_INDEX)
    parser.add_argument("--tensor", help="Full tensor name from index.csv.")
    parser.add_argument("--layer", type=int, help="Layer number, for example 0.")
    parser.add_argument("--role", help="Role from index.csv, for example self_attn.q_proj.")
    parser.add_argument("--rows", type=int, default=8, help="Number of rows to print for 2-D tensors.")
    parser.add_argument("--cols", type=int, default=8, help="Number of columns to print for 2-D tensors.")
    parser.add_argument("--save-csv", type=Path, help="Optional path to save the printed slice as CSV.")
    args = parser.parse_args()

    rows = load_index(args.index)
    row = find_row(rows, args.tensor, args.layer, args.role)
    shape = ast.literal_eval(row["shape"])
    path = Path(row["output_path"])
    tensor = bf16_file_to_fp32(path, shape)

    print("tensor_name:", row["tensor_name"])
    print("file:", path)
    print("shape:", tensor.shape)
    print("dtype after conversion:", tensor.dtype)
    print()

    if tensor.ndim == 1:
        view = tensor[: args.cols]
    elif tensor.ndim == 2:
        view = tensor[: args.rows, : args.cols]
    else:
        slices = tuple(slice(0, min(dim, args.rows if idx == 0 else args.cols)) for idx, dim in enumerate(tensor.shape))
        view = tensor[slices]

    print(view)

    if args.save_csv:
        args.save_csv.parent.mkdir(parents=True, exist_ok=True)
        np.savetxt(args.save_csv, np.asarray(view), delimiter=",")
        print()
        print("saved CSV slice:", args.save_csv)


if __name__ == "__main__":
    main()
