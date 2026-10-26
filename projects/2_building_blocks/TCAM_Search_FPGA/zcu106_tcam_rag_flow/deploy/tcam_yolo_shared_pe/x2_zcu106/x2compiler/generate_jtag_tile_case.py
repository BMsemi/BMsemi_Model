from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re

import numpy as np

from x2_zcu106.runtime.axi_x2_mvm_driver import pack_mvm_batch_stream

from .ir import X2Program


def _tcl_words(values: np.ndarray, *, words_per_line: int = 8) -> str:
    flat = np.asarray(values, dtype=np.uint32).reshape(-1)
    lines = []
    for start in range(0, flat.size, words_per_line):
        words = " ".join(f"0x{int(value):08X}" for value in flat[start : start + words_per_line])
        lines.append(f"    {words}")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Generate an XSCT/JTAG smoke case from one compiled X2 weight tile."
    )
    parser.add_argument("--program", required=True, help="Compiled X2 program directory.")
    parser.add_argument("--output", required=True, help="Output Tcl case file.")
    parser.add_argument("--tile-id", type=int, default=0)
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--seed", type=int, default=20260727)
    args = parser.parse_args()

    program = X2Program.load(args.program)
    matches = [tile for tile in program.tiles if tile.tile_id == args.tile_id]
    if len(matches) != 1:
        raise ValueError(f"Expected one tile with id {args.tile_id}, found {len(matches)}.")
    tile = matches[0]
    if not 1 <= args.batch <= program.target.max_batch:
        raise ValueError(f"--batch must be in 1..{program.target.max_batch}.")

    weights = np.asarray(program.constants[tile.weight_key], dtype=np.int8)
    rng = np.random.default_rng(args.seed)
    vectors = rng.integers(
        -16,
        17,
        size=(args.batch, tile.rows),
        dtype=np.int16,
    ).astype(np.int8)
    expected_i32 = vectors.astype(np.int32) @ weights.astype(np.int32)
    packed = pack_mvm_batch_stream(vectors, weights)
    expected_u32 = np.ascontiguousarray(expected_i32).reshape(-1).view(np.uint32)

    safe_name = re.sub(r"[^A-Za-z0-9_.-]+", "_", program.name)
    case_name = f"{safe_name}_tile_{tile.tile_id}"
    output = Path(args.output)
    binary_path = output.with_suffix(output.suffix + ".bin")
    tcl = (
        "# Generated from a compiled X2 artifact. Do not hand-edit.\n"
        f'set case_name "{case_name}"\n'
        f"set rows {tile.rows}\n"
        f"set cols {tile.cols}\n"
        f"set batch {args.batch}\n"
        "set input_binary [file normalize [file join [file dirname [info script]] "
        f'"{binary_path.name}"]]\n'
        "set packed_words {\n"
        f"{_tcl_words(packed)}\n"
        "}\n"
        "set expected_values {\n"
        f"{_tcl_words(expected_u32)}\n"
        "}\n"
    )

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(tcl, encoding="utf-8")
    ddr_lines = np.zeros((packed.size, 4), dtype="<u4")
    ddr_lines[:, 0] = np.asarray(packed, dtype=np.uint32)
    binary_path.write_bytes(ddr_lines.tobytes())
    report = {
        "schema_version": 1,
        "case_name": case_name,
        "program": str(Path(args.program).resolve()),
        "tile_id": tile.tile_id,
        "weight_key": tile.weight_key,
        "rows": tile.rows,
        "cols": tile.cols,
        "batch": args.batch,
        "seed": args.seed,
        "vectors_i8": vectors.tolist(),
        "expected_i32": expected_i32.tolist(),
        "packed_word_count": int(packed.size),
        "input_binary": str(binary_path.resolve()),
        "input_binary_bytes": int(ddr_lines.nbytes),
        "weight_sha256": hashlib.sha256(
            np.ascontiguousarray(weights).tobytes()
        ).hexdigest(),
        "tcl": str(output.resolve()),
    }
    report_path = output.with_suffix(output.suffix + ".json")
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({"status": "ok", **report}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
