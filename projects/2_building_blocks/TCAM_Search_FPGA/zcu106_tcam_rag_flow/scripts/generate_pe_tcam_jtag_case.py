from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from integrated.pe_tcam import FULL_CARE, TcamEntry, encode_query_rows, encode_tcam_bank


def _pack_i8_words(values: np.ndarray) -> list[int]:
    payload = np.asarray(values, dtype=np.int8).reshape(-1).view(np.uint8)
    padded = np.zeros(((payload.size + 3) // 4) * 4, dtype=np.uint8)
    padded[: payload.size] = payload
    return [
        int.from_bytes(padded[index : index + 4].tobytes(), "little")
        for index in range(0, padded.size, 4)
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate shared-PE TCAM JTAG MVM case.")
    parser.add_argument("output", type=Path)
    parser.add_argument("--physical-cols", type=int, choices=(16, 32), default=16)
    args = parser.parse_args()
    entries = [
        TcamEntry(0, 0xA5A5, FULL_CARE, 0x1000),
        TcamEntry(1, 0xA5A4, FULL_CARE, 0x1001),
        TcamEntry(2, 0xA5A7, FULL_CARE, 0x1002),
        TcamEntry(3, 0xA5A0, FULL_CARE, 0x1003),
    ]
    matrix = encode_tcam_bank(entries, 0)[:, : args.physical_cols]
    query = encode_query_rows(0xA5A5)
    packed_words = _pack_i8_words(matrix)
    packed_words.extend(_pack_i8_words(query))
    expected = query.astype(np.int32) @ matrix.astype(np.int32)
    binary_path = args.output.resolve().with_suffix(".bin")
    line_image = np.zeros(len(packed_words) * 4, dtype="<u4")
    line_image[::4] = np.asarray(packed_words, dtype=np.uint32)
    binary_path.write_bytes(line_image.tobytes())
    document = "\n".join(
        [
            'set case_name "shared_pe_tcam_32x32"',
            "set rows 32",
            f"set cols {args.physical_cols}",
            "set batch 1",
            f'set input_binary "{binary_path.as_posix()}"',
            "set packed_words {" + " ".join(f"0x{word:08X}" for word in packed_words) + "}",
            "set expected_values {" + " ".join(str(int(value)) for value in expected) + "}",
            "",
        ]
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(document, encoding="ascii")
    print(
        f"WROTE {args.output} input={binary_path} "
        f"words={len(packed_words)} outputs={len(expected)}"
    )


if __name__ == "__main__":
    main()
