from __future__ import annotations

import argparse
import ast
import csv
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Iterator

import numpy as np


BANK_ORDER = ("pos_lo", "pos_hi", "neg_lo", "neg_hi")
BANK_TO_MACRO = {name: idx for idx, name in enumerate(BANK_ORDER)}

ROLE_ALIASES = {
    "self_attn.q_proj": "q_proj",
    "self_attn.k_proj": "k_proj",
    "self_attn.v_proj": "v_proj",
    "self_attn.o_proj": "out_proj",
    "mlp.gate_proj": "ffn_up",
    "mlp.up_proj": "ffn_up",
    "mlp.down_proj": "ffn_down",
    "gate_proj": "ffn_up",
    "up_proj": "ffn_up",
    "down_proj": "ffn_down",
    "q_proj": "q_proj",
    "k_proj": "k_proj",
    "v_proj": "v_proj",
    "o_proj": "out_proj",
    "out_proj": "out_proj",
    "ffn_up": "ffn_up",
    "ffn_down": "ffn_down",
    "attn_scores_qkt": "attn_scores_qkt",
    "attn_av": "attn_av",
}

ROLE_TDC = {
    "ffn_up": {"role_id": 1, "coarse_bits": 4, "fine_bits": 3, "bits_per_value": 8},
    "ffn_down": {"role_id": 2, "coarse_bits": 5, "fine_bits": 3, "bits_per_value": 9},
    "q_proj": {"role_id": 3, "coarse_bits": 5, "fine_bits": 3, "bits_per_value": 9},
    "k_proj": {"role_id": 4, "coarse_bits": 5, "fine_bits": 3, "bits_per_value": 9},
    "v_proj": {"role_id": 5, "coarse_bits": 5, "fine_bits": 3, "bits_per_value": 9},
    "out_proj": {"role_id": 6, "coarse_bits": 5, "fine_bits": 3, "bits_per_value": 9},
    "attn_scores_qkt": {"role_id": 7, "coarse_bits": 5, "fine_bits": 3, "bits_per_value": 9},
    "attn_av": {"role_id": 8, "coarse_bits": 5, "fine_bits": 5, "bits_per_value": 11},
}


@dataclass(frozen=True)
class WeightImageConfig:
    weight_bits: int = 7
    block_size: int = 16
    rows_per_x1: int = 32
    cols_per_x1: int = 32
    include_zero_cells: bool = True
    gain_shift: int = 0


@dataclass(frozen=True)
class QuantizedWeight:
    tensor_name: str
    role: str
    compiler_role: str
    layer: int
    source_shape_hf: tuple[int, int]
    crossbar_shape: tuple[int, int]
    weight_bits: int
    block_size: int
    weight_scale: float
    weight_q: np.ndarray

    def metadata(self) -> dict[str, object]:
        return {
            "tensor_name": self.tensor_name,
            "role": self.role,
            "compiler_role": self.compiler_role,
            "layer": self.layer,
            "source_shape_hf": list(self.source_shape_hf),
            "crossbar_shape": list(self.crossbar_shape),
            "weight_bits": self.weight_bits,
            "block_size": self.block_size,
            "weight_scale": self.weight_scale,
        }


@dataclass(frozen=True)
class X1Tile:
    tile_row: int
    tile_col: int
    input_start: int
    output_start: int
    valid_rows: int
    valid_cols: int
    q: np.ndarray

    def metadata(self) -> dict[str, int]:
        return {
            "tile_row": self.tile_row,
            "tile_col": self.tile_col,
            "input_start": self.input_start,
            "output_start": self.output_start,
            "valid_rows": self.valid_rows,
            "valid_cols": self.valid_cols,
        }


@dataclass(frozen=True)
class PayloadStats:
    payload_words: int
    nonzero_payload_words: int
    byte_count: int
    x1_tiles: int
    logical_weight_elements: int
    logical_cell_banks: int
    selected_weight_elements: int
    selected_cell_banks: int

    def as_dict(self) -> dict[str, int]:
        return asdict(self)


def ceil_div(num: int, den: int) -> int:
    return int(math.ceil(int(num) / max(1, int(den))))


def normalize_role(role: str) -> str:
    key = str(role).strip().lower()
    if key in ROLE_ALIASES:
        return ROLE_ALIASES[key]
    tail = key.rsplit(".", 1)[-1]
    if tail in ROLE_ALIASES:
        return ROLE_ALIASES[tail]
    raise ValueError(f"Unknown X1 compiler role {role!r}")


def payload_word(data: int, macro: int, row: int, col: int) -> int:
    """Pack one X1 programming payload word.

    RTL/window contract: {data[31:24], macro[23:16], col[12:8], row[4:0]}.
    Weight data uses a 4-bit ReRAM level in the low nibble of data[31:24].
    """

    return (
        ((int(data) & 0xFF) << 24)
        | ((int(macro) & 0xFF) << 16)
        | ((int(col) & 0x1F) << 8)
        | (int(row) & 0x1F)
    ) & 0xFFFF_FFFF


def decode_payload_word(word: int) -> dict[str, int]:
    value = int(word) & 0xFFFF_FFFF
    return {
        "data": (value >> 24) & 0xFF,
        "macro": (value >> 16) & 0xFF,
        "col": (value >> 8) & 0x1F,
        "row": value & 0x1F,
    }


def quantize_symmetric(x: np.ndarray, bits: int = 8) -> tuple[np.ndarray, float]:
    arr = np.asarray(x, dtype=np.float32)
    max_abs = float(np.max(np.abs(arr))) if arr.size else 0.0
    qmax = (1 << (int(bits) - 1)) - 1
    scale = max(max_abs / float(qmax), 1.0e-8)
    q = np.clip(np.rint(arr / scale), -qmax - 1, qmax).astype(np.int8)
    return q, float(scale)


def quantize_block16(x: np.ndarray, bits: int = 7, block_size: int = 16) -> np.ndarray:
    arr = np.asarray(x, dtype=np.float32)
    if arr.ndim == 0:
        raise ValueError("quantize_block16 expects at least one dimension")

    bits = int(bits)
    block_size = int(block_size)
    if bits >= 16:
        return arr.copy()
    if bits < 2:
        raise ValueError(f"bits must be >= 2, got {bits}")
    if block_size <= 0:
        raise ValueError(f"block_size must be positive, got {block_size}")

    orig_shape = arr.shape
    dim = arr.shape[-1]
    pad = (-dim) % block_size
    if pad:
        pad_width = [(0, 0)] * arr.ndim
        pad_width[-1] = (0, pad)
        arr = np.pad(arr, pad_width)

    blocks = arr.reshape(*arr.shape[:-1], -1, block_size)
    qmax = (1 << (bits - 1)) - 1
    amax = np.maximum(np.max(np.abs(blocks), axis=-1, keepdims=True), 1.0e-8)
    scale = amax / float(qmax)
    q = np.clip(np.rint(blocks / scale), -qmax, qmax)
    dq = (q * scale).reshape(*arr.shape[:-1], -1)
    return dq[..., :dim].reshape(orig_shape).astype(np.float32, copy=False)


def weight_block16(w_in_out: np.ndarray, bits: int, block_size: int = 16) -> np.ndarray:
    # Match src.crossbar_wrapper.weight_block16: block along the HF input dimension.
    w_out_in = np.asarray(w_in_out, dtype=np.float32).T
    return quantize_block16(w_out_in, bits=bits, block_size=block_size).T


def load_index(path: str | Path) -> list[dict[str, str]]:
    with Path(path).open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def select_index_row(rows: list[dict[str, str]], *, layer: int, role: str) -> dict[str, str]:
    for row in rows:
        if str(row.get("layer", "")) == str(int(layer)) and row.get("role") == role:
            return row
    available = sorted({f"L{r.get('layer')}:{r.get('role')}" for r in rows})
    raise KeyError(f"No tensor for layer={layer} role={role!r}. Available examples: {available[:8]}")


def _resolve_output_path(output_path: str | Path, index_path: Path | None) -> Path:
    path = Path(output_path)
    if path.is_absolute() or path.exists() or index_path is None:
        return path

    base = Path(index_path).resolve().parent
    candidates = (base / path, base.parent / path)
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


def _load_raw_bf16(path: Path, shape: tuple[int, ...]) -> np.ndarray:
    raw = np.fromfile(path, dtype="<u2")
    expected = int(np.prod(shape))
    if raw.size != expected:
        raise ValueError(f"{path} has {raw.size} BF16 values, expected {expected} for shape {shape}")
    return (raw.astype(np.uint32) << np.uint32(16)).view("<f4").reshape(shape)


def load_tensor_from_index_row(row: dict[str, str], *, index_path: str | Path | None = None) -> np.ndarray:
    path = _resolve_output_path(row["output_path"], Path(index_path) if index_path is not None else None)
    shape = tuple(int(v) for v in ast.literal_eval(row["shape"]))
    export_format = row.get("export_format", "")
    dtype = row.get("dtype", "").upper()
    if export_format == "raw-bf16":
        if dtype != "BF16":
            raise ValueError(f"raw-bf16 row {row['tensor_name']} has unsupported dtype {dtype}")
        return _load_raw_bf16(path, shape)
    if export_format == "npy-fp32":
        return np.load(path).astype(np.float32, copy=False)
    raise ValueError(f"Unsupported export_format {export_format!r}")


def quantize_weight_from_hf(
    weight_hf_out_in: np.ndarray,
    *,
    tensor_name: str,
    role: str,
    layer: int,
    config: WeightImageConfig | None = None,
) -> QuantizedWeight:
    cfg = config or WeightImageConfig()
    hf = np.asarray(weight_hf_out_in, dtype=np.float32)
    if hf.ndim != 2:
        raise ValueError(f"Expected a 2-D HF weight [out,in], got {hf.shape}")

    w_in_out = hf.T.copy()
    if int(cfg.weight_bits) < 8:
        w_in_out = weight_block16(w_in_out, bits=cfg.weight_bits, block_size=cfg.block_size)
    q, scale = quantize_symmetric(w_in_out, bits=8)
    return QuantizedWeight(
        tensor_name=tensor_name,
        role=role,
        compiler_role=normalize_role(role),
        layer=int(layer),
        source_shape_hf=tuple(int(v) for v in hf.shape),
        crossbar_shape=tuple(int(v) for v in q.shape),
        weight_bits=int(cfg.weight_bits),
        block_size=int(cfg.block_size),
        weight_scale=float(scale),
        weight_q=q,
    )


def quantize_weight_from_index(
    index: str | Path,
    *,
    layer: int,
    role: str,
    config: WeightImageConfig | None = None,
) -> QuantizedWeight:
    index_path = Path(index)
    rows = load_index(index_path)
    row = select_index_row(rows, layer=layer, role=role)
    weight = load_tensor_from_index_row(row, index_path=index_path)
    return quantize_weight_from_hf(
        weight,
        tensor_name=row["tensor_name"],
        role=row["role"],
        layer=int(row["layer"]),
        config=config,
    )


def decompose_int8_pos_neg(weight_q: np.ndarray) -> dict[str, np.ndarray]:
    w = np.asarray(weight_q, dtype=np.int16)
    if np.any(w < -128) or np.any(w > 127):
        raise ValueError("Signed INT8 weights must be in [-128, 127]")
    pos = np.maximum(w, 0).astype(np.int16)
    neg = np.maximum(-w, 0).astype(np.int16)
    return {
        "pos_lo": (pos & 0xF).astype(np.uint8),
        "pos_hi": ((pos >> 4) & 0xF).astype(np.uint8),
        "neg_lo": (neg & 0xF).astype(np.uint8),
        "neg_hi": ((neg >> 4) & 0xF).astype(np.uint8),
    }


def reconstruct_int8_from_banks(banks: dict[str, np.ndarray]) -> np.ndarray:
    pos = banks["pos_lo"].astype(np.int16) + (banks["pos_hi"].astype(np.int16) << 4)
    neg = banks["neg_lo"].astype(np.int16) + (banks["neg_hi"].astype(np.int16) << 4)
    return (pos - neg).astype(np.int16)


def iter_x1_tiles(q: np.ndarray, cfg: WeightImageConfig) -> Iterator[X1Tile]:
    arr = np.asarray(q, dtype=np.int8)
    in_dim, out_dim = arr.shape
    for tile_row in range(ceil_div(in_dim, cfg.rows_per_x1)):
        row_start = tile_row * cfg.rows_per_x1
        row_end = min(row_start + cfg.rows_per_x1, in_dim)
        for tile_col in range(ceil_div(out_dim, cfg.cols_per_x1)):
            col_start = tile_col * cfg.cols_per_x1
            col_end = min(col_start + cfg.cols_per_x1, out_dim)
            tile = np.zeros((cfg.rows_per_x1, cfg.cols_per_x1), dtype=np.int8)
            tile[: row_end - row_start, : col_end - col_start] = arr[row_start:row_end, col_start:col_end]
            yield X1Tile(
                tile_row=tile_row,
                tile_col=tile_col,
                input_start=row_start,
                output_start=col_start,
                valid_rows=row_end - row_start,
                valid_cols=col_end - col_start,
                q=tile,
            )


def tile_payload_words(tile: X1Tile, *, include_zero_cells: bool = True) -> Iterator[int]:
    banks = decompose_int8_pos_neg(tile.q)
    for bank_name in BANK_ORDER:
        macro = BANK_TO_MACRO[bank_name]
        bank = banks[bank_name]
        for row in range(bank.shape[0]):
            for col in range(bank.shape[1]):
                data = int(bank[row, col])
                if include_zero_cells or data != 0:
                    yield payload_word(data, macro, row, col)


def payload_words_to_tile(words: Iterable[int], rows: int = 32, cols: int = 32) -> np.ndarray:
    banks = {name: np.zeros((rows, cols), dtype=np.uint8) for name in BANK_ORDER}
    macro_to_bank = {idx: name for name, idx in BANK_TO_MACRO.items()}
    for word in words:
        item = decode_payload_word(int(word))
        if item["macro"] not in macro_to_bank:
            continue
        banks[macro_to_bank[item["macro"]]][item["row"], item["col"]] = item["data"] & 0xF
    return reconstruct_int8_from_banks(banks).astype(np.int8)


def write_payload_bin(words: Iterable[int], path: str | Path) -> PayloadStats:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    payload_words = 0
    nonzero_words = 0
    with out.open("wb") as handle:
        chunk: list[int] = []
        for word in words:
            value = int(word) & 0xFFFF_FFFF
            payload_words += 1
            if ((value >> 24) & 0xFF) != 0:
                nonzero_words += 1
            chunk.append(value)
            if len(chunk) >= 1_000_000:
                np.asarray(chunk, dtype="<u4").tofile(handle)
                chunk.clear()
        if chunk:
            np.asarray(chunk, dtype="<u4").tofile(handle)
    return PayloadStats(
        payload_words=payload_words,
        nonzero_payload_words=nonzero_words,
        byte_count=payload_words * 4,
        x1_tiles=0,
        logical_weight_elements=0,
        logical_cell_banks=0,
        selected_weight_elements=0,
        selected_cell_banks=0,
    )


def _iter_selected_tiles(
    quant: QuantizedWeight,
    cfg: WeightImageConfig,
    *,
    tile_row: int | None = None,
    tile_col: int | None = None,
    max_tiles: int | None = None,
) -> Iterator[X1Tile]:
    emitted = 0
    for tile in iter_x1_tiles(quant.weight_q, cfg):
        if tile_row is not None and tile.tile_row != int(tile_row):
            continue
        if tile_col is not None and tile.tile_col != int(tile_col):
            continue
        yield tile
        emitted += 1
        if max_tiles is not None and emitted >= int(max_tiles):
            break


def emit_weight_image(
    quant: QuantizedWeight,
    *,
    out_bin: str | Path,
    out_json: str | Path,
    config: WeightImageConfig | None = None,
    tile_row: int | None = None,
    tile_col: int | None = None,
    max_tiles: int | None = None,
) -> dict[str, object]:
    cfg = config or WeightImageConfig()
    selected_tiles = list(_iter_selected_tiles(quant, cfg, tile_row=tile_row, tile_col=tile_col, max_tiles=max_tiles))
    if not selected_tiles:
        raise ValueError("No X1 tiles selected for weight-image emission")

    def words() -> Iterator[int]:
        for tile in selected_tiles:
            yield from tile_payload_words(tile, include_zero_cells=cfg.include_zero_cells)

    stats = write_payload_bin(words(), out_bin)
    selected_weight_elements = int(sum(tile.valid_rows * tile.valid_cols for tile in selected_tiles))
    metadata = {
        "format": "x1_weight_image_u32_v1",
        "payload_word_format": "{data[31:24], macro[23:16], col[12:8], row[4:0]}",
        "bank_to_macro": BANK_TO_MACRO,
        "config": asdict(cfg),
        "quantized_weight": quant.metadata(),
        "selection": {
            "tile_row": tile_row,
            "tile_col": tile_col,
            "max_tiles": max_tiles,
            "selected_tiles": [tile.metadata() for tile in selected_tiles[:16]],
            "selected_tile_count": len(selected_tiles),
        },
        "payload": {
            **stats.as_dict(),
            "x1_tiles": len(selected_tiles),
            "logical_weight_elements": int(quant.weight_q.size),
            "logical_cell_banks": int(quant.weight_q.size * len(BANK_ORDER)),
            "selected_weight_elements": selected_weight_elements,
            "selected_cell_banks": selected_weight_elements * len(BANK_ORDER),
            "out_bin": str(out_bin),
        },
    }
    first_tile = selected_tiles[0]
    metadata["tdc_reference"] = tile_tdc_reference(
        first_tile.q,
        role=quant.compiler_role,
        weight_scale=quant.weight_scale,
        gain_shift=cfg.gain_shift,
    )

    out = Path(out_json)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    return metadata


def encode_tdc_signed(value: int, coarse_bits: int, fine_bits: int, gain_shift: int = 0) -> int:
    """Encode a signed reduced X1 output into an approximate SAR/TDC code.

    This mirrors the current RTL magnitude fine-bit window and adds a sign bit
    for compiler-side signed pos-minus-neg crossbar reference values.
    """

    sign = 1 if int(value) < 0 else 0
    mag = abs(int(value))
    if mag == 0:
        coarse = 0
        fine = 0
    else:
        msb_idx = mag.bit_length() - 1
        coarse = min(msb_idx + int(gain_shift), (1 << int(coarse_bits)) - 1)
        fine_mask = (1 << int(fine_bits)) - 1
        hidden_at = max(int(fine_bits) - 1, 0)
        if msb_idx >= hidden_at:
            fine = (mag >> (msb_idx - hidden_at)) & fine_mask
        else:
            fine = (mag << (hidden_at - msb_idx)) & fine_mask
    return ((sign & 1) << (int(coarse_bits) + int(fine_bits))) | (coarse << int(fine_bits)) | fine


def decode_tdc_signed_to_float(code: int, coarse_bits: int, fine_bits: int, gain_shift: int = 0) -> float:
    code = int(code)
    fine_mask = (1 << int(fine_bits)) - 1
    coarse_mask = (1 << int(coarse_bits)) - 1
    fine = code & fine_mask
    coarse = (code >> int(fine_bits)) & coarse_mask
    sign = -1.0 if ((code >> (int(coarse_bits) + int(fine_bits))) & 1) else 1.0
    if coarse == 0 and fine == 0:
        return 0.0
    exponent = int(coarse) - int(gain_shift) - max(int(fine_bits) - 1, 0)
    return sign * float(fine) * (2.0**exponent)


def tile_integer_outputs(q_tile: np.ndarray, activation_q: np.ndarray) -> np.ndarray:
    q = np.asarray(q_tile, dtype=np.int32)
    act = np.asarray(activation_q, dtype=np.int32)
    if act.shape[0] != q.shape[0]:
        raise ValueError(f"Activation length {act.shape[0]} does not match tile rows {q.shape[0]}")
    return act @ q


def default_activation_q(rows: int) -> np.ndarray:
    return ((np.arange(int(rows), dtype=np.int16) * 7) % 31 - 15).astype(np.int8)


def tile_tdc_reference(
    q_tile: np.ndarray,
    *,
    role: str,
    activation_q: np.ndarray | None = None,
    weight_scale: float = 1.0,
    activation_scale: float = 1.0,
    gain_shift: int = 0,
) -> dict[str, object]:
    compiler_role = normalize_role(role)
    policy = ROLE_TDC.get(compiler_role, ROLE_TDC["q_proj"])
    act = default_activation_q(q_tile.shape[0]) if activation_q is None else np.asarray(activation_q, dtype=np.int8)
    raw = tile_integer_outputs(q_tile, act).astype(np.int64)
    codes = np.asarray(
        [encode_tdc_signed(int(v), policy["coarse_bits"], policy["fine_bits"], gain_shift) for v in raw],
        dtype=np.int64,
    )
    approx_int = np.asarray(
        [decode_tdc_signed_to_float(int(c), policy["coarse_bits"], policy["fine_bits"], gain_shift) for c in codes],
        dtype=np.float64,
    )
    approx_float = approx_int * float(weight_scale) * float(activation_scale)
    exact_float = raw.astype(np.float64) * float(weight_scale) * float(activation_scale)
    err = exact_float - approx_float
    return {
        "format": "signed_tdc_float_approx_v1",
        "role": compiler_role,
        "policy": policy,
        "activation_q_first8": [int(v) for v in act[:8]],
        "raw_int_first8": [int(v) for v in raw[:8]],
        "tdc_code_first8": [int(v) for v in codes[:8]],
        "approx_int_first8": [float(v) for v in approx_int[:8]],
        "exact_float_first8": [float(v) for v in exact_float[:8]],
        "approx_float_first8": [float(v) for v in approx_float[:8]],
        "max_abs_float_error": float(np.max(np.abs(err))) if err.size else 0.0,
        "mean_abs_float_error": float(np.mean(np.abs(err))) if err.size else 0.0,
    }


def _parse_optional_index(value: int) -> int | None:
    return None if int(value) < 0 else int(value)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Emit X1 hardware weight payload words from extracted Qwen weights.")
    parser.add_argument("--index", type=Path, required=True, help="index.csv from extract_qwen_layer_weights.py")
    parser.add_argument("--layer", type=int, required=True)
    parser.add_argument("--role", required=True, help="Role from index.csv, e.g. self_attn.q_proj")
    parser.add_argument("--out-bin", type=Path, required=True)
    parser.add_argument("--out-json", type=Path, required=True)
    parser.add_argument("--weight-bits", type=int, default=7)
    parser.add_argument("--block-size", type=int, default=16)
    parser.add_argument("--tile-row", type=int, default=-1, help="X1 tile row to emit; -1 emits all rows")
    parser.add_argument("--tile-col", type=int, default=-1, help="X1 tile col to emit; -1 emits all cols")
    parser.add_argument("--max-tiles", type=int, default=0, help="Optional cap after filtering; 0 means no cap")
    parser.add_argument("--skip-zero-cells", action="store_true", help="Skip zero cell writes; only safe after X1 clear")
    args = parser.parse_args(argv)

    cfg = WeightImageConfig(
        weight_bits=args.weight_bits,
        block_size=args.block_size,
        include_zero_cells=not bool(args.skip_zero_cells),
    )
    quant = quantize_weight_from_index(args.index, layer=args.layer, role=args.role, config=cfg)
    metadata = emit_weight_image(
        quant,
        out_bin=args.out_bin,
        out_json=args.out_json,
        config=cfg,
        tile_row=_parse_optional_index(args.tile_row),
        tile_col=_parse_optional_index(args.tile_col),
        max_tiles=None if int(args.max_tiles) <= 0 else int(args.max_tiles),
    )
    print(json.dumps({"out_bin": str(args.out_bin), "out_json": str(args.out_json), "payload": metadata["payload"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
