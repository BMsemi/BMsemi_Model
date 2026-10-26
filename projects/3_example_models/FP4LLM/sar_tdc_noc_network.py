from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable


ROLE_ALIASES = {
    "gate_proj": "ffn_up",
    "up_proj": "ffn_up",
    "ffn_up": "ffn_up",
    "down_proj": "ffn_down",
    "ffn_down": "ffn_down",
    "q_proj": "q_proj",
    "k_proj": "k_proj",
    "v_proj": "v_proj",
    "o_proj": "out_proj",
    "out_proj": "out_proj",
    "qk_scores": "attn_scores_qkt",
    "attn_scores_qkt": "attn_scores_qkt",
    "av": "attn_av",
    "attn_av": "attn_av",
}


@dataclass(frozen=True)
class SarTdcRolePolicy:
    role: str
    coarse_bits: int
    fine_bits: int
    description: str

    @property
    def payload_bits_per_value(self) -> int:
        return 1 + int(self.coarse_bits) + int(self.fine_bits)


ROLE_POLICIES = {
    "ffn_up": SarTdcRolePolicy("ffn_up", 4, 3, "gate_proj/up_proj packed as c4/f3"),
    "ffn_down": SarTdcRolePolicy("ffn_down", 5, 3, "down_proj packed as c5/f3"),
    "q_proj": SarTdcRolePolicy("q_proj", 5, 3, "query projection packed as c5/f3"),
    "k_proj": SarTdcRolePolicy("k_proj", 5, 3, "key projection packed as c5/f3"),
    "v_proj": SarTdcRolePolicy("v_proj", 5, 3, "value projection packed as c5/f3"),
    "out_proj": SarTdcRolePolicy("out_proj", 5, 3, "attention output projection packed as c5/f3"),
    "attn_scores_qkt": SarTdcRolePolicy("attn_scores_qkt", 5, 3, "QK scores packed as c5/f3"),
    "attn_av": SarTdcRolePolicy("attn_av", 5, 5, "AV output packed as c5/f5"),
}


@dataclass(frozen=True)
class NoCPacketConfig:
    flit_bits: int = 128
    header_bits: int = 64
    value_count: int = 32
    aggregate_tiles: int = 1
    baseline_bits_per_value: int = 32


@dataclass(frozen=True)
class PacketLayout:
    role: str
    coarse_bits: int
    fine_bits: int
    bits_per_value: int
    values_per_tile: int
    aggregate_tiles: int
    values_per_packet: int
    payload_bits_exact: int
    payload_bytes_exact: int
    payload_bits_byte_aligned: int
    payload_bytes_byte_aligned: int
    header_bits: int
    flit_bits: int
    wire_bits_exact_flit_aligned: int
    wire_flits_exact: int
    wire_bits_byte_aligned_flit_aligned: int
    wire_flits_byte_aligned: int
    baseline_payload_bits: int
    baseline_wire_bits: int
    payload_saving_exact_pct: float
    payload_saving_byte_aligned_pct: float
    wire_saving_exact_pct: float
    wire_saving_byte_aligned_pct: float


def ceil_div(num: int, den: int) -> int:
    return int(math.ceil(int(num) / max(1, int(den))))


def pct_saved(old: int, new: int) -> float:
    if old <= 0:
        return 0.0
    return 100.0 * (float(old) - float(new)) / float(old)


def normalize_role(role: str) -> str:
    key = str(role).strip().lower()
    if key not in ROLE_ALIASES:
        raise ValueError(f"Unknown role {role!r}. Known roles: {', '.join(sorted(ROLE_ALIASES))}.")
    return ROLE_ALIASES[key]


def policy_for_role(role: str) -> SarTdcRolePolicy:
    return ROLE_POLICIES[normalize_role(role)]


def packet_layout(role: str, cfg: NoCPacketConfig) -> PacketLayout:
    policy = policy_for_role(role)
    values_per_packet = int(cfg.value_count) * int(cfg.aggregate_tiles)
    bits_per_value = policy.payload_bits_per_value
    payload_bits = values_per_packet * bits_per_value
    payload_bytes = ceil_div(payload_bits, 8)
    payload_bits_byte_aligned = payload_bytes * 8
    baseline_payload_bits = values_per_packet * int(cfg.baseline_bits_per_value)
    baseline_wire_bits = ceil_div(baseline_payload_bits + cfg.header_bits, cfg.flit_bits) * cfg.flit_bits
    exact_wire = ceil_div(payload_bits + cfg.header_bits, cfg.flit_bits) * cfg.flit_bits
    byte_wire = ceil_div(payload_bits_byte_aligned + cfg.header_bits, cfg.flit_bits) * cfg.flit_bits
    return PacketLayout(
        role=policy.role,
        coarse_bits=policy.coarse_bits,
        fine_bits=policy.fine_bits,
        bits_per_value=bits_per_value,
        values_per_tile=int(cfg.value_count),
        aggregate_tiles=int(cfg.aggregate_tiles),
        values_per_packet=values_per_packet,
        payload_bits_exact=payload_bits,
        payload_bytes_exact=payload_bytes,
        payload_bits_byte_aligned=payload_bits_byte_aligned,
        payload_bytes_byte_aligned=ceil_div(payload_bits_byte_aligned, 8),
        header_bits=int(cfg.header_bits),
        flit_bits=int(cfg.flit_bits),
        wire_bits_exact_flit_aligned=exact_wire,
        wire_flits_exact=ceil_div(payload_bits + cfg.header_bits, cfg.flit_bits),
        wire_bits_byte_aligned_flit_aligned=byte_wire,
        wire_flits_byte_aligned=ceil_div(payload_bits_byte_aligned + cfg.header_bits, cfg.flit_bits),
        baseline_payload_bits=baseline_payload_bits,
        baseline_wire_bits=baseline_wire_bits,
        payload_saving_exact_pct=pct_saved(baseline_payload_bits, payload_bits),
        payload_saving_byte_aligned_pct=pct_saved(baseline_payload_bits, payload_bits_byte_aligned),
        wire_saving_exact_pct=pct_saved(baseline_wire_bits, exact_wire),
        wire_saving_byte_aligned_pct=pct_saved(baseline_wire_bits, byte_wire),
    )


def pack_unsigned(values: Iterable[int], bits_per_value: int) -> bytes:
    """Pack integer payload values LSB-first into a byte stream."""

    bits = int(bits_per_value)
    if bits <= 0:
        raise ValueError("bits_per_value must be positive.")
    max_value = (1 << bits) - 1
    acc = 0
    bit_pos = 0
    out = bytearray()
    for value in values:
        item = int(value)
        if item < 0 or item > max_value:
            raise ValueError(f"Value {item} does not fit in {bits} bits.")
        acc |= item << bit_pos
        bit_pos += bits
        while bit_pos >= 8:
            out.append(acc & 0xFF)
            acc >>= 8
            bit_pos -= 8
    if bit_pos:
        out.append(acc & 0xFF)
    return bytes(out)


def unpack_unsigned(payload: bytes, count: int, bits_per_value: int) -> list[int]:
    """Inverse of pack_unsigned for NoC endpoint unpacking."""

    bits = int(bits_per_value)
    mask = (1 << bits) - 1
    acc = 0
    bit_pos = 0
    idx = 0
    out: list[int] = []
    while len(out) < int(count):
        while bit_pos < bits and idx < len(payload):
            acc |= int(payload[idx]) << bit_pos
            bit_pos += 8
            idx += 1
        if bit_pos < bits:
            raise ValueError("Payload ended before all values could be unpacked.")
        out.append(acc & mask)
        acc >>= bits
        bit_pos -= bits
    return out


def role_rows(cfg: NoCPacketConfig) -> list[dict[str, object]]:
    roles = ["ffn_up", "ffn_down", "q_proj", "k_proj", "v_proj", "out_proj", "attn_scores_qkt", "attn_av"]
    return [asdict(packet_layout(role, cfg)) for role in roles]


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="SAR-TDC dynamic NoC packet layout model.")
    parser.add_argument("--flit-bits", type=int, default=128)
    parser.add_argument("--header-bits", type=int, default=64)
    parser.add_argument("--value-count", type=int, default=32, help="Logical output columns per tile packet.")
    parser.add_argument("--aggregate-tiles", type=int, default=4, help="Tiles coalesced per packet to amortize header/tail.")
    parser.add_argument("--out", type=Path, default=Path("results") / "sar_tdc_noc_packet_layout.csv")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = NoCPacketConfig(
        flit_bits=args.flit_bits,
        header_bits=args.header_bits,
        value_count=args.value_count,
        aggregate_tiles=args.aggregate_tiles,
    )
    rows = role_rows(cfg)
    write_csv(args.out, rows)
    print(json.dumps({"config": asdict(cfg), "rows": rows}, indent=2), flush=True)


if __name__ == "__main__":
    main()
