from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Iterable


FMT_SAR_FP = 0x1
REDUCE_LOCAL_ROW = 0x1

ROLE_IDS = {
    "ffn_up": 1,
    "ffn_down": 2,
    "q_proj": 3,
    "k_proj": 4,
    "v_proj": 5,
    "out_proj": 6,
    "attn_scores_qkt": 7,
    "attn_av": 8,
}

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
class RolePolicy:
    role: str
    role_id: int
    coarse_bits: int
    fine_bits: int
    description: str

    @property
    def bits_per_value(self) -> int:
        return 1 + int(self.coarse_bits) + int(self.fine_bits)

    def as_dict(self) -> dict[str, object]:
        out = asdict(self)
        out["bits_per_value"] = self.bits_per_value
        return out


ROLE_POLICIES = {
    "ffn_up": RolePolicy("ffn_up", ROLE_IDS["ffn_up"], 4, 3, "gate/up projection c4/f3"),
    "ffn_down": RolePolicy("ffn_down", ROLE_IDS["ffn_down"], 5, 3, "down projection c5/f3"),
    "q_proj": RolePolicy("q_proj", ROLE_IDS["q_proj"], 5, 3, "query projection c5/f3"),
    "k_proj": RolePolicy("k_proj", ROLE_IDS["k_proj"], 5, 3, "key projection c5/f3"),
    "v_proj": RolePolicy("v_proj", ROLE_IDS["v_proj"], 5, 3, "value projection c5/f3"),
    "out_proj": RolePolicy("out_proj", ROLE_IDS["out_proj"], 5, 3, "attention output c5/f3"),
    "attn_scores_qkt": RolePolicy("attn_scores_qkt", ROLE_IDS["attn_scores_qkt"], 5, 3, "QK scores c5/f3"),
    "attn_av": RolePolicy("attn_av", ROLE_IDS["attn_av"], 5, 5, "AV output c5/f5"),
}


@dataclass(frozen=True)
class NoCHeader:
    fmt: int
    role_id: int
    bits_per_value: int
    coarse_bits: int
    fine_bits: int
    gain_shift: int
    value_count: int
    tile_group: int
    reduce_mode: int
    tail_valid_bits: int
    reserved: int = 0

    def pack(self) -> int:
        return (
            ((self.fmt & 0xF) << 60)
            | ((self.role_id & 0xF) << 56)
            | ((self.bits_per_value & 0x1F) << 51)
            | ((self.coarse_bits & 0x7) << 48)
            | ((self.fine_bits & 0x7) << 45)
            | ((self.gain_shift & 0x1F) << 40)
            | ((self.value_count & 0x3FF) << 30)
            | ((self.tile_group & 0xFFF) << 18)
            | ((self.reduce_mode & 0x7) << 15)
            | ((self.tail_valid_bits & 0x7F) << 8)
            | (self.reserved & 0xFF)
        )

    def as_dict(self) -> dict[str, int]:
        return asdict(self)


def ceil_div(num: int, den: int) -> int:
    return int(math.ceil(int(num) / max(1, int(den))))


def normalize_role(role: str | int) -> str:
    if isinstance(role, int):
        for name, role_id in ROLE_IDS.items():
            if int(role) == role_id:
                return name
        raise ValueError(f"Unknown X1 role id {role!r}.")
    key = str(role).strip().lower()
    if key not in ROLE_ALIASES:
        raise ValueError(f"Unknown role {role!r}. Known roles: {', '.join(sorted(ROLE_ALIASES))}.")
    return ROLE_ALIASES[key]


def role_policy(role: str | int) -> RolePolicy:
    return ROLE_POLICIES[normalize_role(role)]


def tail_valid_bits(total_bits: int) -> int:
    rem = int(total_bits) & 0x7F
    return 0 if rem == 0 else rem


def make_header(
    role: str | int,
    value_count: int,
    *,
    gain_shift: int = 0,
    tile_group: int = 0,
    reduce_mode: int = REDUCE_LOCAL_ROW,
) -> NoCHeader:
    policy = role_policy(role)
    total_bits = 64 + int(value_count) * policy.bits_per_value
    return NoCHeader(
        fmt=FMT_SAR_FP,
        role_id=policy.role_id,
        bits_per_value=policy.bits_per_value,
        coarse_bits=policy.coarse_bits,
        fine_bits=policy.fine_bits,
        gain_shift=int(gain_shift),
        value_count=int(value_count),
        tile_group=int(tile_group),
        reduce_mode=int(reduce_mode),
        tail_valid_bits=tail_valid_bits(total_bits),
    )


def decode_header(header: int) -> NoCHeader:
    word = int(header)
    return NoCHeader(
        fmt=(word >> 60) & 0xF,
        role_id=(word >> 56) & 0xF,
        bits_per_value=(word >> 51) & 0x1F,
        coarse_bits=(word >> 48) & 0x7,
        fine_bits=(word >> 45) & 0x7,
        gain_shift=(word >> 40) & 0x1F,
        value_count=(word >> 30) & 0x3FF,
        tile_group=(word >> 18) & 0xFFF,
        reduce_mode=(word >> 15) & 0x7,
        tail_valid_bits=(word >> 8) & 0x7F,
        reserved=word & 0xFF,
    )


def pack_lsb(values: Iterable[int], bits_per_value: int) -> int:
    bits = int(bits_per_value)
    if bits <= 0:
        raise ValueError("bits_per_value must be positive.")
    mask = (1 << bits) - 1
    out = 0
    for idx, value in enumerate(values):
        item = int(value)
        if item < 0 or item > mask:
            raise ValueError(f"Value {item} does not fit in {bits} bits.")
        out |= item << (idx * bits)
    return out


def unpack_lsb(payload: int, count: int, bits_per_value: int) -> list[int]:
    mask = (1 << int(bits_per_value)) - 1
    return [(int(payload) >> (idx * int(bits_per_value))) & mask for idx in range(int(count))]


def encode_sar_value(value: int, coarse_bits: int, fine_bits: int, gain_shift: int = 0) -> int:
    """Encode a signed integer into sign/coarse/fine SAR-FP payload bits.

    The RTL endpoint currently emits only non-negative X1 magnitudes. The compiler
    model keeps the sign bit so the same packet format can describe signed PE
    results when the endpoint grows beyond the smoke model.
    """

    sign = 1 if int(value) < 0 else 0
    magnitude = abs(int(value))
    if magnitude == 0:
        coarse = 0
        fine = 0
    else:
        msb_idx = magnitude.bit_length() - 1
        max_coarse = (1 << int(coarse_bits)) - 1
        coarse = min(msb_idx + int(gain_shift), max_coarse)
        if msb_idx >= int(fine_bits):
            fine = (magnitude >> (msb_idx - int(fine_bits))) & ((1 << int(fine_bits)) - 1)
        else:
            fine = (magnitude << (int(fine_bits) - msb_idx)) & ((1 << int(fine_bits)) - 1)
    return (sign << (int(coarse_bits) + int(fine_bits))) | (coarse << int(fine_bits)) | fine


def encode_rtl_unsigned_magnitude(magnitude: int, coarse_bits: int, fine_bits: int, gain_shift: int = 0) -> int:
    """Match the present Verilog helper in x1_memory_compiler_noc.v exactly."""

    mag = int(magnitude) & 0x1FF
    msb_idx = 0
    for bit in range(9):
        if mag & (1 << bit):
            msb_idx = bit
    if mag == 0:
        coarse = 0
        fine = 0
    else:
        coarse = msb_idx + int(gain_shift)
        max_coarse = (1 << int(coarse_bits)) - 1
        coarse = min(coarse, max_coarse)
        fine_mask = (1 << int(fine_bits)) - 1
        if msb_idx >= (int(fine_bits) - 1):
            fine = (mag >> (msb_idx - (int(fine_bits) - 1))) & fine_mask
        else:
            fine = (mag << ((int(fine_bits) - 1) - msb_idx)) & fine_mask
    raw_11 = (coarse << 5) | fine
    return raw_11 & ((1 << (1 + int(coarse_bits) + int(fine_bits))) - 1)


def build_noc_flit(
    role: str | int,
    values: Iterable[int],
    *,
    gain_shift: int = 0,
    tile_group: int = 0,
    reduce_mode: int = REDUCE_LOCAL_ROW,
    values_are_encoded: bool = False,
) -> int:
    value_list = list(values)
    header = make_header(
        role,
        len(value_list),
        gain_shift=gain_shift,
        tile_group=tile_group,
        reduce_mode=reduce_mode,
    )
    if values_are_encoded:
        encoded = value_list
    else:
        encoded = [
            encode_sar_value(v, header.coarse_bits, header.fine_bits, header.gain_shift)
            for v in value_list
        ]
    payload = pack_lsb(encoded, header.bits_per_value)
    return ((payload & ((1 << 64) - 1)) << 64) | header.pack()


def split_values_for_128b_flits(values: Iterable[int], bits_per_value: int) -> list[list[int]]:
    per_flit = max(1, 64 // int(bits_per_value))
    value_list = list(values)
    return [value_list[idx : idx + per_flit] for idx in range(0, len(value_list), per_flit)]
