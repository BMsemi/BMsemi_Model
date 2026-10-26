from __future__ import annotations

import math
import re
from dataclasses import asdict, dataclass
from pathlib import Path


def ceil_div(num: int, den: int) -> int:
    return int(math.ceil(int(num) / max(1, int(den))))


def strip_line_comments(text: str) -> str:
    out: list[str] = []
    for line in text.splitlines():
        line = re.sub(r"//.*", "", line)
        line = re.sub(r"#.*", "", line)
        line = re.sub(r"%.*", "", line)
        if line.strip():
            out.append(line)
    return "\n".join(out)


def extract_braced_block(text: str, start: int) -> tuple[str, int]:
    open_idx = text.find("{", start)
    if open_idx < 0:
        raise ValueError("Expected braced block.")
    depth = 1
    idx = open_idx + 1
    while idx < len(text) and depth > 0:
        if text[idx] == "{":
            depth += 1
        elif text[idx] == "}":
            depth -= 1
        idx += 1
    if depth != 0:
        raise ValueError("Unbalanced braces in mapping file.")
    return text[open_idx + 1 : idx - 1], idx


def parse_keyvals(block_text: str) -> dict[str, int]:
    return {
        key.upper(): int(val)
        for key, val in re.findall(r"([A-Za-z_]+)\s*:\s*([0-9]+)", block_text)
    }


def parse_numeric_map(block_text: str, dim: str, default: int = 1) -> tuple[str, int, int]:
    pattern = rf"\b(SpatialMap|TemporalMap)\s*\(\s*([0-9]+)\s*,\s*([0-9]+)\s*\)\s*{dim}\s*;"
    match = re.search(pattern, block_text, flags=re.IGNORECASE)
    if not match:
        return "TemporalMap", int(default), int(default)
    return match.group(1), int(match.group(2)), int(match.group(3))


def classify_layer(name: str) -> str:
    if "Q_Proj" in name:
        return "q_proj"
    if "K_Proj" in name:
        return "k_proj"
    if "V_Proj" in name:
        return "v_proj"
    if "Attn_Scores" in name or "QKt" in name:
        return "attn_scores_qkt"
    if "Attn_AV" in name:
        return "attn_av"
    if "Out_Proj" in name:
        return "out_proj"
    if "FFN_Gate" in name or "FFN_Up" in name or "FFN_1" in name:
        return "ffn_up"
    if "FFN_Down" in name or "FFN_2" in name:
        return "ffn_down"
    return "unknown"


@dataclass(frozen=True)
class LayerSpec:
    phase: str
    network: str
    name: str
    layer_type: str
    role: str
    k: int
    c: int
    r: int
    s: int
    y: int
    x: int
    stride_x: int
    stride_y: int
    k_map: str
    k_tile: int
    k_stride: int
    c_map: str
    c_tile: int
    c_stride: int
    y_map: str
    y_tile: int
    y_stride: int

    @property
    def tokens(self) -> int:
        return ceil_div(self.y, self.stride_y) * ceil_div(self.x, self.stride_x)

    @property
    def in_dim(self) -> int:
        return self.c * self.r * self.s

    @property
    def out_dim(self) -> int:
        return self.k

    @property
    def weight_elements(self) -> int:
        return self.in_dim * self.out_dim

    @property
    def static_weights(self) -> bool:
        return self.role not in {"attn_scores_qkt", "attn_av"}

    def as_dict(self) -> dict[str, object]:
        out = asdict(self)
        out.update(
            {
                "tokens": self.tokens,
                "in_dim": self.in_dim,
                "out_dim": self.out_dim,
                "weight_elements": self.weight_elements,
                "static_weights": self.static_weights,
            }
        )
        return out


def parse_mapping_file(path: str | Path, phase: str) -> tuple[str, list[LayerSpec]]:
    map_path = Path(path)
    text = strip_line_comments(map_path.read_text(encoding="utf-8", errors="ignore"))
    network_match = re.search(r"\bNetwork\s+([A-Za-z_][A-Za-z0-9_]*)\s*\{", text)
    if not network_match:
        raise ValueError(f"Cannot find Network header in {map_path}.")
    network = network_match.group(1)
    layers: list[LayerSpec] = []
    pos = 0
    while True:
        match = re.search(r"\bLayer\s+([A-Za-z_][A-Za-z0-9_]*)\s*\{", text[pos:])
        if not match:
            break
        layer_name = match.group(1)
        layer_start = pos + match.start()
        layer_body, layer_end = extract_braced_block(text, layer_start)
        pos = layer_end

        type_match = re.search(r"\bType\s*:\s*([A-Za-z_][A-Za-z0-9_]*)", layer_body)
        stride_match = re.search(r"\bStride\s*\{", layer_body)
        dim_match = re.search(r"\bDimensions\s*\{", layer_body)
        df_match = re.search(r"\bDataflow\s*\{", layer_body)
        if not dim_match or not df_match:
            continue

        stride_body = ""
        if stride_match:
            stride_body, _ = extract_braced_block(layer_body, stride_match.start())
        dim_body, _ = extract_braced_block(layer_body, dim_match.start())
        df_body, _ = extract_braced_block(layer_body, df_match.start())
        stride = parse_keyvals(stride_body)
        dims = parse_keyvals(dim_body)
        k_map, k_tile, k_stride = parse_numeric_map(df_body, "K")
        c_map, c_tile, c_stride = parse_numeric_map(df_body, "C")
        y_map, y_tile, y_stride = parse_numeric_map(df_body, "Y")

        layers.append(
            LayerSpec(
                phase=phase,
                network=network,
                name=layer_name,
                layer_type=type_match.group(1) if type_match else "CONV",
                role=classify_layer(layer_name),
                k=dims["K"],
                c=dims["C"],
                r=dims.get("R", 1),
                s=dims.get("S", 1),
                y=dims.get("Y", 1),
                x=dims.get("X", 1),
                stride_x=stride.get("X", 1),
                stride_y=stride.get("Y", 1),
                k_map=k_map,
                k_tile=k_tile,
                k_stride=k_stride,
                c_map=c_map,
                c_tile=c_tile,
                c_stride=c_stride,
                y_map=y_map,
                y_tile=y_tile,
                y_stride=y_stride,
            )
        )
    if not layers:
        raise ValueError(f"No layers found in {map_path}.")
    return network, layers
