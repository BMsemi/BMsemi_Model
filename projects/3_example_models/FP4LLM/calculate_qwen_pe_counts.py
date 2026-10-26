from __future__ import annotations

import argparse
import csv
import json
import math
import re
from dataclasses import asdict, dataclass
from pathlib import Path


DEFAULT_PREFILL = Path("mappings") / "Qwen3_1_7B_prefill.m"
DEFAULT_DECODE = Path("mappings") / "Qwen3_1_7B_decode.m"
DEFAULT_OUT_DIR = Path("results") / "qwen_pe_counts"
DEFAULT_TILE_ROWS = 64
DEFAULT_TILE_COLS = 32


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
        raise ValueError("Expected '{' for braced block.")

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
        key.upper(): int(value)
        for key, value in re.findall(r"([A-Za-z_]+)\s*:\s*([0-9]+)", block_text)
    }


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


def parse_extent(value: str, dims: dict[str, int]) -> int:
    value = value.strip()
    if value.isdigit():
        return int(value)
    match = re.fullmatch(r"Sz\(\s*([A-Za-z_]+)\s*\)", value)
    if match:
        return int(dims[match.group(1).upper()])
    raise ValueError(f"Unsupported map extent {value!r}.")


@dataclass(frozen=True)
class MapDirective:
    kind: str
    dim: str
    size: int
    offset: int

    def as_text(self) -> str:
        return f"{self.kind}({self.size},{self.offset}) {self.dim}"


@dataclass(frozen=True)
class LayerSpec:
    phase: str
    network: str
    name: str
    role: str
    k: int
    c: int
    r: int
    s: int
    y: int
    x: int
    stride_y: int
    stride_x: int
    maps: tuple[MapDirective, ...]

    @property
    def layer_number(self) -> int | None:
        match = re.search(r"\bL([0-9]+)_", self.name)
        return int(match.group(1)) if match else None

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
    def macs(self) -> int:
        return self.tokens * self.in_dim * self.out_dim

    def crossbar_row_tiles(self, tile_rows: int) -> int:
        return ceil_div(self.in_dim, tile_rows)

    def crossbar_col_tiles(self, tile_cols: int) -> int:
        return ceil_div(self.out_dim, tile_cols)

    def crossbar_pe_count(self, tile_rows: int, tile_cols: int) -> int:
        return self.crossbar_row_tiles(tile_rows) * self.crossbar_col_tiles(tile_cols)

    def maestro_spatial_pe_count(self) -> int:
        dims = {
            "K": self.k,
            "C": self.c,
            "R": self.r,
            "S": self.s,
            "Y": self.y,
            "X": self.x,
        }
        count = 1
        for directive in self.maps:
            if directive.kind.lower() == "spatialmap":
                count *= ceil_div(dims[directive.dim.upper()], directive.offset)
        return count

    def map_text(self, kind: str | None = None) -> str:
        maps = self.maps
        if kind is not None:
            maps = tuple(item for item in maps if item.kind.lower() == kind.lower())
        return "; ".join(item.as_text() for item in maps)


def parse_map_directives(dataflow_text: str, dims: dict[str, int]) -> tuple[MapDirective, ...]:
    pattern = re.compile(
        r"\b(SpatialMap|TemporalMap)\s*"
        r"\(\s*([A-Za-z0-9_()]+)\s*,\s*([A-Za-z0-9_()]+)\s*\)\s*"
        r"([A-Za-z_]+)\s*;",
        flags=re.IGNORECASE,
    )
    directives: list[MapDirective] = []
    for match in pattern.finditer(dataflow_text):
        directives.append(
            MapDirective(
                kind=match.group(1),
                size=parse_extent(match.group(2), dims),
                offset=parse_extent(match.group(3), dims),
                dim=match.group(4).upper(),
            )
        )
    return tuple(directives)


def parse_mapping_file(path: Path, phase: str) -> tuple[str, list[LayerSpec]]:
    text = strip_line_comments(path.read_text(encoding="utf-8", errors="ignore"))
    network_match = re.search(r"\bNetwork\s+([A-Za-z_][A-Za-z0-9_]*)\s*\{", text)
    if not network_match:
        raise ValueError(f"Cannot find Network header in {path}.")
    network = network_match.group(1)

    layers: list[LayerSpec] = []
    pos = 0
    layer_pattern = re.compile(r"\bLayer\s+([A-Za-z_][A-Za-z0-9_]*)\s*\{")
    while True:
        match = layer_pattern.search(text, pos)
        if not match:
            break
        layer_name = match.group(1)
        layer_body, pos = extract_braced_block(text, match.start())

        dims_match = re.search(r"\bDimensions\s*\{", layer_body)
        if not dims_match:
            raise ValueError(f"Layer {layer_name} in {path} has no Dimensions block.")
        dims_body, _ = extract_braced_block(layer_body, dims_match.start())
        dims = parse_keyvals(dims_body)

        stride: dict[str, int] = {}
        stride_match = re.search(r"\bStride\s*\{", layer_body)
        if stride_match:
            stride_body, _ = extract_braced_block(layer_body, stride_match.start())
            stride = parse_keyvals(stride_body)

        dataflow_match = re.search(r"\bDataflow\s*\{", layer_body)
        dataflow_body = ""
        if dataflow_match:
            dataflow_body, _ = extract_braced_block(layer_body, dataflow_match.start())

        layers.append(
            LayerSpec(
                phase=phase,
                network=network,
                name=layer_name,
                role=classify_layer(layer_name),
                k=dims["K"],
                c=dims["C"],
                r=dims.get("R", 1),
                s=dims.get("S", 1),
                y=dims.get("Y", 1),
                x=dims.get("X", 1),
                stride_y=stride.get("Y", 1),
                stride_x=stride.get("X", 1),
                maps=parse_map_directives(dataflow_body, dims),
            )
        )

    if not layers:
        raise ValueError(f"No layers found in {path}.")
    return network, layers


def infer_layers_per_block(layers: list[LayerSpec]) -> int:
    roles = [layer.role for layer in layers]
    for width in range(1, min(32, len(roles)) + 1):
        if len(roles) % width:
            continue
        pattern = roles[:width]
        if all(roles[idx : idx + width] == pattern for idx in range(0, len(roles), width)):
            return width
    return len(roles)


def block_number(layer: LayerSpec, fallback_index: int, layers_per_block: int) -> int:
    layer_number = layer.layer_number or fallback_index + 1
    return ceil_div(layer_number, layers_per_block)


def layer_rows(
    layers: list[LayerSpec],
    *,
    layers_per_block: int,
    tile_rows: int,
    tile_cols: int,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for idx, layer in enumerate(layers):
        row = {
            "phase": layer.phase,
            "network": layer.network,
            "block": block_number(layer, idx, layers_per_block),
            "layer_name": layer.name,
            "role": layer.role,
            "k": layer.k,
            "c": layer.c,
            "r": layer.r,
            "s": layer.s,
            "y": layer.y,
            "x": layer.x,
            "tokens": layer.tokens,
            "in_dim": layer.in_dim,
            "out_dim": layer.out_dim,
            "crossbar_tile_rows": tile_rows,
            "crossbar_tile_cols": tile_cols,
            "crossbar_row_tiles": layer.crossbar_row_tiles(tile_rows),
            "crossbar_col_tiles": layer.crossbar_col_tiles(tile_cols),
            "pe_count": layer.crossbar_pe_count(tile_rows, tile_cols),
            "maestro_spatial_pe_count": layer.maestro_spatial_pe_count(),
            "macs": layer.macs,
            "weight_elements": layer.weight_elements,
            "spatial_maps": layer.map_text("SpatialMap"),
            "all_maps": layer.map_text(),
        }
        rows.append(row)
    return rows


def aggregate_rows(rows: list[dict[str, object]], keys: tuple[str, ...]) -> list[dict[str, object]]:
    grouped: dict[tuple[object, ...], dict[str, object]] = {}
    for row in rows:
        key = tuple(row[item] for item in keys)
        out = grouped.setdefault(
            key,
            {
                **{item: row[item] for item in keys},
                "layers": 0,
                "pe_count": 0,
                "maestro_spatial_pe_count": 0,
                "macs": 0,
                "weight_elements": 0,
                "max_layer_pe_count": 0,
                "max_layer_name": "",
                "roles": [],
            },
        )
        out["layers"] = int(out["layers"]) + 1
        out["pe_count"] = int(out["pe_count"]) + int(row["pe_count"])
        out["maestro_spatial_pe_count"] = int(out["maestro_spatial_pe_count"]) + int(row["maestro_spatial_pe_count"])
        out["macs"] = int(out["macs"]) + int(row["macs"])
        out["weight_elements"] = int(out["weight_elements"]) + int(row["weight_elements"])
        if int(row["pe_count"]) > int(out["max_layer_pe_count"]):
            out["max_layer_pe_count"] = int(row["pe_count"])
            out["max_layer_name"] = str(row["layer_name"])
        roles = list(out["roles"])
        role = str(row["role"])
        if role not in roles:
            roles.append(role)
        out["roles"] = roles

    result = []
    for out in grouped.values():
        out["roles"] = ",".join(str(role) for role in out["roles"])
        result.append(out)
    return sorted(result, key=lambda item: tuple(item[key] for key in keys))


def combined_block_rows(block_phase_rows: list[dict[str, object]]) -> list[dict[str, object]]:
    grouped: dict[int, dict[str, object]] = {}
    for row in block_phase_rows:
        block = int(row["block"])
        phase = str(row["phase"])
        out = grouped.setdefault(
            block,
            {
                "block": block,
                "prefill_pe_count": 0,
                "decode_pe_count": 0,
                "total_pe_count": 0,
                "prefill_maestro_spatial_pe_count": 0,
                "decode_maestro_spatial_pe_count": 0,
                "total_maestro_spatial_pe_count": 0,
                "prefill_macs": 0,
                "decode_macs": 0,
                "total_macs": 0,
                "prefill_layers": 0,
                "decode_layers": 0,
            },
        )
        pe_count = int(row["pe_count"])
        spatial_pe_count = int(row["maestro_spatial_pe_count"])
        macs = int(row["macs"])
        out[f"{phase}_pe_count"] = pe_count
        out[f"{phase}_maestro_spatial_pe_count"] = spatial_pe_count
        out[f"{phase}_macs"] = macs
        out[f"{phase}_layers"] = int(row["layers"])
        out["total_pe_count"] = int(out["total_pe_count"]) + pe_count
        out["total_maestro_spatial_pe_count"] = int(out["total_maestro_spatial_pe_count"]) + spatial_pe_count
        out["total_macs"] = int(out["total_macs"]) + macs
    return sorted(grouped.values(), key=lambda item: int(item["block"]))


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def phase_totals(block_phase_rows: list[dict[str, object]]) -> dict[str, dict[str, int]]:
    totals: dict[str, dict[str, int]] = {}
    for row in block_phase_rows:
        phase = str(row["phase"])
        stats = totals.setdefault(
            phase,
            {
                "blocks": 0,
                "layers": 0,
                "pe_count": 0,
                "maestro_spatial_pe_count": 0,
                "macs": 0,
                "weight_elements": 0,
            },
        )
        stats["blocks"] += 1
        stats["layers"] += int(row["layers"])
        stats["pe_count"] += int(row["pe_count"])
        stats["maestro_spatial_pe_count"] += int(row["maestro_spatial_pe_count"])
        stats["macs"] += int(row["macs"])
        stats["weight_elements"] += int(row["weight_elements"])
    return totals


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Calculate Qwen PE counts per transformer block from the prefill/decode "
            "MAESTRO mapping files."
        )
    )
    parser.add_argument("--prefill", type=Path, default=DEFAULT_PREFILL)
    parser.add_argument("--decode", type=Path, default=DEFAULT_DECODE)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--tile-rows", type=int, default=DEFAULT_TILE_ROWS)
    parser.add_argument("--tile-cols", type=int, default=DEFAULT_TILE_COLS)
    parser.add_argument(
        "--layers-per-block",
        type=int,
        default=0,
        help="Mapped layers in one transformer block. Default 0 auto-detects the repeated role pattern.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)

    prefill_network, prefill_layers = parse_mapping_file(args.prefill, "prefill")
    decode_network, decode_layers = parse_mapping_file(args.decode, "decode")

    inferred_prefill = infer_layers_per_block(prefill_layers)
    inferred_decode = infer_layers_per_block(decode_layers)
    if args.layers_per_block > 0:
        layers_per_block = int(args.layers_per_block)
    elif inferred_prefill == inferred_decode:
        layers_per_block = inferred_prefill
    else:
        raise ValueError(
            "Prefill/decode repeated layer patterns do not match. "
            f"Use --layers-per-block explicitly. prefill={inferred_prefill}, decode={inferred_decode}"
        )

    all_layers = prefill_layers + decode_layers
    rows = layer_rows(
        all_layers,
        layers_per_block=layers_per_block,
        tile_rows=int(args.tile_rows),
        tile_cols=int(args.tile_cols),
    )
    block_phase = aggregate_rows(rows, ("phase", "block"))
    block_combined = combined_block_rows(block_phase)
    role_phase = aggregate_rows(rows, ("phase", "role"))

    out_dir = args.out_dir
    write_csv(out_dir / "qwen_pe_counts_by_layer.csv", rows)
    write_csv(out_dir / "qwen_pe_counts_by_block_phase.csv", block_phase)
    write_csv(out_dir / "qwen_pe_counts_by_block.csv", block_combined)
    write_csv(out_dir / "qwen_pe_counts_by_phase_role.csv", role_phase)

    totals = phase_totals(block_phase)
    summary = {
        "inputs": {
            "prefill": str(args.prefill),
            "decode": str(args.decode),
            "prefill_network": prefill_network,
            "decode_network": decode_network,
        },
        "pe_definition": {
            "pe_count": "ceil((C*R*S) / tile_rows) * ceil(K / tile_cols)",
            "tile_rows": int(args.tile_rows),
            "tile_cols": int(args.tile_cols),
            "maestro_spatial_pe_count": "product of ceil(dimension / SpatialMap offset) across SpatialMap directives",
        },
        "layers_per_transformer_block": layers_per_block,
        "transformer_blocks": len(block_combined),
        "phase_totals": totals,
        "combined_totals": {
            "blocks": len(block_combined),
            "pe_count": sum(int(row["total_pe_count"]) for row in block_combined),
            "maestro_spatial_pe_count": sum(int(row["total_maestro_spatial_pe_count"]) for row in block_combined),
            "macs": sum(int(row["total_macs"]) for row in block_combined),
        },
        "outputs": {
            "by_layer_csv": str(out_dir / "qwen_pe_counts_by_layer.csv"),
            "by_block_phase_csv": str(out_dir / "qwen_pe_counts_by_block_phase.csv"),
            "by_block_csv": str(out_dir / "qwen_pe_counts_by_block.csv"),
            "by_phase_role_csv": str(out_dir / "qwen_pe_counts_by_phase_role.csv"),
            "summary_json": str(out_dir / "qwen_pe_counts_summary.json"),
        },
    }
    (out_dir / "qwen_pe_counts_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    first = block_combined[0] if block_combined else {}
    print(f"Wrote Qwen PE counts to {out_dir}")
    print(f"Detected {len(block_combined)} transformer blocks with {layers_per_block} mapped layers/block.")
    if first:
        print(
            "Per-block PE count: "
            f"prefill={int(first['prefill_pe_count'])}, "
            f"decode={int(first['decode_pe_count'])}, "
            f"combined={int(first['total_pe_count'])}"
        )
    print(
        "Total PE count across all blocks: "
        f"prefill={totals.get('prefill', {}).get('pe_count', 0)}, "
        f"decode={totals.get('decode', {}).get('pe_count', 0)}, "
        f"combined={summary['combined_totals']['pe_count']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
