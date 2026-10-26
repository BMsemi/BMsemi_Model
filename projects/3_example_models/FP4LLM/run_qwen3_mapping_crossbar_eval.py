from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import platform
import re
import shutil
import socket
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import src.crossbar_wrapper as crossbar_wrapper  # noqa: E402


CrossbarLinear = crossbar_wrapper.CrossbarLinear

ROW_TILE = 64
COL_TILE = 32
ATTN_AV_ROLE = "attn_av"
MLP_GATE_UP_ROLE = "ffn_up"
MLP_DOWN_ROLE = "ffn_down"
DEFAULT_MLP_GATE_UP_TDC_COARSE_BITS = 4
DEFAULT_MLP_GATE_UP_TDC_FINE_BITS = 3
DEFAULT_MLP_DOWN_TDC_COARSE_BITS = 5
DEFAULT_MLP_DOWN_TDC_FINE_BITS = 3


def ceil_div(num: int, den: int) -> int:
    return int(math.ceil(int(num) / int(den)))


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
    i = open_idx + 1
    while i < len(text) and depth > 0:
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
        i += 1

    if depth != 0:
        raise ValueError("Unbalanced braces in mapping file.")
    return text[open_idx + 1 : i - 1], i


def parse_keyvals(block_text: str) -> dict[str, int]:
    return {
        key.upper(): int(val)
        for key, val in re.findall(r"([A-Za-z_]+)\s*:\s*([0-9]+)", block_text)
    }


def classify_layer(name: str) -> str:
    if "Q_Proj" in name:
        return "q_proj"
    if "K_Proj" in name:
        return "k_proj"
    if "V_Proj" in name:
        return "v_proj"
    if "Attn_Scores" in name:
        return "attn_scores_qkt"
    if "Attn_AV" in name:
        return "attn_av"
    if "Out_Proj" in name:
        return "out_proj"
    if "FFN_1" in name:
        return "ffn_up"
    if "FFN_2" in name:
        return "ffn_down"
    return "unknown"


@dataclass(frozen=True)
class LayerSpec:
    phase: str
    network: str
    layer_name: str
    role: str
    k: int
    c: int
    r: int
    s: int
    y: int
    x: int
    stride_y: int
    stride_x: int

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
    def row_tiles(self) -> int:
        return ceil_div(self.in_dim, ROW_TILE)

    @property
    def col_tiles(self) -> int:
        return ceil_div(self.out_dim, COL_TILE)

    @property
    def crossbar_tiles(self) -> int:
        return self.row_tiles * self.col_tiles

    @property
    def macs(self) -> int:
        return self.tokens * self.in_dim * self.out_dim

    @property
    def weight_elements(self) -> int:
        return self.in_dim * self.out_dim

    def row(self) -> dict[str, object]:
        return {
            **asdict(self),
            "tokens": self.tokens,
            "in_dim": self.in_dim,
            "out_dim": self.out_dim,
            "row_tiles": self.row_tiles,
            "col_tiles": self.col_tiles,
            "crossbar_tiles": self.crossbar_tiles,
            "macs": self.macs,
            "weight_elements": self.weight_elements,
        }


def parse_mapping(path: Path, phase: str) -> tuple[str, list[LayerSpec]]:
    text = strip_line_comments(path.read_text(encoding="utf-8"))
    network_match = re.search(r"\bNetwork\s+([A-Za-z0-9_]+)\s*\{", text)
    if not network_match:
        raise ValueError(f"Cannot find Network header in {path}.")
    network = network_match.group(1)

    layers: list[LayerSpec] = []
    pos = 0
    layer_pattern = re.compile(r"\bLayer\s+([A-Za-z0-9_]+)\s*\{")
    while True:
        match = layer_pattern.search(text, pos)
        if not match:
            break
        layer_name = match.group(1)
        block, pos = extract_braced_block(text, match.start())
        dims_match = re.search(r"\bDimensions\s*\{([^}]*)\}", block, flags=re.DOTALL)
        if not dims_match:
            raise ValueError(f"Layer {layer_name} in {path} has no Dimensions block.")
        stride_match = re.search(r"\bStride\s*\{([^}]*)\}", block, flags=re.DOTALL)
        dims = parse_keyvals(dims_match.group(1))
        stride = parse_keyvals(stride_match.group(1)) if stride_match else {}
        layers.append(
            LayerSpec(
                phase=phase,
                network=network,
                layer_name=layer_name,
                role=classify_layer(layer_name),
                k=dims["K"],
                c=dims["C"],
                r=dims.get("R", 1),
                s=dims.get("S", 1),
                y=dims.get("Y", 1),
                x=dims.get("X", 1),
                stride_y=stride.get("Y", 1),
                stride_x=stride.get("X", 1),
            )
        )
    return network, layers


def summarize_layers(layers: list[LayerSpec]) -> dict[str, object]:
    by_phase: dict[str, dict[str, int]] = {}
    by_role: dict[str, dict[str, int]] = {}

    def add(bucket: dict[str, dict[str, int]], key: str, layer: LayerSpec) -> None:
        stats = bucket.setdefault(
            key,
            {
                "layers": 0,
                "tokens": 0,
                "crossbar_tiles": 0,
                "macs": 0,
                "weight_elements": 0,
            },
        )
        stats["layers"] += 1
        stats["tokens"] += layer.tokens
        stats["crossbar_tiles"] += layer.crossbar_tiles
        stats["macs"] += layer.macs
        stats["weight_elements"] += layer.weight_elements

    for layer in layers:
        add(by_phase, layer.phase, layer)
        add(by_role, f"{layer.phase}:{layer.role}", layer)

    return {
        "row_tile": ROW_TILE,
        "col_tile": COL_TILE,
        "total_layers": len(layers),
        "total_crossbar_tiles": sum(layer.crossbar_tiles for layer in layers),
        "total_macs": sum(layer.macs for layer in layers),
        "total_weight_elements": sum(layer.weight_elements for layer in layers),
        "by_phase": by_phase,
        "by_phase_role": by_role,
    }


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def aggregate_iv_saturation(crossbar_infos: list[dict[str, object]]) -> dict[str, object]:
    totals = {
        "iv_transfer_calls": 0,
        "iv_upper_saturation_events": 0,
        "iv_lower_clip_events": 0,
        "iv_unique_saturated_tile_columns": 0,
        "iv_saturated_tiles": 0,
        "iv_full_scale_levels": 0.0,
        "iv_nonlinearity_mode": "",
    }
    for info in crossbar_infos:
        stats = info.get("iv_saturation_stats", {})
        if not isinstance(stats, dict):
            continue
        upper_events = int(stats.get("upper_saturation_events", 0))
        unique_columns = int(stats.get("unique_saturated_columns", 0))
        totals["iv_transfer_calls"] += int(stats.get("transfer_calls", 0))
        totals["iv_upper_saturation_events"] += upper_events
        totals["iv_lower_clip_events"] += int(stats.get("lower_clip_events", 0))
        totals["iv_unique_saturated_tile_columns"] += unique_columns
        totals["iv_saturated_tiles"] += int(upper_events > 0 or unique_columns > 0)
        totals["iv_full_scale_levels"] = float(stats.get("full_scale_levels", totals["iv_full_scale_levels"]))
        totals["iv_nonlinearity_mode"] = str(stats.get("mode", totals["iv_nonlinearity_mode"]))
    return totals


def aggregate_shapes(layers: list[LayerSpec]) -> list[dict[str, object]]:
    grouped: dict[tuple[str, str, int, int, int], dict[str, object]] = {}
    for layer in layers:
        key = (layer.phase, layer.role, layer.tokens, layer.in_dim, layer.out_dim)
        row = grouped.setdefault(
            key,
            {
                "phase": layer.phase,
                "role": layer.role,
                "tokens": layer.tokens,
                "in_dim": layer.in_dim,
                "out_dim": layer.out_dim,
                "layers": 0,
                "row_tiles": layer.row_tiles,
                "col_tiles": layer.col_tiles,
                "crossbar_tiles_per_layer": layer.crossbar_tiles,
                "crossbar_tiles_total": 0,
                "macs_per_layer": layer.macs,
                "macs_total": 0,
                "example_layer": layer.layer_name,
            },
        )
        row["layers"] = int(row["layers"]) + 1
        row["crossbar_tiles_total"] = int(row["crossbar_tiles_total"]) + layer.crossbar_tiles
        row["macs_total"] = int(row["macs_total"]) + layer.macs
    return sorted(grouped.values(), key=lambda r: (str(r["phase"]), str(r["example_layer"])))


def deterministic_seed(*parts: object) -> int:
    h = hashlib.sha256("|".join(map(str, parts)).encode("utf-8")).digest()
    return int.from_bytes(h[:4], "little")


def tdc_config_for_role(
    role: str,
    *,
    enable_tdc: bool,
    tdc_bits: int,
    tdc_coarse_bits: int,
    tdc_fine_bits: int,
    tdc_gain_shift: int,
    tdc_output_format: str,
    tdc_precision_mode: str | None,
    attn_av_tdc_fine_bits: int,
    mlp_gate_up_tdc_coarse_bits: int = DEFAULT_MLP_GATE_UP_TDC_COARSE_BITS,
    mlp_gate_up_tdc_fine_bits: int = DEFAULT_MLP_GATE_UP_TDC_FINE_BITS,
    mlp_down_tdc_coarse_bits: int = DEFAULT_MLP_DOWN_TDC_COARSE_BITS,
    mlp_down_tdc_fine_bits: int = DEFAULT_MLP_DOWN_TDC_FINE_BITS,
) -> crossbar_wrapper.CrossbarTDCConfig:
    coarse_bits = int(tdc_coarse_bits)
    fine_bits = int(tdc_fine_bits)
    total_bits = int(tdc_bits)
    role_name = str(role)
    grow_total_bits = False
    if role_name == ATTN_AV_ROLE:
        fine_bits = int(attn_av_tdc_fine_bits)
        grow_total_bits = True
    elif role_name == MLP_GATE_UP_ROLE:
        coarse_bits = int(mlp_gate_up_tdc_coarse_bits)
        fine_bits = int(mlp_gate_up_tdc_fine_bits)
        grow_total_bits = True
    elif role_name == MLP_DOWN_ROLE:
        coarse_bits = int(mlp_down_tdc_coarse_bits)
        fine_bits = int(mlp_down_tdc_fine_bits)
        grow_total_bits = True

    if grow_total_bits:
        total_bits = max(total_bits, coarse_bits + fine_bits)

    return crossbar_wrapper.CrossbarTDCConfig(
        enabled=bool(enable_tdc),
        tdc_bits=total_bits,
        coarse_bits=coarse_bits,
        fine_bits=fine_bits,
        gain_shift=int(tdc_gain_shift),
        output_format=str(tdc_output_format),
        precision_mode=tdc_precision_mode if not grow_total_bits else None,
    )


def run_functional_shape(
    row: dict[str, object],
    sample_tokens: int,
    weight_bits: int,
    act_bits: int,
    *,
    enable_tdc: bool,
    tdc_bits: int,
    tdc_coarse_bits: int,
    tdc_fine_bits: int,
    tdc_gain_shift: int,
    tdc_output_format: str,
    tdc_precision_mode: str | None,
    attn_av_tdc_fine_bits: int,
    mlp_gate_up_tdc_coarse_bits: int,
    mlp_gate_up_tdc_fine_bits: int,
    mlp_down_tdc_coarse_bits: int,
    mlp_down_tdc_fine_bits: int,
    row_parallel_config: crossbar_wrapper.CrossbarPERowParallelConfig | dict | None,
    column_parallel_config: crossbar_wrapper.CrossbarPEColumnParallelConfig | dict | None,
) -> dict[str, object]:
    phase = str(row["phase"])
    role = str(row["role"])
    in_dim = int(row["in_dim"])
    out_dim = int(row["out_dim"])
    tokens = max(1, min(int(row["tokens"]), sample_tokens))
    seed = deterministic_seed(phase, role, in_dim, out_dim, tokens, weight_bits, act_bits)
    rng = np.random.default_rng(seed)
    weight = rng.normal(0.0, 0.02, size=(in_dim, out_dim)).astype(np.float32)
    x = torch.from_numpy(rng.normal(0.0, 0.25, size=(tokens, in_dim)).astype(np.float32))
    tdc_config = tdc_config_for_role(
        role,
        enable_tdc=enable_tdc,
        tdc_bits=tdc_bits,
        tdc_coarse_bits=tdc_coarse_bits,
        tdc_fine_bits=tdc_fine_bits,
        tdc_gain_shift=tdc_gain_shift,
        tdc_output_format=tdc_output_format,
        tdc_precision_mode=tdc_precision_mode,
        attn_av_tdc_fine_bits=attn_av_tdc_fine_bits,
        mlp_gate_up_tdc_coarse_bits=mlp_gate_up_tdc_coarse_bits,
        mlp_gate_up_tdc_fine_bits=mlp_gate_up_tdc_fine_bits,
        mlp_down_tdc_coarse_bits=mlp_down_tdc_coarse_bits,
        mlp_down_tdc_fine_bits=mlp_down_tdc_fine_bits,
    )

    start = time.perf_counter()
    layer = CrossbarLinear(
        weight,
        weight_bits=weight_bits,
        act_bits=act_bits,
        block_size=16,
        sigma=0.0,
        seed=seed,
        verbose=False,
        tdc_config=tdc_config,
        row_parallel_config=row_parallel_config,
        column_parallel_config=column_parallel_config,
    )
    y = layer(x)
    elapsed_s = time.perf_counter() - start
    stats = layer.stats(tokens=tokens)
    iv_saturation = aggregate_iv_saturation(layer._executor.crossbar_infos())
    y_cpu = y.detach().float().cpu()
    return {
        "phase": phase,
        "role": role,
        "sample_tokens": tokens,
        "in_dim": in_dim,
        "out_dim": out_dim,
        "row_tiles": stats.crossbar_tiles // max(1, ceil_div(out_dim, COL_TILE)),
        "col_tiles": ceil_div(out_dim, COL_TILE),
        "crossbar_tiles": stats.crossbar_tiles,
        "macs": stats.macs,
        "output_shape": list(y_cpu.shape),
        "output_checksum": float(y_cpu.sum().item()),
        "output_l2_norm": float(torch.linalg.vector_norm(y_cpu).item()),
        "elapsed_s": elapsed_s,
        "seed": seed,
        "weight_bits": weight_bits,
        "act_bits": act_bits,
        "tdc_enabled": bool(tdc_config.enabled),
        "tdc_bits": int(tdc_config.tdc_bits),
        "tdc_coarse_bits": int(tdc_config.coarse_bits),
        "tdc_fine_bits": int(tdc_config.fine_bits),
        "tdc_gain_shift": int(tdc_config.gain_shift),
        "tdc_output_format": str(tdc_config.output_format),
        "tdc_precision_mode": tdc_config.precision_mode,
        "tdc_signed_payload_bits": int(tdc_config.signed_payload_bits),
        **iv_saturation,
    }


def parse_args() -> argparse.Namespace:
    ts = time.strftime("%Y%m%d_%H%M%S")
    default_out = SCRIPT_DIR / "results" / f"qwen3_1_7b_crossbar_eval_{ts}"
    parser = argparse.ArgumentParser(description="Evaluate Qwen3-1.7B mapping files with CrossbarLinear.")
    parser.add_argument("--prefill", default=str(SCRIPT_DIR / "mappings" / "Qwen3_1_7B_prefill.m"))
    parser.add_argument("--decode", default=str(SCRIPT_DIR / "mappings" / "Qwen3_1_7B_decode.m"))
    parser.add_argument("--out-dir", default=str(default_out))
    parser.add_argument("--sample-tokens", type=int, default=1)
    parser.add_argument("--weight-bits", type=int, default=7)
    parser.add_argument("--act-bits", type=int, default=7)
    crossbar_wrapper.add_reram_tdc_args(parser)
    crossbar_wrapper.add_reram_pe_parallel_args(parser)
    parser.add_argument(
        "--attn-av-tdc-fine-bits",
        type=int,
        default=5,
        help="Fine SAR-TDC bits to use only for Attn_AV, after softmax. Total TDC bits are grown to preserve coarse range.",
    )
    parser.add_argument(
        "--mlp-gate-up-tdc-coarse-bits",
        type=int,
        default=DEFAULT_MLP_GATE_UP_TDC_COARSE_BITS,
        help="Coarse SAR-TDC bits to use only for MLP gate_proj/up_proj mapped as FFN_1.",
    )
    parser.add_argument(
        "--mlp-gate-up-tdc-fine-bits",
        type=int,
        default=DEFAULT_MLP_GATE_UP_TDC_FINE_BITS,
        help="Fine SAR-TDC bits to use only for MLP gate_proj/up_proj mapped as FFN_1.",
    )
    parser.add_argument(
        "--mlp-down-tdc-coarse-bits",
        type=int,
        default=DEFAULT_MLP_DOWN_TDC_COARSE_BITS,
        help="Coarse SAR-TDC bits to use only for MLP down_proj mapped as FFN_2.",
    )
    parser.add_argument(
        "--mlp-down-tdc-fine-bits",
        type=int,
        default=DEFAULT_MLP_DOWN_TDC_FINE_BITS,
        help="Fine SAR-TDC bits to use only for MLP down_proj mapped as FFN_2.",
    )
    parser.add_argument("--crossbar-rows", type=int, default=64)
    parser.add_argument("--crossbar-cols", type=int, default=32)
    parser.add_argument(
        "--phase-filter",
        choices=("all", "decode", "prefill"),
        default="all",
        help="Run functional smoke inference only for the selected mapping phase.",
    )
    parser.add_argument("--skip-functional", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    global ROW_TILE, COL_TILE
    ROW_TILE = int(args.crossbar_rows)
    COL_TILE = int(args.crossbar_cols)
    crossbar_wrapper.ReRAMLinearExecutor.row_tile = ROW_TILE
    crossbar_wrapper.ReRAMLinearExecutor.col_tile = COL_TILE
    row_parallel_config, column_parallel_config = crossbar_wrapper.pe_parallel_configs_from_args(args)
    base_tdc_config = crossbar_wrapper.make_tdc_config(
        enable_tdc=args.enable_tdc,
        tdc_bits=args.tdc_bits,
        tdc_coarse_bits=args.tdc_coarse_bits,
        tdc_fine_bits=args.tdc_fine_bits,
        tdc_gain_shift=args.tdc_gain_shift,
        tdc_output_format=args.tdc_output_format,
        tdc_precision_mode=args.tdc_precision_mode,
        tdc_timing_enabled=not args.disable_tdc_timing,
        tdc_timing_min_ps=args.tdc_timing_min_ps,
        tdc_timing_max_ps=args.tdc_timing_max_ps,
        tdc_timing_reference_levels=args.tdc_timing_reference_levels,
    )

    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    prefill_path = Path(args.prefill).resolve()
    decode_path = Path(args.decode).resolve()
    prefill_network, prefill_layers = parse_mapping(prefill_path, "prefill")
    decode_network, decode_layers = parse_mapping(decode_path, "decode")
    layers = prefill_layers + decode_layers
    shape_rows = aggregate_shapes(layers)
    functional_shape_rows = [
        row for row in shape_rows if args.phase_filter == "all" or row["phase"] == args.phase_filter
    ]

    shutil.copy2(prefill_path, out_dir / prefill_path.name)
    shutil.copy2(decode_path, out_dir / decode_path.name)
    write_csv(out_dir / "layer_stats.csv", [layer.row() for layer in layers])
    write_csv(out_dir / "shape_stats.csv", shape_rows)

    functional_rows: list[dict[str, object]] = []
    if not args.skip_functional:
        for idx, shape in enumerate(functional_shape_rows, start=1):
            print(
                f"[{idx}/{len(functional_shape_rows)}] crossbar inference "
                f"{shape['phase']} {shape['role']} {shape['in_dim']}->{shape['out_dim']}",
                flush=True,
            )
            functional_rows.append(
                run_functional_shape(
                    shape,
                    sample_tokens=args.sample_tokens,
                    weight_bits=args.weight_bits,
                    act_bits=args.act_bits,
                    enable_tdc=args.enable_tdc,
                    tdc_bits=args.tdc_bits,
                    tdc_coarse_bits=args.tdc_coarse_bits,
                    tdc_fine_bits=args.tdc_fine_bits,
                    tdc_gain_shift=args.tdc_gain_shift,
                    tdc_output_format=args.tdc_output_format,
                    tdc_precision_mode=args.tdc_precision_mode,
                    attn_av_tdc_fine_bits=args.attn_av_tdc_fine_bits,
                    mlp_gate_up_tdc_coarse_bits=args.mlp_gate_up_tdc_coarse_bits,
                    mlp_gate_up_tdc_fine_bits=args.mlp_gate_up_tdc_fine_bits,
                    mlp_down_tdc_coarse_bits=args.mlp_down_tdc_coarse_bits,
                    mlp_down_tdc_fine_bits=args.mlp_down_tdc_fine_bits,
                    row_parallel_config=row_parallel_config,
                    column_parallel_config=column_parallel_config,
                )
            )
            write_csv(out_dir / "functional_inference.csv", functional_rows)

    summary = {
        "name": out_dir.name,
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S %Z"),
        "host": socket.gethostname(),
        "platform": platform.platform(),
        "python": sys.version,
        "torch": torch.__version__,
        "torch_cuda_available": bool(torch.cuda.is_available()),
        "cuda_device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
        "prefill_mapping": str(prefill_path),
        "decode_mapping": str(decode_path),
        "prefill_network": prefill_network,
        "decode_network": decode_network,
        "sample_tokens": args.sample_tokens,
        "phase_filter": args.phase_filter,
        "weight_bits": args.weight_bits,
        "act_bits": args.act_bits,
        "tdc_policy": {
            "enabled": bool(args.enable_tdc),
            "precision_mode": base_tdc_config.precision_mode,
            "default_tdc_bits": base_tdc_config.tdc_bits,
            "default_coarse_bits": base_tdc_config.coarse_bits,
            "default_fine_bits": base_tdc_config.fine_bits,
            "default_signed_payload_bits": base_tdc_config.signed_payload_bits,
            "attn_av_fine_bits": args.attn_av_tdc_fine_bits,
            "attn_av_tdc_bits": max(base_tdc_config.tdc_bits, base_tdc_config.coarse_bits + args.attn_av_tdc_fine_bits),
            "mlp_gate_up": {
                "mapped_role": MLP_GATE_UP_ROLE,
                "qwen_layers": ["gate_proj", "up_proj"],
                "coarse_bits": args.mlp_gate_up_tdc_coarse_bits,
                "fine_bits": args.mlp_gate_up_tdc_fine_bits,
                "tdc_bits": max(
                    base_tdc_config.tdc_bits,
                    args.mlp_gate_up_tdc_coarse_bits + args.mlp_gate_up_tdc_fine_bits,
                ),
            },
            "mlp_down": {
                "mapped_role": MLP_DOWN_ROLE,
                "qwen_layers": ["down_proj"],
                "coarse_bits": args.mlp_down_tdc_coarse_bits,
                "fine_bits": args.mlp_down_tdc_fine_bits,
                "tdc_bits": max(base_tdc_config.tdc_bits, args.mlp_down_tdc_coarse_bits + args.mlp_down_tdc_fine_bits),
            },
            "gain_shift": args.tdc_gain_shift,
            "output_format": args.tdc_output_format,
            "rationale": (
                "Attn_AV consumes post-softmax activations; no later softmax normalizes its PE error. "
                "The MLP Qwen3-1.7B sweep keeps gate/up at c4/f3 and down_proj at c5/f3, because "
                "down_proj is coarse-range limited and failed at c4 even with extra fine bits."
            ),
        },
        "pe_row_parallel_policy": row_parallel_config.as_dict(),
        "pe_column_parallel_policy": column_parallel_config.as_dict(),
        "crossbar_wrapper": {
            "row_tile": ROW_TILE,
            "signed_logical_cols": COL_TILE,
            "physical_cols": COL_TILE * 2,
            "structure": (
                f"{ROW_TILE} input rows x {COL_TILE} signed logical output cols "
                f"({COL_TILE * 2} physical differential cols)"
            ),
        },
        "mapping_summary": summarize_layers(layers),
        "functional_inference": functional_rows,
        "outputs": {
            "layer_stats_csv": str(out_dir / "layer_stats.csv"),
            "shape_stats_csv": str(out_dir / "shape_stats.csv"),
            "functional_inference_csv": str(out_dir / "functional_inference.csv"),
            "summary_json": str(out_dir / "summary.json"),
        },
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary["mapping_summary"], indent=2), flush=True)
    print(f"Wrote results to {out_dir}", flush=True)


if __name__ == "__main__":
    main()
