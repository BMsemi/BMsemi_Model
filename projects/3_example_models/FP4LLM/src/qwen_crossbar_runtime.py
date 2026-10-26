"""Checkpoint-backed Qwen decoder runtime used by the main mapping workflow."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import platform
import shutil
import sys
import time
import types
from pathlib import Path
from typing import Any

os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.models.qwen3.modeling_qwen3 import apply_rotary_pos_emb, repeat_kv


PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from llm_reram_executor import normalize_float_format, quantize_symmetric, round_float_format  # noqa: E402
from reram_crossbar import (  # noqa: E402
    CrossbarArray,
    CrossbarIVNonlinearityConfig,
    CrossbarNoiseConfig,
    CrossbarReliabilityConfig,
    CrossbarTDCConfig,
)
from run_qwen3_mapping_crossbar_eval import LayerSpec, parse_mapping, summarize_layers  # noqa: E402
from src.crossbar_wrapper import CrossbarLinear, quantize_block16, weight_block16  # noqa: E402


DEFAULT_MODEL = "Qwen/Qwen3-1.7B"
DEFAULT_PROMPT = "Explain how ReRAM crossbars perform matrix multiplication."
DEFAULT_DECODE_TEXT = " next"
SECONDS_PER_YEAR = 365.0 * 24.0 * 60.0 * 60.0
ROLES_PER_QWEN_BLOCK = 8
LINEAR_ROLE_BY_MODULE = {
    "q_proj": "q_proj",
    "k_proj": "k_proj",
    "v_proj": "v_proj",
    "o_proj": "out_proj",
    "gate_proj": "ffn_up",
    "up_proj": "ffn_up",
    "down_proj": "ffn_down",
}
EXPECTED_BLOCK_ROLES = [
    "q_proj",
    "k_proj",
    "v_proj",
    "attn_scores_qkt",
    "attn_av",
    "out_proj",
    "ffn_up",
    "ffn_down",
]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run selected or all Qwen3 transformer blocks with software crossbar GEMMs."
    )
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--decode-text", default=DEFAULT_DECODE_TEXT)
    parser.add_argument("--gpu-metadata", type=Path, default=None)
    parser.add_argument("--block-index", type=int, default=0)
    parser.add_argument("--all-blocks", action="store_true", help="Run the full decoder stack on software crossbars.")
    parser.add_argument("--prefill-mapping", type=Path, default=PROJECT_DIR / "mappings" / "Qwen3_1_7B_prefill.m")
    parser.add_argument("--decode-mapping", type=Path, default=PROJECT_DIR / "mappings" / "Qwen3_1_7B_decode.m")
    parser.add_argument("--local-root", default=os.environ.get("HF_MODEL_ROOT", str(Path.home() / "hf_models")))
    parser.add_argument("--cache-dir", default=os.environ.get("HF_HUB_CACHE", r"E:\hf_cache\hub"))
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--dtype", choices=("bfloat16", "float32"), default="bfloat16")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--max-input-tokens", type=int, default=128)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--weight-bits", type=int, default=7)
    parser.add_argument("--act-bits", type=int, default=7)
    parser.add_argument("--block-size", type=int, default=16)
    effects_group = parser.add_mutually_exclusive_group()
    effects_group.add_argument(
        "--enable-hardware-effects-audit",
        dest="hardware_effects_audit",
        action="store_true",
        help="Emit a prompt-aware audit for reliability, read noise, IV/ADC, and SAR-TDC effects. This is the default.",
    )
    effects_group.add_argument(
        "--disable-hardware-effects-audit",
        dest="hardware_effects_audit",
        action="store_false",
        help="Skip the prompt-aware reliability/noise/IV/TDC audit.",
    )
    parser.set_defaults(hardware_effects_audit=True)
    parser.add_argument("--hardware-effect-seed", type=int, default=20260623)
    parser.add_argument("--reliability-raw-ber-ppm", type=float, default=10.0)
    parser.add_argument("--reliability-retention-days", type=float, default=30.0)
    parser.add_argument("--reliability-state-sigma-ln", type=float, default=0.02)
    parser.add_argument("--noise-bandwidth-hz", type=float, default=5_000.0)
    parser.add_argument(
        "--tdc-precision-mode",
        choices=("4b", "8b", "9b", "11b", "16b"),
        default="9b",
        help="SAR-TDC signed payload mode used by the hardware-effects audit.",
    )
    parser.add_argument("--tdc-output-format", choices=("sar_fp", "bfp"), default="sar_fp")
    parser.add_argument("--tdc-gain-shift", type=int, default=0)
    audit_group = parser.add_mutually_exclusive_group()
    audit_group.add_argument(
        "--enable-saturation-audit",
        dest="saturation_audit",
        action="store_true",
        help="Emit a vectorized IV/ADC saturation audit for the actual prompt/decode tensors. This is the default.",
    )
    audit_group.add_argument(
        "--disable-saturation-audit",
        dest="saturation_audit",
        action="store_false",
        help="Skip the vectorized IV/ADC saturation audit.",
    )
    parser.set_defaults(saturation_audit=True)
    parser.add_argument(
        "--iv-nonlinearity-mode",
        choices=("hard_clamp", "tanh"),
        default="hard_clamp",
        help="Saturation transfer model used by the audit.",
    )
    parser.add_argument(
        "--iv-linear-rows-before-clamp",
        type=int,
        default=10,
        help="Worst-case INT4 rows kept linear before clamp when full-scale is not set.",
    )
    parser.add_argument(
        "--iv-adc-full-scale-levels",
        type=float,
        default=0.0,
        help="Override branch product level that reaches ADC/SAR-TDC full-scale in the audit.",
    )
    parser.add_argument(
        "--saturation-row-groups",
        type=int,
        default=4,
        help="PE row groups used by the saturation audit; 4 matches the kanna PE row-group study.",
    )
    parser.add_argument(
        "--saturation-rows-per-group",
        type=int,
        default=0,
        help="Rows per PE row group; overrides --saturation-row-groups when positive.",
    )
    parser.add_argument(
        "--crossbar-backend",
        choices=("ideal_int", "wrapper"),
        default="wrapper",
        help="ideal_int is the fast W/A-quantized crossbar MAC path; wrapper uses CrossbarLinear directly.",
    )
    parser.add_argument(
        "--crossbar-execution", choices=("block_streamed", "eager"), default="block_streamed",
        help="Keep only the current block's wrapper tiles resident (default), or retain all tiles.",
    )
    parser.add_argument("--disable-bf16-boundary", action="store_true")
    parser.add_argument("--endurance-cycles", type=int, default=100_000)
    parser.add_argument("--static-program-cycles", type=int, default=1)
    parser.add_argument("--dynamic-kv-endurance-cycles", type=int, default=10_000_000)
    parser.add_argument("--kv-cache-token-time-s", type=float, default=0.1)
    parser.add_argument("--kv-cache-service-life-years", type=float, default=5.0)
    parser.add_argument(
        "--disable-mapped-attention-gemms",
        action="store_true",
        help="Keep QK^T and AV attention GEMMs on the framework path instead of the mapped software crossbar path.",
    )
    return parser.parse_args(argv)


def dtype_from_name(name: str) -> torch.dtype:
    if name == "bfloat16":
        return torch.bfloat16
    if name == "float32":
        return torch.float32
    raise ValueError(f"Unsupported dtype {name!r}")


def deterministic_seed(*parts: object) -> int:
    digest = hashlib.sha256("|".join(map(str, parts)).encode("utf-8")).digest()
    return int.from_bytes(digest[:4], "little")


def resolve_mapping_path(path: Path) -> Path:
    candidate = path.expanduser()
    if candidate.exists():
        return candidate.resolve()
    fallback = PROJECT_DIR / "mapping" / candidate.name
    if fallback.exists():
        return fallback.resolve()
    raise FileNotFoundError(f"Mapping file not found: {path}")


def select_block_mapping(layers: list[LayerSpec], block_index: int, phase: str) -> list[LayerSpec]:
    start = int(block_index) * ROLES_PER_QWEN_BLOCK
    end = start + ROLES_PER_QWEN_BLOCK
    selected = layers[start:end]
    if len(selected) != ROLES_PER_QWEN_BLOCK:
        raise ValueError(
            f"{phase} mapping has {len(layers)} layers and cannot provide block {block_index + 1} "
            f"entries {start + 1}..{end}."
        )
    roles = [layer.role for layer in selected]
    if roles != EXPECTED_BLOCK_ROLES:
        raise ValueError(
            f"{phase} block {block_index + 1} mapping roles are {roles}, expected {EXPECTED_BLOCK_ROLES}."
        )
    return selected


def layer_by_role(layers: list[LayerSpec]) -> dict[str, LayerSpec]:
    by_role: dict[str, LayerSpec] = {}
    for layer in layers:
        if layer.role in by_role:
            raise ValueError(f"Duplicate mapping role {layer.role!r} in selected block.")
        by_role[layer.role] = layer
    return by_role


def load_mapping_context(
    prefill_path: Path,
    decode_path: Path,
    block_index: int,
) -> dict[str, Any]:
    resolved_prefill = resolve_mapping_path(prefill_path)
    resolved_decode = resolve_mapping_path(decode_path)
    prefill_network, prefill_layers = parse_mapping(resolved_prefill, "prefill")
    decode_network, decode_layers = parse_mapping(resolved_decode, "decode")
    prefill_block = select_block_mapping(prefill_layers, block_index, "prefill")
    decode_block = select_block_mapping(decode_layers, block_index, "decode")
    selected = prefill_block + decode_block
    return {
        "prefill_path": resolved_prefill,
        "decode_path": resolved_decode,
        "prefill_network": prefill_network,
        "decode_network": decode_network,
        "prefill_layers": prefill_block,
        "decode_layers": decode_block,
        "prefill_by_role": layer_by_role(prefill_block),
        "decode_by_role": layer_by_role(decode_block),
        "selected_layers": selected,
        "summary": summarize_layers(selected),
    }


def require_capacity(label: str, actual: int, capacity: int) -> dict[str, Any]:
    actual_i = int(actual)
    capacity_i = int(capacity)
    if actual_i > capacity_i:
        raise ValueError(f"{label} actual value {actual_i} exceeds mapped capacity {capacity_i}.")
    return {"label": label, "actual": actual_i, "capacity": capacity_i, "fits": True}


def validate_sequence_mapping_capacity(
    mapping_ctx: dict[str, Any],
    *,
    prefill_tokens: int,
    decode_tokens: int,
) -> list[dict[str, Any]]:
    decode_context = int(prefill_tokens) + int(decode_tokens)
    prefill_by_role = mapping_ctx["prefill_by_role"]
    decode_by_role = mapping_ctx["decode_by_role"]
    checks: list[dict[str, Any]] = []
    for role in ("q_proj", "k_proj", "v_proj", "out_proj", "ffn_up", "ffn_down"):
        checks.append(require_capacity(f"prefill {role} tokens", prefill_tokens, prefill_by_role[role].tokens))
        checks.append(require_capacity(f"decode {role} tokens", decode_tokens, decode_by_role[role].tokens))
    checks.append(
        require_capacity("prefill attention QK query tokens", prefill_tokens, prefill_by_role["attn_scores_qkt"].tokens)
    )
    checks.append(
        require_capacity("prefill attention QK context", prefill_tokens, prefill_by_role["attn_scores_qkt"].out_dim)
    )
    checks.append(require_capacity("prefill attention AV query tokens", prefill_tokens, prefill_by_role["attn_av"].tokens))
    checks.append(require_capacity("prefill attention AV context", prefill_tokens, prefill_by_role["attn_av"].in_dim))
    checks.append(
        require_capacity("decode attention QK context", decode_context, decode_by_role["attn_scores_qkt"].out_dim)
    )
    checks.append(require_capacity("decode attention QK mapped context Y", decode_context, decode_by_role["attn_scores_qkt"].tokens))
    checks.append(require_capacity("decode attention AV context", decode_context, decode_by_role["attn_av"].in_dim))
    checks.append(require_capacity("decode attention AV mapped context Y", decode_context, decode_by_role["attn_av"].tokens))
    return checks


def write_mapping_artifacts(out_dir: Path, mapping_ctx: dict[str, Any], validation_rows: list[dict[str, Any]]) -> None:
    shutil.copy2(mapping_ctx["prefill_path"], out_dir / Path(mapping_ctx["prefill_path"]).name)
    shutil.copy2(mapping_ctx["decode_path"], out_dir / Path(mapping_ctx["decode_path"]).name)
    write_csv(out_dir / "mapped_block_layers.csv", [layer.row() for layer in mapping_ctx["selected_layers"]])
    write_csv(out_dir / "mapping_validation.csv", validation_rows)


def base_model_and_layers(model: torch.nn.Module) -> tuple[torch.nn.Module, str, torch.nn.ModuleList]:
    base = getattr(model, "model", None) or getattr(model, "transformer", None)
    if base is None:
        raise RuntimeError("Could not find the decoder backbone on the loaded model.")
    for attr in ("layers", "h"):
        layers = getattr(base, attr, None)
        if layers is not None:
            return base, attr, layers
    raise RuntimeError("Could not find decoder layers on the loaded model.")


def tensor_from_output(output: Any) -> torch.Tensor:
    if isinstance(output, tuple):
        output = output[0]
    if not torch.is_tensor(output):
        raise TypeError(f"Expected a tensor output, got {type(output)!r}")
    return output


def capture_tensor(tensor: torch.Tensor) -> torch.Tensor:
    return tensor.detach().to(torch.bfloat16).cpu().contiguous()


def tensor_stats(tensor: torch.Tensor) -> dict[str, Any]:
    values = tensor.float().reshape(-1)
    return {
        "shape": list(tensor.shape),
        "dtype": str(tensor.dtype).replace("torch.", ""),
        "numel": int(tensor.numel()),
        "mean": float(values.mean().item()),
        "std": float(values.std(unbiased=False).item()) if values.numel() > 1 else 0.0,
        "min": float(values.min().item()),
        "max": float(values.max().item()),
        "absmax": float(values.abs().max().item()),
    }


def unravel_index(flat_index: int, shape: list[int]) -> tuple[int, ...]:
    coords = []
    remaining = int(flat_index)
    for dim in reversed(shape):
        coords.append(remaining % int(dim))
        remaining //= int(dim)
    return tuple(reversed(coords))


def bf16_words(tensor: torch.Tensor) -> list[int]:
    bf16 = tensor.to(torch.bfloat16).contiguous()
    f32 = bf16.float().contiguous()
    words = ((f32.view(torch.int32) >> 16) & 0xFFFF).reshape(-1).to(torch.int32)
    return [int(value) for value in words.tolist()]


def write_bf16_text(path: Path, captures: dict[str, dict[str, torch.Tensor]], metadata: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write("# Qwen3-1.7B_Transformer_block_1 software crossbar BF16 values\n")
        handle.write("# This is a software crossbar run; crossbar_used=true.\n")
        handle.write("# bf16_hex is the exact raw BF16 16-bit word stored for comparison.\n")
        handle.write("# bf16_value_as_float32 is the exact BF16 value widened to float32 for readability.\n")
        handle.write(f"# metadata_json={json.dumps(metadata, sort_keys=True)}\n")
        handle.write("phase\ttensor\tflat_index\tindex\tbf16_hex\tbf16_value_as_float32\n")

        total_values = 0
        for phase in ("prefill", "decode"):
            for name, tensor in captures[phase].items():
                shape = list(tensor.shape)
                words = bf16_words(tensor)
                values = tensor.float().reshape(-1).tolist()
                handle.write(f"# phase={phase} tensor={name} dtype=BF16 shape={shape} numel={tensor.numel()}\n")
                for flat_index, (word, value) in enumerate(zip(words, values)):
                    coords = unravel_index(flat_index, shape)
                    handle.write(f"{phase}\t{name}\t{flat_index}\t{coords}\t0x{word:04x}\t{value:.9g}\n")
                total_values += int(tensor.numel())
        handle.write(f"# total_values={total_values}\n")


def write_summary_text(path: Path, metadata: dict[str, Any], captures: dict[str, dict[str, torch.Tensor]]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write("Qwen3-1.7B_Transformer_block_1 Software Crossbar Summary\n")
        handle.write("========================================================\n\n")
        for key in [
            "run_name",
            "model",
            "phase_list",
            "reference_backend",
            "crossbar_used",
            "crossbar_backend",
            "mapping_used",
            "mapped_attention_gemms_enabled",
            "prefill_mapping",
            "decode_mapping",
            "original_layer_count",
            "block_index_zero_based",
            "block_index_one_based",
            "dtype",
            "device",
            "prefill_input_shape",
            "decode_input_shape",
            "elapsed_prefill_s",
            "elapsed_decode_s",
        ]:
            handle.write(f"{key}: {metadata.get(key)}\n")
        handle.write("\nTensor summaries\n")
        handle.write("phase\ttensor\tshape\tdtype\tnumel\tmean\tstd\tmin\tmax\tabsmax\n")
        for phase in ("prefill", "decode"):
            for name, tensor in captures[phase].items():
                stats = tensor_stats(tensor)
                handle.write(
                    f"{phase}\t{name}\t{stats['shape']}\t{stats['dtype']}\t{stats['numel']}\t"
                    f"{stats['mean']:.9g}\t{stats['std']:.9g}\t{stats['min']:.9g}\t"
                    f"{stats['max']:.9g}\t{stats['absmax']:.9g}\n"
                )


def iv_full_scale_levels(adc_full_scale_levels: float, linear_rows_before_clamp: int) -> float:
    configured = float(adc_full_scale_levels)
    if configured > 0.0:
        return configured
    max_int4_level = 15
    return float(max(int(linear_rows_before_clamp), 1) * max_int4_level * max_int4_level)


def row_parallel_ranges(tile_rows: int, groups: int, rows_per_group: int) -> list[slice]:
    tile_rows = int(tile_rows)
    rows_per_group = max(int(rows_per_group), 0)
    if rows_per_group > 0:
        return [
            slice(start, min(start + rows_per_group, tile_rows))
            for start in range(0, tile_rows, rows_per_group)
        ]

    groups = min(max(int(groups), 1), tile_rows)
    ranges: list[slice] = []
    for group_idx in range(groups):
        start = (group_idx * tile_rows) // groups
        end = ((group_idx + 1) * tile_rows) // groups
        if end > start:
            ranges.append(slice(start, end))
    return ranges or [slice(0, tile_rows)]


def saturation_config_from_args(args: argparse.Namespace) -> dict[str, Any]:
    full_scale = iv_full_scale_levels(args.iv_adc_full_scale_levels, args.iv_linear_rows_before_clamp)
    return {
        "enabled": bool(args.saturation_audit),
        "audit_only": True,
        "branch_model": "zero_point_corrected_signed_branch",
        "iv_nonlinearity_mode": args.iv_nonlinearity_mode,
        "iv_adc_full_scale_levels": float(full_scale),
        "iv_linear_rows_before_clamp": int(args.iv_linear_rows_before_clamp),
        "row_tile": 64,
        "col_tile": 32,
        "row_groups": int(args.saturation_row_groups),
        "rows_per_group": int(args.saturation_rows_per_group),
        "note": (
            "Counts actual W7/A7 prompt/decode branch-current magnitudes that exceed full-scale. "
            "The BF16 values remain from the ideal_int software-crossbar value path."
        ),
    }


def empty_saturation_stats() -> dict[str, Any]:
    return {
        "iv_transfer_calls": 0,
        "iv_upper_saturation_events": 0,
        "iv_lower_clip_events": 0,
        "iv_unique_saturated_tile_columns": 0,
        "iv_saturated_tiles": 0,
        "iv_observed_tile_columns": 0,
        "iv_tile_reads": 0,
        "iv_max_branch_level": 0.0,
    }


def add_saturation_stats(dst: dict[str, Any], src: dict[str, Any]) -> None:
    for key in (
        "iv_transfer_calls",
        "iv_upper_saturation_events",
        "iv_lower_clip_events",
        "iv_unique_saturated_tile_columns",
        "iv_saturated_tiles",
        "iv_observed_tile_columns",
        "iv_tile_reads",
    ):
        dst[key] = int(dst.get(key, 0)) + int(src.get(key, 0))
    dst["iv_max_branch_level"] = max(float(dst.get("iv_max_branch_level", 0.0)), float(src.get("iv_max_branch_level", 0.0)))


def audit_quantized_saturation(
    x_q: np.ndarray,
    weight_q: np.ndarray,
    *,
    phase: str,
    module: str,
    role: str,
    call_kind: str,
    batch_matrices: int,
    config: dict[str, Any],
) -> dict[str, Any]:
    x_int = np.asarray(x_q, dtype=np.int16).reshape(-1, np.asarray(x_q).shape[-1])
    w_int = np.asarray(weight_q, dtype=np.int16)
    if w_int.ndim != 2:
        raise ValueError(f"weight_q must be 2D, got {w_int.shape}")
    if x_int.shape[-1] != w_int.shape[0]:
        raise ValueError(f"saturation audit shape mismatch: x {x_int.shape} vs weight {w_int.shape}")

    tokens = int(x_int.shape[0])
    in_dim = int(w_int.shape[0])
    out_dim = int(w_int.shape[1])
    row_tile = int(config.get("row_tile", 64))
    col_tile = int(config.get("col_tile", 32))
    full_scale = float(config["iv_adc_full_scale_levels"])
    ranges = row_parallel_ranges(row_tile, int(config.get("row_groups", 1)), int(config.get("rows_per_group", 0)))
    row_tiles = int(math.ceil(in_dim / row_tile))
    col_tiles = int(math.ceil(out_dim / col_tile))

    stats = empty_saturation_stats()
    stats.update(
        {
            "phase": phase,
            "module": module,
            "role": role,
            "call_kind": call_kind,
            "tokens": tokens,
            "batch_matrices": int(batch_matrices),
            "in_dim": in_dim,
            "out_dim": out_dim,
            "row_tiles": row_tiles,
            "col_tiles": col_tiles,
            "iv_full_scale_levels": full_scale,
            "iv_nonlinearity_mode": str(config.get("iv_nonlinearity_mode", "hard_clamp")),
            "iv_row_groups": len(ranges),
            "iv_rows_per_group": int(config.get("rows_per_group", 0)),
        }
    )

    for r_block in range(row_tiles):
        row_start = r_block * row_tile
        row_end = min(row_start + row_tile, in_dim)
        row_count = row_end - row_start
        x_tile = np.zeros((tokens, row_tile), dtype=np.int16)
        w_tile = np.zeros((row_tile, out_dim), dtype=np.int16)
        x_tile[:, :row_count] = x_int[:, row_start:row_end]
        w_tile[:row_count, :] = w_int[row_start:row_end, :]
        tile_column_masks = [
            np.zeros((min((c_block + 1) * col_tile, out_dim) - c_block * col_tile,), dtype=bool)
            for c_block in range(col_tiles)
        ]
        tile_has_saturation = [False for _ in range(col_tiles)]

        for row_slice in ranges:
            partial = x_tile[:, row_slice].astype(np.int32) @ w_tile[row_slice, :].astype(np.int32)
            branch = np.abs(partial)
            if branch.size:
                stats["iv_max_branch_level"] = max(float(stats["iv_max_branch_level"]), float(np.max(branch)))
            for c_block in range(col_tiles):
                col_start = c_block * col_tile
                col_end = min(col_start + col_tile, out_dim)
                upper = branch[:, col_start:col_end] > full_scale
                upper_count = int(np.count_nonzero(upper))
                stats["iv_upper_saturation_events"] = int(stats["iv_upper_saturation_events"]) + upper_count
                if upper_count:
                    tile_has_saturation[c_block] = True
                    tile_column_masks[c_block] |= np.any(upper, axis=0)

        for c_block, mask in enumerate(tile_column_masks):
            col_start = c_block * col_tile
            col_end = min(col_start + col_tile, out_dim)
            stats["iv_observed_tile_columns"] = int(stats["iv_observed_tile_columns"]) + (col_end - col_start)
            stats["iv_unique_saturated_tile_columns"] = int(stats["iv_unique_saturated_tile_columns"]) + int(np.count_nonzero(mask))
            stats["iv_saturated_tiles"] = int(stats["iv_saturated_tiles"]) + int(tile_has_saturation[c_block])

    stats["iv_tile_reads"] = int(tokens * row_tiles * col_tiles * len(ranges))
    stats["iv_transfer_calls"] = int(stats["iv_tile_reads"]) * 2
    observed = max(int(stats["iv_observed_tile_columns"]), 1)
    stats["iv_clamped_column_fraction"] = float(stats["iv_unique_saturated_tile_columns"]) / float(observed)
    return stats


def audit_float_saturation(
    x_float: np.ndarray,
    weight_float: np.ndarray | None,
    *,
    weight_q: np.ndarray | None = None,
    phase: str,
    module: str,
    role: str,
    call_kind: str,
    weight_bits: int,
    act_bits: int,
    block_size: int,
    float_format: str,
    batch_matrices: int,
    config: dict[str, Any],
) -> dict[str, Any]:
    x_arr = np.asarray(x_float, dtype=np.float32)
    x_2d = round_float_format(x_arr.reshape(-1, x_arr.shape[-1]), float_format)
    x_block = quantize_block16(x_2d, bits=act_bits, block_size=block_size)
    x_q, _x_scale = quantize_symmetric(x_block)

    if weight_q is None:
        if weight_float is None:
            raise ValueError("weight_float or weight_q is required for saturation audit.")
        w_arr = round_float_format(np.asarray(weight_float, dtype=np.float32), float_format)
        w_block = weight_block16(w_arr, bits=weight_bits, block_size=block_size) if weight_bits < 8 else w_arr
        weight_q, _w_scale = quantize_symmetric(w_block)

    return audit_quantized_saturation(
        x_q,
        weight_q,
        phase=phase,
        module=module,
        role=role,
        call_kind=call_kind,
        batch_matrices=batch_matrices,
        config=config,
    )


def audit_batched_attention_saturation(
    x: torch.Tensor,
    weight: torch.Tensor,
    *,
    phase: str,
    role: str,
    weight_bits: int,
    act_bits: int,
    block_size: int,
    float_format: str,
    config: dict[str, Any],
) -> dict[str, Any]:
    in_dim = int(x.shape[-1])
    out_dim = int(weight.shape[-1])
    prefix = tuple(int(v) for v in x.shape[:-2])
    tokens = int(x.shape[-2])
    batch_count = int(math.prod(prefix)) if prefix else 1
    x_np = x.detach().float().cpu().numpy().reshape(batch_count, tokens, in_dim)
    w_np_raw = weight.detach().float().cpu().numpy()
    if weight.ndim == 2:
        w_np = np.broadcast_to(w_np_raw.reshape(1, in_dim, out_dim), (batch_count, in_dim, out_dim))
    else:
        weight_prefix = tuple(int(v) for v in weight.shape[:-2])
        if weight_prefix != prefix:
            raise ValueError(f"batched weight prefix {weight_prefix} does not match x prefix {prefix}.")
        w_np = w_np_raw.reshape(batch_count, in_dim, out_dim)

    aggregate = empty_saturation_stats()
    for x_i, w_i in zip(x_np, w_np):
        row = audit_float_saturation(
            x_i,
            w_i,
            phase=phase,
            module=f"mapped_{role}",
            role=role,
            call_kind="mapped_attention_gemm",
            weight_bits=weight_bits,
            act_bits=act_bits,
            block_size=block_size,
            float_format=float_format,
            batch_matrices=1,
            config=config,
        )
        add_saturation_stats(aggregate, row)

    aggregate.update(
        {
            "phase": phase,
            "module": f"mapped_{role}",
            "role": role,
            "call_kind": "mapped_attention_gemm",
            "tokens": tokens,
            "batch_matrices": batch_count,
            "in_dim": in_dim,
            "out_dim": out_dim,
            "row_tiles": int(math.ceil(in_dim / int(config.get("row_tile", 64)))),
            "col_tiles": int(math.ceil(out_dim / int(config.get("col_tile", 32)))),
            "iv_full_scale_levels": float(config["iv_adc_full_scale_levels"]),
            "iv_nonlinearity_mode": str(config.get("iv_nonlinearity_mode", "hard_clamp")),
            "iv_row_groups": len(row_parallel_ranges(int(config.get("row_tile", 64)), int(config.get("row_groups", 1)), int(config.get("rows_per_group", 0)))),
            "iv_rows_per_group": int(config.get("rows_per_group", 0)),
        }
    )
    observed = max(int(aggregate["iv_observed_tile_columns"]), 1)
    aggregate["iv_clamped_column_fraction"] = float(aggregate["iv_unique_saturated_tile_columns"]) / float(observed)
    return aggregate


def get_replaced_linear(layer: nn.Module, module_name: str) -> nn.Module:
    attn = getattr(layer, "self_attn", None) or getattr(layer, "attention", None)
    mlp = getattr(layer, "mlp", None)
    if module_name in {"q_proj", "k_proj", "v_proj", "o_proj"}:
        return getattr(attn, module_name)
    return getattr(mlp, module_name)


def linear_saturation_rows(
    selected_layer: nn.Module,
    captures: dict[str, dict[str, torch.Tensor]],
    *,
    weight_bits: int,
    act_bits: int,
    block_size: int,
    float_format: str,
    config: dict[str, Any],
) -> list[dict[str, Any]]:
    input_by_module = {
        "q_proj": "layer_00_input_layernorm_output",
        "k_proj": "layer_00_input_layernorm_output",
        "v_proj": "layer_00_input_layernorm_output",
        "o_proj": "layer_00_attn_context_before_o_proj",
        "gate_proj": "layer_00_post_attention_layernorm_output",
        "up_proj": "layer_00_post_attention_layernorm_output",
        "down_proj": "layer_00_mlp_gated_activation_before_down_proj",
    }
    rows: list[dict[str, Any]] = []
    for phase in ("prefill", "decode"):
        for module_name, input_name in input_by_module.items():
            module = get_replaced_linear(selected_layer, module_name)
            weight_q = getattr(module, "weight_q", None)
            if weight_q is None and hasattr(module, "_executor"):
                weight_q = getattr(module._executor, "weight_q", None)
            if weight_q is None:
                continue
            x_tensor = captures[phase][input_name]
            role = LINEAR_ROLE_BY_MODULE[module_name]
            rows.append(
                audit_float_saturation(
                    x_tensor.float().cpu().numpy(),
                    None,
                    weight_q=np.asarray(weight_q),
                    phase=phase,
                    module=module_name,
                    role=role,
                    call_kind="linear",
                    weight_bits=weight_bits,
                    act_bits=act_bits,
                    block_size=block_size,
                    float_format=float_format,
                    batch_matrices=1,
                    config=config,
                )
            )
    return rows


def saturation_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    summary = {
        "rows": len(rows),
        "by_phase": {},
        "by_phase_role": {},
        "total": empty_saturation_stats(),
    }
    for row in rows:
        add_saturation_stats(summary["total"], row)
        phase = str(row["phase"])
        role_key = f"{phase}:{row['role']}"
        by_phase = summary["by_phase"].setdefault(phase, empty_saturation_stats())
        by_role = summary["by_phase_role"].setdefault(role_key, empty_saturation_stats())
        add_saturation_stats(by_phase, row)
        add_saturation_stats(by_role, row)
    total_observed = max(int(summary["total"]["iv_observed_tile_columns"]), 1)
    summary["total"]["iv_clamped_column_fraction"] = (
        float(summary["total"]["iv_unique_saturated_tile_columns"]) / float(total_observed)
    )
    for bucket in (summary["by_phase"], summary["by_phase_role"]):
        for values in bucket.values():
            observed = max(int(values["iv_observed_tile_columns"]), 1)
            values["iv_clamped_column_fraction"] = float(values["iv_unique_saturated_tile_columns"]) / float(observed)
    return summary


def write_saturation_markdown(
    path: Path,
    *,
    config: dict[str, Any],
    summary: dict[str, Any],
    rows: list[dict[str, Any]],
) -> None:
    total = summary["total"]
    lines = [
        "# Software Crossbar Saturation Report",
        "",
        f"- Branch model: `{config['branch_model']}`",
        f"- IV mode: `{config['iv_nonlinearity_mode']}`",
        f"- Full-scale: {config['iv_adc_full_scale_levels']} product levels",
        f"- PE row groups: {config['row_groups']}",
        f"- Upper saturation events: {total['iv_upper_saturation_events']}",
        f"- Clamped tile-columns: {total['iv_unique_saturated_tile_columns']}",
        f"- Saturated tiles: {total['iv_saturated_tiles']}",
        "",
        "## By Phase",
        "",
        "| Phase | Upper saturation events | Clamped tile-columns | Saturated tiles | Max branch level |",
        "|---|---:|---:|---:|---:|",
    ]
    for phase in sorted(summary["by_phase"]):
        row = summary["by_phase"][phase]
        lines.append(
            f"| `{phase}` | {row['iv_upper_saturation_events']} | "
            f"{row['iv_unique_saturated_tile_columns']} | {row['iv_saturated_tiles']} | "
            f"{float(row['iv_max_branch_level']):.0f} |"
        )
    lines.extend(
        [
            "",
            "## By Role",
            "",
            "| Phase | Role | Module / GEMM | Upper saturation events | Clamped tile-columns | Saturated tiles | Max branch level |",
            "|---|---|---|---:|---:|---:|---:|",
        ]
    )
    for row in rows:
        lines.append(
            f"| `{row['phase']}` | `{row['role']}` | `{row['module']}` | "
            f"{row['iv_upper_saturation_events']} | {row['iv_unique_saturated_tile_columns']} | "
            f"{row['iv_saturated_tiles']} | {float(row['iv_max_branch_level']):.0f} |"
        )
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def hardware_effect_configs_from_args(args: argparse.Namespace) -> dict[str, Any]:
    reliability = CrossbarReliabilityConfig(
        enabled=True,
        raw_ber_ppm=float(args.reliability_raw_ber_ppm),
        use_ecc=True,
        ecc_encode_latency_cycles=1,
        ecc_decode_latency_cycles=1,
        retention_model="state_power_law",
        retention_time_s=float(args.reliability_retention_days) * 24.0 * 3600.0,
        retention_lrs_beta_log10_per_decade=-0.02556,
        retention_hrs_beta_log10_per_decade=-0.04041,
        state_sigma_ln=float(args.reliability_state_sigma_ln),
    )
    noise = CrossbarNoiseConfig(enabled=True, bandwidth_hz=float(args.noise_bandwidth_hz))
    iv = CrossbarIVNonlinearityConfig(
        enabled=True,
        mode=str(args.iv_nonlinearity_mode),
        adc_full_scale_levels=float(args.iv_adc_full_scale_levels),
        linear_rows_before_clamp=int(args.iv_linear_rows_before_clamp),
    )
    tdc = CrossbarTDCConfig(
        enabled=True,
        precision_mode=args.tdc_precision_mode,
        gain_shift=int(args.tdc_gain_shift),
        output_format=str(args.tdc_output_format),
    )
    return {
        "enabled": bool(args.hardware_effects_audit),
        "seed": int(args.hardware_effect_seed),
        "row_tile": 64,
        "col_tile": 32,
        "row_groups": int(args.saturation_row_groups),
        "rows_per_group": int(args.saturation_rows_per_group),
        "reliability": reliability.as_dict(),
        "noise": noise.as_dict(),
        "iv": iv.as_dict(),
        "tdc": tdc.as_dict(),
        "branch_model": "zero_point_corrected_signed_branch",
        "note": (
            "Prompt-aware audit over the same W7/A7 activations and weights. "
            "It records the reliability, read-noise, IV/ADC, and TDC error/stress seen by the mapped software-crossbar reads."
        ),
    }


def hardware_empty_stats() -> dict[str, Any]:
    return {
        "reliability_programmed_cells": 0,
        "reliability_changed_cells": 0,
        "reliability_stuck_at_lrs_cells": 0,
        "reliability_stuck_at_hrs_cells": 0,
        "reliability_raw_ber_bit_flips": 0,
        "reliability_retention_drift_cells": 0,
        "reliability_state_spread_cells": 0,
        "effective_weight_changed_entries": 0,
        "reliability_error_samples": 0,
        "reliability_mean_abs_int_error": 0.0,
        "reliability_max_abs_int_error": 0.0,
        "noise_samples": 0,
        "noise_mean_branch_sigma_levels": 0.0,
        "noise_max_branch_sigma_levels": 0.0,
        "iv_transfer_calls": 0,
        "iv_upper_saturation_events": 0,
        "iv_lower_clip_events": 0,
        "iv_unique_saturated_tile_columns": 0,
        "iv_saturated_tiles": 0,
        "iv_observed_tile_columns": 0,
        "iv_max_branch_level": 0.0,
        "tdc_output_values": 0,
        "tdc_payload_bits_per_value": 0,
        "tdc_wire_bits": 0,
        "tdc_clipped_branch_conversions": 0,
        "tdc_mean_abs_int_error": 0.0,
        "tdc_max_abs_int_error": 0.0,
        "tdc_timing_samples": 0,
        "tdc_first_ready_column": -1,
        "tdc_first_ready_column_histogram": {},
        "tdc_first_ready_arrival_min_ps": 0.0,
        "tdc_first_ready_arrival_max_ps": 0.0,
        "tdc_arrival_spread_max_ps": 0.0,
    }


def add_weighted_mean(dst: dict[str, Any], mean_key: str, count_key: str, src: dict[str, Any]) -> None:
    dst_count = int(dst.get(count_key, 0))
    src_count = int(src.get(count_key, 0))
    if src_count <= 0:
        return
    old_total = float(dst.get(mean_key, 0.0)) * dst_count
    new_total = float(src.get(mean_key, 0.0)) * src_count
    dst[mean_key] = (old_total + new_total) / float(dst_count + src_count)


def tdc_first_ready_column_from_histogram(histogram: dict[str, Any]) -> int:
    """Return the most frequently first-ready logical column, with a stable tie-break."""

    if not histogram:
        return -1
    return int(
        min(
            histogram,
            key=lambda column: (-int(histogram[column]), int(column)),
        )
    )


def add_hardware_stats(dst: dict[str, Any], src: dict[str, Any]) -> None:
    add_weighted_mean(dst, "reliability_mean_abs_int_error", "reliability_error_samples", src)
    add_weighted_mean(dst, "tdc_mean_abs_int_error", "tdc_output_values", src)
    add_weighted_mean(dst, "noise_mean_branch_sigma_levels", "noise_samples", src)
    timing_samples = int(src.get("tdc_timing_samples", 0))
    if timing_samples > 0:
        dst_timing_samples = int(dst.get("tdc_timing_samples", 0))
        src_min = float(src.get("tdc_first_ready_arrival_min_ps", 0.0))
        src_max = float(src.get("tdc_first_ready_arrival_max_ps", 0.0))
        if dst_timing_samples <= 0:
            dst["tdc_first_ready_arrival_min_ps"] = src_min
        else:
            dst["tdc_first_ready_arrival_min_ps"] = min(
                float(dst.get("tdc_first_ready_arrival_min_ps", src_min)),
                src_min,
            )
        dst["tdc_first_ready_arrival_max_ps"] = max(
            float(dst.get("tdc_first_ready_arrival_max_ps", 0.0)),
            src_max,
        )
        dst["tdc_arrival_spread_max_ps"] = max(
            float(dst.get("tdc_arrival_spread_max_ps", 0.0)),
            float(src.get("tdc_arrival_spread_max_ps", 0.0)),
        )
        dst["tdc_timing_samples"] = dst_timing_samples + timing_samples

    dst_histogram = dst.setdefault("tdc_first_ready_column_histogram", {})
    if not isinstance(dst_histogram, dict):
        dst_histogram = {}
        dst["tdc_first_ready_column_histogram"] = dst_histogram
    src_histogram = src.get("tdc_first_ready_column_histogram", {})
    if not isinstance(src_histogram, dict):
        src_histogram = {}
    for column, count in src_histogram.items():
        column_key = str(column)
        dst_histogram[column_key] = int(dst_histogram.get(column_key, 0)) + int(count)
    dst["tdc_first_ready_column"] = tdc_first_ready_column_from_histogram(dst_histogram)
    for key in (
        "reliability_programmed_cells",
        "reliability_changed_cells",
        "reliability_stuck_at_lrs_cells",
        "reliability_stuck_at_hrs_cells",
        "reliability_raw_ber_bit_flips",
        "reliability_retention_drift_cells",
        "reliability_state_spread_cells",
        "effective_weight_changed_entries",
        "reliability_error_samples",
        "noise_samples",
        "iv_transfer_calls",
        "iv_upper_saturation_events",
        "iv_lower_clip_events",
        "iv_unique_saturated_tile_columns",
        "iv_saturated_tiles",
        "iv_observed_tile_columns",
        "tdc_output_values",
        "tdc_wire_bits",
        "tdc_clipped_branch_conversions",
    ):
        dst[key] = int(dst.get(key, 0)) + int(src.get(key, 0))
    for key in (
        "reliability_max_abs_int_error",
        "noise_max_branch_sigma_levels",
        "iv_max_branch_level",
        "tdc_max_abs_int_error",
        "tdc_payload_bits_per_value",
    ):
        dst[key] = max(float(dst.get(key, 0.0)), float(src.get(key, 0.0)))


def post_ecc_ber_probability(cfg: CrossbarReliabilityConfig, p_raw: float) -> float:
    n_bits = max(1, int(cfg.ecc_codeword_bits))
    data_bits = max(1, int(cfg.ecc_data_bits))
    p0 = (1.0 - p_raw) ** n_bits
    p1 = n_bits * p_raw * ((1.0 - p_raw) ** max(0, n_bits - 1))
    p_uncorrectable_word = min(1.0, max(0.0, 1.0 - p0 - p1))
    return min(1.0, p_uncorrectable_word / data_bits)


def effective_bit_error_probability(cfg: CrossbarReliabilityConfig) -> float:
    p_raw = max(0.0, float(cfg.raw_ber_ppm)) * 1.0e-6
    if p_raw <= 0.0:
        return 0.0
    if not cfg.use_ecc:
        return min(1.0, p_raw)
    return post_ecc_ber_probability(cfg, p_raw)


def state_log_resistance(noise_cfg: CrossbarNoiseConfig) -> np.ndarray:
    levels = np.arange(16, dtype=float)
    frac = levels / 15.0
    log_hrs = np.log(max(float(noise_cfg.r_hrs_ohm), 1.0e-30))
    log_lrs = np.log(max(float(noise_cfg.r_lrs_ohm), 1.0e-30))
    return log_hrs + frac * (log_lrs - log_hrs)


def log_resistance_to_levels(log_resistance: np.ndarray, state_log: np.ndarray) -> np.ndarray:
    diff = np.abs(np.asarray(log_resistance)[..., None] - state_log)
    return np.argmin(diff, axis=-1).astype(np.int16)


def retention_log_shift(levels: np.ndarray, cfg: CrossbarReliabilityConfig) -> np.ndarray:
    ref = max(float(cfg.retention_reference_s), 1.0e-12)
    t = max(float(cfg.retention_time_s), ref)
    decades = max(np.log10(t / ref), 0.0)
    frac_lrs = np.asarray(levels, dtype=float) / 15.0
    beta = (1.0 - frac_lrs) * float(cfg.retention_hrs_beta_log10_per_decade)
    beta += frac_lrs * float(cfg.retention_lrs_beta_log10_per_decade)
    return np.log(10.0) * beta * decades


def apply_reliability_to_levels(
    levels: np.ndarray,
    *,
    cfg: CrossbarReliabilityConfig,
    noise_cfg: CrossbarNoiseConfig,
    rng: np.random.Generator,
) -> tuple[np.ndarray, dict[str, int]]:
    out = np.asarray(levels, dtype=np.int16).copy()
    original = out.copy()
    stats = {
        "reliability_programmed_cells": int(out.size),
        "reliability_changed_cells": 0,
        "reliability_stuck_at_lrs_cells": 0,
        "reliability_stuck_at_hrs_cells": 0,
        "reliability_raw_ber_bit_flips": 0,
        "reliability_retention_drift_cells": 0,
        "reliability_state_spread_cells": 0,
    }
    if not cfg.enabled or out.size == 0:
        return out.astype(np.int16), stats

    programmable = np.ones(out.shape, dtype=bool)
    p_lrs = np.clip(float(cfg.stuck_at_lrs_prob), 0.0, 1.0)
    if p_lrs > 0.0:
        mask = rng.random(out.shape) < p_lrs
        out[mask] = 15
        programmable[mask] = False
        stats["reliability_stuck_at_lrs_cells"] += int(np.count_nonzero(mask))

    p_hrs = np.clip(float(cfg.stuck_at_hrs_prob), 0.0, 1.0)
    if p_hrs > 0.0:
        mask = (rng.random(out.shape) < p_hrs) & programmable
        out[mask] = 0
        programmable[mask] = False
        stats["reliability_stuck_at_hrs_cells"] += int(np.count_nonzero(mask))

    p_ber = effective_bit_error_probability(cfg)
    if p_ber > 0.0:
        for bit in range(4):
            flips = (rng.random(out.shape) < p_ber) & programmable
            out[flips] ^= 1 << bit
            stats["reliability_raw_ber_bit_flips"] += int(np.count_nonzero(flips))
        out = np.clip(out, 0, 15)

    if float(cfg.retention_time_s) > 0.0 and (
        float(cfg.retention_lrs_beta_log10_per_decade) != 0.0
        or float(cfg.retention_hrs_beta_log10_per_decade) != 0.0
    ):
        state_log = state_log_resistance(noise_cfg)
        before = out.copy()
        log_r = state_log[np.clip(out, 0, 15)]
        log_r = log_r + retention_log_shift(out, cfg)
        out = log_resistance_to_levels(log_r, state_log)
        stats["reliability_retention_drift_cells"] += int(np.count_nonzero(out != before))

    sigma = max(float(cfg.state_sigma_ln), 0.0)
    if sigma > 0.0:
        state_log = state_log_resistance(noise_cfg)
        before = out.copy()
        log_r = state_log[np.clip(out, 0, 15)] + rng.normal(0.0, sigma, size=out.shape)
        out = log_resistance_to_levels(log_r, state_log)
        stats["reliability_state_spread_cells"] += int(np.count_nonzero(out != before))

    out = np.clip(out, 0, 15).astype(np.int16)
    stats["reliability_changed_cells"] = int(np.count_nonzero(out != original))
    return out, stats


def decompose_signed_int8_levels(weights: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    w = np.asarray(weights, dtype=np.int16)
    w_pos = np.maximum(w, 0)
    w_neg = np.maximum(-w, 0)
    return (
        (w_pos & 0xF).astype(np.int16),
        ((w_pos >> 4) & 0xF).astype(np.int16),
        (w_neg & 0xF).astype(np.int16),
        ((w_neg >> 4) & 0xF).astype(np.int16),
    )


def reconstruct_signed_int8_from_levels(
    pos0: np.ndarray,
    pos1: np.ndarray,
    neg0: np.ndarray,
    neg1: np.ndarray,
) -> np.ndarray:
    return (
        np.asarray(pos0, dtype=np.int16)
        + (np.asarray(pos1, dtype=np.int16) << 4)
        - np.asarray(neg0, dtype=np.int16)
        - (np.asarray(neg1, dtype=np.int16) << 4)
    ).astype(np.int16)


def noise_sigma_lookup(noise_cfg: CrossbarNoiseConfig) -> np.ndarray:
    probe = CrossbarArray(input_size=1, output_size=2, noise_config=noise_cfg)
    return probe._read_noise_level_sigma(np.arange(16, dtype=np.int16)).astype(np.float64)


def tdc_sar_reconstruct_magnitude(values: np.ndarray, tdc_cfg: CrossbarTDCConfig) -> tuple[np.ndarray, int]:
    raw = np.asarray(values, dtype=np.float64)
    fine_bits = min(max(int(tdc_cfg.fine_bits), 0), max(int(tdc_cfg.tdc_bits), 1))
    coarse_budget = max(int(tdc_cfg.tdc_bits), 1) - fine_bits
    coarse_bits = min(max(int(tdc_cfg.coarse_bits), 1), max(coarse_budget, 1))
    max_coarse = (1 << coarse_bits) - 1
    mantissa_base = 1 << fine_bits
    max_mantissa = (1 << (fine_bits + 1)) - 1
    gain_shift = max(int(tdc_cfg.gain_shift), 0)
    scaled = np.maximum(raw, 0.0) / float(1 << gain_shift)
    active = scaled >= 0.5
    reconstructed = np.zeros(scaled.shape, dtype=np.float64)
    clipped_high_count = 0
    if not np.any(active):
        return reconstructed, clipped_high_count

    scaled_active = scaled[active]
    coarse = np.floor(np.log2(scaled_active)).astype(np.int32)
    coarse = np.maximum(coarse, 0)
    clipped_high = coarse > max_coarse
    coarse = np.minimum(coarse, max_coarse)
    step_exp = coarse - fine_bits
    mantissa = np.rint(np.ldexp(scaled_active, -step_exp)).astype(np.int32)
    mantissa = np.maximum(mantissa, mantissa_base)
    overflow = mantissa >= (1 << (fine_bits + 1))
    if np.any(overflow):
        can_increment = coarse < max_coarse
        inc = overflow & can_increment
        coarse[inc] += 1
        mantissa[inc] = mantissa_base
        clipped_high |= overflow & ~can_increment
    mantissa = np.minimum(mantissa, max_mantissa)
    mantissa[clipped_high] = max_mantissa
    exponent = coarse + gain_shift
    reconstructed_active = np.rint(np.ldexp(mantissa.astype(np.float64), exponent - fine_bits))
    reconstructed[active] = reconstructed_active
    clipped_high_count = int(np.count_nonzero(clipped_high))
    return reconstructed, clipped_high_count


def tdc_reconstruct_signed(diff: np.ndarray, tdc_cfg: CrossbarTDCConfig, full_scale: float) -> tuple[np.ndarray, int]:
    pos = np.clip(np.maximum(np.asarray(diff, dtype=np.float64), 0.0), 0.0, full_scale)
    neg = np.clip(np.maximum(-np.asarray(diff, dtype=np.float64), 0.0), 0.0, full_scale)
    if str(tdc_cfg.output_format).lower() == "sar_fp":
        pos_rec, pos_clip = tdc_sar_reconstruct_magnitude(pos, tdc_cfg)
        neg_rec, neg_clip = tdc_sar_reconstruct_magnitude(neg, tdc_cfg)
        return pos_rec - neg_rec, pos_clip + neg_clip
    max_code = (1 << max(int(tdc_cfg.tdc_bits), 1)) - 1
    scale = float(1 << max(int(tdc_cfg.gain_shift), 0))
    pos_code = np.clip(np.rint(pos / scale), 0, max_code)
    neg_code = np.clip(np.rint(neg / scale), 0, max_code)
    clipped = int(np.count_nonzero(pos_code >= max_code) + np.count_nonzero(neg_code >= max_code))
    return (pos_code - neg_code) * scale, clipped


def audit_quantized_hardware_effects(
    x_q: np.ndarray,
    weight_q: np.ndarray,
    *,
    phase: str,
    module: str,
    role: str,
    call_kind: str,
    batch_matrices: int,
    config: dict[str, Any],
    rng: np.random.Generator,
) -> dict[str, Any]:
    x_int = np.asarray(x_q, dtype=np.int16).reshape(-1, np.asarray(x_q).shape[-1])
    w_int = np.asarray(weight_q, dtype=np.int16)
    tokens = int(x_int.shape[0])
    in_dim = int(w_int.shape[0])
    out_dim = int(w_int.shape[1])
    row_tile = int(config.get("row_tile", 64))
    col_tile = int(config.get("col_tile", 32))
    row_tiles = int(math.ceil(in_dim / row_tile))
    col_tiles = int(math.ceil(out_dim / col_tile))
    ranges = row_parallel_ranges(row_tile, int(config.get("row_groups", 1)), int(config.get("rows_per_group", 0)))
    rel_cfg = CrossbarReliabilityConfig.from_obj(config["reliability"])
    noise_cfg = CrossbarNoiseConfig.from_obj(config["noise"])
    iv_cfg = CrossbarIVNonlinearityConfig.from_obj(config["iv"])
    tdc_cfg = CrossbarTDCConfig.from_obj(config["tdc"])
    full_scale = iv_full_scale_levels(iv_cfg.adc_full_scale_levels, iv_cfg.linear_rows_before_clamp)
    sigma_lookup = noise_sigma_lookup(noise_cfg)

    row = hardware_empty_stats()
    row.update(
        {
            "phase": phase,
            "module": module,
            "role": role,
            "call_kind": call_kind,
            "tokens": tokens,
            "batch_matrices": int(batch_matrices),
            "in_dim": in_dim,
            "out_dim": out_dim,
            "row_tiles": row_tiles,
            "col_tiles": col_tiles,
            "iv_full_scale_levels": full_scale,
            "tdc_payload_bits_per_value": int(tdc_cfg.signed_payload_bits),
        }
    )

    rel_abs_sum = 0.0
    rel_count = 0
    tdc_abs_sum = 0.0
    tdc_count = 0
    noise_sigma_sum = 0.0
    noise_count = 0

    for r_block in range(row_tiles):
        row_start = r_block * row_tile
        row_end = min(row_start + row_tile, in_dim)
        row_count = row_end - row_start
        x_tile = np.zeros((tokens, row_tile), dtype=np.int16)
        w_tile = np.zeros((row_tile, out_dim), dtype=np.int16)
        x_tile[:, :row_count] = x_int[:, row_start:row_end]
        w_tile[:row_count, :] = w_int[row_start:row_end, :]
        pos0, pos1, neg0, neg1 = decompose_signed_int8_levels(w_tile)
        reliable_levels = []
        for levels in (pos0, pos1, neg0, neg1):
            adjusted, stats = apply_reliability_to_levels(levels, cfg=rel_cfg, noise_cfg=noise_cfg, rng=rng)
            reliable_levels.append(adjusted)
            for key, value in stats.items():
                row[key] = int(row[key]) + int(value)
        rpos0, rpos1, rneg0, rneg1 = reliable_levels
        reliable_w = reconstruct_signed_int8_from_levels(rpos0, rpos1, rneg0, rneg1)
        row["effective_weight_changed_entries"] = int(row["effective_weight_changed_entries"]) + int(
            np.count_nonzero(reliable_w != w_tile)
        )

        tile_column_masks = [
            np.zeros((min((c_block + 1) * col_tile, out_dim) - c_block * col_tile,), dtype=bool)
            for c_block in range(col_tiles)
        ]
        tile_has_saturation = [False for _ in range(col_tiles)]
        reliable_total = np.zeros((tokens, out_dim), dtype=np.int64)
        tdc_total = np.zeros((tokens, out_dim), dtype=np.int64)

        for row_slice in ranges:
            ideal_part = x_tile[:, row_slice].astype(np.int32) @ w_tile[row_slice, :].astype(np.int32)
            reliable_part = x_tile[:, row_slice].astype(np.int32) @ reliable_w[row_slice, :].astype(np.int32)
            reliable_total += reliable_part.astype(np.int64)
            rel_err = reliable_part.astype(np.int64) - ideal_part.astype(np.int64)
            rel_abs = np.abs(rel_err)
            rel_abs_sum += float(np.sum(rel_abs))
            rel_count += int(rel_abs.size)
            if rel_abs.size:
                row["reliability_max_abs_int_error"] = max(
                    float(row["reliability_max_abs_int_error"]),
                    float(np.max(rel_abs)),
                )

            counts = (x_tile[:, row_slice].astype(np.int32) + 128).astype(np.int32)
            var = counts @ (sigma_lookup[rpos0[row_slice, :]] ** 2)
            var += 256.0 * (counts @ (sigma_lookup[rpos1[row_slice, :]] ** 2))
            var += counts @ (sigma_lookup[rneg0[row_slice, :]] ** 2)
            var += 256.0 * (counts @ (sigma_lookup[rneg1[row_slice, :]] ** 2))
            branch_sigma = np.sqrt(np.maximum(var, 0.0))
            if branch_sigma.size:
                noise_sigma_sum += float(np.sum(branch_sigma))
                noise_count += int(branch_sigma.size)
                row["noise_max_branch_sigma_levels"] = max(
                    float(row["noise_max_branch_sigma_levels"]),
                    float(np.max(branch_sigma)),
                )

            tdc_part, tdc_clips = tdc_reconstruct_signed(reliable_part, tdc_cfg, full_scale)
            tdc_total += np.rint(tdc_part).astype(np.int64)
            row["tdc_clipped_branch_conversions"] = int(row["tdc_clipped_branch_conversions"]) + int(tdc_clips)
            branch = np.abs(reliable_part.astype(np.float64))
            if bool(tdc_cfg.timing_enabled) and branch.ndim == 2 and branch.shape[1] > 0:
                # The audit mirrors the CrossbarArray timing model: larger
                # column magnitudes arrive earlier, while the slowest column
                # determines synchronous PE completion.
                timing_magnitude = np.clip(branch, 0.0, full_scale)
                timing_span_ps = float(tdc_cfg.timing_max_ps - tdc_cfg.timing_min_ps)
                arrival_time_ps = float(tdc_cfg.timing_min_ps) + timing_span_ps / (
                    1.0 + timing_magnitude / float(tdc_cfg.timing_reference_levels)
                )
                completion_order = np.argsort(arrival_time_ps, axis=1, kind="stable")
                first_columns = completion_order[:, 0]
                first_times = arrival_time_ps[np.arange(arrival_time_ps.shape[0]), first_columns]
                last_times = np.max(arrival_time_ps, axis=1)
                row["tdc_timing_samples"] = int(row["tdc_timing_samples"]) + int(first_columns.size)
                histogram = row["tdc_first_ready_column_histogram"]
                for column in first_columns.tolist():
                    column_key = str(int(column))
                    histogram[column_key] = int(histogram.get(column_key, 0)) + 1
                batch_min = float(np.min(first_times))
                batch_max = float(np.max(first_times))
                row["tdc_first_ready_arrival_min_ps"] = (
                    batch_min
                    if float(row["tdc_first_ready_arrival_min_ps"]) <= 0.0
                    else min(float(row["tdc_first_ready_arrival_min_ps"]), batch_min)
                )
                row["tdc_first_ready_arrival_max_ps"] = max(
                    float(row["tdc_first_ready_arrival_max_ps"]),
                    batch_max,
                )
                row["tdc_arrival_spread_max_ps"] = max(
                    float(row["tdc_arrival_spread_max_ps"]),
                    float(np.max(last_times - first_times)),
                )
            if branch.size:
                row["iv_max_branch_level"] = max(float(row["iv_max_branch_level"]), float(np.max(branch)))
            for c_block in range(col_tiles):
                col_start = c_block * col_tile
                col_end = min(col_start + col_tile, out_dim)
                upper = branch[:, col_start:col_end] > full_scale
                upper_count = int(np.count_nonzero(upper))
                row["iv_upper_saturation_events"] = int(row["iv_upper_saturation_events"]) + upper_count
                if upper_count:
                    tile_has_saturation[c_block] = True
                    tile_column_masks[c_block] |= np.any(upper, axis=0)

        tdc_err = tdc_total - reliable_total
        tdc_abs = np.abs(tdc_err)
        tdc_abs_sum += float(np.sum(tdc_abs))
        tdc_count += int(tdc_abs.size)
        if tdc_abs.size:
            row["tdc_max_abs_int_error"] = max(float(row["tdc_max_abs_int_error"]), float(np.max(tdc_abs)))

        for c_block, mask in enumerate(tile_column_masks):
            col_start = c_block * col_tile
            col_end = min(col_start + col_tile, out_dim)
            row["iv_observed_tile_columns"] = int(row["iv_observed_tile_columns"]) + (col_end - col_start)
            row["iv_unique_saturated_tile_columns"] = int(row["iv_unique_saturated_tile_columns"]) + int(
                np.count_nonzero(mask)
            )
            row["iv_saturated_tiles"] = int(row["iv_saturated_tiles"]) + int(tile_has_saturation[c_block])

    row["noise_samples"] = noise_count
    row["reliability_error_samples"] = rel_count
    row["noise_mean_branch_sigma_levels"] = noise_sigma_sum / float(noise_count) if noise_count else 0.0
    row["reliability_mean_abs_int_error"] = rel_abs_sum / float(rel_count) if rel_count else 0.0
    row["tdc_output_values"] = tdc_count
    row["tdc_mean_abs_int_error"] = tdc_abs_sum / float(tdc_count) if tdc_count else 0.0
    row["tdc_first_ready_column"] = tdc_first_ready_column_from_histogram(
        row["tdc_first_ready_column_histogram"]
    )
    row["iv_transfer_calls"] = int(tokens * row_tiles * col_tiles * len(ranges) * 2)
    row["tdc_wire_bits"] = int(tdc_count * int(tdc_cfg.signed_payload_bits))
    return row


def audit_float_hardware_effects(
    x_float: np.ndarray,
    weight_float: np.ndarray | None,
    *,
    weight_q: np.ndarray | None = None,
    phase: str,
    module: str,
    role: str,
    call_kind: str,
    weight_bits: int,
    act_bits: int,
    block_size: int,
    float_format: str,
    batch_matrices: int,
    config: dict[str, Any],
    rng: np.random.Generator,
) -> dict[str, Any]:
    x_arr = np.asarray(x_float, dtype=np.float32)
    x_2d = round_float_format(x_arr.reshape(-1, x_arr.shape[-1]), float_format)
    x_block = quantize_block16(x_2d, bits=act_bits, block_size=block_size)
    x_q, _x_scale = quantize_symmetric(x_block)
    if weight_q is None:
        if weight_float is None:
            raise ValueError("weight_float or weight_q is required for hardware effects audit.")
        w_arr = round_float_format(np.asarray(weight_float, dtype=np.float32), float_format)
        w_block = weight_block16(w_arr, bits=weight_bits, block_size=block_size) if weight_bits < 8 else w_arr
        weight_q, _w_scale = quantize_symmetric(w_block)
    return audit_quantized_hardware_effects(
        x_q,
        np.asarray(weight_q),
        phase=phase,
        module=module,
        role=role,
        call_kind=call_kind,
        batch_matrices=batch_matrices,
        config=config,
        rng=rng,
    )


def linear_hardware_effect_rows(
    selected_layer: nn.Module,
    captures: dict[str, dict[str, torch.Tensor]],
    *,
    weight_bits: int,
    act_bits: int,
    block_size: int,
    float_format: str,
    config: dict[str, Any],
) -> list[dict[str, Any]]:
    input_by_module = {
        "q_proj": "layer_00_input_layernorm_output",
        "k_proj": "layer_00_input_layernorm_output",
        "v_proj": "layer_00_input_layernorm_output",
        "o_proj": "layer_00_attn_context_before_o_proj",
        "gate_proj": "layer_00_post_attention_layernorm_output",
        "up_proj": "layer_00_post_attention_layernorm_output",
        "down_proj": "layer_00_mlp_gated_activation_before_down_proj",
    }
    rows: list[dict[str, Any]] = []
    seed = int(config.get("seed", 0))
    for phase in ("prefill", "decode"):
        for module_name, input_name in input_by_module.items():
            module = get_replaced_linear(selected_layer, module_name)
            weight_q = getattr(module, "weight_q", None)
            if weight_q is None and hasattr(module, "_executor"):
                weight_q = getattr(module._executor, "weight_q", None)
            if weight_q is None:
                continue
            role = LINEAR_ROLE_BY_MODULE[module_name]
            rng = np.random.default_rng(deterministic_seed("hardware", seed, phase, module_name, role))
            rows.append(
                audit_float_hardware_effects(
                    captures[phase][input_name].float().cpu().numpy(),
                    None,
                    weight_q=np.asarray(weight_q),
                    phase=phase,
                    module=module_name,
                    role=role,
                    call_kind="linear",
                    weight_bits=weight_bits,
                    act_bits=act_bits,
                    block_size=block_size,
                    float_format=float_format,
                    batch_matrices=1,
                    config=config,
                    rng=rng,
                )
            )
    return rows


def attention_hardware_effect_row(
    x: torch.Tensor,
    weight: torch.Tensor,
    *,
    phase: str,
    role: str,
    weight_bits: int,
    act_bits: int,
    block_size: int,
    float_format: str,
    config: dict[str, Any],
) -> dict[str, Any]:
    in_dim = int(x.shape[-1])
    out_dim = int(weight.shape[-1])
    prefix = tuple(int(v) for v in x.shape[:-2])
    tokens = int(x.shape[-2])
    batch_count = int(math.prod(prefix)) if prefix else 1
    x_np = x.detach().float().cpu().numpy().reshape(batch_count, tokens, in_dim)
    w_np_raw = weight.detach().float().cpu().numpy()
    if weight.ndim == 2:
        w_np = np.broadcast_to(w_np_raw.reshape(1, in_dim, out_dim), (batch_count, in_dim, out_dim))
    else:
        w_np = w_np_raw.reshape(batch_count, in_dim, out_dim)

    aggregate = hardware_empty_stats()
    for idx, (x_i, w_i) in enumerate(zip(x_np, w_np)):
        rng = np.random.default_rng(deterministic_seed("hardware", config.get("seed", 0), phase, role, idx))
        row = audit_float_hardware_effects(
            x_i,
            w_i,
            phase=phase,
            module=f"mapped_{role}",
            role=role,
            call_kind="mapped_attention_gemm",
            weight_bits=weight_bits,
            act_bits=act_bits,
            block_size=block_size,
            float_format=float_format,
            batch_matrices=1,
            config=config,
            rng=rng,
        )
        add_hardware_stats(aggregate, row)
    aggregate.update(
        {
            "phase": phase,
            "module": f"mapped_{role}",
            "role": role,
            "call_kind": "mapped_attention_gemm",
            "tokens": tokens,
            "batch_matrices": batch_count,
            "in_dim": in_dim,
            "out_dim": out_dim,
            "row_tiles": int(math.ceil(in_dim / int(config.get("row_tile", 64)))),
            "col_tiles": int(math.ceil(out_dim / int(config.get("col_tile", 32)))),
        }
    )
    return aggregate


def hardware_effect_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    summary = {"rows": len(rows), "by_phase": {}, "by_phase_role": {}, "total": hardware_empty_stats()}
    for row in rows:
        add_hardware_stats(summary["total"], row)
        phase = str(row["phase"])
        role_key = f"{phase}:{row['role']}"
        add_hardware_stats(summary["by_phase"].setdefault(phase, hardware_empty_stats()), row)
        add_hardware_stats(summary["by_phase_role"].setdefault(role_key, hardware_empty_stats()), row)
    return summary


def write_hardware_effects_markdown(
    path: Path,
    *,
    config: dict[str, Any],
    summary: dict[str, Any],
    rows: list[dict[str, Any]],
) -> None:
    total = summary["total"]
    lines = [
        "# Software Crossbar Hardware Effects Report",
        "",
        "This audit uses the same prompt/decode tensors and W7/A7 quantized weights as the BF16 value run.",
        "",
        "| Effect | What was enabled | What happened |",
        "|---|---|---|",
        (
            "| Reliability | 30-day retention drift, state spread, raw BER with ECC | "
            f"{total['reliability_programmed_cells']} cell writes audited; "
            f"{total['reliability_changed_cells']} cell levels changed; "
            f"{total['effective_weight_changed_entries']} INT8 weights changed |"
        ),
        (
            "| Noise | Thermal, shot, flicker, RTN, readout model at 5 kHz bandwidth | "
            f"mean branch sigma {float(total['noise_mean_branch_sigma_levels']):.6f} levels; "
            f"max branch sigma {float(total['noise_max_branch_sigma_levels']):.6f} levels |"
        ),
        (
            "| IV / ADC | hard-clamp branch full-scale before digitization | "
            f"{total['iv_upper_saturation_events']} saturation events; "
            f"{total['iv_unique_saturated_tile_columns']} clamped tile-columns; "
            f"{total['iv_saturated_tiles']} saturated tiles |"
        ),
        (
            "| TDC | SAR-FP TDC conversion | "
            f"{total['tdc_output_values']} output values converted; "
            f"{total['tdc_clipped_branch_conversions']} branch conversions clipped; "
            f"mean abs integer error {float(total['tdc_mean_abs_int_error']):.6f}; "
            f"max abs integer error {float(total['tdc_max_abs_int_error']):.0f} |"
        ),
        "",
        "## By Phase",
        "",
        "| Phase | Changed cells | Noise mean sigma | Saturation events | Clamped tile-columns | TDC clipped conversions | TDC mean abs int error |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for phase in sorted(summary["by_phase"]):
        row = summary["by_phase"][phase]
        lines.append(
            f"| `{phase}` | {row['reliability_changed_cells']} | "
            f"{float(row['noise_mean_branch_sigma_levels']):.6f} | "
            f"{row['iv_upper_saturation_events']} | {row['iv_unique_saturated_tile_columns']} | "
            f"{row['tdc_clipped_branch_conversions']} | "
            f"{float(row['tdc_mean_abs_int_error']):.6f} |"
        )
    lines.extend(
        [
            "",
            "## By Module",
            "",
            "| Phase | Role | Module / GEMM | Reliability mean / max int error | Noise mean / max sigma | IV saturation events | Clamped tile-columns | TDC mean / max int error |",
            "|---|---|---|---:|---:|---:|---:|---:|",
        ]
    )
    for row in rows:
        lines.append(
            f"| `{row['phase']}` | `{row['role']}` | `{row['module']}` | "
            f"{float(row['reliability_mean_abs_int_error']):.6f} / "
            f"{float(row['reliability_max_abs_int_error']):.0f} | "
            f"{float(row['noise_mean_branch_sigma_levels']):.6f} / "
            f"{float(row['noise_max_branch_sigma_levels']):.6f} | "
            f"{row['iv_upper_saturation_events']} | {row['iv_unique_saturated_tile_columns']} | "
            f"{float(row['tdc_mean_abs_int_error']):.6f} / {float(row['tdc_max_abs_int_error']):.0f} |"
        )
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def add_hooks(layer: torch.nn.Module, captures: dict[str, dict[str, torch.Tensor]], phase_ref: dict[str, str]) -> list[Any]:
    hooks = []

    def save(name: str, tensor: torch.Tensor) -> None:
        captures[phase_ref["phase"]][name] = capture_tensor(tensor)

    def pre_hook(name: str):
        def hook(_module: torch.nn.Module, inputs: tuple[Any, ...]) -> None:
            save(name, tensor_from_output(inputs[0]))

        return hook

    def out_hook(name: str):
        def hook(_module: torch.nn.Module, _inputs: tuple[Any, ...], output: Any) -> None:
            save(name, tensor_from_output(output))

        return hook

    hooks.append(layer.register_forward_pre_hook(pre_hook("layer_00_block_input")))
    hooks.append(layer.register_forward_hook(out_hook("layer_00_block_output")))

    input_norm = getattr(layer, "input_layernorm", None)
    post_norm = getattr(layer, "post_attention_layernorm", None)
    attn = getattr(layer, "self_attn", None) or getattr(layer, "attention", None)
    mlp = getattr(layer, "mlp", None)
    if input_norm is None or post_norm is None or attn is None or mlp is None:
        raise RuntimeError("Qwen decoder layer is missing expected block modules.")

    hooks.append(input_norm.register_forward_hook(out_hook("layer_00_input_layernorm_output")))
    hooks.append(post_norm.register_forward_hook(out_hook("layer_00_post_attention_layernorm_output")))

    for attr, name in [
        ("q_proj", "layer_00_q_proj_output"),
        ("k_proj", "layer_00_k_proj_output"),
        ("v_proj", "layer_00_v_proj_output"),
        ("o_proj", "layer_00_o_proj_output"),
    ]:
        module = getattr(attn, attr, None)
        if module is None:
            raise RuntimeError(f"Attention block is missing {attr}.")
        hooks.append(module.register_forward_hook(out_hook(name)))
    hooks.append(attn.o_proj.register_forward_pre_hook(pre_hook("layer_00_attn_context_before_o_proj")))

    for attr, name in [
        ("gate_proj", "layer_00_mlp_gate_proj_output"),
        ("up_proj", "layer_00_mlp_up_proj_output"),
        ("down_proj", "layer_00_mlp_down_proj_output"),
    ]:
        module = getattr(mlp, attr, None)
        if module is None:
            raise RuntimeError(f"MLP block is missing {attr}.")
        hooks.append(module.register_forward_hook(out_hook(name)))
    hooks.append(mlp.down_proj.register_forward_pre_hook(pre_hook("layer_00_mlp_gated_activation_before_down_proj")))

    return hooks


class IdealIntCrossbarLinear(nn.Module):
    """Fast ideal W/A-quantized crossbar MAC path with BF16 boundary rounding."""

    def __init__(
        self,
        linear: nn.Linear,
        *,
        weight_bits: int,
        act_bits: int,
        block_size: int,
        float_format: str,
    ):
        super().__init__()
        self.in_features = int(linear.in_features)
        self.out_features = int(linear.out_features)
        self.weight_bits = int(weight_bits)
        self.act_bits = int(act_bits)
        self.block_size = int(block_size)
        self.float_format = normalize_float_format(float_format)
        w_in_out = linear.weight.detach().float().cpu().numpy().T
        w_in_out = round_float_format(w_in_out, self.float_format)
        if self.weight_bits < 8:
            w_in_out = weight_block16(w_in_out, bits=self.weight_bits, block_size=self.block_size)
        self.weight_q, self.weight_scale = quantize_symmetric(w_in_out)
        self.bias = None if linear.bias is None else linear.bias.detach().float().cpu().numpy()

    @property
    def weight_elements(self) -> int:
        return self.in_features * self.out_features

    @property
    def physical_cell_banks_per_weight(self) -> int:
        return 4

    @property
    def programmed_cells(self) -> int:
        return self.weight_elements * self.physical_cell_banks_per_weight

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        device = x.device
        orig_shape = tuple(x.shape[:-1])
        x_np = x.detach().float().cpu().numpy().reshape(-1, self.in_features)
        x_np = round_float_format(x_np, self.float_format)
        x_block = quantize_block16(x_np, bits=self.act_bits, block_size=self.block_size)
        x_q, x_scale = quantize_symmetric(x_block)
        y_int = x_q.astype(np.int32) @ self.weight_q.astype(np.int32)
        y = y_int.astype(np.float32) * x_scale * self.weight_scale
        if self.bias is not None:
            y = y + self.bias.reshape(1, -1)
        y = round_float_format(y.reshape(*orig_shape, self.out_features), self.float_format)
        return torch.from_numpy(y.astype(np.float32, copy=False)).to(device)


def crossbar_matmul(
    x: torch.Tensor,
    weight: torch.Tensor,
    *,
    backend: str,
    weight_bits: int,
    act_bits: int,
    block_size: int,
    float_format: str,
) -> torch.Tensor:
    if backend not in {"wrapper", "ideal_int"}:
        raise ValueError(f"Unsupported backend {backend!r}")
    if x.ndim < 2 or weight.ndim < 2:
        raise ValueError("mapped crossbar matmul expects x[..., tokens, in_dim] and weight[..., in_dim, out_dim].")
    if x.shape[-1] != weight.shape[-2]:
        raise ValueError(f"matmul shape mismatch: x last dim {x.shape[-1]} vs weight in dim {weight.shape[-2]}")
    device = x.device
    in_dim = int(x.shape[-1])
    out_dim = int(weight.shape[-1])
    prefix = tuple(int(v) for v in x.shape[:-2])
    tokens = int(x.shape[-2])
    batch_count = int(math.prod(prefix)) if prefix else 1
    x_np = x.detach().float().cpu().numpy().reshape(batch_count, tokens, in_dim)
    w_np_raw = weight.detach().float().cpu().numpy()
    if weight.ndim == 2:
        w_np = np.broadcast_to(w_np_raw.reshape(1, in_dim, out_dim), (batch_count, in_dim, out_dim))
    else:
        weight_prefix = tuple(int(v) for v in weight.shape[:-2])
        if weight_prefix != prefix:
            raise ValueError(f"batched weight prefix {weight_prefix} does not match x prefix {prefix}.")
        w_np = w_np_raw.reshape(batch_count, in_dim, out_dim)

    outputs = []
    for x_i, w_i in zip(x_np, w_np):
        if backend == "wrapper":
            crossbar = CrossbarLinear(
                w_i, weight_bits=weight_bits, act_bits=act_bits,
                block_size=block_size, float_format=float_format,
                noise_config=CrossbarNoiseConfig(enabled=False),
                reliability_config=CrossbarReliabilityConfig(enabled=False),
                iv_config=CrossbarIVNonlinearityConfig(enabled=False),
                tdc_config=CrossbarTDCConfig(enabled=False),
            )
            outputs.append(crossbar(torch.from_numpy(x_i)).numpy())
            continue
        x_i = round_float_format(x_i, float_format)
        w_i = round_float_format(w_i, float_format)
        x_block = quantize_block16(x_i, bits=act_bits, block_size=block_size)
        w_block = weight_block16(w_i, bits=weight_bits, block_size=block_size) if weight_bits < 8 else w_i
        x_q, x_scale = quantize_symmetric(x_block)
        w_q, w_scale = quantize_symmetric(w_block)
        y_int = x_q.astype(np.int32) @ w_q.astype(np.int32)
        outputs.append(y_int.astype(np.float32) * x_scale * w_scale)
    y = np.stack(outputs, axis=0).reshape(*prefix, tokens, out_dim)
    y = round_float_format(y, float_format)
    return torch.from_numpy(y.astype(np.float32, copy=False)).to(device)


class MappingBackedAttentionGemms:
    def __init__(
        self,
        mapping_ctx: dict[str, Any],
        *,
        backend: str,
        weight_bits: int,
        act_bits: int,
        block_size: int,
        float_format: str,
        saturation_config: dict[str, Any] | None = None,
        hardware_effect_config: dict[str, Any] | None = None,
    ):
        self.mapping_ctx = mapping_ctx
        self.backend = backend
        self.weight_bits = int(weight_bits)
        self.act_bits = int(act_bits)
        self.block_size = int(block_size)
        self.float_format = normalize_float_format(float_format)
        self.saturation_config = saturation_config or {"enabled": False}
        self.hardware_effect_config = hardware_effect_config or {"enabled": False}
        self.records: list[dict[str, Any]] = []

    def layer(self, phase: str, role: str) -> LayerSpec:
        return self.mapping_ctx[f"{phase}_by_role"][role]

    def matmul(
        self,
        phase: str,
        role: str,
        x: torch.Tensor,
        weight: torch.Tensor,
    ) -> torch.Tensor:
        layer = self.layer(phase, role)
        in_dim = int(x.shape[-1])
        out_dim = int(weight.shape[-1])
        batch_heads = int(math.prod(x.shape[:-2])) if x.ndim >= 3 else 1
        tokens = int(x.shape[-2]) if x.ndim >= 2 else 1
        if role == "attn_scores_qkt":
            if in_dim != layer.in_dim:
                raise ValueError(f"{phase} QK^T in_dim {in_dim} does not match mapping C {layer.in_dim}.")
            if out_dim > layer.out_dim:
                raise ValueError(f"{phase} QK^T context {out_dim} exceeds mapping K {layer.out_dim}.")
            if tokens > layer.tokens:
                raise ValueError(f"{phase} QK^T query tokens {tokens} exceeds mapping token capacity {layer.tokens}.")
        elif role == "attn_av":
            if in_dim > layer.in_dim:
                raise ValueError(f"{phase} AV context {in_dim} exceeds mapping C {layer.in_dim}.")
            if out_dim != layer.out_dim:
                raise ValueError(f"{phase} AV out_dim {out_dim} does not match mapping K {layer.out_dim}.")
            if tokens > layer.tokens:
                raise ValueError(f"{phase} AV query tokens {tokens} exceeds mapping token capacity {layer.tokens}.")
        else:
            raise ValueError(f"Unsupported mapped attention role {role!r}.")
        y = crossbar_matmul(
            x,
            weight,
            backend=self.backend,
            weight_bits=self.weight_bits,
            act_bits=self.act_bits,
            block_size=self.block_size,
            float_format=self.float_format,
        )
        record = {
            "phase": phase,
            "crossbar_backend": self.backend,
            "mapped_role": role,
            "mapping_layer": layer.layer_name,
            "mapping_in_dim_capacity": layer.in_dim,
            "mapping_out_dim_capacity": layer.out_dim,
            "mapping_token_capacity": layer.tokens,
            "actual_batch_head_matrices": batch_heads,
            "actual_tokens": tokens,
            "actual_in_dim": in_dim,
            "actual_out_dim": out_dim,
            "mapping_row_tiles": layer.row_tiles,
            "mapping_col_tiles": layer.col_tiles,
            "mapping_crossbar_tiles_per_matrix": layer.crossbar_tiles,
            "actual_crossbar_tiles_per_matrix": math.ceil(in_dim / 64) * math.ceil(out_dim / 32),
            "actual_macs": batch_heads * tokens * in_dim * out_dim,
            "weight_bits": self.weight_bits,
            "act_bits": self.act_bits,
            "float_format": self.float_format,
        }
        if bool(self.saturation_config.get("enabled", False)):
            sat = audit_batched_attention_saturation(
                x,
                weight,
                phase=phase,
                role=role,
                weight_bits=self.weight_bits,
                act_bits=self.act_bits,
                block_size=self.block_size,
                float_format=self.float_format,
                config=self.saturation_config,
            )
            for key, value in sat.items():
                if key.startswith("iv_"):
                    record[key] = value
        if bool(self.hardware_effect_config.get("enabled", False)):
            hardware = attention_hardware_effect_row(
                x,
                weight,
                phase=phase,
                role=role,
                weight_bits=self.weight_bits,
                act_bits=self.act_bits,
                block_size=self.block_size,
                float_format=self.float_format,
                config=self.hardware_effect_config,
            )
            for key, value in hardware.items():
                if key in hardware_empty_stats():
                    record[f"hardware_{key}"] = value
        self.records.append(record)
        return y


def load_model(model_path: str, dtype: torch.dtype) -> torch.nn.Module:
    kwargs = {
        "low_cpu_mem_usage": True,
        "trust_remote_code": True,
        "local_files_only": True,
    }
    try:
        return AutoModelForCausalLM.from_pretrained(model_path, dtype=dtype, **kwargs)
    except TypeError:
        return AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=dtype, **kwargs)


def resolve_local_model_dir(local_root: str | Path, model: str) -> Path:
    """Accept either a direct local model directory or a model ID under local_root."""

    model_path = Path(str(model)).expanduser()
    if model_path.exists():
        return model_path.resolve()
    return (Path(local_root).expanduser() / str(model).replace("/", "--")).resolve()


def replace_linear(
    parent: nn.Module,
    attr: str,
    *,
    mapping_ctx: dict[str, Any],
    backend: str,
    weight_bits: int,
    act_bits: int,
    block_size: int,
    float_format: str,
    defer_programming: bool = False,
) -> dict[str, Any]:
    original = getattr(parent, attr)
    if not isinstance(original, nn.Linear):
        raise TypeError(f"{parent.__class__.__name__}.{attr} is not nn.Linear: {type(original)!r}")
    mapped_role = LINEAR_ROLE_BY_MODULE[attr]
    prefill_layer = mapping_ctx["prefill_by_role"][mapped_role]
    decode_layer = mapping_ctx["decode_by_role"][mapped_role]
    for phase, layer in (("prefill", prefill_layer), ("decode", decode_layer)):
        if layer.in_dim != int(original.in_features) or layer.out_dim != int(original.out_features):
            raise ValueError(
                f"{phase} mapping {layer.layer_name} ({layer.in_dim}->{layer.out_dim}) does not match "
                f"{attr} ({original.in_features}->{original.out_features})."
            )
    if backend == "ideal_int":
        replacement = IdealIntCrossbarLinear(
            original,
            weight_bits=weight_bits,
            act_bits=act_bits,
            block_size=block_size,
            float_format=float_format,
        )
        programmed_cells = replacement.programmed_cells
        weight_elements = replacement.weight_elements
    elif backend == "wrapper":
        replacement = CrossbarLinear.from_linear(
            original,
            weight_bits=weight_bits,
            act_bits=act_bits,
            block_size=block_size,
            noise_config=CrossbarNoiseConfig(enabled=False),
            reliability_config=CrossbarReliabilityConfig(enabled=False),
            iv_config=CrossbarIVNonlinearityConfig(enabled=False),
            tdc_config=CrossbarTDCConfig(enabled=False),
            float_format=float_format,
            defer_programming=defer_programming,
        )
        programmed_cells = original.in_features * original.out_features * 4
        weight_elements = original.in_features * original.out_features
    else:
        raise ValueError(f"Unsupported backend {backend!r}")
    setattr(parent, attr, replacement)
    return {
        "module": attr,
        "crossbar_backend": backend,
        "in_features": int(original.in_features),
        "out_features": int(original.out_features),
        "weight_elements": int(weight_elements),
        "programmed_cells": int(programmed_cells),
        "mapped_role": mapped_role,
        "prefill_mapping_layer": prefill_layer.layer_name,
        "decode_mapping_layer": decode_layer.layer_name,
        "prefill_mapping_token_capacity": prefill_layer.tokens,
        "decode_mapping_token_capacity": decode_layer.tokens,
        "mapping_row_tiles": prefill_layer.row_tiles,
        "mapping_col_tiles": prefill_layer.col_tiles,
        "mapping_crossbar_tiles": prefill_layer.crossbar_tiles,
    }


def replace_block_linears(
    layer: nn.Module,
    *,
    mapping_ctx: dict[str, Any],
    backend: str,
    weight_bits: int,
    act_bits: int,
    block_size: int,
    float_format: str,
    defer_programming: bool = False,
) -> list[dict[str, Any]]:
    attn = getattr(layer, "self_attn", None) or getattr(layer, "attention", None)
    mlp = getattr(layer, "mlp", None)
    if attn is None or mlp is None:
        raise RuntimeError("Qwen decoder layer is missing self_attn or mlp.")
    rows = []
    for attr in ("q_proj", "k_proj", "v_proj", "o_proj"):
        row = replace_linear(
            attn,
            attr,
            mapping_ctx=mapping_ctx,
            backend=backend,
            weight_bits=weight_bits,
            act_bits=act_bits,
            block_size=block_size,
            float_format=float_format,
            defer_programming=defer_programming,
        )
        row["scope"] = "self_attn"
        rows.append(row)
    for attr in ("gate_proj", "up_proj", "down_proj"):
        row = replace_linear(
            mlp,
            attr,
            mapping_ctx=mapping_ctx,
            backend=backend,
            weight_bits=weight_bits,
            act_bits=act_bits,
            block_size=block_size,
            float_format=float_format,
            defer_programming=defer_programming,
        )
        row["scope"] = "mlp"
        rows.append(row)
    return rows


def install_mapped_attention_gemms(
    layer: nn.Module,
    *,
    mapping_ctx: dict[str, Any],
    phase_ref: dict[str, str],
    backend: str,
    weight_bits: int,
    act_bits: int,
    block_size: int,
    float_format: str,
    saturation_config: dict[str, Any] | None = None,
    hardware_effect_config: dict[str, Any] | None = None,
) -> MappingBackedAttentionGemms:
    attn = getattr(layer, "self_attn", None) or getattr(layer, "attention", None)
    if attn is None:
        raise RuntimeError("Qwen decoder layer is missing self_attn/attention.")
    runner = MappingBackedAttentionGemms(
        mapping_ctx,
        backend=backend,
        weight_bits=weight_bits,
        act_bits=act_bits,
        block_size=block_size,
        float_format=float_format,
        saturation_config=saturation_config,
        hardware_effect_config=hardware_effect_config,
    )

    def mapped_forward(
        self: nn.Module,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor] | None = None,
        attention_mask: torch.Tensor | None = None,
        past_key_value: Any = None,
        cache_position: torch.Tensor | None = None,
        **kwargs: Any,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        if position_embeddings is None:
            raise RuntimeError("Mapped Qwen attention requires position_embeddings from the decoder backbone.")
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)

        query_states = self.q_norm(self.q_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
        key_states = self.k_norm(self.k_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        if past_key_value is None:
            past_key_value = kwargs.get("past_key_values")
        if past_key_value is not None:
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)

        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)
        phase = phase_ref["phase"]
        attn_weights = runner.matmul(
            phase,
            "attn_scores_qkt",
            query_states,
            key_states.transpose(2, 3).contiguous(),
        )
        attn_weights = attn_weights * self.scaling
        if attention_mask is not None:
            causal_mask = attention_mask[:, :, :, : key_states.shape[-2]]
            attn_weights = attn_weights + causal_mask

        attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        attn_weights = F.dropout(
            attn_weights,
            p=0.0 if not self.training else self.attention_dropout,
            training=self.training,
        )
        attn_output = runner.matmul(phase, "attn_av", attn_weights, value_states)
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        attn_output = self.o_proj(attn_output)
        return attn_output, attn_weights if kwargs.get("output_attentions", False) else None

    attn.forward = types.MethodType(mapped_forward, attn)
    return runner


def install_block_tile_lifecycle(
    layer: nn.Module,
    *,
    block_index: int,
    phase_ref: dict[str, str],
    residency: dict[str, int],
) -> list[Any]:
    executors = [module._executor for module in layer.modules() if isinstance(module, CrossbarLinear)]
    active = False

    def release(module: nn.Module, inputs: Any, output: Any) -> None:
        nonlocal active
        for executor in executors:
            executor.release_tiles()
        if active:
            residency["resident_blocks"] -= 1
            residency["block_release_events"] += 1
            active = False

    def program(module: nn.Module, inputs: Any) -> None:
        nonlocal active
        if active or residency["resident_blocks"]:
            raise RuntimeError("Block-streamed crossbars require sequential, non-reentrant execution.")
        active = True
        residency["resident_blocks"] += 1
        residency["peak_resident_blocks"] = max(residency["peak_resident_blocks"], residency["resident_blocks"])
        residency["block_program_events"] += 1
        try:
            for executor in executors:
                executor.program_tiles()
        except BaseException:
            release(module, inputs, None)
            raise
        print(f"crossbar_block_active={block_index + 1} phase={phase_ref['phase']}", flush=True)

    return [layer.register_forward_pre_hook(program), layer.register_forward_hook(release, always_call=True)]


def prepare_crossbar_blocks(
    model: nn.Module,
    *,
    prefill_path: Path,
    decode_path: Path,
    block_index: int | None,
    phase_ref: dict[str, str],
    backend: str,
    weight_bits: int,
    act_bits: int,
    block_size: int,
    float_format: str,
    mapped_attention: bool = True,
    saturation_config: dict[str, Any] | None = None,
    hardware_effect_config: dict[str, Any] | None = None,
    crossbar_execution: str = "eager",
) -> list[dict[str, Any]]:
    if crossbar_execution not in {"block_streamed", "eager"}:
        raise ValueError(f"Unsupported crossbar execution mode: {crossbar_execution!r}")
    streaming = backend == "wrapper" and crossbar_execution == "block_streamed"
    residency = {"resident_blocks": 0, "peak_resident_blocks": 0,
                 "block_program_events": 0, "block_release_events": 0}
    base, layers_attr, layers = base_model_and_layers(model)
    indices = list(range(len(layers))) if block_index is None else [block_index]
    if not indices or any(index < 0 or index >= len(layers) for index in indices):
        raise ValueError(f"Block selection must be within [0, {len(layers) - 1}].")
    blocks = [
        {
            "index": index,
            "layer": layers[index],
            "mapping_ctx": load_mapping_context(prefill_path, decode_path, index),
        }
        for index in indices
    ]
    if block_index is not None:
        setattr(base, layers_attr, nn.ModuleList([layers[block_index]]))
        layers[block_index].self_attn.layer_idx = 0
    for block in blocks:
        block["linear_rows"] = replace_block_linears(
            block["layer"], mapping_ctx=block["mapping_ctx"], backend=backend,
            weight_bits=weight_bits, act_bits=act_bits, block_size=block_size,
            float_format=float_format, defer_programming=streaming,
        )
        for row in block["linear_rows"]:
            row["block_index"] = block["index"]
        block["attention_runner"] = None
        if mapped_attention:
            block["attention_runner"] = install_mapped_attention_gemms(
                block["layer"], mapping_ctx=block["mapping_ctx"], phase_ref=phase_ref,
                backend=backend,
                weight_bits=weight_bits, act_bits=act_bits, block_size=block_size,
                float_format=float_format, saturation_config=saturation_config,
                hardware_effect_config=hardware_effect_config,
            )
        block["tile_lifecycle"] = residency if streaming else None
        block["lifecycle_hooks"] = install_block_tile_lifecycle(
            block["layer"], block_index=block["index"], phase_ref=phase_ref, residency=residency,
        ) if streaming else []
        print(f"crossbar_block_ready={block['index'] + 1}/{len(layers)}", flush=True)
    return blocks


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def build_endurance_report(
    *,
    metadata: dict[str, Any],
    linear_rows: list[dict[str, Any]],
    endurance_cycles: int,
    static_program_cycles: int,
    dynamic_kv_endurance_cycles: int,
    token_time_s: float,
    service_life_years: float,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    prefill_tokens = int(metadata["prefill_input_shape"][1])
    decode_tokens = int(metadata["decode_input_shape"][1])
    context_slots = max(prefill_tokens + decode_tokens, 1)
    service_life_s = max(service_life_years, 0.0) * SECONDS_PER_YEAR
    token_rate = 1.0 / max(token_time_s, 1.0e-30)
    projected_decode_tokens = service_life_s * token_rate
    projected_kv_cycles_per_slot = projected_decode_tokens / context_slots
    kv_margin = (
        float(dynamic_kv_endurance_cycles) / projected_kv_cycles_per_slot
        if dynamic_kv_endurance_cycles > 0 and projected_kv_cycles_per_slot > 0.0
        else None
    )
    static_weight_elements = sum(int(row["weight_elements"]) for row in linear_rows)
    static_programmed_cells = sum(int(row["programmed_cells"]) for row in linear_rows)
    static_margin = (
        float(endurance_cycles) / float(static_program_cycles)
        if endurance_cycles > 0 and static_program_cycles > 0
        else None
    )
    kv_dim = sum(int(row["out_features"]) for row in linear_rows if row["module"] == "k_proj")
    dynamic_kv_elements_observed = (prefill_tokens + decode_tokens) * kv_dim * 2
    rows = [
        {
            "category": "static_transformer_block_linears",
            "write_policy": "program_once_then_read_for_inference",
            "programmed_elements": static_weight_elements,
            "programmed_cells": static_programmed_cells,
            "observed_or_projected_cycles_per_cell": static_program_cycles,
            "endurance_cycles": endurance_cycles,
            "margin": static_margin,
            "status": "meets_inference_programming" if static_margin is not None and static_margin >= 1.0 else "below_target",
        },
        {
            "category": "dynamic_kv_cache_for_executed_blocks",
            "write_policy": "append_only_new_token",
            "programmed_elements": dynamic_kv_elements_observed,
            "programmed_cells": dynamic_kv_elements_observed,
            "observed_or_projected_cycles_per_cell": projected_kv_cycles_per_slot,
            "endurance_cycles": dynamic_kv_endurance_cycles,
            "margin": kv_margin,
            "status": "meets_projection" if kv_margin is not None and kv_margin >= 1.0 else "below_projection",
        },
    ]
    summary = {
        "static_weight_elements": static_weight_elements,
        "static_programmed_cells": static_programmed_cells,
        "static_program_cycles": static_program_cycles,
        "static_endurance_cycles": endurance_cycles,
        "static_margin": static_margin,
        "dynamic_kv_elements_observed": dynamic_kv_elements_observed,
        "dynamic_kv_endurance_cycles": dynamic_kv_endurance_cycles,
        "kv_cache_context_slots": context_slots,
        "kv_cache_token_time_s": token_time_s,
        "kv_cache_service_life_years": service_life_years,
        "projected_decode_tokens_per_service_life": projected_decode_tokens,
        "projected_kv_write_cycles_per_slot": projected_kv_cycles_per_slot,
        "kv_margin": kv_margin,
    }
    return summary, rows


def write_endurance_markdown(path: Path, summary: dict[str, Any], rows: list[dict[str, Any]]) -> None:
    lines = [
        "# Software Crossbar Endurance Report",
        "",
        f"- Static programmed cells: {summary['static_programmed_cells']}",
        f"- Static endurance margin: {summary['static_margin']}",
        f"- Dynamic K/V projected cycles per slot: {summary['projected_kv_write_cycles_per_slot']}",
        f"- Dynamic K/V endurance margin: {summary['kv_margin']}",
        "",
        "| Category | Status | Programmed cells | Cycles/cell | Endurance cycles | Margin |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            "| {category} | {status} | {cells} | {cycles} | {endurance} | {margin} |".format(
                category=row["category"],
                status=row["status"],
                cells=row["programmed_cells"],
                cycles=row["observed_or_projected_cycles_per_cell"],
                endurance=row["endurance_cycles"],
                margin=row["margin"],
            )
        )
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def load_gpu_metadata(path: Path | None) -> dict[str, Any]:
    if path is None:
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def run(args: argparse.Namespace) -> None:
    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    gpu_metadata = load_gpu_metadata(args.gpu_metadata)
    if gpu_metadata:
        args.prompt = gpu_metadata.get("prefill_prompt", args.prompt)
        args.decode_text = gpu_metadata.get("decode_text", args.decode_text)
    mapping_ctx = load_mapping_context(args.prefill_mapping, args.decode_mapping, args.block_index)

    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested, but torch.cuda.is_available() is False.")

    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    dtype = dtype_from_name(args.dtype)
    float_format = "fp32" if args.disable_bf16_boundary else "bf16"
    saturation_config = saturation_config_from_args(args)
    hardware_effect_config = hardware_effect_configs_from_args(args)

    model_local_dir = resolve_local_model_dir(args.local_root, args.model)
    if not model_local_dir.exists():
        raise FileNotFoundError(f"Local model directory not found: {model_local_dir}")

    print(f"run_name={args.run_name}", flush=True)
    print(f"model={args.model}", flush=True)
    print(f"local_model_dir={model_local_dir}", flush=True)
    print(f"out_dir={out_dir}", flush=True)
    print(f"crossbar_backend={args.crossbar_backend}", flush=True)
    print(f"prefill_mapping={mapping_ctx['prefill_path']}", flush=True)
    print(f"decode_mapping={mapping_ctx['decode_path']}", flush=True)

    tokenizer = AutoTokenizer.from_pretrained(str(model_local_dir), trust_remote_code=True, local_files_only=True)
    model = load_model(str(model_local_dir), dtype=dtype)
    model.eval()

    base, layers_attr, layers = base_model_and_layers(model)
    original_layer_count = len(layers)
    if args.block_index < 0 or args.block_index >= original_layer_count:
        raise ValueError(f"--block-index must be in [0, {original_layer_count - 1}], got {args.block_index}")

    encoded = tokenizer(args.prompt, return_tensors="pt")
    if encoded["input_ids"].shape[1] > args.max_input_tokens:
        encoded = {key: value[:, : args.max_input_tokens] for key, value in encoded.items()}
    decode_ids = tokenizer(args.decode_text, add_special_tokens=False, return_tensors="pt")["input_ids"]
    if decode_ids.numel() < 1:
        raise RuntimeError("--decode-text did not produce any tokens.")
    decode_ids = decode_ids[:, :1]

    if gpu_metadata:
        expected_prefill = gpu_metadata.get("prefill_input_ids")
        expected_decode = gpu_metadata.get("decode_input_ids")
        if expected_prefill is not None and encoded["input_ids"].tolist() != expected_prefill:
            raise RuntimeError("Tokenized prefill IDs do not match --gpu-metadata prefill_input_ids.")
        if expected_decode is not None and decode_ids.tolist() != expected_decode:
            raise RuntimeError("Tokenized decode IDs do not match --gpu-metadata decode_input_ids.")

    phase_ref = {"phase": "prefill"}
    blocks = prepare_crossbar_blocks(
        model,
        prefill_path=args.prefill_mapping,
        decode_path=args.decode_mapping,
        block_index=None if args.all_blocks else args.block_index,
        phase_ref=phase_ref,
        backend=args.crossbar_backend,
        weight_bits=args.weight_bits,
        act_bits=args.act_bits,
        block_size=args.block_size,
        float_format=float_format,
        mapped_attention=not args.disable_mapped_attention_gemms,
        saturation_config=saturation_config,
        hardware_effect_config=hardware_effect_config,
        crossbar_execution=args.crossbar_execution,
    )
    mapping_validation_rows = []
    for block in blocks:
        checks = validate_sequence_mapping_capacity(
            block["mapping_ctx"], prefill_tokens=int(encoded["input_ids"].shape[1]),
            decode_tokens=int(decode_ids.shape[1]),
        )
        mapping_validation_rows.extend(dict(check, block_index=block["index"]) for check in checks)
    mapping_ctx["selected_layers"] = [layer for block in blocks for layer in block["mapping_ctx"]["selected_layers"]]
    mapping_ctx["summary"] = summarize_layers(mapping_ctx["selected_layers"])
    write_mapping_artifacts(out_dir, mapping_ctx, mapping_validation_rows)
    linear_rows = [row for block in blocks for row in block["linear_rows"]]
    model.to(device)

    encoded = {key: value.to(device) for key, value in encoded.items()}
    decode_ids = decode_ids.to(device)

    hooks = []
    for block in blocks:
        block["captures"] = {"prefill": {}, "decode": {}}
        hooks.extend(add_hooks(block["layer"], block["captures"], phase_ref))

    try:
        t0 = time.perf_counter()
        with torch.inference_mode():
            prefill_outputs = base(
                input_ids=encoded["input_ids"],
                attention_mask=encoded.get("attention_mask"),
                use_cache=True,
                output_attentions=False,
                output_hidden_states=False,
            )
        elapsed_prefill_s = time.perf_counter() - t0

        prefill_next_token_id: int | None = None
        prefill_next_token_text = ""
        lm_head = getattr(model, "lm_head", None)
        prefill_hidden = getattr(prefill_outputs, "last_hidden_state", None)
        if lm_head is not None and prefill_hidden is not None:
            with torch.inference_mode():
                prefill_logits = lm_head(
                    prefill_hidden[:, -1, :].to(dtype=lm_head.weight.dtype)
                )
            prefill_next_token_id = int(torch.argmax(prefill_logits, dim=-1).item())
            prefill_next_token_text = tokenizer.decode(
                [prefill_next_token_id], skip_special_tokens=False
            )

        phase_ref["phase"] = "decode"
        decode_attention_mask = torch.cat(
            [
                encoded.get("attention_mask", torch.ones_like(encoded["input_ids"])),
                torch.ones((encoded["input_ids"].shape[0], 1), dtype=encoded["input_ids"].dtype, device=device),
            ],
            dim=1,
        )
        t1 = time.perf_counter()
        with torch.inference_mode():
            decode_outputs = base(
                input_ids=decode_ids,
                attention_mask=decode_attention_mask,
                past_key_values=prefill_outputs.past_key_values,
                use_cache=True,
                output_attentions=False,
                output_hidden_states=False,
            )
        elapsed_decode_s = time.perf_counter() - t1

        decode_next_token_id: int | None = None
        decode_next_token_text = ""
        decode_hidden = getattr(decode_outputs, "last_hidden_state", None)
        if lm_head is not None and decode_hidden is not None:
            with torch.inference_mode():
                decode_logits = lm_head(
                    decode_hidden[:, -1, :].to(dtype=lm_head.weight.dtype)
                )
            decode_next_token_id = int(torch.argmax(decode_logits, dim=-1).item())
            decode_next_token_text = tokenizer.decode(
                [decode_next_token_id], skip_special_tokens=False
            )
    finally:
        for handle in hooks:
            handle.remove()
        for block in blocks:
            for handle in block["lifecycle_hooks"]:
                handle.remove()

    expected = {
        "layer_00_block_input",
        "layer_00_input_layernorm_output",
        "layer_00_q_proj_output",
        "layer_00_k_proj_output",
        "layer_00_v_proj_output",
        "layer_00_attn_context_before_o_proj",
        "layer_00_o_proj_output",
        "layer_00_post_attention_layernorm_output",
        "layer_00_mlp_gate_proj_output",
        "layer_00_mlp_up_proj_output",
        "layer_00_mlp_gated_activation_before_down_proj",
        "layer_00_mlp_down_proj_output",
        "layer_00_block_output",
    }
    for block in blocks:
        for phase in ("prefill", "decode"):
            missing = sorted(expected.difference(block["captures"][phase]))
            if missing:
                raise RuntimeError(f"Block {block['index']} missing {phase} captures: {missing}")
    attention_records = [
        dict(record, block_index=block["index"])
        for block in blocks if block["attention_runner"] is not None
        for record in block["attention_runner"].records
    ]

    saturation_rows: list[dict[str, Any]] = []
    if saturation_config["enabled"]:
        for block in blocks:
            rows = linear_saturation_rows(
                block["layer"], block["captures"],
                weight_bits=args.weight_bits,
                act_bits=args.act_bits,
                block_size=args.block_size,
                float_format=float_format,
                config=saturation_config,
            )
            saturation_rows.extend(dict(row, block_index=block["index"]) for row in rows)
        if attention_records:
            for record in attention_records:
                if "iv_upper_saturation_events" not in record:
                    continue
                saturation_rows.append(
                    {
                        "phase": record["phase"],
                        "block_index": record["block_index"],
                        "module": f"mapped_{record['mapped_role']}",
                        "role": record["mapped_role"],
                        "call_kind": "mapped_attention_gemm",
                        "tokens": record["actual_tokens"],
                        "batch_matrices": record["actual_batch_head_matrices"],
                        "in_dim": record["actual_in_dim"],
                        "out_dim": record["actual_out_dim"],
                        "row_tiles": math.ceil(int(record["actual_in_dim"]) / 64),
                        "col_tiles": math.ceil(int(record["actual_out_dim"]) / 32),
                        "iv_full_scale_levels": record.get("iv_full_scale_levels", saturation_config["iv_adc_full_scale_levels"]),
                        "iv_nonlinearity_mode": record.get("iv_nonlinearity_mode", saturation_config["iv_nonlinearity_mode"]),
                        "iv_row_groups": record.get("iv_row_groups", saturation_config["row_groups"]),
                        "iv_rows_per_group": record.get("iv_rows_per_group", saturation_config["rows_per_group"]),
                        "iv_transfer_calls": record["iv_transfer_calls"],
                        "iv_upper_saturation_events": record["iv_upper_saturation_events"],
                        "iv_lower_clip_events": record["iv_lower_clip_events"],
                        "iv_unique_saturated_tile_columns": record["iv_unique_saturated_tile_columns"],
                        "iv_saturated_tiles": record["iv_saturated_tiles"],
                        "iv_observed_tile_columns": record["iv_observed_tile_columns"],
                        "iv_tile_reads": record["iv_tile_reads"],
                        "iv_max_branch_level": record["iv_max_branch_level"],
                        "iv_clamped_column_fraction": record["iv_clamped_column_fraction"],
                    }
                )
    sat_summary = saturation_summary(saturation_rows)

    hardware_rows: list[dict[str, Any]] = []
    if hardware_effect_config["enabled"]:
        for block in blocks:
            rows = linear_hardware_effect_rows(
                block["layer"], block["captures"],
                weight_bits=args.weight_bits,
                act_bits=args.act_bits,
                block_size=args.block_size,
                float_format=float_format,
                config=hardware_effect_config,
            )
            hardware_rows.extend(dict(row, block_index=block["index"]) for row in rows)
        if attention_records:
            iv_cfg = CrossbarIVNonlinearityConfig.from_obj(hardware_effect_config["iv"])
            for record in attention_records:
                if "hardware_iv_upper_saturation_events" not in record:
                    continue
                row = {
                    "phase": record["phase"],
                    "block_index": record["block_index"],
                    "module": f"mapped_{record['mapped_role']}",
                    "role": record["mapped_role"],
                    "call_kind": "mapped_attention_gemm",
                    "tokens": record["actual_tokens"],
                    "batch_matrices": record["actual_batch_head_matrices"],
                    "in_dim": record["actual_in_dim"],
                    "out_dim": record["actual_out_dim"],
                    "row_tiles": math.ceil(int(record["actual_in_dim"]) / int(hardware_effect_config.get("row_tile", 64))),
                    "col_tiles": math.ceil(int(record["actual_out_dim"]) / int(hardware_effect_config.get("col_tile", 32))),
                    "iv_full_scale_levels": iv_full_scale_levels(
                        iv_cfg.adc_full_scale_levels,
                        iv_cfg.linear_rows_before_clamp,
                    ),
                }
                for key in hardware_empty_stats():
                    row[key] = record.get(f"hardware_{key}", 0)
                hardware_rows.append(row)
    hardware_summary = hardware_effect_summary(hardware_rows)

    captures = {"prefill": {}, "decode": {}}
    for block in blocks:
        for phase in captures:
            for name, tensor in block["captures"][phase].items():
                output_name = name.replace("layer_00_", f"layer_{block['index']:02d}_", 1) if args.all_blocks else name
                captures[phase][output_name] = tensor

    metadata = {
        "run_name": args.run_name,
        "model": args.model,
        "snapshot_path": str(model_local_dir),
        "phase_list": ["prefill", "decode"],
        "reference_backend": "software_crossbar",
        "crossbar_used": True,
        "crossbar_backend": args.crossbar_backend,
        "crossbar_execution": "block_streamed" if blocks[0]["tile_lifecycle"] is not None else "eager",
        "crossbar_tile_lifecycle": blocks[0]["tile_lifecycle"],
        "activation_buffer": "framework tensors on execution device; diagnostic captures on CPU",
        "kv_cache_policy": "retained across blocks and decode calls",
        "crossbar_implementation": "reram_crossbar.py:CrossbarArray.compute" if args.crossbar_backend == "wrapper" else "numpy:int32_matmul",
        "forward_hardware_effects": {"noise": False, "reliability": False, "iv": False, "tdc": False},
        "mapping_used": True,
        "mapped_attention_gemms_enabled": not args.disable_mapped_attention_gemms,
        "prefill_mapping": str(mapping_ctx["prefill_path"]),
        "decode_mapping": str(mapping_ctx["decode_path"]),
        "prefill_mapping_network": mapping_ctx["prefill_network"],
        "decode_mapping_network": mapping_ctx["decode_network"],
        "mapping_block_layers": [layer.row() for layer in mapping_ctx["selected_layers"]],
        "mapping_summary": mapping_ctx["summary"],
        "mapping_validation": mapping_validation_rows,
        "original_layer_count": original_layer_count,
        "execution_scope": "all transformer blocks" if args.all_blocks else "selected transformer block",
        "executed_block_count": len(blocks),
        "executed_block_indices": [block["index"] for block in blocks],
        "block_index_zero_based": None if args.all_blocks else args.block_index,
        "block_index_one_based": None if args.all_blocks else args.block_index + 1,
        "framework_operations": ["embedding", "normalization", "RoPE", "mask", "softmax", "SwiGLU", "residual", "lm_head", "argmax"],
        "decode_input_mode": "supplied_text",
        "dtype": args.dtype,
        "float_format": float_format,
        "weight_bits": args.weight_bits,
        "act_bits": args.act_bits,
        "block_size": args.block_size,
        "device": str(device),
        "cuda_available": torch.cuda.is_available(),
        "prefill_prompt": args.prompt,
        "prefill_prompt_sha256": hashlib.sha256(args.prompt.encode("utf-8")).hexdigest(),
        "prefill_input_shape": list(encoded["input_ids"].shape),
        "prefill_input_ids": encoded["input_ids"].detach().cpu().tolist(),
        "decode_text": args.decode_text,
        "decode_input_shape": list(decode_ids.shape),
        "decode_input_ids": decode_ids.detach().cpu().tolist(),
        "prefill_next_token_id": prefill_next_token_id,
        "prefill_next_token_text": prefill_next_token_text,
        "decode_next_token_id": decode_next_token_id,
        "decode_next_token_text": decode_next_token_text,
        "elapsed_prefill_s": elapsed_prefill_s,
        "elapsed_decode_s": elapsed_decode_s,
        "python": platform.python_version(),
        "platform": platform.platform(),
        "hostname": platform.node(),
        "linear_crossbar_modules": linear_rows,
        "mapped_attention_gemm_records": attention_records,
        "saturation_audit_config": saturation_config,
        "saturation_summary": sat_summary,
        "saturation_records": saturation_rows,
        "hardware_effect_audit_config": hardware_effect_config,
        "hardware_effect_summary": hardware_summary,
        "hardware_effect_records": hardware_rows,
    }
    if not args.all_blocks:
        for phase in ("prefill", "decode"):
            for field in ("id", "text"):
                metadata[f"selected_block_{phase}_next_token_{field}"] = metadata[f"{phase}_next_token_{field}"]
    endurance_summary, endurance_rows = build_endurance_report(
        metadata=metadata,
        linear_rows=linear_rows,
        endurance_cycles=args.endurance_cycles,
        static_program_cycles=args.static_program_cycles,
        dynamic_kv_endurance_cycles=args.dynamic_kv_endurance_cycles,
        token_time_s=args.kv_cache_token_time_s,
        service_life_years=args.kv_cache_service_life_years,
    )
    metadata["endurance_summary"] = endurance_summary

    values_txt = out_dir / f"{args.run_name}_bf16_values.txt"
    summary_txt = out_dir / f"{args.run_name}_summary.txt"
    metadata_txt = out_dir / f"{args.run_name}_metadata.txt"
    metadata_json = out_dir / "metadata.json"
    endurance_json = out_dir / "endurance_report.json"
    endurance_csv = out_dir / "endurance_report.csv"
    endurance_md = out_dir / "endurance_report.md"
    saturation_json = out_dir / "saturation_report.json"
    saturation_csv = out_dir / "saturation_report.csv"
    saturation_md = out_dir / "saturation_report.md"
    hardware_effects_json = out_dir / "hardware_effects_report.json"
    hardware_effects_csv = out_dir / "hardware_effects_report.csv"
    hardware_effects_md = out_dir / "hardware_effects_report.md"

    write_bf16_text(values_txt, captures, metadata)
    write_summary_text(summary_txt, metadata, captures)
    metadata_txt.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    metadata_json.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    endurance_json.write_text(
        json.dumps({"summary": endurance_summary, "rows": endurance_rows}, indent=2) + "\n",
        encoding="utf-8",
    )
    write_csv(endurance_csv, endurance_rows)
    write_endurance_markdown(endurance_md, endurance_summary, endurance_rows)
    saturation_json.write_text(
        json.dumps({"config": saturation_config, "summary": sat_summary, "rows": saturation_rows}, indent=2) + "\n",
        encoding="utf-8",
    )
    write_csv(saturation_csv, saturation_rows)
    write_saturation_markdown(saturation_md, config=saturation_config, summary=sat_summary, rows=saturation_rows)
    hardware_effects_json.write_text(
        json.dumps({"config": hardware_effect_config, "summary": hardware_summary, "rows": hardware_rows}, indent=2)
        + "\n",
        encoding="utf-8",
    )
    write_csv(hardware_effects_csv, hardware_rows)
    write_hardware_effects_markdown(
        hardware_effects_md,
        config=hardware_effect_config,
        summary=hardware_summary,
        rows=hardware_rows,
    )

    print(f"wrote={out_dir}", flush=True)
    print(f"values_txt={values_txt}", flush=True)
    print(f"summary_txt={summary_txt}", flush=True)
    print(f"endurance_json={endurance_json}", flush=True)
    print(f"saturation_json={saturation_json}", flush=True)
    print(f"hardware_effects_json={hardware_effects_json}", flush=True)
    print(f"elapsed_prefill_s={elapsed_prefill_s:.4f}", flush=True)
    print(f"elapsed_decode_s={elapsed_decode_s:.4f}", flush=True)
    print(
        f"prefill_next_token={prefill_next_token_id} {prefill_next_token_text!r}",
        flush=True,
    )
    print(
        f"decode_next_token={decode_next_token_id} {decode_next_token_text!r}",
        flush=True,
    )


if __name__ == "__main__":
    run(parse_args())
