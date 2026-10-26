from __future__ import annotations

import argparse
import contextlib
import hashlib
import io
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
RESULTS_DIR = SCRIPT_DIR / "results"
sys.dont_write_bytecode = True

from reram_crossbar import (  # noqa: E402
    CrossbarArray,
    CrossbarFeatures,
    CrossbarIVNonlinearityConfig,
    CrossbarNoiseConfig,
    CrossbarPEColumnParallelConfig,
    CrossbarPERowParallelConfig,
    CrossbarReliabilityConfig,
    CrossbarTDCConfig,
    sar_tdc_precision_choices,
)
from stateful_kv_cache import (  # noqa: E402
    DEFAULT_KV_CACHE_CONTEXT_TOKENS,
    DEFAULT_KV_CACHE_SERVICE_LIFE_YEARS,
    DEFAULT_KV_CACHE_TOKEN_TIME_S,
    StatefulKVCache,
    StatefulKVCacheConfig,
)
from transformer_runtime_modules import (  # noqa: E402
    BF16ArithmeticModule,
    CausalMaskModule,
    EmbeddingLookupModule,
    HiddenVectorBufferModule,
    KVCacheControlModule,
    ModuleCallRecorder,
    QKNormModule,
    RMSNormModule,
    ResidualAddModule,
    RoPEModule,
    SiLUModule,
    SoftmaxModule,
    SwiGLUMultiplyModule,
    TokenSamplingModule,
    runtime_module_catalog,
)


ATTN_QKT_ROLE = "attn_scores_qkt"
ATTN_AV_ROLE = "attn_av"
DEFAULT_ATTN_AV_TDC_FINE_BITS = 5

NOISE_MODEL_DESCRIPTION = (
    "Per INT4 cell read: I = Vread/R plus thermal+shot white noise, integrated 1/f noise, "
    "RTN variance, and readout RMS noise converted to INT4-level perturbation before ADC."
)
RELIABILITY_MODEL_DESCRIPTION = (
    "Persistent programmed-cell model: stuck-at HRS/LRS defects, optional raw BER or post-ECC "
    "bit flips with latency accounting, legacy or state-power-law retention drift, relaxation drift, "
    "state spread, endurance drift, and end-of-life stuck-cell or multilevel-loss modes."
)
IV_NONLINEARITY_MODEL_DESCRIPTION = (
    "When enabled, active rows are summed on positive and negative columns per INT4 slice, "
    "subtracted as a signed differential current, then hard-clamped or tanh-compressed at ADC full-scale."
)
TDC_MODEL_DESCRIPTION = (
    "When enabled, full positive/negative column sums are locally subtracted both ways (A-B and B-A), "
    "each unipolar branch is coarse/fine TDC-quantized, and the two TDC codes are digitally subtracted "
    "into a BFP-style q_y mantissa with gain exponent E_g."
)
PE_ROW_PARALLEL_MODEL_DESCRIPTION = (
    "Rows inside each PE tile can be sensed as independent row groups, limiting added row current before "
    "ADC/SAR-TDC conversion and digitally summing the row-group partials."
)
PE_COLUMN_PARALLEL_MODEL_DESCRIPTION = (
    "Logical output columns can be sensed in configurable parallel steps so payload width, packet overhead, "
    "and column-parallel output bandwidth can be studied per PE tile."
)


def _seed_part(part: object) -> int:
    if part is None:
        return 0
    if isinstance(part, str):
        digest = hashlib.sha256(part.encode("utf-8")).digest()
        return int.from_bytes(digest[:4], "little")
    return int(part) & 0xFFFFFFFF


def derive_seed(base_seed: int | None, *parts: object) -> int | None:
    if base_seed is None:
        return None
    entropy = [_seed_part(base_seed), *(_seed_part(part) for part in parts)]
    return int(np.random.SeedSequence(entropy).generate_state(1, dtype=np.uint32)[0])


def quantize_symmetric(x: np.ndarray, bits: int = 8) -> tuple[np.ndarray, float]:
    max_abs = float(np.max(np.abs(x))) if x.size else 0.0
    qmax = (1 << (bits - 1)) - 1
    scale = max(max_abs / qmax, 1.0e-8)
    q = np.clip(np.rint(x / scale), -qmax - 1, qmax).astype(np.int8)
    return q, scale


def normalize_float_format(float_format: str | None) -> str:
    fmt = "bf16" if float_format is None else str(float_format).lower()
    aliases = {
        "bfloat16": "bf16",
        "torch.bfloat16": "bf16",
        "float32": "fp32",
        "torch.float32": "fp32",
        "none": "fp32",
    }
    fmt = aliases.get(fmt, fmt)
    if fmt not in {"bf16", "fp32"}:
        raise ValueError(f"Unsupported PE float_format {float_format!r}; use 'bf16' or 'fp32'.")
    return fmt


def round_float_format(x: np.ndarray, float_format: str | None = "bf16") -> np.ndarray:
    """Round PE boundary values to the selected floating format, returned as float32 arrays."""

    fmt = normalize_float_format(float_format)
    arr = np.asarray(x, dtype=np.float32)
    if fmt == "fp32" or arr.size == 0:
        return arr.astype(np.float32, copy=True)

    out = np.ascontiguousarray(arr, dtype=np.float32).copy()
    u = out.view(np.uint32)
    finite = np.isfinite(out)
    rounded = u[finite] + np.uint32(0x7FFF) + ((u[finite] >> np.uint32(16)) & np.uint32(1))
    u[finite] = rounded & np.uint32(0xFFFF0000)
    return out


def quantize_block16(x: np.ndarray, bits: int = 5, block_size: int = 16) -> np.ndarray:
    """Symmetric per-block fixed-point quantization over the last dimension."""
    if bits < 2:
        raise ValueError("bits must be >= 2")
    if block_size <= 0:
        raise ValueError("block_size must be > 0")

    x_float = np.asarray(x, dtype=np.float32)
    if x_float.size == 0:
        return x_float.copy()

    orig_shape = x_float.shape
    if x_float.ndim == 0:
        x_2d = x_float.reshape(1, 1)
        last_dim = 1
    else:
        last_dim = orig_shape[-1]
        x_2d = x_float.reshape(-1, last_dim)

    pad = (block_size - last_dim % block_size) % block_size
    if pad:
        x_padded = np.pad(x_2d, ((0, 0), (0, pad)), mode="constant")
    else:
        x_padded = x_2d

    blocks = x_padded.reshape(x_2d.shape[0], -1, block_size)
    qmax = (1 << (bits - 1)) - 1
    scale = np.maximum(np.max(np.abs(blocks), axis=-1, keepdims=True) / qmax, 1.0e-8)
    q = np.clip(np.rint(blocks / scale), -qmax, qmax)
    quantized = (q * scale).reshape(x_2d.shape[0], -1)[:, :last_dim]
    return quantized.reshape(orig_shape).astype(np.float32)


def layer_norm(x: np.ndarray, eps: float = 1.0e-5) -> np.ndarray:
    mean = np.mean(x, axis=-1, keepdims=True)
    var = np.mean((x - mean) ** 2, axis=-1, keepdims=True)
    return (x - mean) / np.sqrt(var + eps)


def softmax(x: np.ndarray) -> np.ndarray:
    shifted = x - np.max(x, axis=-1, keepdims=True)
    exp_x = np.exp(shifted)
    return exp_x / np.sum(exp_x, axis=-1, keepdims=True)


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def swiglu(x: np.ndarray) -> np.ndarray:
    up, gate = np.split(x, 2, axis=-1)
    return up * sigmoid(gate)


@dataclass(frozen=True)
class LinearStats:
    name: str
    input_dim: int
    output_dim: int
    rows: int
    tokens: int
    crossbar_tiles: int
    macs: int
    ecc_program_latency_cycles: int = 0
    ecc_compute_latency_cycles: int = 0
    ecc_total_latency_cycles: int = 0
    pe_row_groups: int = 1
    pe_active_columns: int = 0
    pe_output_payload_bits_per_value: int = 0
    pe_output_payload_bits_per_token: int = 0
    pe_output_wire_bits_per_token: int = 0
    pe_output_wire_bits: int = 0


def tdc_config_for_attention_role(
    base_config: CrossbarTDCConfig | dict | None,
    role: str,
    *,
    attn_av_tdc_fine_bits: int = DEFAULT_ATTN_AV_TDC_FINE_BITS,
    force_sar_tdc: bool = True,
) -> CrossbarTDCConfig:
    """Return the SAR-TDC policy used by dynamic decode attention PE layers."""

    cfg = CrossbarTDCConfig.from_obj(base_config)
    if force_sar_tdc and not cfg.enabled:
        cfg = CrossbarTDCConfig(
            enabled=True,
            tdc_bits=int(cfg.tdc_bits),
            coarse_bits=int(cfg.coarse_bits),
            fine_bits=int(cfg.fine_bits),
            gain_shift=int(cfg.gain_shift),
            output_format="sar_fp",
        )
    if str(role) != ATTN_AV_ROLE or not cfg.enabled:
        return cfg

    fine_bits = max(int(cfg.fine_bits), int(attn_av_tdc_fine_bits))
    total_bits = max(int(cfg.tdc_bits), int(cfg.coarse_bits) + fine_bits)
    return CrossbarTDCConfig(
        enabled=True,
        tdc_bits=total_bits,
        coarse_bits=int(cfg.coarse_bits),
        fine_bits=fine_bits,
        gain_shift=int(cfg.gain_shift),
        output_format=str(cfg.output_format),
    )


def _attention_backend_name(tdc_config: CrossbarTDCConfig) -> str:
    sar_formats = {"sar", "sar_fp", "fp", "float", "float_like"}
    if tdc_config.enabled and str(tdc_config.output_format).lower() in sar_formats:
        return "dynamic_kv_cache_reram_sar_tdc"
    if tdc_config.enabled:
        return "dynamic_kv_cache_reram_tdc"
    return "dynamic_kv_cache_reram_adc"


def _programmed_cell_count(executor: "ReRAMLinearExecutor") -> int:
    total = 0
    for info in executor.crossbar_infos():
        reliability_stats = info.get("reliability_stats", {})
        if isinstance(reliability_stats, dict):
            total += int(reliability_stats.get("programmed_cells", 0))
    return total


def _record_dynamic_attention_stat(
    records: dict[tuple[int, str], dict[str, object]],
    *,
    layer_idx: int,
    role: str,
    heads: int,
    context_tokens: int,
    executor: "ReRAMLinearExecutor",
    tdc_config: CrossbarTDCConfig,
    float_format: str,
) -> None:
    key = (int(layer_idx), str(role))
    stat = executor.stats(tokens=1)
    record = records.get(key)
    if record is None:
        record = {
            "name": f"L{int(layer_idx):02d}_{role}",
            "role": str(role),
            "layer_idx": int(layer_idx),
            "heads": int(heads),
            "backend": _attention_backend_name(tdc_config),
            "tdc_enabled": bool(tdc_config.enabled),
            "tdc_bits": int(tdc_config.tdc_bits),
            "tdc_coarse_bits": int(tdc_config.coarse_bits),
            "tdc_fine_bits": int(tdc_config.fine_bits),
            "tdc_gain_shift": int(tdc_config.gain_shift),
            "tdc_output_format": str(tdc_config.output_format),
            "float_format": normalize_float_format(float_format),
            "calls": 0,
            "context_tokens_min": int(context_tokens),
            "context_tokens_max": int(context_tokens),
            "context_tokens_total": 0,
            "input_dim_min": int(stat.input_dim),
            "input_dim_max": int(stat.input_dim),
            "output_dim_min": int(stat.output_dim),
            "output_dim_max": int(stat.output_dim),
            "crossbar_tiles": 0,
            "macs": 0,
            "dynamic_programmed_cells": 0,
            "ecc_program_latency_cycles": 0,
            "ecc_compute_latency_cycles": 0,
            "ecc_total_latency_cycles": 0,
        }
        records[key] = record

    record["calls"] = int(record["calls"]) + 1
    record["context_tokens_min"] = min(int(record["context_tokens_min"]), int(context_tokens))
    record["context_tokens_max"] = max(int(record["context_tokens_max"]), int(context_tokens))
    record["context_tokens_total"] = int(record["context_tokens_total"]) + int(context_tokens)
    record["input_dim_min"] = min(int(record["input_dim_min"]), int(stat.input_dim))
    record["input_dim_max"] = max(int(record["input_dim_max"]), int(stat.input_dim))
    record["output_dim_min"] = min(int(record["output_dim_min"]), int(stat.output_dim))
    record["output_dim_max"] = max(int(record["output_dim_max"]), int(stat.output_dim))
    record["crossbar_tiles"] = int(record["crossbar_tiles"]) + int(stat.crossbar_tiles)
    record["macs"] = int(record["macs"]) + int(stat.macs)
    record["dynamic_programmed_cells"] = int(record["dynamic_programmed_cells"]) + _programmed_cell_count(executor)
    record["ecc_program_latency_cycles"] = int(record["ecc_program_latency_cycles"]) + int(
        stat.ecc_program_latency_cycles
    )
    record["ecc_compute_latency_cycles"] = int(record["ecc_compute_latency_cycles"]) + int(
        stat.ecc_compute_latency_cycles
    )
    record["ecc_total_latency_cycles"] = int(record["ecc_total_latency_cycles"]) + int(stat.ecc_total_latency_cycles)


def _record_dynamic_attention_compute_call(
    records: dict[tuple[int, str], dict[str, object]],
    *,
    layer_idx: int,
    role: str,
    heads: int,
    context_tokens: int,
    input_dim: int,
    output_dim: int,
    macs: int,
    tdc_config: CrossbarTDCConfig,
    float_format: str,
) -> None:
    key = (int(layer_idx), str(role))
    record = records.get(key)
    if record is None:
        record = {
            "name": f"L{int(layer_idx):02d}_{role}",
            "role": str(role),
            "layer_idx": int(layer_idx),
            "heads": int(heads),
            "backend": _attention_backend_name(tdc_config),
            "tdc_enabled": bool(tdc_config.enabled),
            "tdc_bits": int(tdc_config.tdc_bits),
            "tdc_coarse_bits": int(tdc_config.coarse_bits),
            "tdc_fine_bits": int(tdc_config.fine_bits),
            "tdc_gain_shift": int(tdc_config.gain_shift),
            "tdc_output_format": str(tdc_config.output_format),
            "float_format": normalize_float_format(float_format),
            "persistent_pe_state": True,
            "calls": 0,
            "context_tokens_min": int(context_tokens),
            "context_tokens_max": int(context_tokens),
            "context_tokens_total": 0,
            "input_dim_min": int(input_dim),
            "input_dim_max": int(input_dim),
            "output_dim_min": int(output_dim),
            "output_dim_max": int(output_dim),
            "crossbar_tiles": 0,
            "macs": 0,
            "dynamic_programmed_cells": 0,
            "ecc_program_latency_cycles": 0,
            "ecc_compute_latency_cycles": 0,
            "ecc_total_latency_cycles": 0,
            "programmed_tile_updates": 0,
            "weight_clip_events": 0,
        }
        records[key] = record

    record["calls"] = int(record["calls"]) + 1
    record["context_tokens_min"] = min(int(record["context_tokens_min"]), int(context_tokens))
    record["context_tokens_max"] = max(int(record["context_tokens_max"]), int(context_tokens))
    record["context_tokens_total"] = int(record["context_tokens_total"]) + int(context_tokens)
    record["input_dim_min"] = min(int(record["input_dim_min"]), int(input_dim))
    record["input_dim_max"] = max(int(record["input_dim_max"]), int(input_dim))
    record["output_dim_min"] = min(int(record["output_dim_min"]), int(output_dim))
    record["output_dim_max"] = max(int(record["output_dim_max"]), int(output_dim))
    record["macs"] = int(record["macs"]) + int(macs)


def _finalize_dynamic_attention_stats(records: dict[tuple[int, str], dict[str, object]]) -> list[dict[str, object]]:
    finalized: list[dict[str, object]] = []
    for key in sorted(records):
        record = dict(records[key])
        calls = max(int(record["calls"]), 1)
        record["context_tokens_mean"] = float(record["context_tokens_total"]) / float(calls)
        finalized.append(record)
    return finalized


class ReRAMLinearExecutor:
    """Dense linear layer implemented as 64x32 signed INT8 ReRAM crossbar tiles."""

    row_tile = 64
    col_tile = 32

    def __init__(
        self,
        name: str,
        weight_float: np.ndarray,
        verbose_crossbar: bool = False,
        act_bits: int = 8,
        act_block_size: int = 16,
        crossbar_features: CrossbarFeatures | dict | None = None,
        noise_config: CrossbarNoiseConfig | dict | None = None,
        reliability_config: CrossbarReliabilityConfig | dict | None = None,
        iv_config: CrossbarIVNonlinearityConfig | dict | None = None,
        tdc_config: CrossbarTDCConfig | dict | None = None,
        row_parallel_config: CrossbarPERowParallelConfig | dict | None = None,
        column_parallel_config: CrossbarPEColumnParallelConfig | dict | None = None,
        noise_seed: int | None = None,
        reliability_seed: int | None = None,
        float_format: str = "bf16",
        defer_programming: bool = False,
    ):
        self.name = name
        self.verbose_crossbar = bool(verbose_crossbar)
        self.act_bits = int(act_bits)
        self.act_block_size = int(act_block_size)
        self.float_format = normalize_float_format(float_format)
        features = CrossbarFeatures.from_obj(crossbar_features)
        self.noise_config = CrossbarNoiseConfig.from_obj(
            noise_config if noise_config is not None else features.noise
        )
        self.reliability_config = CrossbarReliabilityConfig.from_obj(
            reliability_config if reliability_config is not None else features.reliability
        )
        self.iv_config = CrossbarIVNonlinearityConfig.from_obj(iv_config if iv_config is not None else features.iv)
        self.tdc_config = CrossbarTDCConfig.from_obj(tdc_config if tdc_config is not None else features.tdc)
        self.row_parallel_config = CrossbarPERowParallelConfig.from_obj(
            row_parallel_config if row_parallel_config is not None else features.row_parallel
        )
        self.column_parallel_config = CrossbarPEColumnParallelConfig.from_obj(
            column_parallel_config if column_parallel_config is not None else features.column_parallel
        )
        self.noise_seed = features.noise_seed if noise_seed is None else noise_seed
        self.reliability_seed = features.reliability_seed if reliability_seed is None else reliability_seed
        weight_float = round_float_format(weight_float, self.float_format)
        self.weight_q, self.weight_scale = quantize_symmetric(weight_float)
        self.in_dim, self.out_dim = self.weight_q.shape
        self.row_tiles = int(np.ceil(self.in_dim / self.row_tile))
        self.col_tiles = int(np.ceil(self.out_dim / self.col_tile))
        self.crossbar_tiles = self.row_tiles * self.col_tiles
        self._programmed: list[list[tuple[CrossbarArray, np.ndarray]]] = []
        if defer_programming:
            self._validate_tile_release()
        else:
            self.program_tiles()

    def _validate_tile_release(self) -> None:
        if any(config.enabled for config in (
            self.noise_config, self.reliability_config, self.iv_config, self.tdc_config,
        )):
            raise ValueError("Tile streaming requires forward noise, reliability, IV and TDC to be disabled.")

    def program_tiles(self) -> None:
        if not self._programmed:
            self._programmed = self._program_tiles()

    def release_tiles(self) -> None:
        self._validate_tile_release()
        self._programmed.clear()

    def _quiet(self):
        if self.verbose_crossbar:
            return contextlib.nullcontext()
        return contextlib.redirect_stdout(io.StringIO())

    def _program_tiles(self) -> list[list[tuple[CrossbarArray, np.ndarray]]]:
        programmed: list[list[tuple[CrossbarArray, np.ndarray]]] = []
        for r_block in range(self.row_tiles):
            row_start = r_block * self.row_tile
            row_end = min(row_start + self.row_tile, self.in_dim)
            row_tiles: list[tuple[CrossbarArray, np.ndarray]] = []
            for c_block in range(self.col_tiles):
                col_start = c_block * self.col_tile
                col_end = min(col_start + self.col_tile, self.out_dim)
                tile = np.zeros((self.row_tile, self.col_tile), dtype=np.int8)
                tile[: row_end - row_start, : col_end - col_start] = self.weight_q[row_start:row_end, col_start:col_end]
                with self._quiet():
                    crossbar = CrossbarArray(
                        input_size=self.row_tile,
                        output_size=self.col_tile * 2,
                        noise_config=self.noise_config,
                        reliability_config=self.reliability_config,
                        iv_config=self.iv_config,
                        tdc_config=self.tdc_config,
                        row_parallel_config=self.row_parallel_config,
                        column_parallel_config=self.column_parallel_config,
                        noise_seed=derive_seed(self.noise_seed, self.name, r_block, c_block),
                        reliability_seed=derive_seed(self.reliability_seed, self.name, r_block, c_block),
                    )
                    crossbar.program_weights(tile)
                    effective_tile = crossbar.get_effective_weights().astype(np.int32)
                row_tiles.append((crossbar, effective_tile))
            programmed.append(row_tiles)
        return programmed

    def __call__(self, x_float: np.ndarray) -> np.ndarray:
        self.program_tiles()
        original_shape = x_float.shape[:-1]
        x_2d = round_float_format(x_float.reshape(-1, x_float.shape[-1]), self.float_format)
        x_block = quantize_block16(x_2d, bits=self.act_bits, block_size=self.act_block_size)
        x_q, x_scale = quantize_symmetric(x_block)
        y_int = np.zeros((x_2d.shape[0], self.out_dim), dtype=np.int64)

        for token_idx, token in enumerate(x_q.astype(np.int16)):
            for r_block, row_tiles in enumerate(self._programmed):
                row_start = r_block * self.row_tile
                row_end = min(row_start + self.row_tile, self.in_dim)
                x_tile_signed = np.zeros((self.row_tile,), dtype=np.int16)
                x_tile_signed[: row_end - row_start] = token[row_start:row_end]
                x_tile_unsigned = (x_tile_signed.astype(np.int16) + 128).astype(np.uint8)

                for c_block, (crossbar, _weight_tile) in enumerate(row_tiles):
                    col_start = c_block * self.col_tile
                    col_end = min(col_start + self.col_tile, self.out_dim)
                    with self._quiet():
                        if crossbar.tdc_config.enabled:
                            if str(crossbar.tdc_config.output_format).lower() in {"sar", "sar_fp", "fp", "float", "float_like"}:
                                packed = crossbar.compute_tdc_packed_corrected(x_tile_unsigned, zero_point=128)
                                corrected = packed.reconstruct_integer()[: self.col_tile]
                            else:
                                packed = crossbar.compute_tdc_packed(x_tile_unsigned)
                                zero = crossbar.pwm_zero_point_packed_output(128, exponent=packed.exponent)
                                corrected = (
                                    packed.reconstruct_integer()[: self.col_tile]
                                    - zero.reconstruct_integer()[: self.col_tile]
                                )
                        else:
                            raw = crossbar.compute(x_tile_unsigned)
                            logical = raw[: self.col_tile].astype(np.int64)
                            offset_correction = crossbar.pwm_zero_point_output(128)[: self.col_tile].astype(np.int64)
                            corrected = logical - offset_correction
                    y_int[token_idx, col_start:col_end] += corrected[: col_end - col_start]

        y = (y_int.astype(np.float32) * x_scale * self.weight_scale).reshape(*original_shape, self.out_dim)
        return round_float_format(y, self.float_format)

    def crossbar_infos(self) -> list[dict[str, object]]:
        return [crossbar.get_info() for row_tiles in self._programmed for crossbar, _ in row_tiles]

    def latency_model(self) -> dict[str, int]:
        totals = {
            "ecc_program_latency_cycles": 0,
            "ecc_compute_latency_cycles": 0,
            "ecc_total_latency_cycles": 0,
        }
        for info in self.crossbar_infos():
            latency = info.get("latency_model", {})
            if not isinstance(latency, dict):
                continue
            for key in totals:
                totals[key] += int(latency.get(key, 0))
        return totals

    def pe_output_bandwidth(self, tokens: int = 1) -> dict[str, int]:
        totals = {
            "row_groups": 1,
            "active_columns": 0,
            "payload_bits_per_value": 0,
            "payload_bits_per_token": 0,
            "wire_bits_per_token": 0,
            "wire_bits": 0,
        }
        for row_tiles in self._programmed:
            for c_block, (crossbar, _) in enumerate(row_tiles):
                col_start = c_block * self.col_tile
                col_end = min(col_start + self.col_tile, self.out_dim)
                bw = crossbar.estimate_output_bandwidth(logical_outputs=col_end - col_start)
                totals["row_groups"] = max(totals["row_groups"], int(bw["row_groups"]))
                totals["active_columns"] = max(totals["active_columns"], int(bw["active_columns_per_step"]))
                totals["payload_bits_per_value"] = max(
                    totals["payload_bits_per_value"], int(bw["payload_bits_per_value"])
                )
                totals["payload_bits_per_token"] += int(bw["payload_bits"])
                totals["wire_bits_per_token"] += int(bw["wire_bits"])
        totals["wire_bits"] = totals["wire_bits_per_token"] * int(tokens)
        return totals

    def stats(self, tokens: int) -> LinearStats:
        latency = self.latency_model()
        bandwidth = self.pe_output_bandwidth(tokens=tokens)
        return LinearStats(
            name=self.name,
            input_dim=self.in_dim,
            output_dim=self.out_dim,
            rows=self.row_tile,
            tokens=int(tokens),
            crossbar_tiles=self.crossbar_tiles,
            macs=int(tokens) * self.in_dim * self.out_dim,
            ecc_program_latency_cycles=latency["ecc_program_latency_cycles"],
            ecc_compute_latency_cycles=latency["ecc_compute_latency_cycles"],
            ecc_total_latency_cycles=latency["ecc_total_latency_cycles"],
            pe_row_groups=bandwidth["row_groups"],
            pe_active_columns=bandwidth["active_columns"],
            pe_output_payload_bits_per_value=bandwidth["payload_bits_per_value"],
            pe_output_payload_bits_per_token=bandwidth["payload_bits_per_token"],
            pe_output_wire_bits_per_token=bandwidth["wire_bits_per_token"],
            pe_output_wire_bits=bandwidth["wire_bits"],
        )


class PersistentDynamicAttentionMatrix:
    """
    Append-programmed dynamic ReRAM matrix for decode attention.

    Static layers can program every tile once because their weights are known.
    Decode attention is different: K/V rows arrive token by token. This matrix
    keeps the same physical crossbar objects alive, patches only the new K/V
    cells into those objects, and reuses the already programmed tiles for later
    attention reads.
    """

    row_tile = ReRAMLinearExecutor.row_tile
    col_tile = ReRAMLinearExecutor.col_tile

    def __init__(
        self,
        name: str,
        in_dim: int,
        out_dim: int,
        *,
        linear_kwargs: dict[str, object],
        tdc_config: CrossbarTDCConfig,
        float_format: str,
    ):
        self.name = str(name)
        self.in_dim = int(in_dim)
        self.out_dim = int(out_dim)
        if self.in_dim <= 0:
            raise ValueError("in_dim must be > 0")
        if self.out_dim <= 0:
            raise ValueError("out_dim must be > 0")

        self.verbose_crossbar = bool(linear_kwargs.get("verbose_crossbar", False))
        self.act_bits = int(linear_kwargs.get("act_bits", 8))
        self.act_block_size = int(linear_kwargs.get("act_block_size", 16))
        self.float_format = normalize_float_format(float_format)
        self.noise_config = CrossbarNoiseConfig.from_obj(linear_kwargs.get("noise_config"))
        self.reliability_config = CrossbarReliabilityConfig.from_obj(linear_kwargs.get("reliability_config"))
        self.iv_config = CrossbarIVNonlinearityConfig.from_obj(linear_kwargs.get("iv_config"))
        self.tdc_config = CrossbarTDCConfig.from_obj(tdc_config)
        self.row_parallel_config = CrossbarPERowParallelConfig.from_obj(linear_kwargs.get("row_parallel_config"))
        self.column_parallel_config = CrossbarPEColumnParallelConfig.from_obj(linear_kwargs.get("column_parallel_config"))
        self.noise_seed = linear_kwargs.get("noise_seed")
        self.reliability_seed = linear_kwargs.get("reliability_seed")

        self.row_tiles = int(np.ceil(self.in_dim / self.row_tile))
        self.col_tiles = int(np.ceil(self.out_dim / self.col_tile))
        self.weight_q = np.zeros((self.in_dim, self.out_dim), dtype=np.int8)
        self.weight_scale: float | None = None
        self.programmed_tile_updates = 0
        self.weight_clip_events = 0
        self._programmed_tiles: set[tuple[int, int]] = set()
        self._programmed: dict[tuple[int, int], CrossbarArray] = {}

    def _quiet(self):
        if self.verbose_crossbar:
            return contextlib.nullcontext()
        return contextlib.redirect_stdout(io.StringIO())

    def _tile_for(self, r_block: int, c_block: int) -> CrossbarArray:
        key = (int(r_block), int(c_block))
        crossbar = self._programmed.get(key)
        if crossbar is not None:
            return crossbar
        with self._quiet():
            crossbar = CrossbarArray(
                input_size=self.row_tile,
                output_size=self.col_tile * 2,
                noise_config=self.noise_config,
                reliability_config=self.reliability_config,
                iv_config=self.iv_config,
                tdc_config=self.tdc_config,
                row_parallel_config=self.row_parallel_config,
                column_parallel_config=self.column_parallel_config,
                noise_seed=derive_seed(self.noise_seed, self.name, r_block, c_block),
                reliability_seed=derive_seed(self.reliability_seed, self.name, r_block, c_block),
            )
        crossbar.mode = crossbar.MODE_PWM_INT8_ROW_SERIAL
        crossbar._effective_signed = np.zeros((self.row_tile, self.col_tile), dtype=np.int16)
        self._programmed[key] = crossbar
        return crossbar

    def program_patch(self, row_start: int, col_start: int, patch_float: np.ndarray) -> None:
        patch = round_float_format(np.asarray(patch_float, dtype=np.float32), self.float_format)
        if patch.ndim != 2:
            raise ValueError(f"patch_float must be 2D, got shape {patch.shape}")
        row_start = int(row_start)
        col_start = int(col_start)
        row_end = row_start + patch.shape[0]
        col_end = col_start + patch.shape[1]
        if row_start < 0 or col_start < 0 or row_end > self.in_dim or col_end > self.out_dim:
            raise IndexError(
                f"patch {patch.shape} at ({row_start}, {col_start}) exceeds matrix shape "
                f"({self.in_dim}, {self.out_dim})"
            )
        if patch.size == 0:
            return

        if self.weight_scale is None:
            max_abs = float(np.max(np.abs(patch))) if patch.size else 0.0
            if max_abs > 0.0:
                self.weight_scale = max(max_abs / 127.0, 1.0e-8)
        patch_scale = float(self.weight_scale) if self.weight_scale is not None else 1.0e-8
        scaled = np.rint(patch / patch_scale)
        self.weight_clip_events += int(np.count_nonzero((scaled < -128.0) | (scaled > 127.0)))
        patch_q = np.clip(scaled, -128, 127).astype(np.int8)
        self.weight_q[row_start:row_end, col_start:col_end] = patch_q

        r_first = row_start // self.row_tile
        r_last = (row_end - 1) // self.row_tile
        c_first = col_start // self.col_tile
        c_last = (col_end - 1) // self.col_tile
        for r_block in range(r_first, r_last + 1):
            tile_row_start = r_block * self.row_tile
            overlap_row_start = max(row_start, tile_row_start)
            overlap_row_end = min(row_end, tile_row_start + self.row_tile)
            patch_row_start = overlap_row_start - row_start
            patch_row_end = overlap_row_end - row_start
            local_row_start = overlap_row_start - tile_row_start
            for c_block in range(c_first, c_last + 1):
                tile_col_start = c_block * self.col_tile
                overlap_col_start = max(col_start, tile_col_start)
                overlap_col_end = min(col_end, tile_col_start + self.col_tile)
                patch_col_start = overlap_col_start - col_start
                patch_col_end = overlap_col_end - col_start
                local_col_start = overlap_col_start - tile_col_start
                tile_patch = patch_q[patch_row_start:patch_row_end, patch_col_start:patch_col_end]
                self._program_crossbar_patch(
                    self._tile_for(r_block, c_block),
                    local_row_start,
                    local_col_start,
                    tile_patch,
                )
                self._programmed_tiles.add((r_block, c_block))
                self.programmed_tile_updates += 1

    def _program_crossbar_patch(
        self,
        crossbar: CrossbarArray,
        row_start: int,
        col_start: int,
        patch_q: np.ndarray,
    ) -> None:
        patch = np.asarray(patch_q, dtype=np.int8)
        if patch.ndim != 2 or patch.size == 0:
            return
        row_end = int(row_start) + patch.shape[0]
        col_end = int(col_start) + patch.shape[1]
        if row_end > crossbar.input_size or col_end > crossbar.output_size // 2:
            raise IndexError("crossbar patch exceeds tile bounds")

        with self._quiet():
            pos0, pos1, neg0, neg1 = crossbar._decompose_signed_int8(patch)
        rows = slice(int(row_start), row_end)
        cols = slice(int(col_start), col_end)
        crossbar._w_pos0[rows, cols] = pos0
        crossbar._w_pos1[rows, cols] = pos1
        crossbar._w_neg0[rows, cols] = neg0
        crossbar._w_neg1[rows, cols] = neg1
        crossbar._effective_signed = crossbar._reconstruct_signed_int8()
        bits = int(patch.size) * 4 * int(crossbar.cell_bits)
        crossbar._ecc_program_latency_cycles += crossbar._ecc_latency_cycles_for_bits(
            bits,
            crossbar.reliability_config.ecc_encode_latency_cycles,
        )

    def __call__(
        self,
        x_float: np.ndarray,
        *,
        active_input_indices: np.ndarray | None = None,
        active_output_indices: np.ndarray | None = None,
    ) -> np.ndarray:
        x_arr = np.asarray(x_float, dtype=np.float32)
        if x_arr.shape[-1] != self.in_dim:
            raise ValueError(f"input last dimension must be {self.in_dim}, got {x_arr.shape[-1]}")
        original_shape = x_arr.shape[:-1]
        x_2d = round_float_format(x_arr.reshape(-1, x_arr.shape[-1]), self.float_format)
        x_block = quantize_block16(x_2d, bits=self.act_bits, block_size=self.act_block_size)
        x_q, x_scale = quantize_symmetric(x_block)
        y_int = np.zeros((x_2d.shape[0], self.out_dim), dtype=np.int64)
        if not self._programmed_tiles:
            return np.zeros((*original_shape, self.out_dim), dtype=np.float32)

        row_blocks = self._active_blocks(active_input_indices, self.in_dim, self.row_tile, self.row_tiles)
        col_blocks = self._active_blocks(active_output_indices, self.out_dim, self.col_tile, self.col_tiles)
        if not row_blocks or not col_blocks:
            return np.zeros((*original_shape, self.out_dim), dtype=np.float32)

        for token_idx, token in enumerate(x_q.astype(np.int16)):
            for r_block in row_blocks:
                row_start = r_block * self.row_tile
                row_end = min(row_start + self.row_tile, self.in_dim)
                x_tile_signed = np.zeros((self.row_tile,), dtype=np.int16)
                x_tile_signed[: row_end - row_start] = token[row_start:row_end]
                x_tile_unsigned = (x_tile_signed.astype(np.int16) + 128).astype(np.uint8)

                for c_block in col_blocks:
                    if (r_block, c_block) not in self._programmed_tiles:
                        continue
                    crossbar = self._programmed[(r_block, c_block)]
                    col_start = c_block * self.col_tile
                    col_end = min(col_start + self.col_tile, self.out_dim)
                    with self._quiet():
                        if crossbar.tdc_config.enabled:
                            if str(crossbar.tdc_config.output_format).lower() in {
                                "sar",
                                "sar_fp",
                                "fp",
                                "float",
                                "float_like",
                            }:
                                packed = crossbar.compute_tdc_packed_corrected(x_tile_unsigned, zero_point=128)
                                corrected = packed.reconstruct_integer()[: self.col_tile]
                            else:
                                packed = crossbar.compute_tdc_packed(x_tile_unsigned)
                                zero = crossbar.pwm_zero_point_packed_output(128, exponent=packed.exponent)
                                corrected = (
                                    packed.reconstruct_integer()[: self.col_tile]
                                    - zero.reconstruct_integer()[: self.col_tile]
                                )
                        else:
                            raw = crossbar.compute(x_tile_unsigned)
                            logical = raw[: self.col_tile].astype(np.int64)
                            offset_correction = crossbar.pwm_zero_point_output(128)[: self.col_tile].astype(np.int64)
                            corrected = logical - offset_correction
                    y_int[token_idx, col_start:col_end] += corrected[: col_end - col_start]

        weight_scale = float(self.weight_scale) if self.weight_scale is not None else 1.0e-8
        y = (y_int.astype(np.float32) * x_scale * weight_scale).reshape(*original_shape, self.out_dim)
        return round_float_format(y, self.float_format)

    def _active_blocks(
        self,
        active_indices: np.ndarray | None,
        dim: int,
        tile_size: int,
        total_blocks: int,
    ) -> list[int]:
        if active_indices is None:
            return list(range(total_blocks))
        indices = np.asarray(active_indices, dtype=np.int64).reshape(-1)
        if indices.size == 0:
            return []
        valid = indices[(indices >= 0) & (indices < int(dim))]
        if valid.size == 0:
            return []
        return sorted(int(x) for x in np.unique(valid // int(tile_size)))

    def crossbar_infos(self) -> list[dict[str, object]]:
        return [self._programmed[key].get_info() for key in sorted(self._programmed_tiles)]

    def programmed_cell_count(self) -> int:
        total = 0
        for info in self.crossbar_infos():
            reliability_stats = info.get("reliability_stats", {})
            if isinstance(reliability_stats, dict):
                total += int(reliability_stats.get("programmed_cells", 0))
        return total

    def latency_model(self) -> dict[str, int]:
        totals = {
            "ecc_program_latency_cycles": 0,
            "ecc_compute_latency_cycles": 0,
            "ecc_total_latency_cycles": 0,
        }
        for info in self.crossbar_infos():
            latency = info.get("latency_model", {})
            if not isinstance(latency, dict):
                continue
            for key in totals:
                totals[key] += int(latency.get(key, 0))
        return totals

    def pe_output_bandwidth(self, tokens: int = 1) -> dict[str, int]:
        totals = {
            "row_groups": 1,
            "active_columns": 0,
            "payload_bits_per_value": 0,
            "payload_bits_per_token": 0,
            "wire_bits_per_token": 0,
            "wire_bits": 0,
        }
        for key in sorted(self._programmed_tiles):
            _r_block, c_block = key
            crossbar = self._programmed[key]
            col_start = c_block * self.col_tile
            col_end = min(col_start + self.col_tile, self.out_dim)
            bw = crossbar.estimate_output_bandwidth(logical_outputs=col_end - col_start)
            totals["row_groups"] = max(totals["row_groups"], int(bw["row_groups"]))
            totals["active_columns"] = max(totals["active_columns"], int(bw["active_columns_per_step"]))
            totals["payload_bits_per_value"] = max(totals["payload_bits_per_value"], int(bw["payload_bits_per_value"]))
            totals["payload_bits_per_token"] += int(bw["payload_bits"])
            totals["wire_bits_per_token"] += int(bw["wire_bits"])
        totals["wire_bits"] = totals["wire_bits_per_token"] * int(tokens)
        return totals

    def stats(self, tokens: int = 0) -> LinearStats:
        latency = self.latency_model()
        bandwidth = self.pe_output_bandwidth(tokens=tokens)
        return LinearStats(
            name=self.name,
            input_dim=self.in_dim,
            output_dim=self.out_dim,
            rows=self.row_tile,
            tokens=int(tokens),
            crossbar_tiles=len(self._programmed_tiles),
            macs=int(tokens) * self.in_dim * self.out_dim,
            ecc_program_latency_cycles=latency["ecc_program_latency_cycles"],
            ecc_compute_latency_cycles=latency["ecc_compute_latency_cycles"],
            ecc_total_latency_cycles=latency["ecc_total_latency_cycles"],
            pe_row_groups=bandwidth["row_groups"],
            pe_active_columns=bandwidth["active_columns"],
            pe_output_payload_bits_per_value=bandwidth["payload_bits_per_value"],
            pe_output_payload_bits_per_token=bandwidth["payload_bits_per_token"],
            pe_output_wire_bits_per_token=bandwidth["wire_bits_per_token"],
            pe_output_wire_bits=bandwidth["wire_bits"],
        )


class PersistentDynamicAttentionHeadPE:
    """Persistent dynamic attention PE state for one decoder layer/head."""

    def __init__(
        self,
        *,
        layer_idx: int,
        head_idx: int,
        head_dim: int,
        physical_slots: int,
        linear_kwargs: dict[str, object],
        qkt_tdc_config: CrossbarTDCConfig,
        av_tdc_config: CrossbarTDCConfig,
        float_format: str,
    ):
        self.layer_idx = int(layer_idx)
        self.head_idx = int(head_idx)
        self.head_dim = int(head_dim)
        self.physical_slots = int(physical_slots)
        self.key_matrix = PersistentDynamicAttentionMatrix(
            f"L{self.layer_idx:02d}_H{self.head_idx:02d}_{ATTN_QKT_ROLE}",
            self.head_dim,
            self.physical_slots,
            linear_kwargs=linear_kwargs,
            tdc_config=qkt_tdc_config,
            float_format=float_format,
        )
        self.value_matrix = PersistentDynamicAttentionMatrix(
            f"L{self.layer_idx:02d}_H{self.head_idx:02d}_{ATTN_AV_ROLE}",
            self.physical_slots,
            self.head_dim,
            linear_kwargs=linear_kwargs,
            tdc_config=av_tdc_config,
            float_format=float_format,
        )

    def append_token(
        self,
        *,
        key_slot: int,
        value_slot: int,
        key_vector: np.ndarray,
        value_vector: np.ndarray,
    ) -> None:
        key = np.asarray(key_vector, dtype=np.float32).reshape(self.head_dim, 1)
        value = np.asarray(value_vector, dtype=np.float32).reshape(1, self.head_dim)
        self.key_matrix.program_patch(0, int(key_slot), key)
        self.value_matrix.program_patch(int(value_slot), 0, value)

    def compute_scores(self, query: np.ndarray, key_slots: np.ndarray) -> np.ndarray:
        slots = np.asarray(key_slots, dtype=np.int64).reshape(-1)
        if slots.size == 0:
            return np.zeros((0,), dtype=np.float32)
        q = np.asarray(query, dtype=np.float32).reshape(1, self.head_dim)
        full_scores = self.key_matrix(q, active_output_indices=slots)
        return full_scores.reshape(self.physical_slots)[slots].astype(np.float32, copy=False)

    def compute_value(self, probabilities: np.ndarray, value_slots: np.ndarray) -> np.ndarray:
        slots = np.asarray(value_slots, dtype=np.int64).reshape(-1)
        probs = np.asarray(probabilities, dtype=np.float32).reshape(-1)
        if slots.size != probs.size:
            raise ValueError(f"value slot count {slots.size} does not match probabilities {probs.size}")
        if slots.size == 0:
            return np.zeros((self.head_dim,), dtype=np.float32)
        x = np.zeros((1, self.physical_slots), dtype=np.float32)
        x[0, slots] = probs
        return self.value_matrix(x, active_input_indices=slots).reshape(self.head_dim).astype(np.float32, copy=False)


def _accumulate_persistent_attention_matrix_stats(
    records: dict[tuple[int, str], dict[str, object]],
    *,
    layer_idx: int,
    role: str,
    matrix: PersistentDynamicAttentionMatrix,
) -> None:
    record = records.get((int(layer_idx), str(role)))
    if record is None:
        return
    stat = matrix.stats()
    record["crossbar_tiles"] = int(record["crossbar_tiles"]) + int(stat.crossbar_tiles)
    record["dynamic_programmed_cells"] = int(record["dynamic_programmed_cells"]) + matrix.programmed_cell_count()
    record["ecc_program_latency_cycles"] = int(record["ecc_program_latency_cycles"]) + int(
        stat.ecc_program_latency_cycles
    )
    record["ecc_compute_latency_cycles"] = int(record["ecc_compute_latency_cycles"]) + int(
        stat.ecc_compute_latency_cycles
    )
    record["ecc_total_latency_cycles"] = int(record["ecc_total_latency_cycles"]) + int(stat.ecc_total_latency_cycles)
    record["programmed_tile_updates"] = int(record["programmed_tile_updates"]) + int(matrix.programmed_tile_updates)
    record["weight_clip_events"] = int(record["weight_clip_events"]) + int(matrix.weight_clip_events)


def build_weights(rng: np.random.Generator, d_model: int, heads: int, d_ff: int, layers: int) -> list[dict[str, np.ndarray]]:
    del heads
    weights = []
    for _ in range(layers):
        weights.append(
            {
                "qkv": rng.normal(0.0, 0.02, size=(d_model, 3 * d_model)).astype(np.float32),
                "out_proj": rng.normal(0.0, 0.02, size=(d_model, d_model)).astype(np.float32),
                "mlp_up_gate": rng.normal(0.0, 0.02, size=(d_model, 2 * d_ff)).astype(np.float32),
                "mlp_down": rng.normal(0.0, 0.02, size=(d_ff, d_model)).astype(np.float32),
            }
        )
    return weights


def build_embedding_table(rng: np.random.Generator, vocab_size: int, d_model: int) -> np.ndarray:
    vocab = max(1, int(vocab_size))
    return rng.normal(0.0, 0.25, size=(vocab, int(d_model))).astype(np.float32)


def build_lm_head_weight(rng: np.random.Generator, d_model: int, vocab_size: int) -> np.ndarray:
    vocab = max(1, int(vocab_size))
    return rng.normal(0.0, 0.02, size=(int(d_model), vocab)).astype(np.float32)


def build_non_pe_runtime_modules(
    *,
    embedding_table: np.ndarray,
    float_format: str,
    sampler_rng: np.random.Generator,
) -> tuple[ModuleCallRecorder, dict[str, object]]:
    recorder = ModuleCallRecorder()
    arithmetic = BF16ArithmeticModule(float_format=float_format, recorder=recorder)
    silu = SiLUModule(arithmetic=arithmetic, recorder=recorder)
    modules: dict[str, object] = {
        "arithmetic": arithmetic,
        "embedding": EmbeddingLookupModule(embedding_table, arithmetic=arithmetic, recorder=recorder),
        "hidden_buffer": HiddenVectorBufferModule(arithmetic=arithmetic, recorder=recorder),
        "rms_norm": RMSNormModule(arithmetic=arithmetic, recorder=recorder),
        "qk_norm": QKNormModule(arithmetic=arithmetic, recorder=recorder),
        "rope": RoPEModule(arithmetic=arithmetic, recorder=recorder),
        "causal_mask": CausalMaskModule(arithmetic=arithmetic, recorder=recorder),
        "softmax": SoftmaxModule(arithmetic=arithmetic, recorder=recorder),
        "silu": silu,
        "swiglu": SwiGLUMultiplyModule(silu, arithmetic=arithmetic, recorder=recorder),
        "residual_add": ResidualAddModule(arithmetic=arithmetic, recorder=recorder),
        "sampler": TokenSamplingModule(recorder=recorder, rng=sampler_rng),
    }
    return recorder, modules


def _dynamic_attention_matmul(
    name: str,
    x_float: np.ndarray,
    weight_float: np.ndarray,
    *,
    linear_kwargs: dict[str, object],
    tdc_config: CrossbarTDCConfig,
    attention_records: dict[tuple[int, str], dict[str, object]],
    layer_idx: int,
    role: str,
    heads: int,
    context_tokens: int,
    float_format: str,
) -> np.ndarray:
    dynamic_kwargs = dict(linear_kwargs)
    dynamic_kwargs["tdc_config"] = tdc_config
    executor = ReRAMLinearExecutor(name, weight_float, **dynamic_kwargs)
    y = executor(x_float)
    _record_dynamic_attention_stat(
        attention_records,
        layer_idx=layer_idx,
        role=role,
        heads=heads,
        context_tokens=context_tokens,
        executor=executor,
        tdc_config=tdc_config,
        float_format=float_format,
    )
    return y


def run_transformer_demo(
    seq_len: int,
    d_model: int,
    heads: int,
    d_ff: int,
    layers: int,
    seed: int,
    verbose_crossbar: bool = False,
    crossbar_features: CrossbarFeatures | dict | None = None,
    noise_config: CrossbarNoiseConfig | dict | None = None,
    reliability_config: CrossbarReliabilityConfig | dict | None = None,
    iv_config: CrossbarIVNonlinearityConfig | dict | None = None,
    tdc_config: CrossbarTDCConfig | dict | None = None,
    row_parallel_config: CrossbarPERowParallelConfig | dict | None = None,
    column_parallel_config: CrossbarPEColumnParallelConfig | dict | None = None,
    noise_seed: int | None = None,
    reliability_seed: int | None = None,
    float_format: str = "bf16",
    vocab_size: int = 256,
) -> dict[str, object]:
    if d_model % heads != 0:
        raise ValueError("d_model must be divisible by heads.")
    features = CrossbarFeatures.from_obj(crossbar_features)
    noise_cfg = CrossbarNoiseConfig.from_obj(noise_config if noise_config is not None else features.noise)
    reliability_cfg = CrossbarReliabilityConfig.from_obj(
        reliability_config if reliability_config is not None else features.reliability
    )
    iv_cfg = CrossbarIVNonlinearityConfig.from_obj(iv_config if iv_config is not None else features.iv)
    tdc_cfg = CrossbarTDCConfig.from_obj(tdc_config if tdc_config is not None else features.tdc)
    row_parallel_cfg = CrossbarPERowParallelConfig.from_obj(
        row_parallel_config if row_parallel_config is not None else features.row_parallel
    )
    column_parallel_cfg = CrossbarPEColumnParallelConfig.from_obj(
        column_parallel_config if column_parallel_config is not None else features.column_parallel
    )
    noise_seed = features.noise_seed if noise_seed is None else noise_seed
    reliability_seed = features.reliability_seed if reliability_seed is None else reliability_seed
    linear_kwargs = {
        "verbose_crossbar": verbose_crossbar,
        "noise_config": noise_cfg,
        "reliability_config": reliability_cfg,
        "iv_config": iv_cfg,
        "tdc_config": tdc_cfg,
        "row_parallel_config": row_parallel_cfg,
        "column_parallel_config": column_parallel_cfg,
        "noise_seed": noise_seed,
        "reliability_seed": reliability_seed,
        "float_format": float_format,
    }
    rng = np.random.default_rng(seed)
    runtime_rng = np.random.default_rng(derive_seed(seed, "non_pe_runtime_modules"))
    sampling_rng = np.random.default_rng(derive_seed(seed, "token_sampling"))
    head_dim = d_model // heads
    _legacy_input_draw = rng.normal(0.0, 0.25, size=(seq_len, d_model)).astype(np.float32)
    all_weights = build_weights(rng, d_model, heads, d_ff, layers)
    vocab_size = max(1, int(vocab_size))
    token_ids = runtime_rng.integers(0, vocab_size, size=(seq_len,), dtype=np.int64)
    embedding_table = build_embedding_table(runtime_rng, vocab_size, d_model)
    lm_head_weight = build_lm_head_weight(runtime_rng, d_model, vocab_size)
    module_recorder, runtime_modules = build_non_pe_runtime_modules(
        embedding_table=embedding_table,
        float_format=float_format,
        sampler_rng=sampling_rng,
    )
    embedding = runtime_modules["embedding"]
    hidden_buffer = runtime_modules["hidden_buffer"]
    rms_norm = runtime_modules["rms_norm"]
    qk_norm = runtime_modules["qk_norm"]
    rope = runtime_modules["rope"]
    causal_mask = runtime_modules["causal_mask"]
    softmax_module = runtime_modules["softmax"]
    swiglu_module = runtime_modules["swiglu"]
    residual_add = runtime_modules["residual_add"]
    sampler = runtime_modules["sampler"]
    x = hidden_buffer(embedding(token_ids, stage="input_embedding"), stage="hidden_vector_buffer")
    linear_stats: list[LinearStats] = []

    for layer_idx, layer_weights in enumerate(all_weights):
        qkv = ReRAMLinearExecutor(f"L{layer_idx:02d}_qkv", layer_weights["qkv"], **linear_kwargs)
        out_proj = ReRAMLinearExecutor(f"L{layer_idx:02d}_out_proj", layer_weights["out_proj"], **linear_kwargs)
        mlp_up_gate = ReRAMLinearExecutor(f"L{layer_idx:02d}_mlp_up_gate", layer_weights["mlp_up_gate"], **linear_kwargs)
        mlp_down = ReRAMLinearExecutor(f"L{layer_idx:02d}_mlp_down", layer_weights["mlp_down"], **linear_kwargs)

        residual = x
        normed = rms_norm(x, stage=f"L{layer_idx:02d}_attn_rmsnorm")
        qkv_out = qkv(normed)
        q, k, v = np.split(qkv_out, 3, axis=-1)
        q = q.reshape(seq_len, heads, head_dim).transpose(1, 0, 2)
        k = k.reshape(seq_len, heads, head_dim).transpose(1, 0, 2)
        v = v.reshape(seq_len, heads, head_dim).transpose(1, 0, 2)
        q, k = qk_norm(q, k, stage=f"L{layer_idx:02d}_qk_norm")
        q, k = rope(q, k, positions=np.arange(seq_len, dtype=np.int64), stage=f"L{layer_idx:02d}_rope")
        scores = np.matmul(q, np.swapaxes(k, 1, 2)) / np.sqrt(float(head_dim))
        scores = causal_mask(scores, stage=f"L{layer_idx:02d}_causal_mask")
        attn_probs = softmax_module(scores, stage=f"L{layer_idx:02d}_softmax")
        attn = np.matmul(attn_probs, v).transpose(1, 0, 2).reshape(seq_len, d_model)
        x = residual_add(residual, out_proj(attn), stage=f"L{layer_idx:02d}_attn_residual_add")

        residual = x
        normed = rms_norm(x, stage=f"L{layer_idx:02d}_ffn_rmsnorm")
        mlp_hidden = swiglu_module(mlp_up_gate(normed), stage=f"L{layer_idx:02d}_swiglu")
        x = residual_add(residual, mlp_down(mlp_hidden), stage=f"L{layer_idx:02d}_ffn_residual_add")

        linear_stats.extend(
            [
                qkv.stats(seq_len),
                out_proj.stats(seq_len),
                mlp_up_gate.stats(seq_len),
                mlp_down.stats(seq_len),
            ]
        )

    lm_head = ReRAMLinearExecutor("lm_head", lm_head_weight, **linear_kwargs)
    final_hidden = rms_norm(x, stage="final_rmsnorm")
    logits = lm_head(final_hidden)
    sampled_token_ids = sampler(logits, stage="final_token_sampling")
    linear_stats.append(lm_head.stats(seq_len))

    total_tiles = sum(stat.crossbar_tiles for stat in linear_stats)
    total_macs = sum(stat.macs for stat in linear_stats)
    total_ecc_program_latency = sum(stat.ecc_program_latency_cycles for stat in linear_stats)
    total_ecc_compute_latency = sum(stat.ecc_compute_latency_cycles for stat in linear_stats)
    return {
        "seq_len": seq_len,
        "d_model": d_model,
        "heads": heads,
        "head_dim": head_dim,
        "d_ff": d_ff,
        "layers": layers,
        "vocab_size": vocab_size,
        "seed": seed,
        "input_token_ids": token_ids.astype(int).tolist(),
        "sampled_token_ids": sampled_token_ids.astype(int).tolist(),
        "noise_enabled": noise_cfg.enabled,
        "noise_seed": noise_seed,
        "noise_model": NOISE_MODEL_DESCRIPTION,
        "noise_config": noise_cfg.as_dict(),
        "reliability_enabled": reliability_cfg.enabled,
        "reliability_seed": reliability_seed,
        "reliability_model": RELIABILITY_MODEL_DESCRIPTION,
        "reliability_config": reliability_cfg.as_dict(),
        "iv_nonlinearity_enabled": iv_cfg.enabled,
        "iv_nonlinearity_model": IV_NONLINEARITY_MODEL_DESCRIPTION,
        "iv_nonlinearity_config": iv_cfg.as_dict(),
        "tdc_enabled": tdc_cfg.enabled,
        "tdc_model": TDC_MODEL_DESCRIPTION,
        "tdc_config": tdc_cfg.as_dict(),
        "pe_row_parallel_model": PE_ROW_PARALLEL_MODEL_DESCRIPTION,
        "pe_row_parallel_config": row_parallel_cfg.as_dict(),
        "pe_column_parallel_model": PE_COLUMN_PARALLEL_MODEL_DESCRIPTION,
        "pe_column_parallel_config": column_parallel_cfg.as_dict(),
        "float_format": normalize_float_format(float_format),
        "crossbar_shape": "64 rows x 64 physical cols, 32 signed logical cols",
        "non_pe_module_catalog": runtime_module_catalog(),
        "non_pe_module_execution": module_recorder.summary(),
        "linear_ops": [stat.__dict__ for stat in linear_stats],
        "total_crossbar_tiles_programmed": total_tiles,
        "total_linear_macs": total_macs,
        "total_ecc_program_latency_cycles": int(total_ecc_program_latency),
        "total_ecc_compute_latency_cycles": int(total_ecc_compute_latency),
        "total_ecc_latency_cycles": int(total_ecc_program_latency + total_ecc_compute_latency),
        "output_shape": list(x.shape),
        "output_l2_norm": float(np.linalg.norm(x)),
        "output_checksum": float(np.sum(x)),
        "final_logits_shape": list(logits.shape),
        "final_logits_checksum": float(np.sum(logits)),
    }


def run_transformer_decode_demo(
    seq_len: int,
    d_model: int,
    heads: int,
    d_ff: int,
    layers: int,
    seed: int,
    verbose_crossbar: bool = False,
    crossbar_features: CrossbarFeatures | dict | None = None,
    noise_config: CrossbarNoiseConfig | dict | None = None,
    reliability_config: CrossbarReliabilityConfig | dict | None = None,
    iv_config: CrossbarIVNonlinearityConfig | dict | None = None,
    tdc_config: CrossbarTDCConfig | dict | None = None,
    row_parallel_config: CrossbarPERowParallelConfig | dict | None = None,
    column_parallel_config: CrossbarPEColumnParallelConfig | dict | None = None,
    noise_seed: int | None = None,
    reliability_seed: int | None = None,
    float_format: str = "bf16",
    kv_cache_max_tokens: int | None = None,
    kv_cache_token_time_s: float = DEFAULT_KV_CACHE_TOKEN_TIME_S,
    kv_cache_spare_slots: int = 0,
    kv_cache_endurance_cycles: int = 10_000_000,
    kv_cache_remap_life_fraction: float = 0.80,
    kv_cache_bad_cell_probability: float = 0.0,
    kv_cache_seed: int | None = None,
    kv_cache_service_life_years: float = DEFAULT_KV_CACHE_SERVICE_LIFE_YEARS,
    attn_av_tdc_fine_bits: int = DEFAULT_ATTN_AV_TDC_FINE_BITS,
    force_attention_tdc: bool = True,
    vocab_size: int = 256,
) -> dict[str, object]:
    if d_model % heads != 0:
        raise ValueError("d_model must be divisible by heads.")
    features = CrossbarFeatures.from_obj(crossbar_features)
    noise_cfg = CrossbarNoiseConfig.from_obj(noise_config if noise_config is not None else features.noise)
    reliability_cfg = CrossbarReliabilityConfig.from_obj(
        reliability_config if reliability_config is not None else features.reliability
    )
    iv_cfg = CrossbarIVNonlinearityConfig.from_obj(iv_config if iv_config is not None else features.iv)
    tdc_cfg = CrossbarTDCConfig.from_obj(tdc_config if tdc_config is not None else features.tdc)
    row_parallel_cfg = CrossbarPERowParallelConfig.from_obj(
        row_parallel_config if row_parallel_config is not None else features.row_parallel
    )
    column_parallel_cfg = CrossbarPEColumnParallelConfig.from_obj(
        column_parallel_config if column_parallel_config is not None else features.column_parallel
    )
    noise_seed = features.noise_seed if noise_seed is None else noise_seed
    reliability_seed = features.reliability_seed if reliability_seed is None else reliability_seed
    linear_kwargs = {
        "verbose_crossbar": verbose_crossbar,
        "noise_config": noise_cfg,
        "reliability_config": reliability_cfg,
        "iv_config": iv_cfg,
        "tdc_config": tdc_cfg,
        "row_parallel_config": row_parallel_cfg,
        "column_parallel_config": column_parallel_cfg,
        "noise_seed": noise_seed,
        "reliability_seed": reliability_seed,
        "float_format": float_format,
    }
    rng = np.random.default_rng(seed)
    runtime_rng = np.random.default_rng(derive_seed(seed, "non_pe_runtime_modules"))
    sampling_rng = np.random.default_rng(derive_seed(seed, "token_sampling"))
    head_dim = d_model // heads
    _legacy_input_draw = rng.normal(0.0, 0.25, size=(seq_len, d_model)).astype(np.float32)
    all_weights = build_weights(rng, d_model, heads, d_ff, layers)
    vocab_size = max(1, int(vocab_size))
    token_ids = runtime_rng.integers(0, vocab_size, size=(seq_len,), dtype=np.int64)
    embedding_table = build_embedding_table(runtime_rng, vocab_size, d_model)
    lm_head_weight = build_lm_head_weight(runtime_rng, d_model, vocab_size)
    module_recorder, runtime_modules = build_non_pe_runtime_modules(
        embedding_table=embedding_table,
        float_format=float_format,
        sampler_rng=sampling_rng,
    )
    embedding = runtime_modules["embedding"]
    hidden_buffer = runtime_modules["hidden_buffer"]
    rms_norm = runtime_modules["rms_norm"]
    qk_norm = runtime_modules["qk_norm"]
    rope = runtime_modules["rope"]
    causal_mask = runtime_modules["causal_mask"]
    softmax_module = runtime_modules["softmax"]
    swiglu_module = runtime_modules["swiglu"]
    residual_add = runtime_modules["residual_add"]
    sampler = runtime_modules["sampler"]
    qkt_tdc_cfg = tdc_config_for_attention_role(
        tdc_cfg,
        ATTN_QKT_ROLE,
        attn_av_tdc_fine_bits=attn_av_tdc_fine_bits,
        force_sar_tdc=force_attention_tdc,
    )
    av_tdc_cfg = tdc_config_for_attention_role(
        tdc_cfg,
        ATTN_AV_ROLE,
        attn_av_tdc_fine_bits=attn_av_tdc_fine_bits,
        force_sar_tdc=force_attention_tdc,
    )
    executors: list[dict[str, ReRAMLinearExecutor]] = []
    for layer_idx, layer_weights in enumerate(all_weights):
        executors.append(
            {
                "qkv": ReRAMLinearExecutor(f"L{layer_idx:02d}_qkv", layer_weights["qkv"], **linear_kwargs),
                "out_proj": ReRAMLinearExecutor(
                    f"L{layer_idx:02d}_out_proj", layer_weights["out_proj"], **linear_kwargs
                ),
                "mlp_up_gate": ReRAMLinearExecutor(
                    f"L{layer_idx:02d}_mlp_up_gate", layer_weights["mlp_up_gate"], **linear_kwargs
                ),
                "mlp_down": ReRAMLinearExecutor(
                    f"L{layer_idx:02d}_mlp_down", layer_weights["mlp_down"], **linear_kwargs
                ),
            }
        )
    lm_head = ReRAMLinearExecutor("lm_head", lm_head_weight, **linear_kwargs)

    cache_tokens = int(kv_cache_max_tokens) if kv_cache_max_tokens is not None else DEFAULT_KV_CACHE_CONTEXT_TOKENS
    cache = StatefulKVCache(
        StatefulKVCacheConfig(
            layers=layers,
            heads=heads,
            head_dim=head_dim,
            max_tokens=max(1, cache_tokens),
            token_time_s=kv_cache_token_time_s,
            spare_slots=kv_cache_spare_slots,
            endurance_cycles=kv_cache_endurance_cycles,
            remap_life_fraction=kv_cache_remap_life_fraction,
            bad_cell_probability=kv_cache_bad_cell_probability,
            seed=kv_cache_seed if kv_cache_seed is not None else derive_seed(seed, "stateful_kv_cache"),
            service_life_years=kv_cache_service_life_years,
        )
    )
    kv_cache_control = KVCacheControlModule(cache, recorder=module_recorder)
    attention_states: list[list[PersistentDynamicAttentionHeadPE]] = [
        [
            PersistentDynamicAttentionHeadPE(
                layer_idx=layer_idx,
                head_idx=head_idx,
                head_dim=head_dim,
                physical_slots=cache.physical_slots,
                linear_kwargs=linear_kwargs,
                qkt_tdc_config=qkt_tdc_cfg,
                av_tdc_config=av_tdc_cfg,
                float_format=float_format,
            )
            for head_idx in range(heads)
        ]
        for layer_idx in range(layers)
    ]

    outputs: list[np.ndarray] = []
    logits_per_token: list[np.ndarray] = []
    sampled_token_ids: list[int] = []
    cache_trace: list[dict[str, object]] = []
    attention_records: dict[tuple[int, str], dict[str, object]] = {}
    for token_idx in range(seq_len):
        token_id = token_ids[token_idx : token_idx + 1]
        x_token = hidden_buffer(
            embedding(token_id, stage=f"T{token_idx:04d}_embedding"),
            stage=f"T{token_idx:04d}_hidden_vector_buffer",
        )
        now_s = float(token_idx) * max(float(kv_cache_token_time_s), 0.0)
        for layer_idx, ops in enumerate(executors):
            residual = x_token
            normed = rms_norm(x_token, stage=f"T{token_idx:04d}_L{layer_idx:02d}_attn_rmsnorm")
            qkv_out = ops["qkv"](normed)
            q, k, v = np.split(qkv_out, 3, axis=-1)
            q = q.reshape(1, heads, head_dim).transpose(1, 0, 2)
            k_heads = k.reshape(1, heads, head_dim).transpose(1, 0, 2)
            q, k_heads = qk_norm(q, k_heads, stage=f"T{token_idx:04d}_L{layer_idx:02d}_qk_norm")
            q, k_heads = rope(
                q,
                k_heads,
                positions=np.array([token_idx], dtype=np.int64),
                stage=f"T{token_idx:04d}_L{layer_idx:02d}_rope",
            )
            k_vec = k_heads[:, 0, :]
            v_vec = v.reshape(heads, head_dim)
            cached = kv_cache_control.append_and_read(
                layer_idx=layer_idx,
                token_idx=token_idx,
                key=k_vec,
                value=v_vec,
                now_s=now_s,
                stage=f"T{token_idx:04d}_L{layer_idx:02d}_kv_cache",
            )
            context_tokens = int(cached.token_ids.shape[0])
            current_positions = np.flatnonzero(cached.token_ids == int(token_idx))
            if current_positions.size == 0:
                raise RuntimeError(f"new token {token_idx} is missing from the stateful K/V cache")
            current_pos = int(current_positions[-1])
            for head_idx, pe_state in enumerate(attention_states[layer_idx]):
                pe_state.append_token(
                    key_slot=int(cached.key_physical_slots[current_pos]),
                    value_slot=int(cached.value_physical_slots[current_pos]),
                    key_vector=cached.keys[current_pos, head_idx],
                    value_vector=cached.values[current_pos, head_idx],
                )

            key_slots = cached.key_physical_slots.astype(np.int64, copy=False)
            value_slots = cached.value_physical_slots.astype(np.int64, copy=False)
            q_heads = q.reshape(heads, head_dim)
            scores = np.zeros((heads, 1, context_tokens), dtype=np.float32)
            for head_idx in range(heads):
                scores[head_idx, 0] = attention_states[layer_idx][head_idx].compute_scores(
                    q_heads[head_idx],
                    key_slots,
                )
                _record_dynamic_attention_compute_call(
                    attention_records,
                    layer_idx=layer_idx,
                    role=ATTN_QKT_ROLE,
                    heads=heads,
                    context_tokens=context_tokens,
                    input_dim=head_dim,
                    output_dim=context_tokens,
                    macs=head_dim * context_tokens,
                    tdc_config=qkt_tdc_cfg,
                    float_format=float_format,
                )
            scores = scores / np.sqrt(float(head_dim))
            scores = causal_mask(
                scores,
                query_positions=np.array([token_idx], dtype=np.int64),
                key_positions=cached.token_ids,
                stage=f"T{token_idx:04d}_L{layer_idx:02d}_causal_mask",
            )

            probs = softmax_module(scores, stage=f"T{token_idx:04d}_L{layer_idx:02d}_softmax").astype(
                np.float32,
                copy=False,
            )
            head_outputs = np.zeros((heads, head_dim), dtype=np.float32)
            for head_idx in range(heads):
                head_outputs[head_idx] = attention_states[layer_idx][head_idx].compute_value(
                    probs[head_idx, 0],
                    value_slots,
                )
                _record_dynamic_attention_compute_call(
                    attention_records,
                    layer_idx=layer_idx,
                    role=ATTN_AV_ROLE,
                    heads=heads,
                    context_tokens=context_tokens,
                    input_dim=context_tokens,
                    output_dim=head_dim,
                    macs=context_tokens * head_dim,
                    tdc_config=av_tdc_cfg,
                    float_format=float_format,
                )
            attn = head_outputs.reshape(1, d_model)
            x_token = residual_add(
                residual,
                ops["out_proj"](attn),
                stage=f"T{token_idx:04d}_L{layer_idx:02d}_attn_residual_add",
            )

            residual = x_token
            normed = rms_norm(x_token, stage=f"T{token_idx:04d}_L{layer_idx:02d}_ffn_rmsnorm")
            mlp_hidden = swiglu_module(ops["mlp_up_gate"](normed), stage=f"T{token_idx:04d}_L{layer_idx:02d}_swiglu")
            x_token = residual_add(
                residual,
                ops["mlp_down"](mlp_hidden),
                stage=f"T{token_idx:04d}_L{layer_idx:02d}_ffn_residual_add",
            )

        final_hidden = rms_norm(x_token, stage=f"T{token_idx:04d}_final_rmsnorm")
        logits = lm_head(final_hidden)
        logits_per_token.append(logits.reshape(-1))
        sampled = sampler(logits, stage=f"T{token_idx:04d}_token_sampling")
        sampled_token_ids.append(int(sampled.reshape(-1)[0]))
        outputs.append(x_token.reshape(d_model))
        token_summary = cache.step_summary(now_s=now_s)
        token_summary["token_idx"] = int(token_idx)
        cache_trace.append(token_summary)

    x = np.stack(outputs, axis=0).astype(np.float32) if outputs else np.zeros((0, d_model), dtype=np.float32)
    logits_out = (
        np.stack(logits_per_token, axis=0).astype(np.float32)
        if logits_per_token
        else np.zeros((0, vocab_size), dtype=np.float32)
    )
    linear_stats: list[LinearStats] = []
    for ops in executors:
        linear_stats.extend(
            [
                ops["qkv"].stats(seq_len),
                ops["out_proj"].stats(seq_len),
                ops["mlp_up_gate"].stats(seq_len),
                ops["mlp_down"].stats(seq_len),
            ]
        )
    linear_stats.append(lm_head.stats(seq_len))

    total_tiles = sum(stat.crossbar_tiles for stat in linear_stats)
    total_macs = sum(stat.macs for stat in linear_stats)
    total_ecc_program_latency = sum(stat.ecc_program_latency_cycles for stat in linear_stats)
    total_ecc_compute_latency = sum(stat.ecc_compute_latency_cycles for stat in linear_stats)
    for layer_idx, layer_states in enumerate(attention_states):
        for pe_state in layer_states:
            _accumulate_persistent_attention_matrix_stats(
                attention_records,
                layer_idx=layer_idx,
                role=ATTN_QKT_ROLE,
                matrix=pe_state.key_matrix,
            )
            _accumulate_persistent_attention_matrix_stats(
                attention_records,
                layer_idx=layer_idx,
                role=ATTN_AV_ROLE,
                matrix=pe_state.value_matrix,
            )
    attention_stats = _finalize_dynamic_attention_stats(attention_records)
    total_attention_tiles = sum(int(stat["crossbar_tiles"]) for stat in attention_stats)
    total_attention_macs = sum(int(stat["macs"]) for stat in attention_stats)
    total_attention_programmed_cells = sum(int(stat["dynamic_programmed_cells"]) for stat in attention_stats)
    total_attention_ecc_program_latency = sum(int(stat["ecc_program_latency_cycles"]) for stat in attention_stats)
    total_attention_ecc_compute_latency = sum(int(stat["ecc_compute_latency_cycles"]) for stat in attention_stats)
    now_s = float(max(seq_len - 1, 0)) * max(float(kv_cache_token_time_s), 0.0)
    return {
        "seq_len": seq_len,
        "decode_mode": "token_by_token_stateful_kv_cache",
        "d_model": d_model,
        "heads": heads,
        "head_dim": head_dim,
        "d_ff": d_ff,
        "layers": layers,
        "vocab_size": vocab_size,
        "seed": seed,
        "input_token_ids": token_ids.astype(int).tolist(),
        "sampled_token_ids": [int(x) for x in sampled_token_ids],
        "noise_enabled": noise_cfg.enabled,
        "noise_seed": noise_seed,
        "noise_model": NOISE_MODEL_DESCRIPTION,
        "noise_config": noise_cfg.as_dict(),
        "reliability_enabled": reliability_cfg.enabled,
        "reliability_seed": reliability_seed,
        "reliability_model": RELIABILITY_MODEL_DESCRIPTION,
        "reliability_config": reliability_cfg.as_dict(),
        "iv_nonlinearity_enabled": iv_cfg.enabled,
        "iv_nonlinearity_model": IV_NONLINEARITY_MODEL_DESCRIPTION,
        "iv_nonlinearity_config": iv_cfg.as_dict(),
        "tdc_enabled": tdc_cfg.enabled,
        "tdc_model": TDC_MODEL_DESCRIPTION,
        "tdc_config": tdc_cfg.as_dict(),
        "pe_row_parallel_model": PE_ROW_PARALLEL_MODEL_DESCRIPTION,
        "pe_row_parallel_config": row_parallel_cfg.as_dict(),
        "pe_column_parallel_model": PE_COLUMN_PARALLEL_MODEL_DESCRIPTION,
        "pe_column_parallel_config": column_parallel_cfg.as_dict(),
        "attention_compute_backend": _attention_backend_name(qkt_tdc_cfg),
        "attention_pe_state": "persistent_per_layer_head_append_only_kv",
        "force_attention_tdc": bool(force_attention_tdc),
        "attention_qkt_tdc_config": qkt_tdc_cfg.as_dict(),
        "attention_av_tdc_config": av_tdc_cfg.as_dict(),
        "float_format": normalize_float_format(float_format),
        "crossbar_shape": "64 rows x 64 physical cols, 32 signed logical cols",
        "non_pe_module_catalog": runtime_module_catalog(),
        "non_pe_module_execution": module_recorder.summary(),
        "linear_ops": [stat.__dict__ for stat in linear_stats],
        "attention_ops": attention_stats,
        "kv_cache": cache.summary(now_s=now_s),
        "kv_cache_token_trace": cache_trace,
        "total_static_linear_crossbar_tiles_programmed": total_tiles,
        "total_dynamic_attention_crossbar_tiles_programmed": int(total_attention_tiles),
        "total_crossbar_tiles_programmed": int(total_tiles + total_attention_tiles),
        "total_linear_macs": total_macs,
        "total_dynamic_attention_macs": int(total_attention_macs),
        "total_reram_macs": int(total_macs + total_attention_macs),
        "total_dynamic_attention_programmed_cells": int(total_attention_programmed_cells),
        "total_static_linear_ecc_program_latency_cycles": int(total_ecc_program_latency),
        "total_static_linear_ecc_compute_latency_cycles": int(total_ecc_compute_latency),
        "total_dynamic_attention_ecc_program_latency_cycles": int(total_attention_ecc_program_latency),
        "total_dynamic_attention_ecc_compute_latency_cycles": int(total_attention_ecc_compute_latency),
        "total_ecc_program_latency_cycles": int(total_ecc_program_latency + total_attention_ecc_program_latency),
        "total_ecc_compute_latency_cycles": int(total_ecc_compute_latency + total_attention_ecc_compute_latency),
        "total_ecc_latency_cycles": int(
            total_ecc_program_latency
            + total_ecc_compute_latency
            + total_attention_ecc_program_latency
            + total_attention_ecc_compute_latency
        ),
        "output_shape": list(x.shape),
        "output_l2_norm": float(np.linalg.norm(x)),
        "output_checksum": float(np.sum(x)),
        "final_logits_shape": list(logits_out.shape),
        "final_logits_checksum": float(np.sum(logits_out)),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a tiny dense LLM block on the local ReRAM crossbar.")
    parser.add_argument("--seq-len", type=int, default=4)
    parser.add_argument("--d-model", type=int, default=64)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--d-ff", type=int, default=128)
    parser.add_argument("--layers", type=int, default=1)
    parser.add_argument("--vocab-size", type=int, default=256)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--summary", default=str(RESULTS_DIR / "reram_reliability_noise_modeling_summary.json"))
    parser.add_argument("--verbose-crossbar", action="store_true")
    parser.add_argument("--float-format", choices=("bf16", "fp32"), default="bf16")
    parser.set_defaults(stateful_kv_cache=True)
    parser.add_argument(
        "--decode-token-by-token",
        dest="stateful_kv_cache",
        action="store_true",
        help="Use token-by-token decode with stateful K/V cache tracking. This is the default.",
    )
    parser.add_argument(
        "--no-stateful-kv-cache",
        dest="stateful_kv_cache",
        action="store_false",
        help="Use the legacy full-sequence attention path without stateful K/V cache telemetry.",
    )
    parser.add_argument("--kv-cache-max-tokens", type=int, default=DEFAULT_KV_CACHE_CONTEXT_TOKENS)
    parser.add_argument("--kv-cache-token-time-s", type=float, default=DEFAULT_KV_CACHE_TOKEN_TIME_S)
    parser.add_argument("--kv-cache-spare-slots", type=int, default=0)
    parser.add_argument("--kv-cache-endurance-cycles", type=int, default=10_000_000)
    parser.add_argument("--kv-cache-remap-life-fraction", type=float, default=0.80)
    parser.add_argument("--kv-cache-bad-cell-probability", type=float, default=0.0)
    parser.add_argument("--kv-cache-seed", type=int, default=None)
    parser.add_argument("--kv-cache-service-life-years", type=float, default=DEFAULT_KV_CACHE_SERVICE_LIFE_YEARS)

    noise_group = parser.add_mutually_exclusive_group()
    noise_group.add_argument(
        "--enable-noise",
        dest="enable_noise",
        action="store_true",
        help="Enable thermal, shot, flicker, RTN, and readout read-noise models. This is the default.",
    )
    noise_group.add_argument(
        "--disable-noise",
        dest="enable_noise",
        action="store_false",
        help="Disable thermal, shot, flicker, RTN, and readout read-noise models.",
    )
    parser.set_defaults(enable_noise=True)
    parser.add_argument("--noise-seed", type=int, default=None)
    parser.add_argument("--noise-vread", type=float, default=0.1)
    parser.add_argument("--noise-temp-k", type=float, default=300.0)
    parser.add_argument("--noise-fano", type=float, default=1.0)
    parser.add_argument("--noise-r-lrs-ohm", type=float, default=100_000.0)
    parser.add_argument("--noise-r-hrs-ohm", type=float, default=10_000_000.0)
    parser.add_argument("--noise-flicker-af-lrs", type=float, default=1.0e-11)
    parser.add_argument("--noise-flicker-af-hrs", type=float, default=1.0e-9)
    parser.add_argument("--noise-flicker-gamma-lrs", type=float, default=1.0)
    parser.add_argument("--noise-flicker-gamma-hrs", type=float, default=1.0)
    parser.add_argument("--noise-rtn-amp-frac-lrs", type=float, default=3.0e-4)
    parser.add_argument("--noise-rtn-amp-frac-hrs", type=float, default=2.0e-2)
    parser.add_argument("--noise-readout-a-rms", type=float, default=0.0)
    parser.add_argument("--noise-bandwidth-hz", type=float, default=5_000.0)
    parser.add_argument("--noise-flicker-fmin-hz", type=float, default=0.5)

    parser.add_argument("--enable-reliability", action="store_true")
    parser.add_argument("--reliability-seed", type=int, default=None)
    parser.add_argument("--defect-stuck-at-lrs-prob", type=float, default=0.0)
    parser.add_argument("--defect-stuck-at-hrs-prob", type=float, default=0.0)
    parser.add_argument("--reliability-raw-ber-ppm", type=float, default=0.0)
    parser.add_argument("--reliability-use-ecc", action="store_true")
    parser.add_argument("--reliability-ecc-codeword-bits", type=int, default=38)
    parser.add_argument("--reliability-ecc-data-bits", type=int, default=32)
    parser.add_argument("--reliability-ecc-encode-latency-cycles", type=int, default=1)
    parser.add_argument("--reliability-ecc-decode-latency-cycles", type=int, default=1)
    parser.add_argument("--retention-time-s", type=float, default=0.0)
    parser.add_argument("--retention-reference-s", type=float, default=1.0)
    parser.add_argument("--retention-temp-c", type=float, default=25.0)
    parser.add_argument("--retention-temp-reference-c", type=float, default=25.0)
    parser.add_argument("--retention-temp-acceleration-per-100c", type=float, default=1.0)
    parser.add_argument(
        "--retention-model",
        choices=("legacy_log_shift", "state_power_law", "reram_power_law", "power_law"),
        default="legacy_log_shift",
    )
    parser.add_argument("--retention-log10-resistance-drift-per-decade", type=float, default=0.0)
    parser.add_argument("--retention-lrs-beta-log10-per-decade", type=float, default=0.0)
    parser.add_argument("--retention-hrs-beta-log10-per-decade", type=float, default=0.0)
    parser.add_argument("--state-sigma-ln", type=float, default=0.0)
    parser.add_argument("--relaxation-time-s", type=float, default=0.0)
    parser.add_argument("--relaxation-tau-s", type=float, default=2_000.0)
    parser.add_argument("--relaxation-resistance-amplitude-ln", type=float, default=0.0)
    parser.add_argument("--endurance-write-cycles", type=int, default=0)
    parser.add_argument("--endurance-cycles", type=int, default=100_000)
    parser.add_argument("--endurance-resistance-drift-ln-per-life-fraction", type=float, default=0.0)
    parser.add_argument("--endurance-limit-stuck-at-lrs-prob", type=float, default=0.0)
    parser.add_argument("--endurance-limit-stuck-at-hrs-prob", type=float, default=0.0)
    parser.add_argument("--endurance-limit-multilevel-loss-prob", type=float, default=0.0)
    parser.add_argument("--endurance-limit-multilevel-sigma-ln", type=float, default=0.0)
    iv_group = parser.add_mutually_exclusive_group()
    iv_group.add_argument(
        "--enable-iv-nonlinearity",
        dest="enable_iv_nonlinearity",
        action="store_true",
        help="Enable hard-clamp/tanh I-V saturation at ADC full-scale. This is the default.",
    )
    iv_group.add_argument(
        "--disable-iv-nonlinearity",
        dest="enable_iv_nonlinearity",
        action="store_false",
        help="Disable hard-clamp/tanh I-V saturation at ADC full-scale.",
    )
    parser.set_defaults(enable_iv_nonlinearity=True)
    parser.add_argument("--iv-nonlinearity-mode", choices=("hard_clamp", "tanh"), default="hard_clamp")
    parser.add_argument("--iv-linear-rows-before-clamp", type=int, default=10)
    parser.add_argument("--iv-adc-full-scale-levels", type=float, default=0.0)
    parser.add_argument("--enable-tdc", action="store_true")
    parser.add_argument("--tdc-bits", type=int, default=8)
    parser.add_argument("--tdc-coarse-bits", type=int, default=5)
    parser.add_argument("--tdc-fine-bits", type=int, default=3)
    parser.add_argument("--tdc-gain-shift", type=int, default=0)
    parser.add_argument(
        "--tdc-precision-mode",
        choices=sar_tdc_precision_choices(),
        default=None,
        help="Named SAR-TDC signed payload mode: 4b=c2/f1, 8b=c4/f3, 9b=c5/f3, 11b=c5/f5, 16b=c5/f10.",
    )
    parser.add_argument("--tdc-output-format", choices=("sar_fp", "bfp"), default="sar_fp")
    parser.add_argument(
        "--enable-pe-row-parallel",
        action="store_true",
        help="Enable PE row-group sensing before digital partial-sum combine.",
    )
    parser.add_argument("--pe-row-groups", type=int, default=1)
    parser.add_argument(
        "--pe-rows-per-group",
        type=int,
        default=0,
        help="Rows per PE row group; overrides --pe-row-groups when positive.",
    )
    parser.add_argument(
        "--enable-pe-column-parallel",
        action="store_true",
        help="Enable PE column parallel sensing metadata for output bandwidth studies.",
    )
    parser.add_argument(
        "--pe-active-columns",
        type=int,
        default=0,
        help="Logical PE output columns sensed in parallel per step; 0 means all tile columns.",
    )
    parser.add_argument("--pe-packet-header-bits", type=int, default=0)
    parser.add_argument(
        "--pe-payload-bits-per-value",
        type=int,
        default=0,
        help="Override PE output payload width; 0 uses the TDC payload mode or ADC width.",
    )
    parser.add_argument(
        "--attn-av-tdc-fine-bits",
        type=int,
        default=DEFAULT_ATTN_AV_TDC_FINE_BITS,
        help="Fine SAR-TDC bits for dynamic P V attention after softmax. Total TDC bits grow to preserve coarse range.",
    )
    parser.add_argument(
        "--disable-attention-tdc",
        action="store_true",
        help="Ablation: let dynamic decode attention follow --enable-tdc instead of forcing SAR-TDC.",
    )
    return parser.parse_args()


def build_crossbar_configs(
    args: argparse.Namespace,
) -> tuple[
    CrossbarNoiseConfig,
    CrossbarReliabilityConfig,
    CrossbarIVNonlinearityConfig,
    CrossbarTDCConfig,
    CrossbarPERowParallelConfig,
    CrossbarPEColumnParallelConfig,
]:
    noise_config = CrossbarNoiseConfig(
        enabled=bool(args.enable_noise),
        vread=args.noise_vread,
        temp_k=args.noise_temp_k,
        fano=args.noise_fano,
        r_lrs_ohm=args.noise_r_lrs_ohm,
        r_hrs_ohm=args.noise_r_hrs_ohm,
        flicker_af_lrs=args.noise_flicker_af_lrs,
        flicker_af_hrs=args.noise_flicker_af_hrs,
        flicker_gamma_lrs=args.noise_flicker_gamma_lrs,
        flicker_gamma_hrs=args.noise_flicker_gamma_hrs,
        rtn_amp_frac_lrs=args.noise_rtn_amp_frac_lrs,
        rtn_amp_frac_hrs=args.noise_rtn_amp_frac_hrs,
        readout_noise_a_rms=args.noise_readout_a_rms,
        bandwidth_hz=args.noise_bandwidth_hz,
        flicker_fmin_hz=args.noise_flicker_fmin_hz,
    )
    reliability_config = CrossbarReliabilityConfig(
        enabled=bool(args.enable_reliability),
        stuck_at_lrs_prob=args.defect_stuck_at_lrs_prob,
        stuck_at_hrs_prob=args.defect_stuck_at_hrs_prob,
        raw_ber_ppm=args.reliability_raw_ber_ppm,
        use_ecc=bool(args.reliability_use_ecc),
        ecc_codeword_bits=args.reliability_ecc_codeword_bits,
        ecc_data_bits=args.reliability_ecc_data_bits,
        ecc_encode_latency_cycles=args.reliability_ecc_encode_latency_cycles,
        ecc_decode_latency_cycles=args.reliability_ecc_decode_latency_cycles,
        retention_time_s=args.retention_time_s,
        retention_reference_s=args.retention_reference_s,
        retention_temp_c=args.retention_temp_c,
        retention_temp_reference_c=args.retention_temp_reference_c,
        retention_temp_acceleration_per_100c=args.retention_temp_acceleration_per_100c,
        retention_model=args.retention_model,
        retention_log10_resistance_drift_per_decade=args.retention_log10_resistance_drift_per_decade,
        retention_lrs_beta_log10_per_decade=args.retention_lrs_beta_log10_per_decade,
        retention_hrs_beta_log10_per_decade=args.retention_hrs_beta_log10_per_decade,
        state_sigma_ln=args.state_sigma_ln,
        relaxation_time_s=args.relaxation_time_s,
        relaxation_tau_s=args.relaxation_tau_s,
        relaxation_resistance_amplitude_ln=args.relaxation_resistance_amplitude_ln,
        endurance_write_cycles=args.endurance_write_cycles,
        endurance_cycles=args.endurance_cycles,
        endurance_resistance_drift_ln_per_life_fraction=args.endurance_resistance_drift_ln_per_life_fraction,
        endurance_limit_stuck_at_lrs_prob=args.endurance_limit_stuck_at_lrs_prob,
        endurance_limit_stuck_at_hrs_prob=args.endurance_limit_stuck_at_hrs_prob,
        endurance_limit_multilevel_loss_prob=args.endurance_limit_multilevel_loss_prob,
        endurance_limit_multilevel_sigma_ln=args.endurance_limit_multilevel_sigma_ln,
    )
    iv_config = CrossbarIVNonlinearityConfig(
        enabled=bool(args.enable_iv_nonlinearity),
        mode=args.iv_nonlinearity_mode,
        adc_full_scale_levels=args.iv_adc_full_scale_levels,
        linear_rows_before_clamp=args.iv_linear_rows_before_clamp,
    )
    tdc_config = CrossbarTDCConfig(
        enabled=bool(args.enable_tdc),
        tdc_bits=args.tdc_bits,
        coarse_bits=args.tdc_coarse_bits,
        fine_bits=args.tdc_fine_bits,
        gain_shift=args.tdc_gain_shift,
        output_format=args.tdc_output_format,
        precision_mode=args.tdc_precision_mode,
    )
    row_parallel_config = CrossbarPERowParallelConfig(
        enabled=bool(args.enable_pe_row_parallel),
        groups=args.pe_row_groups,
        rows_per_group=args.pe_rows_per_group,
    )
    column_parallel_config = CrossbarPEColumnParallelConfig(
        enabled=bool(args.enable_pe_column_parallel),
        active_columns=args.pe_active_columns,
        packet_header_bits=args.pe_packet_header_bits,
        payload_bits_per_value=args.pe_payload_bits_per_value,
    )
    return noise_config, reliability_config, iv_config, tdc_config, row_parallel_config, column_parallel_config


def main() -> None:
    args = parse_args()
    (
        noise_config,
        reliability_config,
        iv_config,
        tdc_config,
        row_parallel_config,
        column_parallel_config,
    ) = build_crossbar_configs(args)
    common_kwargs = {
        "seq_len": args.seq_len,
        "d_model": args.d_model,
        "heads": args.heads,
        "d_ff": args.d_ff,
        "layers": args.layers,
        "vocab_size": args.vocab_size,
        "seed": args.seed,
        "verbose_crossbar": args.verbose_crossbar,
        "noise_config": noise_config,
        "reliability_config": reliability_config,
        "iv_config": iv_config,
        "tdc_config": tdc_config,
        "row_parallel_config": row_parallel_config,
        "column_parallel_config": column_parallel_config,
        "noise_seed": args.noise_seed,
        "reliability_seed": args.reliability_seed,
        "float_format": args.float_format,
    }
    if args.stateful_kv_cache:
        kv_cache_max_tokens = args.kv_cache_max_tokens if args.kv_cache_max_tokens > 0 else args.seq_len
        summary = run_transformer_decode_demo(
            **common_kwargs,
            kv_cache_max_tokens=kv_cache_max_tokens,
            kv_cache_token_time_s=args.kv_cache_token_time_s,
            kv_cache_spare_slots=args.kv_cache_spare_slots,
            kv_cache_endurance_cycles=args.kv_cache_endurance_cycles,
            kv_cache_remap_life_fraction=args.kv_cache_remap_life_fraction,
            kv_cache_bad_cell_probability=args.kv_cache_bad_cell_probability,
            kv_cache_seed=args.kv_cache_seed,
            kv_cache_service_life_years=args.kv_cache_service_life_years,
            attn_av_tdc_fine_bits=args.attn_av_tdc_fine_bits,
            force_attention_tdc=not bool(args.disable_attention_tdc),
        )
    else:
        summary = run_transformer_demo(**common_kwargs)
    summary_path = Path(args.summary)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote ReRAM reliability/noise modeling summary: {summary_path}")
    print(f"Output shape: {summary['output_shape']}, total crossbar tiles: {summary['total_crossbar_tiles_programmed']}")
    print(
        f"Total ReRAM MACs: {summary.get('total_reram_macs', summary['total_linear_macs'])}, "
        f"output checksum: {summary['output_checksum']:.6f}"
    )
    if "total_dynamic_attention_macs" in summary:
        print(f"Dynamic attention ReRAM MACs: {summary['total_dynamic_attention_macs']}")
    print(f"Total ECC latency cycles: {summary['total_ecc_latency_cycles']}")
    if "kv_cache" in summary:
        kv = summary["kv_cache"]
        lifetime = kv.get("append_only_lifetime_projection", {})
        print(
            "Stateful KV cache: "
            f"K writes={kv['total_key_writes']}, V writes={kv['total_value_writes']}, "
            f"K remaps={kv['key_remaps']}, V remaps={kv['value_remaps']}, "
            f"max age={kv['max_age_s']:.6g}s, "
            f"projected cycles/slot={float(lifetime.get('projected_key_write_cycles_per_slot', 0.0)):.6g}"
        )


if __name__ == "__main__":
    main()
