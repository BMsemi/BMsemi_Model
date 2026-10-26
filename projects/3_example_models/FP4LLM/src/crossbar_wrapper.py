"""
PyTorch nn.Module wrapper around the local FP4LLM/LLM_ReRAM crossbar executor.

The wrapper keeps FP4LLM's W7/A7 block-16 quantization path, uses the same
64-row x 32-signed-logical-column tile geometry as LLM_ReRAM's ReRAMLinearExecutor,
and forwards the lower-level ReRAM safety models for read noise, reliability
drift/defects, and column-current I-V / ADC saturation.
"""

from __future__ import annotations

import argparse
import contextlib
import io
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn


_SRC_DIR = Path(__file__).resolve().parent
_PROJ_DIR = _SRC_DIR.parent
_RERAM_DIR = _PROJ_DIR.parent / "LLM_ReRAM"

# Prefer the FP4LLM-local executor/crossbar; keep sibling LLM_ReRAM as fallback.
if str(_PROJ_DIR) not in sys.path:
    sys.path.insert(0, str(_PROJ_DIR))
if _RERAM_DIR.is_dir() and str(_RERAM_DIR) not in sys.path:
    sys.path.append(str(_RERAM_DIR))

from llm_reram_executor import (  # noqa: E402
    LinearStats,
    ReRAMLinearExecutor,
    derive_seed,
    normalize_float_format,
    quantize_symmetric,
    round_float_format,
)
from reram_crossbar import (  # noqa: E402
    CrossbarArray,
    CrossbarFeatures,
    CrossbarIVNonlinearityConfig,
    CrossbarNoiseConfig,
    CrossbarPEColumnParallelConfig,
    CrossbarPERowParallelConfig,
    CrossbarReliabilityConfig,
    CrossbarTDCConfig,
    TDCPackedOutput,
    sar_tdc_precision_choices,
)


DEFAULT_TILE_ROWS = 64
DEFAULT_TILE_COLS = 32
ReRAMLinearExecutor.row_tile = DEFAULT_TILE_ROWS
ReRAMLinearExecutor.col_tile = DEFAULT_TILE_COLS

# Backward-compatible names used by the docs and older integration snippets.
ReRAMNoiseConfig = CrossbarNoiseConfig
ReRAMReliabilityConfig = CrossbarReliabilityConfig
ReRAMIVNonlinearityConfig = CrossbarIVNonlinearityConfig
ReRAMTDCConfig = CrossbarTDCConfig
ReRAMPERowParallelConfig = CrossbarPERowParallelConfig
ReRAMPEColumnParallelConfig = CrossbarPEColumnParallelConfig


def quantize_block16(x: np.ndarray, bits: int = 5, block_size: int = 16) -> np.ndarray:
    """Per-block symmetric fixed-point quantization, returned as dequantized float."""

    x_arr = np.asarray(x, dtype=np.float32)
    if x_arr.ndim == 0:
        raise ValueError("quantize_block16 expects at least one dimension.")
    bits = int(bits)
    block_size = int(block_size)
    if bits >= 16:
        return x_arr.copy()
    if bits < 2:
        raise ValueError(f"bits must be >= 2, got {bits}.")
    if block_size <= 0:
        raise ValueError(f"block_size must be positive, got {block_size}.")

    orig_shape = x_arr.shape
    d = x_arr.shape[-1]
    pad = (-d) % block_size
    if pad:
        pad_width = [(0, 0)] * x_arr.ndim
        pad_width[-1] = (0, pad)
        x_arr = np.pad(x_arr, pad_width)

    x_blocks = x_arr.reshape(*x_arr.shape[:-1], -1, block_size)
    qmax = (1 << (bits - 1)) - 1
    amax = np.maximum(np.max(np.abs(x_blocks), axis=-1, keepdims=True), 1.0e-8)
    scale = amax / float(qmax)
    x_q = np.clip(np.rint(x_blocks / scale), -qmax, qmax)
    x_dq = (x_q * scale).reshape(*x_arr.shape[:-1], -1)
    return x_dq[..., :d].reshape(orig_shape).astype(np.float32, copy=False)


def weight_block16(w_in_out: np.ndarray, bits: int, block_size: int = 16) -> np.ndarray:
    """
    Per-block-16 symmetric fixed-point quantization along a weight matrix's
    input dimension. Weight shape is executor convention: (in_dim, out_dim).
    """

    w_out_in = np.asarray(w_in_out, dtype=np.float32).T
    return quantize_block16(w_out_in, bits=bits, block_size=block_size).T


def make_iv_config(
    *,
    enable_iv_nonlinearity: bool = True,
    iv_nonlinearity_mode: str = "hard_clamp",
    iv_adc_full_scale_levels: float = 0.0,
    iv_linear_rows_before_clamp: int = 10,
) -> CrossbarIVNonlinearityConfig:
    """Build the ReRAM column-current I-V / ADC saturation config."""

    return CrossbarIVNonlinearityConfig(
        enabled=bool(enable_iv_nonlinearity),
        mode=iv_nonlinearity_mode,
        adc_full_scale_levels=float(iv_adc_full_scale_levels),
        linear_rows_before_clamp=int(iv_linear_rows_before_clamp),
    )


def add_reram_iv_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """Add I-V / ADC saturation CLI flags for FP4LLM scripts."""

    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--enable-iv-nonlinearity",
        dest="enable_iv_nonlinearity",
        action="store_true",
        help="Enable many-row ReRAM column-current ADC clamp/nonlinearity. This is the default.",
    )
    group.add_argument(
        "--disable-iv-nonlinearity",
        dest="enable_iv_nonlinearity",
        action="store_false",
        help="Disable many-row ReRAM column-current ADC clamp/nonlinearity.",
    )
    parser.set_defaults(enable_iv_nonlinearity=True)
    parser.add_argument(
        "--iv-nonlinearity-mode",
        choices=("hard_clamp", "tanh"),
        default="hard_clamp",
        help="Column-current saturation model.",
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
        help="Override differential product level that reaches ADC full-scale.",
    )
    return parser


def iv_config_from_args(args: argparse.Namespace) -> CrossbarIVNonlinearityConfig:
    """Create CrossbarIVNonlinearityConfig from argparse flags."""

    return make_iv_config(
        enable_iv_nonlinearity=getattr(args, "enable_iv_nonlinearity", True),
        iv_nonlinearity_mode=getattr(args, "iv_nonlinearity_mode", "hard_clamp"),
        iv_adc_full_scale_levels=getattr(args, "iv_adc_full_scale_levels", 0.0),
        iv_linear_rows_before_clamp=getattr(args, "iv_linear_rows_before_clamp", 10),
    )


def crossbar_iv_kwargs_from_args(args: argparse.Namespace) -> dict[str, CrossbarIVNonlinearityConfig]:
    """Return kwargs suitable for CrossbarLinear(..., **kwargs)."""

    return {"iv_config": iv_config_from_args(args)}


def make_tdc_config(
    *,
    enable_tdc: bool = False,
    tdc_bits: int = 8,
    tdc_coarse_bits: int = 5,
    tdc_fine_bits: int = 3,
    tdc_gain_shift: int = 0,
    tdc_output_format: str = "sar_fp",
    tdc_precision_mode: str | None = None,
    tdc_timing_enabled: bool = True,
    tdc_timing_min_ps: float = 0.0,
    tdc_timing_max_ps: float = 576.0,
    tdc_timing_reference_levels: float = 256.0,
) -> CrossbarTDCConfig:
    """Build the differential coarse/fine TDC output config."""

    return CrossbarTDCConfig(
        enabled=bool(enable_tdc),
        tdc_bits=int(tdc_bits),
        coarse_bits=int(tdc_coarse_bits),
        fine_bits=int(tdc_fine_bits),
        gain_shift=int(tdc_gain_shift),
        output_format=str(tdc_output_format),
        precision_mode=tdc_precision_mode,
        timing_enabled=bool(tdc_timing_enabled),
        timing_min_ps=float(tdc_timing_min_ps),
        timing_max_ps=float(tdc_timing_max_ps),
        timing_reference_levels=float(tdc_timing_reference_levels),
    )


def add_reram_tdc_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """Add differential TDC CLI flags for FP4LLM scripts."""

    parser.add_argument("--enable-tdc", action="store_true", help="Enable local differential A-B/B-A TDC output.")
    parser.add_argument("--tdc-bits", type=int, default=8, help="Total unipolar TDC code bits per differential branch.")
    parser.add_argument("--tdc-coarse-bits", type=int, default=5, help="Coarse counter bits for reporting/metadata.")
    parser.add_argument("--tdc-fine-bits", type=int, default=3, help="Fine delay-chain bits for reporting/metadata.")
    parser.add_argument("--tdc-gain-shift", type=int, default=0, help="E_g shift: TDC code = round(branch / 2^E_g).")
    parser.add_argument(
        "--tdc-precision-mode",
        choices=sar_tdc_precision_choices(),
        default=None,
        help="Named signed payload SAR-TDC mode: 4b=c2/f1, 8b=c4/f3, 9b=c5/f3, 11b=c5/f5, 16b=c5/f10.",
    )
    parser.add_argument(
        "--tdc-output-format",
        choices=("sar_fp", "bfp"),
        default="sar_fp",
        help="Use SAR-FP coarse-as-exponent output or legacy uniform BFP q_y,E_g output.",
    )
    parser.add_argument("--disable-tdc-timing", action="store_true", help="Disable value-dependent TDC arrival metadata.")
    parser.add_argument("--tdc-timing-min-ps", type=float, default=0.0, help="Earliest column arrival time.")
    parser.add_argument("--tdc-timing-max-ps", type=float, default=576.0, help="Zero/small-signal timeout arrival time.")
    parser.add_argument(
        "--tdc-timing-reference-levels",
        type=float,
        default=256.0,
        help="Column magnitude that halves the TDC arrival-time span.",
    )
    return parser


def tdc_config_from_args(args: argparse.Namespace) -> CrossbarTDCConfig:
    """Create CrossbarTDCConfig from argparse flags."""

    return make_tdc_config(
        enable_tdc=getattr(args, "enable_tdc", False),
        tdc_bits=getattr(args, "tdc_bits", 8),
        tdc_coarse_bits=getattr(args, "tdc_coarse_bits", 5),
        tdc_fine_bits=getattr(args, "tdc_fine_bits", 3),
        tdc_gain_shift=getattr(args, "tdc_gain_shift", 0),
        tdc_output_format=getattr(args, "tdc_output_format", "sar_fp"),
        tdc_precision_mode=getattr(args, "tdc_precision_mode", None),
        tdc_timing_enabled=not getattr(args, "disable_tdc_timing", False),
        tdc_timing_min_ps=getattr(args, "tdc_timing_min_ps", 0.0),
        tdc_timing_max_ps=getattr(args, "tdc_timing_max_ps", 576.0),
        tdc_timing_reference_levels=getattr(args, "tdc_timing_reference_levels", 256.0),
    )


def crossbar_tdc_kwargs_from_args(args: argparse.Namespace) -> dict[str, CrossbarTDCConfig]:
    """Return kwargs suitable for CrossbarLinear(..., **kwargs)."""

    return {"tdc_config": tdc_config_from_args(args)}


def make_pe_row_parallel_config(
    *,
    enable_pe_row_parallel: bool = False,
    pe_row_groups: int = 1,
    pe_rows_per_group: int = 0,
) -> CrossbarPERowParallelConfig:
    """Build the PE row-group primitive for row-current clipping studies."""

    return CrossbarPERowParallelConfig(
        enabled=bool(enable_pe_row_parallel),
        groups=int(pe_row_groups),
        rows_per_group=int(pe_rows_per_group),
    )


def make_pe_column_parallel_config(
    *,
    enable_pe_column_parallel: bool = False,
    pe_active_columns: int = 0,
    pe_packet_header_bits: int = 0,
    pe_payload_bits_per_value: int = 0,
) -> CrossbarPEColumnParallelConfig:
    """Build the PE column-sense primitive for output bandwidth studies."""

    return CrossbarPEColumnParallelConfig(
        enabled=bool(enable_pe_column_parallel),
        active_columns=int(pe_active_columns),
        packet_header_bits=int(pe_packet_header_bits),
        payload_bits_per_value=int(pe_payload_bits_per_value),
    )


def add_reram_pe_parallel_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """Add PE row/column parallel primitive CLI flags."""

    parser.add_argument(
        "--enable-pe-row-parallel",
        action="store_true",
        help="Enable PE row-group sensing before digital partial-sum combine.",
    )
    parser.add_argument(
        "--pe-row-groups",
        type=int,
        default=1,
        help="Number of row groups sensed independently inside each PE tile.",
    )
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
        help="Logical PE output columns sensed in parallel per step; 0 means all columns in the tile.",
    )
    parser.add_argument(
        "--pe-packet-header-bits",
        type=int,
        default=0,
        help="Packet/header bits added per row-group and column-sense step.",
    )
    parser.add_argument(
        "--pe-payload-bits-per-value",
        type=int,
        default=0,
        help="Override PE output payload bits per value; 0 uses TDC signed payload or ADC bits.",
    )
    return parser


def pe_parallel_configs_from_args(
    args: argparse.Namespace,
) -> tuple[CrossbarPERowParallelConfig, CrossbarPEColumnParallelConfig]:
    """Create PE row and column primitive configs from argparse flags."""

    row_parallel_config = make_pe_row_parallel_config(
        enable_pe_row_parallel=getattr(args, "enable_pe_row_parallel", False),
        pe_row_groups=getattr(args, "pe_row_groups", 1),
        pe_rows_per_group=getattr(args, "pe_rows_per_group", 0),
    )
    column_parallel_config = make_pe_column_parallel_config(
        enable_pe_column_parallel=getattr(args, "enable_pe_column_parallel", False),
        pe_active_columns=getattr(args, "pe_active_columns", 0),
        pe_packet_header_bits=getattr(args, "pe_packet_header_bits", 0),
        pe_payload_bits_per_value=getattr(args, "pe_payload_bits_per_value", 0),
    )
    return row_parallel_config, column_parallel_config


def crossbar_pe_parallel_kwargs_from_args(
    args: argparse.Namespace,
) -> dict[str, CrossbarPERowParallelConfig | CrossbarPEColumnParallelConfig]:
    """Return PE primitive kwargs suitable for CrossbarLinear(..., **kwargs)."""

    row_parallel_config, column_parallel_config = pe_parallel_configs_from_args(args)
    return {
        "row_parallel_config": row_parallel_config,
        "column_parallel_config": column_parallel_config,
    }


class CrossbarLinear(nn.Module):
    """
    Drop-in bias-free nn.Linear replacement backed by signed INT8 ReRAM
    crossbar tiles.

    The default tile size is 64 rows x 32 signed logical columns, matching
    LLM_ReRAM's 64 physical rows x 64 physical differential columns.
    ReRAMLinearExecutor.row_tile/col_tile can still be changed
    before constructing layers when an experiment needs a different geometry.
    """

    def __init__(
        self,
        weight: np.ndarray | torch.Tensor,
        *,
        weight_bits: int = 7,
        act_bits: int = 7,
        block_size: int = 16,
        sigma: float = 0.0,
        seed: int = 42,
        verbose: bool = False,
        crossbar_features: CrossbarFeatures | dict | None = None,
        noise_config: CrossbarNoiseConfig | dict | None = None,
        noise_seed: int | None = None,
        reliability_config: CrossbarReliabilityConfig | dict | None = None,
        reliability_seed: int | None = None,
        iv_config: CrossbarIVNonlinearityConfig | dict | None = None,
        tdc_config: CrossbarTDCConfig | dict | None = None,
        row_parallel_config: CrossbarPERowParallelConfig | dict | None = None,
        column_parallel_config: CrossbarPEColumnParallelConfig | dict | None = None,
        enable_iv_nonlinearity: bool | None = None,
        iv_nonlinearity_mode: str = "hard_clamp",
        iv_adc_full_scale_levels: float = 0.0,
        iv_linear_rows_before_clamp: int = 10,
        enable_tdc: bool = False,
        tdc_bits: int = 8,
        tdc_coarse_bits: int = 5,
        tdc_fine_bits: int = 3,
        tdc_gain_shift: int = 0,
        tdc_output_format: str = "sar_fp",
        tdc_precision_mode: str | None = None,
        enable_pe_row_parallel: bool = False,
        pe_row_groups: int = 1,
        pe_rows_per_group: int = 0,
        enable_pe_column_parallel: bool = False,
        pe_active_columns: int = 0,
        pe_packet_header_bits: int = 0,
        pe_payload_bits_per_value: int = 0,
        float_format: str = "bf16",
        defer_programming: bool = False,
    ):
        super().__init__()
        self.float_format = normalize_float_format(float_format)

        if isinstance(weight, torch.Tensor):
            w = weight.detach().float().cpu().numpy()
        else:
            w = np.asarray(weight, dtype=np.float32)
        if w.ndim != 2:
            raise ValueError(f"weight must be 2-D (in_dim, out_dim), got shape {w.shape}.")

        w = round_float_format(w, self.float_format)
        if int(weight_bits) < 8:
            w = weight_block16(w, bits=weight_bits, block_size=block_size)

        if float(sigma) > 0.0:
            rng = np.random.default_rng(seed)
            w = w * (1.0 + rng.standard_normal(w.shape).astype(np.float32) * float(sigma))

        features = CrossbarFeatures.from_obj(crossbar_features)
        self.noise_config = CrossbarNoiseConfig.from_obj(
            noise_config if noise_config is not None else features.noise
        )
        self.reliability_config = CrossbarReliabilityConfig.from_obj(
            reliability_config if reliability_config is not None else features.reliability
        )
        if iv_config is None and enable_iv_nonlinearity is not None:
            iv_config = make_iv_config(
                enable_iv_nonlinearity=bool(enable_iv_nonlinearity),
                iv_nonlinearity_mode=iv_nonlinearity_mode,
                iv_adc_full_scale_levels=iv_adc_full_scale_levels,
                iv_linear_rows_before_clamp=iv_linear_rows_before_clamp,
            )
        self.iv_config = CrossbarIVNonlinearityConfig.from_obj(iv_config if iv_config is not None else features.iv)
        if tdc_config is None and enable_tdc:
            tdc_config = make_tdc_config(
                enable_tdc=True,
                tdc_bits=tdc_bits,
                tdc_coarse_bits=tdc_coarse_bits,
                tdc_fine_bits=tdc_fine_bits,
                tdc_gain_shift=tdc_gain_shift,
                tdc_output_format=tdc_output_format,
                tdc_precision_mode=tdc_precision_mode,
            )
        self.tdc_config = CrossbarTDCConfig.from_obj(tdc_config if tdc_config is not None else features.tdc)
        if row_parallel_config is None and (enable_pe_row_parallel or pe_row_groups > 1 or pe_rows_per_group > 0):
            row_parallel_config = make_pe_row_parallel_config(
                enable_pe_row_parallel=True,
                pe_row_groups=pe_row_groups,
                pe_rows_per_group=pe_rows_per_group,
            )
        if column_parallel_config is None and (
            enable_pe_column_parallel
            or pe_active_columns > 0
            or pe_packet_header_bits > 0
            or pe_payload_bits_per_value > 0
        ):
            column_parallel_config = make_pe_column_parallel_config(
                enable_pe_column_parallel=True,
                pe_active_columns=pe_active_columns,
                pe_packet_header_bits=pe_packet_header_bits,
                pe_payload_bits_per_value=pe_payload_bits_per_value,
            )
        self.row_parallel_config = CrossbarPERowParallelConfig.from_obj(
            row_parallel_config if row_parallel_config is not None else features.row_parallel
        )
        self.column_parallel_config = CrossbarPEColumnParallelConfig.from_obj(
            column_parallel_config if column_parallel_config is not None else features.column_parallel
        )

        self.noise_seed = features.noise_seed if noise_seed is None else noise_seed
        self.reliability_seed = features.reliability_seed if reliability_seed is None else reliability_seed
        if self.noise_seed is None and self.noise_config.enabled:
            self.noise_seed = int(seed) + 1_000_003
        if self.reliability_seed is None and self.reliability_config.enabled:
            self.reliability_seed = int(seed) + 2_000_003

        self.act_bits = int(act_bits)
        self.block_size = int(block_size)
        self.weight_bits = int(weight_bits)
        self.sigma = float(sigma)

        self._executor = ReRAMLinearExecutor(
            "crossbar_linear",
            w,
            verbose_crossbar=verbose,
            act_bits=self.act_bits,
            act_block_size=self.block_size,
            noise_config=self.noise_config,
            reliability_config=self.reliability_config,
            iv_config=self.iv_config,
            tdc_config=self.tdc_config,
            row_parallel_config=self.row_parallel_config,
            column_parallel_config=self.column_parallel_config,
            noise_seed=self.noise_seed,
            reliability_seed=self.reliability_seed,
            float_format=self.float_format,
            defer_programming=defer_programming,
        )

        self.in_features = self._executor.in_dim
        self.out_features = self._executor.out_dim

    @classmethod
    def from_linear(cls, linear: nn.Linear, **kwargs: Any) -> "CrossbarLinear":
        """Build from nn.Linear by transposing torch's (out, in) weight layout."""

        w_in_out = linear.weight.detach().float().cpu().numpy().T
        return cls(w_in_out, **kwargs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run crossbar-backed linear projection on the same device as x."""

        device = x.device
        x_np = x.detach().float().cpu().numpy()
        y_np = self._executor(x_np)
        return torch.from_numpy(y_np.astype(np.float32, copy=False)).to(device)

    def compute_pwm_int8_serial(self, x: np.ndarray | torch.Tensor) -> np.ndarray | torch.Tensor:
        """
        Explicit CrossbarArray MAC path using the already-programmed executor
        weights and the configured 16x16 tile geometry.
        """

        input_is_torch = isinstance(x, torch.Tensor)
        device = x.device if input_is_torch else None
        x_np = x.detach().float().cpu().numpy() if input_is_torch else np.asarray(x, dtype=np.float32)
        x_np = round_float_format(x_np, self.float_format)

        if x_np.shape[-1] != self.in_features:
            raise ValueError(f"input last dimension must be {self.in_features}, got {x_np.shape[-1]}")

        tile_rows = self._executor.row_tile
        tile_cols = self._executor.col_tile
        original_shape = x_np.shape[:-1]
        x_2d = x_np.reshape(-1, self.in_features)

        x_block = quantize_block16(x_2d, bits=self.act_bits, block_size=self.block_size)
        x_q, x_scale = quantize_symmetric(x_block)

        row_tiles = int(np.ceil(self.in_features / tile_rows))
        col_tiles = int(np.ceil(self.out_features / tile_cols))
        y_int = np.zeros((x_2d.shape[0], self.out_features), dtype=np.int64)

        for r_block in range(row_tiles):
            row_start = r_block * tile_rows
            row_end = min(row_start + tile_rows, self.in_features)

            for c_block in range(col_tiles):
                col_start = c_block * tile_cols
                col_end = min(col_start + tile_cols, self.out_features)

                weight_tile = np.zeros((tile_rows, tile_cols), dtype=np.int8)
                weight_tile[: row_end - row_start, : col_end - col_start] = self._executor.weight_q[
                    row_start:row_end, col_start:col_end
                ]

                with contextlib.redirect_stdout(io.StringIO()):
                    crossbar = CrossbarArray(
                        input_size=tile_rows,
                        output_size=tile_cols * 2,
                        noise_config=self.noise_config,
                        reliability_config=self.reliability_config,
                        iv_config=self.iv_config,
                        tdc_config=self.tdc_config,
                        row_parallel_config=self.row_parallel_config,
                        column_parallel_config=self.column_parallel_config,
                        noise_seed=derive_seed(self.noise_seed, "compute_pwm_int8_serial", r_block, c_block),
                        reliability_seed=derive_seed(
                            self.reliability_seed, "compute_pwm_int8_serial", r_block, c_block
                        ),
                    )
                    crossbar.program_weights(weight_tile)

                for token_idx, token in enumerate(x_q.astype(np.int16)):
                    x_tile_signed = np.zeros((tile_rows,), dtype=np.int16)
                    x_tile_signed[: row_end - row_start] = token[row_start:row_end]
                    x_tile_unsigned = (x_tile_signed + 128).astype(np.uint8)

                    with contextlib.redirect_stdout(io.StringIO()):
                        if crossbar.tdc_config.enabled:
                            if str(crossbar.tdc_config.output_format).lower() in {"sar", "sar_fp", "fp", "float", "float_like"}:
                                packed = crossbar.compute_tdc_packed_corrected(x_tile_unsigned, zero_point=128)
                                corrected = packed.reconstruct_integer()[:tile_cols]
                            else:
                                packed = crossbar.compute_tdc_packed(x_tile_unsigned)
                                zero = crossbar.pwm_zero_point_packed_output(128, exponent=packed.exponent)
                                corrected = (
                                    packed.reconstruct_integer()[:tile_cols]
                                    - zero.reconstruct_integer()[:tile_cols]
                                )
                        else:
                            raw = crossbar.compute(x_tile_unsigned)
                            logical = np.asarray(raw[:tile_cols], dtype=np.int64)
                            offset_correction = crossbar.pwm_zero_point_output(128)[:tile_cols].astype(np.int64)
                            corrected = logical - offset_correction
                    y_int[token_idx, col_start:col_end] += corrected[: col_end - col_start]

        y = y_int.astype(np.float32) * x_scale * self._executor.weight_scale
        y = y.reshape(*original_shape, self.out_features).astype(np.float32)
        y = round_float_format(y, self.float_format)
        if input_is_torch:
            return torch.from_numpy(y).to(device)
        return y

    def stats(self, tokens: int = 1) -> LinearStats:
        """Return tile/MAC stats from the underlying executor."""

        return self._executor.stats(tokens)

    def extra_repr(self) -> str:
        iv = "off"
        if self.iv_config.enabled:
            iv = (
                f"{self.iv_config.mode},fs={self.iv_config.adc_full_scale_levels},"
                f"rows={self.iv_config.linear_rows_before_clamp}"
            )
        tdc = "off"
        if self.tdc_config.enabled:
            tdc = (
                f"{self.tdc_config.tdc_bits}b,"
                f"coarse={self.tdc_config.coarse_bits},fine={self.tdc_config.fine_bits},"
                f"payload={self.tdc_config.signed_payload_bits}b,"
                f"mode={self.tdc_config.precision_mode},"
                f"shift={self.tdc_config.gain_shift},fmt={self.tdc_config.output_format}"
            )
        row_parallel = "off"
        if self.row_parallel_config.enabled:
            row_parallel = (
                f"groups={self.row_parallel_config.groups},"
                f"rows_per_group={self.row_parallel_config.rows_per_group}"
            )
        column_parallel = "off"
        if self.column_parallel_config.enabled:
            column_parallel = (
                f"active_cols={self.column_parallel_config.active_columns},"
                f"header={self.column_parallel_config.packet_header_bits},"
                f"payload_override={self.column_parallel_config.payload_bits_per_value}"
            )
        return (
            f"in={self.in_features}, out={self.out_features}, "
            f"weight_bits={self.weight_bits}, act_bits={self.act_bits}, "
            f"block={self.block_size}, sigma={self.sigma}, float_format={self.float_format}, "
            f"iv={iv}, tdc={tdc}, row_parallel={row_parallel}, column_parallel={column_parallel}"
        )


__all__ = [
    "CrossbarArray",
    "CrossbarLinear",
    "LinearStats",
    "CrossbarFeatures",
    "CrossbarIVNonlinearityConfig",
    "CrossbarNoiseConfig",
    "CrossbarPEColumnParallelConfig",
    "CrossbarPERowParallelConfig",
    "CrossbarReliabilityConfig",
    "CrossbarTDCConfig",
    "TDCPackedOutput",
    "ReRAMIVNonlinearityConfig",
    "ReRAMNoiseConfig",
    "ReRAMPEColumnParallelConfig",
    "ReRAMPERowParallelConfig",
    "ReRAMReliabilityConfig",
    "ReRAMTDCConfig",
    "add_reram_iv_args",
    "add_reram_pe_parallel_args",
    "add_reram_tdc_args",
    "crossbar_iv_kwargs_from_args",
    "crossbar_pe_parallel_kwargs_from_args",
    "crossbar_tdc_kwargs_from_args",
    "iv_config_from_args",
    "make_iv_config",
    "make_pe_column_parallel_config",
    "make_pe_row_parallel_config",
    "make_tdc_config",
    "normalize_float_format",
    "pe_parallel_configs_from_args",
    "quantize_block16",
    "quantize_symmetric",
    "round_float_format",
    "sar_tdc_precision_choices",
    "tdc_config_from_args",
    "weight_block16",
]
