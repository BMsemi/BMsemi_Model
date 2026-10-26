from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields, replace
from typing import Any, Mapping

import numpy as np


K_BOLTZMANN = 1.380649e-23
Q_ELECTRON = 1.602176634e-19


SAR_TDC_SIGNED_PAYLOAD_MODES: dict[str, dict[str, int]] = {
    "4b": {"coarse_bits": 2, "fine_bits": 1, "tdc_bits": 3, "signed_payload_bits": 4},
    "8b": {"coarse_bits": 4, "fine_bits": 3, "tdc_bits": 7, "signed_payload_bits": 8},
    "9b": {"coarse_bits": 5, "fine_bits": 3, "tdc_bits": 8, "signed_payload_bits": 9},
    "11b": {"coarse_bits": 5, "fine_bits": 5, "tdc_bits": 10, "signed_payload_bits": 11},
    "16b": {"coarse_bits": 5, "fine_bits": 10, "tdc_bits": 15, "signed_payload_bits": 16},
}

_SAR_TDC_PRECISION_ALIASES = {
    "c2/f1": "4b",
    "c4/f3": "8b",
    "c5/f3": "9b",
    "c5/f5": "11b",
    "c5/f10": "16b",
}


def normalize_sar_tdc_precision_mode(precision_mode: Any) -> str | None:
    if precision_mode is None:
        return None
    mode = str(precision_mode).strip().lower().replace(" ", "")
    if mode in {"", "none", "off", "manual"}:
        return None
    if mode.isdigit():
        mode = f"{mode}b"
    if mode.endswith("bit"):
        mode = f"{mode[:-3]}b"
    mode = _SAR_TDC_PRECISION_ALIASES.get(mode, mode)
    if mode not in SAR_TDC_SIGNED_PAYLOAD_MODES:
        choices = ", ".join(sorted(SAR_TDC_SIGNED_PAYLOAD_MODES))
        raise ValueError(f"Unsupported SAR-TDC precision_mode {precision_mode!r}; use one of: {choices}.")
    return mode


def sar_tdc_precision_settings(precision_mode: Any) -> dict[str, int | str] | None:
    mode = normalize_sar_tdc_precision_mode(precision_mode)
    if mode is None:
        return None
    return {"precision_mode": mode, **SAR_TDC_SIGNED_PAYLOAD_MODES[mode]}


def sar_tdc_precision_choices() -> tuple[str, ...]:
    return tuple(SAR_TDC_SIGNED_PAYLOAD_MODES.keys())


def _coerce_dataclass(cls, value: Any):
    if value is None:
        return cls()
    if isinstance(value, cls):
        return value
    if isinstance(value, Mapping):
        names = {item.name for item in fields(cls)}
        return cls(**{key: val for key, val in value.items() if key in names})
    raise TypeError(f"Cannot convert {type(value)!r} to {cls.__name__}.")


@dataclass(frozen=True)
class CrossbarNoiseConfig:
    """Transient read-noise knobs for INT4 ReRAM cell reads.

    The default resistance calibration targets about 1 uA from a fully-on
    cell at Vread=0.1 V, with a maximum ON/OFF ratio of 100x.
    """

    enabled: bool = True
    vread: float = 0.1
    temp_k: float = 300.0
    fano: float = 1.0
    r_lrs_ohm: float = 100_000.0
    r_hrs_ohm: float = 10_000_000.0
    flicker_af_lrs: float = 1.0e-11
    flicker_af_hrs: float = 1.0e-9
    flicker_gamma_lrs: float = 1.0
    flicker_gamma_hrs: float = 1.0
    rtn_amp_frac_lrs: float = 3.0e-4
    rtn_amp_frac_hrs: float = 2.0e-2
    readout_noise_a_rms: float = 0.0
    bandwidth_hz: float = 5_000.0
    flicker_fmin_hz: float = 0.5

    @classmethod
    def from_obj(cls, value: Any) -> "CrossbarNoiseConfig":
        return _coerce_dataclass(cls, value)

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class CrossbarReliabilityConfig:
    """Persistent programmed-cell reliability knobs."""

    enabled: bool = False
    stuck_at_lrs_prob: float = 0.0
    stuck_at_hrs_prob: float = 0.0
    raw_ber_ppm: float = 0.0
    use_ecc: bool = False
    ecc_codeword_bits: int = 38
    ecc_data_bits: int = 32
    ecc_encode_latency_cycles: int = 1
    ecc_decode_latency_cycles: int = 1
    retention_time_s: float = 0.0
    retention_reference_s: float = 1.0
    retention_temp_c: float = 25.0
    retention_temp_reference_c: float = 25.0
    retention_temp_acceleration_per_100c: float = 1.0
    retention_model: str = "legacy_log_shift"
    retention_log10_resistance_drift_per_decade: float = 0.0
    retention_lrs_beta_log10_per_decade: float = 0.0
    retention_hrs_beta_log10_per_decade: float = 0.0
    state_sigma_ln: float = 0.0
    relaxation_time_s: float = 0.0
    relaxation_tau_s: float = 2_000.0
    relaxation_resistance_amplitude_ln: float = 0.0
    endurance_write_cycles: int = 0
    endurance_cycles: int = 100_000
    endurance_resistance_drift_ln_per_life_fraction: float = 0.0
    endurance_limit_stuck_at_lrs_prob: float = 0.0
    endurance_limit_stuck_at_hrs_prob: float = 0.0
    endurance_limit_multilevel_loss_prob: float = 0.0
    endurance_limit_multilevel_sigma_ln: float = 0.0

    @classmethod
    def from_obj(cls, value: Any) -> "CrossbarReliabilityConfig":
        return _coerce_dataclass(cls, value)

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class CrossbarIVNonlinearityConfig:
    """Column-current saturation knobs for many-row crossbar reads."""

    enabled: bool = True
    mode: str = "hard_clamp"
    adc_full_scale_levels: float = 0.0
    linear_rows_before_clamp: int = 10

    @classmethod
    def from_obj(cls, value: Any) -> "CrossbarIVNonlinearityConfig":
        return _coerce_dataclass(cls, value)

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class CrossbarTDCConfig:
    """Coarse/fine TDC output conversion after local differential subtraction."""

    enabled: bool = False
    tdc_bits: int = 8
    coarse_bits: int = 5
    fine_bits: int = 3
    gain_shift: int = 0
    output_format: str = "sar_fp"
    precision_mode: str | None = None
    signed_payload_bits: int = 0
    timing_enabled: bool = True
    timing_min_ps: float = 0.0
    timing_max_ps: float = 576.0
    timing_reference_levels: float = 256.0

    def __post_init__(self) -> None:
        mode = normalize_sar_tdc_precision_mode(self.precision_mode)
        if mode is not None:
            settings = SAR_TDC_SIGNED_PAYLOAD_MODES[mode]
            object.__setattr__(self, "precision_mode", mode)
            object.__setattr__(self, "coarse_bits", int(settings["coarse_bits"]))
            object.__setattr__(self, "fine_bits", int(settings["fine_bits"]))
            object.__setattr__(self, "tdc_bits", int(settings["tdc_bits"]))
            object.__setattr__(self, "signed_payload_bits", int(settings["signed_payload_bits"]))
        else:
            object.__setattr__(self, "precision_mode", None)
            payload_bits = int(self.signed_payload_bits)
            if payload_bits <= 0:
                payload_bits = int(self.coarse_bits) + int(self.fine_bits) + 1
            object.__setattr__(self, "signed_payload_bits", payload_bits)

        timing_min_ps = float(self.timing_min_ps)
        timing_max_ps = float(self.timing_max_ps)
        timing_reference_levels = float(self.timing_reference_levels)
        if timing_max_ps < timing_min_ps:
            raise ValueError("TDC timing_max_ps must be greater than or equal to timing_min_ps.")
        if timing_reference_levels <= 0.0:
            raise ValueError("TDC timing_reference_levels must be positive.")
        object.__setattr__(self, "timing_enabled", bool(self.timing_enabled))
        object.__setattr__(self, "timing_min_ps", timing_min_ps)
        object.__setattr__(self, "timing_max_ps", timing_max_ps)
        object.__setattr__(self, "timing_reference_levels", timing_reference_levels)

    @classmethod
    def from_obj(cls, value: Any) -> "CrossbarTDCConfig":
        return _coerce_dataclass(cls, value)

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class CrossbarPERowParallelConfig:
    """PE row-sensing grouping for added-current clipping studies."""

    enabled: bool = False
    groups: int = 1
    rows_per_group: int = 0
    combine: str = "digital_sum"

    def __post_init__(self) -> None:
        groups = max(int(self.groups), 1)
        rows_per_group = max(int(self.rows_per_group), 0)
        enabled = bool(self.enabled or groups > 1 or rows_per_group > 0)
        combine = str(self.combine).lower()
        if combine not in {"digital_sum"}:
            raise ValueError("CrossbarPERowParallelConfig.combine currently supports only 'digital_sum'.")
        object.__setattr__(self, "enabled", enabled)
        object.__setattr__(self, "groups", groups)
        object.__setattr__(self, "rows_per_group", rows_per_group)
        object.__setattr__(self, "combine", combine)

    @classmethod
    def from_obj(cls, value: Any) -> "CrossbarPERowParallelConfig":
        return _coerce_dataclass(cls, value)

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class CrossbarPEColumnParallelConfig:
    """PE column-sensing parallelism and output packetization metadata."""

    enabled: bool = False
    active_columns: int = 0
    packet_header_bits: int = 0
    payload_bits_per_value: int = 0

    def __post_init__(self) -> None:
        active_columns = max(int(self.active_columns), 0)
        packet_header_bits = max(int(self.packet_header_bits), 0)
        payload_bits_per_value = max(int(self.payload_bits_per_value), 0)
        enabled = bool(self.enabled or active_columns > 0 or packet_header_bits > 0 or payload_bits_per_value > 0)
        object.__setattr__(self, "enabled", enabled)
        object.__setattr__(self, "active_columns", active_columns)
        object.__setattr__(self, "packet_header_bits", packet_header_bits)
        object.__setattr__(self, "payload_bits_per_value", payload_bits_per_value)

    @classmethod
    def from_obj(cls, value: Any) -> "CrossbarPEColumnParallelConfig":
        return _coerce_dataclass(cls, value)

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class TDCPackedOutput:
    """
    Packed TDC boundary value.

    In the legacy BFP format, the TDC emits a signed mantissa-like code q_y.
    The shared exponent E_g says how many raw integer MAC units one TDC LSB
    represents:

        raw_integer ~= q_y * 2**E_g

    In the default SAR-FP format, the coarse phase is a leading-one/range
    detector and the fine phase captures mantissa bits below the leading one:

        raw_integer ~= signed_mantissa * 2**(exponent - mantissa_fraction_bits)
    """

    mantissa: np.ndarray
    exponent: int | np.ndarray
    positive_code: np.ndarray
    negative_code: np.ndarray
    positive_coarse: np.ndarray
    positive_fine: np.ndarray
    negative_coarse: np.ndarray
    negative_fine: np.ndarray
    format: str = "tdc_sar_fp"
    mantissa_fraction_bits: int = 0
    arrival_time_ps: np.ndarray | None = None
    pe_completion_time_ps: float = 0.0

    @property
    def completion_order(self) -> np.ndarray:
        """Logical column indices ordered from earliest to latest completion."""

        if self.arrival_time_ps is None:
            return np.zeros((0,), dtype=np.int32)
        arrival = np.asarray(self.arrival_time_ps, dtype=float)
        return np.argsort(arrival, kind="stable").astype(np.int32)

    @property
    def arrival_spread_ps(self) -> float:
        """Difference between the latest and earliest logical-column arrivals."""

        if self.arrival_time_ps is None:
            return 0.0
        arrival = np.asarray(self.arrival_time_ps, dtype=float)
        return float(np.max(arrival) - np.min(arrival)) if arrival.size else 0.0

    @property
    def decode_scale(self) -> int | np.ndarray:
        exp = np.asarray(self.exponent)
        if self.format == "tdc_sar_fp":
            scale_exp = exp - int(self.mantissa_fraction_bits)
        else:
            scale_exp = exp
        if np.ndim(scale_exp) == 0:
            shift = int(scale_exp)
            if shift >= 0:
                return 1 << shift
            return 2.0**shift
        return np.exp2(scale_exp.astype(np.float32))

    def reconstruct_integer(self) -> np.ndarray:
        if self.format in {"tdc_integer", "tdc_row_group_sum"}:
            return self.mantissa.astype(np.int64)
        if self.format == "tdc_sar_fp":
            exp = np.asarray(self.exponent, dtype=np.int32)
            shift = exp - int(self.mantissa_fraction_bits)
            values = np.ldexp(self.mantissa.astype(np.float64), shift)
            return np.rint(values).astype(np.int64)
        return self.mantissa.astype(np.int64) * int(self.decode_scale)

    def reconstruct_float(self, extra_scale: float = 1.0) -> np.ndarray:
        if self.format == "tdc_sar_fp":
            return self.reconstruct_integer().astype(np.float32) * float(extra_scale)
        return self.mantissa.astype(np.float32) * float(self.decode_scale) * float(extra_scale)

    def as_dict(self) -> dict[str, Any]:
        mantissa = np.asarray(self.mantissa)
        exp = np.asarray(self.exponent)
        decode_scale = self.decode_scale
        scale = np.asarray(decode_scale)
        out = {
            "format": self.format,
            "mantissa_shape": list(mantissa.shape),
            "mantissa_dtype": str(mantissa.dtype),
            "mantissa_min": int(np.min(mantissa)) if mantissa.size else 0,
            "mantissa_max": int(np.max(mantissa)) if mantissa.size else 0,
            "mantissa_fraction_bits": int(self.mantissa_fraction_bits),
        }
        if exp.ndim == 0:
            out["exponent"] = int(exp)
        else:
            out["exponent_shape"] = list(exp.shape)
            out["exponent_min"] = int(np.min(exp)) if exp.size else 0
            out["exponent_max"] = int(np.max(exp)) if exp.size else 0
        if scale.ndim == 0:
            out["decode_scale"] = float(scale) if float(scale) % 1 else int(scale)
        else:
            out["decode_scale_min"] = float(np.min(scale)) if scale.size else 0.0
            out["decode_scale_max"] = float(np.max(scale)) if scale.size else 0.0
        arrival = None if self.arrival_time_ps is None else np.asarray(self.arrival_time_ps, dtype=float)
        out["timing_enabled"] = bool(arrival is not None)
        if arrival is not None:
            out["arrival_time_min_ps"] = float(np.min(arrival)) if arrival.size else 0.0
            out["arrival_time_max_ps"] = float(np.max(arrival)) if arrival.size else 0.0
            out["arrival_spread_ps"] = self.arrival_spread_ps
            out["pe_completion_time_ps"] = float(self.pe_completion_time_ps)
            order = self.completion_order
            out["first_ready_column"] = int(order[0]) if order.size else -1
            out["last_ready_column"] = int(order[-1]) if order.size else -1
        return out


@dataclass(frozen=True)
class CrossbarFeatures:
    """Convenience bundle accepted by executors and wrappers."""

    noise: CrossbarNoiseConfig = field(default_factory=CrossbarNoiseConfig)
    reliability: CrossbarReliabilityConfig = field(default_factory=CrossbarReliabilityConfig)
    iv: CrossbarIVNonlinearityConfig = field(default_factory=CrossbarIVNonlinearityConfig)
    tdc: CrossbarTDCConfig = field(default_factory=CrossbarTDCConfig)
    row_parallel: CrossbarPERowParallelConfig = field(default_factory=CrossbarPERowParallelConfig)
    column_parallel: CrossbarPEColumnParallelConfig = field(default_factory=CrossbarPEColumnParallelConfig)
    noise_seed: int | None = None
    reliability_seed: int | None = None

    @classmethod
    def from_obj(cls, value: Any) -> "CrossbarFeatures":
        if value is None:
            return cls()
        if isinstance(value, cls):
            return value
        if isinstance(value, Mapping):
            noise = value.get("noise", value.get("noise_config"))
            reliability = value.get("reliability", value.get("reliability_config"))
            iv = value.get("iv", value.get("iv_config"))
            tdc = value.get("tdc", value.get("tdc_config"))
            row_parallel = value.get(
                "row_parallel",
                value.get("row_parallel_config", value.get("pe_row_parallel_config")),
            )
            column_parallel = value.get(
                "column_parallel",
                value.get("column_parallel_config", value.get("pe_column_parallel_config")),
            )
            return cls(
                noise=CrossbarNoiseConfig.from_obj(noise),
                reliability=CrossbarReliabilityConfig.from_obj(reliability),
                iv=CrossbarIVNonlinearityConfig.from_obj(iv),
                tdc=CrossbarTDCConfig.from_obj(tdc),
                row_parallel=CrossbarPERowParallelConfig.from_obj(row_parallel),
                column_parallel=CrossbarPEColumnParallelConfig.from_obj(column_parallel),
                noise_seed=value.get("noise_seed"),
                reliability_seed=value.get("reliability_seed"),
            )
        raise TypeError(f"Cannot convert {type(value)!r} to CrossbarFeatures.")

    def as_dict(self) -> dict[str, Any]:
        return {
            "noise": self.noise.as_dict(),
            "reliability": self.reliability.as_dict(),
            "iv": self.iv.as_dict(),
            "tdc": self.tdc.as_dict(),
            "row_parallel": self.row_parallel.as_dict(),
            "column_parallel": self.column_parallel.as_dict(),
            "noise_seed": self.noise_seed,
            "reliability_seed": self.reliability_seed,
        }


class CrossbarArray:
    """Standalone ReRAM crossbar with signed row-serial INT8 execution."""

    MODE_UNSIGNED = "UNSIGNED"
    MODE_SIGNED = "SIGNED"
    MODE_PWM_INT8_ROW_SERIAL = "PWM_INT8_ROW_SERIAL"
    MODE_PWM_INT5_DIFFERENTIAL = "PWM_INT5_DIFFERENTIAL"

    def __init__(
        self,
        input_size: int = 64,
        output_size: int = 64,
        *,
        adc_bits: int = 8,
        cell_bits: int = 4,
        slice_gain0: float = 1.0,
        slice_gain1: float = 1.0,
        safety_checks: bool = True,
        crossbar_features: CrossbarFeatures | Mapping[str, Any] | None = None,
        noise_config: CrossbarNoiseConfig | Mapping[str, Any] | None = None,
        reliability_config: CrossbarReliabilityConfig | Mapping[str, Any] | None = None,
        iv_config: CrossbarIVNonlinearityConfig | Mapping[str, Any] | None = None,
        tdc_config: CrossbarTDCConfig | Mapping[str, Any] | None = None,
        row_parallel_config: CrossbarPERowParallelConfig | Mapping[str, Any] | None = None,
        column_parallel_config: CrossbarPEColumnParallelConfig | Mapping[str, Any] | None = None,
        noise_seed: int | None = None,
        reliability_seed: int | None = None,
    ):
        self.safety_checks = bool(safety_checks)
        if self.safety_checks:
            self._validate_constructor_args(input_size, output_size, adc_bits, cell_bits)

        self.input_size = int(input_size)
        self.output_size = int(output_size)
        self.adc_bits = int(adc_bits)
        self.cell_bits = int(cell_bits)
        self.slice_gain0 = float(slice_gain0)
        self.slice_gain1 = float(slice_gain1)
        self.mode: str | None = None

        features = CrossbarFeatures.from_obj(crossbar_features)
        self.noise_config = CrossbarNoiseConfig.from_obj(noise_config if noise_config is not None else features.noise)
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
        self._noise_rng = np.random.default_rng(self.noise_seed)
        self._reliability_rng = np.random.default_rng(self.reliability_seed)
        self._ecc_program_latency_cycles = 0
        self._ecc_compute_latency_cycles = 0
        self._last_tdc_timing: dict[str, float | int | bool] = {
            "enabled": False,
            "first_ready_column": -1,
            "last_ready_column": -1,
            "arrival_time_min_ps": 0.0,
            "arrival_time_max_ps": 0.0,
            "arrival_spread_ps": 0.0,
            "pe_completion_time_ps": 0.0,
        }
        self._reliability_stats = self._new_reliability_stats()

        logical_outputs = self.output_size // 2
        self._weights_internal = np.zeros((self.input_size, self.output_size), dtype=float)
        self._effective_signed: np.ndarray | None = None
        self._w_pos0 = np.zeros((self.input_size, logical_outputs), dtype=np.uint8)
        self._w_pos1 = np.zeros((self.input_size, logical_outputs), dtype=np.uint8)
        self._w_neg0 = np.zeros((self.input_size, logical_outputs), dtype=np.uint8)
        self._w_neg1 = np.zeros((self.input_size, logical_outputs), dtype=np.uint8)
        self._state_log_resistance = self._make_state_log_resistance()
        self._iv_saturation_stats = self._new_iv_saturation_stats(logical_outputs)

    def _new_reliability_stats(self) -> dict[str, int]:
        return {
            "programmed_cells": 0,
            "stuck_at_lrs_cells": 0,
            "stuck_at_hrs_cells": 0,
            "raw_ber_bit_flips": 0,
            "endurance_limit_stuck_at_lrs_cells": 0,
            "endurance_limit_stuck_at_hrs_cells": 0,
            "endurance_limit_multilevel_loss_cells": 0,
        }

    def _new_iv_saturation_stats(self, logical_outputs: int) -> dict[str, object]:
        return {
            "transfer_calls": 0,
            "upper_saturation_events": 0,
            "lower_clip_events": 0,
            "saturated_columns": np.zeros((max(int(logical_outputs), 0),), dtype=bool),
        }

    def _validate_constructor_args(self, input_size: int, output_size: int, adc_bits: int, cell_bits: int) -> None:
        if int(input_size) <= 0:
            raise ValueError(f"input_size must be positive, got {input_size}")
        if int(output_size) <= 0:
            raise ValueError(f"output_size must be positive, got {output_size}")
        if int(output_size) % 2 != 0:
            raise ValueError(f"output_size must be even for differential outputs, got {output_size}")
        if int(adc_bits) <= 0:
            raise ValueError(f"adc_bits must be positive, got {adc_bits}")
        if int(cell_bits) <= 0:
            raise ValueError(f"cell_bits must be positive, got {cell_bits}")

    def _validate_finite(self, values: np.ndarray, name: str) -> None:
        arr = np.asarray(values, dtype=float)
        if not np.all(np.isfinite(arr)):
            raise ValueError(f"{name} contains NaN or infinite values")

    def _validate_2d_weights(self, weight_matrix: np.ndarray) -> None:
        if np.asarray(weight_matrix).ndim != 2:
            raise ValueError(f"weight_matrix must be 2D, got shape {np.asarray(weight_matrix).shape}")

    def _validate_pwm_weight_range(self, weight_matrix: np.ndarray) -> None:
        values = np.asarray(weight_matrix, dtype=np.int64)
        if values.size and (np.min(values) < -128 or np.max(values) > 127):
            raise ValueError("PWM_INT8_ROW_SERIAL weights must be signed INT8 values in [-128, 127]")

    def _validate_legacy_input_vector(self, input_vector: np.ndarray) -> np.ndarray:
        x = np.asarray(input_vector, dtype=float)
        if x.ndim != 1:
            raise ValueError(f"Input vector must be 1D, got shape {x.shape}")
        if x.shape[0] != self.input_size:
            raise ValueError(f"Input size mismatch: expected {self.input_size}, got {x.shape[0]}")
        self._validate_finite(x, "input_vector")
        return x

    def _validate_pwm_input_vector(self, input_vector: np.ndarray) -> np.ndarray:
        a = np.asarray(input_vector)
        if a.ndim != 1:
            raise ValueError(f"Input vector must be 1D, got shape {a.shape}")
        if a.shape[0] != self.input_size:
            raise ValueError(f"Input size mismatch: expected {self.input_size}, got {a.shape[0]}")

        values = np.asarray(a, dtype=float)
        self._validate_finite(values, "input_vector")
        if values.size and (np.min(values) < 0 or np.max(values) > 255):
            raise ValueError("PWM_INT8_ROW_SERIAL inputs must be unsigned activation values in [0, 255]")
        return a

    @property
    def weights(self) -> np.ndarray:
        return self._weights_internal

    @weights.setter
    def weights(self, val: np.ndarray) -> None:
        arr = np.asarray(val, dtype=float)
        if self.safety_checks:
            self._validate_finite(arr, "weights")
        self._weights_internal = arr

    def program_weights(self, weight_matrix: np.ndarray) -> None:
        self._ecc_program_latency_cycles = 0
        self._ecc_compute_latency_cycles = 0
        self._last_tdc_timing.update(
            {
                "enabled": False,
                "first_ready_column": -1,
                "last_ready_column": -1,
                "arrival_time_min_ps": 0.0,
                "arrival_time_max_ps": 0.0,
                "arrival_spread_ps": 0.0,
                "pe_completion_time_ps": 0.0,
            }
        )
        self._reliability_stats = self._new_reliability_stats()
        wm = np.asarray(weight_matrix)
        if self.safety_checks:
            self._validate_2d_weights(wm)
        logical_outputs = self.output_size // 2

        if wm.shape == (self.input_size, logical_outputs) and np.issubdtype(wm.dtype, np.integer):
            if self.safety_checks:
                self._validate_pwm_weight_range(wm)
            self.mode = self.MODE_PWM_INT8_ROW_SERIAL
            self._w_pos0, self._w_pos1, self._w_neg0, self._w_neg1 = self._decompose_signed_int8(wm)
            self._effective_signed = self._reconstruct_signed_int8()
            self._ecc_program_latency_cycles = self._ecc_latency_cycles_for_bits(
                self._pwm_stored_weight_bits(),
                self.reliability_config.ecc_encode_latency_cycles,
            )
            self._weights_internal.fill(0.0)
            return

        wm_float = wm.astype(float)
        if self.safety_checks:
            self._validate_finite(wm_float, "weight_matrix")
        if wm_float.shape == (self.input_size, logical_outputs):
            self.mode = self.MODE_SIGNED
            self._effective_signed = wm_float.copy()
            self._weights_internal = self._signed_to_differential(wm_float)
            return

        if wm_float.shape == (self.input_size, self.output_size):
            if np.any(wm_float < 0):
                raise ValueError(
                    f"Signed differential mode expects ({self.input_size}, {logical_outputs}) weights, "
                    f"not ({self.input_size}, {self.output_size})."
                )
            self.mode = self.MODE_UNSIGNED
            self._weights_internal = wm_float.copy()
            self._effective_signed = None
            return

        raise ValueError(
            f"Invalid weight shape {wm.shape}. Expected ({self.input_size}, {self.output_size}) unsigned "
            f"or ({self.input_size}, {logical_outputs}) signed/PWM INT8."
        )

    def program_int5_weights(self, weight_matrix: np.ndarray) -> None:
        """Program one differential pair per weight, using existing cell models.

        Explicit entry point leaves the legacy INT8 programming behavior intact.
        This mode senses each column bank, digitizes it, then subtracts the banks.
        TDC and slice gains are not part of this ADC-only path.
        """
        w = np.asarray(weight_matrix)
        if self.cell_bits != 4:
            raise ValueError("INT5 differential mode requires 4-bit cells")
        if self.tdc_config.enabled or self.slice_gain0 != 1 or self.slice_gain1 != 1:
            raise ValueError("INT5 mode supports ADC readout with unit slice gains only")
        if (w.shape != (self.input_size, self.output_size // 2)
                or not np.issubdtype(w.dtype, np.integer)
                or np.any(w < -15) or np.any(w > 15)):
            raise ValueError("INT5 weights must have shape [rows, logical outputs] and integer values in [-15,15]")
        self._reliability_stats = self._new_reliability_stats()
        self._iv_saturation_stats = self._new_iv_saturation_stats(self.output_size // 2)
        self._int5_adc_clips = 0
        self._int5_reads = 0
        self._ecc_compute_latency_cycles = 0
        self._last_tdc_timing = {"enabled": False}
        # Cast before negation so unsigned input arrays are handled correctly.
        signed = w.astype(np.int16)
        self._w_pos0 = self._apply_reliability_to_cell_levels(np.maximum(signed, 0))
        self._w_neg0 = self._apply_reliability_to_cell_levels(np.maximum(-signed, 0))
        # No second slice is programmed, read, or charged in storage accounting.
        self._w_pos1 = np.empty((0, 0), dtype=np.uint8)
        self._w_neg1 = np.empty((0, 0), dtype=np.uint8)
        self.mode = self.MODE_PWM_INT5_DIFFERENTIAL
        self._weights_internal = self._signed_to_differential(signed)
        self._weights_internal[:, 0::2] = self._w_pos0
        self._weights_internal[:, 1::2] = self._w_neg0
        self._ecc_program_latency_cycles = self._ecc_latency_cycles_for_bits(
            self._pwm_stored_weight_bits(), self.reliability_config.ecc_encode_latency_cycles)

    def compute_int5_pulses(self, input_vector: np.ndarray, *, dac_bits: int = 2) -> np.ndarray:
        """Return logical-column raw products for one unsigned DAC read.

        No input bit slicing, sign correction, scaling, or prior-read accumulation
        occurs here. ADC code units are integer cell-level times pulse-count units.
        Both column banks are rounded/clipped to the configured unsigned ADC range.
        """
        if self.mode != self.MODE_PWM_INT5_DIFFERENTIAL:
            raise RuntimeError("Program INT5 weights before computing INT5 pulses")
        if not isinstance(dac_bits, (int, np.integer)) or not 1 <= dac_bits <= 6:
            raise ValueError("dac_bits must be an integer from 1 to 6")
        x = np.asarray(input_vector)
        if (x.shape != (self.input_size,) or not np.issubdtype(x.dtype, np.integer)
                or np.any(x < 0) or np.any(x > (1 << dac_bits) - 1)):
            raise ValueError("Input pulses must be unsigned integer DAC codes, one per row")
        self._int5_reads += 1
        self._ecc_compute_latency_cycles += self._ecc_latency_cycles_for_bits(
            int(np.count_nonzero(x)) * self.output_size * self.cell_bits,
            self.reliability_config.ecc_decode_latency_cycles)
        result = np.zeros(self.output_size // 2, dtype=np.int64)
        for row_slice in self._row_parallel_ranges():
            banks = []
            for levels in (self._w_pos0, self._w_neg0):
                raw = self._column_product_sum(x, levels, row_slice=row_slice)
                sensed = self._apply_iv_transfer(raw)
                rounded = np.rint(sensed)
                self._int5_adc_clips += int(np.count_nonzero(
                    (rounded < 0) | (rounded > (1 << self.adc_bits) - 1)))
                banks.append(self._adc_u(sensed).astype(np.int64))
            result += banks[0] - banks[1]
        return result

    def compute(self, input_vector: np.ndarray) -> np.ndarray:
        if self.mode is None:
            raise RuntimeError("Weights are not programmed.")
        if self.mode == self.MODE_PWM_INT5_DIFFERENTIAL:
            raise ValueError("Use compute_int5_pulses with an explicit DAC width for INT5 mode")
        if self.mode == self.MODE_PWM_INT8_ROW_SERIAL:
            return self._compute_pwm_row_serial_int8(input_vector)

        if self.safety_checks:
            x = self._validate_legacy_input_vector(input_vector)
        else:
            x = np.asarray(input_vector, dtype=float)
            if x.shape[0] != self.input_size:
                raise ValueError(f"Input size mismatch: expected {self.input_size}, got {x.shape[0]}")
        raw = x @ self._weights_internal
        if self.mode == self.MODE_UNSIGNED:
            return raw

        diff = raw[0::2] - raw[1::2]
        out = np.zeros((self.output_size,), dtype=float)
        out[: diff.shape[0]] = diff
        return out

    def get_mode(self) -> str:
        return self.mode or "NOT_PROGRAMMED"

    def get_output_size(self) -> int:
        return self.output_size

    def get_effective_weights(self) -> np.ndarray:
        if self.mode is None:
            raise RuntimeError("Weights are not programmed.")
        if self.mode == self.MODE_PWM_INT5_DIFFERENTIAL:
            return self._w_pos0.astype(np.int16) - self._w_neg0.astype(np.int16)
        if self.mode == self.MODE_UNSIGNED:
            return self._weights_internal.copy()
        if self.mode == self.MODE_PWM_INT8_ROW_SERIAL:
            return self._reconstruct_signed_int8()
        if self._effective_signed is None:
            raise RuntimeError("Signed weights are not available.")
        return self._effective_signed.copy()

    def get_info(self) -> dict[str, object]:
        return {
            "mode": self.get_mode(),
            "int5_read_calls": getattr(self, "_int5_reads", 0) if self.mode == self.MODE_PWM_INT5_DIFFERENTIAL else 0,
            "int5_adc_clip_events": getattr(self, "_int5_adc_clips", 0) if self.mode == self.MODE_PWM_INT5_DIFFERENTIAL else 0,
            "input_size": self.input_size,
            "physical_output_size": self.output_size,
            "logical_outputs": self.output_size // 2,
            "weight_shape": getattr(self._weights_internal, "shape", None),
            "adc_bits": self.adc_bits,
            "cell_bits": self.cell_bits,
            "slice_gain0": self.slice_gain0,
            "slice_gain1": self.slice_gain1,
            "safety_checks": self.safety_checks,
            "noise_enabled": self.noise_config.enabled,
            "noise_seed": self.noise_seed,
            "noise_config": self.noise_config.as_dict(),
            "reliability_enabled": self.reliability_config.enabled,
            "reliability_seed": self.reliability_seed,
            "reliability_config": self.reliability_config.as_dict(),
            "reliability_stats": dict(self._reliability_stats),
            "latency_model": {
                "ecc_program_latency_cycles": int(self._ecc_program_latency_cycles),
                "ecc_compute_latency_cycles": int(self._ecc_compute_latency_cycles),
                "ecc_total_latency_cycles": int(self._ecc_program_latency_cycles + self._ecc_compute_latency_cycles),
                "tdc_timing": dict(self._last_tdc_timing),
            },
            "iv_nonlinearity_enabled": self.iv_config.enabled,
            "iv_nonlinearity_mode": self.iv_config.mode,
            "iv_adc_full_scale_levels": self._iv_adc_full_scale_levels(),
            "iv_linear_rows_before_clamp": self.iv_config.linear_rows_before_clamp,
            "iv_saturation_stats": self._iv_saturation_stats_dict(),
            "tdc_enabled": self.tdc_config.enabled,
            "tdc_config": self.tdc_config.as_dict(),
            "tdc_gain_scale": self._tdc_gain_scale(),
            "tdc_effective_coarse_bits": self._tdc_effective_coarse_bits(),
            "tdc_effective_fine_bits": self._tdc_effective_fine_bits(),
            "pe_row_parallel_config": self.row_parallel_config.as_dict(),
            "pe_row_parallel_groups": len(self._row_parallel_ranges()),
            "pe_column_parallel_config": self.column_parallel_config.as_dict(),
            "pe_output_bandwidth": self.estimate_output_bandwidth(),
            "pwm_stored_weight_bits": self._pwm_stored_weight_bits(),
        }

    def _row_parallel_ranges(self) -> list[slice]:
        if not self.row_parallel_config.enabled:
            return [slice(0, self.input_size)]

        rows_per_group = int(self.row_parallel_config.rows_per_group)
        if rows_per_group > 0:
            return [
                slice(start, min(start + rows_per_group, self.input_size))
                for start in range(0, self.input_size, rows_per_group)
            ]

        groups = min(max(int(self.row_parallel_config.groups), 1), self.input_size)
        ranges: list[slice] = []
        for group_idx in range(groups):
            start = (group_idx * self.input_size) // groups
            end = ((group_idx + 1) * self.input_size) // groups
            if end > start:
                ranges.append(slice(start, end))
        return ranges or [slice(0, self.input_size)]

    def _tdc_payload_bits_per_value(self) -> int:
        configured = int(self.column_parallel_config.payload_bits_per_value)
        if configured > 0:
            return configured
        if self.tdc_config.enabled:
            return max(int(self.tdc_config.signed_payload_bits), 1)
        return max(int(self.adc_bits), 1)

    def _active_columns_per_step(self, logical_outputs: int | None = None) -> int:
        logical = int(logical_outputs if logical_outputs is not None else self.output_size // 2)
        if logical <= 0:
            return 0
        configured = int(self.column_parallel_config.active_columns)
        if not self.column_parallel_config.enabled or configured <= 0:
            return logical
        return min(max(configured, 1), logical)

    def estimate_output_bandwidth(self, logical_outputs: int | None = None) -> dict[str, int]:
        logical = int(logical_outputs if logical_outputs is not None else self.output_size // 2)
        logical = max(logical, 0)
        row_groups = len(self._row_parallel_ranges())
        columns_per_step = self._active_columns_per_step(logical)
        column_steps = (logical + columns_per_step - 1) // columns_per_step if columns_per_step else 0
        payload_bits_per_value = self._tdc_payload_bits_per_value()
        payload_values = logical * row_groups
        payload_bits = payload_values * payload_bits_per_value
        header_bits = int(self.column_parallel_config.packet_header_bits) * row_groups * column_steps
        return {
            "logical_outputs": logical,
            "row_groups": row_groups,
            "active_columns_per_step": columns_per_step,
            "column_steps": column_steps,
            "payload_bits_per_value": payload_bits_per_value,
            "payload_values": payload_values,
            "payload_bits": payload_bits,
            "header_bits": header_bits,
            "wire_bits": payload_bits + header_bits,
        }

    def _signed_to_differential(self, signed_weights: np.ndarray) -> np.ndarray:
        differential = np.zeros((self.input_size, self.output_size), dtype=float)
        differential[:, 0::2] = np.maximum(signed_weights, 0)
        differential[:, 1::2] = np.maximum(-signed_weights, 0)
        return differential

    def _decompose_signed_int8(self, weights: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        w = np.asarray(weights, dtype=np.int16)
        w_pos = np.maximum(w, 0).astype(np.int16)
        w_neg = np.maximum(-w, 0).astype(np.int16)
        pos0 = (w_pos & 0xF).astype(np.uint8)
        pos1 = ((w_pos >> 4) & 0xF).astype(np.uint8)
        neg0 = (w_neg & 0xF).astype(np.uint8)
        neg1 = ((w_neg >> 4) & 0xF).astype(np.uint8)
        return tuple(self._apply_reliability_to_cell_levels(x) for x in (pos0, pos1, neg0, neg1))

    def _reconstruct_signed_int8(self) -> np.ndarray:
        pos = self._w_pos0.astype(np.int16) + (self._w_pos1.astype(np.int16) << 4)
        neg = self._w_neg0.astype(np.int16) + (self._w_neg1.astype(np.int16) << 4)
        return pos - neg

    def _adc_u(self, value: np.ndarray) -> np.ndarray:
        if self.safety_checks:
            self._validate_finite(value, "ADC input")
        max_value = (1 << self.adc_bits) - 1
        return np.clip(np.rint(np.asarray(value, dtype=float)), 0, max_value).astype(np.int32)

    def _iv_adc_full_scale_levels(self) -> float:
        configured = float(self.iv_config.adc_full_scale_levels)
        if configured > 0.0:
            return configured
        return float(self.iv_config.linear_rows_before_clamp) * self._max_cell_level() * self._max_cell_level()

    def _apply_iv_transfer(self, product: np.ndarray) -> np.ndarray:
        values = np.asarray(product, dtype=float)
        if not self.iv_config.enabled:
            return values

        full_scale = max(self._iv_adc_full_scale_levels(), 1.0)
        self._record_iv_saturation(values, full_scale)
        mode = self.iv_config.mode.lower()
        if mode == "hard_clamp":
            sensed = np.clip(values, 0.0, full_scale)
        elif mode == "tanh":
            sensed = full_scale * np.tanh(np.maximum(values, 0.0) / full_scale)
        else:
            raise ValueError(f"Unsupported I-V nonlinearity mode: {self.iv_config.mode}")
        return sensed

    def _record_iv_saturation(self, values: np.ndarray, full_scale: float) -> None:
        stats = self._iv_saturation_stats
        stats["transfer_calls"] = int(stats["transfer_calls"]) + 1
        arr = np.asarray(values, dtype=float)
        upper = arr > float(full_scale)
        lower = arr < 0.0
        upper_count = int(np.count_nonzero(upper))
        lower_count = int(np.count_nonzero(lower))
        stats["upper_saturation_events"] = int(stats["upper_saturation_events"]) + upper_count
        stats["lower_clip_events"] = int(stats["lower_clip_events"]) + lower_count
        if upper_count <= 0:
            return
        columns = stats.get("saturated_columns")
        if not isinstance(columns, np.ndarray) or columns.size == 0:
            return
        logical_outputs = columns.shape[0]
        clipped = np.flatnonzero(np.ravel(upper))
        if clipped.size:
            columns[np.mod(clipped, logical_outputs)] = True

    def _iv_saturation_stats_dict(self) -> dict[str, int | float | str | bool]:
        columns = self._iv_saturation_stats.get("saturated_columns")
        unique_columns = int(np.count_nonzero(columns)) if isinstance(columns, np.ndarray) else 0
        return {
            "enabled": bool(self.iv_config.enabled),
            "mode": str(self.iv_config.mode),
            "full_scale_levels": float(self._iv_adc_full_scale_levels()),
            "transfer_calls": int(self._iv_saturation_stats.get("transfer_calls", 0)),
            "upper_saturation_events": int(self._iv_saturation_stats.get("upper_saturation_events", 0)),
            "lower_clip_events": int(self._iv_saturation_stats.get("lower_clip_events", 0)),
            "unique_saturated_columns": unique_columns,
        }

    def _apply_iv_nonlinearity(self, product: np.ndarray) -> np.ndarray:
        if not self.iv_config.enabled:
            return np.rint(np.asarray(product, dtype=float)).astype(np.int32)
        sensed = self._apply_iv_transfer(product)
        return self._adc_u(sensed)

    def _adc_int4_product(self, input_nibble: int, cell_levels: np.ndarray) -> np.ndarray:
        if input_nibble == 0:
            return np.zeros_like(cell_levels, dtype=np.int32)

        levels = np.asarray(cell_levels, dtype=np.int32)
        product = int(input_nibble) * levels
        if self.noise_config.enabled:
            sigma = self._read_noise_level_sigma(levels) * np.sqrt(max(int(input_nibble), 1))
            product = product + self._noise_rng.normal(0.0, sigma, size=levels.shape)
        return self._adc_u(product)

    def _compute_pwm_row_serial_int8(self, input_vector: np.ndarray) -> np.ndarray:
        a = np.asarray(input_vector)
        if self.safety_checks:
            a = self._validate_pwm_input_vector(a)
        elif a.shape[0] != self.input_size:
            raise ValueError(f"Input size mismatch: expected {self.input_size}, got {a.shape[0]}")

        logical_outputs = self.output_size // 2
        a_u = (a.astype(np.int32) & 0xFF).astype(np.int32)
        a0 = (a_u & 0xF).astype(np.int32)
        a1 = (a_u >> 4).astype(np.int32)
        self._ecc_compute_latency_cycles += self._ecc_latency_cycles_for_bits(
            self._pwm_ecc_read_bits_for_activation(a0, a1),
            self.reliability_config.ecc_decode_latency_cycles,
        )
        if self.tdc_config.enabled:
            return self._compute_pwm_differential_tdc_int8(a_u)
        if self.iv_config.enabled:
            return self._compute_pwm_column_nonlinear_int8(a0, a1)
        acc = np.zeros((logical_outputs,), dtype=np.int32)

        for row in range(self.input_size):
            if a0[row] == 0 and a1[row] == 0:
                continue

            p00 = self._adc_int4_product(int(a0[row]), self._w_pos0[row, :]) - self._adc_int4_product(
                int(a0[row]), self._w_neg0[row, :]
            )
            p01 = self._adc_int4_product(int(a0[row]), self._w_pos1[row, :]) - self._adc_int4_product(
                int(a0[row]), self._w_neg1[row, :]
            )
            p10 = self._adc_int4_product(int(a1[row]), self._w_pos0[row, :]) - self._adc_int4_product(
                int(a1[row]), self._w_neg0[row, :]
            )
            p11 = self._adc_int4_product(int(a1[row]), self._w_pos1[row, :]) - self._adc_int4_product(
                int(a1[row]), self._w_neg1[row, :]
            )

            if self.slice_gain0 != 1.0:
                p00 = np.rint(p00 * self.slice_gain0).astype(np.int32)
                p10 = np.rint(p10 * self.slice_gain0).astype(np.int32)
            if self.slice_gain1 != 1.0:
                p01 = np.rint(p01 * self.slice_gain1).astype(np.int32)
                p11 = np.rint(p11 * self.slice_gain1).astype(np.int32)

            acc += p00 + (p01 << 4) + (p10 << 4) + (p11 << 8)

        out = np.zeros((self.output_size,), dtype=np.int32)
        out[:logical_outputs] = acc
        return out

    def _column_product_sum(
        self,
        input_counts: np.ndarray,
        cell_levels: np.ndarray,
        *,
        include_noise: bool = True,
        row_slice: slice | None = None,
    ) -> np.ndarray:
        counts = np.asarray(input_counts, dtype=np.int32)
        levels = np.asarray(cell_levels, dtype=np.int32)
        if row_slice is not None:
            counts = counts[row_slice]
            levels = levels[row_slice, :]
        products = counts[:, None] * levels
        if include_noise and self.noise_config.enabled:
            sigma = self._read_noise_level_sigma(levels) * np.sqrt(np.maximum(counts[:, None], 0))
            products = products + self._noise_rng.normal(0.0, sigma, size=levels.shape)
        return np.sum(products, axis=0)

    def _differential_column_product(
        self,
        input_counts: np.ndarray,
        positive_levels: np.ndarray,
        negative_levels: np.ndarray,
        *,
        include_noise: bool = True,
        row_slice: slice | None = None,
    ) -> np.ndarray:
        positive = self._apply_iv_nonlinearity(
            self._column_product_sum(input_counts, positive_levels, include_noise=include_noise, row_slice=row_slice)
        )
        negative = self._apply_iv_nonlinearity(
            self._column_product_sum(input_counts, negative_levels, include_noise=include_noise, row_slice=row_slice)
        )
        return (positive - negative).astype(np.int32)

    def _tdc_max_code(self) -> int:
        return (1 << max(int(self.tdc_config.tdc_bits), 1)) - 1

    def _tdc_effective_fine_bits(self) -> int:
        return min(max(int(self.tdc_config.fine_bits), 0), max(int(self.tdc_config.tdc_bits), 1))

    def _tdc_effective_coarse_bits(self) -> int:
        return max(int(self.tdc_config.tdc_bits), 1) - self._tdc_effective_fine_bits()

    def _tdc_gain_scale(self, exponent: int | None = None) -> int:
        shift = self.tdc_config.gain_shift if exponent is None else exponent
        return 1 << max(int(shift), 0)

    def _tdc_output_format(self) -> str:
        fmt = str(getattr(self.tdc_config, "output_format", "sar_fp")).lower()
        if fmt in {"sar", "sar_fp", "fp", "float", "float_like"}:
            return "sar_fp"
        if fmt in {"bfp", "linear", "uniform", "legacy"}:
            return "bfp"
        raise ValueError(f"Unsupported TDC output_format: {self.tdc_config.output_format}")

    def _attach_tdc_timing(self, packed: TDCPackedOutput, magnitude: np.ndarray) -> TDCPackedOutput:
        """Attach lightweight magnitude-dependent logical-column arrival times."""

        if not self.tdc_config.timing_enabled:
            return packed

        cfg = self.tdc_config
        mag = np.maximum(np.abs(np.asarray(magnitude, dtype=float)), 0.0)
        span_ps = cfg.timing_max_ps - cfg.timing_min_ps
        arrival_time_ps = cfg.timing_min_ps + span_ps / (1.0 + mag / cfg.timing_reference_levels)
        pe_completion_time_ps = float(np.max(arrival_time_ps)) if arrival_time_ps.size else 0.0
        arrival_spread_ps = (
            float(np.max(arrival_time_ps) - np.min(arrival_time_ps)) if arrival_time_ps.size else 0.0
        )
        completion_order = np.argsort(arrival_time_ps, kind="stable")

        self._last_tdc_timing = {
            "enabled": True,
            "first_ready_column": int(completion_order[0]) if completion_order.size else -1,
            "last_ready_column": int(completion_order[-1]) if completion_order.size else -1,
            "arrival_time_min_ps": float(np.min(arrival_time_ps)) if arrival_time_ps.size else 0.0,
            "arrival_time_max_ps": float(np.max(arrival_time_ps)) if arrival_time_ps.size else 0.0,
            "arrival_spread_ps": arrival_spread_ps,
            "pe_completion_time_ps": pe_completion_time_ps,
        }
        return replace(
            packed,
            arrival_time_ps=arrival_time_ps,
            pe_completion_time_ps=pe_completion_time_ps,
        )

    def _tdc_u(self, values: np.ndarray, exponent: int | None = None) -> np.ndarray:
        scaled = np.asarray(values, dtype=float) / float(self._tdc_gain_scale(exponent))
        return np.clip(np.rint(scaled), 0, self._tdc_max_code()).astype(np.int32)

    def _tdc_split_code(self, code: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        fine_bits = self._tdc_effective_fine_bits()
        codes = np.asarray(code, dtype=np.int32)
        if fine_bits == 0:
            return codes, np.zeros_like(codes, dtype=np.int32)
        fine_mask = (1 << fine_bits) - 1
        return (codes >> fine_bits).astype(np.int32), (codes & fine_mask).astype(np.int32)

    def _tdc_sar_magnitude_pack(self, values: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        SAR-like unsigned magnitude conversion.

        Coarse is the leading-one position of the gain-normalized magnitude and
        therefore determines the output exponent. Fine stores mantissa bits below
        that leading one. The returned mantissa includes the hidden leading one.
        """

        fine_bits = self._tdc_effective_fine_bits()
        coarse_budget = max(self._tdc_effective_coarse_bits(), 1)
        coarse_bits = min(max(int(self.tdc_config.coarse_bits), 1), coarse_budget)
        max_coarse = (1 << coarse_bits) - 1
        mantissa_base = 1 << fine_bits
        max_mantissa = (1 << (fine_bits + 1)) - 1
        gain_shift = max(int(self.tdc_config.gain_shift), 0)
        gain_scale = float(1 << gain_shift)

        raw = np.asarray(values, dtype=float)
        if self.safety_checks:
            self._validate_finite(raw, "TDC SAR input")
        scaled = np.maximum(raw, 0.0) / gain_scale
        active = scaled >= 0.5

        coarse = np.zeros(scaled.shape, dtype=np.int32)
        fine = np.zeros(scaled.shape, dtype=np.int32)
        mantissa = np.zeros(scaled.shape, dtype=np.int32)
        exponent = np.zeros(scaled.shape, dtype=np.int32)
        code = np.zeros(scaled.shape, dtype=np.int32)
        if not np.any(active):
            return code, coarse, fine, mantissa, exponent

        scaled_active = scaled[active]
        coarse_active = np.floor(np.log2(scaled_active)).astype(np.int32)
        coarse_active = np.maximum(coarse_active, 0)
        clipped_high = coarse_active > max_coarse
        coarse_active = np.minimum(coarse_active, max_coarse)

        step_exp = coarse_active - fine_bits
        mantissa_active = np.rint(np.ldexp(scaled_active, -step_exp)).astype(np.int32)
        mantissa_active = np.maximum(mantissa_active, mantissa_base)

        overflow = mantissa_active >= (1 << (fine_bits + 1))
        if np.any(overflow):
            can_increment = coarse_active < max_coarse
            inc = overflow & can_increment
            coarse_active[inc] += 1
            mantissa_active[inc] = mantissa_base
            clipped_high |= overflow & ~can_increment

        mantissa_active = np.minimum(mantissa_active, max_mantissa)
        mantissa_active[clipped_high] = max_mantissa
        fine_active = mantissa_active - mantissa_base
        code_active = (coarse_active << fine_bits) | fine_active

        coarse[active] = coarse_active
        fine[active] = fine_active
        mantissa[active] = mantissa_active
        exponent[active] = coarse_active + gain_shift
        code[active] = code_active
        return code, coarse, fine, mantissa, exponent

    def _tdc_pack_codes(
        self,
        positive_code: np.ndarray,
        negative_code: np.ndarray,
        exponent: int,
    ) -> TDCPackedOutput:
        logical_outputs = self.output_size // 2
        pos = np.asarray(positive_code, dtype=np.int32)
        neg = np.asarray(negative_code, dtype=np.int32)
        pos_coarse, pos_fine = self._tdc_split_code(pos)
        neg_coarse, neg_fine = self._tdc_split_code(neg)

        packed_arrays = []
        for values in (pos - neg, pos, neg, pos_coarse, pos_fine, neg_coarse, neg_fine):
            out = np.zeros((self.output_size,), dtype=np.int32)
            out[:logical_outputs] = np.asarray(values, dtype=np.int32)[:logical_outputs]
            packed_arrays.append(out)

        return TDCPackedOutput(
            mantissa=packed_arrays[0],
            exponent=int(exponent),
            positive_code=packed_arrays[1],
            negative_code=packed_arrays[2],
            positive_coarse=packed_arrays[3],
            positive_fine=packed_arrays[4],
            negative_coarse=packed_arrays[5],
            negative_fine=packed_arrays[6],
            format="tdc_bfp",
            mantissa_fraction_bits=0,
        )

    def _tdc_sar_pack_magnitudes(
        self,
        positive_magnitude: np.ndarray,
        negative_magnitude: np.ndarray,
    ) -> TDCPackedOutput:
        logical_outputs = self.output_size // 2
        pos_code, pos_coarse, pos_fine, pos_mant, pos_exp = self._tdc_sar_magnitude_pack(positive_magnitude)
        neg_code, neg_coarse, neg_fine, neg_mant, neg_exp = self._tdc_sar_magnitude_pack(negative_magnitude)

        signed_mantissa = pos_mant - neg_mant
        exponent = np.maximum(pos_exp, neg_exp).astype(np.int32)

        packed_arrays = []
        for values in (signed_mantissa, pos_code, neg_code, pos_coarse, pos_fine, neg_coarse, neg_fine):
            out = np.zeros((self.output_size,), dtype=np.int32)
            out[:logical_outputs] = np.asarray(values, dtype=np.int32)[:logical_outputs]
            packed_arrays.append(out)

        exponent_out = np.zeros((self.output_size,), dtype=np.int32)
        exponent_out[:logical_outputs] = exponent[:logical_outputs]
        return TDCPackedOutput(
            mantissa=packed_arrays[0],
            exponent=exponent_out,
            positive_code=packed_arrays[1],
            negative_code=packed_arrays[2],
            positive_coarse=packed_arrays[3],
            positive_fine=packed_arrays[4],
            negative_coarse=packed_arrays[5],
            negative_fine=packed_arrays[6],
            format="tdc_sar_fp",
            mantissa_fraction_bits=self._tdc_effective_fine_bits(),
        )

    def _tdc_integer_packed(self, values: np.ndarray, *, format_name: str = "tdc_integer") -> TDCPackedOutput:
        logical_outputs = self.output_size // 2
        mantissa = np.zeros((self.output_size,), dtype=np.int64)
        mantissa[:logical_outputs] = np.asarray(values, dtype=np.int64)[:logical_outputs]
        zeros = np.zeros((self.output_size,), dtype=np.int32)
        return TDCPackedOutput(
            mantissa=mantissa,
            exponent=0,
            positive_code=zeros,
            negative_code=zeros,
            positive_coarse=zeros,
            positive_fine=zeros,
            negative_coarse=zeros,
            negative_fine=zeros,
            format=format_name,
            mantissa_fraction_bits=0,
        )

    def _tdc_differential_pack(
        self,
        signed_diff: np.ndarray,
        *,
        exponent: int | None = None,
    ) -> TDCPackedOutput:
        if self._tdc_output_format() == "sar_fp":
            del exponent
            diff = np.asarray(signed_diff, dtype=float)
            ab_branch = self._apply_iv_transfer(np.maximum(diff, 0.0))
            ba_branch = self._apply_iv_transfer(np.maximum(-diff, 0.0))
            packed = self._tdc_sar_pack_magnitudes(ab_branch, ba_branch)
            return self._attach_tdc_timing(packed, ab_branch + ba_branch)

        shift = max(int(self.tdc_config.gain_shift if exponent is None else exponent), 0)
        diff = np.asarray(signed_diff, dtype=float)
        ab_branch = self._apply_iv_transfer(np.maximum(diff, 0.0))
        ba_branch = self._apply_iv_transfer(np.maximum(-diff, 0.0))
        ab_code = self._tdc_u(ab_branch, exponent=shift)
        ba_code = self._tdc_u(ba_branch, exponent=shift)
        packed = self._tdc_pack_codes(ab_code, ba_code, shift)
        return self._attach_tdc_timing(packed, ab_branch + ba_branch)

    def _tdc_differential_code(self, signed_diff: np.ndarray) -> np.ndarray:
        packed = self._tdc_differential_pack(signed_diff)
        if packed.format == "tdc_sar_fp":
            return packed.reconstruct_integer().astype(np.int32)
        return packed.mantissa

    def _pwm_full_column_sums(
        self,
        input_counts: np.ndarray,
        *,
        include_noise: bool = True,
        row_slice: slice | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        pos0 = self._column_product_sum(input_counts, self._w_pos0, include_noise=include_noise, row_slice=row_slice)
        pos1 = self._column_product_sum(input_counts, self._w_pos1, include_noise=include_noise, row_slice=row_slice)
        neg0 = self._column_product_sum(input_counts, self._w_neg0, include_noise=include_noise, row_slice=row_slice)
        neg1 = self._column_product_sum(input_counts, self._w_neg1, include_noise=include_noise, row_slice=row_slice)

        if self.slice_gain0 != 1.0:
            pos0 = pos0 * self.slice_gain0
            neg0 = neg0 * self.slice_gain0
        if self.slice_gain1 != 1.0:
            pos1 = pos1 * self.slice_gain1
            neg1 = neg1 * self.slice_gain1
        return pos0 + 16.0 * pos1, neg0 + 16.0 * neg1

    def _compute_pwm_differential_tdc_int8(
        self,
        activation_counts: np.ndarray,
        *,
        include_noise: bool = True,
    ) -> np.ndarray:
        packed = self._compute_pwm_differential_tdc_packed_int8(
            activation_counts,
            include_noise=include_noise,
        )
        if packed.format == "tdc_sar_fp":
            return packed.reconstruct_integer().astype(np.int32)
        return packed.mantissa

    def _compute_pwm_differential_tdc_packed_int8(
        self,
        activation_counts: np.ndarray,
        *,
        include_noise: bool = True,
        exponent: int | None = None,
    ) -> TDCPackedOutput:
        logical_outputs = self.output_size // 2
        row_ranges = self._row_parallel_ranges()
        if len(row_ranges) > 1:
            acc = np.zeros((logical_outputs,), dtype=np.int64)
            for row_slice in row_ranges:
                positive, negative = self._pwm_full_column_sums(
                    activation_counts,
                    include_noise=include_noise,
                    row_slice=row_slice,
                )
                packed_part = self._tdc_differential_pack(positive - negative, exponent=exponent)
                acc += packed_part.reconstruct_integer()[:logical_outputs]
            return self._tdc_integer_packed(acc, format_name="tdc_row_group_sum")

        positive, negative = self._pwm_full_column_sums(activation_counts, include_noise=include_noise)
        signed_diff = positive - negative
        packed = self._tdc_differential_pack(signed_diff, exponent=exponent)
        if packed.mantissa.shape[0] != self.output_size:
            out = np.zeros((self.output_size,), dtype=np.int32)
            out[:logical_outputs] = packed.mantissa[:logical_outputs]
            return TDCPackedOutput(
                mantissa=out,
                exponent=packed.exponent,
                positive_code=packed.positive_code,
                negative_code=packed.negative_code,
                positive_coarse=packed.positive_coarse,
                positive_fine=packed.positive_fine,
                negative_coarse=packed.negative_coarse,
                negative_fine=packed.negative_fine,
                format=packed.format,
                mantissa_fraction_bits=packed.mantissa_fraction_bits,
                arrival_time_ps=packed.arrival_time_ps,
                pe_completion_time_ps=packed.pe_completion_time_ps,
            )
        return packed

    def _compute_pwm_corrected_tdc_packed_int8(
        self,
        activation_counts: np.ndarray,
        *,
        zero_point: int = 128,
        include_noise: bool = True,
    ) -> TDCPackedOutput:
        z = int(np.clip(int(zero_point), 0, 255))
        z_vec = np.full((self.input_size,), z, dtype=np.int32)
        logical_outputs = self.output_size // 2
        row_ranges = self._row_parallel_ranges()
        if len(row_ranges) > 1:
            acc = np.zeros((logical_outputs,), dtype=np.int64)
            for row_slice in row_ranges:
                positive, negative = self._pwm_full_column_sums(
                    activation_counts,
                    include_noise=include_noise,
                    row_slice=row_slice,
                )
                zero_positive, zero_negative = self._pwm_full_column_sums(
                    z_vec,
                    include_noise=False,
                    row_slice=row_slice,
                )
                packed_part = self._tdc_differential_pack(
                    (positive - negative) - (zero_positive - zero_negative)
                )
                acc += packed_part.reconstruct_integer()[:logical_outputs]
            return self._tdc_integer_packed(acc, format_name="tdc_row_group_sum")

        positive, negative = self._pwm_full_column_sums(activation_counts, include_noise=include_noise)
        signed_diff = positive - negative
        zero_positive, zero_negative = self._pwm_full_column_sums(z_vec, include_noise=False)
        corrected_diff = signed_diff - (zero_positive - zero_negative)
        return self._tdc_differential_pack(corrected_diff)

    def compute_tdc_packed(self, input_vector: np.ndarray) -> TDCPackedOutput:
        """Compute PWM INT8 row-serial output as packed TDC mantissa/exponent metadata."""

        if self.mode != self.MODE_PWM_INT8_ROW_SERIAL:
            raise RuntimeError("Packed TDC output is only defined for PWM_INT8_ROW_SERIAL weights.")
        a = np.asarray(input_vector)
        if self.safety_checks:
            a = self._validate_pwm_input_vector(a)
        elif a.shape[0] != self.input_size:
            raise ValueError(f"Input size mismatch: expected {self.input_size}, got {a.shape[0]}")

        a_u = (a.astype(np.int32) & 0xFF).astype(np.int32)
        a0 = (a_u & 0xF).astype(np.int32)
        a1 = (a_u >> 4).astype(np.int32)
        self._ecc_compute_latency_cycles += self._ecc_latency_cycles_for_bits(
            self._pwm_ecc_read_bits_for_activation(a0, a1),
            self.reliability_config.ecc_decode_latency_cycles,
        )
        return self._compute_pwm_differential_tdc_packed_int8(a_u)

    def compute_tdc_packed_corrected(self, input_vector: np.ndarray, zero_point: int = 128) -> TDCPackedOutput:
        """Compute zero-point-corrected PWM output, then SAR/BFP-pack the signed result."""

        if self.mode != self.MODE_PWM_INT8_ROW_SERIAL:
            raise RuntimeError("Packed TDC output is only defined for PWM_INT8_ROW_SERIAL weights.")
        a = np.asarray(input_vector)
        if self.safety_checks:
            a = self._validate_pwm_input_vector(a)
        elif a.shape[0] != self.input_size:
            raise ValueError(f"Input size mismatch: expected {self.input_size}, got {a.shape[0]}")

        a_u = (a.astype(np.int32) & 0xFF).astype(np.int32)
        a0 = (a_u & 0xF).astype(np.int32)
        a1 = (a_u >> 4).astype(np.int32)
        self._ecc_compute_latency_cycles += self._ecc_latency_cycles_for_bits(
            self._pwm_ecc_read_bits_for_activation(a0, a1),
            self.reliability_config.ecc_decode_latency_cycles,
        )
        return self._compute_pwm_corrected_tdc_packed_int8(a_u, zero_point=zero_point)

    def pwm_zero_point_output(self, zero_point: int = 128) -> np.ndarray:
        """Return deterministic PWM output code for an unsigned activation zero-point."""

        if self.mode != self.MODE_PWM_INT8_ROW_SERIAL:
            raise RuntimeError("PWM zero-point output is only defined for PWM_INT8_ROW_SERIAL weights.")
        z = int(np.clip(int(zero_point), 0, 255))
        z_vec = np.full((self.input_size,), z, dtype=np.int32)
        if self.tdc_config.enabled:
            return self._compute_pwm_differential_tdc_int8(z_vec, include_noise=False)

        if self.iv_config.enabled:
            z0 = (z_vec & 0xF).astype(np.int32)
            z1 = (z_vec >> 4).astype(np.int32)
            return self._compute_pwm_column_nonlinear_int8(z0, z1, include_noise=False)

        out = np.zeros((self.output_size,), dtype=np.int32)
        out[: self.output_size // 2] = z * np.sum(self._reconstruct_signed_int8(), axis=0, dtype=np.int64)
        return out

    def pwm_zero_point_packed_output(self, zero_point: int = 128, exponent: int | None = None) -> TDCPackedOutput:
        """Return deterministic packed TDC output for an unsigned activation zero-point."""

        if self.mode != self.MODE_PWM_INT8_ROW_SERIAL:
            raise RuntimeError("PWM zero-point output is only defined for PWM_INT8_ROW_SERIAL weights.")
        z = int(np.clip(int(zero_point), 0, 255))
        z_vec = np.full((self.input_size,), z, dtype=np.int32)
        return self._compute_pwm_differential_tdc_packed_int8(z_vec, include_noise=False, exponent=exponent)

    def _compute_pwm_column_nonlinear_int8(
        self,
        a0: np.ndarray,
        a1: np.ndarray,
        *,
        include_noise: bool = True,
    ) -> np.ndarray:
        logical_outputs = self.output_size // 2
        acc = np.zeros((logical_outputs,), dtype=np.int64)
        for row_slice in self._row_parallel_ranges():
            p00 = self._differential_column_product(
                a0, self._w_pos0, self._w_neg0, include_noise=include_noise, row_slice=row_slice
            )
            p01 = self._differential_column_product(
                a0, self._w_pos1, self._w_neg1, include_noise=include_noise, row_slice=row_slice
            )
            p10 = self._differential_column_product(
                a1, self._w_pos0, self._w_neg0, include_noise=include_noise, row_slice=row_slice
            )
            p11 = self._differential_column_product(
                a1, self._w_pos1, self._w_neg1, include_noise=include_noise, row_slice=row_slice
            )

            if self.slice_gain0 != 1.0:
                p00 = np.rint(p00 * self.slice_gain0).astype(np.int32)
                p10 = np.rint(p10 * self.slice_gain0).astype(np.int32)
            if self.slice_gain1 != 1.0:
                p01 = np.rint(p01 * self.slice_gain1).astype(np.int32)
                p11 = np.rint(p11 * self.slice_gain1).astype(np.int32)

            acc += p00 + (p01 << 4) + (p10 << 4) + (p11 << 8)
        out = np.zeros((self.output_size,), dtype=np.int32)
        out[:logical_outputs] = acc.astype(np.int32)
        return out

    def _max_cell_level(self) -> int:
        return (1 << self.cell_bits) - 1

    def _make_state_log_resistance(self) -> np.ndarray:
        max_level = self._max_cell_level()
        levels = np.arange(max_level + 1, dtype=float)
        frac = levels / max(1, max_level)
        log_hrs = np.log(max(self.noise_config.r_hrs_ohm, 1.0e-30))
        log_lrs = np.log(max(self.noise_config.r_lrs_ohm, 1.0e-30))
        return log_hrs + frac * (log_lrs - log_hrs)

    def _levels_to_log_resistance(self, levels: np.ndarray) -> np.ndarray:
        idx = np.clip(np.rint(levels).astype(np.int16), 0, self._max_cell_level())
        return self._state_log_resistance[idx]

    def _log_resistance_to_levels(self, log_resistance: np.ndarray) -> np.ndarray:
        diff = np.abs(np.asarray(log_resistance)[..., None] - self._state_log_resistance)
        return np.argmin(diff, axis=-1).astype(np.uint8)

    def _apply_reliability_to_cell_levels(self, levels: np.ndarray) -> np.ndarray:
        cfg = self.reliability_config
        out = np.asarray(levels, dtype=np.int16).copy()
        max_level = self._max_cell_level()
        self._reliability_stats["programmed_cells"] += int(out.size)
        if not cfg.enabled:
            return np.clip(out, 0, max_level).astype(np.uint8)

        programmable = np.ones(out.shape, dtype=bool)

        p_lrs = np.clip(float(cfg.stuck_at_lrs_prob), 0.0, 1.0)
        if p_lrs > 0.0:
            lrs_mask = self._reliability_rng.random(out.shape) < p_lrs
            out[lrs_mask] = max_level
            programmable[lrs_mask] = False
            self._reliability_stats["stuck_at_lrs_cells"] += int(np.count_nonzero(lrs_mask))

        p_hrs = np.clip(float(cfg.stuck_at_hrs_prob), 0.0, 1.0)
        if p_hrs > 0.0:
            hrs_mask = (self._reliability_rng.random(out.shape) < p_hrs) & programmable
            out[hrs_mask] = 0
            programmable[hrs_mask] = False
            self._reliability_stats["stuck_at_hrs_cells"] += int(np.count_nonzero(hrs_mask))

        p_ber = self._effective_bit_error_probability()
        if p_ber > 0.0:
            for bit in range(self.cell_bits):
                flips = (self._reliability_rng.random(out.shape) < p_ber) & programmable
                out[flips] ^= 1 << bit
                self._reliability_stats["raw_ber_bit_flips"] += int(np.count_nonzero(flips))
            out = np.clip(out, 0, max_level)

        if self._has_endurance_limit_failure():
            out, programmable = self._apply_endurance_limit_failures(out, programmable)

        if self._has_programmed_cell_drift():
            log_r = self._levels_to_log_resistance(out)
            log_shift = np.zeros(out.shape, dtype=float)

            if self._has_retention_drift():
                log_shift += self._retention_log_resistance_shift(out)

            if float(cfg.relaxation_time_s) > 0.0 and float(cfg.relaxation_resistance_amplitude_ln) != 0.0:
                tau = max(float(cfg.relaxation_tau_s), 1.0e-12)
                log_shift += float(cfg.relaxation_resistance_amplitude_ln) * (
                    1.0 - np.exp(-float(cfg.relaxation_time_s) / tau)
                )

            if int(cfg.endurance_write_cycles) > 0 and float(cfg.endurance_resistance_drift_ln_per_life_fraction) != 0.0:
                life = max(float(cfg.endurance_cycles), 1.0)
                life_fraction = min(max(float(cfg.endurance_write_cycles) / life, 0.0), 1.0)
                log_shift += float(cfg.endurance_resistance_drift_ln_per_life_fraction) * life_fraction

            if float(cfg.state_sigma_ln) > 0.0:
                log_shift += self._reliability_rng.normal(0.0, float(cfg.state_sigma_ln), size=out.shape)

            log_r = np.where(programmable, log_r + log_shift, log_r)
            out = self._log_resistance_to_levels(log_r).astype(np.int16)

        return np.clip(out, 0, max_level).astype(np.uint8)

    def _has_programmed_cell_drift(self) -> bool:
        cfg = self.reliability_config
        return any(
            (
                self._has_retention_drift(),
                float(cfg.relaxation_time_s) > 0.0 and float(cfg.relaxation_resistance_amplitude_ln) != 0.0,
                int(cfg.endurance_write_cycles) > 0
                and float(cfg.endurance_resistance_drift_ln_per_life_fraction) != 0.0,
                float(cfg.state_sigma_ln) > 0.0,
            )
        )

    def _has_retention_drift(self) -> bool:
        cfg = self.reliability_config
        if float(cfg.retention_time_s) <= 0.0:
            return False
        model = str(cfg.retention_model).lower()
        if model in {"state_power_law", "reram_power_law", "power_law"}:
            return any(
                (
                    float(cfg.retention_lrs_beta_log10_per_decade) != 0.0,
                    float(cfg.retention_hrs_beta_log10_per_decade) != 0.0,
                )
            )
        return float(cfg.retention_log10_resistance_drift_per_decade) != 0.0

    def _retention_log_resistance_shift(self, levels: np.ndarray) -> np.ndarray | float:
        cfg = self.reliability_config
        ref = max(float(cfg.retention_reference_s), 1.0e-12)
        t = max(float(cfg.retention_time_s), ref)
        decades = max(np.log10(t / ref), 0.0)
        model = str(cfg.retention_model).lower()

        if model in {"state_power_law", "reram_power_law", "power_law"}:
            max_level = self._max_cell_level()
            frac_lrs = np.asarray(levels, dtype=float) / max(max_level, 1)
            beta_hrs = float(cfg.retention_hrs_beta_log10_per_decade)
            beta_lrs = float(cfg.retention_lrs_beta_log10_per_decade)
            beta = (1.0 - frac_lrs) * beta_hrs + frac_lrs * beta_lrs
            return np.log(10.0) * beta * decades

        temp_factor = max(float(cfg.retention_temp_acceleration_per_100c), 0.0) ** (
            (float(cfg.retention_temp_c) - float(cfg.retention_temp_reference_c)) / 100.0
        )
        return np.log(10.0) * float(cfg.retention_log10_resistance_drift_per_decade) * decades * temp_factor

    def _endurance_at_limit(self) -> bool:
        cfg = self.reliability_config
        if int(cfg.endurance_write_cycles) <= 0:
            return False
        return int(cfg.endurance_write_cycles) >= max(int(cfg.endurance_cycles), 1)

    def _has_endurance_limit_failure(self) -> bool:
        cfg = self.reliability_config
        if not self._endurance_at_limit():
            return False
        return any(
            (
                float(cfg.endurance_limit_stuck_at_lrs_prob) > 0.0,
                float(cfg.endurance_limit_stuck_at_hrs_prob) > 0.0,
                float(cfg.endurance_limit_multilevel_loss_prob) > 0.0
                and float(cfg.endurance_limit_multilevel_sigma_ln) > 0.0,
            )
        )

    def _apply_endurance_limit_failures(
        self,
        levels: np.ndarray,
        programmable: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        cfg = self.reliability_config
        max_level = self._max_cell_level()
        next_levels = np.asarray(levels, dtype=np.int16).copy()
        next_programmable = np.asarray(programmable, dtype=bool).copy()

        candidates = next_programmable.copy()
        p_lrs = np.clip(float(cfg.endurance_limit_stuck_at_lrs_prob), 0.0, 1.0)
        if p_lrs > 0.0:
            mask = (self._reliability_rng.random(next_levels.shape) < p_lrs) & candidates
            next_levels[mask] = max_level
            next_programmable[mask] = False
            candidates[mask] = False
            self._reliability_stats["endurance_limit_stuck_at_lrs_cells"] += int(np.count_nonzero(mask))

        p_hrs = np.clip(float(cfg.endurance_limit_stuck_at_hrs_prob), 0.0, 1.0)
        if p_hrs > 0.0:
            mask = (self._reliability_rng.random(next_levels.shape) < p_hrs) & candidates
            next_levels[mask] = 0
            next_programmable[mask] = False
            candidates[mask] = False
            self._reliability_stats["endurance_limit_stuck_at_hrs_cells"] += int(np.count_nonzero(mask))

        p_loss = np.clip(float(cfg.endurance_limit_multilevel_loss_prob), 0.0, 1.0)
        sigma = max(float(cfg.endurance_limit_multilevel_sigma_ln), 0.0)
        if p_loss > 0.0 and sigma > 0.0:
            mask = (self._reliability_rng.random(next_levels.shape) < p_loss) & candidates
            if np.any(mask):
                log_r = self._levels_to_log_resistance(next_levels[mask])
                log_r = log_r + self._reliability_rng.normal(0.0, sigma, size=log_r.shape)
                next_levels[mask] = self._log_resistance_to_levels(log_r).astype(np.int16)
                self._reliability_stats["endurance_limit_multilevel_loss_cells"] += int(np.count_nonzero(mask))

        return np.clip(next_levels, 0, max_level).astype(np.int16), next_programmable

    def _ecc_codeword_count(self, data_bit_count: int) -> int:
        cfg = self.reliability_config
        data_bits = max(int(cfg.ecc_data_bits), 1)
        return int(np.ceil(max(int(data_bit_count), 0) / data_bits))

    def _ecc_latency_cycles_for_bits(self, data_bit_count: int, cycles_per_codeword: int) -> int:
        cfg = self.reliability_config
        if not (cfg.enabled and cfg.use_ecc):
            return 0
        per_codeword = max(int(cycles_per_codeword), 0)
        return self._ecc_codeword_count(data_bit_count) * per_codeword

    def _pwm_stored_weight_bits(self) -> int:
        return int(
            (self._w_pos0.size + self._w_pos1.size + self._w_neg0.size + self._w_neg1.size) * self.cell_bits
        )

    def _pwm_ecc_read_bits_for_activation(self, a0: np.ndarray, a1: np.ndarray) -> int:
        logical_outputs = self.output_size // 2
        nibble_passes = int(np.count_nonzero(a0)) + int(np.count_nonzero(a1))
        banks_per_pass = 4
        return int(nibble_passes * logical_outputs * banks_per_pass * self.cell_bits)

    def _effective_bit_error_probability(self) -> float:
        cfg = self.reliability_config
        p_raw = max(0.0, float(cfg.raw_ber_ppm)) * 1.0e-6
        if p_raw <= 0.0:
            return 0.0
        if not cfg.use_ecc:
            return min(1.0, p_raw)
        return self._post_ecc_ber_probability(p_raw)

    def _post_ecc_ber_probability(self, p_raw: float) -> float:
        cfg = self.reliability_config
        n_bits = max(1, int(cfg.ecc_codeword_bits))
        data_bits = max(1, int(cfg.ecc_data_bits))
        p0 = (1.0 - p_raw) ** n_bits
        p1 = n_bits * p_raw * ((1.0 - p_raw) ** max(0, n_bits - 1))
        p_uncorrectable_word = min(1.0, max(0.0, 1.0 - p0 - p1))
        return min(1.0, p_uncorrectable_word / data_bits)

    def _flicker_variance(self, i0: np.ndarray, af: np.ndarray, gamma: np.ndarray) -> np.ndarray:
        cfg = self.noise_config
        fmin = max(float(cfg.flicker_fmin_hz), 1.0e-30)
        fmax = max(float(cfg.bandwidth_hz), fmin)
        gamma_arr = np.asarray(gamma, dtype=float)
        integral = np.empty_like(gamma_arr, dtype=float)
        gamma_is_one = np.abs(gamma_arr - 1.0) < 1.0e-9
        integral[gamma_is_one] = np.log(fmax / fmin)
        gamma_other = gamma_arr[~gamma_is_one]
        integral[~gamma_is_one] = (
            np.power(fmax, 1.0 - gamma_other) - np.power(fmin, 1.0 - gamma_other)
        ) / (1.0 - gamma_other)
        return np.maximum(0.0, af * np.square(i0) * integral)

    def _read_noise_level_sigma(self, levels: np.ndarray) -> np.ndarray:
        cfg = self.noise_config
        max_level = self._max_cell_level()
        level_f = np.asarray(levels, dtype=float)
        frac = level_f / max(1, max_level)

        resistance = np.exp(self._levels_to_log_resistance(levels))
        current = abs(float(cfg.vread)) / np.maximum(resistance, 1.0e-30)
        thermal_psd = 4.0 * K_BOLTZMANN * max(float(cfg.temp_k), 0.0) / np.maximum(resistance, 1.0e-30)
        shot_psd = 2.0 * Q_ELECTRON * current * max(float(cfg.fano), 0.0)
        white_var = (thermal_psd + shot_psd) * max(float(cfg.bandwidth_hz), 0.0)

        af = cfg.flicker_af_hrs + frac * (cfg.flicker_af_lrs - cfg.flicker_af_hrs)
        gamma = cfg.flicker_gamma_hrs + frac * (cfg.flicker_gamma_lrs - cfg.flicker_gamma_hrs)
        flicker_var = self._flicker_variance(current, af, gamma)

        rtn_frac = cfg.rtn_amp_frac_hrs + frac * (cfg.rtn_amp_frac_lrs - cfg.rtn_amp_frac_hrs)
        rtn_var = np.square(0.5 * current * rtn_frac)
        readout_var = max(float(cfg.readout_noise_a_rms), 0.0) ** 2
        current_sigma = np.sqrt(np.maximum(0.0, white_var + flicker_var + rtn_var + readout_var))

        current_grid = abs(float(cfg.vread)) / np.exp(self._state_log_resistance)
        current_steps = np.abs(np.diff(current_grid))
        min_step = max(float(np.min(current_steps)) if current_steps.size else 1.0, 1.0e-30)
        idx = np.clip(np.asarray(levels, dtype=np.int16), 0, max_level)
        lo = np.maximum(idx - 1, 0)
        hi = np.minimum(idx + 1, max_level)
        span = np.abs(current_grid[hi] - current_grid[lo]) / np.maximum(1, hi - lo)
        return current_sigma / np.maximum(span, min_step)
