import numpy as np
import sys
import io
from dataclasses import asdict, dataclass, field, fields, replace
from typing import Any, Mapping
try:
    sys.stdout.reconfigure(encoding='utf-8')
except AttributeError:
    # Some environments (e.g., Jupyter) don't support reconfigure
    pass


K_BOLTZMANN = 1.380649e-23
Q_ELECTRON = 1.602176634e-19


SAR_TDC_SIGNED_PAYLOAD_MODES = {
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


def normalize_sar_tdc_precision_mode(precision_mode):
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


def sar_tdc_precision_settings(precision_mode):
    mode = normalize_sar_tdc_precision_mode(precision_mode)
    if mode is None:
        return None
    return {"precision_mode": mode, **SAR_TDC_SIGNED_PAYLOAD_MODES[mode]}


def sar_tdc_precision_choices():
    return tuple(SAR_TDC_SIGNED_PAYLOAD_MODES.keys())


def _coerce_dataclass(cls, value):
    if value is None:
        return cls()
    if isinstance(value, cls):
        return value
    if isinstance(value, Mapping):
        valid = {item.name for item in fields(cls)}
        return cls(**{key: value[key] for key in value if key in valid})
    raise TypeError(f"Cannot build {cls.__name__} from {type(value).__name__}")


@dataclass
class CrossbarNoiseConfig:
    enabled: bool = False
    vread: float = 0.1
    temp_k: float = 300.0
    fano: float = 1.0
    r_lrs_ohm: float = 10000.0
    r_hrs_ohm: float = 1000000.0
    flicker_af_lrs: float = 1.0e-11
    flicker_af_hrs: float = 1.0e-9
    flicker_gamma_lrs: float = 1.0
    flicker_gamma_hrs: float = 1.0
    rtn_amp_frac_lrs: float = 3.0e-4
    rtn_amp_frac_hrs: float = 2.0e-2
    readout_noise_a_rms: float = 0.0
    bandwidth_hz: float = 5000.0
    flicker_fmin_hz: float = 0.5

    @classmethod
    def from_obj(cls, value):
        return _coerce_dataclass(cls, value)

    def as_dict(self):
        return asdict(self)


@dataclass
class CrossbarReliabilityConfig:
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
    relaxation_tau_s: float = 2000.0
    relaxation_resistance_amplitude_ln: float = 0.0
    endurance_write_cycles: int = 0
    endurance_cycles: int = 100000
    endurance_resistance_drift_ln_per_life_fraction: float = 0.0
    endurance_limit_stuck_at_lrs_prob: float = 0.0
    endurance_limit_stuck_at_hrs_prob: float = 0.0
    endurance_limit_multilevel_loss_prob: float = 0.0
    endurance_limit_multilevel_sigma_ln: float = 0.0

    @classmethod
    def from_obj(cls, value):
        return _coerce_dataclass(cls, value)

    def as_dict(self):
        return asdict(self)


@dataclass
class CrossbarIVNonlinearityConfig:
    enabled: bool = False
    mode: str = "hard_clamp"
    adc_full_scale_levels: float = 0.0
    linear_rows_before_clamp: float = 10.0

    @classmethod
    def from_obj(cls, value):
        return _coerce_dataclass(cls, value)

    def as_dict(self):
        return asdict(self)


@dataclass
class CrossbarTDCConfig:
    enabled: bool = False
    tdc_bits: int = 8
    coarse_bits: int = 5
    fine_bits: int = 3
    gain_shift: int = 0
    output_format: str = "sar_fp"
    precision_mode: Any = None
    signed_payload_bits: int = 0
    timing_enabled: bool = True
    timing_min_ps: float = 0.0
    timing_max_ps: float = 576.0
    timing_stage_ps: float = 9.0
    timing_reference_levels: float = 256.0
    timing_encoder_ps: float = 100.0
    timing_reset_ps: float = 150.0

    def __post_init__(self):
        mode = normalize_sar_tdc_precision_mode(self.precision_mode)
        if mode is not None:
            settings = SAR_TDC_SIGNED_PAYLOAD_MODES[mode]
            self.precision_mode = mode
            self.coarse_bits = int(settings["coarse_bits"])
            self.fine_bits = int(settings["fine_bits"])
            self.tdc_bits = int(settings["tdc_bits"])
            self.signed_payload_bits = int(settings["signed_payload_bits"])
        else:
            self.precision_mode = None
            if int(self.signed_payload_bits) <= 0:
                self.signed_payload_bits = int(self.coarse_bits) + int(self.fine_bits) + 1

        self.timing_min_ps = float(self.timing_min_ps)
        self.timing_max_ps = float(self.timing_max_ps)
        self.timing_stage_ps = float(self.timing_stage_ps)
        self.timing_reference_levels = float(self.timing_reference_levels)
        self.timing_encoder_ps = float(self.timing_encoder_ps)
        self.timing_reset_ps = float(self.timing_reset_ps)
        if self.timing_max_ps < self.timing_min_ps:
            raise ValueError("TDC timing_max_ps must be greater than or equal to timing_min_ps.")
        if self.timing_stage_ps <= 0.0:
            raise ValueError("TDC timing_stage_ps must be positive.")
        if self.timing_reference_levels <= 0.0:
            raise ValueError("TDC timing_reference_levels must be positive.")
        if self.timing_encoder_ps < 0.0 or self.timing_reset_ps < 0.0:
            raise ValueError("TDC encoder and reset timing must be non-negative.")
        self.timing_enabled = bool(self.timing_enabled)

    @classmethod
    def from_obj(cls, value):
        return _coerce_dataclass(cls, value)

    def as_dict(self):
        return asdict(self)


@dataclass
class CrossbarPERowParallelConfig:
    """PE row-sensing grouping for added-current clipping studies."""

    enabled: bool = False
    groups: int = 1
    rows_per_group: int = 0
    combine: str = "digital_sum"

    def __post_init__(self):
        self.groups = max(int(self.groups), 1)
        self.rows_per_group = max(int(self.rows_per_group), 0)
        self.enabled = bool(self.enabled or self.groups > 1 or self.rows_per_group > 0)
        self.combine = str(self.combine).lower()
        if self.combine != "digital_sum":
            raise ValueError("CrossbarPERowParallelConfig.combine currently supports only 'digital_sum'.")

    @classmethod
    def from_obj(cls, value):
        return _coerce_dataclass(cls, value)

    def as_dict(self):
        return asdict(self)


@dataclass
class CrossbarPEColumnParallelConfig:
    """PE column-sensing parallelism and output packetization metadata."""

    enabled: bool = False
    active_columns: int = 0
    packet_header_bits: int = 0
    payload_bits_per_value: int = 0

    def __post_init__(self):
        self.active_columns = max(int(self.active_columns), 0)
        self.packet_header_bits = max(int(self.packet_header_bits), 0)
        self.payload_bits_per_value = max(int(self.payload_bits_per_value), 0)
        self.enabled = bool(
            self.enabled
            or self.active_columns > 0
            or self.packet_header_bits > 0
            or self.payload_bits_per_value > 0
        )

    @classmethod
    def from_obj(cls, value):
        return _coerce_dataclass(cls, value)

    def as_dict(self):
        return asdict(self)


@dataclass
class TDCPackedOutput:
    mantissa: np.ndarray
    exponent: Any
    positive_code: np.ndarray
    negative_code: np.ndarray
    positive_coarse: np.ndarray
    positive_fine: np.ndarray
    negative_coarse: np.ndarray
    negative_fine: np.ndarray
    format: str = "tdc_sar_fp"
    mantissa_fraction_bits: int = 0
    tap_index: Any = None
    arrival_time_ps: Any = None
    ready_time_ps: Any = None
    pe_ready_time_ps: float = 0.0
    pe_cycle_time_ps: float = 0.0

    @property
    def completion_order(self):
        """Logical column indices ordered from earliest to latest completion."""

        if self.ready_time_ps is None:
            return np.zeros((0,), dtype=np.int32)
        ready = np.asarray(self.ready_time_ps, dtype=float)
        return np.argsort(ready, kind="stable").astype(np.int32)

    @property
    def decode_scale(self):
        exp = np.asarray(self.exponent)
        if self.format == "tdc_sar_fp":
            scale_exp = exp - int(self.mantissa_fraction_bits)
        else:
            scale_exp = exp
        if np.ndim(scale_exp) == 0:
            shift = int(scale_exp)
            if shift >= 0:
                return 1 << shift
            return 2.0 ** shift
        return np.exp2(scale_exp.astype(np.float32))

    def reconstruct_integer(self):
        if self.format in {"tdc_integer", "tdc_row_group_sum"}:
            return self.mantissa.astype(np.int64)
        if self.format == "tdc_sar_fp":
            exp = np.asarray(self.exponent, dtype=np.int32)
            shift = exp - int(self.mantissa_fraction_bits)
            values = np.ldexp(self.mantissa.astype(np.float64), shift)
            return np.rint(values).astype(np.int64)
        return self.mantissa.astype(np.int64) * int(self.decode_scale)

    def reconstruct_float(self, extra_scale=1.0):
        if self.format == "tdc_sar_fp":
            return self.reconstruct_integer().astype(np.float32) * float(extra_scale)
        return self.mantissa.astype(np.float32) * float(self.decode_scale) * float(extra_scale)

    def as_dict(self):
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
            scale_value = float(scale)
            out["decode_scale"] = int(scale_value) if scale_value.is_integer() else scale_value
        else:
            out["decode_scale_min"] = float(np.min(scale)) if scale.size else 0.0
            out["decode_scale_max"] = float(np.max(scale)) if scale.size else 0.0
        arrival = None if self.arrival_time_ps is None else np.asarray(self.arrival_time_ps, dtype=float)
        ready = None if self.ready_time_ps is None else np.asarray(self.ready_time_ps, dtype=float)
        taps = None if self.tap_index is None else np.asarray(self.tap_index, dtype=np.int32)
        out["timing_enabled"] = bool(arrival is not None)
        if arrival is not None:
            out["arrival_time_min_ps"] = float(np.min(arrival)) if arrival.size else 0.0
            out["arrival_time_max_ps"] = float(np.max(arrival)) if arrival.size else 0.0
            out["pe_ready_time_ps"] = float(self.pe_ready_time_ps)
            out["pe_cycle_time_ps"] = float(self.pe_cycle_time_ps)
            order = self.completion_order
            out["first_ready_column"] = int(order[0]) if order.size else -1
            out["last_ready_column"] = int(order[-1]) if order.size else -1
        if ready is not None:
            out["ready_time_min_ps"] = float(np.min(ready)) if ready.size else 0.0
            out["ready_time_max_ps"] = float(np.max(ready)) if ready.size else 0.0
        if taps is not None:
            out["tap_index_min"] = int(np.min(taps)) if taps.size else 0
            out["tap_index_max"] = int(np.max(taps)) if taps.size else 0
        return out


@dataclass
class CrossbarFeatures:
    noise: CrossbarNoiseConfig = field(default_factory=CrossbarNoiseConfig)
    reliability: CrossbarReliabilityConfig = field(default_factory=CrossbarReliabilityConfig)
    iv: CrossbarIVNonlinearityConfig = field(default_factory=CrossbarIVNonlinearityConfig)
    tdc: CrossbarTDCConfig = field(default_factory=CrossbarTDCConfig)
    row_parallel: CrossbarPERowParallelConfig = field(default_factory=CrossbarPERowParallelConfig)
    column_parallel: CrossbarPEColumnParallelConfig = field(default_factory=CrossbarPEColumnParallelConfig)
    noise_seed: Any = None
    reliability_seed: Any = None

    @classmethod
    def from_obj(cls, value):
        if value is None:
            return cls()
        if isinstance(value, cls):
            return value
        if not isinstance(value, Mapping):
            raise TypeError(f"Cannot build {cls.__name__} from {type(value).__name__}")
        return cls(
            noise=CrossbarNoiseConfig.from_obj(value.get("noise_config", value.get("noise"))),
            reliability=CrossbarReliabilityConfig.from_obj(value.get("reliability_config", value.get("reliability"))),
            iv=CrossbarIVNonlinearityConfig.from_obj(value.get("iv_config", value.get("iv"))),
            tdc=CrossbarTDCConfig.from_obj(value.get("tdc_config", value.get("tdc"))),
            row_parallel=CrossbarPERowParallelConfig.from_obj(
                value.get("row_parallel_config", value.get("pe_row_parallel_config", value.get("row_parallel")))
            ),
            column_parallel=CrossbarPEColumnParallelConfig.from_obj(
                value.get(
                    "column_parallel_config",
                    value.get("pe_column_parallel_config", value.get("column_parallel")),
                )
            ),
            noise_seed=value.get("noise_seed"),
            reliability_seed=value.get("reliability_seed"),
        )

    def as_dict(self):
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
    """
    64×64 crossbar performing analog-style vector-matrix multiplication.

    Supported modes
    ---------------
    1) UNSIGNED (legacy)
        - Weight storage: (64,64) non-negative floats
        - Compute: raw = x @ W  -> 64 outputs

    2) SIGNED (legacy differential, float)
        - Input weights: (64,32) signed floats
        - Internal storage: (64,64) non-negative floats (even=pos, odd=neg)
        - Compute: diff = raw[0::2] - raw[1::2] -> 32 outputs, padded to 64

    3) PWM_INT8_ROW_SERIAL (NEW; per INT8xINT8 PWM row-serial PE doc)
        - Assumptions:
            * Activations a are UNSIGNED INT8 in [0,255] (e.g., post-ReLU)
            * Weights w are SIGNED INT8 in [-128,127]
            * Signed weights stored with column differential: w = w_pos - w_neg
            * 4-bit cells, 8-bit ADC, radix-16 (nibble) slicing + multi-read recombine
        - Storage per logical weight: 4 x 4-bit values (w_pos0, w_pos1, w_neg0, w_neg1)
        - Compute: row-serial, two PWM passes per row (a0 then a1),
          ADC digitizes per-pass partial products, digital shift-add recombines, INT32 accumulates.
        - Output contract: returns 64 elements:
            * [0:32] = computed signed outputs (INT32)
            * [32:64] = 0 padding (keeps compatibility with your pipeline)
    """

    MODE_UNSIGNED = 'UNSIGNED'
    MODE_SIGNED = 'SIGNED'
    MODE_PWM_INT8_ROW_SERIAL = 'PWM_INT8_ROW_SERIAL'

    def __init__(
        self,
        input_size=64,
        output_size=64,
        *,
        adc_bits=8,
        cell_bits=4,
        slice_gain0=1.0,
        slice_gain1=1.0,
        safety_checks=True,
        crossbar_features=None,
        noise_config=None,
        reliability_config=None,
        iv_config=None,
        tdc_config=None,
        row_parallel_config=None,
        column_parallel_config=None,
        noise_seed=None,
        reliability_seed=None,
    ):
        self.safety_checks = bool(safety_checks)
        if self.safety_checks:
            self._validate_constructor_args(input_size, output_size, adc_bits, cell_bits)

        self.input_size = int(input_size)
        self.output_size = int(output_size)

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
        self._last_tdc_timing = {
            "enabled": False,
            "first_ready_column": -1,
            "last_ready_column": -1,
            "arrival_time_min_ps": 0.0,
            "arrival_time_max_ps": 0.0,
            "pe_ready_time_ps": 0.0,
            "pe_cycle_time_ps": 0.0,
        }
        self._reliability_stats = self._new_reliability_stats()

        # legacy storage (float)
        self._weights_internal = np.zeros((self.input_size, self.output_size), dtype=float)
        self._effective_signed = None
        self.mode = None

        # PWM_INT8_ROW_SERIAL storage (4 banks of 4-bit cells, column-differential)
        self.adc_bits = int(adc_bits)
        self.cell_bits = int(cell_bits)
        self.slice_gain0 = float(slice_gain0)  # optional calibration gain for slice 0
        self.slice_gain1 = float(slice_gain1)  # optional calibration gain for slice 1

        logical_out = self.output_size // 2  # because column differential => 2 physical cols per logical output
        self._w_pos0 = np.zeros((self.input_size, logical_out), dtype=np.uint8)
        self._w_pos1 = np.zeros((self.input_size, logical_out), dtype=np.uint8)
        self._w_neg0 = np.zeros((self.input_size, logical_out), dtype=np.uint8)
        self._w_neg1 = np.zeros((self.input_size, logical_out), dtype=np.uint8)
        self._state_log_resistance = self._make_state_log_resistance()
        self._iv_saturation_stats = self._new_iv_saturation_stats(logical_out)

        print(f"[INIT] CrossbarArray created: {self.input_size}×{self.output_size}")
        print(f"       Mode will be auto-detected from weights\n")

    def _new_reliability_stats(self):
        return {
            "programmed_cells": 0,
            "stuck_at_lrs_cells": 0,
            "stuck_at_hrs_cells": 0,
            "raw_ber_bit_flips": 0,
            "endurance_limit_stuck_at_lrs_cells": 0,
            "endurance_limit_stuck_at_hrs_cells": 0,
            "endurance_limit_multilevel_loss_cells": 0,
        }

    def _new_iv_saturation_stats(self, logical_outputs):
        return {
            "transfer_calls": 0,
            "upper_saturation_events": 0,
            "lower_clip_events": 0,
            "saturated_columns": np.zeros((max(int(logical_outputs), 0),), dtype=bool),
        }

    # -----------------------
    # Legacy properties
    # -----------------------
    @property
    def weights(self):
        return self._weights_internal

    @weights.setter
    def weights(self, val):
        arr = np.asarray(val, dtype=float)
        if self.safety_checks:
            self._validate_finite(arr, "weights")
        self._weights_internal = arr

    # -----------------------
    # Safety checks
    # -----------------------
    def _validate_constructor_args(self, input_size, output_size, adc_bits, cell_bits):
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

    def _validate_finite(self, arr, name):
        values = np.asarray(arr, dtype=float)
        if not np.all(np.isfinite(values)):
            raise ValueError(f"{name} contains NaN or infinite values")

    def _validate_2d_weights(self, weight_matrix):
        if np.asarray(weight_matrix).ndim != 2:
            raise ValueError(f"weight_matrix must be 2D, got shape {np.asarray(weight_matrix).shape}")

    def _validate_pwm_weight_range(self, weight_matrix):
        values = np.asarray(weight_matrix, dtype=np.int64)
        if values.size and (np.min(values) < -128 or np.max(values) > 127):
            raise ValueError("PWM_INT8_ROW_SERIAL weights must be signed INT8 values in [-128, 127]")

    def _validate_legacy_input_vector(self, input_vector):
        x = np.asarray(input_vector, dtype=float)
        if x.ndim != 1:
            raise ValueError(f"Input vector must be 1D, got shape {x.shape}")
        if x.shape[0] != self.input_size:
            raise ValueError(f"Input size mismatch: expected {self.input_size}, got {x.shape[0]}")
        self._validate_finite(x, "input_vector")
        return x

    def _validate_pwm_input_vector(self, input_vector):
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

    # -----------------------
    # Legacy helpers
    # -----------------------
    def _detect_mode(self, weight_matrix_float):
        has_negative = np.any(weight_matrix_float < 0)
        if has_negative:
            print(f"[AUTO-DETECT] Negative weights found → SIGNED mode")
            return self.MODE_SIGNED
        print(f"[AUTO-DETECT] All weights ≥ 0 → UNSIGNED mode")
        return self.MODE_UNSIGNED

    def _convert_to_differential_64x64(self, signed_weights_64x32):
        differential = np.zeros((self.input_size, self.output_size), dtype=float)
        differential[:, 0::2] = np.maximum(signed_weights_64x32, 0)
        differential[:, 1::2] = np.maximum(-signed_weights_64x32, 0)
        return differential

    def _print_program_info(self, original_weights):
        if self.mode == self.MODE_UNSIGNED:
            nonzero = np.count_nonzero(self._weights_internal)
            total = self.input_size * self.output_size
            print(f"[PROGRAM] UNSIGNED mode:")
            print(f"  Weight shape: {self._weights_internal.shape}")
            print(f"  Non-zero weights: {nonzero}/{total}")
            print(f"  Output size: {self.output_size}\n")
            return

        if self.mode == self.MODE_SIGNED:
            even_cols = self._weights_internal[:, 0::2]
            odd_cols = self._weights_internal[:, 1::2]
            print(f"[PROGRAM] SIGNED mode (differential):")
            print(f"  Original weight shape: {original_weights.shape}")
            print(f"  Differential storage:  {self._weights_internal.shape}")
            print(f"  Positive columns (even): {np.count_nonzero(even_cols)} non-zero")
            print(f"  Negative columns (odd):  {np.count_nonzero(odd_cols)} non-zero")
            print(f"  Differential outputs: {self.output_size // 2}  (zeros in upper half)")
            print(f"  Total output vector:  {self.output_size}  (fixed, mode-transparent)\n")
            return

        if self.mode == self.MODE_PWM_INT8_ROW_SERIAL:
            logical_out = self.output_size // 2
            print(f"[PROGRAM] PWM_INT8_ROW_SERIAL mode (4-bit cells, 8-bit ADC, row-serial):")
            print(f"  Original signed INT8 weight shape: {original_weights.shape} (expects 64×{logical_out})")
            print(f"  Stored banks per logical weight: (pos0,pos1,neg0,neg1) each 4-bit")
            print(f"  Logical outputs: {logical_out}  (padded to {self.output_size})")
            print(f"  ADC bits: {self.adc_bits}, Cell bits: {self.cell_bits}\n")
            return

    # -----------------------
    # NEW: nibble decomposition per doc
    # -----------------------
    def _decompose_signed_int8_to_4bit_banks(self, w_signed_int):
        """
        Implements:
            w_pos = max(w,0), w_neg = max(-w,0)
            w_pos = w_pos0 + 16*w_pos1, w_neg = w_neg0 + 16*w_neg1
        where each nibble is in [0,15].
        """
        w = np.asarray(w_signed_int, dtype=np.int16)
        w_pos = np.maximum(w, 0).astype(np.int16)
        w_neg = np.maximum(-w, 0).astype(np.int16)

        pos0 = (w_pos & 0xF).astype(np.uint8)
        pos1 = ((w_pos >> 4) & 0xF).astype(np.uint8)
        neg0 = (w_neg & 0xF).astype(np.uint8)
        neg1 = ((w_neg >> 4) & 0xF).astype(np.uint8)
        return tuple(self._apply_reliability_to_cell_levels(x) for x in (pos0, pos1, neg0, neg1))

    def _reconstruct_signed_int8(self):
        pos = self._w_pos0.astype(np.int16) + (self._w_pos1.astype(np.int16) << 4)
        neg = self._w_neg0.astype(np.int16) + (self._w_neg1.astype(np.int16) << 4)
        return pos - neg

    # -----------------------
    # Weight programming
    # -----------------------
    def program_weights(self, weight_matrix):
        """
        Program weights with automatic mode detection.

        - If weights are integer and shape is (64,32): programs PWM_INT8_ROW_SERIAL mode.
        - Else: falls back to legacy UNSIGNED / SIGNED float modes.
        """
        self._ecc_program_latency_cycles = 0
        self._ecc_compute_latency_cycles = 0
        self._last_tdc_timing.update(
            {
                "enabled": False,
                "first_ready_column": -1,
                "last_ready_column": -1,
                "arrival_time_min_ps": 0.0,
                "arrival_time_max_ps": 0.0,
                "pe_ready_time_ps": 0.0,
                "pe_cycle_time_ps": 0.0,
            }
        )
        self._reliability_stats = self._new_reliability_stats()
        self._iv_saturation_stats = self._new_iv_saturation_stats(self.output_size // 2)
        wm = np.asarray(weight_matrix)
        if self.safety_checks:
            self._validate_2d_weights(wm)

        logical_out = self.output_size // 2

        # NEW: INT8 PWM row-serial mode detection
        if wm.shape == (self.input_size, logical_out) and np.issubdtype(wm.dtype, np.integer):
            if self.safety_checks:
                self._validate_pwm_weight_range(wm)
            self.mode = self.MODE_PWM_INT8_ROW_SERIAL

            pos0, pos1, neg0, neg1 = self._decompose_signed_int8_to_4bit_banks(wm)
            self._w_pos0, self._w_pos1, self._w_neg0, self._w_neg1 = pos0, pos1, neg0, neg1
            self._effective_signed = self._reconstruct_signed_int8()
            self._ecc_program_latency_cycles = self._ecc_latency_cycles_for_bits(
                self._pwm_stored_weight_bits(),
                self.reliability_config.ecc_encode_latency_cycles,
            )

            # Keep legacy matrix cleared (not used in this mode)
            self._weights_internal.fill(0.0)
            self._print_program_info(wm)
            return

        # -----------------------
        # Legacy paths (float)
        # -----------------------
        weight_matrix = np.asarray(weight_matrix, dtype=float)
        if self.safety_checks:
            self._validate_finite(weight_matrix, "weight_matrix")

        if weight_matrix.shape == (self.input_size, logical_out):
            # Explicit SIGNED mode (64, 32)
            print(f"[AUTO-DETECT] Input shape {weight_matrix.shape} → SIGNED mode")
            self.mode = self.MODE_SIGNED
            self._effective_signed = weight_matrix.copy()
            self._weights_internal = self._convert_to_differential_64x64(weight_matrix)

        elif weight_matrix.shape == (self.input_size, self.output_size):
            # Auto-detect from values
            self.mode = self._detect_mode(weight_matrix)
            if self.mode == self.MODE_SIGNED:
                raise ValueError(
                    f"Cannot program a ({self.input_size}, {self.output_size}) signed weight matrix "
                    f"into a {self.output_size}-column crossbar in SIGNED (differential) mode. "
                    f"SIGNED mode requires ({self.input_size}, {logical_out}) weights "
                    f"so each signed column maps to a differential pair (even=pos, odd=neg). "
                    f"Pass a ({self.input_size}, {logical_out}) matrix instead."
                )
            self._weights_internal = weight_matrix.copy()
            self._effective_signed = None
        else:
            raise ValueError(
                f"Invalid weight shape: {weight_matrix.shape}. Expected "
                f"({self.input_size},{self.output_size}) for UNSIGNED or "
                f"({self.input_size},{logical_out}) for SIGNED/PWM_INT8."
            )

        self._print_program_info(weight_matrix)

    # -----------------------
    # NEW: ADC model
    # -----------------------
    def _adc_u(self, x_nonneg):
        """
        Unsigned ADC quantization with saturation.
        The doc budgets for 8-bit ADC measuring 4b×4b magnitudes (0..225).
        """
        if self.safety_checks:
            self._validate_finite(x_nonneg, "ADC input")
        maxv = (1 << self.adc_bits) - 1
        x = np.asarray(x_nonneg, dtype=float)
        x = np.clip(np.rint(x), 0, maxv)
        return x.astype(np.int32)

    def _max_cell_level(self):
        return (1 << self.cell_bits) - 1

    def _make_state_log_resistance(self):
        cfg = self.noise_config
        max_level = self._max_cell_level()
        levels = np.arange(max_level + 1, dtype=float)
        frac = levels / max(max_level, 1)
        r_hrs = max(float(cfg.r_hrs_ohm), 1.0e-12)
        r_lrs = max(float(cfg.r_lrs_ohm), 1.0e-12)
        return np.log(r_hrs) + frac * (np.log(r_lrs) - np.log(r_hrs))

    def _levels_to_log_resistance(self, levels):
        idx = np.clip(np.rint(levels).astype(np.int16), 0, self._max_cell_level())
        return self._state_log_resistance[idx]

    def _log_resistance_to_levels(self, log_resistance):
        grid = self._state_log_resistance
        max_level = self._max_cell_level()
        log_hrs = float(grid[0])
        log_lrs = float(grid[-1])
        denom = log_lrs - log_hrs
        if abs(denom) < 1.0e-30:
            return np.zeros_like(log_resistance, dtype=np.uint8)
        frac = (np.asarray(log_resistance, dtype=float) - log_hrs) / denom
        return np.clip(np.rint(frac * max_level), 0, max_level).astype(np.uint8)

    def _effective_bit_error_probability(self):
        cfg = self.reliability_config
        p = max(float(cfg.raw_ber_ppm), 0.0) * 1.0e-6
        if not bool(cfg.use_ecc):
            return min(p, 1.0)
        return self._post_ecc_ber_probability(p)

    def _post_ecc_ber_probability(self, raw_bit_probability):
        cfg = self.reliability_config
        p = min(max(float(raw_bit_probability), 0.0), 1.0)
        n = max(int(cfg.ecc_codeword_bits), 1)
        data_bits = max(int(cfg.ecc_data_bits), 1)
        no_error = (1.0 - p) ** n
        one_error = n * p * ((1.0 - p) ** max(n - 1, 0))
        uncorrectable = np.clip(1.0 - no_error - one_error, 0.0, 1.0)
        return min(float(uncorrectable) / data_bits, 1.0)

    def _ecc_codeword_count(self, data_bit_count):
        cfg = self.reliability_config
        data_bits = max(int(cfg.ecc_data_bits), 1)
        return int(np.ceil(max(int(data_bit_count), 0) / data_bits))

    def _ecc_latency_cycles_for_bits(self, data_bit_count, cycles_per_codeword):
        cfg = self.reliability_config
        if not (bool(cfg.enabled) and bool(cfg.use_ecc)):
            return 0
        per_codeword = max(int(cycles_per_codeword), 0)
        return self._ecc_codeword_count(data_bit_count) * per_codeword

    def _pwm_stored_weight_bits(self):
        return int(
            (
                self._w_pos0.size
                + self._w_pos1.size
                + self._w_neg0.size
                + self._w_neg1.size
            )
            * self.cell_bits
        )

    def _pwm_ecc_read_bits_for_activation(self, a0, a1):
        logical_out = self.output_size // 2
        nibble_passes = int(np.count_nonzero(a0)) + int(np.count_nonzero(a1))
        banks_per_pass = 4
        return int(nibble_passes * logical_out * banks_per_pass * self.cell_bits)

    def _endurance_at_limit(self):
        cfg = self.reliability_config
        if int(cfg.endurance_write_cycles) <= 0:
            return False
        return int(cfg.endurance_write_cycles) >= max(int(cfg.endurance_cycles), 1)

    def _has_endurance_limit_failure(self):
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

    def _apply_endurance_limit_failures(self, levels, programmable):
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

    def _has_programmed_cell_drift(self):
        cfg = self.reliability_config
        return any(
            (
                self._has_retention_drift(),
                float(cfg.relaxation_time_s) > 0.0 and float(cfg.relaxation_resistance_amplitude_ln) != 0.0,
                int(cfg.endurance_write_cycles) > 0 and float(cfg.endurance_resistance_drift_ln_per_life_fraction) != 0.0,
                float(cfg.state_sigma_ln) > 0.0,
            )
        )

    def _has_retention_drift(self):
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

    def _retention_log_resistance_shift(self, levels):
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
        return (
            np.log(10.0)
            * float(cfg.retention_log10_resistance_drift_per_decade)
            * decades
            * temp_factor
        )

    def _apply_reliability_to_cell_levels(self, cell_levels):
        levels = np.asarray(cell_levels, dtype=np.int16).copy()
        cfg = self.reliability_config
        max_level = self._max_cell_level()
        self._reliability_stats["programmed_cells"] += int(levels.size)
        if not bool(cfg.enabled):
            return np.clip(levels, 0, max_level).astype(np.uint8)

        programmable = np.ones(levels.shape, dtype=bool)
        p_lrs = np.clip(float(cfg.stuck_at_lrs_prob), 0.0, 1.0)
        if p_lrs > 0.0:
            mask = self._reliability_rng.random(levels.shape) < p_lrs
            levels[mask] = max_level
            programmable[mask] = False
            self._reliability_stats["stuck_at_lrs_cells"] += int(np.count_nonzero(mask))

        p_hrs = np.clip(float(cfg.stuck_at_hrs_prob), 0.0, 1.0)
        if p_hrs > 0.0:
            mask = (self._reliability_rng.random(levels.shape) < p_hrs) & programmable
            levels[mask] = 0
            programmable[mask] = False
            self._reliability_stats["stuck_at_hrs_cells"] += int(np.count_nonzero(mask))

        p_ber = self._effective_bit_error_probability()
        if p_ber > 0.0:
            for bit in range(self.cell_bits):
                flips = (self._reliability_rng.random(levels.shape) < p_ber) & programmable
                levels[flips] ^= (1 << bit)
                self._reliability_stats["raw_ber_bit_flips"] += int(np.count_nonzero(flips))
            levels = np.clip(levels, 0, max_level)

        if self._has_endurance_limit_failure():
            levels, programmable = self._apply_endurance_limit_failures(levels, programmable)

        if self._has_programmed_cell_drift():
            log_r = self._levels_to_log_resistance(levels)
            shift = np.zeros_like(log_r, dtype=float)

            if self._has_retention_drift():
                shift += self._retention_log_resistance_shift(levels)

            if float(cfg.relaxation_time_s) > 0.0 and float(cfg.relaxation_resistance_amplitude_ln) != 0.0:
                tau = max(float(cfg.relaxation_tau_s), 1.0e-12)
                shift += float(cfg.relaxation_resistance_amplitude_ln) * (1.0 - np.exp(-float(cfg.relaxation_time_s) / tau))

            if int(cfg.endurance_write_cycles) > 0 and float(cfg.endurance_resistance_drift_ln_per_life_fraction) != 0.0:
                life = max(float(cfg.endurance_cycles), 1.0)
                frac = min(max(float(cfg.endurance_write_cycles) / life, 0.0), 1.0)
                shift += float(cfg.endurance_resistance_drift_ln_per_life_fraction) * frac

            if float(cfg.state_sigma_ln) > 0.0:
                shift += self._reliability_rng.normal(0.0, float(cfg.state_sigma_ln), size=levels.shape)

            log_r = np.where(programmable, log_r + shift, log_r)
            levels = self._log_resistance_to_levels(log_r).astype(np.int16)

        return np.clip(levels, 0, max_level).astype(np.uint8)

    def _flicker_variance(self, i0, af, gamma):
        cfg = self.noise_config
        fmin = max(float(cfg.flicker_fmin_hz), 1.0e-12)
        fmax = max(float(cfg.bandwidth_hz), fmin)
        gamma = np.asarray(gamma, dtype=float)
        near_one = np.isclose(gamma, 1.0)
        integral = np.empty_like(gamma, dtype=float)
        integral[near_one] = np.log(fmax / fmin)
        not_one = ~near_one
        integral[not_one] = (
            fmax ** (1.0 - gamma[not_one]) - fmin ** (1.0 - gamma[not_one])
        ) / (1.0 - gamma[not_one])
        return np.maximum(np.asarray(af, dtype=float), 0.0) * (np.asarray(i0, dtype=float) ** 2) * np.maximum(integral, 0.0)

    def _read_noise_level_sigma(self, levels):
        cfg = self.noise_config
        max_level = self._max_cell_level()
        idx = np.clip(np.rint(levels).astype(np.int16), 0, max_level)
        frac = idx.astype(float) / max(max_level, 1)

        resistance = np.exp(self._state_log_resistance[idx])
        current = abs(float(cfg.vread)) / np.maximum(resistance, 1.0e-30)
        bandwidth = max(float(cfg.bandwidth_hz), 0.0)

        thermal_var = 4.0 * K_BOLTZMANN * max(float(cfg.temp_k), 0.0) * bandwidth / np.maximum(resistance, 1.0e-30)
        shot_var = 2.0 * Q_ELECTRON * current * max(float(cfg.fano), 0.0) * bandwidth
        flicker_af = (1.0 - frac) * float(cfg.flicker_af_hrs) + frac * float(cfg.flicker_af_lrs)
        flicker_gamma = (1.0 - frac) * float(cfg.flicker_gamma_hrs) + frac * float(cfg.flicker_gamma_lrs)
        flicker_var = self._flicker_variance(current, flicker_af, flicker_gamma)
        rtn_amp_frac = (1.0 - frac) * float(cfg.rtn_amp_frac_hrs) + frac * float(cfg.rtn_amp_frac_lrs)
        rtn_var = (current * np.maximum(rtn_amp_frac, 0.0)) ** 2 / 4.0
        readout_var = max(float(cfg.readout_noise_a_rms), 0.0) ** 2
        current_sigma = np.sqrt(np.maximum(thermal_var + shot_var + flicker_var + rtn_var + readout_var, 0.0))

        current_grid = abs(float(cfg.vread)) / np.exp(self._state_log_resistance)
        current_steps = np.abs(np.diff(current_grid))
        min_step = max(float(np.min(current_steps)) if current_steps.size else 1.0, 1.0e-30)
        lo = np.maximum(idx - 1, 0)
        hi = np.minimum(idx + 1, max_level)
        span = np.abs(current_grid[hi] - current_grid[lo]) / np.maximum(1, hi - lo)
        return current_sigma / np.maximum(span, min_step)

    def _adc_int4_product(self, input_nibble, cell_levels):
        if int(input_nibble) == 0:
            return np.zeros(np.asarray(cell_levels).shape, dtype=np.int32)
        levels = np.asarray(cell_levels, dtype=np.int32)
        product = int(input_nibble) * levels
        if bool(self.noise_config.enabled):
            sigma = self._read_noise_level_sigma(levels) * np.sqrt(max(int(input_nibble), 1))
            product = product + self._noise_rng.normal(0.0, sigma, size=levels.shape)
        return self._adc_u(product)

    def _iv_adc_full_scale_levels(self):
        configured = float(self.iv_config.adc_full_scale_levels)
        if configured > 0.0:
            return configured
        return float(self.iv_config.linear_rows_before_clamp) * self._max_cell_level() * self._max_cell_level()

    def _apply_iv_transfer(self, product):
        values = np.asarray(product, dtype=float)
        if not bool(self.iv_config.enabled):
            return values

        full_scale = max(self._iv_adc_full_scale_levels(), 1.0)
        self._record_iv_saturation(values, full_scale)
        mode = str(self.iv_config.mode).lower()
        if mode == "hard_clamp":
            return np.clip(values, 0.0, full_scale)
        if mode == "tanh":
            return full_scale * np.tanh(np.maximum(values, 0.0) / full_scale)
        raise ValueError(f"Unsupported IV nonlinearity mode: {self.iv_config.mode}")

    def _record_iv_saturation(self, values, full_scale):
        stats = self._iv_saturation_stats
        stats["transfer_calls"] = int(stats["transfer_calls"]) + 1
        arr = np.asarray(values, dtype=float)
        upper = arr > float(full_scale)
        lower = arr < 0.0
        upper_count = int(np.count_nonzero(upper))
        stats["upper_saturation_events"] = int(stats["upper_saturation_events"]) + upper_count
        stats["lower_clip_events"] = int(stats["lower_clip_events"]) + int(np.count_nonzero(lower))
        if upper_count <= 0:
            return
        columns = stats.get("saturated_columns")
        if not isinstance(columns, np.ndarray) or columns.size == 0:
            return
        clipped = np.flatnonzero(np.ravel(upper))
        if clipped.size:
            columns[np.mod(clipped, columns.shape[0])] = True

    def _iv_saturation_stats_dict(self):
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

    def _apply_iv_nonlinearity(self, product):
        if not bool(self.iv_config.enabled):
            return np.rint(np.asarray(product, dtype=float)).astype(np.int32)
        return self._adc_u(self._apply_iv_transfer(product))

    def _row_parallel_ranges(self):
        if not self.row_parallel_config.enabled:
            return [slice(0, self.input_size)]

        rows_per_group = int(self.row_parallel_config.rows_per_group)
        if rows_per_group > 0:
            return [
                slice(start, min(start + rows_per_group, self.input_size))
                for start in range(0, self.input_size, rows_per_group)
            ]

        groups = min(max(int(self.row_parallel_config.groups), 1), self.input_size)
        ranges = []
        for group_idx in range(groups):
            start = (group_idx * self.input_size) // groups
            end = ((group_idx + 1) * self.input_size) // groups
            if end > start:
                ranges.append(slice(start, end))
        return ranges or [slice(0, self.input_size)]

    def _tdc_payload_bits_per_value(self):
        configured = int(self.column_parallel_config.payload_bits_per_value)
        if configured > 0:
            return configured
        if self.tdc_config.enabled:
            return max(int(self.tdc_config.signed_payload_bits), 1)
        return max(int(self.adc_bits), 1)

    def _active_columns_per_step(self, logical_outputs=None):
        logical = int(logical_outputs if logical_outputs is not None else self.output_size // 2)
        if logical <= 0:
            return 0
        configured = int(self.column_parallel_config.active_columns)
        if not self.column_parallel_config.enabled or configured <= 0:
            return logical
        return min(max(configured, 1), logical)

    def estimate_output_bandwidth(self, logical_outputs=None):
        logical = max(int(logical_outputs if logical_outputs is not None else self.output_size // 2), 0)
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

    def _column_product_sum(self, input_counts, cell_levels, *, include_noise=True, row_slice=None):
        counts = np.asarray(input_counts, dtype=np.int32)
        levels = np.asarray(cell_levels, dtype=np.int32)
        if row_slice is not None:
            counts = counts[row_slice]
            levels = levels[row_slice, :]
        product = counts[:, None] * levels
        if include_noise and bool(self.noise_config.enabled):
            sigma = self._read_noise_level_sigma(levels) * np.sqrt(np.maximum(counts[:, None], 0))
            product = product + self._noise_rng.normal(0.0, sigma, size=levels.shape)
        return np.sum(product, axis=0)

    def _differential_column_product(
        self,
        input_counts,
        pos_levels,
        neg_levels,
        *,
        include_noise=True,
        row_slice=None,
    ):
        pos = self._apply_iv_nonlinearity(
            self._column_product_sum(
                input_counts,
                pos_levels,
                include_noise=include_noise,
                row_slice=row_slice,
            )
        )
        neg = self._apply_iv_nonlinearity(
            self._column_product_sum(
                input_counts,
                neg_levels,
                include_noise=include_noise,
                row_slice=row_slice,
            )
        )
        return (pos - neg).astype(np.int32)

    def _compute_pwm_column_nonlinear_int8(self, a0, a1, *, include_noise=True):
        logical_out = self.output_size // 2
        acc = np.zeros((logical_out,), dtype=np.int64)
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
        out[:logical_out] = acc.astype(np.int32)
        return out

    def _tdc_max_code(self):
        return (1 << max(int(self.tdc_config.tdc_bits), 1)) - 1

    def _tdc_effective_fine_bits(self):
        return min(max(int(self.tdc_config.fine_bits), 0), max(int(self.tdc_config.tdc_bits), 1))

    def _tdc_effective_coarse_bits(self):
        return max(int(self.tdc_config.tdc_bits), 1) - self._tdc_effective_fine_bits()

    def _tdc_gain_scale(self, exponent=None):
        shift = self.tdc_config.gain_shift if exponent is None else exponent
        return 1 << max(int(shift), 0)

    def _tdc_output_format(self):
        fmt = str(getattr(self.tdc_config, "output_format", "sar_fp")).lower()
        if fmt in {"sar", "sar_fp", "fp", "float", "float_like"}:
            return "sar_fp"
        if fmt in {"bfp", "linear", "uniform", "legacy"}:
            return "bfp"
        raise ValueError(f"Unsupported TDC output_format: {self.tdc_config.output_format}")

    def _attach_tdc_timing(self, packed, magnitude):
        """
        Attach a simple current-to-time delay-chain model.

        Larger column magnitude crosses the sensing threshold earlier. Arrival
        time is quantized to delay-chain stages; a synchronous PE becomes ready
        when its slowest logical column has been encoded.
        """

        if not self.tdc_config.timing_enabled:
            return packed

        cfg = self.tdc_config
        mag = np.maximum(np.abs(np.asarray(magnitude, dtype=float)), 0.0)
        span_ps = cfg.timing_max_ps - cfg.timing_min_ps
        analog_arrival_ps = cfg.timing_min_ps + span_ps / (1.0 + mag / cfg.timing_reference_levels)
        max_tap = int(np.ceil(span_ps / cfg.timing_stage_ps))
        tap_index = np.rint((analog_arrival_ps - cfg.timing_min_ps) / cfg.timing_stage_ps).astype(np.int32)
        tap_index = np.clip(tap_index, 0, max_tap)
        arrival_time_ps = cfg.timing_min_ps + tap_index.astype(float) * cfg.timing_stage_ps
        arrival_time_ps = np.minimum(arrival_time_ps, cfg.timing_max_ps)
        ready_time_ps = arrival_time_ps + cfg.timing_encoder_ps
        pe_ready_time_ps = float(np.max(ready_time_ps)) if ready_time_ps.size else float(cfg.timing_encoder_ps)
        pe_cycle_time_ps = pe_ready_time_ps + cfg.timing_reset_ps
        completion_order = np.argsort(ready_time_ps, kind="stable")

        self._last_tdc_timing = {
            "enabled": True,
            "first_ready_column": int(completion_order[0]) if completion_order.size else -1,
            "last_ready_column": int(completion_order[-1]) if completion_order.size else -1,
            "arrival_time_min_ps": float(np.min(arrival_time_ps)) if arrival_time_ps.size else 0.0,
            "arrival_time_max_ps": float(np.max(arrival_time_ps)) if arrival_time_ps.size else 0.0,
            "pe_ready_time_ps": pe_ready_time_ps,
            "pe_cycle_time_ps": pe_cycle_time_ps,
        }
        return replace(
            packed,
            tap_index=tap_index,
            arrival_time_ps=arrival_time_ps,
            ready_time_ps=ready_time_ps,
            pe_ready_time_ps=pe_ready_time_ps,
            pe_cycle_time_ps=pe_cycle_time_ps,
        )

    def _tdc_u(self, values, exponent=None):
        scaled = np.asarray(values, dtype=float) / float(self._tdc_gain_scale(exponent))
        return np.clip(np.rint(scaled), 0, self._tdc_max_code()).astype(np.int32)

    def _tdc_split_code(self, code):
        fine_bits = self._tdc_effective_fine_bits()
        codes = np.asarray(code, dtype=np.int32)
        if fine_bits == 0:
            return codes, np.zeros_like(codes, dtype=np.int32)
        fine_mask = (1 << fine_bits) - 1
        return (codes >> fine_bits).astype(np.int32), (codes & fine_mask).astype(np.int32)

    def _tdc_sar_magnitude_pack(self, values):
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

    def _tdc_pack_codes(self, positive_code, negative_code, exponent):
        logical_out = self.output_size // 2
        pos = np.asarray(positive_code, dtype=np.int32)
        neg = np.asarray(negative_code, dtype=np.int32)
        pos_coarse, pos_fine = self._tdc_split_code(pos)
        neg_coarse, neg_fine = self._tdc_split_code(neg)

        packed_arrays = []
        for values in (pos - neg, pos, neg, pos_coarse, pos_fine, neg_coarse, neg_fine):
            out = np.zeros((self.output_size,), dtype=np.int32)
            out[:logical_out] = np.asarray(values, dtype=np.int32)[:logical_out]
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

    def _tdc_sar_pack_magnitudes(self, positive_magnitude, negative_magnitude):
        logical_out = self.output_size // 2
        pos_code, pos_coarse, pos_fine, pos_mant, pos_exp = self._tdc_sar_magnitude_pack(positive_magnitude)
        neg_code, neg_coarse, neg_fine, neg_mant, neg_exp = self._tdc_sar_magnitude_pack(negative_magnitude)

        signed_mantissa = pos_mant - neg_mant
        exponent = np.maximum(pos_exp, neg_exp).astype(np.int32)

        packed_arrays = []
        for values in (signed_mantissa, pos_code, neg_code, pos_coarse, pos_fine, neg_coarse, neg_fine):
            out = np.zeros((self.output_size,), dtype=np.int32)
            out[:logical_out] = np.asarray(values, dtype=np.int32)[:logical_out]
            packed_arrays.append(out)

        exponent_out = np.zeros((self.output_size,), dtype=np.int32)
        exponent_out[:logical_out] = exponent[:logical_out]
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

    def _tdc_integer_packed(self, values, *, format_name="tdc_integer"):
        logical_out = self.output_size // 2
        mantissa = np.zeros((self.output_size,), dtype=np.int64)
        mantissa[:logical_out] = np.asarray(values, dtype=np.int64)[:logical_out]
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

    def _tdc_differential_pack(self, signed_diff, *, exponent=None):
        diff = np.asarray(signed_diff, dtype=float)
        ab_branch = self._apply_iv_transfer(np.maximum(diff, 0.0))
        ba_branch = self._apply_iv_transfer(np.maximum(-diff, 0.0))
        if self._tdc_output_format() == "sar_fp":
            packed = self._tdc_sar_pack_magnitudes(ab_branch, ba_branch)
            return self._attach_tdc_timing(packed, ab_branch + ba_branch)

        shift = max(int(self.tdc_config.gain_shift if exponent is None else exponent), 0)
        ab_code = self._tdc_u(ab_branch, exponent=shift)
        ba_code = self._tdc_u(ba_branch, exponent=shift)
        packed = self._tdc_pack_codes(ab_code, ba_code, shift)
        return self._attach_tdc_timing(packed, ab_branch + ba_branch)

    def _pwm_full_column_sums(self, input_counts, *, include_noise=True, row_slice=None):
        pos0 = self._column_product_sum(
            input_counts, self._w_pos0, include_noise=include_noise, row_slice=row_slice
        )
        pos1 = self._column_product_sum(
            input_counts, self._w_pos1, include_noise=include_noise, row_slice=row_slice
        )
        neg0 = self._column_product_sum(
            input_counts, self._w_neg0, include_noise=include_noise, row_slice=row_slice
        )
        neg1 = self._column_product_sum(
            input_counts, self._w_neg1, include_noise=include_noise, row_slice=row_slice
        )

        if self.slice_gain0 != 1.0:
            pos0 = pos0 * self.slice_gain0
            neg0 = neg0 * self.slice_gain0
        if self.slice_gain1 != 1.0:
            pos1 = pos1 * self.slice_gain1
            neg1 = neg1 * self.slice_gain1
        return pos0 + 16.0 * pos1, neg0 + 16.0 * neg1

    def _compute_pwm_differential_tdc_packed_int8(self, activation_counts, *, include_noise=True, exponent=None):
        logical_out = self.output_size // 2
        row_ranges = self._row_parallel_ranges()
        if len(row_ranges) > 1:
            acc = np.zeros((logical_out,), dtype=np.int64)
            for row_slice in row_ranges:
                positive, negative = self._pwm_full_column_sums(
                    activation_counts,
                    include_noise=include_noise,
                    row_slice=row_slice,
                )
                packed_part = self._tdc_differential_pack(positive - negative, exponent=exponent)
                acc += packed_part.reconstruct_integer()[:logical_out]
            return self._tdc_integer_packed(acc, format_name="tdc_row_group_sum")

        positive, negative = self._pwm_full_column_sums(activation_counts, include_noise=include_noise)
        return self._tdc_differential_pack(positive - negative, exponent=exponent)

    def _compute_pwm_differential_tdc_int8(self, activation_counts, *, include_noise=True):
        packed = self._compute_pwm_differential_tdc_packed_int8(activation_counts, include_noise=include_noise)
        if packed.format == "tdc_sar_fp":
            return packed.reconstruct_integer().astype(np.int32)
        return packed.mantissa.astype(np.int32)

    def _compute_pwm_corrected_tdc_packed_int8(self, activation_counts, *, zero_point=128, include_noise=True):
        z = int(np.clip(int(zero_point), 0, 255))
        z_vec = np.full((self.input_size,), z, dtype=np.int32)
        logical_out = self.output_size // 2
        row_ranges = self._row_parallel_ranges()
        if len(row_ranges) > 1:
            acc = np.zeros((logical_out,), dtype=np.int64)
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
                acc += packed_part.reconstruct_integer()[:logical_out]
            return self._tdc_integer_packed(acc, format_name="tdc_row_group_sum")

        positive, negative = self._pwm_full_column_sums(activation_counts, include_noise=include_noise)
        zero_positive, zero_negative = self._pwm_full_column_sums(z_vec, include_noise=False)
        return self._tdc_differential_pack((positive - negative) - (zero_positive - zero_negative))

    def compute_tdc_packed(self, input_vector):
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

    def compute_tdc_packed_corrected(self, input_vector, zero_point=128):
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

    def pwm_zero_point_output(self, zero_point=128):
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

    def pwm_zero_point_packed_output(self, zero_point=128, exponent=None):
        if self.mode != self.MODE_PWM_INT8_ROW_SERIAL:
            raise RuntimeError("PWM zero-point output is only defined for PWM_INT8_ROW_SERIAL weights.")
        z = int(np.clip(int(zero_point), 0, 255))
        z_vec = np.full((self.input_size,), z, dtype=np.int32)
        return self._compute_pwm_differential_tdc_packed_int8(z_vec, include_noise=False, exponent=exponent)

    def _compute_pwm_row_serial_int8(self, a_u8_vec):
        """
        Implements the doc's nibble-slicing + two-pass PWM per row:

            a = a0 + 16*a1, with a0=a&0xF, a1=a>>4
            w0 = (w_pos0 - w_neg0), w1 = (w_pos1 - w_neg1)
            p00=a0*w0, p01=a0*w1, p10=a1*w0, p11=a1*w1
            a*w = p00 + (p01<<4) + (p10<<4) + (p11<<8)

        Accumulates across rows into INT32.
        """
        logical_out = self.output_size // 2

        a = np.asarray(a_u8_vec)
        if self.safety_checks:
            a = self._validate_pwm_input_vector(a)
        elif a.shape[0] != self.input_size:
            raise ValueError(f"Input size mismatch: expected {self.input_size}, got {a.shape[0]}")

        # Force to 0..255
        a_u = (a.astype(np.int32) & 0xFF).astype(np.int32)
        a0 = (a_u & 0xF).astype(np.int32)     # 0..15
        a1 = (a_u >> 4).astype(np.int32)      # 0..15
        self._ecc_compute_latency_cycles += self._ecc_latency_cycles_for_bits(
            self._pwm_ecc_read_bits_for_activation(a0, a1),
            self.reliability_config.ecc_decode_latency_cycles,
        )

        if bool(self.tdc_config.enabled):
            return self._compute_pwm_differential_tdc_int8(a_u)

        if bool(self.iv_config.enabled):
            return self._compute_pwm_column_nonlinear_int8(a0, a1)

        acc = np.zeros((logical_out,), dtype=np.int32)

        # Row-serial: one row contributes per "integration window"
        for i in range(self.input_size):
            ai0 = int(a0[i])
            ai1 = int(a1[i])
            if ai0 == 0 and ai1 == 0:
                continue

            # Slice 0 (LS nibble) per polarity
            wpos0 = self._w_pos0[i, :].astype(np.int32)
            wneg0 = self._w_neg0[i, :].astype(np.int32)
            # Slice 1 (MS nibble) per polarity
            wpos1 = self._w_pos1[i, :].astype(np.int32)
            wneg1 = self._w_neg1[i, :].astype(np.int32)

            # Pass for a0
            if ai0 != 0:
                adc_pos00 = self._adc_int4_product(ai0, wpos0)
                adc_neg00 = self._adc_int4_product(ai0, wneg0)
                p00 = (adc_pos00 - adc_neg00).astype(np.int32)
                if self.slice_gain0 != 1.0:
                    p00 = np.rint(p00 * self.slice_gain0).astype(np.int32)

                adc_pos01 = self._adc_int4_product(ai0, wpos1)
                adc_neg01 = self._adc_int4_product(ai0, wneg1)
                p01 = (adc_pos01 - adc_neg01).astype(np.int32)
                if self.slice_gain1 != 1.0:
                    p01 = np.rint(p01 * self.slice_gain1).astype(np.int32)
            else:
                p00 = 0
                p01 = 0

            # Pass for a1
            if ai1 != 0:
                adc_pos10 = self._adc_int4_product(ai1, wpos0)
                adc_neg10 = self._adc_int4_product(ai1, wneg0)
                p10 = (adc_pos10 - adc_neg10).astype(np.int32)
                if self.slice_gain0 != 1.0:
                    p10 = np.rint(p10 * self.slice_gain0).astype(np.int32)

                adc_pos11 = self._adc_int4_product(ai1, wpos1)
                adc_neg11 = self._adc_int4_product(ai1, wneg1)
                p11 = (adc_pos11 - adc_neg11).astype(np.int32)
                if self.slice_gain1 != 1.0:
                    p11 = np.rint(p11 * self.slice_gain1).astype(np.int32)
            else:
                p10 = 0
                p11 = 0

            # Exact digital recombination (shift-add)
            # a*w = p00 + (p01<<4) + (p10<<4) + (p11<<8)
            acc += (p00 + (p01 << 4) + (p10 << 4) + (p11 << 8)).astype(np.int32)

        out = np.zeros((self.output_size,), dtype=np.int32)
        out[:logical_out] = acc
        return out

    # -----------------------
    # Compute
    # -----------------------
    def compute(self, input_vector):
        """
        Mode-aware MVM / MAC.
        Output is always length 64.
        """
        if self.mode is None:
            raise RuntimeError("Weights not programmed yet. Call program_weights() first.")

        if self.mode == self.MODE_PWM_INT8_ROW_SERIAL:
            return self._compute_pwm_row_serial_int8(input_vector)

        # legacy float modes
        if self.safety_checks:
            input_vector = self._validate_legacy_input_vector(input_vector)
        elif len(input_vector) != self.input_size:
            raise ValueError(f"Input size mismatch: expected {self.input_size}, got {len(input_vector)}")

        raw = np.dot(np.asarray(input_vector, dtype=float), self._weights_internal)  # (output_size,)

        if self.mode == self.MODE_UNSIGNED:
            return raw

        # SIGNED
        diff = raw[0::2] - raw[1::2]  # (32,)
        result = np.zeros(self.output_size, dtype=float)
        result[: len(diff)] = diff
        return result

    def compute_batch(self, input_vectors):
        """
        Compute a batch from the currently programmed crossbar state.

        The ideal PWM path reconstructs the signed values from the four
        programmed 4-bit differential cell banks, then evaluates every input
        vector together.  Non-ideal noise, IV, TDC, or row-group models retain
        the scalar path because those models update per-read state.
        """
        if self.mode is None:
            raise RuntimeError("Weights not programmed yet. Call program_weights() first.")

        vectors = np.asarray(input_vectors)
        if vectors.ndim != 2 or vectors.shape[1] != self.input_size:
            raise ValueError(
                f"Expected input batch [B,{self.input_size}], got {vectors.shape}"
            )

        use_scalar_model = (
            bool(self.noise_config.enabled)
            or bool(self.iv_config.enabled)
            or bool(self.tdc_config.enabled)
            or len(self._row_parallel_ranges()) > 1
        )
        if self.mode != self.MODE_PWM_INT8_ROW_SERIAL or use_scalar_model:
            return np.stack([self.compute(vector) for vector in vectors], axis=0)

        values = np.asarray(vectors, dtype=np.int64)
        if self.safety_checks and values.size and (
            int(values.min()) < 0 or int(values.max()) > 255
        ):
            raise ValueError(
                "PWM_INT8_ROW_SERIAL inputs must be unsigned activation "
                "values in [0, 255]"
            )
        activations = (values & 0xFF).astype(np.int32)
        programmed_weights = self._reconstruct_signed_int8().astype(
            np.int32,
            copy=False,
        )
        accumulators = activations @ programmed_weights
        output = np.zeros(
            (activations.shape[0], self.output_size),
            dtype=np.int32,
        )
        output[:, : programmed_weights.shape[1]] = accumulators.astype(
            np.int32,
            copy=False,
        )
        return output

    # -----------------------
    # Introspection helpers
    # -----------------------
    def get_mode(self):
        return self.mode if self.mode else "NOT_PROGRAMMED"

    def get_output_size(self):
        if self.mode is None:
            raise RuntimeError("Weights not programmed yet.")
        return self.output_size

    def get_effective_weights(self):
        """
        Logical weight matrix:
          - UNSIGNED: (64,64)
          - SIGNED  : (64,32) signed float
          - PWM_INT8_ROW_SERIAL: (64,32) signed int16 (reference)
        """
        if self.mode is None:
            raise RuntimeError("Weights not programmed yet.")

        if self.mode == self.MODE_UNSIGNED:
            return self._weights_internal.copy()

        if self.mode == self.MODE_PWM_INT8_ROW_SERIAL:
            return self._reconstruct_signed_int8()

        # SIGNED float: reconstruct signed from differential
        even = self._weights_internal[:, 0::2]
        odd = self._weights_internal[:, 1::2]
        return even - odd

    def get_info(self):
        info = {
            'mode': self.get_mode(),
            'input_size': self.input_size,
            'physical_output_size': self.output_size,
            'weight_shape': getattr(self._weights_internal, "shape", None),
            'safety_checks': self.safety_checks,
            'noise_config': self.noise_config.as_dict(),
            'reliability_config': self.reliability_config.as_dict(),
            'iv_config': self.iv_config.as_dict(),
            'iv_nonlinearity_enabled': bool(self.iv_config.enabled),
            'iv_nonlinearity_mode': self.iv_config.mode,
            'iv_adc_full_scale_levels': self._iv_adc_full_scale_levels(),
            'iv_linear_rows_before_clamp': self.iv_config.linear_rows_before_clamp,
            'iv_saturation_stats': self._iv_saturation_stats_dict(),
            'tdc_enabled': bool(self.tdc_config.enabled),
            'tdc_config': self.tdc_config.as_dict(),
            'tdc_gain_scale': self._tdc_gain_scale(),
            'tdc_effective_coarse_bits': self._tdc_effective_coarse_bits(),
            'tdc_effective_fine_bits': self._tdc_effective_fine_bits(),
            'pe_row_parallel_config': self.row_parallel_config.as_dict(),
            'pe_row_parallel_groups': len(self._row_parallel_ranges()),
            'pe_column_parallel_config': self.column_parallel_config.as_dict(),
            'pe_output_bandwidth': self.estimate_output_bandwidth(),
            'noise_seed': self.noise_seed,
            'reliability_seed': self.reliability_seed,
            'reliability_stats': dict(self._reliability_stats),
            'latency_model': {
                'ecc_program_latency_cycles': int(self._ecc_program_latency_cycles),
                'ecc_compute_latency_cycles': int(self._ecc_compute_latency_cycles),
                'ecc_total_latency_cycles': int(
                    self._ecc_program_latency_cycles + self._ecc_compute_latency_cycles
                ),
                'tdc_timing': dict(self._last_tdc_timing),
            },
        }
        if self.mode == self.MODE_PWM_INT8_ROW_SERIAL:
            info.update({
                'logical_outputs': self.output_size // 2,
                'adc_bits': self.adc_bits,
                'cell_bits': self.cell_bits,
                'slice_gain0': self.slice_gain0,
                'slice_gain1': self.slice_gain1,
                'pwm_stored_weight_bits': self._pwm_stored_weight_bits(),
            })
        return info
