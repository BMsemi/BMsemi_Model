"""MIRA-style numerical reference: W[in, out], INT5 tiles and INT6 groups.

This is not an ADC/noise/energy model. Scales are encoded explicitly, but
rescaling uses float64. Hardware fixed-point widths require a separate model.
"""
from dataclasses import dataclass
from typing import Callable

import numpy as np


@dataclass
class ScaledWeights:
    values: np.ndarray
    block_scales: np.ndarray
    scale_codes: np.ndarray
    tensor_exponent: int
    block_size: int

    def reconstruct(self):
        scales = np.repeat(np.repeat(self.block_scales, self.block_size, 0),
                           self.block_size, 1)
        return self.values * scales[:self.values.shape[0], :self.values.shape[1]] * 2.0 ** self.tensor_exponent


@dataclass
class ScaledInputs:
    values: np.ndarray
    scale_codes: np.ndarray
    fractional_bits: int
    block_size: int

    @property
    def scales(self):
        return self.scale_codes.astype(np.float64) * 2.0 ** -self.fractional_bits

    def reconstruct(self):
        return self.values * np.repeat(self.scales, self.block_size)[:self.values.size]


def _finite(values, ndim):
    a = np.asarray(values, dtype=np.float64)
    if a.ndim != ndim or a.size == 0 or not np.all(np.isfinite(a)):
        raise ValueError(f"Expected a nonempty finite {ndim}-dimensional array")
    return a


def _block_size(value):
    if not isinstance(value, (int, np.integer)) or value <= 0:
        raise ValueError("block_size must be a positive integer")


def e4m3_positive_grid():
    """Nonnegative finite E4M3FN codes 0..126 (bias 7, max 448)."""
    codes = np.arange(127)
    exponent, mantissa = codes >> 3, codes & 7
    return np.where(exponent == 0, mantissa * 2.0 ** -9,
                    (1.0 + mantissa / 8.0) * 2.0 ** (exponent - 7))


def encode_weight_scale(value):
    """Nearest E4M3FN; ties to even code; positive scales never round to zero."""
    grid = e4m3_positive_grid()
    if not np.isfinite(value) or value < 0 or value > 448:
        raise ValueError("Weight scale must be finite and in [0, 448]")
    if value == 0:
        return 0, 0.0
    error = np.abs(grid - value)
    candidates = np.flatnonzero(error == error.min())
    even = candidates[candidates % 2 == 0]
    code = max(1, int(even[0] if even.size else candidates[0]))
    return code, float(grid[code])


def quantize_weights(weights, block_size=16):
    """One E4M3FN scale per 2D tile, power-of-two tensor scale, q in [-15,15]."""
    w = _finite(weights, 2)
    _block_size(block_size)
    peak = float(np.abs(w).max())
    exponent = int(np.ceil(np.log2(peak) - np.log2(15 * 448))) if peak else 0
    if not -127 <= exponent <= 127:
        raise ValueError("Weight tensor scale exceeds finite E8M0 exponent range")
    normalized = w / (2.0 ** exponent)
    shape = tuple((s + block_size - 1) // block_size for s in w.shape)
    scales = np.zeros(shape, dtype=np.float64)
    codes = np.zeros(shape, dtype=np.uint8)
    q = np.zeros(w.shape, dtype=np.int64)
    for bi in range(shape[0]):
        for bj in range(shape[1]):
            region = (slice(bi * block_size, (bi + 1) * block_size),
                      slice(bj * block_size, (bj + 1) * block_size))
            tile = normalized[region]
            code, scale = encode_weight_scale(min(448.0, float(np.abs(tile).max()) / 15))
            codes[bi, bj], scales[bi, bj] = code, scale
            if scale:
                q[region] = np.clip(np.rint(tile / scale), -15, 15).astype(np.int64)
    return ScaledWeights(q, scales, codes, exponent, block_size)


def quantize_inputs(inputs, block_size=16, fractional_bits=None):
    """Symmetric INT6 [-31,31], unsigned 16-bit fixed-point group scales.

    An explicit fractional_bits models a calibrated binary point. If omitted,
    choose one for this input vector to fit its largest scale (reference only).
    Nonzero scales underflowing the chosen format clamp to one scale LSB.
    Scale overflow is an error rather than silent saturation.
    """
    x = _finite(inputs, 1)
    _block_size(block_size)
    raw = np.array([np.abs(x[i:i + block_size]).max() / 31
                    for i in range(0, x.size, block_size)])
    if fractional_bits is None:
        peak = float(raw.max())
        fractional_bits = int(np.floor(np.log2(65535) - np.log2(peak))) if peak else 16
        fractional_bits = max(-126, min(126, fractional_bits))
    if not isinstance(fractional_bits, (int, np.integer)) or not -126 <= fractional_bits <= 126:
        raise ValueError("fractional_bits must be an integer in [-126,126]")
    scaled = np.ldexp(raw, int(fractional_bits))
    rounded = np.rint(scaled)
    if np.any(rounded > 65535) or not np.all(np.isfinite(rounded)):
        raise ValueError("Input scale overflow: choose fewer fractional bits")
    codes = np.where(raw > 0, np.maximum(rounded, 1), 0).astype(np.uint16)
    scales = codes.astype(np.float64) * 2.0 ** -fractional_bits
    q = np.zeros(x.shape, dtype=np.int64)
    for group, scale in enumerate(scales):
        region = slice(group * block_size, (group + 1) * block_size)
        if scale:
            q[region] = np.clip(np.rint(x[region] / scale), -31, 31).astype(np.int64)
    return ScaledInputs(q, codes, int(fractional_bits), block_size)


def differential_cells(weights):
    """One pair of 4-bit cells per signed INT5 weight, with no bit slicing."""
    q = np.asarray(weights)
    if not np.issubdtype(q.dtype, np.integer) or np.any(np.abs(q) > 15):
        raise ValueError("Differential weights must be integers in [-15,15]")
    return np.maximum(q, 0), np.maximum(-q, 0)


def ideal_unsigned_read(positive_cells, negative_cells, pulses):
    """Stateless ideal differential read. Pulses are nonnegative DAC codes."""
    return pulses @ positive_cells - pulses @ negative_cells


def execute(weights: ScaledWeights, inputs: ScaledInputs, dac_bits=2,
            read: Callable = ideal_unsigned_read):
    """Read each tile/cycle, reconstruct signed partial sums, then apply scales.

    read(positive_cells, negative_cells, unsigned_pulses) must return one raw,
    unscaled dot product per logical output column, independently each call.
    A real PWM adapter must establish units, mapping, and reset behavior first.
    Signed inputs use separate positive/negative reads, not two's-complement
    pulses. This is a correctness reference, not MIRA's timing specification.
    """
    if weights.values.shape[0] != inputs.values.size or weights.block_size != inputs.block_size:
        raise ValueError("Weight input dimension and group size must match inputs")
    if not isinstance(dac_bits, (int, np.integer)) or not 1 <= dac_bits <= 6:
        raise ValueError("dac_bits must be an integer from 1 to 6")
    b = weights.block_size
    output = np.zeros(weights.values.shape[1], dtype=np.float64)
    trace = []
    for bi in range(weights.block_scales.shape[0]):
        row = slice(bi * b, (bi + 1) * b)
        x = inputs.values[row]
        for bj in range(weights.block_scales.shape[1]):
            col = slice(bj * b, (bj + 1) * b)
            positive, negative = differential_cells(weights.values[row, col])
            # Stateful adapters program once per logical tile, retaining defects
            # and noise RNG state across all sign passes and DAC cycles.
            prepare = getattr(read, "prepare_tile", None)
            if prepare is not None:
                prepare((bi, bj), positive, negative)
            partial = np.zeros(positive.shape[1], dtype=np.float64)
            cycles = []
            # INT6 symmetric magnitude needs five bits; sign handled separately.
            for shift in range(0, 5, dac_bits):
                chunk = (np.abs(x) >> shift) & ((1 << dac_bits) - 1)
                cycle = np.zeros_like(partial)
                for sign in (1, -1):
                    pulses = np.where((x >= 0) if sign == 1 else (x < 0), chunk, 0)
                    if np.any(pulses):
                        value = np.asarray(read(positive, negative, pulses), dtype=np.float64)
                        if value.shape != partial.shape or not np.all(np.isfinite(value)):
                            raise ValueError("Backend must return a finite raw output vector")
                        cycle += sign * value
                partial += cycle * (1 << shift)
                cycles.append({"shift": shift, "raw_output": cycle.tolist()})
            scale = (weights.block_scales[bi, bj] * 2.0 ** weights.tensor_exponent
                     * inputs.scales[bi])
            output[col] += partial * scale
            trace.append({"input_block": bi, "output_block": bj,
                          "integer_partial": partial.tolist(), "combined_scale": float(scale),
                          "cycles": cycles})
    return output, trace


def requantize_output(output, fractional_bits):
    """Optional signed 16-bit destination, nearest-even rounding and saturation."""
    if not isinstance(fractional_bits, (int, np.integer)) or not -126 <= fractional_bits <= 126:
        raise ValueError("output fractional bits must be an integer in [-126,126]")
    values = _finite(output, 1)
    with np.errstate(over="ignore"):
        rounded = np.rint(np.ldexp(values, int(fractional_bits)))
    saturated = int(np.count_nonzero((rounded < -32768) | (rounded > 32767)))
    codes = np.clip(rounded, -32768, 32767).astype(np.int16)
    return codes, codes.astype(np.float64) * 2.0 ** -fractional_bits, saturated
