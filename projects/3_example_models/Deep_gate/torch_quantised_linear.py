"""
Torch Quantised Linear Layer - Python translation of torch_quantised_linear.jinja2

This implements a quantised linear (fully-connected) layer using:
  - int8 weights
  - int32 biases (adjusted/pre-shifted)
  - per-channel requantisation multiplier and shift
"""

import numpy as np


def sat_round_doubling_high_mul(acc: int, mul: int) -> int:
    """
    Saturating rounding doubling high multiply (32-bit).

    Computes (acc * mul + 2^30) >> 31, saturating at INT32_MAX
    when both inputs are INT32_MIN.
    """
    INT32_MIN = -(1 << 31)
    INT32_MAX = (1 << 31) - 1

    if acc == INT32_MIN and mul == INT32_MIN:
        return INT32_MAX

    wide = acc * mul  # int64 equivalent (Python ints are arbitrary precision)
    return (wide + (1 << 30)) >> 31


def round_shift_right(acc: int, shift: int) -> int:
    """
    Rounding right shift.

    Shifts acc right by `shift` bits with round-half-up behaviour
    (ties round away from negative infinity).
    """
    if shift <= 0:
        return acc

    mask = (1 << shift) - 1
    half = 1 << (shift - 1)
    rem = acc & mask
    base = acc >> shift
    round_up = (rem > half) or (rem == half and acc >= 0)
    return base + (1 if round_up else 0)


def torch_quantised_linear_forward(
    input_data: np.ndarray,       # int8, shape (input_features,)
    weights: np.ndarray,          # int8, shape (output_features, input_features)
    bias: np.ndarray,             # int32, shape (output_features,)
    requant_multiplier: np.ndarray,  # int32, shape (output_features,)
    requant_shift: np.ndarray,    # int32, shape (output_features,)
    output_zeropoint: int = 0,
    output_dtype=np.int8,
) -> np.ndarray:
    """
    Quantised linear (fully-connected) forward pass.

    For each output neuron:
      1. Accumulate: acc = bias + sum(input * weight)  [int32]
      2. Requantise:  acc = sat_round_doubling_high_mul(acc, multiplier)
                      acc = round_shift_right(acc, shift)
      3. Apply output transformations (clamp + add zero-point)

    Args:
        input_data:          1-D int8 array of length input_features.
        weights:             2-D int8 array, one row per output neuron.
        bias:                1-D int32 array of per-output adjusted biases.
        requant_multiplier:  1-D int32 array of per-output multipliers.
        requant_shift:       1-D int32 array of per-output right-shifts.
        output_zeropoint:    Scalar zero-point added after requantisation.
        output_dtype:        NumPy dtype for the returned array (default int8).

    Returns:
        1-D array of shape (output_features,) with dtype `output_dtype`.
    """
    output_features = weights.shape[0]
    output = np.empty(output_features, dtype=np.int32)

    for oc in range(output_features):
        # Accumulate dot product in int32
        acc = int(bias[oc])
        for ic in range(len(input_data)):
            acc += int(input_data[ic]) * int(weights[oc, ic])

        # Requantise
        acc = sat_round_doubling_high_mul(acc, int(requant_multiplier[oc]))
        acc = round_shift_right(acc, int(requant_shift[oc]))

        # Output transformation: clamp to dtype range and add zero-point
        info = np.iinfo(output_dtype)
        acc = max(info.min, min(info.max, acc + output_zeropoint))

        output[oc] = acc

    return output.astype(output_dtype)


# ---------------------------------------------------------------------------
# Vectorised (fast) version using NumPy — equivalent result
# ---------------------------------------------------------------------------

def torch_quantised_linear_forward_vectorised(
    input_data: np.ndarray,
    weights: np.ndarray,
    bias: np.ndarray,
    requant_multiplier: np.ndarray,
    requant_shift: np.ndarray,
    output_zeropoint: int = 0,
    output_dtype=np.int8,
) -> np.ndarray:
    """
    Vectorised NumPy equivalent of torch_quantised_linear_forward.
    Produces the same results as the scalar version but much faster.
    """
    INT32_MIN = np.int64(-(1 << 31))
    INT32_MAX = np.int64((1 << 31) - 1)

    # Step 1: dot product + bias  (int32 accumulation)
    acc = bias.astype(np.int64) + weights.astype(np.int64) @ input_data.astype(np.int64)

    # Step 2a: sat_round_doubling_high_mul (vectorised)
    mul = requant_multiplier.astype(np.int64)
    both_min = (acc == INT32_MIN) & (mul == INT32_MIN)
    wide = acc * mul
    acc = (wide + (1 << 30)) >> 31
    acc[both_min] = INT32_MAX

    # Step 2b: round_shift_right (vectorised)
    shift = requant_shift.astype(np.int64)
    mask = (1 << shift) - 1
    half = np.where(shift > 0, 1 << (shift - 1), 0)
    rem = acc & mask
    base = acc >> shift
    round_up = (rem > half) | ((rem == half) & (acc >= 0))
    acc = np.where(shift > 0, base + round_up.astype(np.int64), acc)

    # Step 3: clamp + zero-point
    info = np.iinfo(output_dtype)
    acc = np.clip(acc + output_zeropoint, info.min, info.max)

    return acc.astype(output_dtype)
