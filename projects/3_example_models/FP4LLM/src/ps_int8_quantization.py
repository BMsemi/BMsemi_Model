"""PS-side format bridge for signed-INT8 X2 MVM engines.

This module prepares operands and reconstructs integer results. It performs no
GEMM, nonlinear model operation, MMIO, or FPGA dispatch. Scales span the full
reduction dimension and must be reused unchanged across hardware C tiles.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator

import numpy as np


def unpack_bf16(words: np.ndarray) -> np.ndarray:
    """Decode raw BF16 words without interpreting them as integer magnitudes."""
    words = np.asarray(words)
    if words.dtype.kind != 'u' or words.dtype.itemsize != 2:
        raise TypeError('Expected an array of uint16 BF16 words.')
    return (words.astype(np.uint32) << 16).view(np.float32)


def round_bf16(values: np.ndarray) -> np.ndarray:
    """BF16 round-to-nearest-even, retained in a float32 container."""
    values = np.asarray(values)
    if values.dtype.kind != 'f':
        raise TypeError('Expected floating values; decode raw BF16 with unpack_bf16 first.')
    result = np.array(values, dtype=np.float32, copy=True, order='C')
    if not np.isfinite(result).all():
        raise ValueError('INT8 conversion requires finite operands.')
    words = result.view(np.uint32)
    words += np.uint32(0x7FFF) + ((words >> 16) & np.uint32(1))
    words &= np.uint32(0xFFFF0000)
    if not np.isfinite(result).all():
        raise ValueError('Operand overflows BF16.')
    return result


def _quantize_rows(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    # Float64 scale calculation avoids an underflowing scale for BF16 subnormals.
    peak = np.max(np.abs(values.astype(np.float64)), axis=1)
    scale = np.where(peak == 0, 1.0, peak / 127.0)
    codes = np.clip(np.rint(values / scale[:, None]), -127, 127).astype(np.int8)
    return np.ascontiguousarray(codes), scale


@dataclass(frozen=True)
class MvmTile:
    engine_id: int
    token_start: int
    channel_start: int
    output_start: int
    activations: np.ndarray  # signed INT8 [batch, C]; pass directly to existing packer
    weights: np.ndarray  # signed INT8 [C, K]


@dataclass(frozen=True)
class PreparedMVM:
    activations: np.ndarray  # [T,C]
    weights: np.ndarray  # [C,K]
    activation_scales: np.ndarray  # [T], fixed over all C chunks
    weight_scales: np.ndarray  # [K], fixed over all C chunks

    def tiles(self) -> Iterator[MvmTile]:
        """Describe <=32x32 MVM transfers, output tiles assigned to engines 0-3.

        This is a work iterator, not a concurrent FPGA scheduler. The caller must
        merge all returned C partials before calling reconstruct().
        """
        tokens, channels = self.activations.shape
        outputs = self.weights.shape[1]
        for t in range(0, tokens, 32):
            for k in range(0, outputs, 32):
                for c in range(0, channels, 32):
                    yield MvmTile(
                        (k // 32) % 4, t, c, k,
                        np.ascontiguousarray(self.activations[t:t+32, c:c+32]),
                        np.ascontiguousarray(self.weights[c:c+32, k:k+32]),
                    )

    def reconstruct(self, integer_sum: np.ndarray, *, bf16: bool = True) -> np.ndarray:
        """Apply PS-side scales to a complete integer reduction, optionally BF16 round.

        integer_sum must already contain the sum of every FPGA C-tile result.
        This function does not compute or merge those results on the host.
        """
        integer_sum = np.asarray(integer_sum)
        expected = (self.activations.shape[0], self.weights.shape[1])
        if integer_sum.shape != expected or integer_sum.dtype.kind != 'i':
            raise ValueError(f'Expected signed integer accumulator array with shape {expected}.')
        output = (integer_sum.astype(np.float64) * self.activation_scales[:, None]
                  * self.weight_scales[None, :]).astype(np.float32)
        if not np.isfinite(output).all():
            raise ValueError('Reconstructed output overflows float32.')
        return round_bf16(output) if bf16 else output


def prepare_mvm(activations: np.ndarray, weights_kc: np.ndarray) -> PreparedMVM:
    """Direct BF16 -> INT8 conversion, without the old W7/A7 intermediate step.

    Input X is [T,C]; checkpoint weights are [K,C]. Use unpack_bf16() first for
    raw checkpoint words. Both projection and dynamic attention operands can
    use this bridge by supplying the right operand in [K,C] orientation.
    """
    x, w = round_bf16(activations), round_bf16(weights_kc)
    if x.ndim != 2 or w.ndim != 2 or min(*x.shape, *w.shape) == 0:
        raise ValueError('Expected non-empty [T,C] inputs and [K,C] weights.')
    if x.shape[1] != w.shape[1]:
        raise ValueError('Input and weight reduction dimensions do not match.')
    # Conservative bound for a complete INT32 sum; each 32-row tile is smaller.
    if x.shape[1] * 127 * 127 > np.iinfo(np.int32).max:
        raise ValueError('Full reduction can overflow INT32; a wider accumulator is required.')
    x_q, sx = _quantize_rows(x)
    w_q, sw = _quantize_rows(w)
    return PreparedMVM(x_q, np.ascontiguousarray(w_q.T), sx, sw)
