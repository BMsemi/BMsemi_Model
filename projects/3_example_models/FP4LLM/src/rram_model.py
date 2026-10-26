"""
RRAM conductance noise models for FP4 weight storage.
Three error modes:
  1. Relative programming noise:   G_noisy = G_target × (1 + σ × N(0,1))
  2. Stuck cells:                  cell forced to G_min or G_max with prob p_stuck
  3. Sign path mismatch:           sign bit flipped with prob p_sign
"""
import torch
import numpy as np
from .fp4_utils import (quantize_to_fp4, dequantize_fp4, FP4_MAGNITUDES,
                         FP4_MAX, FP4_INT_EQUIV, WEIGHT_TARGET_NAMES)

# Conductance levels proportional to FP4 magnitudes (normalized to G_max = 1)
G_LEVELS = [m / FP4_MAX for m in FP4_MAGNITUDES]   # {0, 1/12, 1/6, 1/4, 1/3, 1/2, 2/3, 1}


# ── Conductance noise ─────────────────────────────────────────────────────────

def add_conductance_noise(w_fp4_float: torch.Tensor,
                           sigma_rel: float,
                           p_stuck: float = 0.0,
                           p_sign_flip: float = 0.0,
                           w_scale: float = None) -> torch.Tensor:
    """
    Simulate RRAM read-back noise on an FP4-quantized weight tensor.

    Assumes w_fp4_float has already been FP4-quantized with per-channel scale
    (values ∈ FP4 code × per_channel_scale). Returns noisy float tensor.

    Physical model: conductance noise perturbs the analog readout continuously.
    No re-quantization after noise — the noisy float is used directly.
    Re-quantizing with a per-tensor scale would incorrectly destroy per-channel
    structure, causing catastrophic accuracy loss unrelated to RRAM noise.

    Stuck-at-max uses per-row max magnitude to approximate per-channel G_max.
    """
    if sigma_rel == 0 and p_stuck == 0 and p_sign_flip == 0:
        return w_fp4_float

    w = w_fp4_float.float()
    sign = torch.sign(w)
    mag = w.abs()

    # Per-row max magnitude — approximates per-channel G_max for stuck-cell model
    if w.dim() >= 2:
        row_max = mag.amax(dim=-1, keepdim=True).expand_as(mag)
    else:
        row_max = mag.amax().expand_as(mag)

    # Relative noise on magnitude conductance (multiplicative Gaussian)
    if sigma_rel > 0:
        noise = 1.0 + sigma_rel * torch.randn_like(mag)
        mag = (mag * noise).clamp(min=0.0)

    # Stuck-at-zero or stuck-at-max cells
    if p_stuck > 0:
        stuck_mask = torch.rand_like(mag) < p_stuck
        stuck_type = torch.rand_like(mag) < 0.5   # True=stuck-at-max, False=stuck-at-zero
        mag = torch.where(stuck_mask & stuck_type, row_max, mag)
        mag = torch.where(stuck_mask & ~stuck_type, torch.zeros_like(mag), mag)

    # Sign path mismatch
    if p_sign_flip > 0:
        flip_mask = torch.rand_like(mag) < p_sign_flip
        sign = torch.where(flip_mask & (sign != 0), -sign, sign)

    noisy_w = sign * mag
    return noisy_w.to(w_fp4_float.dtype)


# ── RRAM noise injection hook ─────────────────────────────────────────────────

class RRAMNoiseInjector:
    """
    Injects RRAM noise into linear layer weights at each forward pass.
    Represents: program once into RRAM → every inference reads noisy weights.
    """
    def __init__(self, model, sigma_rel: float, p_stuck: float = 0.0,
                 p_sign_flip: float = 0.0, target_names=None):
        self.sigma_rel = sigma_rel
        self.p_stuck = p_stuck
        self.p_sign_flip = p_sign_flip
        self.hooks = []

        if target_names is None:
            target_names = WEIGHT_TARGET_NAMES

        for name, module in model.named_modules():
            if not isinstance(module, torch.nn.Linear):
                continue
            if not any(t in name for t in target_names):
                continue
            h = module.register_forward_pre_hook(self._make_hook(module))
            self.hooks.append(h)

    def _make_hook(self, module):
        def hook(mod, inputs):
            w_noisy = add_conductance_noise(
                mod.weight,
                sigma_rel=self.sigma_rel,
                p_stuck=self.p_stuck,
                p_sign_flip=self.p_sign_flip,
            )
            mod.weight = torch.nn.Parameter(w_noisy, requires_grad=False)
            return inputs
        return hook

    def remove(self):
        for h in self.hooks:
            h.remove()
        self.hooks = []


# ── Conductance level analysis (for RQ3 paper figures) ────────────────────────

def conductance_level_table() -> dict:
    """
    Returns a summary dict of RRAM conductance levels for FP4 E2M1 weights.
    Useful for generating paper tables and verifying GEMM equivalence.
    """
    mag_levels = FP4_MAGNITUDES          # [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0]
    g_norm = [m / FP4_MAX for m in mag_levels]   # normalized to [0, 1]
    int_equiv = [0, 1, 2, 3, 4, 6, 8, 12]       # integer code = mag / 0.5

    spacings = [g_norm[i+1] - g_norm[i] for i in range(len(g_norm)-1)]
    ratios   = [g_norm[i+1] / g_norm[i] if g_norm[i] > 0 else float('inf')
                for i in range(len(g_norm)-1)]

    return {
        'magnitudes':    mag_levels,
        'g_normalized':  g_norm,
        'int_equiv':     int_equiv,
        'spacings':      spacings,
        'spacing_ratios': ratios,
        'note': ('Integer equivalents are {0,1,2,3,4,6,8,12} — NOT contiguous. '
                 'Gaps at {5,7,9,10,11}. Non-uniform conductance spacing required.'),
    }


def verify_gemm_equivalence(N: int = 256, K: int = 512, seed: int = 42) -> dict:
    """
    Verify that FP4 GEMM == integer-code GEMM × (0.5 × s_W) × (0.5 × s_A).
    W: (N, K) random float → quantize to FP4
    A: (K,)  random float → quantize to FP4
    Returns error statistics.
    """
    torch.manual_seed(seed)
    W_raw = torch.randn(N, K)
    A_raw = torch.randn(K)

    W_codes, s_W = quantize_to_fp4(W_raw)
    A_codes, s_A = quantize_to_fp4(A_raw)

    W_fp4 = dequantize_fp4(W_codes, s_W)
    A_fp4 = dequantize_fp4(A_codes, s_A)

    # Float GEMM on FP4-quantized values
    out_fp4 = W_fp4 @ A_fp4

    # Integer-code GEMM × scale²
    W_int = torch.tensor([FP4_INT_EQUIV[c] for c in W_codes.flatten().tolist()],
                          dtype=torch.float32).reshape(N, K)
    A_int = torch.tensor([FP4_INT_EQUIV[c] for c in A_codes.flatten().tolist()],
                          dtype=torch.float32)
    scale_sq = (0.5 * s_W) * (0.5 * s_A)
    out_int = (W_int @ A_int) * scale_sq

    err = (out_fp4 - out_int).abs()
    return {
        'max_abs_error':  err.max().item(),
        'mean_abs_error': err.mean().item(),
        'relative_error': (err / (out_fp4.abs() + 1e-8)).mean().item(),
        'scale_W': s_W,
        'scale_A': s_A,
        'scale_product_used': scale_sq,
    }
