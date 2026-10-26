"""
FP4 E2M1 core utilities.
Bit layout: [s][e1][e0][m], bias=1
Magnitude set: {0, 0.5, 1, 1.5, 2, 3, 4, 6}
"""
import torch
import numpy as np

# ── Codebook ──────────────────────────────────────────────────────────────────
# Index 0-7: positive; 8-15: negative (sign bit = 1)
FP4_VALUES = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
              -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0]
FP4_MAGNITUDES = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0]
FP4_MAX = 6.0

# Quantization thresholds (midpoints between consecutive magnitude levels)
_THRESHOLDS = torch.tensor([0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0], dtype=torch.float32)
# For each code, the integer equivalent when scale = 0.5
FP4_INT_EQUIV = [0, 1, 2, 3, 4, 6, 8, 12,    # positive codes 0-7
                  0,-1,-2,-3,-4,-6,-8,-12]    # negative codes 8-15

_CODEBOOK = torch.tensor(FP4_VALUES, dtype=torch.float32)
_MAG_TABLE = torch.tensor(FP4_MAGNITUDES, dtype=torch.float32)


# ── Quantize / Dequantize ─────────────────────────────────────────────────────

def quantize_to_fp4(tensor: torch.Tensor, scale: float = None) -> tuple:
    """
    Quantize float tensor to FP4 E2M1 codes (int64, 0-15).
    scale = amax / FP4_MAX by default (per-tensor).
    Returns (codes, scale).
    """
    t = tensor.float()
    if scale is None:
        amax = t.abs().max().item()
        scale = amax / FP4_MAX if amax > 0 else 1.0

    scaled = (t / scale).clamp(-FP4_MAX, FP4_MAX)
    sign_bit = (scaled < 0).long()
    mag = scaled.abs()

    thresholds = _THRESHOLDS.to(tensor.device)
    mag_idx = torch.bucketize(mag, thresholds)   # 0-7

    codes = sign_bit * 8 + mag_idx
    return codes, scale


def dequantize_fp4(codes: torch.Tensor, scale: float) -> torch.Tensor:
    """FP4 codes → float tensor."""
    cb = _CODEBOOK.to(codes.device)
    return cb[codes.long()] * scale


def quantize_dequantize_fp4(tensor: torch.Tensor, scale: float = None) -> tuple:
    """Round-trip quantization for simulating FP4 inference. Returns (tensor_q, scale)."""
    codes, s = quantize_to_fp4(tensor, scale)
    return dequantize_fp4(codes, s).to(tensor.dtype), s


# ── Block-scale (NVFP4 style) ─────────────────────────────────────────────────

def quantize_fp4_block(tensor: torch.Tensor, block_size: int = 16,
                        scale_fmt: str = 'e8m0') -> tuple:
    """
    Block-scaled FP4 quantization with selectable scale format.
    scale_fmt:
      'e8m0'  — nearest power-of-2 ≥ amax/6 (MXFP4 / OCP MX spec)
      'e4m3'  — FP8 E4M3 representation of amax/6 (true NVFP4)
      'fp32'  — exact float32 scale, upper-bound accuracy reference
    tensor: (..., N) where N must be divisible by block_size.
    Returns (codes: same shape, scales: (..., N//block_size)).
    """
    orig_shape = tensor.shape
    n = tensor.shape[-1]
    assert n % block_size == 0, f"last dim {n} not divisible by block_size {block_size}"

    t = tensor.float().reshape(-1, n // block_size, block_size)  # (B, G, K)
    amax = t.abs().amax(dim=-1, keepdim=True)                    # (B, G, 1)
    raw_scale = (amax / FP4_MAX).clamp(min=1e-38)               # (B, G, 1)

    if scale_fmt == 'e8m0':
        log2_scale = torch.log2(raw_scale).ceil()
        scales = (2.0 ** log2_scale).squeeze(-1)                 # (B, G)
    elif scale_fmt == 'e4m3':
        if hasattr(torch, 'float8_e4m3fn'):
            # float8_e4m3fn has no inf — values above 448 become NaN, not inf.
            # Clamp before cast to prevent NaN propagation from large-scale blocks.
            _FP8_E4M3_MAX = 448.0
            scales = raw_scale.squeeze(-1).clamp(max=_FP8_E4M3_MAX).to(torch.float8_e4m3fn).float()
        else:
            scales = raw_scale.squeeze(-1)                       # fallback to fp32
    elif scale_fmt == 'fp32':
        scales = raw_scale.squeeze(-1)
    else:
        raise ValueError(f"Unknown scale_fmt: {scale_fmt!r}. Choose 'e8m0', 'e4m3', or 'fp32'.")

    scales = scales.clamp(min=1e-38)

    codes_list = []
    for g in range(t.shape[1]):
        s = scales[:, g].unsqueeze(-1)                           # (B, 1)
        c, _ = quantize_to_fp4(t[:, g, :] / s * FP4_MAX, scale=FP4_MAX)
        codes_list.append(c)
    codes = torch.stack(codes_list, dim=1)                       # (B, G, K)
    codes = codes.reshape(orig_shape)
    scales = scales.reshape(orig_shape[:-1] + (n // block_size,))
    return codes, scales


def dequantize_fp4_block(codes: torch.Tensor, scales: torch.Tensor,
                          block_size: int = 16) -> torch.Tensor:
    orig_shape = codes.shape
    n = codes.shape[-1]
    cb = _CODEBOOK.to(codes.device)

    flat_codes = codes.reshape(-1, n // block_size, block_size)  # (B, G, K)
    flat_scales = scales.reshape(-1, n // block_size)            # (B, G)

    vals = cb[flat_codes.long()]                                  # (B, G, K)
    vals = vals * flat_scales.unsqueeze(-1)                       # scale per block
    return vals.reshape(orig_shape).to(codes.dtype if codes.is_floating_point() else torch.float32)


# ── Weight quantization hook ──────────────────────────────────────────────────

# Modules targeted for weight quantization (standard LLM linear layers)
WEIGHT_TARGET_NAMES = [
    'q_proj', 'k_proj', 'v_proj', 'o_proj',
    'gate_proj', 'up_proj', 'down_proj',
    'dense', 'fc1', 'fc2',
    # lm_head intentionally excluded: it is tied to embed_tokens in most LLMs,
    # so in-place quantization would also destroy the embedding table.
]


def apply_fp4_weights(model, target_names=None, block_size=None,
                       scale_fmt: str = 'e8m0') -> dict:
    """
    In-place: replace linear weights with FP4-quantized float equivalents.
    block_size=None  → per-output-channel FP32 scale (one scale per row).
    block_size=16    → per-block-16 scale; format controlled by scale_fmt.
    scale_fmt        → 'e8m0' (MXFP4), 'e4m3' (true NVFP4), or 'fp32' (ideal).

    lm_head is excluded because it is typically tied to embed_tokens.
    Returns {layer_name: {'scale_shape': ..., 'codes_shape': ...}}.
    """
    if target_names is None:
        target_names = WEIGHT_TARGET_NAMES

    info = {}
    for name, module in model.named_modules():
        if not isinstance(module, torch.nn.Linear):
            continue
        if not any(t in name for t in target_names):
            continue
        with torch.no_grad():
            w = module.weight.float()          # [out, in]
            if block_size is None:
                # Per-output-channel: one scale per row of the weight matrix.
                amax = w.abs().amax(dim=1, keepdim=True).clamp(min=1e-38)  # [out, 1]
                scale = amax / FP4_MAX         # [out, 1]
                codes, _ = quantize_to_fp4(w, scale=scale)
                w_q = dequantize_fp4(codes, scale).to(module.weight.dtype)
                info[name] = {'scale_shape': tuple(scale.shape),
                              'codes_shape': codes.shape}
            else:
                codes, scales = quantize_fp4_block(w, block_size, scale_fmt=scale_fmt)
                w_q = dequantize_fp4_block(codes, scales, block_size).to(module.weight.dtype)
                info[name] = {'scales_shape': scales.shape, 'codes_shape': codes.shape,
                              'scale_fmt': scale_fmt}
            module.weight.copy_(w_q)
    return info


# ── Activation quantization hooks ────────────────────────────────────────────

class FP4ActivationQuantizer:
    """
    Wraps a set of nn.Linear modules to quantize their *input* activations to FP4
    (per-token scale) before the matmul and dequantize after.
    """
    def __init__(self, model, target_names=None):
        self.hooks = []
        if target_names is None:
            target_names = WEIGHT_TARGET_NAMES
        for name, module in model.named_modules():
            if not isinstance(module, torch.nn.Linear):
                continue
            if not any(t in name for t in target_names):
                continue
            h = module.register_forward_pre_hook(self._pre_hook)
            self.hooks.append(h)

    @staticmethod
    def _pre_hook(module, inputs):
        x = inputs[0].float()
        # per-token scale: normalize each [batch, seq] position to [-6, 6], quantize, restore
        amax = x.abs().amax(dim=-1, keepdim=True).clamp(min=1e-38)
        scale = amax / FP4_MAX
        x_norm = x / scale
        x_norm_q, _ = quantize_dequantize_fp4(x_norm, scale=1.0)
        x_q = x_norm_q * scale
        return (x_q.to(inputs[0].dtype),) + inputs[1:]

    def remove(self):
        for h in self.hooks:
            h.remove()
        self.hooks = []


class INT8ActivationQuantizer:
    """Per-token INT8 activation quantization (W4A8 operating point)."""
    INT8_MAX = 127.0

    def __init__(self, model, target_names=None):
        self.hooks = []
        if target_names is None:
            target_names = WEIGHT_TARGET_NAMES
        for name, module in model.named_modules():
            if not isinstance(module, torch.nn.Linear):
                continue
            if not any(t in name for t in target_names):
                continue
            h = module.register_forward_pre_hook(self._pre_hook)
            self.hooks.append(h)

    @staticmethod
    def _pre_hook(module, inputs):
        x = inputs[0].float()
        amax = x.abs().amax(dim=-1, keepdim=True).clamp(min=1e-38)
        scale = amax / INT8ActivationQuantizer.INT8_MAX
        x_q = (x / scale).round().clamp(-INT8ActivationQuantizer.INT8_MAX,
                                         INT8ActivationQuantizer.INT8_MAX)
        return (( x_q * scale).to(inputs[0].dtype),) + inputs[1:]

    def remove(self):
        for h in self.hooks:
            h.remove()
        self.hooks = []


class MXFP4ActivationQuantizer:
    """
    Per-16-element block FP4 activation quantization with E8M0 (power-of-two)
    block scales — matching the OCP MX specification (MXFP4).

    NOTE: this uses E8M0 scales (nearest power of two), NOT E4M3 as in NVIDIA's
    proprietary NVFP4.  Call this MXFP4_W4A4_B16, not NVFP4_W4A4.
    """
    BLOCK_SIZE = 16

    def __init__(self, model, target_names=None):
        self.hooks = []
        if target_names is None:
            target_names = WEIGHT_TARGET_NAMES
        for name, module in model.named_modules():
            if not isinstance(module, torch.nn.Linear):
                continue
            if not any(t in name for t in target_names):
                continue
            h = module.register_forward_pre_hook(self._pre_hook)
            self.hooks.append(h)

    @classmethod
    def _pre_hook(cls, module, inputs):
        x = inputs[0]
        orig_shape, orig_dtype = x.shape, x.dtype
        hidden = orig_shape[-1]

        pad = (-hidden) % cls.BLOCK_SIZE
        xf = torch.nn.functional.pad(x.float(), (0, pad)) if pad else x.float()
        hidden_p = hidden + pad

        # [tokens, n_blocks, BLOCK_SIZE]
        flat = xf.reshape(-1, hidden_p // cls.BLOCK_SIZE, cls.BLOCK_SIZE)
        scale = flat.abs().amax(dim=-1, keepdim=True).clamp(min=1e-38) / FP4_MAX

        # quantize in normalised space, restore with block scale
        x_norm = flat / scale
        x_q_norm, _ = quantize_dequantize_fp4(x_norm, scale=1.0)
        x_q = (x_q_norm * scale).reshape(orig_shape[:-1] + (hidden_p,))[..., :hidden]
        return (x_q.to(orig_dtype),) + inputs[1:]

    def remove(self):
        for h in self.hooks:
            h.remove()
        self.hooks = []


# Backward-compat alias — old scripts used the NVFP4 name
NVFP4ActivationQuantizer = MXFP4ActivationQuantizer


class TrueNVFP4ActivationQuantizer:
    """
    Per-16-element block FP4 activation quantization with FP8 E4M3 block scales.
    This matches NVIDIA's published NVFP4 format: each group-of-16 gets its own
    FP8 E4M3 scale, providing significantly more scale precision than E8M0's
    power-of-two rounding (especially for blocks where optimal scale falls between
    two consecutive powers of 2).

    Hardware implication vs MXFP4 (E8M0): scale application requires an FP8→FP16
    scale multiply rather than a simple exponent shift.
    """
    BLOCK_SIZE = 16

    def __init__(self, model, target_names=None):
        self.hooks = []
        if target_names is None:
            target_names = WEIGHT_TARGET_NAMES
        for name, module in model.named_modules():
            if not isinstance(module, torch.nn.Linear):
                continue
            if not any(t in name for t in target_names):
                continue
            h = module.register_forward_pre_hook(self._pre_hook)
            self.hooks.append(h)

    @classmethod
    def _pre_hook(cls, module, inputs):
        x = inputs[0]
        orig_shape, orig_dtype = x.shape, x.dtype
        hidden = orig_shape[-1]

        pad = (-hidden) % cls.BLOCK_SIZE
        xf = torch.nn.functional.pad(x.float(), (0, pad)) if pad else x.float()
        hidden_p = hidden + pad

        flat = xf.reshape(-1, hidden_p // cls.BLOCK_SIZE, cls.BLOCK_SIZE)
        raw_scale = flat.abs().amax(dim=-1, keepdim=True).clamp(min=1e-38) / FP4_MAX

        # E4M3 scale: quantize the ideal scale to FP8 E4M3 precision.
        # float8_e4m3fn has no inf — overflow (> 448) gives NaN, so clamp first.
        if hasattr(torch, 'float8_e4m3fn'):
            _FP8_E4M3_MAX = 448.0
            scale = (raw_scale.clamp(max=_FP8_E4M3_MAX)
                              .to(torch.float8_e4m3fn).float()
                              .clamp(min=1e-38))
        else:
            scale = raw_scale  # exact scale as fallback when FP8 dtype unavailable

        x_norm = flat / scale
        x_q_norm, _ = quantize_dequantize_fp4(x_norm, scale=1.0)
        x_q = (x_q_norm * scale).reshape(orig_shape[:-1] + (hidden_p,))[..., :hidden]
        return (x_q.to(orig_dtype),) + inputs[1:]

    def remove(self):
        for h in self.hooks:
            h.remove()
        self.hooks = []


class IdealScaleActivationQuantizer:
    """
    Per-16-element block FP4 with exact FP32 block scales — upper-bound accuracy
    reference. No scale quantization error; isolates the FP4 code grid from scale format.
    """
    BLOCK_SIZE = 16

    def __init__(self, model, target_names=None):
        self.hooks = []
        if target_names is None:
            target_names = WEIGHT_TARGET_NAMES
        for name, module in model.named_modules():
            if not isinstance(module, torch.nn.Linear):
                continue
            if not any(t in name for t in target_names):
                continue
            h = module.register_forward_pre_hook(self._pre_hook)
            self.hooks.append(h)

    @classmethod
    def _pre_hook(cls, module, inputs):
        x = inputs[0]
        orig_shape, orig_dtype = x.shape, x.dtype
        hidden = orig_shape[-1]

        pad = (-hidden) % cls.BLOCK_SIZE
        xf = torch.nn.functional.pad(x.float(), (0, pad)) if pad else x.float()
        hidden_p = hidden + pad

        flat = xf.reshape(-1, hidden_p // cls.BLOCK_SIZE, cls.BLOCK_SIZE)
        scale = flat.abs().amax(dim=-1, keepdim=True).clamp(min=1e-38) / FP4_MAX

        x_norm = flat / scale
        x_q_norm, _ = quantize_dequantize_fp4(x_norm, scale=1.0)
        x_q = (x_q_norm * scale).reshape(orig_shape[:-1] + (hidden_p,))[..., :hidden]
        return (x_q.to(orig_dtype),) + inputs[1:]

    def remove(self):
        for h in self.hooks:
            h.remove()
        self.hooks = []


# ── FP8 utilities (E4M3 and E5M2 for comparison) ─────────────────────────────

FP8_E4M3_MAX = 448.0
FP8_E5M2_MAX = 57344.0


def quantize_to_fp8_e4m3(tensor: torch.Tensor, scale: float = None) -> tuple:
    """Simulate FP8 E4M3 quantization using float8_e4m3fn if available, else clamp+round."""
    if hasattr(torch, 'float8_e4m3fn'):
        t = tensor.float()
        if scale is None:
            amax = t.abs().max().item()
            scale = amax / FP8_E4M3_MAX if amax > 0 else 1.0
        t_scaled = (t / scale).clamp(-FP8_E4M3_MAX, FP8_E4M3_MAX)
        t_fp8 = t_scaled.to(torch.float8_e4m3fn).float() * scale
        return t_fp8.to(tensor.dtype), scale
    else:
        # Fallback: use float16 clamped to FP8 E4M3 range as approximation
        t = tensor.float()
        if scale is None:
            amax = t.abs().max().item()
            scale = amax / FP8_E4M3_MAX if amax > 0 else 1.0
        t_q = (t / scale).clamp(-FP8_E4M3_MAX, FP8_E4M3_MAX)
        # Approximate FP8 E4M3: 4-bit exponent, 3-bit mantissa
        # Round to nearest representable by rounding mantissa to 3 bits
        t_q16 = t_q.half().float()  # rough approximation
        return (t_q16 * scale).to(tensor.dtype), scale


# ── Utility: layer name classification ───────────────────────────────────────

def classify_layer(name: str) -> str:
    """Return 'attn', 'mlp', or 'other' based on layer name."""
    if any(x in name for x in ['q_proj', 'k_proj', 'v_proj', 'o_proj',
                                'query', 'key', 'value', 'out_proj']):
        return 'attn'
    if any(x in name for x in ['gate_proj', 'up_proj', 'down_proj', 'fc1', 'fc2',
                                'mlp', 'ffn', 'dense']):
        return 'mlp'
    return 'other'
