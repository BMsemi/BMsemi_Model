"""
Inspect JongYeop/Qwen2.5-3B-Instruct-FP4-W4A4 (NVFP4 W4A4).

Decodes directly from safetensors — no compressed-tensors library needed.

NVFP4 storage layout per linear layer:
  weight_packed        uint8  [out, in//2]       two FP4 nibbles per byte
  weight_scale         fp8_e4m3fn [out, in//16]  per-16-element group scale
  weight_global_scale  float32 [1]               per-tensor global scale
  input_global_scale   float32 [1]               activation global scale (calibrated)

Dequantization: w = fp4_val(code) × group_scale × global_scale

Outputs:
  results/nvfp4_inspect/
    layer_summary.csv         per-layer shape, scale stats, skipped flag
    weight_code_hist.csv      FP4 code frequency per layer
    global_scale_summary.csv  weight + activation global scales per layer
    nvfp4_vs_plain_fp4.csv    NVFP4 dequant vs our plain FP4 quant error
    nvfp4_inspect.png         plots
"""
import os, sys, json
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import torch
import pandas as pd
import matplotlib.pyplot as plt
import safetensors.torch as st
from collections import defaultdict
from src.fp4_utils import (FP4_VALUES, FP4_MAGNITUDES, quantize_to_fp4,
                            dequantize_fp4, FP4_MAX, classify_layer)

MODEL_PATH = os.path.expanduser(
    "~/.cache/huggingface/hub/models--JongYeop--Qwen2.5-3B-Instruct-FP4-W4A4"
    "/snapshots/c2c2fabc282fe67ee2f78b28d3046118df0b3760"
)
OUT_DIR = "results/nvfp4_inspect"
os.makedirs(OUT_DIR, exist_ok=True)

FP4_CODEBOOK = torch.tensor(FP4_VALUES, dtype=torch.float32)   # shape (16,)


# ── Decode helpers ─────────────────────────────────────────────────────────────

def unpack_nibbles(packed: torch.Tensor) -> torch.Tensor:
    """
    uint8 [out, in//2] → int64 codes [out, in]
    Low nibble = even column, high nibble = odd column.
    """
    packed = packed.to(torch.int32)
    lo = packed & 0x0F
    hi = (packed >> 4) & 0x0F
    # interleave: col 0 from lo, col 1 from hi, col 2 from lo, ...
    out = torch.stack([lo, hi], dim=-1)          # [out, in//2, 2]
    return out.reshape(packed.shape[0], -1)      # [out, in]


def dequant_nvfp4(weight_packed: torch.Tensor,
                   weight_scale: torch.Tensor,
                   weight_global_scale: torch.Tensor) -> torch.Tensor:
    """
    Decode NVFP4 weight to float32.
    weight_packed: uint8 [out, in//2]
    weight_scale:  fp8_e4m3fn [out, in//16]  (per group of 16 in-features)
    weight_global_scale: float32 scalar
    """
    codes   = unpack_nibbles(weight_packed)              # [out, in]  int32
    fp4_val = FP4_CODEBOOK[codes]                        # [out, in]  float32

    # group scale: broadcast from [out, in//16] to [out, in]
    g_scale = weight_scale.float()                       # [out, in//16]
    g_scale = g_scale.repeat_interleave(16, dim=1)      # [out, in]

    return fp4_val * g_scale * weight_global_scale.float()


# ── Load all tensors ──────────────────────────────────────────────────────────

print("Loading safetensors...")
tensors = st.load_file(os.path.join(MODEL_PATH, "model.safetensors"))

# Group by layer base name
layer_data = defaultdict(dict)
for key, val in tensors.items():
    parts = key.rsplit(".", 1)
    if len(parts) == 2:
        layer_data[parts[0]][parts[1]] = val

print(f"Total tensor keys: {len(tensors)}")
print(f"Total layer base names: {len(layer_data)}")


# ── Identify quantized vs skipped layers ──────────────────────────────────────

quantized_layers = {}
skipped_layers   = {}

for base, fields in layer_data.items():
    if "weight_packed" in fields:
        quantized_layers[base] = fields
    elif "weight" in fields:
        skipped_layers[base] = fields

print(f"\nQuantized layers: {len(quantized_layers)}")
print(f"Unquantized layers: {len(skipped_layers)}")
print(f"Skipped: {list(skipped_layers.keys())}")


# ── Per-layer analysis ────────────────────────────────────────────────────────

layer_rows    = []
code_hist_rows = []
compare_rows  = []

for base, fields in quantized_layers.items():
    wp  = fields["weight_packed"]            # uint8 [out, in//2]
    ws  = fields["weight_scale"]             # fp8_e4m3fn [out, in//16]
    wgs = fields["weight_global_scale"]      # float32 [1]
    igs = fields.get("input_global_scale", torch.tensor([float('nan')]))

    out_f, in_half = wp.shape
    in_f = in_half * 2

    # Decode
    codes   = unpack_nibbles(wp)             # [out_f, in_f]
    w_nvfp4 = dequant_nvfp4(wp, ws, wgs)    # [out_f, in_f] float32

    # Code histogram
    code_counts = torch.bincount(codes.flatten(), minlength=16).float()
    code_freq   = code_counts / code_counts.sum()

    # Group scale statistics
    gs_flat = ws.float().flatten()
    gs_log2 = torch.log2(gs_flat.abs().clamp(min=1e-38))

    # Our plain FP4 quantization (per-tensor) on the same raw weight
    # We don't have the original BF16 weight, so compare NVFP4 dequant vs
    # re-quantizing it with plain per-tensor scale
    codes_plain, scale_plain = quantize_to_fp4(w_nvfp4)
    w_plain = dequant_nvfp4_plain = dequantize_fp4(codes_plain, scale_plain)

    err = (w_nvfp4 - w_plain).abs()
    rel_err = err / (w_nvfp4.abs().clamp(min=1e-6))

    layer_type = classify_layer(base)

    layer_rows.append({
        "layer":               base,
        "layer_type":          layer_type,
        "out_features":        out_f,
        "in_features":         in_f,
        "weight_global_scale": wgs.item(),
        "input_global_scale":  igs.item(),
        "group_scale_mean":    gs_flat.mean().item(),
        "group_scale_std":     gs_flat.std().item(),
        "group_scale_log2_mean": gs_log2.mean().item(),
        "group_scale_log2_std":  gs_log2.std().item(),
        "w_max_abs":           w_nvfp4.abs().max().item(),
        "w_std":               w_nvfp4.std().item(),
        "zero_fraction":       (codes % 8 == 0).float().mean().item(),
        "nvfp4_vs_plain_max_err":  err.max().item(),
        "nvfp4_vs_plain_rel_err":  rel_err.mean().item(),
    })

    for code_idx in range(16):
        code_hist_rows.append({
            "layer":      base,
            "layer_type": layer_type,
            "code":       code_idx,
            "value":      FP4_VALUES[code_idx],
            "frequency":  code_freq[code_idx].item(),
        })

print(f"\nAnalysed {len(layer_rows)} quantized layers.")


# ── Aggregate code histogram ──────────────────────────────────────────────────

df_layers  = pd.DataFrame(layer_rows)
df_codes   = pd.DataFrame(code_hist_rows)

# Global code histogram across all layers
global_freq = df_codes.groupby("code")["frequency"].mean()
mag_freq = pd.Series(
    [global_freq[i] + global_freq[i+8] for i in range(8)],
    index=[str(m) for m in FP4_MAGNITUDES]
)

print("\n=== Global FP4 Code Frequencies (NVFP4 weights) ===")
print(f"{'Code':>5}  {'Value':>6}  {'Frequency':>10}")
for code in range(16):
    print(f"  {code:>3}  {FP4_VALUES[code]:>6.2f}  {global_freq[code]:.6f}")

print("\n=== Magnitude-level aggregated ===")
for mag, freq in zip(FP4_MAGNITUDES, mag_freq.values):
    bar = "█" * int(freq * 200)
    print(f"  G={mag:.1f}:  {freq:.6f}  {bar}")

print("\n=== Layer summary (first 10) ===")
print(df_layers[["layer", "layer_type", "out_features", "in_features",
                  "weight_global_scale", "zero_fraction",
                  "nvfp4_vs_plain_rel_err"]].head(10).to_string(index=False))

print("\n=== Group scale log2 stats by layer type ===")
print(df_layers.groupby("layer_type")[
    ["group_scale_log2_mean", "group_scale_log2_std",
     "weight_global_scale", "nvfp4_vs_plain_rel_err"]
].agg(["mean", "std"]).to_string())


# ── Adaptive sensing policy from NVFP4 ───────────────────────────────────────

print("\n=== Adaptive Sensing Policy (from NVFP4 calibrated weights) ===")
print("Magnitude level → mean frequency → relative sensing time")
total = mag_freq.sum()
for mag, freq in zip(FP4_MAGNITUDES, mag_freq.values):
    rel_time = freq / total
    bar = "▓" * int(rel_time * 100)
    print(f"  G={mag:.1f} (G/Gmax={mag/6:.4f}):  freq={freq:.4f}  "
          f"sensing_time={rel_time:.4f}  {bar}")


# ── Save CSVs ─────────────────────────────────────────────────────────────────

df_layers.to_csv(os.path.join(OUT_DIR, "layer_summary.csv"), index=False)
df_codes.to_csv(os.path.join(OUT_DIR, "weight_code_hist.csv"), index=False)
print(f"\nCSVs saved to {OUT_DIR}/")


# ── Plots ─────────────────────────────────────────────────────────────────────

fig, axes = plt.subplots(2, 3, figsize=(16, 9))

# (a) Global code histogram
ax = axes[0, 0]
ax.bar(range(16), [global_freq[i] for i in range(16)],
       color=["steelblue"]*8 + ["tomato"]*8, alpha=0.8)
ax.set_xticks(range(16))
ax.set_xticklabels([f"{FP4_VALUES[i]:.1f}" for i in range(16)], rotation=90, fontsize=7)
ax.set_xlabel("FP4 code value")
ax.set_ylabel("Mean frequency")
ax.set_title("(a) Global FP4 Code Histogram\n(blue=positive, red=negative)")

# (b) Magnitude-level frequency (collapsed ±)
ax = axes[0, 1]
ax.bar([str(m) for m in FP4_MAGNITUDES], mag_freq.values, color="steelblue", alpha=0.8)
ax.set_xlabel("FP4 magnitude level")
ax.set_ylabel("Mean frequency (|pos| + |neg|)")
ax.set_title("(b) Magnitude-Level Usage\n(NVFP4 calibrated weights)")

# (c) Group scale log2 distribution per layer type
ax = axes[0, 2]
for ltype, color in [("attn", "steelblue"), ("mlp", "darkorange")]:
    sub = df_layers[df_layers["layer_type"] == ltype]["group_scale_log2_mean"]
    ax.hist(sub, bins=30, alpha=0.6, label=ltype, color=color)
ax.set_xlabel("mean log₂(group scale) per layer")
ax.set_ylabel("Count")
ax.set_title("(c) NVFP4 Group Scale Distribution\nby layer type")
ax.legend()

# (d) Weight global scale per layer
ax = axes[1, 0]
attn_gs = df_layers[df_layers["layer_type"]=="attn"]["weight_global_scale"].values
mlp_gs  = df_layers[df_layers["layer_type"]=="mlp"]["weight_global_scale"].values
ax.scatter(range(len(attn_gs)), attn_gs, s=15, label="attn", alpha=0.7)
ax.scatter(range(len(mlp_gs)),  mlp_gs,  s=15, label="mlp",  alpha=0.7)
ax.set_xlabel("Layer index (within type)")
ax.set_ylabel("Weight global scale")
ax.set_title("(d) Weight Global Scale per Layer")
ax.legend()

# (e) Zero fraction (fraction of weights quantised to 0)
ax = axes[1, 1]
colors = ["steelblue" if t=="attn" else "darkorange" for t in df_layers["layer_type"]]
ax.bar(range(len(df_layers)), df_layers["zero_fraction"], color=colors, alpha=0.8)
ax.set_xlabel("Layer index")
ax.set_ylabel("Fraction of weights = 0")
ax.set_title("(e) Zero-Weight Fraction per Layer")
from matplotlib.patches import Patch
ax.legend(handles=[Patch(color="steelblue", label="attn"),
                   Patch(color="darkorange", label="mlp")])

# (f) NVFP4 vs plain FP4 relative error per layer
ax = axes[1, 2]
ax.bar(range(len(df_layers)), df_layers["nvfp4_vs_plain_rel_err"],
       color=colors, alpha=0.8)
ax.set_xlabel("Layer index")
ax.set_ylabel("Mean relative error")
ax.set_title("(f) NVFP4 vs Plain FP4 Mismatch\n(per-tensor rescale of same codes)")
ax.legend(handles=[Patch(color="steelblue", label="attn"),
                   Patch(color="darkorange", label="mlp")])

plt.suptitle("NVFP4 Checkpoint Inspection — Qwen2.5-3B-Instruct-FP4-W4A4", fontsize=13)
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "nvfp4_inspect.png"), dpi=200)
print(f"Plot saved: {OUT_DIR}/nvfp4_inspect.png")
