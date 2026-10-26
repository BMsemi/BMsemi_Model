"""
RQ7 + RQ8 — Distribution-Aware Sensing & Extension to NVFP4/FP8

RQ7: Can FP4 code distributions guide adaptive sensing/precharge policies?
     (Also analysed in FP8/BF16/FP16 for richer distribution information)
RQ8: Once plain FP4 is understood, how do block scaling and FP8 scale
     factors change the hardware requirements?

This script:
  1. Captures weight + activation tensors from all linear layers
  2. Represents each in FP4, FP8-E4M3, FP8-E5M2, BF16, FP16
  3. Computes per-format code utilisation, entropy, dynamic range
  4. Designs adaptive sensing policy from FP4 code frequencies
  5. RQ8: analyses NVFP4 block-scale distributions + FP8 scale statistics

Outputs: results/rq7_rq8/
"""
import os, sys, gc, json
os.environ["CUDA_VISIBLE_DEVICES"] = "4"              # must be before torch import
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
sys.path.insert(0, os.path.dirname(__file__))

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import defaultdict
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from tqdm import tqdm
from src.fp4_utils import (quantize_to_fp4, quantize_fp4_block,
                            FP4_MAGNITUDES, FP4_VALUES, FP4_MAX,
                            classify_layer, WEIGHT_TARGET_NAMES)

# ── Configuration ─────────────────────────────────────────────────────────────

MODEL_NAME  = "Qwen/Qwen2.5-1.5B"    # base model — instruct models give ~100K PPL on raw wikitext
SEQ_LEN     = 256
MAX_SAMPLES = 16       # fewer needed for distribution capture
SEED        = 42
NVFP4_BLOCK = 16       # NVFP4-style block size
OUT_DIR     = "results/rq7_rq8"
os.makedirs(OUT_DIR, exist_ok=True)


# ── Quantisation helpers ──────────────────────────────────────────────────────

def fp8_e4m3_values(tensor: torch.Tensor) -> torch.Tensor:
    """Approximate FP8 E4M3 (8-bit float, bias=7) via torch.float8_e4m3fn if available."""
    t = tensor.float()
    if hasattr(torch, 'float8_e4m3fn'):
        amax = t.abs().max().item()
        scale = amax / 448.0 if amax > 0 else 1.0
        return (t / scale).clamp(-448, 448).to(torch.float8_e4m3fn).float() * scale
    # Fallback: quantise to 8-bit using rounding to 128 uniform levels in log-space
    amax = t.abs().max().item()
    if amax == 0:
        return t.clone()
    scale = amax / 448.0
    # 4-bit exponent, 3-bit mantissa → 128 positive levels + sign
    # Simulate via float16 in reduced range as approximation
    t_q = (t / scale).half().float() * scale
    return t_q


def fp8_e5m2_values(tensor: torch.Tensor) -> torch.Tensor:
    t = tensor.float()
    if hasattr(torch, 'float8_e5m2'):
        amax = t.abs().max().item()
        scale = amax / 57344.0 if amax > 0 else 1.0
        return (t / scale).clamp(-57344, 57344).to(torch.float8_e5m2).float() * scale
    amax = t.abs().max().item()
    if amax == 0:
        return t.clone()
    scale = amax / 57344.0
    t_q = (t / scale).half().float() * scale
    return t_q


# ── Distribution analyser ─────────────────────────────────────────────────────

def analyse_tensor(t: torch.Tensor, name: str, tensor_type: str) -> dict:
    """
    Compute distribution statistics for a single tensor in multiple formats.
    tensor_type: 'weight' or 'activation'
    """
    t_flat = t.float().flatten()
    n      = t_flat.numel()

    result = {
        'name': name,
        'type': tensor_type,
        'layer_type': classify_layer(name),
        'numel': n,
    }

    # ── FP4 ───────────────────────────────────────────────────────────────────
    codes_fp4, scale_fp4 = quantize_to_fp4(t_flat)
    codes_fp4_np = codes_fp4.cpu().numpy()
    fp4_hist     = np.bincount(codes_fp4_np, minlength=16) / n

    # Separate positive magnitude codes (0-7) for sensing policy
    mag_hist = fp4_hist[:8] + fp4_hist[8:]   # combine +/- for same magnitude level
    # Entropy of FP4 code distribution
    p_fp4     = fp4_hist[fp4_hist > 0]
    entropy_fp4 = -np.sum(p_fp4 * np.log2(p_fp4))
    # Utilisation: fraction of the 16 codes actually used
    n_used_fp4  = (fp4_hist > 0).sum()

    result.update({
        'fp4_scale': scale_fp4,
        'fp4_code_hist': fp4_hist.tolist(),
        'fp4_mag_hist': mag_hist.tolist(),    # per-magnitude-level usage
        'fp4_entropy': float(entropy_fp4),
        'fp4_codes_used': int(n_used_fp4),
        'fp4_max_code_freq': float(fp4_hist.max()),
        'fp4_zero_fraction': float(fp4_hist[0] + fp4_hist[8]),  # codes for 0 magnitude
    })

    # ── FP8 E4M3 ─────────────────────────────────────────────────────────────
    t_fp8e4m3 = fp8_e4m3_values(t_flat)
    fp8_amax  = t_fp8e4m3.abs().max().item()
    fp8_amin  = t_fp8e4m3[t_fp8e4m3 != 0].abs().min().item() if (t_fp8e4m3 != 0).any() else fp8_amax
    result.update({
        'fp8_e4m3_max':  fp8_amax,
        'fp8_e4m3_min_nonzero': fp8_amin,
        'fp8_e4m3_dr':  fp8_amax / fp8_amin if fp8_amin > 0 else 1.0,
        'fp8_e4m3_std':  float(t_fp8e4m3.std()),
    })

    # ── BF16 ─────────────────────────────────────────────────────────────────
    t_bf16 = t_flat.bfloat16().float()
    result.update({
        'bf16_max': float(t_bf16.abs().max()),
        'bf16_std': float(t_bf16.std()),
    })

    # ── FP16 ─────────────────────────────────────────────────────────────────
    t_fp16 = t_flat.half().float()
    result.update({
        'fp16_max': float(t_fp16.abs().max()),
        'fp16_std': float(t_fp16.std()),
    })

    # ── Distribution shape ───────────────────────────────────────────────────
    result.update({
        'raw_max':  float(t_flat.abs().max()),
        'raw_std':  float(t_flat.std()),
        'raw_mean': float(t_flat.mean()),
        'raw_kurtosis': float(((t_flat - t_flat.mean()) ** 4).mean() / (t_flat.std() ** 4 + 1e-8)),
    })
    return result


# ── NVFP4 block-scale analyser (RQ8) ─────────────────────────────────────────

def analyse_nvfp4(t: torch.Tensor, name: str) -> dict:
    """Analyse block-scale distribution for NVFP4."""
    t_2d = t.float().reshape(-1, t.shape[-1] if t.dim() > 1 else t.numel())
    n_elem = t_2d.shape[-1]

    if n_elem < NVFP4_BLOCK:
        return {}

    codes, scales = quantize_fp4_block(t_2d, block_size=NVFP4_BLOCK)
    scales_flat   = scales.float().flatten()

    log2_scales = torch.log2(scales_flat.abs().clamp(min=1e-38))
    result = {
        'name': name,
        'nvfp4_block_size': NVFP4_BLOCK,
        'scale_log2_mean': float(log2_scales.mean()),
        'scale_log2_std':  float(log2_scales.std()),
        'scale_log2_min':  float(log2_scales.min()),
        'scale_log2_max':  float(log2_scales.max()),
        'n_blocks': int(scales_flat.numel()),
        'codes_shape': list(codes.shape),
    }

    # Code histogram under block scaling
    codes_flat = codes.long().flatten().cpu().numpy()
    hist = np.bincount(codes_flat, minlength=16) / len(codes_flat)
    result['nvfp4_code_hist'] = hist.tolist()
    result['nvfp4_entropy']   = float(-np.sum(p * np.log2(p) for p in hist if p > 0))
    return result


# ── Adaptive sensing policy ───────────────────────────────────────────────────

def compute_sensing_policy(all_stats: list) -> dict:
    """
    Aggregate FP4 magnitude-level frequencies across all layers and tensor types.
    Adaptive sensing: allocate sensing time proportional to code frequency.
    """
    # Aggregate magnitude histogram across all weight tensors
    agg_weight_hist = np.zeros(8)
    agg_act_hist    = np.zeros(8)
    n_weight, n_act = 0, 0

    for st in all_stats:
        if 'fp4_mag_hist' not in st:
            continue
        h = np.array(st['fp4_mag_hist'])
        if st['type'] == 'weight':
            agg_weight_hist += h * st['numel']
            n_weight        += st['numel']
        else:
            agg_act_hist    += h * st['numel']
            n_act           += st['numel']

    if n_weight > 0: agg_weight_hist /= n_weight
    if n_act    > 0: agg_act_hist    /= n_act

    # Sensing time proportional to frequency (more frequent → more sensing cycles)
    w_time = agg_weight_hist / agg_weight_hist.sum() if agg_weight_hist.sum() > 0 else agg_weight_hist
    a_time = agg_act_hist    / agg_act_hist.sum()    if agg_act_hist.sum() > 0    else agg_act_hist

    policy = {
        'magnitude_levels': FP4_MAGNITUDES,
        'weight': {
            'frequency':      agg_weight_hist.tolist(),
            'sensing_time_norm': w_time.tolist(),
            'rank_by_frequency': np.argsort(-agg_weight_hist).tolist(),
        },
        'activation': {
            'frequency':      agg_act_hist.tolist(),
            'sensing_time_norm': a_time.tolist(),
            'rank_by_frequency': np.argsort(-agg_act_hist).tolist(),
        },
        'note': ('Sensing time normalised to [0,1] per level. '
                 'Level 0 (G=0) is rarely used in well-trained models '
                 'and can tolerate longer settle time.'),
    }
    return policy


# ── Capture activations via hooks ─────────────────────────────────────────────

class ActivationCapture:
    def __init__(self, model, max_tokens: int = 4096):
        self.max_tokens = max_tokens
        self.buffers    = defaultdict(list)
        self.hooks      = []
        for name, module in model.named_modules():
            if not isinstance(module, nn.Linear):
                continue
            if not any(t in name for t in WEIGHT_TARGET_NAMES):
                continue
            h = module.register_forward_pre_hook(self._make_hook(name))
            self.hooks.append(h)

    def _make_hook(self, name):
        def hook(module, inputs):
            buf = self.buffers[name]
            total = sum(b.numel() for b in buf)
            if total >= self.max_tokens * module.in_features:
                return
            buf.append(inputs[0].detach().float().reshape(-1, inputs[0].shape[-1]).cpu())
        return hook

    def get_tensor(self, name) -> torch.Tensor:
        bufs = self.buffers.get(name, [])
        if not bufs:
            return torch.empty(0)
        return torch.cat(bufs, dim=0)

    def remove(self):
        for h in self.hooks:
            h.remove()
        self.hooks = []


# ── Plotting ──────────────────────────────────────────────────────────────────

def plot_sensing_policy(policy: dict):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    mags   = [str(m) for m in policy['magnitude_levels']]
    w_freq = policy['weight']['frequency']
    a_freq = policy['activation']['frequency']

    for ax, freq, title in [
        (axes[0], w_freq,  "Weight Conductance Level Usage"),
        (axes[1], a_freq,  "Activation Conductance Level Usage"),
    ]:
        bars = ax.bar(mags, freq)
        ax.set_xlabel("FP4 Magnitude Level")
        ax.set_ylabel("Normalised Frequency")
        ax.set_title(title)
        # Annotate relative sensing time
        for bar, f in zip(bars, freq):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.002,
                    f'{f:.3f}', ha='center', va='bottom', fontsize=7)

    plt.suptitle("RQ7: Adaptive Sensing Policy — FP4 Code Frequencies")
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "rq7_sensing_policy.png"), dpi=200)
    print("  Saved: rq7_sensing_policy.png")


def plot_entropy_comparison(all_stats: list):
    """Compare FP4 entropy across layers and tensor types."""
    df = pd.DataFrame(all_stats)
    if 'fp4_entropy' not in df.columns:
        return

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for ax, tensor_type in [(axes[0], 'weight'), (axes[1], 'activation')]:
        sub = df[df['type'] == tensor_type].reset_index(drop=True)
        colors = {'attn': 'steelblue', 'mlp': 'darkorange', 'other': 'gray'}
        bar_colors = [colors.get(t, 'gray') for t in sub['layer_type']]
        ax.bar(range(len(sub)), sub['fp4_entropy'], color=bar_colors, alpha=0.8)
        ax.axhline(4.0, color='red', linestyle='--', label='Max entropy (4 bits)')
        ax.set_xticks(range(len(sub)))
        ax.set_xticklabels([n.split('.')[-1] for n in sub['name']], rotation=90, fontsize=6)
        ax.set_ylabel("FP4 Code Entropy (bits)")
        ax.set_title(f"{tensor_type.capitalize()} Distribution Entropy")
        from matplotlib.patches import Patch
        legend_handles = [Patch(color=c, label=t) for t, c in colors.items()]
        ax.legend(handles=legend_handles + [plt.Line2D([0],[0],color='red',linestyle='--',label='Max entropy')],
                  fontsize=8)

    plt.suptitle("RQ7: FP4 Code Distribution Entropy (higher = more uniform = less adaptive gain)")
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "rq7_fp4_entropy.png"), dpi=200)
    print("  Saved: rq7_fp4_entropy.png")


def plot_rq8_scales(nvfp4_stats: list):
    """RQ8: NVFP4 block scale distributions."""
    if not nvfp4_stats:
        return
    df = pd.DataFrame(nvfp4_stats)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    axes[0].scatter(range(len(df)), df['scale_log2_mean'], s=20, label='mean')
    axes[0].fill_between(range(len(df)),
                          df['scale_log2_mean'] - df['scale_log2_std'],
                          df['scale_log2_mean'] + df['scale_log2_std'],
                          alpha=0.3, label='±1σ')
    axes[0].set_xlabel("Layer index")
    axes[0].set_ylabel("log₂(E8M0 scale)")
    axes[0].set_title("NVFP4 Block Scale per Layer")
    axes[0].legend()

    axes[1].hist(df['scale_log2_mean'], bins=30, alpha=0.7, label='log₂ scale (mean)')
    axes[1].set_xlabel("log₂(block scale)")
    axes[1].set_ylabel("Count (layers)")
    axes[1].set_title("Distribution of NVFP4 Block Scales")

    plt.suptitle("RQ8: NVFP4 Block Scale Statistics")
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "rq8_nvfp4_scales.png"), dpi=200)
    print("  Saved: rq8_nvfp4_scales.png")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    torch.manual_seed(SEED)
    print(f"Loading {MODEL_NAME}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model     = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, dtype=torch.bfloat16, device_map="auto"
    )
    model.eval()

    all_stats    = []
    nvfp4_stats  = []

    # ── Analyse weights (static) ──────────────────────────────────────────────
    print("\n=== Weight distribution analysis ===")
    for name, module in tqdm(list(model.named_modules()), desc="Weights"):
        if not isinstance(module, nn.Linear):
            continue
        if not any(t in name for t in WEIGHT_TARGET_NAMES):
            continue
        w = module.weight.detach().float()
        st = analyse_tensor(w, name, 'weight')
        all_stats.append(st)

        # NVFP4 block-scale analysis
        nv = analyse_nvfp4(w, name)
        if nv:
            nvfp4_stats.append(nv)

    # ── Capture activations ───────────────────────────────────────────────────
    print("\n=== Activation distribution capture ===")
    act_capture = ActivationCapture(model, max_tokens=MAX_SAMPLES * SEQ_LEN)

    ds   = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    text = "\n\n".join(ds["text"])
    text = text[:MAX_SAMPLES * SEQ_LEN * 6]   # pre-truncate to avoid tokenizer length warning
    enc  = tokenizer(text, return_tensors="pt")
    ids  = enc.input_ids[:, :MAX_SAMPLES * SEQ_LEN]
    device = next(model.parameters()).device

    with torch.no_grad():
        for i in range(MAX_SAMPLES):
            s, e = i * SEQ_LEN, (i + 1) * SEQ_LEN
            if e > ids.size(1):
                break
            model(ids[:, s:e].to(device))

    act_capture.remove()

    # Analyse captured activations
    for name, module in model.named_modules():
        if not isinstance(module, nn.Linear):
            continue
        if not any(t in name for t in WEIGHT_TARGET_NAMES):
            continue
        a = act_capture.get_tensor(name)
        if a.numel() == 0:
            continue
        st = analyse_tensor(a, name, 'activation')
        all_stats.append(st)

    del model
    gc.collect()
    torch.cuda.empty_cache()

    # ── Compute sensing policy ────────────────────────────────────────────────
    print("\n=== Adaptive Sensing Policy ===")
    policy = compute_sensing_policy(all_stats)
    print("Weight magnitude level frequencies:")
    for i, (mag, freq) in enumerate(zip(policy['magnitude_levels'], policy['weight']['frequency'])):
        rank = policy['weight']['rank_by_frequency'].index(i) + 1
        print(f"  G={mag:.1f}: {freq:.4f}  (rank {rank})")
    print(f"\n{policy['note']}")

    # ── Save results ──────────────────────────────────────────────────────────
    df_all = pd.DataFrame(all_stats)
    # Drop large histogram lists for CSV (keep in JSON)
    df_csv = df_all.drop(columns=['fp4_code_hist', 'fp4_mag_hist', 'nvfp4_code_hist'],
                          errors='ignore')
    df_csv.to_csv(os.path.join(OUT_DIR, "rq7_all_stats.csv"), index=False)

    # JSON with full data
    def safe(x):
        if isinstance(x, (np.integer,)): return int(x)
        if isinstance(x, (np.floating,)): return float(x)
        if isinstance(x, (np.ndarray,)): return x.tolist()
        return x

    with open(os.path.join(OUT_DIR, "rq7_rq8_results.json"), "w") as f:
        json.dump({'all_stats': all_stats, 'nvfp4_stats': nvfp4_stats,
                   'sensing_policy': policy}, f, indent=2, default=safe)

    # ── Plots ─────────────────────────────────────────────────────────────────
    print("\n=== Generating plots ===")
    plot_sensing_policy(policy)
    plot_entropy_comparison(all_stats)
    plot_rq8_scales(nvfp4_stats)

    # ── Summary by format & layer type ───────────────────────────────────────
    print("\n=== Summary ===")
    weight_stats = [s for s in all_stats if s['type'] == 'weight']
    print(f"Total weight tensors analysed: {len(weight_stats)}")
    print(f"Total activation tensors analysed: {sum(1 for s in all_stats if s['type']=='activation')}")

    # Average FP4 entropy by layer type
    for ltype in ['attn', 'mlp']:
        subset = [s for s in weight_stats if s.get('layer_type') == ltype]
        if subset:
            avg_ent = np.mean([s['fp4_entropy'] for s in subset])
            avg_zero = np.mean([s['fp4_zero_fraction'] for s in subset])
            print(f"  {ltype}: avg FP4 entropy={avg_ent:.3f} bits, zero_fraction={avg_zero:.3f}")

    print(f"\nAll results saved to {OUT_DIR}/")


if __name__ == "__main__":
    main()
