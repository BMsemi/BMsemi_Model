"""
RQ11 — Block-Scale Format Comparison: E8M0 vs E4M3 vs FP32 (ideal)

Question: how much does the scale format (E8M0 power-of-2 vs FP8 E4M3 vs ideal FP32)
affect model accuracy for block-16 FP4 (W4A4)?

Hardware context:
  E8M0  — exponent-only shift, cheapest hardware (MXFP4 / OCP MX spec)
  E4M3  — FP8 mantissa+exponent scale, more expressive; matches true NVFP4 as shipped
           by NVIDIA (e.g. in Blackwell NVLink/SHARP and HuggingFace FP4 checkpoints)
  FP32  — exact scale, no quantisation error; upper-bound accuracy reference

For each variant:
  - Weights: block-16 FP4 with the given scale format
  - Activations: block-16 FP4 with the given scale format
  - Both combined: W4A4 with matching scale format

Operating points evaluated:
  BF16        — floating-point baseline
  W4A16-ch    — FP4 weights (per-channel), BF16 activations [held constant]
  MXFP4-W4A4-B16  — E8M0 weights + E8M0 activations
  NVFP4-W4A4-B16  — E4M3 weights + E4M3 activations  (true NVFP4)
  Ideal-W4A4  — FP32 weights + FP32 activations   (upper bound)

Outputs: results/rq11/
"""
import os, sys, gc
os.environ["CUDA_VISIBLE_DEVICES"] = "4"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
import torch.nn as nn
import pandas as pd
import matplotlib.pyplot as plt
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from tqdm import tqdm

from src.fp4_utils import (
    apply_fp4_weights,
    MXFP4ActivationQuantizer,      # E8M0 block-16
    TrueNVFP4ActivationQuantizer,  # E4M3 block-16
    IdealScaleActivationQuantizer, # FP32 block-16
)

# ── Configuration ──────────────────────────────────────────────────────────────

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_NAME  = "Qwen/Qwen2.5-1.5B"
SEQ_LEN     = 512
MAX_SAMPLES = 32
SEED        = 42
BLOCK_SIZE  = 16
OUT_DIR     = os.path.join(_SCRIPT_DIR, "results", "rq11")
os.makedirs(OUT_DIR, exist_ok=True)

# Hardware cost summary (qualitative, for paper table)
HARDWARE_COST = {
    'BF16':        'baseline FP ops',
    'W4A16-ch':    'FP4 weight decode only',
    'MXFP4-W4A4-B16':  'shift-only scale (barrel shifter)',
    'NVFP4-W4A4-B16':  'FP8→FP16 scale multiply',
    'Ideal-W4A4':  'FP32 scale multiply (reference only)',
}


# ── PPL evaluator ─────────────────────────────────────────────────────────────

def evaluate_ppl(model, tokenizer) -> float:
    ds   = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    text = "\n\n".join(ds["text"])
    text = text[:MAX_SAMPLES * SEQ_LEN * 6]
    enc  = tokenizer(text, add_special_tokens=False, return_tensors="pt")
    ids  = enc.input_ids[:, :MAX_SAMPLES * SEQ_LEN]
    device = next(model.parameters()).device
    ids  = ids.to(device)
    nlls = []
    with torch.no_grad():
        for i in range(MAX_SAMPLES):
            s, e = i * SEQ_LEN, (i + 1) * SEQ_LEN
            if e > ids.size(1):
                break
            chunk = ids[:, s:e]
            out   = model(chunk, labels=chunk.clone())
            nlls.append(out.loss.item())
    return torch.exp(torch.tensor(nlls).mean()).item()


# ── Scale error analysis ───────────────────────────────────────────────────────

def scale_error_analysis(model) -> dict:
    """
    For each linear weight matrix, compute the relative error introduced by
    E8M0 and E4M3 scale quantization compared to ideal FP32 scale.
    Returns aggregate statistics.
    """
    from src.fp4_utils import FP4_MAX, quantize_fp4_block
    e8m0_errs, e4m3_errs = [], []

    for name, module in model.named_modules():
        if not isinstance(module, nn.Linear):
            continue
        w = module.weight.float()
        n = w.shape[-1]
        if n % BLOCK_SIZE != 0:
            continue

        # Reshape to blocks
        t = w.reshape(-1, n // BLOCK_SIZE, BLOCK_SIZE)
        amax = t.abs().amax(dim=-1, keepdim=True).clamp(min=1e-38)
        ideal_scale = (amax / FP4_MAX).squeeze(-1)  # (B, G)

        # E8M0 scale
        import torch, numpy as np
        log2s = torch.log2(ideal_scale).ceil()
        e8m0_scale = (2.0 ** log2s)
        e8m0_err = ((e8m0_scale - ideal_scale).abs() / ideal_scale).mean().item()
        e8m0_errs.append(e8m0_err)

        # E4M3 scale
        if hasattr(torch, 'float8_e4m3fn'):
            e4m3_scale = ideal_scale.to(torch.float8_e4m3fn).float().clamp(min=1e-38)
            e4m3_err = ((e4m3_scale - ideal_scale).abs() / ideal_scale).mean().item()
        else:
            e4m3_err = 0.0  # no quantisation if dtype unavailable
        e4m3_errs.append(e4m3_err)

    import numpy as np
    return {
        'e8m0_scale_err_mean': float(np.mean(e8m0_errs)),
        'e8m0_scale_err_max':  float(np.max(e8m0_errs)),
        'e4m3_scale_err_mean': float(np.mean(e4m3_errs)),
        'e4m3_scale_err_max':  float(np.max(e4m3_errs)),
        'n_layers': len(e8m0_errs),
    }


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    torch.manual_seed(SEED)
    results = []

    def load_fresh():
        m = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME, dtype=torch.bfloat16, device_map="auto"
        )
        tok = AutoTokenizer.from_pretrained(MODEL_NAME)
        m.eval()
        return m, tok

    # ── BF16 baseline ────────────────────────────────────────────────────────
    print("\n[1/5] BF16 baseline")
    model, tokenizer = load_fresh()
    ppl_bf16 = evaluate_ppl(model, tokenizer)
    results.append({'variant': 'BF16', 'w_scale': '-', 'a_scale': '-',
                    'ppl': ppl_bf16, 'hardware_cost': HARDWARE_COST['BF16']})
    print(f"  PPL = {ppl_bf16:.3f}")
    del model; gc.collect(); torch.cuda.empty_cache()

    # ── W4A16 per-channel (weight only, fixed reference) ─────────────────────
    print("\n[2/5] W4A16-ch  (per-channel FP4 weights, BF16 activations)")
    model, tokenizer = load_fresh()
    apply_fp4_weights(model, block_size=None)   # per-channel
    ppl_w4a16 = evaluate_ppl(model, tokenizer)
    results.append({'variant': 'W4A16-ch', 'w_scale': 'per-channel FP32', 'a_scale': 'BF16',
                    'ppl': ppl_w4a16, 'hardware_cost': HARDWARE_COST['W4A16-ch']})
    print(f"  PPL = {ppl_w4a16:.3f}  ΔPPL = {ppl_w4a16 - ppl_bf16:+.3f}")
    del model; gc.collect(); torch.cuda.empty_cache()

    # ── MXFP4-W4A4-B16 — E8M0 (current MXFP4 / OCP MX) ──────────────────────────
    print("\n[3/5] MXFP4-W4A4-B16  (E8M0 block-16 weights + E8M0 activations)")
    model, tokenizer = load_fresh()
    scale_errs = scale_error_analysis(model)
    print(f"  Scale error analysis:")
    print(f"    E8M0: mean={scale_errs['e8m0_scale_err_mean']:.4f}  "
          f"max={scale_errs['e8m0_scale_err_max']:.4f}")
    print(f"    E4M3: mean={scale_errs['e4m3_scale_err_mean']:.6f}  "
          f"max={scale_errs['e4m3_scale_err_max']:.6f}")
    apply_fp4_weights(model, block_size=BLOCK_SIZE, scale_fmt='e8m0')
    act_q = MXFP4ActivationQuantizer(model)
    ppl_mxfp4 = evaluate_ppl(model, tokenizer)
    act_q.remove()
    results.append({'variant': 'MXFP4-W4A4-B16', 'w_scale': 'E8M0 block-16', 'a_scale': 'E8M0 block-16',
                    'ppl': ppl_mxfp4, 'hardware_cost': HARDWARE_COST['MXFP4-W4A4-B16']})
    print(f"  PPL = {ppl_mxfp4:.3f}  ΔPPL vs BF16 = {ppl_mxfp4 - ppl_bf16:+.3f}")
    del model; gc.collect(); torch.cuda.empty_cache()

    # ── True NVFP4-W4A4-B16 — E4M3 block scales ──────────────────────────────────
    print("\n[4/5] NVFP4-W4A4-B16  (E4M3 block-16 weights + E4M3 activations)")
    model, tokenizer = load_fresh()
    apply_fp4_weights(model, block_size=BLOCK_SIZE, scale_fmt='e4m3')
    act_q = TrueNVFP4ActivationQuantizer(model)
    ppl_nvfp4 = evaluate_ppl(model, tokenizer)
    act_q.remove()
    results.append({'variant': 'NVFP4-W4A4-B16', 'w_scale': 'E4M3 block-16', 'a_scale': 'E4M3 block-16',
                    'ppl': ppl_nvfp4, 'hardware_cost': HARDWARE_COST['NVFP4-W4A4-B16']})
    print(f"  PPL = {ppl_nvfp4:.3f}  ΔPPL vs BF16 = {ppl_nvfp4 - ppl_bf16:+.3f}")
    del model; gc.collect(); torch.cuda.empty_cache()

    # ── Ideal-scale W4A4 — FP32 block scales (upper bound) ───────────────────
    print("\n[5/5] Ideal-W4A4  (FP32 block-16 weights + FP32 activations, upper bound)")
    model, tokenizer = load_fresh()
    apply_fp4_weights(model, block_size=BLOCK_SIZE, scale_fmt='fp32')
    act_q = IdealScaleActivationQuantizer(model)
    ppl_ideal = evaluate_ppl(model, tokenizer)
    act_q.remove()
    results.append({'variant': 'Ideal-W4A4', 'w_scale': 'FP32 block-16', 'a_scale': 'FP32 block-16',
                    'ppl': ppl_ideal, 'hardware_cost': HARDWARE_COST['Ideal-W4A4']})
    print(f"  PPL = {ppl_ideal:.3f}  ΔPPL vs BF16 = {ppl_ideal - ppl_bf16:+.3f}")
    del model; gc.collect(); torch.cuda.empty_cache()

    # ── Save and summarise ────────────────────────────────────────────────────
    df = pd.DataFrame(results)
    df['delta_ppl_vs_bf16'] = df['ppl'] - ppl_bf16
    df.to_csv(os.path.join(OUT_DIR, "rq11_scale_comparison.csv"), index=False)

    print("\n" + "=" * 70)
    print("RQ11 — Block-Scale Format Comparison Summary")
    print("=" * 70)
    print(f"\n  Model: {MODEL_NAME}   Block size: {BLOCK_SIZE}")
    print(f"\n  {'Variant':<18} {'W scale':<22} {'A scale':<22} "
          f"{'PPL':>7} {'ΔPPL':>7}  Hardware cost")
    print("  " + "-" * 90)
    for _, r in df.iterrows():
        print(f"  {r['variant']:<18} {r['w_scale']:<22} {r['a_scale']:<22} "
              f"{r['ppl']:>7.3f} {r['delta_ppl_vs_bf16']:>+7.3f}  {r['hardware_cost']}")

    # Derived conclusions
    gap_e8m0_vs_ideal = ppl_mxfp4 - ppl_ideal
    gap_e4m3_vs_ideal = ppl_nvfp4 - ppl_ideal
    gap_e8m0_vs_e4m3  = ppl_mxfp4 - ppl_nvfp4
    print(f"\n  Scale format cost:")
    print(f"    E8M0 vs ideal:  ΔPPL = {gap_e8m0_vs_ideal:+.3f}")
    print(f"    E4M3 vs ideal:  ΔPPL = {gap_e4m3_vs_ideal:+.3f}")
    print(f"    E8M0 vs E4M3:   ΔPPL = {gap_e8m0_vs_e4m3:+.3f}")
    if abs(gap_e8m0_vs_e4m3) < 0.5:
        print("  → E8M0 shift-only scale is sufficient: ΔPPL vs E4M3 < 0.5")
        print("    Hardware recommendation: use shift-only E8M0 scale path.")
    else:
        print(f"  → E4M3 provides {-gap_e8m0_vs_e4m3:.2f} PPL improvement over E8M0.")
        print("    Hardware recommendation: evaluate E4M3 FP8-scale multiply cost.")
    print(f"\n  Scale error summary (weight matrices, {scale_errs['n_layers']} layers):")
    print(f"    E8M0 rel. scale error: mean={scale_errs['e8m0_scale_err_mean']:.4f}  "
          f"max={scale_errs['e8m0_scale_err_max']:.4f}")
    print(f"    E4M3 rel. scale error: mean={scale_errs['e4m3_scale_err_mean']:.6f}  "
          f"max={scale_errs['e4m3_scale_err_max']:.6f}")

    # ── Plot ──────────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    variants  = df['variant'].tolist()
    ppls      = df['ppl'].tolist()
    deltas    = df['delta_ppl_vs_bf16'].tolist()
    colors = ['#2ecc71', '#3498db', '#e67e22', '#e74c3c', '#9b59b6']

    ax = axes[0]
    bars = ax.bar(variants, ppls, color=colors[:len(variants)], alpha=0.85)
    ax.axhline(ppl_bf16, color='green', linestyle='--', linewidth=1.5,
               label=f'BF16 baseline ({ppl_bf16:.2f})')
    ax.set_ylabel("Perplexity (wikitext-2)")
    ax.set_title("PPL by Scale Format")
    ax.legend(fontsize=9)
    ax.set_xticks(range(len(variants)))
    ax.set_xticklabels(variants, rotation=15, ha='right')
    ax.grid(axis='y', alpha=0.3)
    for bar, val in zip(bars, ppls):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.1,
                f'{val:.2f}', ha='center', va='bottom', fontsize=8)

    ax = axes[1]
    subset = df[df['variant'] != 'BF16'].copy()
    bars2 = ax.bar(subset['variant'], subset['delta_ppl_vs_bf16'],
                   color=colors[1:len(subset) + 1], alpha=0.85)
    ax.axhline(0, color='black', linewidth=0.8)
    ax.axhline(0.5, color='red', linestyle=':', alpha=0.6, label='±0.5 PPL threshold')
    ax.axhline(-0.5, color='red', linestyle=':', alpha=0.6)
    ax.set_ylabel("ΔPPL vs BF16 baseline")
    ax.set_title("PPL Degradation by Scale Format")
    ax.set_xticks(range(len(subset)))
    ax.set_xticklabels(subset['variant'], rotation=15, ha='right')
    ax.legend(fontsize=9)
    ax.grid(axis='y', alpha=0.3)
    for bar, val in zip(bars2, subset['delta_ppl_vs_bf16']):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + (0.05 if val >= 0 else -0.15),
                f'{val:+.2f}', ha='center', va='bottom', fontsize=8)

    plt.suptitle(f"RQ11 — Scale Format Impact on W4A4 Accuracy "
                 f"({MODEL_NAME.split('/')[-1]}, block={BLOCK_SIZE})", fontsize=11)
    plt.tight_layout()
    fname = os.path.join(OUT_DIR, "rq11_scale_comparison.png")
    plt.savefig(fname, dpi=200)
    print(f"\n  Saved: {fname}")
    print(f"All results saved to {OUT_DIR}/")


if __name__ == "__main__":
    main()
