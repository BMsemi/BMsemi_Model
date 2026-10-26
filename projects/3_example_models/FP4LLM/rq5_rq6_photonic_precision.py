"""
RQ5 + RQ6 — Photonic Accumulation Precision & Layer-wise Sensitivity

RQ5: What ENOB, SNR, and dynamic range are required for FP4 MAC accumulation?
RQ6: Which layers and tensor types require higher sensing / accumulation precision?

Two sub-experiments:
  A) Capture output distributions → compute per-layer ENOB requirements
  B) ADC precision sweep → measure PPL vs. accumulator bit-width per layer type

Outputs: results/rq5_rq6/
"""
import os, sys, gc, json
os.environ["CUDA_VISIBLE_DEVICES"] = "4"              # must be before torch import
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
sys.path.insert(0, os.path.dirname(__file__))

import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from tqdm import tqdm
from src.fp4_utils import apply_fp4_weights, classify_layer
from src.photonic_model import (LinearOutputCapture,
                                 AccumulatorPrecisionLimiter,
                                 required_accumulator_bits)

# ── Configuration ─────────────────────────────────────────────────────────────

MODEL_NAME  = "Qwen/Qwen2.5-1.5B"    # base model — instruct models give ~100K PPL on raw wikitext
SEQ_LEN     = 512
MAX_SAMPLES = 32
SEED        = 42
OUT_DIR     = "results/rq5_rq6"
os.makedirs(OUT_DIR, exist_ok=True)

# ADC bit-widths to sweep (RQ6 sensitivity)
ADC_BITS_SWEEP = [4, 6, 8, 10, 12, 16]


# ── PPL helper ────────────────────────────────────────────────────────────────

def evaluate_ppl(model, tokenizer, max_samples=MAX_SAMPLES) -> float:
    ds   = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    text = "\n\n".join(ds["text"])
    text = text[:max_samples * SEQ_LEN * 6]   # truncate before tokenizing
    enc  = tokenizer(text, add_special_tokens=False, return_tensors="pt")
    ids  = enc.input_ids[:, :max_samples * SEQ_LEN]

    device = next(model.parameters()).device
    ids    = ids.to(device)
    nlls   = []
    with torch.no_grad():
        for i in range(max_samples):
            s, e = i * SEQ_LEN, (i + 1) * SEQ_LEN
            if e > ids.size(1):
                break
            chunk = ids[:, s:e]
            out   = model(chunk, labels=chunk.clone())
            nlls.append(out.loss.item())
    return torch.exp(torch.tensor(nlls).mean()).item()


# ── Experiment A: distribution capture ────────────────────────────────────────

def run_distribution_capture(model, tokenizer) -> dict:
    print("  Capturing output distributions...")
    capture = LinearOutputCapture(model, max_samples=64)

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

    capture.remove()
    return capture.compute_stats()


def plot_distribution_stats(stats: dict):
    """Heatmap of per-layer ENOB requirements."""
    names    = list(stats.keys())
    enob_vals = [stats[n]['enob']          for n in names]
    bits_vals = [stats[n]['required_bits'] for n in names]
    dr_vals   = [np.log2(stats[n]['dynamic_range']) for n in names]
    types     = [stats[n]['layer_type']    for n in names]

    colors = {'attn': 'steelblue', 'mlp': 'darkorange', 'other': 'gray'}
    bar_colors = [colors.get(t, 'gray') for t in types]

    fig, axes = plt.subplots(3, 1, figsize=(max(14, len(names) * 0.4), 11))

    for ax, vals, ylabel, title in [
        (axes[0], enob_vals, "ENOB",           "ENOB Required per Layer"),
        (axes[1], bits_vals, "Accumulator bits","Required Accumulator Bits"),
        (axes[2], dr_vals,   "log₂(DR)",        "log₂(Dynamic Range)"),
    ]:
        ax.bar(range(len(names)), vals, color=bar_colors, alpha=0.8)
        ax.set_xticks(range(len(names)))
        ax.set_xticklabels([n.split('.')[-1] for n in names], rotation=90, fontsize=6)
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.grid(True, axis='y', alpha=0.3)

    # Legend
    from matplotlib.patches import Patch
    legend_handles = [Patch(color=c, label=t) for t, c in colors.items()]
    axes[0].legend(handles=legend_handles, loc='upper right')

    plt.suptitle(f"RQ5+6: Photonic Accumulation Precision — {MODEL_NAME.split('/')[-1]}")
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "rq5_enob_per_layer.png"), dpi=200)
    print(f"  Saved: rq5_enob_per_layer.png")


# ── Experiment B: ADC precision sweep ─────────────────────────────────────────

def run_adc_sweep(model, tokenizer, layer_stats: dict) -> list:
    """
    Sweep ADC bit-width applied globally to all linear layers.
    Also sweep per layer type (attn only / mlp only).
    """
    results = []

    # 1) Baseline (no ADC limitation)
    baseline_ppl = evaluate_ppl(model, tokenizer)
    results.append({'scope': 'baseline', 'n_bits': None, 'ppl': baseline_ppl})
    print(f"  baseline PPL={baseline_ppl:.3f}")

    # 2) Global ADC precision sweep
    for n_bits in tqdm(ADC_BITS_SWEEP, desc="Global ADC sweep"):
        limiter = AccumulatorPrecisionLimiter(model, n_bits=n_bits)
        ppl     = evaluate_ppl(model, tokenizer)
        limiter.remove()
        results.append({'scope': 'all', 'n_bits': n_bits, 'ppl': ppl})
        print(f"  all layers  n_bits={n_bits}  PPL={ppl:.3f}")

    # 3) Per layer-type sweep (attn only, mlp only)
    attn_layers = [n for n, s in layer_stats.items() if s['layer_type'] == 'attn']
    mlp_layers  = [n for n, s in layer_stats.items() if s['layer_type'] == 'mlp']

    for scope, layer_list in [('attn_only', attn_layers), ('mlp_only', mlp_layers)]:
        for n_bits in tqdm(ADC_BITS_SWEEP, desc=f"{scope} ADC sweep"):
            limiters = [AccumulatorPrecisionLimiter(model, n_bits=n_bits, target_layer=ln)
                        for ln in layer_list]
            ppl = evaluate_ppl(model, tokenizer)
            for l in limiters:
                l.remove()
            results.append({'scope': scope, 'n_bits': n_bits, 'ppl': ppl})
            print(f"  {scope}  n_bits={n_bits}  PPL={ppl:.3f}")

    return results


def plot_adc_sweep(sweep_results: list, baseline_ppl: float):
    df  = pd.DataFrame(sweep_results)
    sub = df[df['scope'] != 'baseline'].copy()

    fig, ax = plt.subplots(figsize=(9, 5))
    for scope in ['all', 'attn_only', 'mlp_only']:
        ss = sub[sub['scope'] == scope].sort_values('n_bits')
        ax.plot(ss['n_bits'], ss['ppl'], 'o-', label=scope)
    ax.axhline(baseline_ppl, color='red', linestyle='--', label=f'FP4 W4A16 baseline ({baseline_ppl:.2f})')
    ax.set_xlabel("ADC / Accumulator bit-width")
    ax.set_ylabel("Perplexity (wikitext-2)")
    ax.set_title("RQ6: Layer-wise Sensitivity to Accumulation Precision")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "rq6_adc_sweep.png"), dpi=200)
    print("  Saved: rq6_adc_sweep.png")


# ── Analytical ENOB table ─────────────────────────────────────────────────────

def print_analytical_enob():
    print("\n=== Analytical ENOB vs. K (FP4×FP4) ===")
    print(f"{'K':>6}  {'max_sum':>10}  {'DR':>12}  {'bits':>6}  {'ENOB':>7}  {'SNR(dB)':>9}")
    print("-" * 55)
    for K in [64, 128, 256, 512, 1024, 2048, 4096, 8192]:
        r = required_accumulator_bits(K)
        print(f"{r['k']:>6}  {r['max_partial_sum']:>10.0f}  "
              f"{r['dynamic_range']:>12.0f}  {r['required_bits']:>6}  "
              f"{r['enob']:>7.2f}  {r['snr_db']:>9.2f}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    torch.manual_seed(SEED)
    print(f"Loading {MODEL_NAME}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model     = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, dtype=torch.bfloat16, device_map="auto"
    )
    model.eval()
    apply_fp4_weights(model)

    print_analytical_enob()

    # Experiment A
    print("\n=== Experiment A: Output Distribution Capture ===")
    layer_stats = run_distribution_capture(model, tokenizer)

    df_dist = pd.DataFrame.from_dict(layer_stats, orient='index')
    df_dist.index.name = 'layer'
    df_dist.to_csv(os.path.join(OUT_DIR, "rq5_layer_stats.csv"))
    print(f"\n  Captured {len(df_dist)} layers")
    print(df_dist[['layer_type', 'required_bits', 'enob', 'dynamic_range',
                    'max_abs', 'snr_db_needed']].to_string())
    plot_distribution_stats(layer_stats)

    # Summary by layer type
    df_dist['layer_type_cat'] = df_dist['layer_type']
    grp = df_dist.groupby('layer_type')[['enob', 'required_bits', 'dynamic_range']].agg(['mean', 'max'])
    print("\n--- ENOB by layer type ---")
    print(grp.to_string())

    # Experiment B
    print("\n=== Experiment B: ADC Precision Sweep ===")
    sweep_results = run_adc_sweep(model, tokenizer, layer_stats)

    df_sweep = pd.DataFrame(sweep_results)
    df_sweep.to_csv(os.path.join(OUT_DIR, "rq6_adc_sweep.csv"), index=False)

    baseline_ppl = df_sweep.loc[df_sweep['scope'] == 'baseline', 'ppl'].values[0]
    plot_adc_sweep(sweep_results, baseline_ppl)

    # Save combined JSON
    summary = {
        'layer_stats': {k: {kk: float(vv) if isinstance(vv, (float, np.floating)) else vv
                             for kk, vv in v.items()} for k, v in layer_stats.items()},
        'adc_sweep':   sweep_results,
    }
    with open(os.path.join(OUT_DIR, "rq5_rq6_results.json"), "w") as f:
        json.dump(summary, f, indent=2, default=str)

    print(f"\nAll results saved to {OUT_DIR}/")


if __name__ == "__main__":
    main()
