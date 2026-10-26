"""
RQ9 — Partial Sum Overflow: Accumulator Width vs. PPL

For each operating point (WFP4AINT8, MXFP4_W4A4_B16):
  Step 0: Capture linear-layer output distributions — min, max, std, 99th/99.9th
          percentile, dynamic range. These ARE the final partial sums the
          accumulator must hold; intermediates are always strictly smaller.
  Step 1: Sweep accumulator bit-width N ∈ ACC_BITS_SWEEP.
          Simulate N-bit overflow by quantizing each linear layer's output to
          N bits (sign-magnitude, dynamic range matched to observed amax).
          Measure PPL at each N to find the knee.

Operating point determines which quantizers are applied to weights/activations
BEFORE the output clamp:
  WFP4AINT8   — apply_fp4_weights (per-channel) + INT8ActivationQuantizer
  MXFP4_W4A4_B16  — apply_fp4_weights (block-16)    + NVFP4ActivationQuantizer

Outputs: results/rq9/
"""
import os, sys, gc, json
os.environ["CUDA_VISIBLE_DEVICES"] = "4"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
sys.path.insert(0, os.path.dirname(__file__))

import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from tqdm import tqdm

from src.fp4_utils import (apply_fp4_weights,
                           INT8ActivationQuantizer, NVFP4ActivationQuantizer)
from src.photonic_model import LinearOutputCapture, AccumulatorPrecisionLimiter

# ── Configuration ──────────────────────────────────────────────────────────────

MODEL_NAME  = "Qwen/Qwen2.5-1.5B"
SEQ_LEN     = 512
MAX_SAMPLES = 32
SEED        = 42
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUT_DIR     = os.path.join(_SCRIPT_DIR, "results", "rq9")
os.makedirs(OUT_DIR, exist_ok=True)

# Accumulator bit-widths to sweep (signed, so range = ±2^(N-1))
ACC_BITS_SWEEP = [8, 10, 12, 14, 16, 18, 20, 24, 32]

OPERATING_POINTS = {
    "WFP4AINT8":  dict(block_size=None,  act_cls=INT8ActivationQuantizer),
    "MXFP4_W4A4_B16": dict(block_size=16,    act_cls=NVFP4ActivationQuantizer),
}


# ── PPL helper ─────────────────────────────────────────────────────────────────

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


# ── Step 0: partial sum distribution capture ───────────────────────────────────

def capture_psum_stats(model, tokenizer, op_name: str) -> dict:
    """
    Capture the distribution of linear-layer outputs (= final partial sums).
    Returns per-layer stats: min, max, std, p99, p999, dynamic_range.
    """
    print(f"  [{op_name}] Capturing partial sum distributions...")
    capture = LinearOutputCapture(model, max_samples=MAX_SAMPLES)

    ds   = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    text = "\n\n".join(ds["text"])
    text = text[:MAX_SAMPLES * SEQ_LEN * 6]   # truncate before tokenizing
    enc  = tokenizer(text, add_special_tokens=False, return_tensors="pt")
    ids  = enc.input_ids[:, :MAX_SAMPLES * SEQ_LEN]
    device = next(model.parameters()).device

    with torch.no_grad():
        for i in range(MAX_SAMPLES):
            s, e = i * SEQ_LEN, (i + 1) * SEQ_LEN
            if e > ids.size(1):
                break
            model(ids[:, s:e].to(device))

    capture.remove()
    raw = capture.compute_stats()

    # augment with extra percentiles needed for overflow analysis
    stats = {}
    for name, st in raw.items():
        vals_sample = []   # re-collect just for percentiles
        stats[name] = {
            'op':            op_name,
            'layer':         name,
            'layer_type':    st['layer_type'],
            'in_features':   st['in_features'],
            'out_features':  st['out_features'],
            'max_abs':       st['max_abs'],
            'min_abs_nz':    st['min_abs_nonzero'],
            'mean_abs':      st['mean_abs'],
            'std':           st['std'],
            'p99':           st['p99'],
            'p999':          float(st.get('p999', st['p99'])),   # fallback if absent
            'dynamic_range': st['dynamic_range'],
            'required_bits': st['required_bits'],
            'enob':          st['enob'],
        }
    return stats


# ── Step 1: overflow sweep ──────────────────────────────────────────────────────

def run_overflow_sweep(model, tokenizer, op_name: str) -> list:
    results = []

    # baseline (no clamping) under this operating point
    baseline_ppl = evaluate_ppl(model, tokenizer)
    results.append({'op': op_name, 'n_bits': 'inf', 'ppl': baseline_ppl,
                    'overflow_scope': 'none'})
    print(f"  [{op_name}] baseline PPL = {baseline_ppl:.3f}")

    for n_bits in tqdm(ACC_BITS_SWEEP, desc=f"  [{op_name}] overflow sweep"):
        limiter = AccumulatorPrecisionLimiter(model, n_bits=n_bits)
        ppl = evaluate_ppl(model, tokenizer)
        limiter.remove()
        results.append({'op': op_name, 'n_bits': n_bits, 'ppl': ppl,
                        'overflow_scope': 'all'})
        print(f"  [{op_name}] n_bits={n_bits:2d}  PPL={ppl:.3f}  "
              f"ΔPPL={ppl - baseline_ppl:+.3f}")

    return results


# ── Plot ────────────────────────────────────────────────────────────────────────

def plot_results(sweep_df: pd.DataFrame, psum_dfs: dict):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # Left: PPL vs accumulator bits per operating point
    ax = axes[0]
    colors = {'WFP4AINT8': 'steelblue', 'MXFP4_W4A4_B16': 'darkorange'}
    for op_name in OPERATING_POINTS:
        sub = sweep_df[(sweep_df['op'] == op_name) &
                       (sweep_df['n_bits'] != 'inf')].copy()
        sub['n_bits'] = sub['n_bits'].astype(int)
        sub = sub.sort_values('n_bits')
        baseline = sweep_df[(sweep_df['op'] == op_name) &
                             (sweep_df['n_bits'] == 'inf')]['ppl'].values[0]
        ax.plot(sub['n_bits'], sub['ppl'], 'o-',
                label=op_name, color=colors[op_name])
        ax.axhline(baseline, linestyle='--', color=colors[op_name], alpha=0.4,
                   label=f'{op_name} baseline ({baseline:.2f})')

    ax.set_xlabel("Accumulator bit-width N")
    ax.set_ylabel("Perplexity (wikitext-2)")
    ax.set_title("RQ9: Partial Sum Overflow — PPL vs Accumulator Width")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # Right: per-layer max_abs distribution for both OPs
    ax2 = axes[1]
    for op_name, df in psum_dfs.items():
        vals = df['max_abs'].values
        ax2.ecdf(vals, label=op_name, color=colors[op_name])
    ax2.set_xlabel("Max |partial sum| per layer")
    ax2.set_ylabel("CDF")
    ax2.set_title("Layer-wise Partial Sum Range")
    ax2.set_xscale('log')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.suptitle(f"RQ9 — Accumulator Overflow: {MODEL_NAME.split('/')[-1]}")
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "rq9_overflow.png"), dpi=200)
    print("  Saved: rq9_overflow.png")


# ── Main ────────────────────────────────────────────────────────────────────────

def main():
    torch.manual_seed(SEED)

    all_sweep   = []
    psum_dfs    = {}
    all_psum    = []

    for op_name, cfg in OPERATING_POINTS.items():
        print(f"\n{'='*60}")
        print(f"Operating point: {op_name}")
        print(f"{'='*60}")

        model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME, dtype=torch.bfloat16, device_map="auto"
        )
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        model.eval()

        # Apply quantization for this operating point
        apply_fp4_weights(model, block_size=cfg['block_size'])
        act_q = cfg['act_cls'](model)

        # Step 0 — partial sum distribution
        print("\n  --- Step 0: partial sum distribution ---")
        psum_stats = capture_psum_stats(model, tokenizer, op_name)
        df_psum = pd.DataFrame(psum_stats.values())
        df_psum.to_csv(os.path.join(OUT_DIR, f"rq9_psum_{op_name}.csv"), index=False)
        psum_dfs[op_name] = df_psum
        all_psum.append(df_psum)

        # Summary
        print(f"\n  Partial sum stats ({op_name}) — across {len(df_psum)} layers:")
        print(f"    max_abs  : {df_psum['max_abs'].max():.4f}  "
              f"(median {df_psum['max_abs'].median():.4f})")
        print(f"    p99      : {df_psum['p99'].max():.4f}  "
              f"(median {df_psum['p99'].median():.4f})")
        print(f"    DR       : {df_psum['dynamic_range'].max():.1f}  "
              f"(median {df_psum['dynamic_range'].median():.1f})")
        print(f"    req_bits : {df_psum['required_bits'].max()}  "
              f"(median {df_psum['required_bits'].median():.1f})")

        # Step 1 — overflow sweep
        print("\n  --- Step 1: overflow sweep ---")
        sweep = run_overflow_sweep(model, tokenizer, op_name)
        all_sweep.extend(sweep)

        act_q.remove()
        del model
        gc.collect()
        torch.cuda.empty_cache()

    # Save
    sweep_df = pd.DataFrame(all_sweep)
    sweep_df.to_csv(os.path.join(OUT_DIR, "rq9_sweep.csv"), index=False)

    psum_all_df = pd.concat(all_psum, ignore_index=True)
    psum_all_df.to_csv(os.path.join(OUT_DIR, "rq9_psum_all.csv"), index=False)

    # Combined summary table
    print("\n=== OVERFLOW SWEEP SUMMARY ===")
    pivot = sweep_df[sweep_df['n_bits'] != 'inf'].pivot(
        index='n_bits', columns='op', values='ppl')
    print(pivot.to_string())

    # Analytical worst-case reminder
    print("\n=== ANALYTICAL WORST CASE (FP4×FP4, K=hidden_dim) ===")
    from src.photonic_model import required_accumulator_bits
    for K in [896, 1536, 2048, 4096]:   # common hidden dims
        r = required_accumulator_bits(K)
        print(f"  K={K}: max_psum={r['max_partial_sum']:.0f}  "
              f"req_bits={r['required_bits']}  ENOB={r['enob']:.1f}")

    plot_results(sweep_df, psum_dfs)

    with open(os.path.join(OUT_DIR, "rq9_results.json"), "w") as f:
        json.dump({
            'sweep': all_sweep,
            'psum_summary': {
                op: {
                    'max_abs_max':  float(df['max_abs'].max()),
                    'max_abs_med':  float(df['max_abs'].median()),
                    'p99_max':      float(df['p99'].max()),
                    'req_bits_max': int(df['required_bits'].max()),
                    'req_bits_med': float(df['required_bits'].median()),
                }
                for op, df in psum_dfs.items()
            }
        }, f, indent=2, default=str)

    print(f"\nAll results saved to {OUT_DIR}/")


if __name__ == "__main__":
    main()
