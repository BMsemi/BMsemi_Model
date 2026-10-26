"""
RQ9b — Partial Sum Bit-Growth vs Number of Accumulated Rows

Question: if the RRAM column accumulates K rows simultaneously, how many
accumulator bits are needed?  Verify the formula (M+N) + log2(K).

Two perspectives:
  (A) Analytical worst case:
        max partial sum = K × max_w × max_a
        bits = ceil(log2(K × max_w × max_a)) + 1  [sign]
        For FP4 UQ3.2: max_w = max_a = 24, so bits = ceil(log2(576K)) + 1

  (B) Empirical: hook each linear layer, compute partial dot products for
        K = 1,2,4,...,in_features, record the actual max |partial sum| seen
        during inference.  Compare to worst case.

      empirical_bits(K) = ceil(log2(empirical_max_psum(K))) + 1

The gap (worst_case − empirical) is the safety margin hardware can exploit.

Outputs: results/rq9b/
"""
import os, sys, gc
os.environ["CUDA_VISIBLE_DEVICES"] = "4"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
sys.path.insert(0, os.path.dirname(__file__))

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset

from src.fp4_utils import (apply_fp4_weights,
                           INT8ActivationQuantizer, NVFP4ActivationQuantizer,
                           FP4_MAX)

# ── Configuration ──────────────────────────────────────────────────────────────

MODEL_NAME  = "Qwen/Qwen2.5-1.5B"
SEQ_LEN     = 512
MAX_SAMPLES = 16          # fewer samples — partial sum computation is expensive
SEED        = 42
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUT_DIR     = os.path.join(_SCRIPT_DIR, "results", "rq9b")
os.makedirs(OUT_DIR, exist_ok=True)

# K checkpoints: rows accumulated simultaneously
# Always ends at the actual layer in_features (added dynamically per layer)
K_BASE = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096]

# UQ3.2 max magnitude code (from barrel shift: "11" << 3 = 24)
UQ32_MAX = 24

OPERATING_POINTS = {
    "WFP4AINT8":  dict(block_size=None, act_cls=INT8ActivationQuantizer,  act_max=127),
    "MXFP4_W4A4_B16": dict(block_size=16,   act_cls=NVFP4ActivationQuantizer, act_max=UQ32_MAX),
}


# ── (A) Analytical ─────────────────────────────────────────────────────────────

def analytical_bits(K_vals, w_max=UQ32_MAX, a_max=UQ32_MAX):
    """
    Worst-case signed accumulator bits for K products of w_max × a_max.
    Returns list of (K, max_psum, required_bits) — no sign bit convention:
      required_bits = ceil(log2(K × w_max × a_max + 1)) + 1  (signed)
    """
    rows = []
    for K in K_vals:
        max_psum = K * w_max * a_max
        bits = int(np.ceil(np.log2(max_psum + 1))) + 1   # +1 sign
        formula_bits = int(np.ceil(np.log2(w_max * a_max))) + int(np.ceil(np.log2(K))) + 1
        rows.append({
            'K':            K,
            'max_psum':     max_psum,
            'bits_exact':   bits,
            'bits_formula': formula_bits,   # (M+N) + ceil(log2(K)) + 1
        })
    return pd.DataFrame(rows)


def print_analytical_table(w_max=UQ32_MAX, a_max=UQ32_MAX):
    MN = int(np.ceil(np.log2(w_max * a_max)))   # M+N effective bits for product
    print(f"\n=== Analytical worst case: FP4 UQ3.2 ===")
    print(f"  w_max={w_max} (UQ3.2 code),  a_max={a_max}")
    print(f"  Single product max = {w_max*a_max},  (M+N) = ceil(log2({w_max*a_max})) = {MN} bits")
    print(f"  Formula: (M+N) + ceil(log2(K)) + 1(sign)  =  {MN} + ceil(log2(K)) + 1")
    print()
    print(f"  {'K':>6}  {'max_psum':>12}  {'bits_exact':>12}  {'formula (M+N)+log2K+1':>24}")
    print("  " + "-"*58)
    df = analytical_bits(K_BASE, w_max, a_max)
    for _, r in df.iterrows():
        print(f"  {int(r.K):>6}  {r.max_psum:>12.0f}  {r.bits_exact:>12d}  "
              f"  {r.bits_formula:>6d}   (M+N={MN}, log2(K)={int(np.ceil(np.log2(r.K+1)))})")
    return df


# ── (B) Empirical partial sum capture ──────────────────────────────────────────

class PartialSumGrowthCapture:
    """
    For each linear layer: computes partial dot products for K=1,2,...,in_features
    and records max |partial sum| per K checkpoint.

    NOTE: This is compute-intensive because it recomputes sub-GEMMs.
    We limit to first MAX_COLS_PER_LAYER output channels for speed.
    """
    MAX_OUT_SAMPLE = 64    # sample this many output channels per layer

    def __init__(self, model, k_checkpoints):
        self.k_checkpoints = k_checkpoints
        self.records = []   # list of dicts
        self.hooks = []

        for name, module in model.named_modules():
            if not isinstance(module, nn.Linear):
                continue
            h = module.register_forward_hook(self._make_hook(name, module))
            self.hooks.append(h)

    def _make_hook(self, name, module):
        def hook(mod, inputs, output):
            x = inputs[0].detach().float()   # [B, T, in_features]
            w = mod.weight.detach().float()  # [out_features, in_features]
            in_f = w.shape[1]

            # Sample a subset of output channels to keep compute manageable
            out_idx = torch.arange(min(self.MAX_OUT_SAMPLE, w.shape[0]),
                                   device=w.device)
            w_sub = w[out_idx]               # [out_sample, in_features]

            # Flatten batch & seq
            x_flat = x.reshape(-1, in_f)    # [B*T, in_features]

            for K in self.k_checkpoints:
                if K > in_f:
                    continue
                # Partial product: x[:, :K] @ w_sub[:, :K].T
                psum = x_flat[:, :K] @ w_sub[:, :K].t()  # [B*T, out_sample]
                max_abs = psum.abs().max().item()

                self.records.append({
                    'layer': name,
                    'K': K,
                    'in_features': in_f,
                    'max_abs_psum': max_abs,
                })
        return hook

    def remove(self):
        for h in self.hooks:
            h.remove()
        self.hooks = []

    def aggregate(self) -> pd.DataFrame:
        """Max over all samples per (layer, K)."""
        df = pd.DataFrame(self.records)
        return df.groupby(['layer', 'K', 'in_features'])['max_abs_psum'].max().reset_index()


def run_empirical(model, tokenizer, k_checkpoints):
    capture = PartialSumGrowthCapture(model, k_checkpoints)

    ds   = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    text = "\n\n".join(ds["text"])
    text = text[:MAX_SAMPLES * SEQ_LEN * 6]   # truncate before tokenizing
    enc  = tokenizer(text, add_special_tokens=False, return_tensors="pt")
    ids  = enc.input_ids[:, :MAX_SAMPLES * SEQ_LEN]
    device = next(model.parameters()).device

    print(f"  Running {MAX_SAMPLES} batches for empirical partial sum stats...")
    with torch.no_grad():
        for i in range(MAX_SAMPLES):
            s, e = i * SEQ_LEN, (i + 1) * SEQ_LEN
            if e > ids.size(1):
                break
            model(ids[:, s:e].to(device))

    capture.remove()
    return capture.aggregate()


# ── Plotting ───────────────────────────────────────────────────────────────────

def plot_bit_growth(empirical_df: pd.DataFrame, analytical_df: pd.DataFrame,
                    op_name: str, w_max: int, a_max: int):
    """
    Plot required bits vs K for worst-case and empirical (all layers + median).
    """
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    MN = int(np.ceil(np.log2(w_max * a_max)))

    # ── Left: max partial sum vs K ───────────────────────────────────────────
    ax = axes[0]

    # Per-layer empirical curves (light)
    for layer, grp in empirical_df.groupby('layer'):
        in_f = grp['in_features'].iloc[0]
        sub  = grp[grp['K'] <= in_f].sort_values('K')
        ax.plot(sub['K'], sub['max_abs_psum'], alpha=0.15, color='steelblue', lw=0.8)

    # Median empirical
    med = empirical_df.groupby('K')['max_abs_psum'].median().reset_index()
    ax.plot(med['K'], med['max_abs_psum'], 'o-', color='steelblue',
            lw=2, label='empirical median')

    # Worst-case analytical
    ax.plot(analytical_df['K'], analytical_df['max_psum'], 's--',
            color='crimson', lw=2, label=f'worst case (K×{w_max}×{a_max})')

    ax.set_xscale('log', base=2)
    ax.set_yscale('log', base=2)
    ax.set_xlabel("K (rows accumulated)")
    ax.set_ylabel("|partial sum| max")
    ax.set_title(f"[{op_name}] Partial Sum Magnitude vs K")
    ax.legend()
    ax.grid(True, alpha=0.3, which='both')

    # ── Right: required bits vs K ────────────────────────────────────────────
    ax2 = axes[1]

    # Analytical formula: (M+N) + ceil(log2(K)) + 1
    K_fine = np.array([2**i for i in range(13)])
    bits_formula = MN + np.ceil(np.log2(K_fine + 1)).astype(int) + 1
    ax2.plot(K_fine, bits_formula, 's--', color='crimson', lw=2,
             label=f'formula: {MN}+⌈log₂K⌉+1')

    # Empirical bits per K
    def bits_needed(v):
        return int(np.ceil(np.log2(v + 1))) + 1 if v > 0 else 1

    emp_bits = med.copy()
    emp_bits['bits'] = emp_bits['max_abs_psum'].apply(bits_needed)
    ax2.plot(emp_bits['K'], emp_bits['bits'], 'o-', color='steelblue', lw=2,
             label='empirical median')

    # Deep_gate reference (K=1, row-serial)
    ax2.axvline(1, color='green', linestyle=':', alpha=0.7, label='K=1 row-serial')
    ax2.axhline(bits_formula[0], color='green', linestyle=':', alpha=0.7)

    ax2.set_xscale('log', base=2)
    ax2.set_xlabel("K (rows accumulated)")
    ax2.set_ylabel("Required accumulator bits")
    ax2.set_title(f"[{op_name}] Accumulator Bits vs K\n"
                  f"(M+N)=⌈log₂({w_max}×{a_max})⌉={MN}")
    ax2.legend()
    ax2.grid(True, alpha=0.3, which='both')

    # Annotate useful K points
    for K_mark, label in [(1, 'row-serial'), (64, 'subarray-64'),
                           (256, 'subarray-256'), (4096, 'full col')]:
        if K_mark in analytical_df['K'].values:
            b = analytical_df[analytical_df['K'] == K_mark]['bits_exact'].values[0]
            ax2.annotate(f"K={K_mark}\n{b}b",
                         xy=(K_mark, b), xytext=(K_mark * 1.4, b + 0.5),
                         fontsize=7, arrowprops=dict(arrowstyle='->', lw=0.8))

    plt.suptitle(f"RQ9b — Partial Sum Bit Growth: {op_name} | {MODEL_NAME.split('/')[-1]}")
    plt.tight_layout()
    fname = os.path.join(OUT_DIR, f"rq9b_bit_growth_{op_name}.png")
    plt.savefig(fname, dpi=200)
    print(f"  Saved: {fname}")


# ── (C) FP16 decoder equivalence ──────────────────────────────────────────────

FP16_MANTISSA_BITS = 11    # 10 explicit + 1 implicit leading 1
FP16_EPS           = 2 ** -(FP16_MANTISSA_BITS - 1)   # ≈ 0.05%


def fp16_decoder_table(analytical_df: pd.DataFrame, w_max: int, a_max: int,
                       rq9_sweep_csv: str = None) -> pd.DataFrame:
    """
    For each K: compare bits the fixed-point partial sum needs vs
    what FP16 sub-sum decoder provides.  Optionally cross-reference
    with the RQ9 PPL sweep to read off the model-accuracy cost.

    Key insight: FP16 always provides exactly 11 significant bits (per element,
    with its own exponent). Bits lost = max(0, bits_needed_worst − 11).

    NOTE: FP16 per-element decode ≠ N=11 per-tensor fixed-point (RQ9 limiter).
    RQ9 uses a shared scale per layer tensor; layer outputs span 23-34 bit dynamic
    range, so N=11 per-tensor zeros out small values. FP16 per-element preserves all
    values at ~0.1% relative error. RQ9 N=10/N=12 bracket where the per-tensor
    fixed-point knee lies; actual FP16 decode quality should be much better.
    """
    rq9_df = None
    if rq9_sweep_csv and os.path.exists(rq9_sweep_csv):
        _raw = pd.read_csv(rq9_sweep_csv)
        rq9_df = _raw[~np.isinf(_raw['n_bits'])].copy()  # drop inf (baseline) rows

    rows = []
    for _, r in analytical_df.iterrows():
        K         = int(r['K'])
        bits_wc   = int(r['bits_exact'])
        bits_lost = max(0, bits_wc - FP16_MANTISSA_BITS)
        fp16_abs_lsb = r['max_psum'] / (2 ** FP16_MANTISSA_BITS)

        # From RQ9: report N=10 and N=12 as bracketing values (N=11 not in sweep).
        # Per-tensor fixed-point knee is at N=14; FP16 per-element should behave
        # significantly better than N=10/N=12 per-tensor.
        ppl_bracket = {}
        if rq9_df is not None:
            for op in rq9_df['op'].unique():
                for n in [10, 12, 14]:
                    sub = rq9_df[(rq9_df['op'] == op) &
                                 (rq9_df['n_bits'].round().astype(int) == n)]
                    if not sub.empty:
                        ppl_bracket[f'ppl_n{n}_{op}'] = round(sub['ppl'].values[0], 3)

        row = {
            'K':                K,
            'bits_needed_wc':   bits_wc,
            'fp16_bits':        FP16_MANTISSA_BITS,
            'bits_lost':        bits_lost,
            'fp16_rel_err_pct': round(FP16_EPS * 100, 4),
            'fp16_abs_lsb':     round(fp16_abs_lsb, 2),
            'verdict':          'LOSSLESS' if bits_lost == 0 else f'-{bits_lost}b',
        }
        row.update(ppl_bracket)
        rows.append(row)

    return pd.DataFrame(rows)


def plot_fp16_equivalence(fp16_df: pd.DataFrame, rq9_sweep_csv: str = None):
    """
    Three-panel figure:
      Left:  bits_needed_wc vs K, with FP16=11b horizontal line
      Middle: bits_lost vs K (how much precision FP16 sacrifices)
      Right: PPL at N=11 from RQ9 (flat horizontal — the model accuracy cost)
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    K_vals = fp16_df['K'].values

    # ── Left: bits needed vs K with FP16 ceiling ────────────────────────────
    ax = axes[0]
    ax.plot(K_vals, fp16_df['bits_needed_wc'], 's-', color='crimson',
            lw=2, label='worst-case bits needed')
    ax.axhline(FP16_MANTISSA_BITS, color='steelblue', lw=2, linestyle='--',
               label=f'FP16 decoder = {FP16_MANTISSA_BITS} bits')
    ax.fill_between(K_vals, FP16_MANTISSA_BITS, fp16_df['bits_needed_wc'],
                    alpha=0.15, color='crimson', label='bits lost to FP16')
    ax.set_xscale('log', base=2)
    ax.set_xlabel("K (rows per subarray)")
    ax.set_ylabel("Accumulator bits")
    ax.set_title("Bits needed (worst case)\nvs FP16 decoder ceiling")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # ── Middle: bits lost vs K ───────────────────────────────────────────────
    ax2 = axes[1]
    bars = ax2.bar(range(len(K_vals)), fp16_df['bits_lost'],
                   color=['seagreen' if b == 0 else 'darkorange' if b <= 3
                          else 'crimson' for b in fp16_df['bits_lost']])
    ax2.axhline(0, color='black', lw=0.8)
    ax2.set_xticks(range(len(K_vals)))
    ax2.set_xticklabels([str(k) for k in K_vals], rotation=45, fontsize=8)
    ax2.set_xlabel("K (rows per subarray)")
    ax2.set_ylabel("Bits lost vs integer accumulator")
    ax2.set_title("Precision lost by FP16 conversion\n"
                  "(green=lossless, orange≤3b, red>3b)")
    ax2.grid(True, axis='y', alpha=0.3)

    # ── Right: PPL impact — read from RQ9 if available ──────────────────────
    ax3 = axes[2]
    rq9_loaded = False
    if rq9_sweep_csv and os.path.exists(rq9_sweep_csv):
        rq9 = pd.read_csv(rq9_sweep_csv)
        colors_op = {'WFP4AINT8': 'steelblue', 'MXFP4_W4A4_B16': 'darkorange'}
        for op in rq9['op'].unique():
            sub = rq9[(rq9['op'] == op) & ~np.isinf(rq9['n_bits'])].copy()
            sub['n_bits'] = sub['n_bits'].astype(int)
            sub = sub.sort_values('n_bits')
            bl_rows = rq9[(rq9['op'] == op) & np.isinf(rq9['n_bits'])]
            baseline = bl_rows['ppl'].values[0] if len(bl_rows) else float('nan')
            ax3.plot(sub['n_bits'], sub['ppl'], 'o-',
                     label=op, color=colors_op.get(op, 'gray'))
            ax3.axhline(baseline, linestyle=':', color=colors_op.get(op, 'gray'), alpha=0.5)
        ax3.axvline(FP16_MANTISSA_BITS, color='purple', lw=2, linestyle='--',
                    label=f'FP16 decoder N=11')
        ax3.set_xlabel("Accumulator N bits (from RQ9)")
        ax3.set_ylabel("PPL")
        ax3.set_title("PPL at each N (RQ9)\nVertical = FP16 operating point")
        ax3.legend(fontsize=8)
        ax3.grid(True, alpha=0.3)
        rq9_loaded = True

    if not rq9_loaded:
        ax3.text(0.5, 0.5,
                 "Run rq9_overflow.py first\nto populate this panel",
                 ha='center', va='center', transform=ax3.transAxes, fontsize=11)
        ax3.set_title("PPL vs N (RQ9 — pending)")

    plt.suptitle("RQ9b Step C — FP16 Sub-sum Decoder Equivalence Analysis",
                 fontsize=11)
    plt.tight_layout()
    fname = os.path.join(OUT_DIR, "rq9b_fp16_equivalence.png")
    plt.savefig(fname, dpi=200)
    print(f"  Saved: {fname}")


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    torch.manual_seed(SEED)

    # (A) Analytical table — always printed, independent of inference
    print("\n" + "="*65)
    print("(A) ANALYTICAL WORST CASE")
    print("="*65)
    for op_name, cfg in OPERATING_POINTS.items():
        a_max_int = cfg['act_max']
        print(f"\n  [{op_name}]  w_max={UQ32_MAX}  a_max={a_max_int}")
        analytical_bits(K_BASE, UQ32_MAX, a_max_int)   # will be used in plot

    ana_fp4fp4 = print_analytical_table(UQ32_MAX, UQ32_MAX)
    print()
    ana_fp4int8 = print_analytical_table(UQ32_MAX, 127)

    ana_fp4fp4.to_csv(os.path.join(OUT_DIR, "analytical_fp4fp4.csv"), index=False)
    ana_fp4int8.to_csv(os.path.join(OUT_DIR, "analytical_fp4int8.csv"), index=False)

    # (B) Empirical — run inference for each operating point
    for op_name, cfg in OPERATING_POINTS.items():
        print(f"\n{'='*65}")
        print(f"(B) EMPIRICAL: {op_name}")
        print(f"{'='*65}")

        model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME, dtype=torch.bfloat16, device_map="auto"
        )
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        model.eval()

        apply_fp4_weights(model, block_size=cfg['block_size'])
        act_q = cfg['act_cls'](model)

        emp_df = run_empirical(model, tokenizer, K_BASE)
        emp_df.to_csv(os.path.join(OUT_DIR, f"empirical_{op_name}.csv"), index=False)

        # Summary
        med = emp_df.groupby('K')['max_abs_psum'].median()
        print(f"\n  Empirical median max |partial sum| per K:")
        ana = analytical_bits(K_BASE, UQ32_MAX, cfg['act_max'])
        for K in K_BASE:
            if K not in med.index:
                continue
            emp_max  = med[K]
            wc_max   = ana[ana['K'] == K]['max_psum'].values[0]
            emp_bits = int(np.ceil(np.log2(emp_max + 1))) + 1
            wc_bits  = ana[ana['K'] == K]['bits_exact'].values[0]
            margin   = wc_bits - emp_bits
            print(f"    K={K:5d}:  empirical {emp_max:10.1f} ({emp_bits:2d}b)  "
                  f"worst-case {wc_max:12.0f} ({wc_bits:2d}b)  "
                  f"margin={margin:+d}b")

        plot_bit_growth(emp_df, ana, op_name, UQ32_MAX, cfg['act_max'])

        act_q.remove()
        del model
        gc.collect()
        torch.cuda.empty_cache()

    # (C) FP16 decoder equivalence — uses analytical FP4×FP4 table + RQ9 results if ready
    print(f"\n{'='*65}")
    print("(C) FP16 SUB-SUM DECODER EQUIVALENCE")
    print(f"{'='*65}")
    rq9_csv = os.path.join(_SCRIPT_DIR, "results", "rq9", "rq9_sweep.csv")
    fp16_df = fp16_decoder_table(ana_fp4fp4, UQ32_MAX, UQ32_MAX, rq9_sweep_csv=rq9_csv)
    fp16_df.to_csv(os.path.join(OUT_DIR, "fp16_decoder_equivalence.csv"), index=False)

    print(f"\n  FP16 decoder: {FP16_MANTISSA_BITS} significant bits, "
          f"rel_error={FP16_EPS*100:.4f}% (constant, independent of K)\n")
    print(f"  {'K':>6}  {'bits_needed':>12}  {'FP16_bits':>10}  "
          f"{'bits_lost':>10}  {'verdict':>10}")
    print("  " + "-"*55)
    for _, r in fp16_df.iterrows():
        print(f"  {int(r.K):>6}  {int(r.bits_needed_wc):>12}  "
              f"{int(r.fp16_bits):>10}  {int(r.bits_lost):>10}  {r.verdict:>10}")

    # Cross-reference with RQ9 PPL if available
    if os.path.exists(rq9_csv):
        rq9 = pd.read_csv(rq9_csv)
        rq9_finite = rq9[~np.isinf(rq9['n_bits'])].copy()
        print(f"\n  RQ9 PPL at N=10/12/14 (brackets around FP16's 11-bit precision):")
        print(f"  (NOTE: RQ9 uses per-tensor fixed-point; FP16 decode is per-element")
        print(f"   → actual FP16 quality should be much better than N=10/12 suggests)")
        print(f"\n  {'Op':<20} {'baseline':>10} {'N=10':>8} {'N=12':>8} {'N=14':>8}")
        print("  " + "-"*52)
        for op in sorted(rq9['op'].unique()):
            base_row = rq9[(rq9['op'] == op) & np.isinf(rq9['n_bits'])]
            base = base_row['ppl'].values[0] if len(base_row) else float('nan')
            ppls = {}
            for n in [10, 12, 14]:
                sub = rq9_finite[(rq9_finite['op'] == op) &
                                 (rq9_finite['n_bits'].round().astype(int) == n)]
                ppls[n] = sub['ppl'].values[0] if len(sub) else float('nan')
            print(f"  {op:<20} {base:>10.3f} {ppls[10]:>8.3f} {ppls[12]:>8.3f} {ppls[14]:>8.3f}")
        print(f"\n  Knee at N=14 (Δ<0.5 PPL). Per-tensor N=11 knee is NOT the FP16 target.")
        print(f"  FP16 per-element decode expected to match baseline (see progress_main.md §3).")
    else:
        print(f"\n  [RQ9 results not yet available — run rq9_overflow.py first]")
        print(f"  → Once RQ9 is done, re-run this script for the full cross-reference.")

    plot_fp16_equivalence(fp16_df, rq9_sweep_csv=rq9_csv)
    print(f"\nAll results saved to {OUT_DIR}/")


if __name__ == "__main__":
    main()
