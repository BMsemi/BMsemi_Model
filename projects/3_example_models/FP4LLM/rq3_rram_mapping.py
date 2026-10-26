"""
RQ3 — RRAM Numerical Mapping
Can E2M1 FP4 be mapped into a signed fixed-point-equivalent RRAM
conductance representation without changing the GEMM result?

This script:
  1. Prints the FP4 → conductance level table
  2. Verifies GEMM equivalence analytically (integer codes × scale²)
  3. Analyses conductance spacing (non-uniformity)
  4. Sweeps inner-product size K to show required conductance precision
  5. Plots conductance levels and spacing

No model needed — pure numerical analysis.
Outputs: results/rq3/
"""
import os, sys, json
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from src.fp4_utils import (FP4_VALUES, FP4_MAGNITUDES, FP4_INT_EQUIV,
                            FP4_MAX, quantize_to_fp4, dequantize_fp4)
from src.rram_model import conductance_level_table, verify_gemm_equivalence

OUT_DIR = "results/rq3"
os.makedirs(OUT_DIR, exist_ok=True)


# ── 1. Conductance level table ─────────────────────────────────────────────────

def print_mapping_table():
    print("\n=== FP4 E2M1 → RRAM Conductance Mapping ===")
    print(f"{'Code':>5}  {'Bits':>6}  {'Value':>6}  {'IntEquiv':>9}  "
          f"{'G/G_max':>8}  {'G_norm (× sign)':>16}")
    print("-" * 65)
    for code in range(16):
        val      = FP4_VALUES[code]
        int_eq   = FP4_INT_EQUIV[code]
        mag_idx  = code % 8
        g_norm   = FP4_MAGNITUDES[mag_idx] / FP4_MAX
        bits_str = format(code, '04b')
        print(f"{code:>5}  {bits_str:>6}  {val:>6.2f}  {int_eq:>9}  {g_norm:>8.4f}  "
              f"  sign={'+'if val>=0 else '-'}  G={g_norm:.4f}")

    info = conductance_level_table()
    print("\n--- Conductance level spacing (positive side) ---")
    print(f"Magnitudes:     {info['magnitudes']}")
    print(f"G_normalized:   {[f'{g:.4f}' for g in info['g_normalized']]}")
    print(f"Integer equivs: {info['int_equiv']}")
    print(f"Spacings (ΔG):  {[f'{s:.4f}' for s in info['spacings']]}")
    print(f"Ratios G[i+1]/G[i]: {[f'{r:.3f}' for r in info['spacing_ratios']]}")
    print(f"\n{info['note']}")
    return info


# ── 2. GEMM equivalence verification ─────────────────────────────────────────

def run_gemm_equivalence_tests():
    print("\n=== GEMM Equivalence: FP4 float == integer_code × scale² ===")
    configs = [
        (64, 128), (256, 512), (1024, 4096), (4096, 4096)
    ]
    results = []
    for N, K in configs:
        r = verify_gemm_equivalence(N, K)
        results.append({'N': N, 'K': K, **r})
        print(f"  N={N:5d}, K={K:5d}  max_err={r['max_abs_error']:.2e}  "
              f"rel_err={r['relative_error']:.2e}  "
              f"s_W={r['scale_W']:.4f}  s_A={r['scale_A']:.4f}  "
              f"s²={r['scale_product_used']:.6f}")

    # Use relative error: float32 accumulation grows with K, so absolute threshold is too tight.
    # rel_err < 1e-3 confirms there is no structural mismatch — only float32 rounding.
    ok = all(r['relative_error'] < 1e-3 for r in results)
    print(f"\n  Equivalence verified: {ok}")
    print("  (residual = float32 accumulation rounding, not a structural error;")
    print("   exact equality holds in exact arithmetic)")
    return results


# ── 3. Required conductance precision vs K ───────────────────────────────────

def conductance_precision_vs_K():
    """
    For a given number of accumulations K, what relative conductance precision
    is needed to keep the MAC error below 1 LSB of the FP4 product?
    """
    Ks = [64, 128, 256, 512, 1024, 2048, 4096, 8192]
    fp4_min_product = 0.5 * 0.5   # smallest non-zero product of two FP4 values
    fp4_max_product = 6.0 * 6.0   # 36.0

    rows = []
    for K in Ks:
        max_sum = K * fp4_max_product          # worst-case partial sum
        lsb     = fp4_min_product              # resolution required
        dynamic_range   = max_sum / lsb        # = 144 × K
        required_bits   = int(np.ceil(np.log2(dynamic_range))) + 1  # +sign
        enob            = np.log2(dynamic_range)
        # Conductance tolerance: 1 LSB error at scale = step between adjacent G levels
        # Smallest G step is 1/12 (between 0.5 and 1 → ΔG=1/12)
        # Required σ_G/G < 1 / (required_bits × dynamic_range^(1/K))
        sigma_req = lsb / max_sum              # relative error budget
        rows.append({
            'K': K,
            'max_partial_sum': max_sum,
            'dynamic_range':   dynamic_range,
            'required_bits':   required_bits,
            'enob':            round(enob, 2),
            'sigma_conductance_budget': sigma_req,
        })
        print(f"  K={K:5d}  DR={dynamic_range:.0f}  bits={required_bits}  "
              f"ENOB={enob:.1f}  σ_budget={sigma_req:.3e}")
    return rows


# ── 4. Plots ──────────────────────────────────────────────────────────────────

def make_plots(cond_info, precision_rows):
    fig = plt.figure(figsize=(15, 5))
    gs  = gridspec.GridSpec(1, 3, figure=fig)

    # (a) Conductance level positions
    ax1 = fig.add_subplot(gs[0])
    mags = cond_info['magnitudes']
    g_norm = cond_info['g_normalized']
    ax1.scatter(range(len(mags)), g_norm, s=80, zorder=5)
    for i, (m, g) in enumerate(zip(mags, g_norm)):
        ax1.annotate(f"{m}", (i, g), textcoords="offset points", xytext=(4,4), fontsize=8)
    ax1.set_xticks(range(len(mags)))
    ax1.set_xticklabels([f"lvl{i}" for i in range(len(mags))], rotation=45)
    ax1.set_ylabel("G / G_max")
    ax1.set_title("(a) FP4 Conductance Levels\n(non-uniform spacing)")

    # (b) Spacing between levels
    ax2 = fig.add_subplot(gs[1])
    spacings = cond_info['spacings']
    ax2.bar(range(len(spacings)), spacings)
    ax2.set_xlabel("Level transition index")
    ax2.set_ylabel("ΔG / G_max")
    ax2.set_title("(b) Conductance Spacing\n(largest gap at levels 6→7)")
    ax2.set_xticks(range(len(spacings)))
    ax2.set_xticklabels([f"{FP4_MAGNITUDES[i]:.1f}→{FP4_MAGNITUDES[i+1]:.1f}"
                          for i in range(len(spacings))], rotation=45, fontsize=7)

    # (c) Required accumulator bits vs K
    ax3 = fig.add_subplot(gs[2])
    Ks   = [r['K'] for r in precision_rows]
    bits = [r['required_bits'] for r in precision_rows]
    enob = [r['enob'] for r in precision_rows]
    ax3.plot(Ks, bits, 'o-', label='Required bits')
    ax3.plot(Ks, enob, 's--', label='ENOB')
    ax3.set_xscale('log')
    ax3.set_xlabel("Inner product dimension K")
    ax3.set_ylabel("Bits")
    ax3.set_title("(c) Accumulator Precision vs. K\n(FP4×FP4 MAC)")
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "rq3_rram_mapping.png"), dpi=200)
    print(f"\nPlot saved: {OUT_DIR}/rq3_rram_mapping.png")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print("=" * 70)
    print("RQ3: RRAM Conductance Mapping + GEMM Equivalence")
    print("=" * 70)

    cond_info    = print_mapping_table()
    gemm_results = run_gemm_equivalence_tests()
    print("\n=== Required Accumulator Precision vs. K ===")
    precision_rows = conductance_precision_vs_K()

    # Save JSON summary
    summary = {
        'conductance_table':    cond_info,
        'gemm_equivalence':     gemm_results,
        'precision_vs_K':       precision_rows,
    }
    # Convert numpy types for JSON serialisation
    def to_py(obj):
        if isinstance(obj, (np.integer,)): return int(obj)
        if isinstance(obj, (np.floating,)): return float(obj)
        if isinstance(obj, (np.ndarray,)): return obj.tolist()
        return obj

    with open(os.path.join(OUT_DIR, "rq3_results.json"), "w") as f:
        json.dump(summary, f, indent=2, default=to_py)

    make_plots(cond_info, precision_rows)
    print(f"\nAll results saved to {OUT_DIR}/")


if __name__ == "__main__":
    main()
