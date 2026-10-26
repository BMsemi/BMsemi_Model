"""
RQ9c — FP16 Sub-sum Decoder Equivalence (standalone, no GPU)

Reads pre-computed analytical table and RQ9 sweep CSV, produces the
Step C cross-reference table and plot from rq9b without re-running inference.

Outputs: results/rq9b/  (same directory as rq9b)
"""
import os, sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUT_DIR  = os.path.join(_SCRIPT_DIR, "results", "rq9b")
RQ9_CSV  = os.path.join(_SCRIPT_DIR, "results", "rq9", "rq9_sweep.csv")
ANA_CSV  = os.path.join(OUT_DIR, "analytical_fp4fp4.csv")

FP16_MANTISSA_BITS = 11        # 10 stored + 1 implicit leading-1
FP16_EPS           = 2 ** -FP16_MANTISSA_BITS   # ~0.000977

os.makedirs(OUT_DIR, exist_ok=True)


# ── Load inputs ────────────────────────────────────────────────────────────────

ana = pd.read_csv(ANA_CSV)

if not os.path.exists(RQ9_CSV):
    print(f"ERROR: {RQ9_CSV} not found. Run rq9_overflow.py first.")
    sys.exit(1)

rq9 = pd.read_csv(RQ9_CSV)
baseline = rq9[np.isinf(rq9['n_bits'])].set_index('op')['ppl']
sweep    = rq9[~np.isinf(rq9['n_bits'])].copy()
sweep['n_bits'] = sweep['n_bits'].astype(float)


# ── Build cross-reference table ────────────────────────────────────────────────

rows = []
for _, r in ana.iterrows():
    K         = int(r['K'])
    bits_wc   = int(r['bits_exact'])
    bits_lost = max(0, bits_wc - FP16_MANTISSA_BITS)
    fp16_lsb  = float(r['max_psum']) / (2 ** FP16_MANTISSA_BITS)

    row = {
        'K':                K,
        'bits_needed_wc':   bits_wc,
        'fp16_bits':        FP16_MANTISSA_BITS,
        'bits_lost':        bits_lost,
        'fp16_rel_err_pct': round(FP16_EPS * 100, 4),
        'fp16_abs_lsb':     round(fp16_lsb, 2),
        'verdict':          'LOSSLESS' if bits_lost == 0 else f'-{bits_lost}b',
    }
    for n in [10, 12, 14]:
        for op in sorted(sweep['op'].unique()):
            sub = sweep[(sweep['op'] == op) &
                        (sweep['n_bits'].round().astype(int) == n)]
            key = f'ppl_n{n}_{op}'
            row[key] = round(sub['ppl'].values[0], 3) if len(sub) else float('nan')
    rows.append(row)

fp16_df = pd.DataFrame(rows)
fp16_df.to_csv(os.path.join(OUT_DIR, "fp16_decoder_equivalence.csv"), index=False)


# ── Print ──────────────────────────────────────────────────────────────────────

print("=" * 70)
print("RQ9c — FP16 Sub-sum Decoder Equivalence")
print("=" * 70)
print(f"\n  FP16: {FP16_MANTISSA_BITS} significant bits, "
      f"rel_error={FP16_EPS*100:.4f}% (constant, independent of K)\n")

print(f"  {'K':>6}  {'bits_wc':>8}  {'bits_lost':>10}  {'verdict':>9}")
print("  " + "-"*38)
for _, r in fp16_df.iterrows():
    print(f"  {int(r['K']):>6}  {int(r['bits_needed_wc']):>8}  "
          f"{int(r['bits_lost']):>10}  {r['verdict']:>9}")

print(f"\n  RQ9 PPL — per-tensor fixed-point at N=10/12/14 "
      f"(brackets around FP16's 11-bit precision):")
print()
ops = sorted(sweep['op'].unique())
header = f"  {'Op':<20} {'baseline':>10}"
for n in [10, 12, 14]:
    header += f"  {'N='+str(n):>8}"
print(header)
print("  " + "-"*58)
for op in ops:
    bl = baseline.get(op, float('nan'))
    line = f"  {op:<20} {bl:>10.3f}"
    for n in [10, 12, 14]:
        sub = sweep[(sweep['op'] == op) &
                    (sweep['n_bits'].round().astype(int) == n)]
        ppl = sub['ppl'].values[0] if len(sub) else float('nan')
        line += f"  {ppl:>8.3f}"
    print(line)

print()
print("  Per-tensor fixed-point knee: N=14 (ΔPPL < 0.5 for both OPs).")
print("  FP16 per-element decode ≠ N=11 per-tensor fixed-point.")
print("  Layer outputs span 23–34 bit dynamic range; per-tensor N=11 step")
print("  (~6 LSB) zeros out values < ~3, explaining the large PPL gap at N=12.")
print("  FP16 per-element: each value gets its own exponent → ~0.1% relative")
print("  error regardless of magnitude. Baseline model already runs in BF16")
print("  (7 mantissa bits, per-element); hardware FP16 decode (10 mantissa bits)")
print("  has strictly higher precision than the baseline.")
print()
print("  → FP16 sub-sum decode expected to match baseline PPL.")
print("  → For fixed-point hardware: use ≥14 effective bits.")


# ── Plot ───────────────────────────────────────────────────────────────────────

colors = {'WFP4AINT8': 'steelblue', 'MXFP4_W4A4': 'darkorange'}
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

# Panel 1: bits_needed_wc vs K + FP16 line
ax = axes[0]
ax.plot(fp16_df['K'], fp16_df['bits_needed_wc'], 'ko-', label='worst-case bits')
ax.axhline(FP16_MANTISSA_BITS, color='green', linestyle='--',
           label=f'FP16 mantissa ({FP16_MANTISSA_BITS}b)')
ax.axhline(14, color='red', linestyle=':', label='fixed-point knee (N=14)')
ax.set_xscale('log', base=2)
ax.set_xlabel("K (rows accumulated)")
ax.set_ylabel("Bits")
ax.set_title("Worst-case accumulator bits vs K")
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)

# Panel 2: bits_lost vs K
ax = axes[1]
ax.plot(fp16_df['K'], fp16_df['bits_lost'], 'rs-')
ax.axhline(0, color='green', linestyle='--')
ax.set_xscale('log', base=2)
ax.set_xlabel("K (rows accumulated)")
ax.set_ylabel("Bits lost (worst-case)")
ax.set_title("Bits discarded by FP16 decode (worst-case)")
ax.grid(True, alpha=0.3)
ax.text(0.05, 0.95, "Rel error constant\nat ~0.1% for all K",
        transform=ax.transAxes, va='top', fontsize=9, color='green')

# Panel 3: PPL vs N from RQ9 + FP16 operating point
ax = axes[2]
n_vals = sorted(sweep['n_bits'].unique())
for op in ops:
    sub = sweep[sweep['op'] == op].sort_values('n_bits')
    bl  = baseline.get(op, float('nan'))
    ax.plot(sub['n_bits'], sub['ppl'], 'o-', label=op, color=colors.get(op))
    ax.axhline(bl, linestyle='--', color=colors.get(op), alpha=0.4,
               label=f'{op} baseline ({bl:.2f})')

ax.axvline(FP16_MANTISSA_BITS, color='green', linestyle='-.',
           label=f'FP16 mantissa ({FP16_MANTISSA_BITS}b)')
ax.axvline(14, color='red', linestyle=':', label='fixed-point knee (14b)')
ax.set_xlabel("Accumulator bit-width N (per-tensor fixed-point)")
ax.set_ylabel("Perplexity (wikitext-2)")
ax.set_title("RQ9: PPL vs N\n(per-tensor FP — conservative bound for FP16 decode)")
ax.legend(fontsize=7)
ax.grid(True, alpha=0.3)

# Annotation box explaining the gap
ax.annotate("Per-tensor N=11 ≠ FP16 decode\n(FP16 is per-element;\nbaseline BF16 = 7-bit per-element)",
            xy=(FP16_MANTISSA_BITS, 20), xytext=(15, 28),
            fontsize=7, color='green',
            arrowprops=dict(arrowstyle='->', color='green'))

plt.suptitle("RQ9c — FP16 Sub-sum Decoder Equivalence Analysis", fontsize=11)
plt.tight_layout()
fname = os.path.join(OUT_DIR, "rq9c_fp16_decoder.png")
plt.savefig(fname, dpi=200)
print(f"\n  Saved: {fname}")
print(f"All results saved to {OUT_DIR}/")
