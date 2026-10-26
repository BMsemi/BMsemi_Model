"""
RQ10 — Subarray Size K Sweep

Question: what subarray size K (rows accumulated per sub-sum) can the RRAM
array use without degrading model accuracy?

Hardware trade-off (for a layer with in_features = d):
  P = ceil(d / K)   subarrays needed to cover the full inner dimension
  ↑ K  →  larger array, wider ADC (more bits), shallower FP16 tree (fewer adds)
  ↓ K  →  smaller array, narrower ADC, deeper FP16 tree, more scale factors

Block-alignment constraint (MXFP4_W4A4_B16 only):
  With block-16 FP4 scaling, each block of 16 elements shares one scale pair
  (s_w,b × s_a,b). K MUST be a multiple of 16; otherwise elements from different
  blocks with different scales would be mixed in one integer accumulator D_b and
  a single scale cannot be applied correctly:
    Wrong (K=24): mixes block 0 (elements 0–15) and block 1 (elements 16–23) in D
    Correct (K=16): D_b = Σ_{i} n_w,i × n_a,i for elements within one block
  K_SWEEP therefore starts at 16 for MXFP4_W4A4_B16. WFP4AINT8 uses per-channel
  (not block-16) weights + per-token INT8 activations, so any K is valid for it.

Simulation:
  For each K, intercept every nn.Linear (after FP4/INT8 activation quantization
  pre-hooks), replace the matmul with a partitioned version:
    1. Split inner dimension into chunks of K
    2. Compute each partial sum in float32
    3. Round partial sum to FP16  ← simulates sub-sum decoder
    4. Accumulate P chunks in float32  ← simulates FP16 tree reduction
  This gives the combined effect of FP16 decode precision + tree accumulation error.

Operating points:
  MXFP4_W4A4_B16  — FP4 weights (block-16) + FP4 activations (block-16); K must be ≥16 and multiple of 16
  WFP4AINT8       — FP4 weights (per-channel) + INT8 activations (per-token); any K valid

Outputs: results/rq10/
"""
import os, sys, gc
os.environ["CUDA_VISIBLE_DEVICES"] = "4"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from tqdm import tqdm

from src.fp4_utils import (apply_fp4_weights,
                           INT8ActivationQuantizer, NVFP4ActivationQuantizer)

# ── Configuration ──────────────────────────────────────────────────────────────

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_NAME  = "Qwen/Qwen2.5-1.5B"
SEQ_LEN     = 512
MAX_SAMPLES = 32
SEED        = 42
OUT_DIR     = os.path.join(_SCRIPT_DIR, "results", "rq10")
os.makedirs(OUT_DIR, exist_ok=True)

# K values to sweep (rows per subarray).
# Must be multiples of 16 for MXFP4_W4A4_B16 (block-alignment constraint — see docstring).
# K=1/2/4/8 are sub-block and not physically realizable with block-16 scaling.
# K=inf (baseline) = no partitioning; handled separately in run_k_sweep().
K_SWEEP = [16, 32, 64, 128, 256, 512, 1024, 2048, 4096]

OPERATING_POINTS = {
    "MXFP4_W4A4_B16": dict(block_size=16,   act_cls=NVFP4ActivationQuantizer),
    "WFP4AINT8":  dict(block_size=None, act_cls=INT8ActivationQuantizer),
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


# ── Subarray partitioned GEMM hook ────────────────────────────────────────────

class SubarraySplitHook:
    """
    Replaces each nn.Linear forward with a K-partitioned version:
      - Splits W's input dimension into ceil(in_f / K) chunks of size K
      - Computes partial GEMM per chunk (float32)
      - Rounds each partial sum to FP16  (sub-sum decoder simulation)
      - Accumulates in float32           (FP16 tree simulation)

    Precondition: FP4/INT8 quantization pre-hooks already fired, so
    inputs[0] and module.weight already carry quantized values.
    """
    def __init__(self, model, k_subarray: int):
        self.k = k_subarray
        self.hooks = []
        for _, module in model.named_modules():
            if isinstance(module, nn.Linear):
                h = module.register_forward_hook(self._make_hook(k_subarray))
                self.hooks.append(h)

    @staticmethod
    def _make_hook(k: int):
        def hook(module, inputs, output):
            x  = inputs[0].float()       # [*, in_f] — already quantized
            w  = module.weight.float()   # [out_f, in_f] — already FP4-quantized
            in_f = w.shape[1]

            if k >= in_f:
                # No partitioning: recompute full dot in fp32 (bias-free),
                # round to FP16, then add bias once.  Using `output` directly
                # would double-count any bias already baked into the forward.
                psum = x @ w.T
                result = psum.half().float()
                if module.bias is not None:
                    result = result + module.bias.float()
                return result.to(output.dtype)

            n_chunks = (in_f + k - 1) // k
            acc = torch.zeros(*x.shape[:-1], w.shape[0],
                              device=x.device, dtype=torch.float32)
            for i in range(n_chunks):
                s = i * k
                e = min(s + k, in_f)
                psum = x[..., s:e] @ w[:, s:e].T   # float32 partial sum
                acc  = acc + psum.half().float()    # decode to FP16, accumulate

            if module.bias is not None:
                acc = acc + module.bias.float()
            return acc.to(output.dtype)
        return hook

    def remove(self):
        for h in self.hooks:
            h.remove()
        self.hooks = []


# ── Analytical companion ───────────────────────────────────────────────────────

def analytical_row(K: int, in_features_list: list) -> dict:
    """
    For a given K and a list of layer in_features, compute:
      - required ADC bits (worst-case FP4×FP4)
      - P = in_f / K for each layer
      - FP16 tree error bound O(log2(P) * 2^-11)
    """
    wc_bits = int(np.ceil(np.log2(K * 24 * 24 + 1))) + 1  # FP4×FP4 worst case
    results = {'K': K, 'adc_bits_wc_fp4fp4': wc_bits}
    for d in sorted(set(in_features_list)):
        P = int(np.ceil(d / K))
        tree_err_pct = np.log2(max(P, 1)) * (2**-11) * 100
        results[f'P_d{d}'] = P
        results[f'tree_err_pct_d{d}'] = round(tree_err_pct, 4)
    return results


# ── Sweep ──────────────────────────────────────────────────────────────────────

def run_k_sweep(model, tokenizer, op_name: str) -> list:
    results = []

    # Baseline: quantized model, no subarray partitioning
    baseline_ppl = evaluate_ppl(model, tokenizer)
    results.append({'op': op_name, 'K': 'inf', 'ppl': baseline_ppl,
                    'delta_ppl': 0.0})
    print(f"  [{op_name}] baseline (no partitioning) PPL = {baseline_ppl:.3f}")

    for K in tqdm(K_SWEEP, desc=f"  [{op_name}] K sweep"):
        hook = SubarraySplitHook(model, K)
        ppl  = evaluate_ppl(model, tokenizer)
        hook.remove()
        delta = ppl - baseline_ppl
        results.append({'op': op_name, 'K': K, 'ppl': ppl, 'delta_ppl': delta})
        print(f"  [{op_name}]  K={K:5d}  PPL={ppl:.3f}  ΔPPL={delta:+.3f}")

    return results


# ── Plot ────────────────────────────────────────────────────────────────────────

def plot_results(sweep_df: pd.DataFrame, ana_df: pd.DataFrame,
                 in_features_list: list):
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    colors = {'MXFP4_W4A4_B16': 'darkorange', 'WFP4AINT8': 'steelblue'}

    # ── Panel 1: PPL vs K ───────────────────────────────────────────────────
    ax = axes[0]
    finite = sweep_df[sweep_df['K'] != 'inf'].copy()
    finite['K'] = finite['K'].astype(int)
    for op in finite['op'].unique():
        sub = finite[finite['op'] == op].sort_values('K')
        bl  = float(sweep_df[(sweep_df['op'] == op) &
                              (sweep_df['K'] == 'inf')]['ppl'].values[0])
        ax.plot(sub['K'], sub['ppl'], 'o-', label=op, color=colors.get(op))
        ax.axhline(bl, linestyle='--', alpha=0.4, color=colors.get(op),
                   label=f'{op} baseline ({bl:.2f})')
    ax.set_xscale('log', base=2)
    ax.set_xlabel("K (rows per subarray)")
    ax.set_ylabel("Perplexity (wikitext-2)")
    ax.set_title("PPL vs Subarray Size K")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # ── Panel 2: ΔPPL vs K ─────────────────────────────────────────────────
    ax = axes[1]
    for op in finite['op'].unique():
        sub = finite[finite['op'] == op].sort_values('K')
        ax.plot(sub['K'], sub['delta_ppl'], 'o-', label=op, color=colors.get(op))
    ax.axhline(0, color='gray', linestyle=':')
    ax.axhline(0.5, color='red', linestyle='--', alpha=0.5, label='±0.5 PPL threshold')
    ax.axhline(-0.5, color='red', linestyle='--', alpha=0.5)
    ax.set_xscale('log', base=2)
    ax.set_xlabel("K (rows per subarray)")
    ax.set_ylabel("ΔPPL vs no-partitioning baseline")
    ax.set_title("PPL Degradation from K Partitioning")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # ── Panel 3: ADC bits + P vs K ─────────────────────────────────────────
    ax = axes[2]
    ax2 = ax.twinx()
    K_vals = ana_df['K'].values
    ax.plot(K_vals, ana_df['adc_bits_wc_fp4fp4'], 'ks-', label='ADC bits (worst-case)')
    ax.axhline(14, color='red', linestyle=':', label='per-tensor knee N=14')
    ax.set_xscale('log', base=2)
    ax.set_xlabel("K (rows per subarray)")
    ax.set_ylabel("ADC bits (worst-case FP4×FP4)")
    ax.set_title("Hardware Cost vs K")
    ax.legend(loc='upper left', fontsize=8)
    ax.grid(True, alpha=0.3)

    for d in sorted(set(in_features_list)):
        col = f'P_d{d}'
        if col in ana_df.columns:
            ax2.plot(K_vals, ana_df[col], '--', alpha=0.6,
                     label=f'P (d={d})')
    ax2.set_ylabel("Number of subarrays P")
    ax2.set_yscale('log', base=2)
    ax2.legend(loc='upper right', fontsize=8)

    plt.suptitle(f"RQ10 — Subarray K Sweep ({MODEL_NAME.split('/')[-1]})",
                 fontsize=11)
    plt.tight_layout()
    fname = os.path.join(OUT_DIR, "rq10_k_sweep.png")
    plt.savefig(fname, dpi=200)
    print(f"  Saved: {fname}")


# ── Main ────────────────────────────────────────────────────────────────────────

def main():
    torch.manual_seed(SEED)

    # Analytical companion table
    # Qwen2.5-1.5B: d_model=1536, d_ffn intermediate=8960
    in_features_list = [1536, 8960]
    ana_rows = [analytical_row(K, in_features_list) for K in K_SWEEP]
    ana_df = pd.DataFrame(ana_rows)
    ana_df.to_csv(os.path.join(OUT_DIR, "rq10_analytical.csv"), index=False)

    print("\n=== Analytical: ADC bits + P for Qwen2.5-1.5B ===")
    print(f"  {'K':>6}  {'ADC bits':>10}  {'P(d=1536)':>12}  {'P(d=8960)':>12}  "
          f"{'tree_err%(d=1536)':>18}")
    for _, r in ana_df.iterrows():
        print(f"  {int(r.K):>6}  {int(r.adc_bits_wc_fp4fp4):>10}  "
              f"{int(r.P_d1536):>12}  {int(r.P_d8960):>12}  "
              f"{r.tree_err_pct_d1536:>18.4f}%")

    # Sweep
    all_results = []
    for op_name, cfg in OPERATING_POINTS.items():
        print(f"\n{'='*60}")
        print(f"Operating point: {op_name}")
        print(f"{'='*60}")

        model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME, dtype=torch.bfloat16, device_map="auto"
        )
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        model.eval()

        apply_fp4_weights(model, block_size=cfg['block_size'])
        act_q = cfg['act_cls'](model)

        sweep = run_k_sweep(model, tokenizer, op_name)
        all_results.extend(sweep)

        act_q.remove()
        del model
        gc.collect()
        torch.cuda.empty_cache()

    # Save and print summary
    sweep_df = pd.DataFrame(all_results)
    sweep_df.to_csv(os.path.join(OUT_DIR, "rq10_sweep.csv"), index=False)

    print("\n=== K SWEEP SUMMARY ===")
    finite = sweep_df[sweep_df['K'] != 'inf'].copy()
    finite['K'] = finite['K'].astype(int)
    pivot_ppl   = finite.pivot(index='K', columns='op', values='ppl')
    pivot_delta = finite.pivot(index='K', columns='op', values='delta_ppl')
    print("\nPPL:")
    print(pivot_ppl.to_string())
    print("\nΔPPL vs no-partitioning baseline:")
    print(pivot_delta.to_string())

    plot_results(sweep_df, ana_df, in_features_list)
    print(f"\nAll results saved to {OUT_DIR}/")


if __name__ == "__main__":
    main()
