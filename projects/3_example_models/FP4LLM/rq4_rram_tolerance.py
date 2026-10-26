"""
RQ4 — RRAM Variation Tolerance
How much conductance error, read noise, stuck-cell error, and
sign-path mismatch can FP4 LLM inference tolerate?

Protocol:
  1. Quantize all weights to FP4 (once, stored as float)
  2. At each forward pass, re-inject RRAM noise → simulates read noise
  3. Sweep σ_rel, p_stuck, p_sign_flip independently
  4. Measure wikitext-2 PPL

Outputs: results/rq4/
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
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from tqdm import tqdm
from src.fp4_utils import apply_fp4_weights
from src.rram_model import add_conductance_noise, WEIGHT_TARGET_NAMES

# ── Configuration ─────────────────────────────────────────────────────────────

MODEL_NAME  = "Qwen/Qwen2.5-1.5B"    # base model — instruct models give ~100K PPL on raw wikitext
SEQ_LEN     = 512
MAX_SAMPLES = 32
SEED        = 42
OUT_DIR     = "results/rq4"
os.makedirs(OUT_DIR, exist_ok=True)

# Sweep parameters — three independent experiments
SIGMA_REL_SWEEP   = [0.0, 0.001, 0.005, 0.01, 0.05, 0.1, 0.2]
P_STUCK_SWEEP     = [0.0, 0.0001, 0.001, 0.005, 0.01, 0.05]
P_SIGN_FLIP_SWEEP = [0.0, 0.0001, 0.0005, 0.001, 0.005, 0.01]


# ── RRAM noise hook (applied at each forward pass) ────────────────────────────

class RRAMReadNoiseHook:
    """
    Injects RRAM read noise into FP4-quantized weights at each forward pass.
    More realistic than programming noise: each read is independently noisy.
    """
    def __init__(self, model, sigma_rel, p_stuck, p_sign_flip):
        self.sigma_rel   = sigma_rel
        self.p_stuck     = p_stuck
        self.p_sign_flip = p_sign_flip
        self.hooks       = []

        for name, module in model.named_modules():
            if not isinstance(module, nn.Linear):
                continue
            if not any(t in name for t in WEIGHT_TARGET_NAMES):
                continue
            # Store clean FP4 weight so we can restore after each call
            module._fp4_weight_clean = module.weight.data.clone()
            h = module.register_forward_pre_hook(self._make_hook())
            self.hooks.append((module, h))

    def _make_hook(self):
        def hook(module, inputs):
            w_noisy = add_conductance_noise(
                module._fp4_weight_clean,
                sigma_rel=self.sigma_rel,
                p_stuck=self.p_stuck,
                p_sign_flip=self.p_sign_flip,
            )
            module.weight = nn.Parameter(w_noisy, requires_grad=False)
            return inputs
        return hook

    def remove(self):
        for module, h in self.hooks:
            h.remove()
            if hasattr(module, '_fp4_weight_clean'):
                module.weight = nn.Parameter(module._fp4_weight_clean, requires_grad=False)
                del module._fp4_weight_clean
        self.hooks = []


# ── PPL evaluator ─────────────────────────────────────────────────────────────

def evaluate_ppl(model, tokenizer) -> float:
    ds   = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    text = "\n\n".join(ds["text"])
    text = text[:MAX_SAMPLES * SEQ_LEN * 6]   # truncate before tokenizing
    enc  = tokenizer(text, add_special_tokens=False, return_tensors="pt")
    ids  = enc.input_ids[:, :MAX_SAMPLES * SEQ_LEN]

    device = next(model.parameters()).device
    ids    = ids.to(device)
    nlls   = []
    with torch.no_grad():
        for i in range(MAX_SAMPLES):
            s, e = i * SEQ_LEN, (i + 1) * SEQ_LEN
            if e > ids.size(1):
                break
            chunk = ids[:, s:e]
            out   = model(chunk, labels=chunk.clone())
            nlls.append(out.loss.item())

    return torch.exp(torch.tensor(nlls).mean()).item()


# ── Sweep runner ──────────────────────────────────────────────────────────────

CHECKPOINT_CSV = os.path.join(OUT_DIR, "rq4_results.csv")


def load_checkpoint() -> list:
    """Return already-completed rows from a prior partial run."""
    if os.path.exists(CHECKPOINT_CSV):
        return pd.read_csv(CHECKPOINT_CSV).to_dict('records')
    return []


def save_checkpoint(results: list):
    pd.DataFrame(results).to_csv(CHECKPOINT_CSV, index=False)


def run_sweep(model, tokenizer, sweep_param: str, sweep_values: list,
              done: set) -> list:
    """Run one sweep, skipping values already in `done`. Checkpoints after each point."""
    results = []
    for val in tqdm(sweep_values, desc=sweep_param):
        if (sweep_param, val) in done:
            print(f"  {sweep_param}={val:.0e}  (skipped — already done)")
            continue
        kwargs = {'sigma_rel': 0.0, 'p_stuck': 0.0, 'p_sign_flip': 0.0}
        kwargs[sweep_param] = val

        hook = RRAMReadNoiseHook(model, **kwargs)
        ppl  = evaluate_ppl(model, tokenizer)
        hook.remove()

        row = {'sweep': sweep_param, 'value': val, 'ppl': ppl}
        results.append(row)
        print(f"  {sweep_param}={val:.0e}  PPL={ppl:.3f}")
    return results


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    torch.manual_seed(SEED)
    print(f"Loading {MODEL_NAME}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model     = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, dtype=torch.bfloat16, device_map="auto"
    )
    model.eval()

    # Pre-quantize weights once
    print("Quantizing weights to FP4...")
    apply_fp4_weights(model)

    # Resume from checkpoint if available
    all_results = load_checkpoint()
    done = {(r['sweep'], r['value']) for r in all_results}
    if done:
        print(f"  Resuming: {len(done)} data points already complete.")

    sweeps = [
        ("Sweep 1: Relative conductance noise", 'sigma_rel',   SIGMA_REL_SWEEP),
        ("Sweep 2: Stuck-cell probability",      'p_stuck',     P_STUCK_SWEEP),
        ("Sweep 3: Sign path mismatch",          'p_sign_flip', P_SIGN_FLIP_SWEEP),
    ]
    for label, param, values in sweeps:
        print(f"\n--- {label} ---")
        new_rows = run_sweep(model, tokenizer, param, values, done)
        all_results += new_rows
        done.update((r['sweep'], r['value']) for r in new_rows)
        save_checkpoint(all_results)   # flush after each sweep

    # Final save (full)
    df = pd.DataFrame(all_results)
    df.to_csv(CHECKPOINT_CSV, index=False)

    # Determine BF16 baseline PPL (sigma=0 case)
    baseline = df[(df['sweep'] == 'sigma_rel') & (df['value'] == 0.0)]['ppl'].values
    baseline_ppl = baseline[0] if len(baseline) else float('nan')

    # ── Tolerance plot ────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    sweep_defs = [
        ('sigma_rel',   SIGMA_REL_SWEEP,   'Relative conductance noise σ_rel',   axes[0]),
        ('p_stuck',     P_STUCK_SWEEP,     'Stuck-cell probability p_stuck',      axes[1]),
        ('p_sign_flip', P_SIGN_FLIP_SWEEP, 'Sign-path mismatch p_sign_flip',      axes[2]),
    ]
    for param, vals, label, ax in sweep_defs:
        sub = df[df['sweep'] == param].sort_values('value')
        ax.plot(sub['value'], sub['ppl'], 'o-', markersize=5)
        ax.axhline(baseline_ppl, color='red', linestyle='--', label=f'FP4 clean ({baseline_ppl:.2f})')
        ax.set_xscale('log')
        ax.set_xlabel(label)
        ax.set_ylabel("Perplexity (wikitext-2)")
        ax.set_title(f"RQ4 — {param}")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    plt.suptitle("RQ4: RRAM Variation Tolerance for FP4 LLM Inference", fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "rq4_tolerance.png"), dpi=200)

    print("\n=== RESULTS ===")
    print(df.to_string(index=False))
    print(f"\nSaved to {OUT_DIR}/")


if __name__ == "__main__":
    main()
