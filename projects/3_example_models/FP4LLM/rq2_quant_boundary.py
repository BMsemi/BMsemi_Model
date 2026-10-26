"""
RQ2 — Quantize/Dequantize Boundary
Where should activations be quantized into FP4, and where must outputs
be restored to higher precision?

Tests five activation quantization sites while keeping weights FP4:
  weight_only  : W4A16  — quantize only weights
  attn_only    : W4A4 attention projections, MLP stays BF16
  mlp_only     : W4A4 MLP projections, attention stays BF16
  all_linear   : W4A4 everywhere
  no_quant     : BF16 baseline (no quantization at all)

Key question: which activation site contributes most to PPL degradation?
Outputs: results/rq2/
"""
import os, sys, gc, json

# CUDA_VISIBLE_DEVICES must be set before torch import, or CUDA ignores it.
os.environ["CUDA_VISIBLE_DEVICES"] = "4"          # single GPU for clean placement
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
sys.path.insert(0, os.path.dirname(__file__))

import torch
import torch.nn as nn
import pandas as pd
import matplotlib.pyplot as plt
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from tqdm import tqdm
from src.fp4_utils import (apply_fp4_weights, quantize_dequantize_fp4,
                            FP4_MAX, WEIGHT_TARGET_NAMES)

# ── Configuration ─────────────────────────────────────────────────────────────

MODEL_NAME  = "Qwen/Qwen2.5-1.5B"  # base model — instruct models give ~100K PPL on raw wikitext
SEQ_LEN     = 512
MAX_SAMPLES = 32
SEED        = 42
OUT_DIR     = "results/rq2"
os.makedirs(OUT_DIR, exist_ok=True)

ATTN_NAMES = ['q_proj', 'k_proj', 'v_proj', 'o_proj', 'query', 'key', 'value', 'out_proj']
MLP_NAMES  = ['gate_proj', 'up_proj', 'down_proj', 'fc1', 'fc2', 'dense']

VARIANTS = {
    "no_quant":   {"quantize_weights": False, "act_site": None},
    "weight_only":{"quantize_weights": True,  "act_site": None},
    "attn_only":  {"quantize_weights": True,  "act_site": "attn"},
    "mlp_only":   {"quantize_weights": True,  "act_site": "mlp"},
    "all_linear": {"quantize_weights": True,  "act_site": "all"},
}


# ── Activation quantization hook ─────────────────────────────────────────────

class SiteQuantizer:
    """Quantize activations only at selected sites (pre-hook on nn.Linear)."""
    def __init__(self, model, site: str):
        self.hooks = []
        for name, module in model.named_modules():
            if not isinstance(module, nn.Linear):
                continue
            is_attn = any(x in name for x in ATTN_NAMES)
            is_mlp  = any(x in name for x in MLP_NAMES)

            should = (site == "all") or (site == "attn" and is_attn) or (site == "mlp" and is_mlp)
            if should:
                h = module.register_forward_pre_hook(self._pre_hook)
                self.hooks.append(h)

    @staticmethod
    def _pre_hook(module, inputs):
        x = inputs[0].float()
        # Per-token scale: each [batch, seq] position gets its own scale.
        # This is the standard W4A4 activation quantization approach.
        amax = x.abs().amax(dim=-1, keepdim=True).clamp(min=1e-38)  # [b, s, 1]
        scale = amax / FP4_MAX                          # [b, s, 1]
        x_norm = x / scale                              # normalize to [-6, 6] per token
        x_norm_q, _ = quantize_dequantize_fp4(x_norm, scale=1.0)   # round-trip at scale 1
        x_q = x_norm_q * scale                         # restore original magnitude
        return (x_q.to(inputs[0].dtype),) + inputs[1:]

    def remove(self):
        for h in self.hooks:
            h.remove()
        self.hooks = []


# ── PPL evaluation (wikitext-2 non-overlapping chunks) ───────────────────────

def evaluate_ppl(model, tokenizer, first_call: bool = False) -> float:
    ds   = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    text = "\n\n".join(ds["text"])
    # Truncate text before tokenizing — feeding 299K tokens to the tokenizer
    # triggers Qwen's position-embedding guard and corrupts predictions.
    # ~6 chars/token is a safe over-estimate; we slice to the exact token count after.
    text = text[:MAX_SAMPLES * SEQ_LEN * 6]
    enc  = tokenizer(text, add_special_tokens=False, return_tensors="pt")
    ids  = enc.input_ids[:, :MAX_SAMPLES * SEQ_LEN]

    # Use the device of the model's first parameter so it works for
    # single-GPU, multi-GPU (device_map), and CPU setups.
    model_device = next(model.parameters()).device
    ids = ids.to(model_device)

    n_chunks = ids.size(1) // SEQ_LEN
    print(f"   tokens={ids.size(1)}, chunks={n_chunks}, model_device={model_device}")

    if first_call:
        # Sanity-check: decode first 8 tokens to verify tokenizer is working.
        sample = tokenizer.convert_ids_to_tokens(ids[0, :8].tolist())
        print(f"   first 8 tokens: {sample}")

    nlls = []
    with torch.no_grad():
        for i in range(MAX_SAMPLES):
            s, e = i * SEQ_LEN, (i + 1) * SEQ_LEN
            if e > ids.size(1):
                break
            chunk  = ids[:, s:e]
            labels = chunk.clone()
            out    = model(chunk, labels=labels)
            nlls.append(out.loss.item())

    ppl = torch.exp(torch.tensor(nlls).mean()).item()
    return ppl


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    torch.manual_seed(SEED)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    results = []

    for i, (vname, cfg) in enumerate(tqdm(VARIANTS.items(), desc="Variants")):
        print(f"\n--- {vname} ---")
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME, dtype=torch.bfloat16, device_map="auto"
        )
        model.eval()

        # Diagnostic: confirm dtype and device after load.
        p = next(model.parameters())
        print(f"   model dtype={p.dtype}, device={p.device}")

        act_q = None
        if cfg["quantize_weights"]:
            apply_fp4_weights(model)
        if cfg["act_site"] is not None:
            act_q = SiteQuantizer(model, cfg["act_site"])

        ppl = evaluate_ppl(model, tokenizer, first_call=(i == 0))
        print(f"   PPL = {ppl:.3f}")

        results.append({
            "variant":           vname,
            "quantize_weights":  cfg["quantize_weights"],
            "act_site":          str(cfg["act_site"]),
            "ppl":               ppl,
        })

        if act_q:
            act_q.remove()
        del model
        gc.collect()
        torch.cuda.empty_cache()

    # ── Save + plot ───────────────────────────────────────────────────────────
    df = pd.DataFrame(results)
    df.to_csv(os.path.join(OUT_DIR, "rq2_results.csv"), index=False)

    baseline_ppl = df.loc[df['variant'] == 'no_quant', 'ppl'].values[0]
    df['ppl_delta'] = df['ppl'] - baseline_ppl

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    axes[0].bar(df['variant'], df['ppl'])
    axes[0].axhline(baseline_ppl, color='red', linestyle='--', label='BF16 baseline')
    axes[0].set_ylabel("Perplexity (wikitext-2)")
    axes[0].set_title("Absolute PPL by Quantization Site")
    axes[0].tick_params(axis='x', rotation=30)
    axes[0].legend()

    axes[1].bar(df['variant'], df['ppl_delta'], color='orange')
    axes[1].axhline(0, color='black', linestyle='-')
    axes[1].set_ylabel("ΔPPL vs. BF16 baseline")
    axes[1].set_title("PPL Penalty by Activation Quantization Site")
    axes[1].tick_params(axis='x', rotation=30)

    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "rq2_boundary.png"), dpi=200)
    print("\n=== RESULTS ===")
    print(df[['variant', 'ppl', 'ppl_delta']].to_string(index=False))
    print(f"\nSaved to {OUT_DIR}/")


if __name__ == "__main__":
    main()
