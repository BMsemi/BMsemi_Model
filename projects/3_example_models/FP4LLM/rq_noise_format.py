"""
RQ-Noise-Format — Quantization Format vs RRAM Noise Tolerance
=============================================================
Question: Does storing weights in UQ3.2 (5-bit FP4) instead of full precision
increase the effective RRAM noise margin?

Two fundamentally different noise models:

  bf16_noise   — continuous perturbation in value space (rq_noise.py model)
                   w_eff = w * (1 + ε)
                 Upper bound on damage; noise acts continuously across all values.

  fmt_decode   — quantize → noise → re-decode  (correct physical RRAM model)
                   w_q     = round_to_grid(w,  bits, scale)   # stored conductance target
                   w_noisy = w_q * (1 + ε)                   # conductance drifts
                   w_eff   = round_to_grid(w_noisy, bits, scale)  # ADC decodes nearest bin
                 Noise only matters when it crosses a bin boundary. Below the crossing
                 threshold the decoded weight is IDENTICAL to the stored quantized value.

Formats compared:
  bf16   — continuous (no grid)
  uq32   — 5-bit symmetric FXP, per-block-16 scale  (UQ3.2, our proposal)
  int4   — 4-bit symmetric FXP, per-block-16 scale
  int8   — 8-bit symmetric FXP, per-block-16 scale

Predicted shape:
  bf16:  monotone degradation from σ=0.01
  uq32:  flat until σ≈0.10-0.15, then cliff  (bin-crossing threshold)
  int4:  flat until σ≈0.06, cliff at lower PPL floor (quantization error)
  int8:  flat until σ≈0.004 (bins very narrow), then degrades like bf16

σ sweep: 0.0, 0.01, 0.03, 0.05, 0.10, 0.15, 0.20
  σ=0.0 gives quantization-only PPL (no noise), the floor for each format.

Outputs: results/rq_noise_format/
"""

import os, sys, gc, json
os.environ["CUDA_VISIBLE_DEVICES"] = "4"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_LIST = [
    "Qwen/Qwen3-1.7B",
    "Qwen/Qwen3-4B",
    "Qwen/Qwen3-8B",
]

RRAM_PROJ_NAMES = {"q_proj", "k_proj", "v_proj", "o_proj",
                   "gate_proj", "up_proj", "down_proj"}

FORMATS     = ["bf16", "uq32", "int4", "int8"]
FORMAT_BITS = {"uq32": 5, "int4": 4, "int8": 8}   # bf16 has no grid

SIGMA_VALUES = [0.0, 0.01, 0.03, 0.05, 0.10, 0.15, 0.20]
BLOCK_SIZE   = 16   # per-block-16 scale granularity (matches UQ3.2 design)

SEQ_LEN     = 512
MAX_SAMPLES = 32

OUT_DIR     = os.path.join(_SCRIPT_DIR, "results", "rq_noise_format")
OUTPUT_JSON = os.path.join(OUT_DIR, "rq_noise_format_results.json")
os.makedirs(OUT_DIR, exist_ok=True)


# ── Noise models ───────────────────────────────────────────────────────────────

def noise_bf16(w: torch.Tensor, sigma: float) -> torch.Tensor:
    """Continuous multiplicative noise in value space (no quantization grid)."""
    if sigma == 0.0:
        return w
    orig = w.dtype
    eps  = torch.randn_like(w.float()) * sigma
    return (w.float() * (1.0 + eps)).to(orig)


def noise_fmt_decode(w: torch.Tensor, sigma: float, bits: int,
                     block_size: int = BLOCK_SIZE) -> torch.Tensor:
    """
    Physical RRAM model: quantize, drift, re-decode with fixed scale.

    The per-block scale is a digital register — it is not affected by
    conductance noise. Only the individual weight conductances drift.
    Re-decoding to the nearest bin captures the step-function threshold:
    below the bin-crossing point the decoded weight is bit-exact.
    """
    orig_shape = w.shape
    orig_dtype = w.dtype
    maxval = (1 << (bits - 1)) - 1

    w2 = w.reshape(-1, block_size).float()

    # Step 1: compute scale and encode
    scale  = w2.abs().amax(dim=1, keepdim=True).clamp(min=1e-8) / maxval
    w_int  = (w2 / scale).round().clamp(-maxval, maxval)
    w_q    = w_int * scale                             # reconstructed stored value

    if sigma == 0.0:
        return w_q.reshape(orig_shape).to(orig_dtype)

    # Step 2: conductance drifts multiplicatively
    eps      = torch.randn_like(w_q) * sigma
    w_noisy  = w_q * (1.0 + eps)

    # Step 3: re-decode using the SAME scale (bin boundaries unchanged)
    w_dec = (w_noisy / scale).round().clamp(-maxval, maxval) * scale

    return w_dec.reshape(orig_shape).to(orig_dtype)


# ── Weight patch/restore manager ───────────────────────────────────────────────

class FormatNoiseManager:
    """Patches RRAM projection weights in-place. Restores on .remove()."""

    def __init__(self, model, sigma: float, fmt: str,
                 block_size: int = BLOCK_SIZE, seed: int = 42):
        self._saved:  dict = {}
        self._params: list = []

        bits = FORMAT_BITS.get(fmt)   # None for bf16
        torch.manual_seed(seed)

        for mod_name, module in model.named_modules():
            proj = mod_name.split(".")[-1]
            if proj not in RRAM_PROJ_NAMES or not hasattr(module, "weight"):
                continue
            w  = module.weight.data
            w2 = w.view(w.shape[0], -1)

            if fmt == "bf16":
                w_eff = noise_bf16(w2, sigma)
            else:
                w_eff = noise_fmt_decode(w2, sigma, bits, block_size)

            self._saved[id(module.weight)] = w.clone()
            module.weight.data = w_eff.view_as(w)
            self._params.append(module.weight)

    def remove(self):
        for p in self._params:
            orig = self._saved.get(id(p))
            if orig is not None:
                p.data = orig
        self._saved.clear()
        self._params.clear()


# ── PPL evaluation ─────────────────────────────────────────────────────────────

def evaluate_ppl(model, tokenizer) -> float:
    ds   = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    text = "\n\n".join(ds["text"])
    enc  = tokenizer(text[:MAX_SAMPLES * SEQ_LEN * 6],
                     add_special_tokens=False, return_tensors="pt")
    ids  = enc.input_ids[:, :MAX_SAMPLES * SEQ_LEN].to(next(model.parameters()).device)
    nlls = []
    with torch.no_grad():
        for i in range(MAX_SAMPLES):
            s, e = i * SEQ_LEN, (i + 1) * SEQ_LEN
            if e > ids.size(1):
                break
            chunk = ids[:, s:e]
            nlls.append(model(chunk, labels=chunk.clone()).loss.item())
    return torch.exp(torch.tensor(nlls).mean()).item()


def exp_key(fmt: str, sigma: float) -> str:
    return f"{fmt}_s{sigma:.2f}"


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    all_results: dict = {}
    if os.path.exists(OUTPUT_JSON):
        with open(OUTPUT_JSON) as f:
            all_results = json.load(f)
        print(f"Loaded cached results: {list(all_results.keys())}")

    for model_name in MODEL_LIST:
        short = model_name.split("/")[-1]
        print(f"\n====== {short} ======")

        if model_name not in all_results:
            all_results[model_name] = {}

        needed = []
        if "baseline" not in all_results[model_name]:
            needed.append(("baseline", None, None))
        for fmt in FORMATS:
            for sigma in SIGMA_VALUES:
                key = exp_key(fmt, sigma)
                if key not in all_results[model_name]:
                    needed.append((key, fmt, sigma))

        if not needed:
            print("  all variants cached, skipping")
            continue

        try:
            model = AutoModelForCausalLM.from_pretrained(
                model_name, torch_dtype=torch.bfloat16, device_map="auto"
            )
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            model.eval()
            p = next(model.parameters())
            print(f"  dtype={p.dtype}  device={p.device}")
        except Exception as e:
            print(f"  LOAD ERROR: {e}")
            all_results[model_name]["load_error"] = str(e)
            with open(OUTPUT_JSON, "w") as f:
                json.dump(all_results, f, indent=2)
            continue

        for key, fmt, sigma in needed:
            if key in all_results[model_name]:
                continue
            tag = "baseline" if fmt is None else f"{fmt} σ={sigma:.2f}"
            print(f"  -- {tag}", end="", flush=True)
            mgr = None
            try:
                if fmt is not None:
                    mgr = FormatNoiseManager(model, sigma, fmt)
                ppl = evaluate_ppl(model, tokenizer)
                if mgr:
                    mgr.remove()
                all_results[model_name][key] = ppl
                base   = all_results[model_name].get("baseline", ppl)
                suffix = f"  Δ{ppl - base:+.3f}" if fmt is not None else ""
                print(f"  PPL={ppl:.3f}{suffix}")
            except Exception as e:
                if mgr:
                    mgr.remove()
                print(f"  ERROR: {e}")
                all_results[model_name][key] = {"error": str(e)}
            with open(OUTPUT_JSON, "w") as f:
                json.dump(all_results, f, indent=2)

        del model
        gc.collect()
        torch.cuda.empty_cache()

    # ── Summary tables ─────────────────────────────────────────────────────────
    rows = []
    for model_name, variants in all_results.items():
        if "load_error" in variants:
            continue
        short    = model_name.split("/")[-1]
        baseline = variants.get("baseline")
        if not isinstance(baseline, (int, float)):
            continue
        for fmt in FORMATS:
            for sigma in SIGMA_VALUES:
                key = exp_key(fmt, sigma)
                val = variants.get(key)
                if not isinstance(val, (int, float)):
                    continue
                rows.append({
                    "model":     short,
                    "fmt":       fmt,
                    "sigma":     sigma,
                    "ppl":       val,
                    "ppl_base":  baseline,
                    "delta_ppl": val - baseline,
                })

    df = pd.DataFrame(rows)
    if df.empty:
        print("\nNo complete results yet.")
        return

    df.to_csv(os.path.join(OUT_DIR, "rq_noise_format_results.csv"), index=False)

    print("\n=== ΔPPL by format and σ (averaged over models) ===")
    piv = df.pivot_table(index="fmt", columns="sigma",
                         values="delta_ppl", aggfunc="mean")
    print(piv.round(3).to_string())

    # Noise-tolerance threshold: smallest σ where ΔPPL > 0.5
    print("\n=== σ threshold for ΔPPL > 0.5 (averaged over models) ===")
    for fmt in FORMATS:
        sub = df[df["fmt"] == fmt].groupby("sigma")["delta_ppl"].mean().sort_index()
        exceeded = sub[sub > 0.5]
        thr = f"σ={exceeded.index[0]:.2f}" if len(exceeded) else ">0.20"
        print(f"  {fmt:6s}:  first ΔPPL>0.5 at {thr}")

    # ── Plot ───────────────────────────────────────────────────────────────────
    models = df["model"].unique()
    fig, axes = plt.subplots(1, len(models), figsize=(5 * len(models), 5), squeeze=False)
    fig.suptitle(
        "RQ-Noise-Format: Quantization Format vs RRAM Noise Tolerance\n"
        "bf16 = continuous noise  |  others = quantize→noise→re-decode",
        fontsize=11, fontweight="bold")

    fmt_styles = {
        "bf16": dict(color="crimson",    ls="-",  lw=2.5, label="BF16 (continuous noise)"),
        "uq32": dict(color="steelblue",  ls="-",  lw=2.5, label="UQ3.2 decode (5-bit, ours)"),
        "int4": dict(color="darkorange", ls="--", lw=1.8, label="INT4 decode (4-bit)"),
        "int8": dict(color="forestgreen",ls="--", lw=1.8, label="INT8 decode (8-bit)"),
    }

    for ci, mdl in enumerate(models):
        ax  = axes[0][ci]
        sub = df[df["model"] == mdl]
        for fmt, sty in fmt_styles.items():
            fsub = sub[sub["fmt"] == fmt].sort_values("sigma")
            if fsub.empty:
                continue
            ax.plot(fsub["sigma"], fsub["delta_ppl"], marker="o", **sty)
        ax.axhline(0,       color="black", lw=0.8,  ls=":")
        ax.axhspan(0, 0.2,  alpha=0.12, color="green",  label="near-lossless (<0.2)")
        ax.axvline(0.03,    color="gray", lw=0.8, ls="--", label="σ=0.03 (rq_noise threshold)")
        ax.set_title(mdl, fontsize=9, fontweight="bold")
        ax.set_xlabel("Noise σ", fontsize=8)
        ax.set_ylabel("ΔPPL vs BF16 baseline", fontsize=8)
        ax.set_xticks(SIGMA_VALUES)
        ax.tick_params(axis="x", labelsize=7)
        ax.legend(fontsize=6.5)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    fig_path = os.path.join(OUT_DIR, "rq_noise_format.png")
    plt.savefig(fig_path, dpi=180, bbox_inches="tight")
    plt.close()
    print(f"\nPlot: {fig_path}")
    print(f"Results: {OUT_DIR}/")


if __name__ == "__main__":
    main()
