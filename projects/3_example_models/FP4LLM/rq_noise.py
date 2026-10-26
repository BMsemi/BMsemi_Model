"""
RQ-Noise — RRAM Conductance Mismatch Sensitivity
=================================================
Question: At what slice_gain mismatch level does UQ3.2 quality fall below the
near-lossless threshold (+0.2 ΔPPL)?

Hardware context:
  reram_crossbar.py slice_gain0/slice_gain1 model cell-to-cell gain variation
  in the differential crossbar (positive and negative polarity cells).
  slice_gain ~ N(1, σ²) → each cell's output current is scaled by ~(1 + ε).
  Net effect on a linear layer: effective weight = w * gain_factor per element.

This script models that as multiplicative weight noise in PyTorch:
  w_eff = w * (1 + ε),  ε ~ N(0, σ²)  (element-wise, independent per weight)

σ sweep: {0.01, 0.03, 0.05, 0.10}
Scope: QKVO projections + FFN (all RRAM-mapped linear layers)
Noise is re-sampled once per forward pass (worst-case: no averaging across tokens).

Outputs: results/rq_noise/
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

# ── Configuration ──────────────────────────────────────────────────────────────

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_LIST = [
    "Qwen/Qwen3-1.7B",
    "Qwen/Qwen3-4B",
    "Qwen/Qwen3-8B",
]

SIGMA_VALUES = [0.0, 0.01, 0.03, 0.05, 0.10]

SEQ_LEN     = 512
MAX_SAMPLES = 32

OUT_DIR     = os.path.join(_SCRIPT_DIR, "results", "rq_noise")
OUTPUT_JSON = os.path.join(OUT_DIR, "rq_noise_results.json")
os.makedirs(OUT_DIR, exist_ok=True)

RRAM_PROJ_NAMES = {"q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"}


# ── Noise hook ─────────────────────────────────────────────────────────────────

class MultiplicativeWeightNoise:
    """
    Attaches forward pre-hooks that perturb linear layer weights with
    element-wise multiplicative noise: w_eff = w * (1 + ε), ε ~ N(0, σ²).

    This models BM Labs slice_gain ~ N(1, σ²) mismatch: each cell's
    conductance is scaled by an independent gain factor.
    """

    def __init__(self, model, sigma: float):
        self.sigma = sigma
        self.hooks: list = []
        if sigma == 0.0:
            return
        for name, module in model.named_modules():
            layer_name = name.split(".")[-1]
            if isinstance(module, torch.nn.Linear) and layer_name in RRAM_PROJ_NAMES:
                h = module.register_forward_pre_hook(self._make_hook())
                self.hooks.append(h)

    def _make_hook(self):
        sigma = self.sigma
        def _hook(module, inputs):
            noise = torch.randn_like(module.weight) * sigma
            module.weight.data.mul_(1.0 + noise)
        return _hook

    def remove(self):
        for h in self.hooks:
            h.remove()
        self.hooks.clear()


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
            if e > ids.size(1): break
            chunk = ids[:, s:e]
            nlls.append(model(chunk, labels=chunk.clone()).loss.item())
    return torch.exp(torch.tensor(nlls).mean()).item()


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

        needed = [s for s in SIGMA_VALUES if f"sigma_{s}" not in all_results[model_name]]
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

        for sigma in needed:
            key = f"sigma_{sigma}"
            if key in all_results[model_name]:
                continue
            label = "baseline" if sigma == 0.0 else f"σ={sigma}"
            print(f"  -- {label}", end="", flush=True)
            try:
                noise_mgr = MultiplicativeWeightNoise(model, sigma)
                ppl = evaluate_ppl(model, tokenizer)
                noise_mgr.remove()
                all_results[model_name][key] = ppl
                base = all_results[model_name].get("sigma_0.0", ppl)
                suffix = f"  Δ{ppl - base:+.3f}" if sigma > 0.0 else ""
                print(f"  PPL={ppl:.3f}{suffix}")
            except Exception as e:
                if "noise_mgr" in dir() and noise_mgr:
                    noise_mgr.remove()
                print(f"  ERROR: {e}")
                all_results[model_name][key] = {"error": str(e)}
            with open(OUTPUT_JSON, "w") as f:
                json.dump(all_results, f, indent=2)

        del model
        gc.collect()
        torch.cuda.empty_cache()

    # ── Summary ────────────────────────────────────────────────────────────────
    rows = []
    for model_name, variants in all_results.items():
        if "load_error" in variants:
            continue
        short    = model_name.split("/")[-1]
        baseline = variants.get("sigma_0.0")
        if not isinstance(baseline, (int, float)):
            continue
        for sigma in SIGMA_VALUES:
            key = f"sigma_{sigma}"
            val = variants.get(key)
            if not isinstance(val, (int, float)):
                continue
            rows.append({
                "model": short,
                "sigma": sigma,
                "ppl": val,
                "ppl_base": baseline,
                "delta_ppl": val - baseline,
            })

    df = pd.DataFrame(rows)
    if df.empty:
        print("\nNo complete results yet.")
        return

    df.to_csv(os.path.join(OUT_DIR, "rq_noise_results.csv"), index=False)

    print("\n=== ΔPPL vs noise level (σ) ===")
    piv = df.pivot_table(index="model", columns="sigma", values="delta_ppl", aggfunc="first")
    print(piv.round(3).to_string())

    # ── Plot ───────────────────────────────────────────────────────────────────
    models = df["model"].unique()
    fig, ax = plt.subplots(figsize=(7, 4))
    fig.suptitle("RQ-Noise: RRAM Conductance Mismatch Sensitivity\n"
                 "Multiplicative weight noise σ (models slice_gain ~ N(1,σ²))",
                 fontsize=11, fontweight="bold")

    colors = [f"C{i}" for i in range(len(models))]
    for mdl, col in zip(models, colors):
        sub = df[df["model"] == mdl].sort_values("sigma")
        nonzero = sub[sub["sigma"] > 0]
        ax.plot(nonzero["sigma"], nonzero["delta_ppl"],
                marker="o", linewidth=2, color=col, label=mdl)

    ax.axhline(0, color="black", linewidth=0.8, linestyle=":")
    ax.axhspan(0, 0.2, alpha=0.12, color="green", label="near-lossless (<0.2)")
    ax.axhspan(0.2, 3.0, alpha=0.06, color="orange", label="acceptable (<3.0)")
    ax.set_xlabel("Gain mismatch σ (slice_gain std dev)", fontsize=9)
    ax.set_ylabel("ΔPPL vs baseline", fontsize=9)
    ax.set_xticks(SIGMA_VALUES[1:])
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    p = os.path.join(OUT_DIR, "rq_noise.png")
    plt.savefig(p, dpi=180, bbox_inches="tight")
    plt.close()
    print(f"\nPlot: {p}")
    print(f"Results: {OUT_DIR}/")


if __name__ == "__main__":
    main()
