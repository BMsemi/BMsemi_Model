"""
RQ5-fix — Transport Precision with Hardware-Valid E8M0 Power-of-2 Scale
========================================================================
Previous rq_transport.py used real-valued scale:
    scale = amax / maxval                    ← not hardware-realizable

This script uses the hardware-correct E8M0 scale:
    E     = floor(log2(max|x_token|))       ← 8-bit exponent, sent once per token
    scale = 2^(E - (N-2))                   ← power-of-2, implemented as barrel shift

Derivation of 2^(E-(N-2)):
  amax ∈ [2^E, 2^(E+1))  →  amax/scale ∈ [2^(N-2), 2^(N-1))
  maxval = 2^(N-1)-1      →  no overflow guaranteed
  max element maps to integer in [2^(N-2), maxval], using 50–100% of range.

Real-valued scale uses 100% of range by definition; E8M0 loses at most 1 effective bit
on the max element — the extra quantization error vs real-valued scale is the key
measurement of this experiment.

Scenarios and models are identical to rq_transport.py for direct comparison.

Outputs: results/rq_transport_e8m0/
  rq_transport_e8m0_results.json
  rq_transport_e8m0_results.csv
  rq_transport_e8m0_comparison.csv   ← delta vs real-scale results
  rq_transport_e8m0.png
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
    "Qwen/Qwen2.5-1.5B",
    "Qwen/Qwen3-1.7B",
    "Qwen/Qwen3-4B",
    "Qwen/Qwen3-8B",
]

N_BITS    = [8, 10, 12, 14, 16]
SCENARIOS = ["layer_to_layer", "attn_to_mlp"]

SEQ_LEN     = 512
MAX_SAMPLES = 32

OUT_DIR      = os.path.join(_SCRIPT_DIR, "results", "rq_transport_e8m0")
ORIG_JSON    = os.path.join(_SCRIPT_DIR, "results", "rq_transport", "rq_transport_results.json")
OUTPUT_JSON  = os.path.join(OUT_DIR, "rq_transport_e8m0_results.json")
os.makedirs(OUT_DIR, exist_ok=True)


# ── E8M0 per-token quantizer (hardware-valid) ──────────────────────────────────

def per_token_e8m0_fixedpoint(x: torch.Tensor, n_bits: int) -> torch.Tensor:
    """Per-token symmetric FXP with E8M0 power-of-2 scale.

    scale = 2^(E - (N-2))  where E = floor(log2(max|x_token|))

    Hardware: E is stored as 1 byte per token (E8M0); dequantize via barrel shift.
    Guarantees no overflow: amax/scale ∈ [2^(N-2), 2^(N-1)) ⊂ [0, maxval].
    """
    maxval = (1 << (n_bits - 1)) - 1
    amax   = x.abs().amax(dim=-1, keepdim=True).clamp(min=1e-8)
    E      = torch.floor(torch.log2(amax))              # per-token exponent (E8M0)
    scale  = torch.pow(2.0, E - (n_bits - 2))           # power-of-2 scale
    x_q    = torch.clamp(torch.round(x / scale), -maxval, maxval)
    return x_q * scale


# ── Hook managers (identical structure to rq_transport.py) ─────────────────────

def _get_decoder_layers(model):
    for attr in ("model.layers", "transformer.h"):
        try:
            obj = model
            for part in attr.split("."):
                obj = getattr(obj, part)
            if hasattr(obj, "__iter__"):
                return list(obj)
        except AttributeError:
            pass
    raise RuntimeError(f"Cannot locate decoder layers in {type(model).__name__}")


def _get_post_attn_layernorms(model):
    norms = []
    for layer in _get_decoder_layers(model):
        for attr in ("post_attention_layernorm", "post_feedforward_layernorm", "ln_2"):
            if hasattr(layer, attr):
                norms.append(getattr(layer, attr))
                break
    if not norms:
        raise RuntimeError("Cannot locate post_attention_layernorm")
    return norms


class LayerToLayerHook:
    def __init__(self, model, n_bits):
        self.n_bits = n_bits
        self.hooks  = []
        for layer in _get_decoder_layers(model):
            self.hooks.append(layer.register_forward_hook(self._hook))

    def _hook(self, module, inputs, output):
        if isinstance(output, torch.Tensor):
            return per_token_e8m0_fixedpoint(output.float(), self.n_bits).to(output.dtype)
        hidden = output[0]
        q    = per_token_e8m0_fixedpoint(hidden.float(), self.n_bits).to(hidden.dtype)
        rest = output[1:] if isinstance(output, tuple) else tuple(output.to_tuple()[1:])
        return (q,) + rest

    def remove(self):
        for h in self.hooks: h.remove()
        self.hooks.clear()


class AttnToMlpHook:
    def __init__(self, model, n_bits):
        self.n_bits = n_bits
        self.hooks  = []
        for norm in _get_post_attn_layernorms(model):
            self.hooks.append(norm.register_forward_pre_hook(self._hook))

    def _hook(self, module, inputs):
        hidden = inputs[0]
        q = per_token_e8m0_fixedpoint(hidden.float(), self.n_bits).to(hidden.dtype)
        return (q,) + inputs[1:]

    def remove(self):
        for h in self.hooks: h.remove()
        self.hooks.clear()


# ── PPL evaluation ─────────────────────────────────────────────────────────────

def evaluate_ppl(model, tokenizer) -> float:
    ds   = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    text = "\n\n".join(ds["text"])
    text = text[: MAX_SAMPLES * SEQ_LEN * 6]
    enc  = tokenizer(text, add_special_tokens=False, return_tensors="pt")
    ids  = enc.input_ids[:, : MAX_SAMPLES * SEQ_LEN].to(next(model.parameters()).device)
    nlls = []
    with torch.no_grad():
        for i in range(MAX_SAMPLES):
            s, e = i * SEQ_LEN, (i + 1) * SEQ_LEN
            if e > ids.size(1): break
            chunk = ids[:, s:e]
            nlls.append(model(chunk, labels=chunk.clone()).loss.item())
    return torch.exp(torch.tensor(nlls).mean()).item()


def run_variant(model, tokenizer, scenario, n_bits):
    hook = None
    if n_bits is not None:
        hook = LayerToLayerHook(model, n_bits) if scenario == "layer_to_layer" \
               else AttnToMlpHook(model, n_bits)
    ppl = evaluate_ppl(model, tokenizer)
    if hook: hook.remove()
    return ppl


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    all_results: dict = {}
    if os.path.exists(OUTPUT_JSON):
        with open(OUTPUT_JSON) as f:
            all_results = json.load(f)
        print(f"Loaded cached E8M0 results: {list(all_results.keys())}")

    for model_name in MODEL_LIST:
        short = model_name.split("/")[-1]
        print(f"\n====== {short} ======")

        if model_name not in all_results:
            all_results[model_name] = {}

        needed = []
        if "baseline" not in all_results[model_name]:
            needed.append(("baseline", None, None))
        for sc in SCENARIOS:
            for nb in N_BITS:
                key = f"{sc}_INT{nb}"
                if key not in all_results[model_name]:
                    needed.append((key, sc, nb))

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

        for key, sc, nb in needed:
            if key in all_results[model_name]:
                continue
            tag = "baseline" if nb is None else f"{sc} INT{nb}"
            print(f"  -- {tag}", end="", flush=True)
            try:
                ppl = run_variant(model, tokenizer, sc, nb)
                all_results[model_name][key] = ppl
                base = all_results[model_name].get("baseline", ppl)
                suffix = f"  Δ{ppl - base:+.3f}" if nb is not None else ""
                print(f"  PPL={ppl:.3f}{suffix}")
            except Exception as e:
                print(f"  ERROR: {e}")
                all_results[model_name][key] = {"error": str(e)}
            with open(OUTPUT_JSON, "w") as f:
                json.dump(all_results, f, indent=2)

        del model
        gc.collect()
        torch.cuda.empty_cache()

    # ── Build comparison table ─────────────────────────────────────────────────
    orig = {}
    if os.path.exists(ORIG_JSON):
        with open(ORIG_JSON) as f:
            orig = json.load(f)

    rows = []
    for model_name, variants in all_results.items():
        if "load_error" in variants:
            continue
        short    = model_name.split("/")[-1]
        baseline = variants.get("baseline")
        if not isinstance(baseline, (int, float)):
            continue
        orig_v = orig.get(model_name, {})

        for sc in SCENARIOS:
            for nb in N_BITS:
                key = f"{sc}_INT{nb}"
                val = variants.get(key)
                if not isinstance(val, (int, float)):
                    continue
                orig_val = orig_v.get(key)
                row = {
                    "model": short, "scenario": sc, "n_bits": nb,
                    "ppl_e8m0": val, "ppl_base": baseline,
                    "delta_e8m0": val - baseline,
                }
                if isinstance(orig_val, (int, float)):
                    row["ppl_realscale"]   = orig_val
                    row["delta_realscale"] = orig_val - baseline
                    row["e8m0_overhead"]   = val - orig_val   # extra PPL cost of E8M0
                rows.append(row)

    df = pd.DataFrame(rows)
    if df.empty:
        print("\nNo complete results yet.")
        return

    df.to_csv(os.path.join(OUT_DIR, "rq_transport_e8m0_results.csv"), index=False)

    # ── Print ΔPPL table: E8M0 vs real-scale ──────────────────────────────────
    print("\n\n=== ΔPPL: E8M0 scale vs BF16 baseline ===")
    piv = df.pivot_table(index=["model", "scenario"], columns="n_bits",
                         values="delta_e8m0", aggfunc="first")
    print(piv.round(3).to_string())

    if "delta_realscale" in df.columns and "e8m0_overhead" in df.columns:
        print("\n=== E8M0 overhead vs real-valued scale (extra ΔPPL from power-of-2 rounding) ===")
        piv2 = df.pivot_table(index=["model", "scenario"], columns="n_bits",
                              values="e8m0_overhead", aggfunc="first")
        print(piv2.round(3).to_string())

        df.to_csv(os.path.join(OUT_DIR, "rq_transport_e8m0_comparison.csv"), index=False)

    # ── Plot: side-by-side E8M0 vs real-scale ─────────────────────────────────
    models = df["model"].unique()
    fig, axes = plt.subplots(len(models), 2, figsize=(12, 3.5 * len(models)), squeeze=False)
    fig.suptitle("RQ5-fix: Transport Precision — E8M0 vs Real-Valued Scale\n"
                 "(per-token FXP at layer→layer boundary, ΔPPL vs BF16)",
                 fontsize=12, fontweight="bold")

    for ri, mdl in enumerate(models):
        sub = df[df["model"] == mdl]
        for ci, sc in enumerate(SCENARIOS):
            ax  = axes[ri][ci]
            ssub = sub[sub["scenario"] == sc].sort_values("n_bits")

            if not ssub.empty:
                ax.plot(ssub["n_bits"], ssub["delta_e8m0"],
                        marker="o", linewidth=2, label="E8M0 (hardware)", color="steelblue")
                if "delta_realscale" in ssub.columns:
                    ax.plot(ssub["n_bits"], ssub["delta_realscale"],
                            marker="s", linewidth=2, linestyle="--",
                            label="Real-valued scale", color="gray")

            ax.axhline(0, linestyle=":", color="black", linewidth=0.8)
            ax.axhspan(0, 0.2, alpha=0.12, color="green", label="near-lossless band (<0.2)")
            base = sub["ppl_base"].iloc[0] if not sub.empty else 0
            ax.set_title(f"{mdl} — {sc.replace('_', '→')}\n(baseline PPL {base:.2f})",
                         fontsize=9, fontweight="bold")
            ax.set_xlabel("Fixed-point bits N", fontsize=8)
            ax.set_ylabel("ΔPPL vs BF16", fontsize=8)
            ax.set_xticks(N_BITS)
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3)

    plt.tight_layout()
    p = os.path.join(OUT_DIR, "rq_transport_e8m0.png")
    plt.savefig(p, dpi=180, bbox_inches="tight")
    plt.close()
    print(f"\nPlot: {p}")
    print(f"Results: {OUT_DIR}/")


if __name__ == "__main__":
    main()
