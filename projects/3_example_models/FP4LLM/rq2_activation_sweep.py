"""
RQ2 — Activation Precision at RRAM Input: QKVO Projection Sweep
================================================================
Question: Given 5-bit UQ3.2 RRAM weights, what activation format is required
at the RRAM input for QKVO projections to preserve model quality?

Hardware context:
  - RRAM arrays compute W×A where W is 5-bit UQ3.2 (FP4-mapped)
  - Activations (A) must be quantized before entering the crossbar
  - Scale granularity must match RRAM peripheral capability
  - Target: A5 UQ3.2 per-block-16 (native to RRAM; E8M0 shift scale)

Experiment design:
  Quantize the INPUT activation to each projection layer via pre-hooks.
  Q, K, V share the same input (post-layernorm hidden state).
  O takes the attention context as input (different tensor).

  Module subsets tested:
    Q, K, V, O            — individual sensitivity
    QK                    — attention score inputs together
    QKVO                  — all projections (full RRAM mapping)

  Granularities:
    tok  — per-token (1 scale covers full hidden_dim per token)
    b16  — per-block-16 (1 scale per 16 elements; matches MXFP4/UQ3.2)
    b32  — per-block-32 (coarser, lower overhead)

  Activation bits: 4, 5, 6, 8
  Scale: real-valued symmetric (best-case upper bound at each N bits;
         E8M0 overhead shown to be <0.1 PPL at same N from RQ5)

Key: {module}_{granularity}_A{bits}   e.g. QKVO_b16_A5, Q_tok_A8

Outputs: results/rq2_activation/
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

# Module subsets: projection layer attribute names inside self_attn
MODULES = {
    "Q":    ["q_proj"],
    "K":    ["k_proj"],
    "V":    ["v_proj"],
    "O":    ["o_proj"],
    "QK":   ["q_proj", "k_proj"],
    "QKVO": ["q_proj", "k_proj", "v_proj", "o_proj"],
}

# Granularity: None = per-token, int = block size
GRANULARITIES = {
    "tok": None,
    "b16": 16,
    "b32": 32,
}

A_BITS = [4, 5, 6, 8]

# Experiments to run: (module_key, gran_key, n_bits)
# QKVO at all granularities × all bits  → main RQ2 question
# Individual modules at b16 × key bits  → sensitivity (RQ3 data)
EXPERIMENTS: list[tuple[str, str, int]] = []
for abits in A_BITS:
    for gran in ["tok", "b16"]:
        EXPERIMENTS.append(("QKVO", gran, abits))
for mod in ["Q", "K", "V", "O", "QK"]:
    for abits in [4, 5, 8]:
        EXPERIMENTS.append((mod, "b16", abits))

SEQ_LEN     = 512
MAX_SAMPLES = 32

OUT_DIR     = os.path.join(_SCRIPT_DIR, "results", "rq2_activation")
OUTPUT_JSON = os.path.join(OUT_DIR, "rq2_activation_results.json")
os.makedirs(OUT_DIR, exist_ok=True)


# ── Quantizers ─────────────────────────────────────────────────────────────────

def per_token_fixedpoint(x: torch.Tensor, n_bits: int) -> torch.Tensor:
    """Symmetric FXP per token row (real-valued scale)."""
    maxval = (1 << (n_bits - 1)) - 1
    amax   = x.abs().amax(dim=-1, keepdim=True).clamp(min=1e-8)
    scale  = amax / maxval
    return torch.clamp(torch.round(x / scale), -maxval, maxval) * scale


def per_block_fixedpoint(x: torch.Tensor, n_bits: int, block_size: int) -> torch.Tensor:
    """Symmetric FXP per block of `block_size` elements along last dim."""
    *batch, d = x.shape
    if d % block_size != 0:
        # pad to next multiple, quantize, unpad
        pad = block_size - (d % block_size)
        x = torch.nn.functional.pad(x, (0, pad))
        *batch, d = x.shape

    n_blocks = d // block_size
    x_b  = x.reshape(*batch, n_blocks, block_size)
    maxval = (1 << (n_bits - 1)) - 1
    amax   = x_b.abs().amax(dim=-1, keepdim=True).clamp(min=1e-8)
    scale  = amax / maxval
    x_q    = torch.clamp(torch.round(x_b / scale), -maxval, maxval) * scale
    result = x_q.reshape(*batch, d)

    # Remove padding if added
    orig_d = x.shape[-1] - (block_size - (x.shape[-1] % block_size) if x.shape[-1] % block_size != 0 else 0)
    return result[..., :orig_d] if result.shape[-1] != orig_d else result


def quantize(x: torch.Tensor, n_bits: int, block_size) -> torch.Tensor:
    """Dispatch to per-token or per-block quantizer."""
    if block_size is None:
        return per_token_fixedpoint(x.float(), n_bits).to(x.dtype)
    return per_block_fixedpoint(x.float(), n_bits, block_size).to(x.dtype)


# ── Hook manager ───────────────────────────────────────────────────────────────

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


class ActivationQuantManager:
    """Attaches pre-hooks to specified projection inputs in every decoder layer."""

    def __init__(self, model, proj_names: list[str], n_bits: int, block_size):
        self.hooks: list = []
        for layer in _get_decoder_layers(model):
            attn = getattr(layer, "self_attn", None)
            if attn is None:
                continue
            for pname in proj_names:
                proj = getattr(attn, pname, None)
                if proj is None:
                    continue
                # Capture block_size and n_bits in closure
                def _make_hook(nb, bs):
                    def _hook(module, inputs):
                        x = inputs[0]
                        return (quantize(x, nb, bs),) + inputs[1:]
                    return _hook
                h = proj.register_forward_pre_hook(_make_hook(n_bits, block_size))
                self.hooks.append(h)

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


def exp_key(module: str, gran: str, abits: int) -> str:
    return f"{module}_{gran}_A{abits}"


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
            needed.append(("baseline", None, None, None))
        for mod, gran, abits in EXPERIMENTS:
            key = exp_key(mod, gran, abits)
            if key not in all_results[model_name]:
                needed.append((key, mod, gran, abits))

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

        for entry in needed:
            key, mod, gran, abits = entry
            if key in all_results[model_name]:
                continue
            tag = key if mod else "baseline"
            print(f"  -- {tag}", end="", flush=True)
            try:
                mgr = None
                if mod is not None:
                    mgr = ActivationQuantManager(
                        model,
                        proj_names=MODULES[mod],
                        n_bits=abits,
                        block_size=GRANULARITIES[gran],
                    )
                ppl = evaluate_ppl(model, tokenizer)
                if mgr: mgr.remove()
                all_results[model_name][key] = ppl
                base = all_results[model_name].get("baseline", ppl)
                suffix = f"  Δ{ppl - base:+.3f}" if mod is not None else ""
                print(f"  PPL={ppl:.3f}{suffix}")
            except Exception as e:
                if "mgr" in dir() and mgr: mgr.remove()
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
        if "load_error" in variants: continue
        short    = model_name.split("/")[-1]
        baseline = variants.get("baseline")
        if not isinstance(baseline, (int, float)): continue
        for mod, gran, abits in EXPERIMENTS:
            key = exp_key(mod, gran, abits)
            val = variants.get(key)
            if not isinstance(val, (int, float)): continue
            rows.append({
                "model": short, "module": mod, "granularity": gran,
                "n_bits": abits, "ppl": val,
                "ppl_base": baseline, "delta_ppl": val - baseline,
            })

    df = pd.DataFrame(rows)
    if df.empty:
        print("\nNo complete results yet.")
        return

    df.to_csv(os.path.join(OUT_DIR, "rq2_activation_results.csv"), index=False)

    # Table 1: QKVO — granularity × bits
    print("\n\n=== QKVO: ΔPPL by granularity and activation bits ===")
    qkvo = df[df["module"] == "QKVO"]
    if not qkvo.empty:
        piv = qkvo.pivot_table(index=["model", "granularity"], columns="n_bits",
                               values="delta_ppl", aggfunc="first")
        print(piv.round(3).to_string())

    # Table 2: module sensitivity at b16
    print("\n=== Module sensitivity (per-block-16) ===")
    sens = df[(df["granularity"] == "b16") & (df["module"].isin(["Q","K","V","O","QK","QKVO"]))]
    if not sens.empty:
        piv2 = sens.pivot_table(index=["model", "module"], columns="n_bits",
                                values="delta_ppl", aggfunc="first")
        print(piv2.round(3).to_string())

    # ── Plot ───────────────────────────────────────────────────────────────────
    models = df["model"].unique()
    fig, axes = plt.subplots(2, len(models), figsize=(5 * len(models), 10), squeeze=False)
    fig.suptitle("RQ2: Activation Precision at RRAM Input (QKVO Projections)\n"
                 "ΔPPL vs BF16 baseline, real-valued symmetric scale",
                 fontsize=12, fontweight="bold")

    gran_styles = {"tok": ("--", "o"), "b16": ("-", "s"), "b32": (":", "^")}
    gran_colors = {"tok": "gray", "b16": "steelblue", "b32": "darkorange"}

    for ci, mdl in enumerate(models):
        sub = df[df["model"] == mdl]

        # Top row: QKVO granularity comparison
        ax = axes[0][ci]
        for gran in ["tok", "b16", "b32"]:
            gsub = sub[(sub["module"] == "QKVO") & (sub["granularity"] == gran)].sort_values("n_bits")
            if gsub.empty: continue
            ls, mk = gran_styles[gran]
            ax.plot(gsub["n_bits"], gsub["delta_ppl"],
                    linestyle=ls, marker=mk, linewidth=2,
                    color=gran_colors[gran], label=f"QKVO {gran}")
        ax.axhline(0, color="black", linewidth=0.8, linestyle=":")
        ax.axhspan(0, 0.2, alpha=0.1, color="green", label="<0.2 target")
        ax.axhspan(0, 3.0, alpha=0.05, color="orange", label="<3.0 acceptable")
        ax.set_title(f"{mdl}\nGranularity comparison (QKVO)", fontsize=9, fontweight="bold")
        ax.set_xlabel("Activation bits (A)", fontsize=8)
        ax.set_ylabel("ΔPPL vs BF16", fontsize=8)
        ax.set_xticks(A_BITS)
        ax.legend(fontsize=7); ax.grid(True, alpha=0.3)

        # Bottom row: module sensitivity at b16
        ax2 = axes[1][ci]
        mods_to_plot = ["Q", "K", "V", "O", "QK", "QKVO"]
        colors = [f"C{i}" for i in range(len(mods_to_plot))]
        for mod, col in zip(mods_to_plot, colors):
            msub = sub[(sub["module"] == mod) & (sub["granularity"] == "b16")].sort_values("n_bits")
            if msub.empty: continue
            ax2.plot(msub["n_bits"], msub["delta_ppl"],
                     marker="o", linewidth=1.8, color=col, label=mod)
        ax2.axhline(0, color="black", linewidth=0.8, linestyle=":")
        ax2.axhspan(0, 0.2, alpha=0.1, color="green")
        ax2.axhspan(0, 3.0, alpha=0.05, color="orange")
        ax2.set_title(f"{mdl}\nModule sensitivity (per-block-16)", fontsize=9, fontweight="bold")
        ax2.set_xlabel("Activation bits (A)", fontsize=8)
        ax2.set_ylabel("ΔPPL vs BF16", fontsize=8)
        ax2.set_xticks([4, 5, 8])
        ax2.legend(fontsize=7, ncol=2); ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    p = os.path.join(OUT_DIR, "rq2_activation.png")
    plt.savefig(p, dpi=180, bbox_inches="tight")
    plt.close()
    print(f"\nPlot: {p}")
    print(f"Results: {OUT_DIR}/")


if __name__ == "__main__":
    main()
