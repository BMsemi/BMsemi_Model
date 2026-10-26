"""
RQ-Compute — Fixed-Point Precision Inside Transformer Blocks
=============================================================
Question: can attention and FFN compute survive per-token fixed-point quantization
at the same INT-N precision used for photonic transport (INT10)?

If yes: the entire pipeline — attention, transport, FFN — can run in fixed-point
with a single FP conversion only at lm_head.

Four scenarios (hooks inside each block):
  qkv       : Q, K, V projection outputs quantized  (before attention matmuls)
  attn_io   : Q, K, V outputs + o_proj input quantized  (full attention I/O)
  ffn_hidden: down_proj input quantized  (SwiGLU hidden state, post-SiLU)
  all_compute: all of the above combined

Quantization scheme: per-token symmetric fixed-point (same as RQ-Transport)
  scale = max|x[token,:]| / (2^(N-1) - 1)
  x_q = clamp(round(x / scale), ...) * scale
  → analogous to shift-encoding: E8M0 exponent + INT-N mantissa per token

N values tested: 8, 10, 12, 14, 16
Models: Qwen3-1.7B, Qwen3-4B, Qwen3-8B (successful in RQ-Transport)

Outputs: results/rq_compute/
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

N_BITS    = [8, 10, 12, 14, 16]
SCENARIOS = ["qkv", "attn_io", "ffn_hidden", "all_compute"]

SEQ_LEN     = 512
MAX_SAMPLES = 32

OUT_DIR = os.path.join(_SCRIPT_DIR, "results", "rq_compute")
os.makedirs(OUT_DIR, exist_ok=True)
OUTPUT_JSON = os.path.join(OUT_DIR, "rq_compute_results.json")


# ── Per-token symmetric fixed-point quantizer ──────────────────────────────────

def per_token_fixedpoint(x: torch.Tensor, n_bits: int) -> torch.Tensor:
    """Quantize x to n-bit symmetric fixed-point, per token row (dim=-1 is feature dim)."""
    maxval = (1 << (n_bits - 1)) - 1
    # Works for any shape [..., seq_len, feature_dim] by treating last dim as feature
    amax   = x.abs().amax(dim=-1, keepdim=True).clamp(min=1e-8)
    scale  = amax / maxval
    x_q    = torch.clamp(torch.round(x / scale), -maxval, maxval)
    return x_q * scale


# ── Locate sub-modules ─────────────────────────────────────────────────────────

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


def _get_attn_submodule(layer, names):
    """Try a list of attribute names on layer.self_attn, return first found."""
    attn = getattr(layer, "self_attn", None)
    if attn is None:
        raise RuntimeError("No self_attn in layer")
    for name in names:
        if hasattr(attn, name):
            return getattr(attn, name)
    raise RuntimeError(f"None of {names} found in {type(attn).__name__}")


def _get_mlp_submodule(layer, names):
    """Try a list of attribute names on layer.mlp, return first found."""
    mlp = getattr(layer, "mlp", None)
    if mlp is None:
        raise RuntimeError("No mlp in layer")
    for name in names:
        if hasattr(mlp, name):
            return getattr(mlp, name)
    raise RuntimeError(f"None of {names} found in {type(mlp).__name__}")


# ── Hook classes ───────────────────────────────────────────────────────────────

class OutputQuantHook:
    """Register a forward_hook on a module to quantize its output tensor."""
    def __init__(self, module, n_bits: int):
        self.n_bits = n_bits
        self.handle = module.register_forward_hook(self._hook)

    def _hook(self, module, inputs, output):
        if isinstance(output, torch.Tensor):
            return per_token_fixedpoint(output.float(), self.n_bits).to(output.dtype)
        # Tuple output — only quantize first element
        h = per_token_fixedpoint(output[0].float(), self.n_bits).to(output[0].dtype)
        return (h,) + output[1:]

    def remove(self):
        self.handle.remove()


class PreInputQuantHook:
    """Register a forward_pre_hook on a module to quantize its first input."""
    def __init__(self, module, n_bits: int):
        self.n_bits = n_bits
        self.handle = module.register_forward_pre_hook(self._hook)

    def _hook(self, module, inputs):
        x = inputs[0]
        q = per_token_fixedpoint(x.float(), self.n_bits).to(x.dtype)
        return (q,) + inputs[1:]

    def remove(self):
        self.handle.remove()


class ComputeHookManager:
    """Attaches the appropriate set of hooks for a given scenario."""

    def __init__(self, model, scenario: str, n_bits: int):
        self.hooks: list = []
        layers = _get_decoder_layers(model)

        for layer in layers:
            if scenario in ("qkv", "attn_io", "all_compute"):
                # Quantize Q, K, V projection outputs
                for proj_names in (
                    ["q_proj"],
                    ["k_proj"],
                    ["v_proj"],
                ):
                    try:
                        mod = _get_attn_submodule(layer, proj_names)
                        self.hooks.append(OutputQuantHook(mod, n_bits))
                    except RuntimeError:
                        pass

            if scenario in ("attn_io", "all_compute"):
                # Quantize context vector entering o_proj
                try:
                    o_proj = _get_attn_submodule(layer, ["o_proj"])
                    self.hooks.append(PreInputQuantHook(o_proj, n_bits))
                except RuntimeError:
                    pass

            if scenario in ("ffn_hidden", "all_compute"):
                # Quantize post-SiLU hidden state entering down_proj
                try:
                    down = _get_mlp_submodule(layer, ["down_proj", "c_proj"])
                    self.hooks.append(PreInputQuantHook(down, n_bits))
                except RuntimeError:
                    pass

    def remove(self):
        for h in self.hooks:
            h.remove()
        self.hooks.clear()


# ── PPL evaluation ─────────────────────────────────────────────────────────────

def evaluate_ppl(model, tokenizer) -> float:
    ds   = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    text = "\n\n".join(ds["text"])
    text = text[: MAX_SAMPLES * SEQ_LEN * 6]
    enc  = tokenizer(text, add_special_tokens=False, return_tensors="pt")
    ids  = enc.input_ids[:, : MAX_SAMPLES * SEQ_LEN]
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


def run_variant(model, tokenizer, scenario: str, n_bits: int | None) -> float:
    hook_mgr = None
    if n_bits is not None:
        hook_mgr = ComputeHookManager(model, scenario, n_bits)
    ppl = evaluate_ppl(model, tokenizer)
    if hook_mgr is not None:
        hook_mgr.remove()
    return ppl


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    all_results: dict = {}
    if os.path.exists(OUTPUT_JSON):
        with open(OUTPUT_JSON) as f:
            all_results = json.load(f)
        print(f"Loaded existing results: {list(all_results.keys())}")

    for model_name in MODEL_LIST:
        short = model_name.split("/")[-1]
        print(f"\n====== {short} ======")

        if model_name not in all_results:
            all_results[model_name] = {}

        needed = []
        if "baseline" not in all_results[model_name]:
            needed.append(("baseline", None, None))
        for scenario in SCENARIOS:
            for nb in N_BITS:
                key = f"{scenario}_INT{nb}"
                if key not in all_results[model_name]:
                    needed.append((key, scenario, nb))

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

        for key, scenario, nb in needed:
            if key in all_results[model_name]:
                print(f"  -- {key}  [skip]")
                continue
            tag = "baseline" if nb is None else f"{scenario} INT{nb}"
            print(f"  -- {tag}", end="", flush=True)
            try:
                ppl = run_variant(model, tokenizer, scenario, nb)
                all_results[model_name][key] = ppl
                base = all_results[model_name].get("baseline", ppl)
                delta = f"  Δ{ppl - base:+.3f}" if nb is not None else ""
                print(f"  PPL={ppl:.3f}{delta}")
            except Exception as e:
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
        short = model_name.split("/")[-1]
        baseline = variants.get("baseline")
        if not isinstance(baseline, (int, float)):
            continue
        for scenario in SCENARIOS:
            for nb in N_BITS:
                key = f"{scenario}_INT{nb}"
                val = variants.get(key)
                if not isinstance(val, (int, float)):
                    continue
                rows.append({
                    "model": short, "scenario": scenario, "n_bits": nb,
                    "ppl": val, "ppl_base": baseline,
                    "delta_ppl": val - baseline,
                    "rel_deg_pct": (val - baseline) / baseline * 100,
                })

    df = pd.DataFrame(rows)
    if df.empty:
        print("\nNo complete results yet.")
        return

    df.to_csv(os.path.join(OUT_DIR, "rq_compute_results.csv"), index=False)
    print("\n\n=== SUMMARY: ΔPPL vs baseline ===")
    piv = df.pivot_table(
        index=["model", "scenario"], columns="n_bits",
        values="delta_ppl", aggfunc="first"
    )
    print(piv.round(3).to_string())

    # ── Plot ───────────────────────────────────────────────────────────────────
    models = df["model"].unique()
    fig, axes = plt.subplots(len(models), len(SCENARIOS),
                             figsize=(4 * len(SCENARIOS), 3.5 * len(models)),
                             squeeze=False)
    fig.suptitle("RQ-Compute: Fixed-Point Precision Inside Attention + FFN\n"
                 "(per-token symmetric FXP, ΔPPL vs BF16 baseline)",
                 fontsize=12, fontweight="bold")

    sc_titles = {
        "qkv":         "Q/K/V projection outputs",
        "attn_io":     "Q/K/V + o_proj input",
        "ffn_hidden":  "FFN: down_proj input (post-SiLU)",
        "all_compute": "All compute (attn + FFN)",
    }

    for ri, mdl in enumerate(models):
        sub = df[df["model"] == mdl]
        base = sub["ppl_base"].iloc[0]
        for ci, sc in enumerate(SCENARIOS):
            ax = axes[ri][ci]
            ssub = sub[sub["scenario"] == sc].sort_values("n_bits")
            if not ssub.empty:
                ax.plot(ssub["n_bits"], ssub["delta_ppl"],
                        marker="o", linewidth=2, color=f"C{ri}")
                for _, row in ssub.iterrows():
                    ax.annotate(f"{row['delta_ppl']:+.2f}",
                                xy=(row["n_bits"], row["delta_ppl"]),
                                xytext=(0, 6), textcoords="offset points",
                                ha="center", fontsize=7)
            ax.axhline(0, linestyle="--", color="gray", linewidth=1,
                       label=f"BF16 ({base:.2f})")
            ax.axhspan(0, base * 0.02, alpha=0.1, color="green", label="<2% band")
            ax.set_title(f"{mdl}\n{sc_titles[sc]}", fontsize=8, fontweight="bold")
            ax.set_xlabel("Fixed-point bits N", fontsize=8)
            ax.set_ylabel("ΔPPL", fontsize=8)
            ax.set_xticks(N_BITS)
            ax.legend(fontsize=7)
            ax.grid(True, alpha=0.3)

    plt.tight_layout()
    p = os.path.join(OUT_DIR, "rq_compute.png")
    plt.savefig(p, dpi=180, bbox_inches="tight")
    plt.close()
    print(f"\nPlot saved: {p}")
    print(f"Results saved to {OUT_DIR}/")


if __name__ == "__main__":
    main()
