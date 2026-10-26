"""
RQ-Transport — Photonic Inter-Layer Transport Precision
=======================================================
Question: what is the minimum number of fixed-point bits needed to transport
the hidden-state tensor across a photonic link without degrading PPL?

Two transport points:
  layer_to_layer : hook on each decoder block output (output[0] of the layer)
  attn_to_mlp    : hook on post_attention_layernorm input (residual just before MLP)

Quantization scheme (per-token symmetric fixed-point):
  scale = max(|x|, dim=-1, keepdim=True) / (2^(N-1) - 1)
  x_q   = clamp(round(x / scale), -(2^(N-1)-1), 2^(N-1)-1) * scale

N values tested: 8, 10, 12, 14, 16 (plus unquantized baseline)

Models: all suitable cached base models detected at runtime.

Outputs: results/rq_transport/
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
    "Qwen/Qwen2.5-0.5B",
    "Qwen/Qwen2.5-1.5B",
    "Qwen/Qwen2.5-7B",
    "Qwen/Qwen3-1.7B",
    "Qwen/Qwen3-4B",
    "Qwen/Qwen3-8B",
    "meta-llama/Meta-Llama-3.1-8B",
    "mistralai/Mistral-7B-v0.1",
]

# Trust-remote-code models
TRUST_REMOTE = {"THUDM/GLM-4-9B-Chat"}

N_BITS    = [8, 10, 12, 14, 16]       # fixed-point widths to test
SCENARIOS = ["layer_to_layer", "attn_to_mlp"]

SEQ_LEN     = 512
MAX_SAMPLES = 32
SEED        = 42

OUT_DIR = os.path.join(_SCRIPT_DIR, "results", "rq_transport")
os.makedirs(OUT_DIR, exist_ok=True)
OUTPUT_JSON = os.path.join(OUT_DIR, "rq_transport_results.json")


# ── Per-token symmetric fixed-point quantizer ──────────────────────────────────

def per_token_fixedpoint(x: torch.Tensor, n_bits: int) -> torch.Tensor:
    """Quantize x to n-bit symmetric fixed-point per token row."""
    maxval = (1 << (n_bits - 1)) - 1          # 2^(N-1) - 1
    amax   = x.abs().amax(dim=-1, keepdim=True).clamp(min=1e-8)
    scale  = amax / maxval
    x_q    = torch.clamp(torch.round(x / scale), -maxval, maxval)
    return x_q * scale


# ── Hook managers ──────────────────────────────────────────────────────────────

class LayerToLayerHook:
    """Quantize each decoder block's output hidden states (output[0])."""

    def __init__(self, model, n_bits: int):
        self.n_bits = n_bits
        self.hooks  = []
        layers = _get_decoder_layers(model)
        for layer in layers:
            h = layer.register_forward_hook(self._hook)
            self.hooks.append(h)

    def _hook(self, module, inputs, output):
        # Some model families return a plain Tensor; others return a tuple/ModelOutput.
        if isinstance(output, torch.Tensor):
            return per_token_fixedpoint(output.float(), self.n_bits).to(output.dtype)
        hidden = output[0]
        q = per_token_fixedpoint(hidden.float(), self.n_bits).to(hidden.dtype)
        # ModelOutput supports tuple slicing via to_tuple(), giving a real tuple.
        rest = output[1:] if isinstance(output, tuple) else tuple(output.to_tuple()[1:])
        return (q,) + rest

    def remove(self):
        for h in self.hooks:
            h.remove()
        self.hooks.clear()


class AttnToMlpHook:
    """Quantize the residual stream at the post_attention_layernorm input
    (i.e. the hidden state after attention, before MLP)."""

    def __init__(self, model, n_bits: int):
        self.n_bits = n_bits
        self.hooks  = []
        norms = _get_post_attn_layernorms(model)
        for norm in norms:
            h = norm.register_forward_pre_hook(self._hook)
            self.hooks.append(h)

    def _hook(self, module, inputs):
        hidden = inputs[0]
        q = per_token_fixedpoint(hidden.float(), self.n_bits).to(hidden.dtype)
        return (q,) + inputs[1:]

    def remove(self):
        for h in self.hooks:
            h.remove()
        self.hooks.clear()


# ── Utilities to locate layers in diverse model families ──────────────────────

def _get_decoder_layers(model):
    """Return the list of transformer decoder block modules."""
    # Qwen2 / Qwen3 / LLaMA / Mistral architecture variants
    for attr in ("model.layers", "transformer.h"):
        try:
            layers = model
            for part in attr.split("."):
                layers = getattr(layers, part)
            if hasattr(layers, "__iter__"):
                return list(layers)
        except AttributeError:
            pass
    raise RuntimeError(f"Cannot locate decoder layers in {type(model).__name__}")


def _get_post_attn_layernorms(model):
    """Return list of post_attention_layernorm (or equivalent) modules."""
    norms = []
    layers = _get_decoder_layers(model)
    for layer in layers:
        # LLaMA / Qwen2 / Qwen3 use post_attention_layernorm
        # Mistral uses the same naming
        for attr in ("post_attention_layernorm", "post_feedforward_layernorm",
                     "ln_2"):
            if hasattr(layer, attr):
                norms.append(getattr(layer, attr))
                break
    if not norms:
        raise RuntimeError("Cannot locate post_attention_layernorm")
    return norms


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
    """Attach hooks for the scenario/n_bits, evaluate PPL, remove hooks."""
    hook_obj = None
    if n_bits is not None:
        if scenario == "layer_to_layer":
            hook_obj = LayerToLayerHook(model, n_bits)
        else:
            hook_obj = AttnToMlpHook(model, n_bits)
    ppl = evaluate_ppl(model, tokenizer)
    if hook_obj is not None:
        hook_obj.remove()
    return ppl


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    # Load existing results
    all_results: dict = {}
    if os.path.exists(OUTPUT_JSON):
        with open(OUTPUT_JSON) as f:
            all_results = json.load(f)
        print(f"Loaded existing results for: {list(all_results.keys())}")

    for model_name in MODEL_LIST:
        short = model_name.split("/")[-1]
        print(f"\n====== {short} ======")

        if model_name not in all_results:
            all_results[model_name] = {}

        # Enumerate all keys to compute
        needed = []
        for scenario in SCENARIOS:
            # baseline: run once per model (same for both scenarios, store under "baseline")
            if "baseline" not in all_results[model_name]:
                needed.append(("baseline", None, None))
            for nb in N_BITS:
                key = f"{scenario}_INT{nb}"
                if key not in all_results[model_name]:
                    needed.append((key, scenario, nb))

        if not needed:
            print("  all variants cached, skipping")
            continue

        # Load model once per model_name
        try:
            trust = model_name in TRUST_REMOTE
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.bfloat16,
                device_map="auto",
                trust_remote_code=trust,
            )
            tokenizer = AutoTokenizer.from_pretrained(
                model_name, trust_remote_code=trust
            )
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
            tag = f"baseline" if nb is None else f"{scenario} INT{nb}"
            print(f"  -- {tag}", end="", flush=True)
            try:
                ppl = run_variant(model, tokenizer, scenario, nb)
                all_results[model_name][key] = ppl
                print(f"  PPL={ppl:.3f}")
            except Exception as e:
                print(f"  ERROR: {e}")
                all_results[model_name][key] = {"error": str(e)}

            # Checkpoint after each variant
            with open(OUTPUT_JSON, "w") as f:
                json.dump(all_results, f, indent=2)

        del model
        gc.collect()
        torch.cuda.empty_cache()

    # ── Build summary table ────────────────────────────────────────────────────
    rows = []
    for model_name, variants in all_results.items():
        if "load_error" in variants:
            continue
        short = model_name.split("/")[-1]
        baseline_ppl = variants.get("baseline")
        if not isinstance(baseline_ppl, (int, float)):
            continue
        for scenario in SCENARIOS:
            for nb in N_BITS:
                key = f"{scenario}_INT{nb}"
                val = variants.get(key)
                if not isinstance(val, (int, float)):
                    continue
                rows.append({
                    "model":       short,
                    "scenario":    scenario,
                    "n_bits":      nb,
                    "ppl":         val,
                    "ppl_base":    baseline_ppl,
                    "delta_ppl":   val - baseline_ppl,
                    "rel_deg_pct": (val - baseline_ppl) / baseline_ppl * 100,
                })

    df = pd.DataFrame(rows)
    if df.empty:
        print("\nNo results to summarize yet.")
        return

    df.to_csv(os.path.join(OUT_DIR, "rq_transport_results.csv"), index=False)

    print("\n\n=== SUMMARY (delta PPL vs baseline) ===")
    piv = df.pivot_table(
        index=["model", "scenario"], columns="n_bits",
        values="delta_ppl", aggfunc="first"
    )
    print(piv.round(3).to_string())

    # ── Plot ───────────────────────────────────────────────────────────────────
    models  = df["model"].unique()
    n_rows  = len(models)
    fig, axes = plt.subplots(n_rows, 2, figsize=(12, 3.5 * n_rows), squeeze=False)

    for row_idx, model_short in enumerate(models):
        sub = df[df["model"] == model_short]
        for col_idx, scenario in enumerate(SCENARIOS):
            ax  = axes[row_idx][col_idx]
            ssub = sub[sub["scenario"] == scenario].sort_values("n_bits")
            if ssub.empty:
                ax.set_visible(False)
                continue
            ax.plot(ssub["n_bits"], ssub["ppl"], marker="o", label=model_short)
            bppl = ssub["ppl_base"].iloc[0]
            ax.axhline(bppl, linestyle="--", color="gray", linewidth=0.8,
                       label=f"baseline ({bppl:.2f})")
            ax.set_title(f"{model_short} — {scenario}")
            ax.set_xlabel("Fixed-point bits (N)")
            ax.set_ylabel("Word PPL")
            ax.legend(fontsize=7)
            ax.set_xticks(N_BITS)

    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "rq_transport.png"), dpi=180)
    print(f"\nResults saved to {OUT_DIR}/")


if __name__ == "__main__":
    main()
