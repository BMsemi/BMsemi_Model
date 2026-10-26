"""
RQ-Broadcast — Optical Broadcast Path Analysis
===============================================
Question: If photonic optical links carry the fan-out broadcast activations
(one input → many PEs), what precision is needed to stay near-lossless?

Hardware context:
  In the BM Labs cost model, OPTICAL_ALLOUT fires when average PE mesh hops > 4.
  The three natural broadcast points in transformer LLMs are:

  1. FFN shared input   — post-LN2 hidden state → gate_proj AND up_proj simultaneously
                          (same tensor, two weight matrices, one optical broadcast)
  2. QKV shared input   — post-LN1 hidden state → q_proj AND k_proj AND v_proj
                          (same tensor, three weight matrices, one broadcast)
  3. KV post-projection — k_proj/v_proj outputs broadcast to multiple Q-head PE groups
                          (GQA: 8 KV heads → 32-64 Q heads in LLaMA3/Qwen3)

Each path is quantized with the same E8M0 per-token FXP used in RQ5 transport,
since the optical encoding is identical regardless of which path uses the link.

Experiments:
  broadcast_ffn_input  — quantize MLP input (gate+up shared)
  broadcast_qkv_input  — quantize attention input (q+k+v shared)
  broadcast_kv_output  — quantize k_proj, v_proj outputs (GQA fanout)
  broadcast_all        — all three simultaneously

Bit widths: 8, 10, 12  (matching RQ5 transport sweep)

If broadcast paths are near-lossless at INT10, optical is justified there.
The combined result (broadcast_all at INT10) answers whether the full
optical broadcast strategy degrades quality.

Outputs: results/rq_broadcast/
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

BROADCAST_PATHS = ["broadcast_ffn_input", "broadcast_qkv_input",
                   "broadcast_kv_output", "broadcast_all"]
N_BITS = [8, 10, 12]

SEQ_LEN     = 512
MAX_SAMPLES = 32

OUT_DIR     = os.path.join(_SCRIPT_DIR, "results", "rq_broadcast")
OUTPUT_JSON = os.path.join(OUT_DIR, "rq_broadcast_results.json")
os.makedirs(OUT_DIR, exist_ok=True)


# ── E8M0 per-token quantizer (same as RQ5) ────────────────────────────────────

def e8m0_quantize(x: torch.Tensor, n_bits: int) -> torch.Tensor:
    """Per-token E8M0 FXP. Hardware: barrel shift only, no FP multiply."""
    maxval = (1 << (n_bits - 1)) - 1
    amax   = x.abs().amax(dim=-1, keepdim=True).clamp(min=1e-8)
    E      = torch.floor(torch.log2(amax))
    scale  = torch.pow(2.0, E - (n_bits - 2))
    return torch.clamp(torch.round(x / scale), -maxval, maxval) * scale


# ── Hook managers ──────────────────────────────────────────────────────────────

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


class BroadcastQuantManager:
    """
    Attaches hooks at the three optical broadcast points:

    FFN input:   MLP module pre-hook  — hidden state before gate_proj + up_proj
    QKV input:   Attention pre-hook   — hidden state before q + k + v projections
    KV output:   k_proj / v_proj post-hooks — K, V values before GQA fanout
    """

    def __init__(self, model, path: str, n_bits: int):
        self.hooks: list = []
        layers = _get_decoder_layers(model)

        do_ffn  = path in ("broadcast_ffn_input",  "broadcast_all")
        do_qkv  = path in ("broadcast_qkv_input",  "broadcast_all")
        do_kv   = path in ("broadcast_kv_output",  "broadcast_all")

        def pre_hook_positional(nb):
            """For modules called with positional args (e.g. MLP)."""
            def _hook(module, inputs):
                if not inputs or not isinstance(inputs[0], torch.Tensor):
                    return inputs
                xq = e8m0_quantize(inputs[0].float(), nb).to(inputs[0].dtype)
                return (xq,) + inputs[1:]
            return _hook

        def pre_hook_kwargs(nb):
            """For modules called with keyword args (e.g. Qwen3Attention).
            with_kwargs=True gives (module, args, kwargs) signature."""
            def _hook(module, args, kwargs):
                # hidden_states may arrive as positional or keyword arg
                if args and isinstance(args[0], torch.Tensor):
                    xq = e8m0_quantize(args[0].float(), nb).to(args[0].dtype)
                    return (xq,) + args[1:], kwargs
                if "hidden_states" in kwargs and isinstance(kwargs["hidden_states"], torch.Tensor):
                    x = kwargs["hidden_states"]
                    kwargs = {**kwargs, "hidden_states": e8m0_quantize(x.float(), nb).to(x.dtype)}
                    return args, kwargs
                return args, kwargs
            return _hook

        def post_hook(nb):
            def _hook(module, inputs, output):
                if isinstance(output, torch.Tensor):
                    return e8m0_quantize(output.float(), nb).to(output.dtype)
                return output
            return _hook

        for layer in layers:
            mlp  = getattr(layer, "mlp",       None)
            attn = getattr(layer, "self_attn",  None)

            if do_ffn and mlp is not None:
                self.hooks.append(mlp.register_forward_pre_hook(pre_hook_positional(n_bits)))

            if do_qkv and attn is not None:
                self.hooks.append(
                    attn.register_forward_pre_hook(pre_hook_kwargs(n_bits), with_kwargs=True)
                )

            if do_kv and attn is not None:
                for proj_name in ("k_proj", "v_proj"):
                    proj = getattr(attn, proj_name, None)
                    if proj is not None:
                        self.hooks.append(proj.register_forward_hook(post_hook(n_bits)))

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


def exp_key(path: str, n_bits: int) -> str:
    return f"{path}_INT{n_bits}"


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
        for path in BROADCAST_PATHS:
            for nb in N_BITS:
                key = exp_key(path, nb)
                if key not in all_results[model_name]:
                    needed.append((key, path, nb))

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

        for key, path, nb in needed:
            if key in all_results[model_name]:
                continue
            tag = "baseline" if path is None else f"{path} INT{nb}"
            print(f"  -- {tag}", end="", flush=True)
            try:
                mgr = None
                if path is not None:
                    mgr = BroadcastQuantManager(model, path, nb)
                ppl = evaluate_ppl(model, tokenizer)
                if mgr: mgr.remove()
                all_results[model_name][key] = ppl
                base = all_results[model_name].get("baseline", ppl)
                suffix = f"  Δ{ppl - base:+.3f}" if path is not None else ""
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
        for path in BROADCAST_PATHS:
            for nb in N_BITS:
                key = exp_key(path, nb)
                val = variants.get(key)
                if not isinstance(val, (int, float)): continue
                rows.append({
                    "model": short, "path": path, "n_bits": nb,
                    "ppl": val, "ppl_base": baseline, "delta_ppl": val - baseline,
                })

    df = pd.DataFrame(rows)
    if df.empty:
        print("\nNo complete results yet.")
        return

    df.to_csv(os.path.join(OUT_DIR, "rq_broadcast_results.csv"), index=False)

    print("\n=== ΔPPL by broadcast path and bit width ===")
    piv = df.pivot_table(index=["model", "path"], columns="n_bits",
                         values="delta_ppl", aggfunc="first")
    print(piv.round(3).to_string())

    # ── Plot ───────────────────────────────────────────────────────────────────
    models = df["model"].unique()
    fig, axes = plt.subplots(1, len(models), figsize=(5 * len(models), 5), squeeze=False)
    fig.suptitle("RQ-Broadcast: Optical Broadcast Path Precision Analysis\n"
                 "ΔPPL vs BF16 baseline, E8M0 per-token FXP at broadcast point",
                 fontsize=11, fontweight="bold")

    path_colors = {
        "broadcast_ffn_input":  "steelblue",
        "broadcast_qkv_input":  "darkorange",
        "broadcast_kv_output":  "green",
        "broadcast_all":        "crimson",
    }
    path_labels = {
        "broadcast_ffn_input":  "FFN gate+up input",
        "broadcast_qkv_input":  "QKV shared input",
        "broadcast_kv_output":  "K/V post-proj (GQA)",
        "broadcast_all":        "All broadcast paths",
    }

    for ci, mdl in enumerate(models):
        ax = axes[0][ci]
        sub = df[df["model"] == mdl]
        for path in BROADCAST_PATHS:
            psub = sub[sub["path"] == path].sort_values("n_bits")
            if psub.empty: continue
            ax.plot(psub["n_bits"], psub["delta_ppl"],
                    marker="o", linewidth=2,
                    color=path_colors[path], label=path_labels[path])
        ax.axhline(0, color="black", linewidth=0.8, linestyle=":")
        ax.axhspan(0, 0.2,  alpha=0.12, color="green",  label="near-lossless (<0.2)")
        ax.axhspan(0, 3.0,  alpha=0.05, color="orange")
        ax.set_title(f"{mdl}", fontsize=9, fontweight="bold")
        ax.set_xlabel("Bit width N", fontsize=8)
        ax.set_ylabel("ΔPPL vs BF16", fontsize=8)
        ax.set_xticks(N_BITS)
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    p = os.path.join(OUT_DIR, "rq_broadcast.png")
    plt.savefig(p, dpi=180, bbox_inches="tight")
    plt.close()
    print(f"\nPlot: {p}")
    print(f"Results: {OUT_DIR}/")


if __name__ == "__main__":
    main()
