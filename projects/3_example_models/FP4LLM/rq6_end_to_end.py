"""
RQ6 — End-to-End Composition: RRAM Activation + Photonic Transport
===================================================================
Question: When 5-bit RRAM QKVO activation quantization and INT10/INT12 E8M0
residual transport are applied SIMULTANEOUSLY, does the combined system stay
within the quality budget?

This is the central result — both degradation sources are active at once,
as they would be in the real hardware.

Ablation structure (the key paper table):
  Condition            | QKVO activation quant | Transport quant
  ─────────────────────|──────────────────────|────────────────
  baseline             | none                  | none
  compute_only_A5      | A5 per-block-16       | none
  compute_only_A6      | A6 per-block-16       | none
  transport_only_INT10 | none                  | INT10 E8M0
  transport_only_INT12 | none                  | INT12 E8M0
  e2e_A5_INT10         | A5 per-block-16       | INT10 E8M0   ← aggressive target
  e2e_A5_INT12         | A5 per-block-16       | INT12 E8M0   ← safe transport
  e2e_A6_INT10         | A6 per-block-16       | INT10 E8M0   ← safe compute
  e2e_A6_INT12         | A6 per-block-16       | INT12 E8M0   ← fully safe

Quality thresholds:
  Near-lossless: ΔPPL < 0.2  (transport / residual standard)
  Acceptable:    ΔPPL < 3.0  (RRAM compute budget from RQ1)

Hooks do not interfere: activation pre-hooks fire inside each layer;
transport post-hook fires at layer output → correct hardware order.

Outputs: results/rq6_e2e/
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

# (key, act_bits, act_block, transport_bits)
# act_block=None → no activation quant; transport_bits=None → no transport quant
EXPERIMENTS = [
    ("compute_only_A5",      5,    16,   None),
    ("compute_only_A6",      6,    16,   None),
    ("transport_only_INT10", None, None, 10),
    ("transport_only_INT12", None, None, 12),
    ("e2e_A5_INT10",         5,    16,   10),
    ("e2e_A5_INT12",         5,    16,   12),
    ("e2e_A6_INT10",         6,    16,   10),
    ("e2e_A6_INT12",         6,    16,   12),
]

QKVO_PROJS = ["q_proj", "k_proj", "v_proj", "o_proj"]

SEQ_LEN     = 512
MAX_SAMPLES = 32

OUT_DIR     = os.path.join(_SCRIPT_DIR, "results", "rq6_e2e")
OUTPUT_JSON = os.path.join(OUT_DIR, "rq6_e2e_results.json")
os.makedirs(OUT_DIR, exist_ok=True)


# ── Quantizers ─────────────────────────────────────────────────────────────────

def per_block_fixedpoint(x: torch.Tensor, n_bits: int, block_size: int) -> torch.Tensor:
    """Symmetric FXP per block, real-valued scale (best-case activation quality)."""
    *batch, d = x.shape
    pad = (block_size - d % block_size) % block_size
    if pad:
        x = torch.nn.functional.pad(x, (0, pad))
    nb = x.shape[-1] // block_size
    xb = x.reshape(*batch, nb, block_size)
    maxval = (1 << (n_bits - 1)) - 1
    amax   = xb.abs().amax(dim=-1, keepdim=True).clamp(min=1e-8)
    scale  = amax / maxval
    xq     = torch.clamp(torch.round(xb / scale), -maxval, maxval) * scale
    out    = xq.reshape(*batch, x.shape[-1])
    return out[..., :d] if pad else out


def per_token_e8m0_fixedpoint(x: torch.Tensor, n_bits: int) -> torch.Tensor:
    """Per-token E8M0 power-of-2 scale — hardware-valid barrel-shift transport."""
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


class ActivationQuantManager:
    """Pre-hooks on QKVO projection inputs: quantizes activations before RRAM MVM."""

    def __init__(self, model, n_bits: int, block_size: int):
        self.hooks: list = []
        for layer in _get_decoder_layers(model):
            attn = getattr(layer, "self_attn", None)
            if attn is None:
                continue
            for pname in QKVO_PROJS:
                proj = getattr(attn, pname, None)
                if proj is None:
                    continue
                def _make_hook(nb, bs):
                    def _hook(module, inputs):
                        x = inputs[0]
                        q = per_block_fixedpoint(x.float(), nb, bs).to(x.dtype)
                        return (q,) + inputs[1:]
                    return _hook
                self.hooks.append(proj.register_forward_pre_hook(_make_hook(n_bits, block_size)))

    def remove(self):
        for h in self.hooks:
            h.remove()
        self.hooks.clear()


class TransportQuantManager:
    """Post-hooks on decoder layer outputs: E8M0 transport quantization."""

    def __init__(self, model, n_bits: int):
        self.hooks: list = []
        for layer in _get_decoder_layers(model):
            def _make_hook(nb):
                def _hook(module, inputs, output):
                    if isinstance(output, torch.Tensor):
                        return per_token_e8m0_fixedpoint(output.float(), nb).to(output.dtype)
                    hidden = output[0]
                    q    = per_token_e8m0_fixedpoint(hidden.float(), nb).to(hidden.dtype)
                    rest = output[1:] if isinstance(output, tuple) \
                           else tuple(output.to_tuple()[1:])
                    return (q,) + rest
                return _hook
            self.hooks.append(layer.register_forward_hook(_make_hook(n_bits)))

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


def run_experiment(model, tokenizer, act_bits, act_block, transport_bits):
    act_mgr  = ActivationQuantManager(model, act_bits, act_block) \
               if act_bits is not None else None
    trans_mgr = TransportQuantManager(model, transport_bits) \
                if transport_bits is not None else None
    ppl = evaluate_ppl(model, tokenizer)
    if act_mgr:   act_mgr.remove()
    if trans_mgr: trans_mgr.remove()
    return ppl


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
        for key, ab, ag, tb in EXPERIMENTS:
            if key not in all_results[model_name]:
                needed.append((key, ab, ag, tb))

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

        for key, ab, ag, tb in needed:
            if key in all_results[model_name]:
                continue
            print(f"  -- {key}", end="", flush=True)
            try:
                ppl = run_experiment(model, tokenizer, ab, ag, tb)
                all_results[model_name][key] = ppl
                base = all_results[model_name].get("baseline", ppl)
                suffix = f"  Δ{ppl - base:+.3f}" if key != "baseline" else ""
                print(f"  PPL={ppl:.3f}{suffix}")
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
        if "load_error" in variants: continue
        short    = model_name.split("/")[-1]
        baseline = variants.get("baseline")
        if not isinstance(baseline, (int, float)): continue
        for key, ab, ag, tb in EXPERIMENTS:
            val = variants.get(key)
            if not isinstance(val, (int, float)): continue
            rows.append({
                "model": short, "condition": key,
                "act_bits": ab, "act_gran": ag, "transport_bits": tb,
                "ppl": val, "ppl_base": baseline, "delta_ppl": val - baseline,
            })

    df = pd.DataFrame(rows)
    if df.empty:
        print("\nNo complete results yet.")
        return

    df.to_csv(os.path.join(OUT_DIR, "rq6_e2e_results.csv"), index=False)

    # ── Print ablation table ───────────────────────────────────────────────────
    print("\n\n=== RQ6 Ablation: ΔPPL vs BF16 baseline ===")
    print(f"{'Condition':<25} {'Activation':<14} {'Transport':<12}", end="")
    for mdl in df['model'].unique():
        print(f"  {mdl:>12}", end="")
    print()
    print("-" * 85)

    cond_order = [k for k, *_ in EXPERIMENTS]
    for key, ab, ag, tb in EXPERIMENTS:
        act_str   = f"A{ab}-b{ag}" if ab else "—"
        trans_str = f"INT{tb} E8M0" if tb else "—"
        print(f"  {key:<23} {act_str:<14} {trans_str:<12}", end="")
        for mdl in df['model'].unique():
            row = df[(df['model'] == mdl) & (df['condition'] == key)]
            if row.empty:
                print(f"  {'—':>12}", end="")
            else:
                d = row['delta_ppl'].values[0]
                print(f"  {d:>+11.3f}", end="")
        print()

    # ── Additivity check ──────────────────────────────────────────────────────
    print("\n=== Additivity check: e2e vs sum of components ===")
    print(f"{'Model':<14}  {'A5+INT10 e2e':>14}  {'A5+INT10 sum':>14}  {'overhead':>10}")
    for mdl in df['model'].unique():
        sub = df[df['model'] == mdl].set_index('condition')['delta_ppl']
        try:
            e2e = sub['e2e_A5_INT10']
            s   = sub.get('compute_only_A5', 0) + sub.get('transport_only_INT10', 0)
            print(f"  {mdl:<12}  {e2e:>+13.3f}  {s:>+13.3f}  {e2e - s:>+9.3f}")
        except KeyError:
            pass

    # ── Plot ──────────────────────────────────────────────────────────────────
    models = df['model'].unique()
    fig, axes = plt.subplots(1, len(models), figsize=(5 * len(models), 6), squeeze=False)
    fig.suptitle("RQ6: End-to-End Composition\nRRAM Activation Quant + E8M0 Transport — ΔPPL vs BF16",
                 fontsize=12, fontweight="bold")

    colors = {
        "compute_only_A5":      "#4e79a7",
        "compute_only_A6":      "#76b7b2",
        "transport_only_INT10": "#f28e2b",
        "transport_only_INT12": "#ffbe7d",
        "e2e_A5_INT10":         "#e15759",
        "e2e_A5_INT12":         "#ff9d9a",
        "e2e_A6_INT10":         "#59a14f",
        "e2e_A6_INT12":         "#8cd17d",
    }
    labels = {
        "compute_only_A5":      "Compute A5-b16 only",
        "compute_only_A6":      "Compute A6-b16 only",
        "transport_only_INT10": "Transport INT10 E8M0 only",
        "transport_only_INT12": "Transport INT12 E8M0 only",
        "e2e_A5_INT10":         "E2E: A5-b16 + INT10 (aggressive)",
        "e2e_A5_INT12":         "E2E: A5-b16 + INT12",
        "e2e_A6_INT10":         "E2E: A6-b16 + INT10",
        "e2e_A6_INT12":         "E2E: A6-b16 + INT12 (safe)",
    }

    x     = range(len(EXPERIMENTS))
    xlabs = [k for k, *_ in EXPERIMENTS]

    for ci, mdl in enumerate(models):
        ax  = axes[0][ci]
        sub = df[df['model'] == mdl].set_index('condition')['delta_ppl']
        vals  = [sub.get(k, float('nan')) for k, *_ in EXPERIMENTS]
        barcs = [colors[k] for k, *_ in EXPERIMENTS]
        bars  = ax.bar(x, vals, color=barcs, edgecolor='k', linewidth=0.5, alpha=0.88)
        for bar, v in zip(bars, vals):
            if not pd.isna(v):
                ax.text(bar.get_x() + bar.get_width()/2, v + 0.005,
                        f"{v:+.3f}", ha='center', va='bottom', fontsize=7.5)
        ax.axhline(0,    color='black', linewidth=0.8, linestyle=':')
        ax.axhline(0.2,  color='green', linewidth=1,   linestyle='--', alpha=0.6,
                   label='near-lossless (0.2)')
        ax.axhline(3.0,  color='orange', linewidth=1,  linestyle='--', alpha=0.6,
                   label='acceptable (3.0)')
        ax.set_title(mdl, fontsize=10, fontweight='bold')
        ax.set_xticks(list(x))
        ax.set_xticklabels(xlabs, rotation=40, ha='right', fontsize=7)
        ax.set_ylabel('ΔPPL vs BF16', fontsize=9)
        ax.legend(fontsize=7); ax.grid(True, axis='y', alpha=0.3)

    # Shared legend for bar colors
    from matplotlib.patches import Patch
    legend_els = [Patch(facecolor=colors[k], label=labels[k]) for k, *_ in EXPERIMENTS]
    fig.legend(handles=legend_els, loc='lower center', ncol=2, fontsize=8,
               bbox_to_anchor=(0.5, -0.18))
    plt.tight_layout()
    p = os.path.join(OUT_DIR, "rq6_e2e.png")
    plt.savefig(p, dpi=180, bbox_inches='tight')
    plt.close()
    print(f"\nPlot: {p}")
    print(f"Results: {OUT_DIR}/")


if __name__ == "__main__":
    main()
