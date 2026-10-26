"""
RQ-AllOn — Full System Composition Experiment
==============================================
Central validation: all non-idealities active simultaneously.

Configs (the core paper table):
  baseline       BF16 weights, no quantization anywhere
  quant_only     UQ3.2 weights + A6-b16 QKVO activations, no transport/compute quant
  transport_only BF16 weights, INT10 residual + broadcast precision assignment
  full_ideal     UQ3.2 + A6-b16 + INT10 transport + broadcast + INT8 compute, no noise
  full_s001      full_ideal + σ=0.01 noise  ← hardware-feasible under calibration
  full_s003      full_ideal + σ=0.03 noise  ← expected degradation (raw hardware)

Hardware pipeline order (each token, each layer):
  1. Noisy weight read (σ multiplicative, models RRAM cell mismatch)
  2. Optical QKV broadcast: INT8 E8M0 at self_attn input
  3. A6-b16 activation re-encode at each QKVO projection input (RRAM format)
  4. INT8 FXP on QKV output, attn_io, ffn_hidden (compute-internal)
  5. Optical KV fanout: INT8 E8M0 on k/v proj outputs (GQA broadcast)
  6. Optical FFN broadcast: INT10 E8M0 at MLP input
  7. INT10 E8M0 residual transport at layer output (inter-block photonic link)

Weight quantization: block-16 symmetric 5-bit on all RRAM-mapped projection weights.
This approximates UQ3.2 (MXFP4) without requiring torchao.

Outputs: results/rq_allon/
"""

import os, sys, gc, json
os.environ["CUDA_VISIBLE_DEVICES"] = "4"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
import torch.nn.functional as F
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import Optional
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset

# ── Configuration ──────────────────────────────────────────────────────────────

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_LIST = [
    "Qwen/Qwen3-1.7B",
    "Qwen/Qwen3-4B",
    "Qwen/Qwen3-8B",
]

RRAM_LAYERS = {"q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"}
QKVO_LAYERS = {"q_proj", "k_proj", "v_proj", "o_proj"}

SEQ_LEN     = 512
MAX_SAMPLES = 32

OUT_DIR     = os.path.join(_SCRIPT_DIR, "results", "rq_allon")
OUTPUT_JSON = os.path.join(OUT_DIR, "rq_allon_results.json")
os.makedirs(OUT_DIR, exist_ok=True)


@dataclass
class SystemConfig:
    label:          str
    weight_bits:    Optional[int]   = None   # None = BF16; 5 = UQ3.2 approx
    act_bits:       Optional[int]   = None   # None = off; 6 = A6-b16 on QKVO
    transport_bits: Optional[int]   = None   # None = off; 10 = INT10 E8M0 residual
    broadcast:      bool            = False  # INT8 QKV/KV + INT10 FFN optical broadcast
    compute_bits:   Optional[int]   = None   # None = off; 8 = INT8 FXP inside attn+FFN
    noise_sigma:    float           = 0.0


CONFIGS = [
    SystemConfig("baseline"),
    SystemConfig("quant_only",     weight_bits=5, act_bits=6),
    SystemConfig("transport_only", transport_bits=10, broadcast=True),
    SystemConfig("full_ideal",     weight_bits=5, act_bits=6, transport_bits=10,
                                   broadcast=True, compute_bits=8),
    SystemConfig("full_s001",      weight_bits=5, act_bits=6, transport_bits=10,
                                   broadcast=True, compute_bits=8, noise_sigma=0.01),
    SystemConfig("full_s003",      weight_bits=5, act_bits=6, transport_bits=10,
                                   broadcast=True, compute_bits=8, noise_sigma=0.03),
]


# ── Quantizers ─────────────────────────────────────────────────────────────────

def block_quant(x: torch.Tensor, bits: int, block_size: int = 16) -> torch.Tensor:
    """Symmetric per-block FXP, returns dequantized float."""
    orig = x.shape
    d = orig[-1]
    pad = (block_size - d % block_size) % block_size
    xp = F.pad(x, (0, pad)) if pad else x
    xb = xp.reshape(*xp.shape[:-1], -1, block_size)
    qmax = (1 << (bits - 1)) - 1
    scale = xb.abs().amax(dim=-1, keepdim=True).clamp(min=1e-8) / qmax
    xq = torch.clamp(torch.round(xb / scale), -qmax, qmax)
    return (xq * scale).reshape(*xp.shape[:-1], -1)[..., :d].reshape(orig)


def e8m0_quant(x: torch.Tensor, bits: int) -> torch.Tensor:
    """Per-token E8M0 FXP — barrel shift only, no FP multiply."""
    maxval = (1 << (bits - 1)) - 1
    amax   = x.abs().amax(dim=-1, keepdim=True).clamp(min=1e-8)
    scale  = torch.pow(2.0, torch.floor(torch.log2(amax)) - (bits - 2))
    return torch.clamp(torch.round(x / scale), -maxval, maxval) * scale


# ── Weight quantization (in-place, restored after eval) ───────────────────────

class WeightQuantManager:
    """Quantize all RRAM-mapped weights to block-16 N-bit symmetric FXP in-place."""

    def __init__(self, model, bits: int, block_size: int = 16):
        self.saved: dict[str, torch.Tensor] = {}
        for name, module in model.named_modules():
            if isinstance(module, torch.nn.Linear) and name.split(".")[-1] in RRAM_LAYERS:
                w = module.weight.data
                self.saved[name] = w.clone()
                module.weight.data = block_quant(w.float(), bits, block_size).to(w.dtype)

    def restore(self, model):
        for name, module in model.named_modules():
            if name in self.saved:
                module.weight.data = self.saved[name]
        self.saved.clear()


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


class NoiseManager:
    def __init__(self, model, sigma: float):
        self.hooks: list = []
        if sigma == 0.0:
            return
        for name, module in model.named_modules():
            if isinstance(module, torch.nn.Linear) and name.split(".")[-1] in RRAM_LAYERS:
                def _hook(mod, inp, sigma=sigma):
                    mod.weight.data.mul_(1.0 + torch.randn_like(mod.weight) * sigma)
                self.hooks.append(module.register_forward_pre_hook(_hook))

    def remove(self):
        for h in self.hooks: h.remove()
        self.hooks.clear()


class ActivationQuantManager:
    """A6 per-block-16 on QKVO projection inputs."""
    def __init__(self, model, bits: int):
        self.hooks: list = []
        for layer in _get_decoder_layers(model):
            attn = getattr(layer, "self_attn", None)
            if attn is None: continue
            for pname in QKVO_LAYERS:
                proj = getattr(attn, pname, None)
                if proj is None: continue
                def _hook(mod, inputs, nb=bits):
                    x = inputs[0]
                    return (block_quant(x.float(), nb).to(x.dtype),) + inputs[1:]
                self.hooks.append(proj.register_forward_pre_hook(_hook))

    def remove(self):
        for h in self.hooks: h.remove()
        self.hooks.clear()


class BroadcastManager:
    """INT8 E8M0 on QKV input + KV output; INT10 E8M0 on FFN input."""
    def __init__(self, model):
        self.hooks: list = []
        for layer in _get_decoder_layers(model):
            attn = getattr(layer, "self_attn", None)
            mlp  = getattr(layer, "mlp",       None)

            # QKV input: INT8 E8M0 (post-LN1 hidden state → all three projections)
            if attn is not None:
                def _attn_pre(mod, args, kwargs, bits=8):
                    if args and isinstance(args[0], torch.Tensor):
                        xq = e8m0_quant(args[0].float(), bits).to(args[0].dtype)
                        return (xq,) + args[1:], kwargs
                    if "hidden_states" in kwargs:
                        x = kwargs["hidden_states"]
                        return args, {**kwargs, "hidden_states": e8m0_quant(x.float(), bits).to(x.dtype)}
                    return args, kwargs
                self.hooks.append(attn.register_forward_pre_hook(_attn_pre, with_kwargs=True))

                # KV output: INT8 E8M0 (K/V post-projection → GQA head groups)
                for pname in ("k_proj", "v_proj"):
                    proj = getattr(attn, pname, None)
                    if proj is not None:
                        def _kv_post(mod, inp, out, bits=8):
                            if isinstance(out, torch.Tensor):
                                return e8m0_quant(out.float(), bits).to(out.dtype)
                        self.hooks.append(proj.register_forward_hook(_kv_post))

            # FFN input: INT10 E8M0 (post-LN2 hidden state → gate + up projections)
            if mlp is not None:
                def _ffn_pre(mod, inputs, bits=10):
                    if not inputs or not isinstance(inputs[0], torch.Tensor): return inputs
                    xq = e8m0_quant(inputs[0].float(), bits).to(inputs[0].dtype)
                    return (xq,) + inputs[1:]
                self.hooks.append(mlp.register_forward_pre_hook(_ffn_pre))

    def remove(self):
        for h in self.hooks: h.remove()
        self.hooks.clear()


class ComputeQuantManager:
    """INT8 FXP on QKV output, attention output (attn_io), and FFN hidden state."""
    def __init__(self, model, bits: int):
        self.hooks: list = []
        for layer in _get_decoder_layers(model):
            attn = getattr(layer, "self_attn", None)
            mlp  = getattr(layer, "mlp",       None)

            # QKV outputs: quantize after projection, before reshape to heads
            if attn is not None:
                for pname in ("q_proj", "k_proj", "v_proj"):
                    proj = getattr(attn, pname, None)
                    if proj is not None:
                        def _post(mod, inp, out, nb=bits):
                            if isinstance(out, torch.Tensor):
                                return e8m0_quant(out.float(), nb).to(out.dtype)
                        self.hooks.append(proj.register_forward_hook(_post))

                # Attention output: quantize o_proj input
                o_proj = getattr(attn, "o_proj", None)
                if o_proj is not None:
                    def _o_pre(mod, inputs, nb=bits):
                        x = inputs[0]
                        return (e8m0_quant(x.float(), nb).to(x.dtype),) + inputs[1:]
                    self.hooks.append(o_proj.register_forward_pre_hook(_o_pre))

            # FFN hidden state: quantize down_proj input
            if mlp is not None:
                down = getattr(mlp, "down_proj", None)
                if down is not None:
                    def _down_pre(mod, inputs, nb=bits):
                        x = inputs[0]
                        return (e8m0_quant(x.float(), nb).to(x.dtype),) + inputs[1:]
                    self.hooks.append(down.register_forward_pre_hook(_down_pre))

    def remove(self):
        for h in self.hooks: h.remove()
        self.hooks.clear()


class TransportManager:
    """INT10 E8M0 on decoder layer outputs (inter-block residual stream)."""
    def __init__(self, model, bits: int):
        self.hooks: list = []
        for layer in _get_decoder_layers(model):
            def _hook(mod, inp, out, nb=bits):
                hidden = out[0] if isinstance(out, tuple) else out
                q = e8m0_quant(hidden.float(), nb).to(hidden.dtype)
                return (q,) + out[1:] if isinstance(out, tuple) else q
            self.hooks.append(layer.register_forward_hook(_hook))

    def remove(self):
        for h in self.hooks: h.remove()
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

        needed = [c for c in CONFIGS if c.label not in all_results[model_name]]
        if not needed:
            print("  all variants cached, skipping")
            continue

        try:
            model = AutoModelForCausalLM.from_pretrained(
                model_name, torch_dtype=torch.bfloat16, device_map="auto"
            )
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            model.eval()
            print(f"  dtype={next(model.parameters()).dtype}  device={next(model.parameters()).device}")
        except Exception as e:
            print(f"  LOAD ERROR: {e}")
            all_results[model_name]["load_error"] = str(e)
            with open(OUTPUT_JSON, "w") as f:
                json.dump(all_results, f, indent=2)
            continue

        for cfg in needed:
            if cfg.label in all_results[model_name]:
                continue
            print(f"  -- {cfg.label}", end="", flush=True)
            managers = []
            wq = None
            try:
                if cfg.weight_bits:
                    wq = WeightQuantManager(model, cfg.weight_bits)
                if cfg.noise_sigma > 0:
                    managers.append(NoiseManager(model, cfg.noise_sigma))
                if cfg.broadcast:
                    managers.append(BroadcastManager(model))
                if cfg.act_bits:
                    managers.append(ActivationQuantManager(model, cfg.act_bits))
                if cfg.compute_bits:
                    managers.append(ComputeQuantManager(model, cfg.compute_bits))
                if cfg.transport_bits:
                    managers.append(TransportManager(model, cfg.transport_bits))

                ppl = evaluate_ppl(model, tokenizer)

                for m in managers: m.remove()
                if wq: wq.restore(model)

                all_results[model_name][cfg.label] = ppl
                base = all_results[model_name].get("baseline", ppl)
                suffix = f"  Δ{ppl - base:+.3f}" if cfg.label != "baseline" else ""
                print(f"  PPL={ppl:.3f}{suffix}")
            except Exception as e:
                for m in managers: m.remove()
                if wq: wq.restore(model)
                print(f"  ERROR: {e}")
                all_results[model_name][cfg.label] = {"error": str(e)}
            with open(OUTPUT_JSON, "w") as f:
                json.dump(all_results, f, indent=2)

        del model
        gc.collect()
        torch.cuda.empty_cache()

    # ── Summary table ──────────────────────────────────────────────────────────
    rows = []
    for model_name, variants in all_results.items():
        if "load_error" in variants: continue
        short    = model_name.split("/")[-1]
        baseline = variants.get("baseline")
        if not isinstance(baseline, (int, float)): continue
        for cfg in CONFIGS:
            val = variants.get(cfg.label)
            if not isinstance(val, (int, float)): continue
            rows.append({
                "model": short, "config": cfg.label,
                "ppl": val, "delta_ppl": val - baseline,
            })

    df = pd.DataFrame(rows)
    if df.empty:
        print("\nNo complete results yet.")
        return

    df.to_csv(os.path.join(OUT_DIR, "rq_allon_results.csv"), index=False)

    print("\n=== Full system composition: ΔPPL vs BF16 baseline ===")
    piv = df.pivot_table(index="config", columns="model", values="delta_ppl", aggfunc="first")
    cfg_order = [c.label for c in CONFIGS]
    piv = piv.reindex([c for c in cfg_order if c in piv.index])
    print(piv.round(3).to_string())

    # ── Plot ───────────────────────────────────────────────────────────────────
    models = df["model"].unique()
    x = range(len(CONFIGS))
    labels = [c.label for c in CONFIGS]
    fig, axes = plt.subplots(1, len(models), figsize=(5 * len(models), 5), squeeze=False)
    fig.suptitle("RQ-AllOn: Full System Composition\n"
                 "ΔPPL vs BF16 — all non-idealities active simultaneously",
                 fontsize=11, fontweight="bold")

    for ci, mdl in enumerate(models):
        ax = axes[0][ci]
        sub = df[df["model"] == mdl].set_index("config")
        vals = [sub.loc[c.label, "delta_ppl"] if c.label in sub.index else float("nan")
                for c in CONFIGS]
        colors = ["steelblue" if v <= 0.2 else "darkorange" if v <= 3.0 else "crimson"
                  for v in vals]
        bars = ax.bar(x, vals, color=colors, alpha=0.8, edgecolor="black", linewidth=0.5)
        ax.axhline(0,   color="black",  linewidth=0.8, linestyle=":")
        ax.axhline(0.2, color="green",  linewidth=1.0, linestyle="--", label="near-lossless (0.2)")
        ax.axhline(3.0, color="orange", linewidth=1.0, linestyle="--", label="acceptable (3.0)")
        ax.set_xticks(list(x))
        ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=7)
        ax.set_title(mdl, fontsize=9, fontweight="bold")
        ax.set_ylabel("ΔPPL vs BF16", fontsize=8)
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    p = os.path.join(OUT_DIR, "rq_allon.png")
    plt.savefig(p, dpi=180, bbox_inches="tight")
    plt.close()
    print(f"\nPlot: {p}")
    print(f"Results: {OUT_DIR}/")


if __name__ == "__main__":
    main()
