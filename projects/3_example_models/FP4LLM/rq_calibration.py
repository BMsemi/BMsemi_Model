"""
RQ-Calibration — RRAM Noise Calibration Recovery
=================================================
Question: Can simple post-MAC gain correction recover quality lost to RRAM
conductance noise?

Noise model: w_eff = w * (1 + ε), ε ~ N(0, σ²)  (same as rq_noise.py)

Calibration strategies (oracle: we know W and W_eff, so this is the upper
bound on what any physical calibration scheme can achieve):
  none         — raw noisy weights, no correction (baseline from rq_noise.py)
  per_tile     — one Frobenius-norm scalar per 32×32 crossbar tile
  per_channel  — one L2-norm scalar per output channel (row of weight matrix)

σ values tested: 0.01, 0.03, 0.05  (σ=0.03 is the critical threshold)
Tile size: 32  (matches BM Labs physical crossbar)

Finding target: per_channel recovers σ=0.03 to ΔPPL < +0.2 → validates the
noise mitigation path and closes Layer-D of the reviewer validation framework.

Outputs: results/rq_calibration/
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
SIGMA_VALUES     = [0.01, 0.03, 0.05]
CALIB_STRATEGIES = ["none", "per_tile", "per_channel"]
TILE_SIZE        = 32   # BM Labs physical crossbar

SEQ_LEN     = 512
MAX_SAMPLES = 32

OUT_DIR     = os.path.join(_SCRIPT_DIR, "results", "rq_calibration")
OUTPUT_JSON = os.path.join(OUT_DIR, "rq_calibration_results.json")
os.makedirs(OUT_DIR, exist_ok=True)


# ── Noise + calibration ────────────────────────────────────────────────────────

def apply_noise_and_calibrate(w: torch.Tensor, sigma: float, strategy: str,
                               tile_size: int = TILE_SIZE) -> torch.Tensor:
    """
    Inject multiplicative noise then correct via oracle gain calibration.
    w must be 2-D (out_features, in_features).
    Oracle: we have access to both W and W_eff, so this is the upper bound
    on what any physical calibration implementation can achieve.
    """
    orig_dtype = w.dtype
    w   = w.float()
    eps = torch.randn_like(w) * sigma
    w_eff = w * (1.0 + eps)

    if strategy == "none":
        return w_eff.to(orig_dtype)

    if strategy == "per_channel":
        # One scalar per output channel: ratio of L2 norms (row-wise)
        nw  = w.norm(dim=1, keepdim=True).clamp(min=1e-8)
        nwe = w_eff.norm(dim=1, keepdim=True).clamp(min=1e-8)
        return (w_eff * (nw / nwe)).to(orig_dtype)

    if strategy == "per_tile":
        rows, cols = w.shape
        r_tiles = rows // tile_size
        c_tiles = cols // tile_size
        w_cal   = w_eff.clone()

        # Vectorized main block (avoids Python loops over thousands of tiles)
        if r_tiles > 0 and c_tiles > 0:
            Rm = r_tiles * tile_size
            Cm = c_tiles * tile_size
            # View: (r_tiles, tile_size, c_tiles, tile_size) — see note below
            # w[a*T+b, c*T+d] lands at view[a, b, c, d] due to row-major layout
            wm  = w    [:Rm, :Cm].view(r_tiles, tile_size, c_tiles, tile_size)
            wem = w_eff[:Rm, :Cm].view(r_tiles, tile_size, c_tiles, tile_size)
            nw  = wm .pow(2).sum(dim=(1, 3), keepdim=True).sqrt().clamp(min=1e-8)
            nwe = wem.pow(2).sum(dim=(1, 3), keepdim=True).sqrt().clamp(min=1e-8)
            w_cal[:Rm, :Cm] = (wem * (nw / nwe)).view(Rm, Cm)

        # Edge row strip (rows % tile_size != 0)
        if rows % tile_size:
            r0 = r_tiles * tile_size
            for c in range(0, cols, tile_size):
                wb  = w    [r0:, c:c+tile_size]
                web = w_eff[r0:, c:c+tile_size]
                s   = wb.norm().item() / web.norm().clamp(min=1e-8).item()
                w_cal[r0:, c:c+tile_size] = web * s

        # Edge col strip (cols % tile_size != 0)
        if cols % tile_size:
            c0 = c_tiles * tile_size
            for r in range(0, r_tiles * tile_size, tile_size):
                wb  = w    [r:r+tile_size, c0:]
                web = w_eff[r:r+tile_size, c0:]
                s   = wb.norm().item() / web.norm().clamp(min=1e-8).item()
                w_cal[r:r+tile_size, c0:] = web * s

        return w_cal.to(orig_dtype)

    return w_eff.to(orig_dtype)


# ── Weight patch/restore manager ───────────────────────────────────────────────

class CalibNoiseManager:
    """
    Patches RRAM projection weights in-place (noise + calibration).
    Restores original weights on .remove().
    Seed is fixed so repeated calls with the same σ give identical noise
    (comparable to rq_noise.py which also uses a fixed seed).
    """

    def __init__(self, model, sigma: float, strategy: str,
                 tile_size: int = TILE_SIZE, seed: int = 42):
        self._saved:  dict = {}  # id(param) → original data tensor
        self._params: list = []

        torch.manual_seed(seed)
        for mod_name, module in model.named_modules():
            proj = mod_name.split(".")[-1]
            if proj not in RRAM_PROJ_NAMES:
                continue
            if not hasattr(module, "weight"):
                continue
            w  = module.weight.data
            w2 = w.view(w.shape[0], -1)     # ensure 2-D
            wc = apply_noise_and_calibrate(w2, sigma, strategy, tile_size)
            self._saved[id(module.weight)] = w.clone()
            module.weight.data = wc.view_as(w)
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


def exp_key(sigma: float, strategy: str) -> str:
    return f"s{sigma:.2f}_{strategy}"


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
        for sigma in SIGMA_VALUES:
            for strategy in CALIB_STRATEGIES:
                key = exp_key(sigma, strategy)
                if key not in all_results[model_name]:
                    needed.append((key, sigma, strategy))

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

        for key, sigma, strategy in needed:
            if key in all_results[model_name]:
                continue
            tag = "baseline" if sigma is None else f"σ={sigma:.2f} {strategy}"
            print(f"  -- {tag}", end="", flush=True)
            mgr = None
            try:
                if sigma is not None:
                    mgr = CalibNoiseManager(model, sigma, strategy)
                ppl = evaluate_ppl(model, tokenizer)
                if mgr:
                    mgr.remove()
                all_results[model_name][key] = ppl
                base   = all_results[model_name].get("baseline", ppl)
                suffix = f"  Δ{ppl - base:+.3f}" if sigma is not None else ""
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
        for sigma in SIGMA_VALUES:
            for strategy in CALIB_STRATEGIES:
                key = exp_key(sigma, strategy)
                val = variants.get(key)
                if not isinstance(val, (int, float)):
                    continue
                rows.append({
                    "model":     short,
                    "sigma":     sigma,
                    "strategy":  strategy,
                    "ppl":       val,
                    "ppl_base":  baseline,
                    "delta_ppl": val - baseline,
                })

    df = pd.DataFrame(rows)
    if df.empty:
        print("\nNo complete results yet.")
        return

    df.to_csv(os.path.join(OUT_DIR, "rq_calibration_results.csv"), index=False)

    print("\n=== ΔPPL by σ and calibration strategy (averaged over models) ===")
    piv = df.pivot_table(index=["model", "sigma"], columns="strategy",
                         values="delta_ppl", aggfunc="first")
    print(piv.round(3).to_string())

    # Recovery ratio: how much of the noise damage does calibration remove?
    print("\n=== Recovery vs no-correction (%) ===")
    for model_name in df["model"].unique():
        sub = df[df["model"] == model_name]
        for sigma in SIGMA_VALUES:
            ssub = sub[sub["sigma"] == sigma]
            base_dmg = ssub[ssub["strategy"] == "none"]["delta_ppl"].values
            if not len(base_dmg) or base_dmg[0] <= 0:
                continue
            dmg = base_dmg[0]
            print(f"  {model_name}  σ={sigma:.2f}  raw_ΔPPL={dmg:+.3f}", end="")
            for strat in ["per_tile", "per_channel"]:
                row = ssub[ssub["strategy"] == strat]["delta_ppl"].values
                if len(row):
                    pct = 100.0 * (1.0 - row[0] / dmg)
                    print(f"  |  {strat}: {row[0]:+.3f} ({pct:.0f}% recovered)", end="")
            print()

    # ── Plot ───────────────────────────────────────────────────────────────────
    models = df["model"].unique()
    fig, axes = plt.subplots(1, len(models), figsize=(5 * len(models), 5), squeeze=False)
    fig.suptitle(
        "RQ-Calibration: Oracle Gain Correction for RRAM Noise\n"
        "ΔPPL vs BF16 baseline — multiplicative noise σ, wikitext-2",
        fontsize=11, fontweight="bold")

    strat_styles = {
        "none":        dict(color="crimson",    ls="-",  lw=2, label="No correction"),
        "per_tile":    dict(color="darkorange",  ls="--", lw=2, label="Per-tile (32×32)"),
        "per_channel": dict(color="steelblue",  ls="-.", lw=2, label="Per-channel"),
    }

    for ci, mdl in enumerate(models):
        ax  = axes[0][ci]
        sub = df[df["model"] == mdl]
        for strat, sty in strat_styles.items():
            ssub = sub[sub["strategy"] == strat].sort_values("sigma")
            if ssub.empty:
                continue
            ax.plot(ssub["sigma"], ssub["delta_ppl"], marker="o", **sty)
        ax.axhline(0,       color="black", linewidth=0.8, linestyle=":")
        ax.axhspan(0, 0.2,  alpha=0.12,   color="green",  label="near-lossless (<0.2)")
        ax.axvline(0.03,    color="gray",  linewidth=0.8, linestyle="--", label="σ=0.03 threshold")
        ax.set_title(mdl, fontsize=9, fontweight="bold")
        ax.set_xlabel("Noise σ", fontsize=8)
        ax.set_ylabel("ΔPPL vs BF16", fontsize=8)
        ax.set_xticks(SIGMA_VALUES)
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    fig_path = os.path.join(OUT_DIR, "rq_calibration.png")
    plt.savefig(fig_path, dpi=180, bbox_inches="tight")
    plt.close()
    print(f"\nPlot: {fig_path}")
    print(f"Results: {OUT_DIR}/")


if __name__ == "__main__":
    main()
