"""
RQ1 — FP4 Model Feasibility
Can LLM linear layers use plain FP4 weights (and optionally activations) while
preserving acceptable perplexity and task accuracy?

Variants:
  baseline  : BF16, no quantization
  W4A16     : FP4 weights (per-output-channel scale), BF16 activations
  W4A4      : FP4 weights + per-token FP4 activations on all linear layers

Metrics: wikitext word-PPL, piqa (0-shot acc), arc_easy (0-shot acc)

NOTE: Use BASE models only — instruct/chat models give ~100K PPL on raw wikitext
because they expect a chat template, masking all quantization effects.

Outputs: results/rq1/
"""
import os, sys, json, gc

# MUST be set before torch import — CUDA initialises on first import and ignores later changes.
os.environ["CUDA_VISIBLE_DEVICES"] = "4"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
sys.path.insert(0, os.path.dirname(__file__))

import torch
import pandas as pd
import matplotlib.pyplot as plt
from transformers import AutoModelForCausalLM, AutoTokenizer
from lm_eval import simple_evaluate
from lm_eval.models.huggingface import HFLM
from src.fp4_utils import (apply_fp4_weights, FP4ActivationQuantizer,
                           INT8ActivationQuantizer, NVFP4ActivationQuantizer,
                           TrueNVFP4ActivationQuantizer)

# ── Configuration ─────────────────────────────────────────────────────────────

# BASE models only. Confirmed working: Qwen/Qwen2.5-1.5B → ~11.3 wikitext PPL.
# Add larger base models as available (Qwen/Qwen2.5-7B, meta-llama/Meta-Llama-3.1-8B).
MODEL_LIST = [
    "Qwen/Qwen2.5-1.5B",          # fast sanity check — confirmed baseline ~11.3 PPL
    # "Qwen/Qwen2.5-7B",           # uncomment when downloaded
    # "meta-llama/Meta-Llama-3.1-8B",
]

TASKS    = ["wikitext", "piqa", "arc_easy"]
VARIANTS = ["baseline", "W4A16", "WFP4AINT8", "W4A4", "MXFP4_W4A4_B16", "NVFP4_W4A4_B16"]

# Remap old JSON keys to current variant names so skip logic works on existing results
_KEY_REMAP = {"W4A8": "WFP4AINT8", "NVFP4_W4A4": "MXFP4_W4A4_B16"}
OUT_DIR  = "results/rq1"
os.makedirs(OUT_DIR, exist_ok=True)


# ── Evaluation ────────────────────────────────────────────────────────────────

def run_variant(model, tokenizer, variant: str) -> dict:
    act_quant = None

    if variant in ("W4A16", "WFP4AINT8", "W4A4"):
        apply_fp4_weights(model)                        # per-output-channel FP4 weights
    elif variant == "MXFP4_W4A4_B16":
        apply_fp4_weights(model, block_size=16, scale_fmt='e8m0')  # E8M0 block-16 weights
    elif variant == "NVFP4_W4A4_B16":
        apply_fp4_weights(model, block_size=16, scale_fmt='e4m3')  # E4M3 block-16 weights

    if variant == "WFP4AINT8":
        act_quant = INT8ActivationQuantizer(model)      # per-token INT8 activations
    elif variant == "W4A4":
        act_quant = FP4ActivationQuantizer(model)       # per-token FP4 activations
    elif variant == "MXFP4_W4A4_B16":
        act_quant = NVFP4ActivationQuantizer(model)     # per-block-16 E8M0 FP4 activations
    elif variant == "NVFP4_W4A4_B16":
        act_quant = TrueNVFP4ActivationQuantizer(model) # per-block-16 E4M3 FP4 activations

    lm  = HFLM(pretrained=model, tokenizer=tokenizer, batch_size=1, max_length=2048)
    res = simple_evaluate(model=lm, tasks=TASKS, num_fewshot=0, batch_size=1)

    if act_quant:
        act_quant.remove()
    del lm
    gc.collect()
    torch.cuda.empty_cache()
    return res["results"]


def main():
    all_results  = {}
    output_json  = os.path.join(OUT_DIR, "rq1_results.json")

    # Load existing results and remap old key names to current variant names
    if os.path.exists(output_json):
        with open(output_json) as f:
            raw = json.load(f)
        for model_name, variants in raw.items():
            all_results[model_name] = {}
            for k, v in variants.items():
                new_key = _KEY_REMAP.get(k, k)
                all_results[model_name][new_key] = v
        print(f"Loaded existing results: {list(next(iter(all_results.values())).keys())}")

    for model_name in MODEL_LIST:
        print(f"\n====== {model_name} ======")
        all_results[model_name] = {}

        for variant in VARIANTS:
            if variant in all_results.get(model_name, {}) and "error" not in all_results[model_name][variant]:
                ppl = all_results[model_name][variant].get("wikitext", {}).get("word_perplexity,none", "?")
                print(f"  -- {variant}  [skipped, cached PPL={ppl}]")
                continue
            print(f"  -- {variant}")
            try:
                model = AutoModelForCausalLM.from_pretrained(
                    model_name, dtype=torch.bfloat16, device_map="auto"
                )
                tokenizer = AutoTokenizer.from_pretrained(model_name)
                model.eval()

                p = next(model.parameters())
                print(f"     dtype={p.dtype}  device={p.device}")

                result = run_variant(model, tokenizer, variant)
                all_results[model_name][variant] = result

                ppl  = result.get("wikitext",  {}).get("word_perplexity,none", float("nan"))
                piqa = result.get("piqa",       {}).get("acc,none",             float("nan"))
                arc  = result.get("arc_easy",   {}).get("acc,none",             float("nan"))
                print(f"     PPL={ppl:.2f}  piqa={piqa:.3f}  arc_easy={arc:.3f}")

            except Exception as e:
                print(f"  ERROR: {e}")
                all_results[model_name][variant] = {"error": str(e)}
            finally:
                try:
                    del model
                    gc.collect()
                    torch.cuda.empty_cache()
                except Exception:
                    pass

            # Checkpoint after every variant
            with open(output_json, "w") as f:
                json.dump(all_results, f, indent=2)

    # ── Summary table ─────────────────────────────────────────────────────────
    rows = []
    for model_name, variants in all_results.items():
        for variant, res in variants.items():
            if "error" in res:
                continue
            rows.append({
                "model":         model_name,
                "variant":       variant,
                "ppl":           res.get("wikitext",  {}).get("word_perplexity,none", float("nan")),
                "piqa_acc":      res.get("piqa",       {}).get("acc,none",             float("nan")),
                "arc_easy_acc":  res.get("arc_easy",   {}).get("acc,none",             float("nan")),
            })

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(OUT_DIR, "rq1_results.csv"), index=False)
    print("\n\n=== SUMMARY ===")
    print(df.to_string(index=False))

    # ── PPL bar chart ─────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    for model_name in df["model"].unique():
        sub   = df[df["model"] == model_name]
        short = model_name.split("/")[-1]
        axes[0].bar([f"{short}\n{v}" for v in sub["variant"]], sub["ppl"], label=short)
        axes[1].bar([f"{short}\n{v}" for v in sub["variant"]], sub["piqa_acc"], label=short)

    axes[0].set_ylabel("Word Perplexity (wikitext, lower=better)")
    axes[0].set_title("RQ1 — FP4 Feasibility: PPL")
    axes[0].tick_params(axis="x", rotation=30)

    axes[1].set_ylabel("Accuracy (0-shot, higher=better)")
    axes[1].set_title("RQ1 — FP4 Feasibility: PIQA Accuracy")
    axes[1].tick_params(axis="x", rotation=30)
    axes[1].set_ylim(0, 1)

    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "rq1_feasibility.png"), dpi=200)
    print(f"\nResults saved to {OUT_DIR}/")


if __name__ == "__main__":
    main()
