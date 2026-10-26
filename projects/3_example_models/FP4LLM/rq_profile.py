"""
Throughput & memory profiling: WFP4AINT8 vs NVFP4_W4A4

Measures for each variant on the same base model (Qwen/Qwen2.5-1.5B):
  - Peak VRAM at load time (model weights only)
  - Peak VRAM during inference
  - Prefill throughput (tokens/sec, prompt processing)
  - Decode throughput  (tokens/sec, autoregressive generation)
  - Time-to-first-token (TTFT)

Optionally also profiles an NVFP4 HuggingFace checkpoint via compressed-tensors
to show real quantized-weight memory footprint.

Outputs: results/profile/
"""
import os, sys, json, time, gc
os.environ["CUDA_VISIBLE_DEVICES"] = "4"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
sys.path.insert(0, os.path.dirname(__file__))

import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from transformers import AutoModelForCausalLM, AutoTokenizer
from src.fp4_utils import (apply_fp4_weights, INT8ActivationQuantizer,
                           TrueNVFP4ActivationQuantizer)

# ── Configuration ──────────────────────────────────────────────────────────────

BASE_MODEL   = "Qwen/Qwen2.5-1.5B"
# HF NVFP4 checkpoint — loaded separately to show real compressed weight memory.
# Set to None to skip (requires compressed-tensors and the model to be cached).
HF_NVFP4_MODEL = None   # skip: different model size/family, not a fair comparison

PROMPT_SHORT = "The quick brown fox jumps over the lazy dog."        # prefill bench
PROMPT_LONG  = PROMPT_SHORT * 32                                     # ~256 tokens
N_DECODE_TOKENS = 128    # tokens to generate for decode throughput
N_WARMUP        = 2      # warmup forward passes before timing
N_REPEATS       = 5      # timed repetitions
SEED            = 42

OUT_DIR = "results/profile"
os.makedirs(OUT_DIR, exist_ok=True)


# ── Helpers ────────────────────────────────────────────────────────────────────

def reset_peak():
    torch.cuda.reset_peak_memory_stats()

def peak_vram_gb() -> float:
    return torch.cuda.max_memory_allocated() / 1e9

def current_vram_gb() -> float:
    return torch.cuda.memory_allocated() / 1e9

def tokenize(tokenizer, text: str, device):
    return tokenizer(text, return_tensors="pt",
                     add_special_tokens=True).input_ids.to(device)


def bench_prefill(model, input_ids, n_warmup=N_WARMUP, n_repeats=N_REPEATS):
    """Return (mean_ms, std_ms, tokens_per_sec)."""
    device = next(model.parameters()).device
    ids = input_ids.to(device)
    n_tokens = ids.size(1)

    # warmup
    with torch.no_grad():
        for _ in range(n_warmup):
            _ = model(ids)
    torch.cuda.synchronize()

    times = []
    with torch.no_grad():
        for _ in range(n_repeats):
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            _ = model(ids)
            torch.cuda.synchronize()
            times.append(time.perf_counter() - t0)

    mean_s = np.mean(times)
    std_s  = np.std(times)
    return mean_s * 1e3, std_s * 1e3, n_tokens / mean_s   # ms, ms, tok/s


def bench_decode(model, tokenizer, prompt: str,
                 n_tokens=N_DECODE_TOKENS, n_warmup=N_WARMUP, n_repeats=N_REPEATS):
    """Return (ttft_ms, decode_tok_per_sec). Uses greedy generation."""
    device = next(model.parameters()).device
    input_ids = tokenize(tokenizer, prompt, device)

    # warmup
    with torch.no_grad():
        for _ in range(n_warmup):
            _ = model.generate(input_ids, max_new_tokens=n_tokens,
                                do_sample=False, use_cache=True)
    torch.cuda.synchronize()

    ttfts, decode_rates = [], []
    with torch.no_grad():
        for _ in range(n_repeats):
            torch.cuda.synchronize()
            t_start = time.perf_counter()
            # first token (prefill + 1 decode step)
            _ = model(input_ids)
            torch.cuda.synchronize()
            t_first = time.perf_counter()

            # full generation
            t_gen_start = time.perf_counter()
            out = model.generate(input_ids, max_new_tokens=n_tokens,
                                 do_sample=False, use_cache=True)
            torch.cuda.synchronize()
            t_gen_end = time.perf_counter()

            n_generated = out.size(1) - input_ids.size(1)
            ttfts.append((t_first - t_start) * 1e3)
            # decode rate = total tokens / total generation time
            decode_rates.append(n_generated / (t_gen_end - t_gen_start))

    return np.mean(ttfts), np.mean(decode_rates)


def profile_variant(model, tokenizer, variant_name: str) -> dict:
    device = next(model.parameters()).device
    short_prompt_ids = tokenize(tokenizer, PROMPT_SHORT, device)
    long_prompt_ids  = tokenize(tokenize, PROMPT_LONG,  device) if False else \
                       tokenize(tokenizer, PROMPT_LONG, device)

    reset_peak()
    vram_loaded = current_vram_gb()   # weight footprint

    print(f"  [{variant_name}] VRAM loaded: {vram_loaded:.2f} GB")

    # prefill with long prompt
    reset_peak()
    prefill_ms, prefill_std_ms, prefill_tps = bench_prefill(model, long_prompt_ids)
    vram_prefill = peak_vram_gb()
    print(f"  [{variant_name}] Prefill  {prefill_tps:.0f} tok/s  "
          f"({prefill_ms:.1f}±{prefill_std_ms:.1f} ms)  "
          f"peak VRAM {vram_prefill:.2f} GB")

    # decode
    reset_peak()
    ttft_ms, decode_tps = bench_decode(model, tokenizer, PROMPT_SHORT)
    vram_decode = peak_vram_gb()
    print(f"  [{variant_name}] TTFT {ttft_ms:.1f} ms  "
          f"Decode {decode_tps:.1f} tok/s  peak VRAM {vram_decode:.2f} GB")

    return {
        "variant":        variant_name,
        "vram_loaded_gb": round(vram_loaded,    3),
        "vram_prefill_gb":round(vram_prefill,   3),
        "vram_decode_gb": round(vram_decode,    3),
        "prefill_ms":     round(prefill_ms,     2),
        "prefill_std_ms": round(prefill_std_ms, 2),
        "prefill_tps":    round(prefill_tps,    1),
        "ttft_ms":        round(ttft_ms,        2),
        "decode_tps":     round(decode_tps,     1),
        "prompt_len_tok": long_prompt_ids.size(1),
    }


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    torch.manual_seed(SEED)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    results = []

    variants = [
        ("baseline",         lambda m: None),
        ("WFP4AINT8",        lambda m: (apply_fp4_weights(m),
                                        INT8ActivationQuantizer(m))),
        ("NVFP4_W4A4_B16",   lambda m: (apply_fp4_weights(m, block_size=16, scale_fmt='e4m3'),
                                        TrueNVFP4ActivationQuantizer(m))),
    ]

    for variant_name, setup_fn in variants:
        print(f"\n=== {variant_name} ===")
        model = AutoModelForCausalLM.from_pretrained(
            BASE_MODEL, dtype=torch.bfloat16, device_map="auto"
        )
        tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
        model.eval()

        hooks = setup_fn(model)   # apply quantization (may return quantizer handles)

        rec = profile_variant(model, tokenizer, variant_name)
        results.append(rec)

        # clean up hooks if any
        if isinstance(hooks, tuple):
            for h in hooks:
                if hasattr(h, "remove"):
                    h.remove()

        del model
        gc.collect()
        torch.cuda.empty_cache()

    # ── HF NVFP4 checkpoint (real weight compression) ─────────────────────────
    if HF_NVFP4_MODEL:
        print(f"\n=== HF_NVFP4 ({HF_NVFP4_MODEL.split('/')[-1]}) ===")
        try:
            model = AutoModelForCausalLM.from_pretrained(
                HF_NVFP4_MODEL, dtype=torch.bfloat16, device_map="auto"
            )
            tok2 = AutoTokenizer.from_pretrained(HF_NVFP4_MODEL)
            model.eval()
            rec = profile_variant(model, tok2, "HF_NVFP4")
            rec["note"] = f"real compressed weights from {HF_NVFP4_MODEL}"
            results.append(rec)
            del model
            gc.collect()
            torch.cuda.empty_cache()
        except Exception as e:
            print(f"  HF NVFP4 load failed: {e}")

    # ── Save & plot ────────────────────────────────────────────────────────────
    df = pd.DataFrame(results)
    df.to_csv(os.path.join(OUT_DIR, "profile_results.csv"), index=False)
    with open(os.path.join(OUT_DIR, "profile_results.json"), "w") as f:
        json.dump(results, f, indent=2)

    print("\n=== PROFILE SUMMARY ===")
    print(df[["variant", "vram_loaded_gb", "prefill_tps",
              "decode_tps", "ttft_ms"]].to_string(index=False))

    # plot
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    variants_plot = df["variant"].tolist()
    x = range(len(variants_plot))

    axes[0].bar(x, df["vram_loaded_gb"], color="steelblue")
    axes[0].set_title("Peak VRAM at load (GB)")
    axes[0].set_xticks(x); axes[0].set_xticklabels(variants_plot, rotation=20)

    axes[1].bar(x, df["prefill_tps"], color="darkorange")
    axes[1].set_title("Prefill throughput (tok/s)")
    axes[1].set_xticks(x); axes[1].set_xticklabels(variants_plot, rotation=20)

    axes[2].bar(x, df["decode_tps"], color="seagreen")
    axes[2].set_title("Decode throughput (tok/s)")
    axes[2].set_xticks(x); axes[2].set_xticklabels(variants_plot, rotation=20)

    plt.suptitle(f"WFP4AINT8 vs NVFP4_W4A4_B16 — {BASE_MODEL.split('/')[-1]}", fontsize=11)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "profile_throughput.png"), dpi=200)
    print(f"\nResults saved to {OUT_DIR}/")


if __name__ == "__main__":
    main()
