"""
Photonic Interconnect Bandwidth Analysis — updated model set (2025 models)

Architecture note:
  Transported tensor is the residual stream [L, hidden_dim] at every boundary.
  - attn_to_mlp:    o_proj + residual → MLP input  (shape [L, hidden_dim])
  - layer_to_layer: down_proj + residual → next layer (shape [L, hidden_dim])
  lm_head (hidden_dim → vocab_size) fires once at model end, NOT between layers.

For MoE models the residual stream hidden_dim is still the model hidden_size;
the MoE expert intermediate dimension is NOT what crosses the photonic link.

Shift-encoding formats (BF16-shift, FP16-shift):
  One shared power-of-2 exponent (E8M0) is stored per token row.
  Dequantisation is a barrel shift — no FP multiplier.
  Overhead: exp_bits per token (8 for BF16, 5 for FP16).
  Total bits (prefill):  L * hidden_dim * bpe  +  L * exp_bits
  Empirical minimum from RQ-Transport: BF16→INT10, FP16→INT12.
"""
import os, pandas as pd
OUT_DIR = "results/photonic_bandwidth"
os.makedirs(OUT_DIR, exist_ok=True)

# ── Model configurations ──────────────────────────────────────────────────────
MODELS = [
    # Qwen3 dense — all BF16, vocab=151936 (v for 0.6B–8B via HF)
    ("Qwen3-0.6B",        1024,  28, "BF16", 151936, "dense"),
    ("Qwen3-1.7B",        2048,  28, "BF16", 151936, "dense"),
    ("Qwen3-4B",          2560,  36, "BF16", 151936, "dense"),
    ("Qwen3-8B",          4096,  36, "BF16", 151936, "dense"),
    ("Qwen3-14B",         5120,  40, "BF16", 151936, "dense"),
    ("Qwen3-32B",         5120,  64, "BF16", 151936, "dense"),
    # Qwen3 MoE
    ("Qwen3-30B-A3B",     2048,  48, "BF16", 151936, "MoE 128e/8a"),
    ("Qwen3-235B-A22B",   4096,  94, "BF16", 151936, "MoE 128e/8a"),
    # Gemma 4
    ("Gemma4-31B",        5376,  60, "BF16", 262144, "dense"),
    # GLM-Z1
    ("GLM-Z1-9B",         4096,  40, "BF16", 151552, "dense"),
    # Llama 3.1
    ("Llama-3.1-8B",      4096,  32, "BF16", 128256, "dense"),
    ("Llama-3.1-70B",     8192,  80, "BF16", 128256, "dense"),
    ("Llama-3.1-405B",   16384, 126, "BF16", 128256, "dense"),
    # Kimi-K2
    ("Kimi-K2",           7168,  61, "BF16", 163840, "MoE"),
    # DeepSeek-V3-0324
    ("DeepSeek-V3-0324",  7168,  61, "BF16", 129280, "MoE 256e/8a"),
]

CONTEXT_LENGTHS_PREFILL = [128,256,512,1024,2048,4096,8192,16384,32768,65536]

# FORMATS: name → (bpe, description, exp_overhead_per_token)
#   bpe                    = bits per element for payload
#   exp_overhead_per_token = extra bits added once per token row (0 for non-shift)
#   Total bits (L tokens)  = L*hidden_dim*bpe + L*exp_overhead_per_token
#
# Shift formats use one shared E8M0 exponent per token:
#   BF16-shift: empirical near-lossless (RQ-Transport INT10 knee), 8-bit E8M0 exp
#   FP16-shift: theoretical min for FP16 (10m+1s+1guard=12 bits), 5-bit E5 exp
FORMATS = {
    "FP16":       (16,  "float16 (1s+5e+10m)",                          0),
    "BF16":       (16,  "bfloat16 native (1s+8e+7m)",                   0),
    "BF16-shift": (10,  "INT10+E8M0 shift/token (RQ-Transport: near-lossless)", 8),
    "FP16-shift": (12,  "INT12+E5 shift/token (theoretical min, 10m+1s+1guard)", 5),
    "BF16-FXP":   ( 8,  "INT8+per-tensor scale (7m+1s, theoretical min)", 0),
    "INT8":       ( 8,  "INT8+per-tensor scale",                         0),
    "FP8-E4M3":   ( 8,  "float8 E4M3",                                  0),
    "INT4":       ( 4,  "INT4+per-tensor scale (FP4/MXFP4)",            0),
}

SCENARIOS = {"attn_to_mlp": "attn_to_mlp", "layer_to_layer": "layer_to_layer"}

rows = []
for name,hd,nl,dtype,vocab,notes in MODELS:
    for sc in SCENARIOS:
        nx = nl if sc=="attn_to_mlp" else nl-1
        for L in CONTEXT_LENGTHS_PREFILL:
            for fmt,(bpe,desc,exp_oh) in FORMATS.items():
                # payload bits + per-token exponent overhead
                bs = L * hd * bpe + L * exp_oh
                rows.append(dict(
                    model=name, hidden_dim=hd, num_layers=nl, dtype=dtype,
                    vocab=vocab, notes=notes, scenario=sc, mode="prefill",
                    seq_len=L, format=fmt, bpe=bpe, exp_overhead=exp_oh,
                    Mbits_single=bs/1e6, Mbits_total=bs*nx/1e6))
        for fmt,(bpe,desc,exp_oh) in FORMATS.items():
            # decode: L=1 token
            bs = hd * bpe + exp_oh
            rows.append(dict(
                model=name, hidden_dim=hd, num_layers=nl, dtype=dtype,
                vocab=vocab, notes=notes, scenario=sc, mode="decode",
                seq_len=1, format=fmt, bpe=bpe, exp_overhead=exp_oh,
                Mbits_single=bs/1e6, Mbits_total=bs*nx/1e6))

df = pd.DataFrame(rows)
df.to_csv(os.path.join(OUT_DIR,"photonic_bandwidth.csv"), index=False)
print(f"CSV saved: {len(df)} rows")

# ── Print summary tables ──────────────────────────────────────────────────────
def fmt_b(x):
    if x >= 1000: return f"{x/1000:.1f}Gb"
    if x >= 1:    return f"{x:.1f}Mb"
    return f"{x*1000:.0f}kb"

print("\n=== Model config table ===")
print(f"{'Model':<22} {'d':>5} {'L':>3} dtype  vocab    notes")
for name,hd,nl,dtype,vocab,notes in MODELS:
    print(f"{name:<22} {hd:>5} {nl:>3} {dtype}  {vocab:>6}  {notes}")

print("\n=== Prefill, single hop (layer_to_layer), Mbits — shift vs native ===")
sub = df[(df["format"].isin(["BF16","BF16-shift","FP16-shift","INT8","INT4"])) &
         (df["scenario"]=="layer_to_layer") & (df["mode"]=="prefill")]
piv = sub.pivot_table(index=["model","format"], columns="seq_len",
                      values="Mbits_single", aggfunc="first")
sel = [c for c in piv.columns if c in [512, 4096, 16384, 65536]]
print(piv[sel].applymap(lambda x: fmt_b(x)).to_string())

print("\n=== Shift format: exponent overhead fraction (per-token exp / total bits) ===")
print(f"{'Model':<22} {'d':>5} {'BF16-shift overhead':>22} {'FP16-shift overhead':>22}")
for name,hd,nl,dtype,vocab,notes in MODELS:
    bf_frac = 8 / (hd*10 + 8) * 100
    fp_frac = 5 / (hd*12 + 5) * 100
    print(f"{name:<22} {hd:>5}   {bf_frac:.4f}%                 {fp_frac:.4f}%")

print("\n=== Decode (L=1), single hop, all formats (layer_to_layer) — Mbits ===")
dec = df[(df["scenario"]=="layer_to_layer")&(df["mode"]=="decode")]
piv2 = dec.pivot_table(index="model", columns="format",
                       values="Mbits_single", aggfunc="first")
print(piv2.applymap(lambda x: fmt_b(x)).to_string())
