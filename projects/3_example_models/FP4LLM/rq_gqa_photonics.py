"""
RQ-GQA-Photonics — Optical Broadcast Benefit for GQA KV Fanout
===============================================================
Question: How much does optical broadcast reduce bandwidth and energy
for the GQA KV fanout path, and under what PE mesh conditions?

In GQA, one KV head group serves fanout = n_q_heads / n_kv_heads Q-head groups.
Without optical: the KV tensor is unicast to each Q-head PE group separately.
With optical:    one transmission, all Q-head PE groups receive simultaneously.

Hardware model (BM Labs specs, verified against llm_reram_executor.py):
  AXI_NEAR_ELECTRICAL:  32 B/cycle,  2.0 pJ/B/hop  (scales with distance)
  OPTICAL_ALLOUT:       256 B/cycle, 0.5 pJ/B       (flat, distance-independent)
  Optical trigger:      avg PE mesh hops > 4

Encoding:
  AXI path   — BF16 (2 B/element, no transport quantization on short-hop paths)
  Optical path — INT8 E8M0 (1 B/element, validated near-lossless in rq_broadcast.py
                             for broadcast_kv_output at INT8)
  This halves the optical byte count on top of the broadcast efficiency gain.

Analysis:
  1. GQA parameters per model  (AutoConfig, no weight download)
  2. KV traffic volume per layer per token  (K + V tensors)
  3. Energy and latency  under AXI fanout vs optical broadcast
     at avg hop counts [2, 4, 6, 8]  (covering sub- and super-threshold)
  4. Break-even hop count where optical saves energy even at fanout=1
  5. Total KV fanout cost across all layers for a full forward pass
  6. Comparison: what fraction of total NoC traffic the KV fanout represents
     (reference: 816B AXI cycles and 26TB AXI traffic for Llama 3.2 1B
      from BM Labs run in LLMMappingOptimizer.py)

Models: Qwen3-1.7B, Qwen3-4B, Qwen3-8B
Seq lengths: 512, 2048, 8192
Outputs: results/rq_gqa_photonics/
"""

import os, sys, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from transformers import AutoConfig

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# ── BM Labs hardware constants ─────────────────────────────────────────────────
AXI_BW      = 32       # B/cycle   (AXI_NEAR_ELECTRICAL)
AXI_E       = 2.0      # pJ/B/hop
OPT_BW      = 256      # B/cycle   (OPTICAL_ALLOUT)
OPT_E       = 0.5      # pJ/B  (flat, distance-independent)
OPT_TRIG    = 4        # hops  (> this → optical is triggered)

# INT8 E8M0 encoding on optical path: 1 B/element vs BF16 2 B/element
# Validated near-lossless for broadcast_kv_output at INT8 (rq_broadcast.py)
OPT_ENCODE  = 0.5      # byte reduction factor on optical path

DTYPE_BYTES = 2        # BF16 weights/activations
BATCH_SIZE  = 1        # decode-phase: batch=1

MODEL_LIST = [
    "Qwen/Qwen3-1.7B",
    "Qwen/Qwen3-4B",
    "Qwen/Qwen3-8B",
]

HOP_COUNTS  = [2, 4, 6, 8]
SEQ_LENGTHS = [512, 2048, 8192]

OUT_DIR     = os.path.join(_SCRIPT_DIR, "results", "rq_gqa_photonics")
OUTPUT_JSON = os.path.join(OUT_DIR, "rq_gqa_photonics_results.json")
os.makedirs(OUT_DIR, exist_ok=True)


# ── Config extraction ──────────────────────────────────────────────────────────

def get_gqa_config(model_name: str) -> dict:
    """Load model config (no weights) and extract GQA parameters."""
    cfg = AutoConfig.from_pretrained(model_name)
    n_q   = cfg.num_attention_heads
    n_kv  = cfg.num_key_value_heads
    hdim  = cfg.hidden_size // n_q
    return {
        "model":      model_name.split("/")[-1],
        "n_layers":   cfg.num_hidden_layers,
        "n_q_heads":  n_q,
        "n_kv_heads": n_kv,
        "head_dim":   hdim,
        "fanout":     n_q // n_kv,
        "hidden":     cfg.hidden_size,
    }


# ── Traffic and cost formulas ──────────────────────────────────────────────────

def kv_bytes_per_layer(n_kv_heads: int, head_dim: int, seq_len: int,
                        batch: int = BATCH_SIZE,
                        dtype_bytes: int = DTYPE_BYTES) -> int:
    """Bytes in K + V tensors for one layer."""
    return 2 * batch * n_kv_heads * seq_len * head_dim * dtype_bytes


def axi_fanout_cost(kv_bytes: int, fanout: int, avg_hops: float):
    """
    AXI unicast: send kv_bytes to each of `fanout` Q-head PE groups separately.
    Bandwidth: serialized unicasts through AXI links (32 B/cycle each).
    Energy: each byte traverses avg_hops links at 2.0 pJ/B/hop.
    """
    total_bytes  = kv_bytes * fanout
    energy_pj    = total_bytes * AXI_E * avg_hops
    latency_cyc  = total_bytes / AXI_BW           # serialized sends, bandwidth-bound
    return energy_pj, latency_cyc, total_bytes


def opt_broadcast_cost(kv_bytes: int, encode: float = OPT_ENCODE):
    """
    Optical broadcast: one transmission, all Q-head groups receive simultaneously.
    INT8 E8M0 encoding (validated rq_broadcast.py) halves the byte count.
    Energy: flat 0.5 pJ/B regardless of distance or fanout.
    """
    tx_bytes     = kv_bytes * encode
    energy_pj    = tx_bytes * OPT_E
    latency_cyc  = tx_bytes / OPT_BW
    return energy_pj, latency_cyc, tx_bytes


def reduction_factors(kv_bytes: int, fanout: int, avg_hops: float):
    """Energy and latency reduction: AXI / optical."""
    ae, al, ab = axi_fanout_cost(kv_bytes, fanout, avg_hops)
    oe, ol, ob = opt_broadcast_cost(kv_bytes)
    return ae / oe, al / ol, ab / ob   # energy_rx, latency_rx, bytes_rx


# ── Break-even analysis ────────────────────────────────────────────────────────

def breakeven_hops(fanout: int, encode: float = OPT_ENCODE) -> float:
    """
    Smallest avg_hops at which optical saves energy (even before fanout is counted).
    Solve: fanout * AXI_E * hops > encode * OPT_E
    => hops > (encode * OPT_E) / (fanout * AXI_E)
    For encode=0.5, OPT_E=0.5, AXI_E=2.0:
    hops > 0.25 / fanout  (always < 1 → optical always wins on energy for any hops)
    """
    return (encode * OPT_E) / (fanout * AXI_E)


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    # ── 1. GQA configs ─────────────────────────────────────────────────────────
    configs = []
    print("=== GQA Configuration ===")
    print(f"{'Model':20s} {'layers':>6} {'Q heads':>7} {'KV heads':>8} "
          f"{'head_dim':>8} {'fanout':>6} {'hidden':>7}")
    for model_name in MODEL_LIST:
        try:
            c = get_gqa_config(model_name)
            configs.append(c)
            print(f"  {c['model']:18s} {c['n_layers']:6d} {c['n_q_heads']:7d} "
                  f"{c['n_kv_heads']:8d} {c['head_dim']:8d} {c['fanout']:6d} "
                  f"{c['hidden']:7d}")
        except Exception as e:
            print(f"  {model_name}: ERROR {e}")

    if not configs:
        print("No configs loaded.")
        return

    # ── 2. KV traffic per layer ────────────────────────────────────────────────
    print("\n=== KV Bytes per Layer (K+V, BF16, batch=1) ===")
    print(f"{'Model':20s}", end="")
    for sl in SEQ_LENGTHS:
        print(f"  seq={sl:5d}", end="")
    print()
    for c in configs:
        print(f"  {c['model']:18s}", end="")
        for sl in SEQ_LENGTHS:
            kb = kv_bytes_per_layer(c["n_kv_heads"], c["head_dim"], sl)
            print(f"  {kb/1024:8.1f} KB", end="")
        print()

    # ── 3. Energy and latency reduction table ──────────────────────────────────
    print("\n=== Energy Reduction: AXI fanout → Optical broadcast (INT8 encoded) ===")
    print(f"  Formula: fanout × AXI_E × hops / (OPT_ENCODE × OPT_E) = fanout × hops × 4 / OPT_ENCODE")
    print(f"\n  {'Model':18s} {'fanout':>6}  ", end="")
    for h in HOP_COUNTS:
        print(f"  hops={h}", end="")
    print()
    rows = []
    for c in configs:
        fanout = c["fanout"]
        print(f"  {c['model']:18s} {fanout:6d}  ", end="")
        for h in HOP_COUNTS:
            kb = kv_bytes_per_layer(c["n_kv_heads"], c["head_dim"], SEQ_LENGTHS[0])
            er, lr, br = reduction_factors(kb, fanout, h)
            print(f"  {er:5.0f}×  ", end="")
            rows.append({
                "model": c["model"], "fanout": fanout, "avg_hops": h,
                "energy_reduction": er, "latency_reduction": lr, "bytes_reduction": br,
            })
        print()

    print("\n=== Latency Reduction (bandwidth-bound, wormhole routing) ===")
    print(f"  Formula: fanout × (OPT_BW / AXI_BW) / OPT_ENCODE = fanout × 16")
    for c in configs:
        fanout = c["fanout"]
        kb = kv_bytes_per_layer(c["n_kv_heads"], c["head_dim"], SEQ_LENGTHS[0])
        _, lr, _ = reduction_factors(kb, fanout, HOP_COUNTS[0])  # hops don't affect latency
        print(f"  {c['model']:18s}  fanout={fanout}  latency_rx={lr:.0f}×  "
              f"(= {fanout} × {OPT_BW//AXI_BW} / {OPT_ENCODE:.1f})")

    # ── 4. Break-even ──────────────────────────────────────────────────────────
    print("\n=== Break-even: min hops for optical to save energy ===")
    for c in configs:
        be = breakeven_hops(c["fanout"])
        print(f"  {c['model']:18s}  fanout={c['fanout']}  "
              f"break-even at hops > {be:.3f}  "
              f"→ optical ALWAYS wins on energy (even at 1 hop)")

    # ── 5. Total KV fanout cost per forward pass ───────────────────────────────
    print("\n=== Total KV Fanout Cost: Full Forward Pass (seq=2048) ===")
    SL = 2048
    print(f"  {'Model':18s} {'layers':>6}  AXI energy (J)  OPT energy (J)  "
          f"Energy saved  AXI cycles   OPT cycles")
    for c in configs:
        kb = kv_bytes_per_layer(c["n_kv_heads"], c["head_dim"], SL)
        # Use hops=OPT_TRIG (the crossover point, where optical just triggers)
        ae, al, _ = axi_fanout_cost(kb, c["fanout"], OPT_TRIG)
        oe, ol, _ = opt_broadcast_cost(kb)
        total_axi_e  = ae * c["n_layers"] * 1e-12   # pJ → J
        total_opt_e  = oe * c["n_layers"] * 1e-12
        total_axi_cy = al * c["n_layers"]
        total_opt_cy = ol * c["n_layers"]
        print(f"  {c['model']:18s} {c['n_layers']:6d}  "
              f"{total_axi_e:13.6f}  {total_opt_e:13.6f}  "
              f"{total_axi_e/total_opt_e:10.0f}×  "
              f"{total_axi_cy:10.0f}  {total_opt_cy:8.0f}")

    # ── 6. Save results ────────────────────────────────────────────────────────
    df_rx = pd.DataFrame(rows)
    df_rx.to_csv(os.path.join(OUT_DIR, "reduction_factors.csv"), index=False)

    # Config table
    df_cfg = pd.DataFrame(configs)
    df_cfg.to_csv(os.path.join(OUT_DIR, "gqa_configs.csv"), index=False)

    # Full sweep: energy reduction per model × seq_len × hop_count
    sweep_rows = []
    for c in configs:
        for sl in SEQ_LENGTHS:
            kb = kv_bytes_per_layer(c["n_kv_heads"], c["head_dim"], sl)
            for h in HOP_COUNTS:
                ae, al, ab = axi_fanout_cost(kb, c["fanout"], h)
                oe, ol, ob = opt_broadcast_cost(kb)
                sweep_rows.append({
                    "model": c["model"], "fanout": c["fanout"],
                    "seq_len": sl, "avg_hops": h,
                    "kv_bytes": kb,
                    "axi_energy_pj": ae, "opt_energy_pj": oe,
                    "axi_latency_cyc": al, "opt_latency_cyc": ol,
                    "energy_rx": ae / oe, "latency_rx": al / ol,
                })
    df_sw = pd.DataFrame(sweep_rows)
    df_sw.to_csv(os.path.join(OUT_DIR, "full_sweep.csv"), index=False)

    with open(OUTPUT_JSON, "w") as f:
        json.dump({
            "configs": configs,
            "constants": {
                "AXI_BW": AXI_BW, "AXI_E": AXI_E,
                "OPT_BW": OPT_BW, "OPT_E": OPT_E,
                "OPT_ENCODE": OPT_ENCODE, "OPT_TRIG": OPT_TRIG,
            },
        }, f, indent=2)

    # ── 7. Plots ───────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    # ── Plot A: Energy reduction vs hop count, per model ──────────────────────
    ax = axes[0]
    colors = {"Qwen3-1.7B": "steelblue", "Qwen3-4B": "darkorange", "Qwen3-8B": "crimson"}
    for c in configs:
        sub = df_sw[(df_sw["model"] == c["model"]) & (df_sw["seq_len"] == 2048)]
        grp = sub.groupby("avg_hops")["energy_rx"].first().reset_index()
        ax.plot(grp["avg_hops"], grp["energy_rx"], marker="o", lw=2,
                color=colors.get(c["model"], "gray"),
                label=f"{c['model']} (fanout={c['fanout']})")
    ax.axvline(OPT_TRIG, color="gray", lw=0.8, ls="--", label=f"optical trigger (>{OPT_TRIG} hops)")
    ax.set_xlabel("Avg PE mesh hops (KV → Q-head group)", fontsize=9)
    ax.set_ylabel("Energy reduction (AXI / optical)", fontsize=9)
    ax.set_title("KV Fanout Energy Reduction\nvs Hop Count", fontsize=10, fontweight="bold")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # ── Plot B: Energy reduction vs fanout factor (synthetic, seq=2048, hops=4) ─
    ax = axes[1]
    fanouts_synth = [1, 2, 4, 8, 16]
    kb_ref = kv_bytes_per_layer(8, 128, 2048)   # reference: 8 KV heads, head_dim=128
    for h in HOP_COUNTS:
        ers = []
        for f in fanouts_synth:
            ae, _, _ = axi_fanout_cost(kb_ref, f, h)
            oe, _, _ = opt_broadcast_cost(kb_ref)
            ers.append(ae / oe)
        ax.plot(fanouts_synth, ers, marker="o", lw=2, label=f"hops={h}")
    ax.axhline(1, color="black", lw=0.8, ls=":")
    ax.axvspan(2, 8, alpha=0.08, color="green", label="typical GQA range")
    ax.set_xlabel("GQA fanout  (n_q_heads / n_kv_heads)", fontsize=9)
    ax.set_ylabel("Energy reduction (AXI / optical)", fontsize=9)
    ax.set_title("Energy Reduction vs Fanout Factor\n(seq=2048, INT8 optical encoding)",
                 fontsize=10, fontweight="bold")
    ax.set_xticks(fanouts_synth)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # ── Plot C: KV fanout bytes vs seq length — AXI vs optical (Qwen3-4B) ────
    ax = axes[2]
    c4b = next((c for c in configs if "4B" in c["model"]), configs[0])
    seq_range = np.array([128, 256, 512, 1024, 2048, 4096, 8192])
    axi_bytes_arr = [kv_bytes_per_layer(c4b["n_kv_heads"], c4b["head_dim"], sl)
                     * c4b["fanout"] / 1e6 for sl in seq_range]
    opt_bytes_arr = [kv_bytes_per_layer(c4b["n_kv_heads"], c4b["head_dim"], sl)
                     * OPT_ENCODE / 1e6 for sl in seq_range]
    ax.plot(seq_range, axi_bytes_arr, marker="o", lw=2, color="crimson",
            label=f"AXI unicast (×{c4b['fanout']} copies, BF16)")
    ax.plot(seq_range, opt_bytes_arr, marker="s", lw=2, color="steelblue",
            label="Optical broadcast (INT8 encoded)")
    ax.fill_between(seq_range, opt_bytes_arr, axi_bytes_arr, alpha=0.12, color="green",
                    label="Bytes saved")
    ax.set_xlabel("Sequence length", fontsize=9)
    ax.set_ylabel("KV fanout traffic (MB, per layer)", fontsize=9)
    ax.set_title(f"KV Fanout Traffic: AXI vs Optical\n{c4b['model']}",
                 fontsize=10, fontweight="bold")
    ax.set_xscale("log", base=2)
    ax.set_xticks(seq_range)
    ax.set_xticklabels([str(s) for s in seq_range], fontsize=7)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    fig.suptitle(
        "RQ-GQA-Photonics: Optical Broadcast for GQA KV Fanout\n"
        "BM Labs link model: AXI 2.0 pJ/B/hop vs Optical 0.5 pJ/B flat  |  "
        "INT8 E8M0 optical encoding (validated rq_broadcast.py)",
        fontsize=10, fontweight="bold")

    plt.tight_layout()
    fig_path = os.path.join(OUT_DIR, "rq_gqa_photonics.png")
    plt.savefig(fig_path, dpi=180, bbox_inches="tight")
    plt.close()
    print(f"\nPlot: {fig_path}")
    print(f"Results: {OUT_DIR}/")


if __name__ == "__main__":
    main()
