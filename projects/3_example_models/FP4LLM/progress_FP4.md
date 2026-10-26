# FP4 + CIM + Photonics — LLM Edge Inference Research Progress

## One-sentence Direction
We investigate plain FP4 as a hardware-native compute format for LLM linear layers by defining
the quantize/dequantize boundaries, mapping FP4 values into RRAM-compatible fixed-point
conductance levels, and characterizing the accumulation precision and sensing policy required
for photonic/CIM MACs to preserve model accuracy.

---

## Approach

### Three-layer co-design target

```
┌─────────────────────────────────────────────────────────────────┐
│  LLM Model Layer  (our software role)                           │
│  • FP4 E2M1 quantization of weights (per-output-channel scale) │
│  • Identify quantize/dequantize boundaries (RQ1, RQ2)          │
│  • Characterise weight/activation distributions (RQ7, RQ8)     │
└────────────────────────┬────────────────────────────────────────┘
                         │ FP4 codes (4-bit per weight)
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│  RRAM Crossbar Layer  (colleague hardware — Deep_gate + MEMRISYS)│
│  • 5-bit multi-level ReRAM cell (32 states, MEMRISYS-2025)      │
│  • FP4 magnitude stored as 5-bit Q-2 integer via barrel shift   │
│    ("1m" << e, max shift 3, levels {0,2,4,6,8,12,16,24})        │
│  • Sign in 1-bit SLC cell; row-serial digital CIM (Deep_gate)   │
│  • 9-bit ADC per column (single-row partial product ≤ 360)      │
│  • INT32 digital accumulator; 2× cell reduction vs INT8         │
│  • Noise/variation tolerance (RQ4)                              │
└────────────────────────┬────────────────────────────────────────┘
                         │ INT32 partial sums
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│  Photonic Accumulator Layer  (photonics colleague)              │
│  • Optical dot-product across crossbar columns                  │
│  • Required ENOB per layer type (RQ5, RQ6)                      │
│  • Row-serial decomposition relaxes photonic DR from ~21 bits   │
│    (full-column accumulation) to ~9 bits (single-row partial)   │
└─────────────────────────────────────────────────────────────────┘
```

### Hardware backing established (justifies proceeding)

| Claim | Evidence |
|-------|----------|
| FP4 in 4-bit cells is lossless | FP4 E2M1 is a 4-bit format; barrel-shift encodes all 8 magnitudes exactly in 5-bit Q-2 integers; verified numerically for all 16 codes |
| 4× cell reduction vs INT8 (sign-routed) | Baseline: Deep_gate INT8 = 4 cells/weight (pos0,pos1,neg0,neg1 nibble-sliced differential). FP4 with S⁺/S⁻ digital routing = 1 five-bit magnitude cell/weight (sign handled by XOR routing, not stored). If sign requires a separate SLC cell → 2 cells → 2× reduction. Claim requires explicit baseline. |
| 8–9-bit ADC suffices | Row-serial reduces per-pass max product to ≤24×15=360 (9-bit); Deep_gate verified with 8-bit ADC at INT8 |
| Digital CIM accuracy | Deep_gate INT8 pipeline verified at 0.0 max error on 5-test-case autoencoder |
| 5-bit ReRAM is ready | MEMRISYS-2025 demonstrates 32 well-separated resistance states; stable at 125°C; FP4 uses only 8 of 32 states (every 4th level), giving 4× noise margin |
| W4A16 FP4 is viable | RQ1: baseline 13.31 → W4A16 16.40 (+23% PPL); RQ2: baseline 11.26 → W4A16 13.97 |
| MXFP4_W4A4_B16 beats W4A16 | RQ1b: MXFP4_W4A4_B16 PPL=15.99 (+20%) < W4A16 PPL=16.40 (+23%); block-16 activation quantization is effectively free |
| WFP4AINT8 ≈ W4A16 | RQ1b: WFP4AINT8 PPL=16.83 vs W4A16=16.40, gap of 0.43; validates Deep_gate 8-bit ADC design |
| MXFP4 (E8M0) scale is shift-only | E8M0 block scale = power-of-2 = barrel shift; no multiplier needed; adds ≈6% register overhead per row |
| E8M0 scale sufficient vs E4M3 | RQ11: MXFP4-E8M0 PPL=13.486 vs NVFP4-E4M3 PPL=13.234, gap=0.252 < 0.5 threshold; barrel-shifter scale path is adequate |

### Block-Scale FP4 Extension Path (MXFP4 → True NVFP4)

```
Plain FP4 (current)
  Weight:  1 × 4-bit cell + 1-bit sign  →  2 cells/weight
  Scale:   1 per output channel (stored in digital logic)
  Decode:  barrel shift  →  5-bit integer  →  UQ3.2 output

MXFP4 / E8M0 block-16 (implemented, RQ1b/RQ9/RQ10)
  Weight:  same 4-bit cell storage
  Scale:   E8M0 per-16-element group  →  shift amount register (4 bits)
           global scale  →  one final shift
  Decode:  barrel shift weight  →  accumulate 16  →  group shift  →  sum groups  →  global shift
  Benefit: per-group calibration reduces per-tensor reconstruction error from 52–77% (RQ8 data) to ~44%

True NVFP4 / E4M3 block-16 (RQ11, new)
  Scale:   FP8 E4M3 per-16-element group  →  FP8→FP16 multiply (not just a shift)
  Benefit: ~5.8% mean scale error vs 44% for E8M0; PPL expected ~13.13 vs 13.49 (RQ11 in progress)
```

### What "our role" contributes to each colleague

**To RRAM colleague:**
- Per-layer FP4 code histograms → which conductance levels need tightest margin
- Per-output-channel scale values → conductance programming targets
- Layer-wise noise tolerance curve (RQ4) → acceptable σ_rel before PPL threshold

**To photonics colleague:**
- Per-layer **partial sum** distributions (min, max, std, 99.9th percentile, DR) → required ENOB (RQ5/RQ9)
- Layer sensitivity ranking → which layers need highest photonic precision (RQ6)
- Row-serial framing → reduces photonic DR requirement from 21 bits to 9 bits

---

## Run Queue (2026-05-07)

### Immediate — complete the current RQ set

| Priority | Command | Why | Est. time |
|----------|---------|-----|-----------|
| 🟡 1st | `python rq_profile.py` | Throughput & memory profiling | ~20 min |

### Deferred — paper completeness

| Priority | Command | Why | Est. time |
|----------|---------|-----|-----------|
| 🟡 3rd | `python rq4_rram_tolerance.py` | RRAM noise tolerance curve for RRAM colleague | ~20 min |
| 🟡 4th | `python rq5_rq6_photonic_precision.py` | ENOB spec for photonics colleague | ~30 min |
| 🟡 5th | `python rq7_rq8_distribution.py` | Distribution analysis + block scale comparison | ~45 min |
| 🟢 6th | `python rq_profile.py` | Throughput profiling | ~20 min |

---

## Research Questions & Status

| RQ | Question | Script | Status |
|----|----------|--------|--------|
| RQ1 | FP4 model feasibility (PPL + task accuracy) | `rq1_fp4_feasibility.py` | DONE ✓ |
| RQ2 | Quantize/dequantize boundary sweep | `rq2_quant_boundary.py` | DONE ✓ |
| RQ3 | RRAM conductance mapping + GEMM equivalence | `rq3_rram_mapping.py` | DONE ✓ |
| RQ1b | WFP4AINT8 + MXFP4_W4A4_B16 feasibility (extended) | `rq1_fp4_feasibility.py` | DONE ✓ |
| RQ4 | RRAM variation tolerance | `rq4_rram_tolerance.py` | DONE ✓ |
| RQ5 | Photonic accumulation ENOB / SNR / dynamic range | `rq5_rq6_photonic_precision.py` | DONE ✓ |
| RQ6 | Layer-wise sensitivity to accumulation precision | `rq5_rq6_photonic_precision.py` | DONE ✓ |
| RQ7 | Distribution-aware sensing (FP4/FP8/BF16/FP16) | `rq7_rq8_distribution.py` | DONE ✓ |
| RQ8 | Extension to NVFP4 / FP8 with block scaling | `rq7_rq8_distribution.py` | DONE ✓ |
| RQ9 | Partial sum overflow — accumulator width vs. PPL | `rq9_overflow.py` | DONE ✓ |
| RQ9b | Partial-sum growth by K + FP16 decoder equivalence | `rq9b_partial_sum_growth.py` | DONE ✓ |
| RQ9c | FP16 sub-sum decoder standalone cross-reference | `rq9c_fp16_decoder.py` | DONE ✓ |
| RQ10 | Subarray size K sweep — hardware vs. PPL | `rq10_subarray_k_sweep.py` | DONE ✓ |
| RQ11 | Block-scale format: E8M0 vs E4M3 vs FP32 | `rq11_scale_format_comparison.py` | DONE ✓ |
| RQP | Throughput & memory profiling: WFP4AINT8 vs NVFP4_W4A4_B16 | `rq_profile.py` | DONE ✓ |

**Naming convention note:** WFP4AINT8 = FP4 E2M1 weights + INT8 activations (ours).
This is distinct from "W4A8" models on HuggingFace which use INT4 weights + INT8 activations (integer PTQ).

---

## FP4 E2M1 Format Reference

Bit layout: `[s][e1][e0][m]`  (MSB→LSB), 4 bits, bias = 1

```
Code  Bits   Value     Integer equiv (×0.5)
 0    0000    0.0           0
 1    0001    0.5          +1
 2    0010    1.0          +2
 3    0011    1.5          +3
 4    0100    2.0          +4
 5    0101    3.0          +6
 6    0110    4.0          +8
 7    0111    6.0         +12
 8    1000   -0.0           0
 9    1001   -0.5          -1
10    1010   -1.0          -2
11    1011   -1.5          -3
12    1100   -2.0          -4
13    1101   -3.0          -6
14    1110   -4.0          -8
15    1111   -6.0         -12
```

Key insight: integer equivalents are {0, ±1, ±2, ±3, ±4, ±6, ±8, ±12} — NOT contiguous.
Gaps at {±5, ±7, ±9, ±10, ±11}. This has direct implications for RRAM conductance spacing.

---

## Models to Use

| Purpose | Model | Notes |
|---------|-------|-------|
| Fast dev/debug | **Qwen/Qwen2.5-1.5B** | ≈3 GB, ~11.26 PPL baseline confirmed ✓ |
| Main results | Qwen/Qwen2.5-7B, meta-llama/Meta-Llama-3.1-8B | Base models only for raw wikitext PPL |
| Cross-arch | mistralai/Mistral-7B-v0.1 | Base model, GQA architecture |

**Important:** Always use BASE models (not instruct/chat) for wikitext PPL. Instruct models give
~100K PPL on raw text (near-random) because they expect a chat template, making quantization
effects invisible. Always run `python rq2_quant_boundary.py` (not instruct variants).

---

## Experiment Design

### RQ1 — FP4 Feasibility
**Variants:**
- `baseline`: BF16 (no quantization)
- `W4A16`: weights FP4, activations BF16
- `W4A4`: weights + activations FP4 (per-token activation scale)

**Metrics:** wikitext word-PPL (via lm_eval), piqa (0-shot acc), arc_easy (0-shot acc)

#### Results (2026-05-06) — Model: Qwen/Qwen2.5-1.5B

| Variant      | PPL (word) | ΔPPL     | PIQA acc | ARC_easy acc | Interpretation                                         |
|--------------|-----------|----------|----------|--------------|--------------------------------------------------------|
| baseline     | 13.31     | 0.00     | 75.63%   | 75.34%       | BF16, no quantization                                  |
| MXFP4_W4A4_B16   | **15.99** | **+2.68**| 74.48%   | 69.44%       | **Best quantized — beats W4A16** ★                    |
| W4A16        | 16.40     | +3.09    | 74.43%   | 69.87%       | FP4 weights only, BF16 activations                     |
| WFP4AINT8         | 16.83     | +3.52    | 73.83%   | 67.80%       | FP4 weights + INT8 activations (Deep_gate native)      |
| W4A4         | 3325.81   | +3312.50 | 55.22%   | 32.07%       | FP4 per-token activations — catastrophic               |

> Note: lm_eval uses **word-perplexity** (normalized by whitespace tokens), so absolute values
> are higher than the token-PPL reported by our custom `evaluate_ppl` in RQ2 (~11–14).
> The relative ordering and trends are directly comparable.

**Key findings:**
1. **MXFP4_W4A4_B16 beats W4A16** (15.99 vs 16.40 PPL, +20% vs +23% relative): per-block-16
   FP4 quantization of *both* weights and activations outperforms per-channel FP4 weights
   with full-precision activations. Block scaling provides better calibration than per-channel
   alone, and quantizing activations at block granularity adds negligible noise.
2. **WFP4AINT8 ≈ W4A16** (16.83 vs 16.40 PPL, only 0.43 PPL gap): INT8 activations cost almost
   nothing over BF16. Directly validates Deep_gate's 8-bit ADC architecture — the ADC
   precision is not the bottleneck.
3. **Per-token W4A4 is catastrophic** (3325 PPL, 250× baseline): granularity matters more
   than bit-width. Block-16 FP4 works; per-token FP4 does not.
4. **MXFP4_W4A4_B16 is the recommended operating point**: full W4A4 at block-16 granularity
   gives the best accuracy of all quantized variants, with the added benefit of 2× activation
   memory bandwidth reduction over W4A16. (MXFP4 here = E8M0 block-16; true NVFP4 E4M3
   evaluated in RQ11 and expected to be slightly better.)

Output files: `results/rq1/rq1_results.csv`, `results/rq1/rq1_results.json`, `results/rq1/rq1_feasibility.png`

---

### RQ2 — Quantize/Dequantize Boundary
**Activation quantization sites:**
1. `weight_only` — no activation quantization (W4A16 baseline)
2. `attn_only` — FP4 activations inside attention projections only
3. `mlp_only` — FP4 activations inside MLP projections only
4. `all_linear` — all linear layer activations (W4A4)

**Key question:** Does attn or MLP dominate the PPL penalty?
**Dequantize boundary:** output of linear layer is always dequantized to BF16 before residual add.

**Quantization scheme (final):**
- Weights: per-output-channel FP4 scale (one scale per row of weight matrix).
  Per-tensor failed catastrophically because a single outlier forces a large scale and
  rounds all small weights to zero (most weights zero → dead layers).
- Activations: per-token FP4 scale (one scale per [batch, seq] position over hidden dim).
  Standard W4A4 practice; handles the wide magnitude variation across token positions.
- `lm_head` excluded from weight quantization — tied to `embed_tokens` in Qwen2.5,
  so in-place quantization would destroy the embedding table.

#### Results (2026-05-05) — Model: Qwen/Qwen2.5-1.5B, wikitext-2-raw-v1

| Variant      | PPL     | ΔPPL    | Interpretation                          |
|--------------|---------|---------|------------------------------------------|
| no_quant     | 11.26   | 0.00    | BF16 baseline                            |
| weight_only  | 13.97   | +2.71   | W4A16 — FP4 weights only, reasonable ✓   |
| attn_only    | 67.27   | +56.01  | FP4 attn activations, 5× worse than W4A16|
| mlp_only     | 133.35  | +122.09 | FP4 MLP activations, 2× worse than attn  |
| all_linear   | 4250.26 | +4239.00| W4A4 everywhere — too aggressive          |

**Key findings:**
1. **W4A16 is viable**: +2.71 PPL (+24% relative) — FP4 weights with BF16 activations
   is an acceptable operating point.
2. **MLP activations dominate** over attention activations (122 vs 56 ΔPPL). MLP intermediate
   activations (post-gate projection) have larger dynamic range and are more sensitive to
   coarse FP4 quantization.
3. **Attention activations are more tolerant**: the softmax normalisation and residual
   pathway partially absorb FP4 rounding in Q/K/V/O projections.
4. **Full W4A4 is too aggressive** at per-token FP4 scale: 4250 PPL vs 11.26 baseline
   (+3756 absolute ΔPPL; note PPL ratio and reconstruction error are different units —
   the 376× PPL ratio is directionally informative, not directly comparable to the
   52–77% tensor reconstruction error from RQ8). Viable W4A4 requires block scaling.
5. **Dequantize boundary recommendation**: quantize weights to FP4; keep activations in BF16
   (W4A16 operating point). If activation quantization is needed for power, restrict to
   attention layers only and accept the ~56 PPL penalty.

Plot: `results/rq2/rq2_boundary.png`

---

### RQ3 — RRAM Conductance Mapping ✓
**Sign-magnitude decomposition:**
- 1 binary cell (SLC) for sign
- 3 conductance levels (8 states) for magnitude from {0, 0.5, 1, 1.5, 2, 3, 4, 6}

**Differential pair alternative:**
- G+ = max(w, 0) / G_unit, G− = max(−w, 0) / G_unit
- Output: I_i = Σ (G+_ij − G−_ij) × V_j → proportional to GEMM

**Proof of GEMM equivalence:**
- With per-tensor scales s_W, s_A: GEMM = s_W × s_A × (integer_code_W · integer_code_A)
- All FP4 values = integer_code × 0.5, so FP4_GEMM = 0.25 × integer_GEMM

**Conductance spacing analysis:**
- Non-uniform: levels at {0, 1, 2, 3, 4, 6, 8, 12} × G_unit
- Largest gap: between 8 and 12 (ratio 1.5)
- Implication: mid-range levels need tighter programming tolerance

#### Results (2026-05-04)

**Conductance level table (positive side, G normalised to G_max = 1):**

| Magnitude | G/G_max | Integer equiv | ΔG to next | Ratio to next |
|-----------|---------|---------------|------------|---------------|
| 0.0       | 0.0000  | 0             | 0.0833     | ∞             |
| 0.5       | 0.0833  | 1             | 0.0833     | 2.000×        |
| 1.0       | 0.1667  | 2             | 0.0833     | 1.500×        |
| 1.5       | 0.2500  | 3             | 0.0833     | 1.333×        |
| 2.0       | 0.3333  | 4             | 0.1667     | 1.500×        |
| 3.0       | 0.5000  | 6             | 0.1667     | 1.333×        |
| 4.0       | 0.6667  | 8             | 0.3333     | 1.500×        |
| 6.0       | 1.0000  | 12            | —          | —             |

**GEMM equivalence verification (GEMM_fp4 == integer_GEMM × s²):**

| N    | K    | max_abs_err | rel_err  | s_W    | s_A    | s²       |
|------|------|-------------|----------|--------|--------|----------|
| 64   | 128  | 3.81e-06    | 2.67e-07 | 0.6388 | 0.4736 | 0.075623 |
| 256  | 512  | 1.53e-05    | 6.59e-07 | 0.7715 | 0.6988 | 0.134777 |
| 1024 | 4096 | 1.35e-04    | 9.79e-04 | 0.8508 | 0.6581 | 0.139986 |
| 4096 | 4096 | 1.68e-04    | 7.34e-04 | 0.8747 | 0.6591 | 0.144128 |

Equivalence verified ✓ — residual is float32 accumulation rounding only (rel_err < 1e-3).

**Required accumulator precision (analytical, FP4×FP4):**

| K    | Dynamic Range | Required bits | ENOB  | σ_conductance budget |
|------|---------------|---------------|-------|----------------------|
| 64   | 9,216         | 15            | 13.2  | 1.09e-04             |
| 128  | 18,432        | 16            | 14.2  | 5.43e-05             |
| 256  | 36,864        | 17            | 15.2  | 2.71e-05             |
| 512  | 73,728        | 18            | 16.2  | 1.36e-05             |
| 1024 | 147,456       | 19            | 17.2  | 6.78e-06             |
| 2048 | 294,912       | 20            | 18.2  | 3.39e-06             |
| 4096 | 589,824       | 21            | 19.2  | 1.70e-06             |
| 8192 | 1,179,648     | 22            | 20.2  | 8.48e-07             |

Plot: `results/rq3/rq3_rram_mapping.png`

---

### RQ4 — RRAM Variation Tolerance ✓

**Script:** `rq4_rram_tolerance.py` → `results/rq4/`
**Operating point:** W4A16 (per-channel FP4 weights, BF16 activations), baseline PPL = 13.965

**Noise models:**
1. Relative conductance noise: G_noisy = G_target × (1 + σ_rel × N(0,1))
2. Stuck cells: p_stuck probability of cell stuck at G_min (0) or G_max
3. Sign path mismatch: p_sign_flip probability of sign bit flip

**Simulation note:** Noise is applied in float space (continuous readout model). No re-quantization
after noise addition — re-quantizing with a per-tensor scale would incorrectly destroy per-channel
structure, producing catastrophic PPL unrelated to RRAM noise. Stuck-at-max uses per-row max
magnitude to approximate per-channel G_max.

#### Results (2026-05-07) — Qwen2.5-1.5B, wikitext-2

**Sweep 1 — Relative conductance noise (σ_rel):**

| σ_rel | PPL    | ΔPPL   | Verdict |
|-------|--------|--------|---------|
| 0.000 | 13.965 | 0.000  | baseline |
| 0.001 | 13.976 | +0.011 | negligible |
| 0.005 | 13.980 | +0.015 | negligible |
| 0.010 | 13.991 | +0.026 | negligible |
| **0.050** | **14.333** | **+0.368** | **safe (< 0.5 threshold)** |
| 0.100 | 15.278 | +1.313 | moderate degradation |
| 0.200 | 22.558 | +8.593 | severe |

**Sweep 2 — Stuck-cell probability (p_stuck):**

| p_stuck | PPL    | ΔPPL   | Verdict |
|---------|--------|--------|---------|
| 0.0000  | 13.965 | 0.000  | baseline |
| **0.0001** | **14.084** | **+0.119** | **safe** |
| 0.0010  | 15.032 | +1.067 | degraded (> 0.5) |
| 0.0050  | 21.096 | +7.131 | severe |
| 0.0100  | 36.615 | +22.65 | failed |
| 0.0500  | 181421 | +181k  | catastrophic |

**Sweep 3 — Sign path mismatch (p_sign_flip):**

| p_sign_flip | PPL    | ΔPPL   | Verdict |
|-------------|--------|--------|---------|
| 0.0000      | 13.965 | 0.000  | baseline |
| 0.0001      | 14.547 | +0.582 | borderline (eval noise ~±0.1) |
| 0.0005      | 14.216 | +0.251 | safe |
| 0.0010      | 14.504 | +0.539 | borderline |
| 0.0050      | 17.713 | +3.748 | severe |
| 0.0100      | 26.758 | +12.79 | failed |

Note: non-monotonic behavior at low p_sign_flip (0.0001 > 0.0005) is evaluation noise (±0.1–0.2 PPL
over 32×512 tokens). True safe threshold is approximately p_sign_flip ≈ 0.0005–0.001.

**Key findings:**

1. **σ_rel tolerance: ≤ 5%** (ΔPPL < 0.5 at σ_rel=0.05). FP4's 4× wider level spacing
   (every 4th of 32 RRAM levels) provides 4× better noise margin vs using all 32 levels.
   Typical RRAM σ_rel at programmed state = 1–3% → FP4 weights are safe.

2. **Stuck-cell tolerance: p_stuck < 0.01%** (1 in 10,000 cells). At p_stuck=0.001 (0.1%)
   ΔPPL=+1.07 already exceeds the threshold. Requires high-quality cells or 1-bit ECC.
   MEMRISYS-2025 post-ECC BER < 10⁻⁶ = 0.0001% → safely within the safe zone.

3. **Sign path is most sensitive**: p_sign_flip must be < 0.05–0.1% to stay within 0.5 ΔPPL.
   Sign bit error is more damaging than conductance noise because it flips the contribution
   from S⁺ to S⁻ (or vice versa), doubling the error for that element.

4. **FP4 quantization noise dominates**: +2.68 PPL from FP4 quantization (RQ1) dwarfs the
   hardware noise at safe operating points. RRAM noise is not the accuracy bottleneck.

---

### RQ5 + RQ6 — Photonic Accumulation Precision ✓

**Script:** `rq5_rq6_photonic_precision.py` → `results/rq5_rq6/`

#### Results (2026-05-07) — Qwen2.5-1.5B

**Experiment A — Full layer output ENOB (RQ5):**

These numbers reflect the **full dot-product output** dynamic range across all 196 targeted linear layers.

| Layer type | ENOB mean | ENOB max | ENOB min | Required bits mean | DR mean  |
|------------|-----------|----------|----------|--------------------|----------|
| attn       | 27.99     | 34.55    | 22.13    | 29.5               | 1.07×10⁹ |
| mlp        | 28.25     | 33.68    | 21.95    | 29.8               | 1.33×10⁹ |

**Experiment B — ADC sweep: PPL vs accumulator bits N (RQ6):**

Per-tensor fixed-point quantization applied to layer outputs (simulates full-DR analog accumulation):

| Scope      | N=4    | N=6    | N=8    | N=10   | N=12   | N=16   | baseline |
|------------|--------|--------|--------|--------|--------|--------|----------|
| all layers | 1.4M   | 10,069 | 36.64  | 24.37  | 16.70  | 13.97  | 13.97    |
| attn only  | 29,918 | 9,136  | 14.81  | **13.97** | 13.98  | 13.97  | 13.97    |
| mlp only   | 89,221 | 376.6  | 31.02  | 24.39  | 16.71  | **13.98** | 13.97  |

ΔPPL vs baseline:

| Scope      | N=8    | N=10   | N=12   | N=16   |
|------------|--------|--------|--------|--------|
| all layers | +22.7  | +10.4  | +2.7   | ≈0     |
| attn only  | +0.85  | ≈0     | ≈0     | ≈0     |
| mlp only   | +17.1  | +10.4  | +2.7   | ≈0     |

**Key findings:**

1. **Full-DR analog accumulation requires ~28–30 bits ENOB** — far beyond any analog photonic system. This is the requirement if the photonic core must accumulate across the full dot-product dimension in the analog domain.

2. **MLP layers drive the precision requirement**: Attn-only is safe at N=8 (ΔPPL=+0.85), but MLP-only needs N≥16 for near-baseline accuracy. MLP layers have higher dynamic range (DR ~1.33×10⁹ mean vs 1.07×10⁹ for attn).

3. **Attn is more tolerant**: N=10 suffices for attention-only quantization (≈0 ΔPPL). This is consistent with RQ2's finding that attention activations tolerate more quantization than MLP.

4. **Critical interpretation — per-tensor fixed-point ≠ FP16 per-element decode**: The same caveat as RQ9 applies. The N=12–16 requirement is for per-tensor fixed-point on the FULL layer output. The hardware FP16 decode is per-element floating-point per subarray, which handles the full DR trivially — same as the BF16 baseline. The ENOB numbers are relevant for full-column analog accumulation only.

5. **Row-serial decomposition eliminates the ENOB bottleneck**: The Deep_gate architecture accumulates one row at a time digitally. In this model the photonic/ADC only sees a single-row partial product (≤9 bits for FP4×4-bit cell). The ~28-bit full-DR requirement applies only if using column-parallel analog accumulation across all rows simultaneously.

**Deliverable to photonics colleague:**
- Full-column analog accumulation: needs ~28–30 ENOB → not practical with current photonics
- Row-serial + digital accumulation (recommended): needs only 9-bit ADC per row → achievable
- Attn vs MLP split: if partial analog accumulation is used, restrict to attn layers where N=8 suffices

---

### RQ7 + RQ8 — Distribution-Aware Sensing ✓

**Script:** `rq7_rq8_distribution.py` → `results/rq7_rq8/`

#### Results (2026-05-07) — Qwen2.5-1.5B, 196 weight + 196 activation tensors

**RQ7 — FP4 code distribution (per-tensor scale, Qwen2.5-1.5B):**

| Magnitude | Weight freq | Act freq | Sensing time (weight) |
|-----------|------------|---------|----------------------|
| 0.0       | 0.6337     | 0.9481  | 0.6337               |
| 0.5       | 0.3306     | 0.0456  | 0.3306               |
| 1.0       | 0.0304     | 0.0044  | 0.0304               |
| 1.5       | 0.0040     | 0.0011  | 0.0040               |
| 2.0       | 0.0011     | 0.0005  | 0.0011               |
| 3.0       | 0.0002     | 0.0002  | 0.0002               |
| 4.0       | ~0.0000    | 0.0001  | ~0                   |
| 6.0       | ~0.0000    | ~0.0000 | ~0                   |

**Entropy and zero fraction by layer type:**

| Tensor type | Layer | FP4 entropy | Zero fraction |
|-------------|-------|-------------|---------------|
| Weight      | attn  | 2.583 bits  | 0.446         |
| Weight      | MLP   | 2.005 bits  | 0.648         |
| Activation  | attn  | 1.539 bits  | 0.887         |
| Activation  | MLP   | 1.262 bits  | 0.951         |

*Note: activation stats use per-token FP4 scale (naive). With MXFP4 block-16, activations would be far more uniformly distributed — the high zero fractions here confirm why per-token scaling fails (RQ2).*

**Key findings (RQ7):**

1. **Weights: highly concentrated at zero and 0.5**: 63.4% at G=0, 33.1% at G=0.5 → 96.5% of cells in just two levels. Only 3.5% at higher levels. This is from FP4 rounding — not pruning.
   - Implication: ~63% of RRAM cells store G_min (zero conductance) → relaxes overall programming requirements
   - G=4.0 and G=6.0 are essentially unused (< 0.01%) → their cells can have the most relaxed tolerance

2. **Activations: even more extreme under per-token scale**: 94.8% zeros, entropy=1.3–1.5 bits. A single large activation in a token forces a large per-token scale → all other elements round to zero. Confirms RQ2: per-token FP4 activation quantization is catastrophic.

3. **Adaptive sensing gain is limited**: The distribution is dominated by 2 levels (G=0, G=0.5). Adaptive sensing helps most when all levels are used; with 96.5% in 2 levels, there is little room for reallocation. The main practical benefit: G=4.0 and G=6.0 cells can be programmed with relaxed tolerance.

4. **Contrast with NVFP4 checkpoint** (from checkpoint inspection, earlier): NVFP4 achieves near-uniform distribution (entropy ≈ 4 bits max) because E4M3 block-16 scales calibrate each group to use the full FP4 range. Our per-tensor FP4 (entropy 2.0–2.6 bits) wastes ~30–50% of representational capacity. This is the underlying reason block-16 scaling recovers accuracy (RQ1/RQ11).

**RQ8 — NVFP4 block scale analysis:**
- E8M0 block scales (log₂) span a wide range across layers
- Attn layers: larger scales (higher weight magnitude range) than MLP
- Scale std is significant (~0.5–0.7 in log₂ units) → per-block calibration is important
- These results are consistent with RQ11: E8M0's 44.6% mean scale error comes from the power-of-2 rounding of these continuous log₂-scale values

Plots: `results/rq7_rq8/rq7_sensing_policy.png`, `rq7_fp4_entropy.png`, `rq8_nvfp4_scales.png`

#### NVFP4 Checkpoint Inspection Results (2026-05-05)
Model: `JongYeop/Qwen2.5-3B-Instruct-FP4-W4A4`
Script: `inspect_nvfp4.py` → `results/nvfp4_inspect/`

**Layer coverage:**
- 252 quantized (all `nn.Linear` in transformer body)
- 75 unquantized: all LayerNorms, `embed_tokens`, `lm_head`, `model.norm`

**NVFP4 storage per linear layer (verified from safetensors):**

| Tensor | dtype | Shape | Role |
|--------|-------|-------|------|
| `weight_packed` | uint8 | [out, in//2] | two FP4 nibbles per byte |
| `weight_scale` | float8_e4m3fn | [out, in//16] | per-16-element group scale |
| `weight_global_scale` | float32 | [1] | per-tensor global scale |
| `input_global_scale` | float32 | [1] | calibrated activation scale |

**FP4 code distribution (global average across 252 layers):**

| Magnitude | Freq (|pos|+|neg|) | Sensing time (norm) |
|-----------|-------------------|---------------------|
| 0.0       | 0.0793            | 0.0793              |
| 0.5       | 0.1450            | 0.1450              |
| 1.0       | 0.1311            | 0.1311              |
| 1.5       | 0.1168            | 0.1168              |
| 2.0       | 0.1470            | 0.1470              |
| 3.0       | 0.1444            | 0.1444              |
| 4.0       | 0.1280            | 0.1280              |
| 6.0       | 0.1084            | 0.1084              |

Key finding: distribution is **near-uniform** (expected ~6.25% per code). NVFP4 calibration
spreads codes almost maximally → code entropy ≈ 4 bits (maximum). This means **adaptive
sensing time allocation has limited gain under NVFP4** — all levels need similar precision.

**Group scale statistics (weight_scale, log₂):**

| Layer type | log₂(scale) mean | log₂(scale) std | weight_global_scale mean |
|------------|-----------------|-----------------|--------------------------|
| attn       | 5.85            | 0.54            | 7218                     |
| mlp        | 4.79            | 0.71            | 3889                     |

Attn layers have larger scales (higher effective weight magnitude range) than MLP layers.

**Plain FP4 (per-tensor) vs NVFP4 relative reconstruction error:**

| Layer type | Mean rel. error |
|------------|----------------|
| attn       | 0.52           |
| mlp        | 0.77           |

→ **Plain per-tensor FP4 has 52–77% relative error vs NVFP4** on the same weight matrix.
This is the key quantitative argument for block scaling in RQ8: without per-group scales,
the same 4-bit codes represent very different values, losing most of the calibration benefit.

---

### RQ9 — Partial Sum Overflow

**The problem:** When K FP4 values are accumulated in a column of a CIM crossbar, the
partial sum can exceed the accumulator's representable range, causing either wrap-around
(catastrophic) or saturation (graceful clipping). The question is: what accumulator width
is *sufficient* for each operating point, and what is the PPL cost of under-provisioning?

**Analytical worst case:**

| Operating point | Max single product | Max partial sum (K=4096) | Bits needed |
|----------------|-------------------|--------------------------|-------------|
| FP4×FP4        | 6×6 = 36          | 36 × 4096 = 147,456      | 18 bits     |
| FP4×FP4 (UQ3.2)| 24×24 = 576       | 576 × 4096 = 2,359,296   | 22 bits     |
| FP4×INT8       | 24×127 = 3,048    | 3,048 × 4096 = 12,484,608| 24 bits     |
| NVFP4×NVFP4 (block)| 576 per 16    | 576 × 16 = 9,216 (intra) | 14 bits     |

Key insight: these are *worst-case* bounds. Real weight/activation distributions rarely hit
the maximum simultaneously. The empirical overflow rate and PPL curve tell us the practical
operating point.

**Row-serial mitigation (Deep_gate):** accumulator only ever sees one row's partial product
at a time (max = 24 × 15 = 360 for FP4×4-bit cell, 9 bits). Digital accumulator then sums
row contributions in INT32 — overflow is structurally impossible at this stage. This is the
key architectural reason Deep_gate doesn't need wide ADCs.

**Experiment design (RQ9):**

*Step 0 — Partial sum distribution capture (feeds photonics colleague directly):*
Hook each linear layer to record the running partial sum after every K=1,2,4,…,K_full
accumulation steps. Collect: min, max, mean, std, 99th percentile, 99.9th percentile,
dynamic range = max(|psum|) / percentile_01(|psum|). Report per layer and per layer type
(attn vs. MLP). This is the primary input for the photonic ENOB requirement.

*Step 1 — Overflow sweep:*
1. Hook all linear layers to clamp partial sums to ±(2^(N−1) − 1) at the final accumulation
2. Sweep N ∈ {8, 10, 12, 14, 16, 18, 20, 24, 32} for FP4×FP4 operating point
3. Measure PPL per N; find the *knee* — minimum N before PPL degrades noticeably
4. Repeat for WFP4AINT8 (FP4 × INT8) and MXFP4_W4A4_B16 (block-serial)
5. Per-layer breakdown: which layers overflow first at each N?

**Expected finding:** PPL is flat until N drops below ~18 (FP4×FP4) or ~24 (FP4×INT8),
then degrades sharply. The plateau gives the hardware team a verified accumulator spec.
The partial sum distribution from Step 0 should predict the knee: overflow probability
becomes non-negligible when 2^N < 99.9th-percentile partial sum.

**Hardware deliverable to photonics colleague:**
- Partial sum DR per layer → minimum ENOB the photonic accumulator must provide
- Overflow rate vs. N curve → practical clipping threshold (tolerated clipping rate <0.1%)
- Per-layer table: (layer, max_psum, 99.9th psum, min safe N, overflow rate at N−2)

#### Results (2026-05-06) — Model: Qwen/Qwen2.5-1.5B, wikitext-2-raw-v1

**Step 0 — Partial sum distribution (across all linear layers):**

| Operating point | max |psum| (max layer) | max |psum| (median layer) | p99 (max layer) | req_bits (max) | req_bits (median) |
|-----------------|----------------------|------------------------|-----------------|----------------|-------------------|
| WFP4AINT8       | 4576.0               | 15.6                   | 288.0           | 34 bits        | 29 bits           |
| MXFP4_W4A4_B16      | 6112.0               | 15.5                   | 288.0           | 34 bits        | 26 bits           |

Key finding: empirical max|psum| is ~6000, far below the analytical FP4×FP4 worst case at K=4096
(589,824). Typical (median) layer has max|psum| only ~15.5. The worst-case outlier layers drive the
accumulator spec; average layers need far fewer bits. req_bits (analytical worst case) = 22 bits at
K=4096; empirical max requires 34 bits — driven by BF16-precision intermediates, not raw FP4 codes.

**Step 1 — Overflow sweep (PPL vs accumulator width N):**

| N bits | WFP4AINT8 PPL | MXFP4_W4A4_B16 PPL | Interpretation |
|--------|---------------|----------------|----------------|
| inf    | 14.37         | 13.49          | baseline (no clamping) |
| 32     | 14.37         | 13.51          | no effect (same as inf) |
| 24     | 14.33         | 13.43          | no effect |
| 20     | 14.27         | 13.57          | marginal |
| 18     | 14.40         | 13.50          | marginal |
| **16** | **14.32**     | **13.59**      | **knee: ΔPPL < 0.5** |
| 14     | 14.50         | 13.80          | acceptable (+0.13 / +0.31) |
| 12     | 16.96         | 17.45          | severe (+2.59 / +3.97) |
| 10     | 25.13         | 24.52          | degraded |
| 8      | 38.07         | 34.14          | failed |

**Knee at N=14–16**: ΔPPL < 0.5 for both operating points. N=14 gives marginal degradation
(+0.13 WFP4AINT8, +0.31 MXFP4_W4A4_B16). N=16 is safe with near-zero degradation.
Hardware spec for photonics: 16-bit accumulator sufficient; 14-bit acceptable with slight penalty.

---

### RQ9b — Partial-Sum Growth by K + FP16 Decoder Equivalence ✓

Script: `rq9b_partial_sum_growth.py` → `results/rq9b/`

Three sub-questions in one script:

**(A) Analytical worst-case partial sum growth with K:**

| K   | max partial sum (FP4×FP4) | bits needed | max partial sum (FP4×INT8) | bits needed |
|-----|--------------------------|-------------|---------------------------|-------------|
| 1   | 576                      | 11          | 3,048                     | 13          |
| 16  | 9,216                    | 15          | 48,768                    | 17          |
| 64  | 36,864                   | 17          | 195,072                   | 19          |
| 256 | 147,456                  | 19          | 780,288                   | 21          |
| 1536| 884,736                  | 21          | 4,681,728                 | 24          |
| 4096| 2,359,296                | 22          | 12,484,608                | 25          |

**(B) Empirical partial sum growth (median max|psum| across layers):**
Empirical values are 2–5 orders of magnitude below worst case, confirming that the
tail of the distribution (not the average) drives the accumulator spec.

**(C) FP16 decoder equivalence analysis:**

FP16 has 11 significant mantissa bits. Worst-case bits needed for K partial products:
- K=1: 11 bits → LOSSLESS in FP16
- K=2: 12 bits → −1 bit precision loss
- K=4: 13 bits → −2 bits
- K=16: 15 bits → −4 bits
- K=256: 19 bits → −8 bits

**Key insight**: FP16 decode is PER-ELEMENT with its own exponent, NOT equivalent to
per-tensor N-bit fixed-point. The RQ9 fixed-point knee at N=14 is a CONSERVATIVE bound.
Real FP16 decode quality (per-element) uses round-to-nearest-even (PyTorch `.half()` default),
giving ½-ULP relative error = 2⁻¹¹ ≈ 0.049% — not 0.1% (which would be 1-ULP truncation).
Truncation gives 2⁻¹⁰ ≈ 0.098%. Our simulation uses `.half()` so the 0.049% bound applies.
Either way, hardware should use FP16 decode without concern.

---

### RQ9c — FP16 Decoder Standalone Cross-Reference ✓

Script: `rq9c_fp16_decoder.py` → `results/rq9b/` (same directory)
Reads pre-computed analytical + RQ9 CSVs; no GPU needed. Confirms RQ9b findings with
cross-reference table: for N=14 fixed-point (bracketing FP16's 11-bit resolution), both
operating points show ΔPPL < 0.5, validating FP16 decode as adequate.

---

### RQ10 — Subarray Size K Sweep ✓

**Script:** `rq10_subarray_k_sweep.py` → `results/rq10/`

**Setup:** For each K (rows per subarray sub-block), intercept every nn.Linear:
- Split inner dimension into P = ceil(in_f / K) chunks of K
- Compute each partial sum in float32
- Round partial sum to FP16 (simulates sub-sum decoder)
- Accumulate P chunks in float32 (simulates FP16 tree reduction)

**Results (2026-05-06), Qwen/Qwen2.5-1.5B:**

**Block-alignment constraint:** For MXFP4_W4A4_B16, K must be a multiple of 16 (block size).
K=1/2/4/8 mix elements from different scale blocks in one accumulator — not physically
realizable. WFP4AINT8 has no block constraint (per-channel weights, per-token INT8 activations).

| K    | MXFP4_W4A4_B16 PPL | ΔPPL   | WFP4AINT8 PPL | ΔPPL   | Note |
|------|---------------------|--------|---------------|--------|------|
| inf  | 13.486              |  0.000 | 14.370        |  0.000 | baseline (no partitioning) |
| 1    | 13.525              | +0.039 | 14.313        | −0.057 | below block-16 constraint for MXFP4 |
| 2    | 13.505              | +0.020 | 14.350        | −0.020 | below block-16 constraint for MXFP4 |
| 4    | 13.567              | +0.082 | 14.333        | −0.037 | below block-16 constraint for MXFP4 |
| 8    | 13.599              | +0.113 | 14.315        | −0.055 | below block-16 constraint for MXFP4 |
| 16   | 13.563              | +0.077 | 14.322        | −0.048 | min valid K for MXFP4_W4A4_B16 |
| 32   | 13.459              | −0.027 | 14.305        | −0.065 | |
| 64   | 13.533              | +0.047 | 14.372        | +0.002 | |
| 128  | 13.514              | +0.028 | 14.301        | −0.070 | |
| 256  | 13.621              | +0.135 | 14.273        | −0.097 | |
| 512  | 13.564              | +0.078 | 14.284        | −0.086 | |
| 1024 | 13.451              | −0.034 | 14.296        | −0.075 | |
| 2048 | 13.595              | +0.110 | 14.318        | −0.052 | |
| 4096 | 13.530              | +0.044 | 14.284        | −0.086 | |

K=2048/4096 now valid — bias double-count bug fixed (`k ≥ in_f` branch recomputes `x @ w.T`
rather than reusing `output` which already contained RoPE q/k biases from the standard forward).

**Key findings:**

1. **FP16 decode negligibly affects accuracy for ALL K (1–4096)**: Every tested K has |ΔPPL| < 0.14.
   The FP16 tree accumulation introduces no meaningful error regardless of the number of chunks.

2. **Block-alignment constraint is a hardware constraint, not a precision constraint**: K=1/2/4/8
   show similar ΔPPL to K=16–4096 in the float simulation. The block constraint (K must be multiple
   of 16 for MXFP4_W4A4_B16) is a hardware rule: elements from different scale blocks cannot share
   one integer accumulator with a single scale. The float simulation has scale already baked into
   weight values, so it doesn't capture this physical limitation. The constraint stands from first
   principles regardless of these simulation results.

3. **Hardware recommendation**: K can be chosen freely (subject to block-16 constraint for MXFP4)
   based on area/power tradeoffs. Larger K → fewer subarrays, wider ADC. Smaller K → more
   subarrays, narrower ADC, deeper FP16 tree.

4. **WFP4AINT8 ΔPPL slightly negative** for many K values: evaluation noise (±0.1 PPL over
   32×512 tokens). No physical meaning.

**Analytical companion (hardware cost table):**

| K    | ADC bits (wc) | P (d=1536) | P (d=8960) | Tree err % (d=1536) |
|------|---------------|------------|------------|---------------------|
| 16   | 15            | 96         | 560        | 0.32                |
| 32   | 16            | 48         | 280        | 0.27                |
| 64   | 17            | 24         | 140        | 0.22                |
| 128  | 18            | 12         | 70         | 0.18                |
| 256  | 19            | 6          | 35         | 0.13                |
| 512  | 20            | 3          | 18         | 0.08                |
| 1024 | 21            | 2          | 9          | 0.05                |

Plot: `results/rq10/rq10_k_sweep.png`

---

### RQ11 — Block-Scale Format Comparison (E8M0 vs E4M3 vs FP32) ✓

**Script:** `rq11_scale_format_comparison.py` → `results/rq11/`

#### Results (2026-05-07) — Model: Qwen/Qwen2.5-1.5B, block size 16

| Variant     | W scale          | A scale          | PPL    | ΔPPL vs BF16 | Hardware cost |
|-------------|------------------|------------------|--------|--------------|---------------|
| BF16        | —                | —                | 11.257 | +0.000       | baseline |
| W4A16-ch    | per-channel FP32 | BF16             | 13.965 | +2.708       | FP4 weight decode only |
| MXFP4-W4A4-B16  | E8M0 block-16    | E8M0 block-16    | 13.486 | +2.229       | shift-only (barrel shifter) |
| NVFP4-W4A4  | E4M3 block-16    | E4M3 block-16    | 13.234 | +1.977       | FP8→FP16 scale multiply |
| Ideal-W4A4  | FP32 block-16    | FP32 block-16    | 13.130 | +1.873       | exact (reference only) |

**Scale format cost (PPL gaps):**

| Comparison         | ΔPPL  | Interpretation |
|--------------------|-------|----------------|
| E8M0 vs ideal      | +0.356 | full cost of power-of-2 rounding |
| E4M3 vs ideal      | +0.104 | residual cost of FP8 scale quantization |
| **E8M0 vs E4M3**   | **+0.252** | **hardware tradeoff: shift vs multiply** |

**Scale error (197 weight layers):**
- E8M0: mean 44.6% relative scale error (power-of-2 rounding can be up to 2× off)
- E4M3: mean  5.8% relative scale error (3 mantissa bits; max 10.0%)

**Key findings:**

1. **E8M0 shift-only scale is sufficient** (ΔPPL vs E4M3 = 0.252 < 0.5 threshold).
   Hardware recommendation: use the barrel-shift E8M0 scale path. No FP8 scale multiplier needed.

2. **Block-16 beats per-channel even with coarse E8M0 scales**: MXFP4-W4A4-B16 (13.486) < W4A16-ch (13.965).
   Scale granularity (per-16 vs per-channel) matters far more than scale format precision.

3. **Scale quantization error matters little**: E8M0 has 44% mean scale error yet costs only +0.356 PPL
   vs ideal FP32 scales. FP4 code quantization dominates; scale rounding is a minor second-order effect.

4. **E4M3 closes 71% of the E8M0 gap to ideal** (0.252 → 0.104 remaining PPL). If accuracy is
   critical and the FP8 scale multiply is acceptable, E4M3 is the right choice.

5. **NVFP4-W4A4 (E4M3) vs W4A16-ch**: 13.234 vs 13.965 — E4M3 W4A4 still clearly better than
   per-channel W4A4 with BF16 activations, by 0.731 PPL.

**Paper narrative (Outcome B — best hardware story):**
> Shift-only E8M0 block scales (MXFP4) are sufficient for this RRAM-CIM FP4 design.
> Despite 44% mean scale rounding error, the PPL penalty vs ideal-scale block FP4 is only 0.356,
> and vs true NVFP4 (E4M3) only 0.252 — well within the acceptable threshold. The RRAM array
> core is format-agnostic; only the lightweight scale-path logic changes between MXFP4 and NVFP4.

Plot: `results/rq11/rq11_scale_comparison.png`

---

### RQP — Throughput & Memory Profiling ✓

Model: Qwen/Qwen2.5-1.5B, GPU: single A100 (CUDA_VISIBLE_DEVICES=4), bfloat16 base.
Script: `rq_profile.py`. Results: `results/profile/`.

| Variant | VRAM loaded (GB) | Prefill (tok/s) | Decode (tok/s) | TTFT (ms) | vs baseline |
|---|---|---|---|---|---|
| baseline (BF16) | 3.087 | 18,352 | 68.9 | 15.5 ms | 1× |
| WFP4AINT8 | 3.096 | 9,248 | 33.4 | 31.1 ms | ~2× slower |
| NVFP4_W4A4_B16 | 3.096 | 4,775 | 17.6 | 61.9 ms | ~4× slower |

**Key findings:**

- **VRAM identical (~3.09 GB) across all variants**: software simulation stores weights as bfloat16
  regardless of quantization; real hardware compression would reduce NVFP4_W4A4_B16 by ~4× and WFP4AINT8 by ~2×.
- **Throughput drops are simulation overhead, not real hardware behavior**: forward hooks perform
  quantize→dequantize on every activation in Python, adding overhead proportional to hook complexity.
  On Blackwell (native FP4 tensor cores), FP4 variants would be *faster* than BF16, not slower.
- **WFP4AINT8 ~2× slower**: per-token INT8 activation scaling is relatively cheap but still adds hook overhead.
- **NVFP4_W4A4_B16 ~4× slower**: per-block-16 E4M3 scale lookup is more expensive than per-token INT8.
- **TTFT scales proportionally** with prefill slowdown (15.5 → 31.1 → 61.9 ms), confirming overhead is
  proportional to computation, not fixed cost.

**Interpretation for hardware design:** Simulation throughput is not a proxy for real FP4 hardware speed.
These numbers characterize the *accuracy* of each quantization scheme (via PPL in other RQs); the throughput
results confirm there is no numerical instability (all variants complete cleanly) but hardware benchmarks
require native FP4 kernel support (e.g., cuBLAS FP4 on H100/B100).

Plot: `results/profile/profile_throughput.png`

---

## Key Equations

**ENOB from dynamic range:**
```
ENOB = log2(DR)
where DR = max_output / min_nonzero_output
```

**Required accumulator bits for partial sum of K FP4 products:**
```
S⁺ and S⁻ are each UNSIGNED accumulators — the sign only enters at S⁺ − S⁻.
ADC reads one unsigned accumulator at a time; no sign bit needed there.

Bits for S⁺ or S⁻ (unsigned, FP4×FP4 UQ3.2 integer products):
  max product = 24 × 24 = 576   (UQ3.2 integer codes, max = 24)
  max S = K × 576
  bits = ceil(log2(K × 576 + 1))   (unsigned, no sign bit)

  K=16:   ceil(log2(9217))  = 14 bits
  K=64:   ceil(log2(36865)) = 16 bits
  K=1024: ceil(log2(589825))= 20 bits
  K=4096: ceil(log2(2359297))= 22 bits

Final output (S⁺ − S⁻) is signed and requires one additional sign bit → 23 bits max at K=4096.
(The old formula "ceil(log2(144K))+1" used FP4 magnitude 6×6=36 and added a sign bit prematurely.)
```

**FP4 integer equivalence (scale = 0.5):**
```
GEMM_fp4(W, A) = 0.25 × GEMM_int(W_int, A_int)
where W_int = W / 0.5  (integer codes)
      A_int = A / 0.5
```

**RRAM conductance noise model:**
```
G_noisy = G_target × (1 + σ_rel × ε),  ε ~ N(0, 1)
w_noisy = G_noisy × G_unit × sign    (dequantized)
```

**FP4 → 5-bit UQ3.2 via barrel shift:**

The FP4 E2M1 magnitude bits `[e1][e0][m]` map to a 5-bit unsigned fixed-point integer
(UQ3.2 — 3 integer bits, 2 fractional bits) via a single variable left shift:

```
if (e == 0):
    n = m << 1          // denormal/zero: gives 0 or 2
else:
    n = {1'b1, m} << e  // normalized: prepend implicit leading 1, shift by raw exponent
```

where the shift amount is exactly the raw 2-bit exponent `e ∈ {1, 2, 3}` — no bias subtraction needed.

Complete mapping (verified numerically):

| FP4 magnitude | e | m | shift formula  | UQ3.2 int | binary  |
|---------------|---|---|----------------|-----------|---------|
| 0.0           | 0 | 0 | 0              | 0         | `00000` |
| 0.5           | 0 | 1 | `1` << 1       | 2         | `00010` |
| 1.0           | 1 | 0 | `10` << 1      | 4         | `00100` |
| 1.5           | 1 | 1 | `11` << 1      | 6         | `00110` |
| 2.0           | 2 | 0 | `10` << 2      | 8         | `01000` |
| 3.0           | 2 | 1 | `11` << 2      | 12        | `01100` |
| 4.0           | 3 | 0 | `10` << 3      | 16        | `10000` |
| 6.0           | 3 | 1 | `11` << 3      | 24        | `11000` |

Used 5-bit levels: `{0, 2, 4, 6, 8, 12, 16, 24}` — 8 of 32 states, all even.
Level gaps: `[2, 2, 2, 2, 4, 4, 8]` — gaps grow at higher magnitudes (better noise margin for large weights).

**Hardware decoder overhead (minimal):**

**Why UQ3.2?**
A 5-bit unsigned integer `n ∈ [0, 31]` decoded with implicit scale 2⁻² gives values
`x = n · 2⁻²  ∈ [0, 7.75]`. Standard fixed-point notation calls this UQ3.2
(3 integer bits, 2 fractional bits). For our FP4 mapping `n = 4|x_FP4|`, so
`|x_FP4| = n · 2⁻²` — the scale factor of 4 is just the implicit Q-point shift.
Only 8 of the 32 representable codes are used: `{0, 2, 4, 6, 8, 12, 16, 24}`.

Paper-safe sentence:
> Each FP4 E2M1 magnitude is exactly linearized into a 5-bit unsigned fixed-point value nₓ
> with implicit scale 2⁻² (i.e., |x| = nₓ · 2⁻², UQ3.2), using only the sparse code set
> {0, 2, 4, 6, 8, 12, 16, 24}. The full signed value is represented as a 1-bit sign plus
> this 5-bit UQ3.2 magnitude stored in the ReRAM cell.

The conversion requires only:
1. Extract `e[1:0]` and `m` from the 3 magnitude bits
2. Form 2-bit significand `{1'b1, m}` (prepend implicit leading 1)
3. Variable left shift by `e` — a 2-bit-controlled barrel shifter (only 3 possible shifts: ×2, ×4, ×8)
4. One mux to select the `e=0` denormal/zero case

This is a tiny combinational block — far smaller than a LUT-based nonlinear decoder.
**For weights, this runs offline during ReRAM programming** (not on the critical path).
For activations, it is lightweight runtime logic.

**Clean statement:** The FP4 E2M1 → 5-bit UQ3.2 conversion for ReRAM magnitude programming
requires at most a 2-bit-controlled left shifter plus one denormal mux. The shift amount
is exactly the raw exponent `e` — no bias arithmetic needed.

**Worked example:**
```
FP4 code 0111  →  e=3, m=1
  significand {1,m} = 11₂  (= 3₁₀)
  11₂ << 3  →  n = 24  (= 11000₂)
  |x| = 24 · 2⁻²  =  6.0  ✓
```

> Note: minimum nonzero FP4 value is 2⁻¹ = 0.5, but the UQ3.2 decode scale is 2⁻²
> because the smallest used UQ3.2 code is 2 (not 1). This is why `n_min = 2`, not 1.

---

### FP4-Compatible 5-bit RRAM CIM Compute Flow

```
FP4 weights ──► sign extraction + FP4→UQ3.2 ──► program 5-bit RRAM magnitudes
                                                          │
                                              5-bit RRAM compute array
                                         (stores unsigned magnitudes n_w;
                                          inside array: fixed-point MAC)
                                                     │         │
FP4 activations ─► sign extraction + FP4→UQ3.2      │         │
                   ─► row / input drive ─────────────►         │
                                                               │
                          ┌──────────────────────────────────── ┘
                          │   S⁺ = Σ_{s_w·s_a = +1}  n_w · n_a   (positive accumulator)
                          │   S⁻ = Σ_{s_w·s_a = −1}  n_w · n_a   (negative accumulator)
                          │
                          └──► S⁺ − S⁻  ──► right-shift by 4 + scale restore
                                            ──► higher-precision output / FP-domain
```

**Full output formula:**
```
y  =  (a_w · a_a / 16) · (S⁺ − S⁻)

where
  a_w, a_a  = per-output-channel (or per-block) float scales for weights and activations
  n_w, n_a  = UQ3.2 integer codes (from barrel-shift decoder)
  S⁺        = sum of products where sign_w · sign_a = +1
  S⁻        = sum of products where sign_w · sign_a = −1
  /16       = right-shift by 4 (two 2-bit fractional parts cancel)
```

**Signed-CIM decomposition** is the main hardware overhead: the array must maintain two
separate accumulators (S⁺, S⁻) and subtract at the end, mirroring the differential-pair
conductance approach used in analog CIM. In Deep_gate's row-serial architecture this is
done digitally — each row fires once, contributing to S⁺ or S⁻ based on the XOR of the
weight and activation sign bits.

**MAC arithmetic:** `FP4_w × FP4_a = (n_w / 4) × (n_a / 4) = n_w · n_a / 16`.
Accumulate integer products into S⁺ or S⁻; right-shift the final (S⁺ − S⁻) by 4 and
multiply by the float scale product `a_w · a_a`.

---

## Findings from INT8 Hardware Work

### MEMRISYS-2025 Paper — 5-bit Multi-Level ReRAM (Jain et al.)

**Key specs (directly calibrate RQ4 noise model):**
- **32 resistance states (S0–S31)** demonstrated with well-separated levels, minimal overlap
- **Thermal stability**: S0 and S31 stable over 100,000 s at room temp and 125°C
- **BER with 1-bit ECC** (38-bit codeword, 32 data bits):
  ```
  Post-ECC BER = (1 - (1-p)^38 - 38p(1-p)^37) / 32
  ```
- **Relaxation at 125°C** (two-phase): initial rapid drift (first few minutes), then stabilization
  - LRS (S0): resistance drifts **upward** (partial filament rupture / vacancy back-diffusion)
  - Intermediate (S9): resistance drifts **downward** (vacancy migration reinforces filament)

- **Relaxation model calibration for RQ4**: the two-phase drift at 125°C means
  freshly-programmed cells have higher effective σ_rel for ~60–120 s after write, then
  stabilise. The RQ4 sweep should include a "fresh" vs "stabilised" noise regime.

---

### Deep_gate_AD — INT8 Crossbar CIM Simulator

**System**: 10-layer FC autoencoder (640→128→…→8→…→640) for anomaly detection  
**Hardware model**: 32×32 crossbar, `PWM_INT8_ROW_SERIAL`, 4-bit cells, 8-bit ADC

**Crossbar weight encoding (INT8 → 4-bit cells):**
```
INT8 weight w  →  w_pos = max(w,0), w_neg = max(-w,0)
w_pos = w_pos0 + 16×w_pos1   (two 4-bit banks)
w_neg = w_neg0 + 16×w_neg1   (two 4-bit banks)
Total: 4 cells per INT8 weight
```

**Perfect functional accuracy verified** (FC0–FC9, 5 examples, max_err = 0.0 across all layers).

**Key architectural insight — why 8-bit ADC suffices despite large accumulations:**  
The crossbar uses **row-serial** computation: each row fires separately, contributing one
4-bit × 4-bit partial product (max = 15×15 = 225, fits in 8 bits). Digital accumulators
then sum the row contributions. The ADC only ever sees a single-row partial product.

**Direct implications for FP4 upgrade:**

| Property | Deep_gate INT8 | FP4 upgrade |
|----------|---------------|-------------|
| Cells per weight | 4 × 4-bit banks | 1 × 4-bit cell (FP4 code directly) |
| Sign handling | pos/neg differential bank pair | Same differential approach, or 1-bit sign cell |
| ADC per column | 8-bit (reads one row's partial product) | Same 8-bit if row-serial retained |
| Accumulation | INT32 digital counter | INT32 digital counter (same) |
| Crossbar utilisation | 3.1% (32×32, small model) | ~100% for LLM layers (4096×4096) |

**FP4 cell-count vs INT8 (Deep_gate differential nibble-sliced baseline = 4 cells/weight):**
- Baseline INT8: 4 cells/weight (pos0, pos1, neg0, neg1 — two 4-bit nibble pairs)
- FP4 with S⁺/S⁻ digital routing (sign via XOR, not stored): 1 five-bit magnitude cell → **4× reduction**
- FP4 with separate 1-bit SLC sign cell: 2 cells (magnitude + sign) → **2× reduction**
The 4× claim requires explicitly stating that sign is handled by routing, not stored in RRAM.

**ADC precision re-framing (important for RQ5):**  
The Deep_gate approach shows that row-serial decomposition relaxes the ADC requirement
from "accumulate over K rows" (needs log2(K×36/0.25) ≈ 21 bits at K=4096) down to
"read single-row contribution" (8 bits). The photonic MAC design could adopt the same
strategy: optically compute one-row-at-a-time and digitally accumulate, trading latency
for dramatic reduction in required photonic-domain precision.

---

## Collaboration Interface

### To RRAM colleague:
- FP4 weight distributions per layer (code histograms, per-tensor scales)
- Conductance target levels: {0, 0.5, 1, 1.5, 2, 3, 4, 6} × s_W / 6
- Sign/magnitude statistics
- Variation tolerance per layer: max σ_rel before PPL exceeds threshold

### To photonics colleague:
- Per-layer **partial sum distributions** (min, max, mean, std, 99th/99.9th percentile,
  dynamic range) — this is the signal the photonic accumulator actually sees at each column,
  not the final layer output; the two differ because residual adds and normalisations happen after
- Required accumulator ENOB per layer type (attn vs. MLP), derived from partial sum DR
- Overflow rate vs. accumulator width curve (RQ9) → practical clipping threshold and
  minimum safe accumulator spec per operating point
- Layer-wise sensitivity ranking

---

## Results Directory
```
results/
├── rq1/    PPL and accuracy tables
├── rq2/    Boundary sweep results
├── rq3/    Conductance mapping analysis
├── rq9/    Overflow sweep (PPL vs accumulator N) + partial sum distributions
├── rq9b/   Partial-sum growth by K, FP16 decoder equivalence tables + plots
├── rq10/   Subarray K sweep (PPL + analytical hardware cost)
└── rq11/   Scale format comparison (E8M0 vs E4M3 vs FP32)
```

---

## Environment
- Libraries: torch, transformers, lm_eval, datasets, numpy, matplotlib, pandas, tqdm
- GPU: multi-GPU, CUDA_VISIBLE_DEVICES as needed
- Models cached at: `/users/grad/zhangjintao/models/`
