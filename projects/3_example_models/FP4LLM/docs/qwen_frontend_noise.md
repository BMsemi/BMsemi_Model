# Qwen frontend noise comparison

Run from the FP4LLM project root using a Python environment with PyTorch,
Transformers supporting Qwen3, and NumPy:

```powershell
python qwen_frontend.py --model /path/to/Qwen3-1.7B --local-files-only --plan "all:crossbar" --crossbar-config configs/qwen_crossbar_noise.json
```

Omit `--local-files-only` to allow Hugging Face model downloads. Use
`--prompt`, `--reference-text`, `--max-new-tokens`, and `--seed` to define the
experiment. The clean run disables both read noise and reliability; the noisy
run enables both. This measures their combined effect, not read noise alone.
Both runs retain identical W7/A7 quantization and BF16 tensor boundaries.
The example reliability settings (10 ppm raw BER, 0.02 log-resistance state
spread, ECC off) are illustrative assumptions, not measured board parameters.
Read noise uses the simulator's physical noise defaults. IV and TDC are off
in both arms to avoid mixing their effects into the comparison.

On this Windows workspace, `run_qwen_noise_comparison.ps1` uses the downloaded
checkpoint at `D:\hf_models\Qwen--Qwen3-1.7B` and the `llmqwen` environment.
It starts a two-token `Hello world` probe and one continuation token, logging
progress to `results/qwen_noise_comparison/run.log`. This is a minimal full-model
sensitivity run, not a representative language-quality evaluation. Both weight
shards were downloaded at revision `70d244cc86ccca08cf5af4e1e306ecf908b1ad5e`
and verified against their Hugging Face SHA-256 values.

`all:crossbar` explicitly means all PyTorch linear modules, including the LM
head and every decoder projection, plus dynamic attention QK/AV matmuls.
Embeddings, normalization, RoPE, softmax, and elementwise operations stay in
PyTorch. This is a software device simulation, not FPGA execution. Dynamic
attention matrices are programmed from current K/V and probability operands;
the framework KV cache stores activations. Dynamic matrices use fixed cell-fault
seeds and read-dependent noise seeds; no physical cache endurance is simulated.

The frontend batches the original row-serial cell-read arithmetic with NumPy.
Tests compare its exact integer outputs against the original simulator with
identical random seeds across repeated clean and noisy reads. This preserves
per-cell noise, ADC rounding, differential subtraction, and RNG draw order.
The clean case uses the algebraically identical integer dot product after the
same quantization, rather than constructing noiseless cells. Its outputs are
checked against the existing tile simulator. Noise standard deviations use a
lookup table over the 16 cell levels, calculated by the original noise model.
For noise/fault-only runs, up to 1,024 independent columns (32 logical tiles)
are evaluated in one NumPy batch. Row sums and cell ADC operations remain
independent for each column. IV, TDC or ECC settings restore 32-column batches.
Random seeds are assigned per computational batch, so this has the same
independent-cell noise distribution, but a different seeded realization from
a run that constructs every 32-column tile separately.
Within a forward call, each faulted tile is reconstructed once and used for all
tokens, resetting each read to its reproducible noise seed. Tests verify exact
agreement with token-by-token reconstruction over repeated calls. Nearest
resistance-state lookup uses two neighboring levels, with explicit tie handling
checked against the original exhaustive search.

Tiles are reconstructed on demand with the same reliability seed at every read,
so programmed faults stay fixed. Read-noise seeds additionally include a read
counter. This is deterministic fault replay, not additional physical programming
or an endurance-write simulation. It avoids retaining all cell objects in RAM.
Compact weights and the embedding still occupy memory. Full 1.7B simulation
is expensive; no automatic small-model fallback is performed.

Outputs under `results/qwen_noise_comparison` include both configurations,
mapped module names, generated tokens/text, saved logits, and `comparison.json`.
Reference perplexity and KL compare identical teacher-forced tokens, avoiding
the confounding context divergence of free-running generations. Positive
perplexity change means degradation on that reference text. One short reference
and one seed cannot establish general benchmark quality. Repeat with independent
seeds and a representative held-out corpus for a robust estimate.

`python -m unittest test_qwen_frontend -v` exercises a tiny randomly initialized
Qwen3 through actual crossbars, including cached decoding and comparison metrics.
These tests are implementation checks, not Qwen3-1.7B quality results.

If the clean arm has completed, `--resume-clean` reuses its saved logits and
output only when model path, prompt, reference, seed, and device configuration
match. The Windows launcher exposes this as `-ResumeClean`.

## Completed full-checkpoint probe

The downloaded Qwen3-1.7B checkpoint completed both arms with seed 1234,
raw prompt/reference `Hello world` (token IDs 9707, 1879), and one generated
continuation token. All 197 linear modules and attention QK/AV in all 28
decoder blocks were mapped through the crossbar frontend.

| Measurement | Clean | Read noise + reliability |
|---|---:|---:|
| Reference-text perplexity | 267,979.25 | 1,875,842.875 |
| Reference negative log-likelihood | 12.498665 | 14.444569 |
| Continuation | ` of` (315) | `!` (0) |

Perplexity increased by 599.9955% (6.99996 times). Logit RMSE was 1.36036,
relative L2 difference was 41.1458%, and mean KL(clean || noisy) was 2.21203.
Teacher-forced top-token agreement was 0/1 evaluated reference transitions.
These metrics measure the combined effect of noise and faults against the
same quantized clean path. The reference has only one scored transition, and
the generated continuation has only one token, so this cannot establish
general language-quality degradation. In particular, the changed continuation
alone does not demonstrate worse prose.

Full settings, logits, tokens, and results are saved under
`results/qwen_noise_comparison/`, including `comparison.json` and `run.log`.
Six frontend regression tests passed; their log is
`results/qwen_frontend_tests.log`.
