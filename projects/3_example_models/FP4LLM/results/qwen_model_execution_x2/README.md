# Qwen3-1.7B mixed FPGA + software-crossbar prefill

## Completed scope

- Prompt: `Hello` (token ID `9707`, position 0).
- Transformer block 0: consumed from the verified physical X2 FPGA artifact; it was **not rerun**.
- Transformer blocks 1 through 27: executed in order by the Python FP4LLM software-crossbar `CrossbarLinear` wrapper.
- Final RMS normalization and language-model head: executed by PyTorch after block 27.
- Resulting next token: ID `25`, decoded as `':'`.

This is a mixed-arithmetic prefill experiment. The FPGA block used INT8 linear arithmetic, while the continuation used W7/A7 blockwise quantization with BF16 boundaries. It is not an all-FPGA run. Decode was not executed because a correct decode path must merge the FPGA block-0 K/V cache with caches from blocks 1–27.

## FPGA processing elements and engine allocation

Transformer block 0 used all six physical engines in the Qwen6 FPGA overlay:

| Hardware | Count | Function |
|---|---:|---|
| LPC matrix engines | 4 (engines 0–3) | INT8 matrix/vector products with INT32 tile outputs |
| MAC processing elements per LPC | 4 (`N_PE=4`) | Parallel MAC lanes inside each LPC |
| Physical MAC PEs | 16 | Four LPCs × four MAC PEs |
| NLPC engines | 2 (engines 4–5) | INT32 partial-sum reduction and vector/nonlinear operations |

The four LPC engines completed 49,280 MVM commands: engine 0 completed 12,368 and engines 1–3 completed 12,304 each. This is a command count, not an additional physical-PE count. The fixed FPGA footprint is 16 MAC PEs plus two NLPC engines.

The LPC operates on at most 32 input channels and 32 output channels per tile. Large Qwen projections therefore produce many INT32 partial results. NLPC opcode 1 combines those partial results until each complete projection output is available.

## What the NLPC performed

NLPC means the nonlinear/reduction processing portion of this overlay. It does more than activation functions: it also performs integer reduction and elementwise vector operations that are unsuitable for the LPC matrix datapath.

| Opcode | Operation | Engine 4 calls | Engine 5 calls | Total calls | Work in block 0 |
|---:|---|---:|---:|---:|---|
| 0 | `PROBE` | 1 | 1 | 2 | Verify both NLPC cores and the expected hardware ABI before model execution |
| 1 | `ADD_I32` | 24,304 | 24,256 | 48,560 | Merge the LPCs' INT32 partial sums across input-channel tiles |
| 2 | `RMS_NORM` | 4 | 2 | 6 | Input RMSNorm, Q norm, K norm and post-attention RMSNorm |
| 3 | `ROPE` | 2 | 2 | 4 | Apply rotary position encoding to Q and K |
| 4 | `CAUSAL_SOFTMAX` | 16 | 0 | 16 | Scale, causally mask and normalize attention scores for 16 query heads |
| 5 | `SILU_MUL` | 1 | 0 | 1 | Calculate `SiLU(gate) × up` in the MLP |
| 6 | `RESIDUAL_ADD` | 2 | 0 | 2 | Add the attention residual and the MLP residual |

Rows are split into contiguous halves between NLPC engines 4 and 5 when there are enough independent rows. A one-row `Hello` tensor has no second half, so its single-row residual, MLP activation and related operations execute only on engine 4. Q/K head rows are numerous enough to use both engines. Each NLPC command supports at most 4096 rows and 6144 columns.

The block-0 FPGA dataflow was:

```text
input
  -> NLPC input RMSNorm
  -> LPC Q/K/V projections
  -> NLPC Q/K normalization and RoPE
  -> LPC QK^T attention scores
  -> NLPC scaled causal softmax
  -> LPC probability x V
  -> LPC output projection
  -> NLPC residual add and post-attention RMSNorm
  -> LPC gate/up projections
  -> NLPC SiLU(gate) x up
  -> LPC down projection
  -> NLPC final residual add
```

The processing system did not provide a numerical fallback. It was limited to quantization/dequantization, tensor layout and copies, and DMA scheduling.

## NLPC nonlinear arithmetic and numerical effects

`ADD_I32` is not nonlinear. It adds two INT32 partial tensors using an INT64 temporary and stops with an overflow status if the result cannot be represented in INT32. The remaining NLPC operations use FP32 internally and explicitly round their outputs to BF16.

### RMSNorm

For each row, the NLPC computes

```text
inv_rms = 1 / sqrt(mean(x^2) + epsilon)
y = BF16(BF16(x * inv_rms) * gain)
```

The verified Qwen configuration uses `epsilon = 1e-6`. Normalization compresses or expands the entire row according to its RMS energy. The deliberate BF16 rounding before the gain multiplication reproduces Qwen's expected boundary behavior but removes low mantissa bits twice: after normalization and at the final output.

### Rotary position encoding

RoPE is a position-dependent rotation rather than an activation nonlinearity. For paired channels, the hardware evaluates the cosine and sine products separately, rounds each product to BF16, adds them, and rounds the result again:

```text
y = BF16(BF16(x * cos) + BF16(rotate_half(x) * sin))
```

This can introduce a small phase/amplitude error relative to an unrounded FP32 rotation. For this single-token prefill, the position is zero, but the same hardware path and BF16 boundaries are still exercised.

### Scaled causal softmax

For each attention row, the NLPC multiplies scores by `1/sqrt(128)`, masks positions beyond the causal boundary, subtracts the valid-row maximum, evaluates `exp`, sums the valid exponentials, divides by that sum and rounds probabilities to BF16. Maximum subtraction avoids exponential overflow. BF16 rounding can suppress very small attention probabilities or slightly change probability ratios, especially when logits are close together. With the one-token `Hello` input, each head has only one valid attention position, so the ideal probability is 1.0 and this run does not stress long-context softmax sensitivity.

### SiLU gate multiplication

The MLP activation uses a numerically stable sigmoid:

```text
sigmoid(g) = 1 / (1 + exp(-g))                  when g >= 0
sigmoid(g) = exp(g) / (1 + exp(g))              when g < 0
y = BF16(BF16(g * sigmoid(g)) * up)
```

SiLU suppresses large negative gate values, behaves approximately linearly for large positive values, and smoothly gates values around zero. BF16 rounding occurs after the SiLU result and again after multiplication by the `up` branch.

### Residual addition

Attention and MLP residuals are added in FP32 and rounded once to BF16:

```text
y = BF16(residual + branch_output)
```

When the operands differ greatly in magnitude, BF16 can discard a small increment. This is a floating-point boundary effect, not analog crossbar saturation.

### Measured block-0 numerical agreement

The physical FPGA block exactly matched the host implementation of the same INT8-linear/FP32-NLPC arithmetic:

| Comparison | Relative L2 error | Maximum absolute error | Exact match |
|---|---:|---:|---|
| FPGA output vs INT8 host reference | 0 | 0 | Yes |
| FPGA output vs host NLPC-kernel output | 0 | 0 | Yes |
| FPGA output vs BF16 model reference | 0.0646815631 | 0.25 | No |
| FPGA K cache vs host | 0 | 0 | Yes |
| FPGA V cache vs host | 0 | 0 | Yes |

The BF16-reference difference measures the complete mixed-precision FPGA block and cannot be attributed to the NLPC alone. It includes INT8 projection quantization, accumulation/reconstruction, and BF16 boundaries. The exact host-kernel match demonstrates that the physical FPGA reproduced the intended hardware arithmetic for this input.

## Software-crossbar PE usage for blocks 1–27

The software continuation uses logical 64-row × 32-column crossbar tiles. These are simulated PE/tile instances executed on the CPU; they are not additional physical FPGA PEs and are not the 32 Slurm CPU cores.

The mapping-level prefill footprint is 18,496 logical PEs per transformer block:

```text
Q projection                 2,048
K projection                 1,024
V projection                 1,024
QK^T attention                  32
attention x V                   32
output projection            2,048
shared gate/up mapping       6,144
down projection              6,144
                            ------
mapping total               18,496 logical PEs/block
```

Across 27 software blocks this is 499,392 mapped logical PE instances. The mapping count treats gate and up as one reusable `ffn_up` role.

The runtime telemetry counts gate and up as two separately instantiated projections. Consequently, the actual wrapper activity for this one-token run is:

| Runtime tile category | Per block | Across 27 blocks |
|---|---:|---:|
| Static Q/K/V/O and gate/up/down projection tiles | 24,576 | 663,552 |
| Dynamic one-token attention tiles across 16 heads | 96 | 2,592 |
| Total simulated tile instances | 24,672 | 666,144 |

Blocks are data-dependent and therefore execute in order. Block-streamed programming released one block before activating the next, so peak transformer-block residency was one. The final RMS normalization and language-model head ran in ordinary PyTorch and are not included in the software-crossbar PE totals.

## Crossbar nonideality effects in this run

The continuation exercised W7/A7 blockwise quantization, a block size of 16, integer crossbar MAC reconstruction and BF16 boundaries. Those effects are part of the reported output.

The following optional device-nonideality models were deliberately disabled in the forward execution:

| Optional effect | Status in this run | What it would model if enabled |
|---|---|---|
| Read/program noise | Disabled | Conductance/read variation and output dispersion |
| Reliability/bit errors | Disabled | Raw bit errors, retention drift and state variation |
| I–V/ADC nonlinearity | Disabled | Hard-clamp or tanh transfer behavior, ADC full-scale clipping and saturated tile columns |
| SAR-TDC quantization/timing | Disabled | Coarse/fine conversion error, finite output precision and timing-dependent conversion behavior |
| Saturation audit | Disabled | Counts of clipped branches, columns and tile reads for the actual prompt tensors |

Therefore, this result must not be interpreted as a measurement of analog saturation, read noise, retention error or TDC conversion error. Its software-side accuracy effects come from W7/A7 quantization and BF16 rounding. A separate run with the hardware-effect and saturation audits enabled is required to quantify those device nonidealities. On the FPGA LPC, TDC timing was also disabled for this Qwen6 build; the measured FPGA result still includes its real INT8 datapath and the NLPC arithmetic described above.

## End-to-end execution flow and files

The hybrid run has a strict handoff boundary. The FPGA creates the hidden state after transformer block 0, and the Python runner consumes that tensor as the input to transformer block 1. Embedding and block 0 must not be run again in Python.

```text
Pinned Qwen checkpoint + "Hello"
  |
  |  export_fpga_bundle.py / fpga/qwen6/fetch_reference_bundle.py
  v
Qwen6 FPGA deployment bundle
  |
  |  main.py --execution-target fpga6
  v
src/qwen_fpga6.py
  |---- ps_int8_quantization.py: BF16/INT8 preparation and reconstruction
  |---- axi_x2_mvm_driver.py: LPC engines 0-3
  |---- fpga/qwen6/hls/qwen_nlpc.cpp: NLPC engines 4-5
  v
fpga_block1_prefill.npz
  |  output = block-0 hidden state [1, 2048]
  |  key_cache/value_cache = block-0 prefill cache
  |
  |  run_qwen_fpga_continuation.py
  v
Load pinned Qwen3-1.7B weights
  |
  |---- mappings/Qwen3_1_7B_prefill.m
  |---- mappings/Qwen3_1_7B_decode.m (shape/capacity validation only here)
  |---- src/qwen_crossbar_runtime.py
  |---- src/crossbar_wrapper.py
  |---- reram_crossbar.py
  v
Software-crossbar blocks 1 -> 2 -> ... -> 27
  |
  v
PyTorch final RMSNorm -> tied language-model head -> argmax
  |
  v
continuation_outputs.npz + metadata/CSV/log/checksum artifacts
```

### FPGA preparation and execution files

| File | Role in the FPGA stage |
|---|---|
| `export_fpga_bundle.py` | Tokenizes the prompt, reads the pinned Qwen checkpoint and exports block-0 BF16-rounded inputs, rotary constants and weights into a `qwen6-bundle-v1` NPZ file. |
| `fpga/qwen6/fetch_reference_bundle.py` | Fetches the same reference bundle directly from the pinned Hugging Face revision when a local checkpoint is not used. |
| `main.py` | Parses the command line. `--execution-target fpga6` dispatches to `src.qwen_fpga6.run_from_args`. |
| `src/qwen_fpga6.py` | Validates bundle shapes and metadata, loads the overlay, schedules the complete block-0 graph, collects engine counts and writes the FPGA outputs. |
| `src/ps_int8_quantization.py` | Quantizes BF16 model operands for LPC INT8 execution and reconstructs floating-point outputs from INT32 accumulators and scales. |
| `fpga/qwen6/overlay/qwen6.bit` and matching `.hwh` | Programs the ZCU106 with four LPC and two NLPC engines. The actual board path is recorded in `fpga_summary.json`. |
| `axi_x2_mvm_driver.py` | Board-side AXI/DMA driver for LPC engines 0–3. It is supplied through `--fpga-driver` and dynamically loaded by `src/qwen_fpga6.py`. |
| `fpga/qwen6/hls/qwen_nlpc.cpp` | Synthesizable implementation of NLPC opcodes 0–6, including validation, overflow/nonfinite reporting and BF16 rounding. |
| `fpga/qwen6/build_overlay.tcl` | Connects the four LPC and two NLPC IP blocks when constructing the Qwen6 overlay. |
| `fpga/qwen6/board_smoke.py` | Independent bring-up test that verifies all six physical engines before the real Qwen bundle is run. |

The compact `hello_input.npz` in the original handoff records the input embedding, positional constants and configuration used for validation. It is not the complete weight-bearing deployment bundle; the block weights came from the separately validated pinned bundle identified by `hello.source.json`.

During execution, `main.py` passes the bundle, bitstream and driver to `src/qwen_fpga6.py`. `Fpga6Backend.open` checks that the overlay exposes four LPCs and two Qwen NLPC cores, programs the FPGA and probes both NLPC ABIs. `run_block` then follows the block graph shown above. Each large matrix product is split into LPC tiles, and NLPC `ADD_I32` merges the channel-tile partial sums. The same NLPC engines execute normalization, RoPE, softmax, gating and residual operations.

### FPGA output and validation files

| File | Produced/used for |
|---|---|
| `fpga_block1_prefill.npz` | Primary handoff. `output` is the BF16-rounded hidden state entering block 1; `key_cache` and `value_cache` hold block-0 prefill cache tensors. |
| `fpga_summary.json` | Physical-board flag, arithmetic modes, engine operation counts, projection shapes, stage timings, bundle metadata and bitstream path. |
| `board_hello_run.txt` | Captured board console command and successful completion status. |
| `hello.source.json` | Pinned checkpoint revision, prompt/token IDs and source-tensor hashes. The Python continuation validates this provenance before running. |
| `accuracy.json` | FPGA output and K/V cache comparisons against the INT8, BF16 and host-NLPC references. |
| `host_validation.json` and `host_validation_outputs.npz` | Separate host-only reference description and tensors from `fpga/qwen6/validate_reference_host.py`; retained in the original `qwen_model_execution_on_x2` handoff. |
| `SHA256SUMS.txt` | Integrity checks for the files in each packaged result directory. |

### Python continuation files and call flow

The production continuation is implemented by the copy of `run_qwen_fpga_continuation.py` in this directory:

1. It opens `fpga_block1_prefill.npz`, selects the `output` array and validates the required shape `[1, 2048]`.
2. It reads `hello.source.json` and requires the exact checkpoint revision `70d244cc86ccca08cf5af4e1e306ecf908b1ad5e`, prompt `Hello` and token ID `9707`.
3. It loads `Qwen/Qwen3-1.7B` from the local model store using `src/qwen_crossbar_runtime.py` helpers.
4. For each original model block index 1–27, `load_mapping_context` reads the prefill and decode mapping entries and verifies dimensions and token capacities. Only the prefill mapping drives this prefill forward pass.
5. `replace_block_linears` replaces Q/K/V/O and gate/up/down PyTorch linear modules with `CrossbarLinear` modules from `src/crossbar_wrapper.py`.
6. `install_mapped_attention_gemms` routes `QK^T` and probability-times-V through the same mapped crossbar path.
7. The runner supplies the FPGA output as `inputs_embeds`, installs only layers 1–27 into the temporary decoder stack and calls the model with position ID 0. Therefore embedding and block 0 are skipped.
8. Block-streaming hooks program one block's simulated tiles before its forward call and release them afterward. The output of one block immediately becomes the next block's input.
9. After block 27, the normal Qwen final RMSNorm and tied language-model head run in PyTorch. Argmax over the resulting logits produced token ID 25 (`:`).
10. The runner restores the original model structure, saves results and emits metadata, timing, mapping and checksum artifacts.

The key software files are:

| File | Role in the Python stage |
|---|---|
| `results/qwen_model_execution_x2/run_qwen_fpga_continuation.py` | Exact orchestration script used for blocks 1–27, final norm, LM head and artifact generation. |
| `src/qwen_crossbar_runtime.py` | Qwen loading, mapping validation, module replacement, mapped attention and block-streaming lifecycle implementation. |
| `src/crossbar_wrapper.py` | `CrossbarLinear` PyTorch adapter, blockwise W7/A7 quantization and tile orchestration. |
| `reram_crossbar.py` | Low-level `CrossbarArray.compute` implementation used by the wrapper backend. |
| `run_qwen3_mapping_crossbar_eval.py` | Mapping parser and `LayerSpec`/tile-dimension definitions reused by the runtime. |
| `mappings/Qwen3_1_7B_prefill.m` | Prefill dimensions, token capacity and logical crossbar mapping for all 28 blocks. |
| `mappings/Qwen3_1_7B_decode.m` | Decode mapping loaded for consistency/capacity checks; decode itself was not executed. |

### Python result files

| File | Contents |
|---|---|
| `continuation_outputs.npz` | Copied FPGA block-0 output, outputs after each software block 1–27, final normalized hidden state and full vocabulary logits. |
| `metadata.json` | Commit and model provenance, block range, quantization, shapes, next token, thread/Slurm configuration, timing and tile lifecycle. |
| `block_timings.csv` | One measured execution time for every software block index 1–27. |
| `linear_crossbar_modules.csv` | Per-block Q/K/V/O and MLP module dimensions, mapping layers and logical tile counts. |
| `mapped_attention_gemms.csv` | Actual one-token QK and attention-value shapes, head multiplicity, MACs and tile counts. |
| `mapping_validation.csv` | Evidence that runtime tensor sizes fit their mapped prefill/decode capacities. |
| `run.log` | Model loading, all block-ready events, active/release progression, per-block completion times and final status. |
| `README.md` | Scope, arithmetic interpretation, PE/NLPC details, execution flow and reproduction procedure. |
| `SHA256SUMS.txt` | SHA-256 checksums for every delivered artifact other than the checksum file itself. |

## Parallel execution

Slurm job `15941` on `bose01` reserved 32 CPU cores and 131072 MiB (128 GiB) of RAM. The blocks remain sequential because each block consumes the preceding block's hidden state; thread parallelism is used inside PyTorch/NumPy crossbar kernels. Peak resident software-crossbar blocks: 1.

## Timing

- Blocks 1–27 plus final norm: 2433.757421 s
- Language-model head: 0.028530 s
- Total measured execution: 2463.979441 s

Per-block timings are in `block_timings.csv`.

## Reproduction steps

1. Check out project commit `7f8bcee0cf9cf4e6e24687359b614c1abbf3b829`.
2. Use Qwen/Qwen3-1.7B revision `70d244cc86ccca08cf5af4e1e306ecf908b1ad5e`.
3. Reserve the compute node, for example:

   ```bash
   srun --pty --nodelist=bose01 --cpus-per-task=32 --mem=128G --time=12:00:00 bash
   ```

   X11 is not required; this run is headless.

4. Activate the environment and constrain numerical libraries to the allocation:

   ```bash
   source ~/venvs/fp4llm-qwen3/bin/activate
   export OMP_NUM_THREADS=32 MKL_NUM_THREADS=32 OPENBLAS_NUM_THREADS=32 NUMEXPR_NUM_THREADS=32
   ```

5. Run the continuation from the project root:

   ```bash
   python results/qwen_model_execution_x2/run_qwen_fpga_continuation.py --input-package results/qwen_model_execution_on_x2 --out-dir results/qwen_model_execution_x2_reproduced --local-root ~/hf_models --start-block-index 1 --end-block-index 27 --backend wrapper --weight-bits 7 --act-bits 7 --block-size 16 --dtype bfloat16 --threads 32
   ```

6. Verify every original artifact from this run with `(cd results/qwen_model_execution_x2 && sha256sum -c SHA256SUMS.txt)`.

## Artifacts

- `continuation_outputs.npz`: FPGA block-0 output, each software block output, final normalized hidden state, and logits.
- `metadata.json`: provenance, configuration, versions, shapes, timings, and next-token result.
- `block_timings.csv`: one timing row per software-executed transformer block.
- `linear_crossbar_modules.csv`: mapping and tile metadata for each crossbar-backed projection.
- `mapped_attention_gemms.csv`: executed QK^T and attention-value crossbar operations.
- `run.log`: captured console output from the execution.
- `run_qwen_fpga_continuation.py`: exact runner used.
- `fpga_block1_prefill.npz`, `hello.source.json`, `accuracy.json`, `fpga_summary.json`: copied source provenance.
- `SHA256SUMS.txt`: checksums for the folder contents.
