# X1 LLM Compiler

This folder is the deployable compiler path for mapping FP4LLM/Qwen projection
weights and tile schedules onto X1 memory-compute hardware.

It keeps only the hardware-facing pieces:

- model/layer mapping
- X1 tile planning
- CSR/ISA lowering
- SAR/TDC NoC packet encoding
- streaming fabric descriptors
- firmware descriptor-driver emission
- real weight-image emission for X1 cores

Generated artifacts, Python runtime smoke tests, LM evaluation glue, testbench
emitters, and caches are intentionally not kept in this folder.

## Important Files

| File | Purpose |
|---|---|
| `x1_llm_compiler/mapping.py` | generic layer mapping parser |
| `x1_llm_compiler/qwen_mapping.py` | Qwen config to compiler layers |
| `x1_llm_compiler/planner.py` | layer/tile task planning |
| `x1_llm_compiler/isa.py` | X1 CSR/ISA operations |
| `x1_llm_compiler/noc.py` | SAR/TDC NoC header and payload encoding |
| `x1_llm_compiler/fabric.py` | streaming descriptors and X1UP fabric plan |
| `x1_llm_compiler/deployment.py` | hardware capacity/deployment report |
| `x1_llm_compiler/firmware_driver.py` | Caravel management-core descriptor firmware |
| `x1_llm_compiler/weight_image.py` | real tensor to packed X1 weight payload image |
| `x1_llm_compiler/cost_model.py` | latency and energy estimates for a manifest and stream plan |
| `x1_llm_compiler/mapping_optimizer.py` | mapping search and latency/energy ranking |

## Weight Image Format

`weight_image.py` loads extracted Qwen tensors from `index.csv`, converts HF
`[out,in]` weights into crossbar `[in,out]`, applies FP4LLM-style block-16
weight quantization, quantizes to signed int8, and decomposes each weight into
four 4-bit X1 banks:

| Macro | Bank |
|---:|---|
| 0 | positive low nibble |
| 1 | positive high nibble |
| 2 | negative low nibble |
| 3 | negative high nibble |

Each programming word is a little-endian `uint32`:

```text
{data[31:24], macro[23:16], col[12:8], row[4:0]}
```

The payload byte count is `payload_words * 4`. A full 32x32 X1 tile with four
banks emits `4096` payload words, or `16384` bytes.

## Commands

From this folder:

```powershell
& "$env:USERPROFILE\anaconda3\python.exe" -m x1_llm_compiler.cli compile-qwen-config --model-id Qwen/Qwen3-0.6B --model-name Qwen3-0.6B --dst 3 --out artifacts/qwen3_0_6b_x1_manifest.json
& "$env:USERPROFILE\anaconda3\python.exe" -m x1_llm_compiler.cli capacity-report --manifest artifacts/qwen3_0_6b_x1_manifest.json --max-x1 20000 --out-json artifacts/qwen3_0_6b_deployment.json --out-md artifacts/qwen3_0_6b_deployment.md
& "$env:USERPROFILE\anaconda3\python.exe" -m x1_llm_compiler.cli emit-stream-plan --manifest artifacts/qwen3_0_6b_x1_manifest.json --out-json artifacts/qwen3_0_6b_stream_plan.json --out-csv artifacts/qwen3_0_6b_stream_plan.csv
& "$env:USERPROFILE\anaconda3\python.exe" -m x1_llm_compiler.cli emit-firmware --stream-plan artifacts/qwen3_0_6b_stream_plan.json --out-c artifacts/x1_streaming_descriptor_firmware.c --out-json artifacts/x1_streaming_descriptor_firmware.json
```

Estimate a compiled plan:

```powershell
& "$env:USERPROFILE\anaconda3\python.exe" -m x1_llm_compiler.cli estimate-cost --manifest artifacts/qwen3_0_6b_x1_manifest.json --stream-plan artifacts/qwen3_0_6b_stream_plan.json --out-json artifacts/qwen3_0_6b_cost.json
```

Search mapping choices:

```powershell
& "$env:USERPROFILE\anaconda3\python.exe" -m x1_llm_compiler.cli optimize-mapping --prefill ../mappings/Qwen3_1_7B_prefill.m --decode ../mappings/Qwen3_1_7B_decode.m --out-json artifacts/qwen3_1_7b_mapping_optimization.json --out-manifest artifacts/qwen3_1_7b_best_manifest.json --out-stream-json artifacts/qwen3_1_7b_best_stream_plan.json
```

The optimizer also writes the selected MAESTRO-style mappings by default to
`artifacts/best_prefill_mapping.m` and `artifacts/best_decode_mapping.m`.
Override those paths with `--out-prefill-mapping` and
`--out-decode-mapping` when needed. Hardware-only choices such as PE cluster
size, weight residency, DMA window size, NoC coalescing, and reduction
placement remain recorded in the optimization JSON and compiled manifest.

The cost-model defaults are architectural placeholders. Calibrate
`CostModelConfig` with measured DMA, programming, PE, ADC, reduction, and NoC
values before treating absolute latency or energy numbers as hardware claims.

Real weight image emission:

```powershell
& "$env:USERPROFILE\anaconda3\python.exe" -m x1_llm_compiler.cli emit-weight-image --index extracted_pe\index.csv --layer 0 --role self_attn.q_proj --out-bin artifacts\qproj_tile00.u32.bin --out-json artifacts\qproj_tile00.json --tile-row 0 --tile-col 0
```
