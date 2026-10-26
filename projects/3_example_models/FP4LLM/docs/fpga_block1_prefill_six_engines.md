# Qwen block 1 prefill on six ZCU106 engines

The `main.py --execution-target fpga6` path is implemented for **block index 0,
prefill only**. It requires the new Qwen6 overlay. The old TCAM/YOLO six-MVM
overlay is incompatible. The board is controlled through its Linux console on
**COM17, 115200 baud, 8 data bits, no parity, 1 stop bit, no flow control**.
Ethernet, SSH and an IP address are not required. Physical board validation is
pending: COM17 opens, but the latest checks received no output after Enter.
The new overlay has passed synthesis, routing and 100 MHz timing checks, and
its bitstream has been generated. See `fpga/qwen6/validation.md` for measured
build/test status, remaining tool warnings and the quantization accuracy limit.

## Execution boundary

| Location | Work | Data type |
|---|---|---|
| Offline preparation | Tokenization, embedding lookup, checkpoint extraction, positional constants | Real BF16 weights in FP32 containers |
| ARM PS | Quantization, scale reconstruction, layout/copies, DMA scheduling | INT8 codes and scale metadata |
| FPGA engines 0–3 | Q/K/V/O, attention products, gate/up/down projections | INT8 multiply, INT32 tile result |
| FPGA engines 4–5 | Merge all C-tile partial sums | INT32 with overflow check |
| FPGA engines 4–5 | RMSNorm, Q/K norm, RoPE, scaled causal softmax, SiLU × up, residual adds | FP32 arithmetic, BF16 rounding |

There is no CPU model-arithmetic fallback. The tests and board-smoke script use
CPU reference arithmetic separately, as validation. Quantization/dequantization
on PS is the explicitly allowed exception.

Only the first transformer block runs. No decode, later blocks, final norm,
LM head or sampling is performed. Outputs are hidden states and rotated K/V
cache tensors. Software mode remains the default; its default is W7/A7.
FPGA mode defaults to W8/A8 and requires BF16 operands.

## Dependency order

1. Input RMSNorm on 4–5.
2. Q/K/V projections on 0–3.
3. Learned per-head Q/K norm and half-vector RoPE on 4–5.
4. Q × K transpose on 0–3; scale, causal mask and stable softmax on 4–5.
5. Probability × V on 0–3, using each query's visible prefix.
6. Head layout copy on PS; output projection on 0–3.
7. First residual add and post-attention RMSNorm on 4–5.
8. Gate/up projections on 0–3; SiLU(gate) × up on 4–5.
9. Down projection on 0–3; second residual add on 4–5.

For Qwen3-1.7B: hidden width 2048, FFN width 6144, 16 query heads, 8 KV heads,
head width 128. Query head h reads KV head floor(h/2). RMS epsilon is 1e-6.
RoPE base is 1,000,000 and positions start at zero. Scaled-RoPE, sliding-window,
bias and non-SiLU variants are rejected.

## Spatial and temporal scheduling

Each LPC transfer covers at most 32 tokens × 32 input channels × 32 output
channels. One wave assigns output columns 0:32, 32:64, 64:96 and 96:128 to
engines 0–3. All available LPC jobs start before waiting. Later waves advance
128 output columns. Each output wave iterates over every 32-channel C chunk.

The first C result initializes an accumulator buffer. Every subsequent C result
is added by NLPC engine 4 or 5. Accumulators travel through PS DDR between
commands; the PS copies them but never adds them. Two independent additions
can run concurrently. The same two NLPC engines handle nonlinear stages later.

The build uses four MAC lanes per LPC (`N_PE=4`) and disables TCAM timing.
A 32-column tile does not imply 32 simultaneous multipliers. Weights stream
from DDR and are reused across the token batch; the whole model is not on-chip.

NLPC rows are divided into contiguous halves, at most 4096 rows per engine
per command. A complete reduction row stays on one engine. Softmax retains
the global query offset. Maximum NLPC row width is 6144; this block runner
supports 1–128 tokens.

PS quantization uses one scale per activation row and one per weight output
channel over the full C dimension. Scales stay fixed across C tiles. All-zero
rows use scale 1; codes are signed [-127,127]. No PS zero-point is added.
The original LPC RTL implements its own PWM offset. A 6144-channel sum is
bounded by 99,096,576, safely inside INT32. Dequantization follows the complete
integer reduction.

For P × V, quantization sees only each query's visible prefix. Otherwise future
V rows could change earlier INT8 scales despite masking. A regression test
changes the final token and verifies all earlier outputs remain identical.

## Source files

- `src/ps_int8_quantization.py`: PS format conversion.
- `src/qwen_fpga6.py`: drivers, scheduler, block graph and report.
- `export_fpga_bundle.py`: real weights, tokens, embeddings and rotary constants.
- `fpga/qwen6/hls/qwen_nlpc.cpp`: synthesizable nonlinear and integer-add core.
- `fpga/qwen6/hls/build.tcl`: HLS simulation, synthesis, optional RTL co-simulation.
- `fpga/qwen6/build_overlay.tcl`: four-LPC/two-NLPC overlay, approximately 100 MHz.
- `fpga/qwen6/board_smoke.py`: physical-engine tests with synthetic operands.
- `test_qwen_fpga6.py`: local kernel, tiling, causality and failure tests.

The reused LPC RTL/driver is in
`../../TCAM_Search_FPGA/zcu106_tcam_rag_flow/xforge_zcu106/x2_zcu106`.
Connection scripts are also present in the OneDrive `tcam_rag_zcu106` folder.
The new build leaves TCAM/YOLO deployment files separate.

## Build

From FP4LLM on this Windows workstation:

```powershell
$env:QWEN_HLS_COSIM = '1'
& 'C:/Vitis_HLS/2022.2/bin/vitis_hls.bat' -f fpga/qwen6/hls/build.tcl
& 'C:/Vivado/2022.2/bin/vivado.bat' -mode batch -source fpga/qwen6/build_overlay.tcl
```

Defaults: `D:/fpga_build/qwen6_nlpc` for HLS, `D:/fpga_build/q6` for Vivado.
Overrides: `QWEN_HLS_BUILD`, `QWEN_HLS_IP`, `QWEN_BD_BUILD`, `QWEN_X2_RTL`.
`QWEN_VALIDATE_ONLY=1` stops after block-design validation. Successful
implementation and timing checks copy bit/HWH/XSA to `fpga/qwen6/overlay/`.
Never combine a bitstream and HWH from different builds.

## Prepare the real 11-token input

Use a complete BF16 checkpoint with config, tokenizer and weight safetensors:

The workstation exporter requires NumPy and Hugging Face tokenizers
(`pip install -r fpga/qwen6/requirements-export.txt`). The board uses its
existing PYNQ environment and NumPy; the hardware runner does not import PyTorch.

```text
python export_fpga_bundle.py --checkpoint /path/to/Qwen3-1.7B --prompt "Explain how ReRAM crossbars perform matrix multiplication." --out block1_11_tokens.npz --max-tokens 11
```

This is the saved 11-token prompt. Export does not silently truncate. It exports
all seven projection matrices and four learned norm gains, not the few weight
positions shown in the website. It does not execute a transformer block.

The saved metadata points to a checkpoint on another user's PC; that directory
is absent here. The local QKV safetensors contains activations, not model
weights. Use the actual checkpoint or explicitly identify a different reference
checkpoint; do not substitute synthetic weights for a real Qwen test.

A clearly identified reference bundle is now available locally at
`fpga/qwen6/reference/block1_11_tokens.npz`, using the visualization's pinned
Qwen revision. It contains all required weights and the exact saved 11 tokens.
Its `.source.json` records provenance and BF16 byte hashes. It is not claimed
to match the unavailable historical checkpoint. To reproduce the download:

```text
python fpga/qwen6/fetch_reference_bundle.py --revision 70d244cc86ccca08cf5af4e1e306ecf908b1ad5e --prompt "Explain how ReRAM crossbars perform matrix multiplication." --out fpga/qwen6/reference/block1_11_tokens.npz --max-tokens 11
```

The downloader accepts an optional per-command `--proxy` and refuses responses
that ignore its byte ranges. It does not download the full model or upload the prompt.

## Run on ZCU106/PYNQ

Use the Linux shell on COM17 for these commands. Check that the installed board
image has the required runtime:

```text
uname -a
python3 -c "import pynq, numpy; print('PYNQ and NumPy available')"
```

Copy the large deployment ZIP to the board using USB storage or an SD data
partition; there is no need to reimage the boot card. The project also has a
serial file-upload helper, but transferring the full weight bundle at 115200
baud is slow. Extract the package on the board and use `RUN_ON_BOARD.txt`.
The program runs on the board's ARM processor; UART carries terminal commands
and output, while the existing AXI/DMA interfaces connect the ARM to the FPGA.

After the build succeeds, create a self-contained board package:

```text
python fpga/qwen6/package_deploy.py --bundle fpga/qwen6/reference/block1_11_tokens.npz
```

This creates `fpga/qwen6/deploy/qwen6_block1.zip`, containing the runner,
compiler imports, original DMA driver, matched bit/HWH files, weight/input
bundle, file hashes, and `RUN_ON_BOARD.txt` with concrete commands. It refuses
to package missing build artifacts. The package is marked as not board-tested.

Copy FP4LLM source, the bundle, new bit/HWH pair, and the original X2 driver
onto the board. Run using the PYNQ/NumPy Python environment with FPGA access.

```text
python fpga/qwen6/board_smoke.py --bitstream fpga/qwen6/overlay/qwen6.bit --driver /path/to/tcam_rag_zcu106/xforge_zcu106/x2_zcu106/runtime/axi_x2_mvm_driver.py
python main.py --execution-target fpga6 --block-index 0 --prompt "Explain how ReRAM crossbars perform matrix multiplication." --fpga-bundle block1_11_tokens.npz --fpga-bitstream fpga/qwen6/overlay/qwen6.bit --fpga-driver /path/to/tcam_rag_zcu106/xforge_zcu106/x2_zcu106/runtime/axi_x2_mvm_driver.py --out-dir results/fpga_block1
```

The runner checks the HWH before programming and probes hardware ABI afterward.
It writes `fpga_summary.json` and `fpga_block1_prefill.npz` after a successful
run. The report contains completed commands per engine, stage shapes and elapsed
times. It never labels compiler estimates or emulator output as board results.

NLPC uses packed 32-bit memory words, unlike the LPC's padded DDR transport.
Its register offsets are explicit in HLS and match the generated driver header.
ABI probe is `0x514E0001`. Status detects invalid shapes, invalid commands,
INT32 overflow and nonfinite results. An NLPC timeout retains its DMA buffers;
the backend attempts to reset the overlay before releasing those allocations.
A faulted backend cannot continue; open a new backend. Physical reset/drain
behavior remains part of board validation.

## Limits

INT8 execution approximates BF16 Qwen. NLPC uses FP32 arithmetic with BF16
rounding; exp/sqrt and accumulation order can differ from PyTorch. Real-model
language accuracy and board throughput remain unmeasured. On the 11-token
reference case, host validation matched the INT8 reference exactly but had
14.70% relative L2 block-output error against a BF16-rounded NumPy reference;
see the validation record. This is not an acceptable-accuracy claim.

Repeated PS allocation, DDR transfers and command submission can dominate this
initial implementation. No speedup is claimed. HLS estimates are not routed
system utilization. The earlier compiler .m files describe a logical fabric;
this fixed physical 32×32 tiler does not claim to preserve that logical PE
allocation unchanged.
