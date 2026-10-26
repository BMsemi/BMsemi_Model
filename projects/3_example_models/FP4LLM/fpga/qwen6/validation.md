# Qwen6 validation record

Date: 2026-09-23. Tools: Vivado and Vitis HLS 2022.2.
Target: ZCU106, xczu7ev-ffvc1156-2-e.

## Completed

- **17 local tests passed**, including precision-default/entry-point compatibility,
  PS quantization, real BF16 checkpoint parsing, driver validation/cache/ABI
  transport, four-LPC concurrent launch, tail tiles, FPGA-kernel integer merge,
  nonlinear kernels, grouped-query block execution with 11 synthetic tokens,
  future-token isolation, integer overflow and timeout behavior: local tests.
  See [test output](reports/python_tests.txt).
  The PyTorch-dependent legacy software suites were not run because PyTorch
  and Transformers are absent on this workstation. The entry-point regression
  checks argument forwarding with a runtime stub.
- HLS C simulation: passed.
- HLS synthesis: passed. Per NLPC estimate: **8,383 LUT, 5,122 FF, 12 DSP,
  14 BRAM18K**. These are estimates for one NLPC, not the complete overlay.
  See [HLS report](reports/nlpc_hls_synthesis.rpt).
- NLPC Verilog C/RTL co-simulation: **15 transactions passed**, including every
  opcode, shape/opcode/overflow failures, causal row offset, nonuniform softmax,
  positive/negative SiLU, two-row RoPE, and width 6144.
  See [RTL simulation report](reports/nlpc_rtl_cosim.rpt).
  The vendor floating-point simulation emitted DSP OPMODE warnings; the
  transaction comparison passed. This does not replace routed timing or board tests.
- AXI-Lite offsets matched the HLS-generated `xqwen_nlpc_hw.h`.
- Vivado block-design validation passed for four LPCs and two NLPCs.
- LPC RTL simulation passed with the actual `N_PE=4, TDC_TIMING_ENABLE=0`
  configuration: 11-token full tiles, 32-token tails, single-element and short
  batches, signed values and output backpressure. The existing TDC-enabled
  regression also passed after the shared RTL sync. See
  [LPC simulation results](reports/lpc_rtl_tests.txt).
- All four generated BD wrappers were inspected: they pass `N_PE(4)` and
  `TDC_TIMING_ENABLE(0)` into the actual RTL, not merely BD metadata.

## Full overlay implementation

Synthesis passed with no errors or critical warnings. The synthesized design
uses **152,648 LUTs (66.25%), 109,077 registers (23.67%), 15 BRAM tiles (4.81%),
and 28 DSPs (1.62%)**. These are pre-placement counts, not routed utilization.
See [synthesis utilization](reports/overlay_synthesis_utilization.rpt).

Bitstream generation, placement and routing passed. The final design uses
**149,040 LUTs (64.69%), 108,243 registers (23.49%), 15 BRAM tiles (4.81%)
and 28 DSPs (1.62%)**. See [routed utilization](reports/overlay_utilization_routed.rpt).
The full routed timing summary at **100 MHz** reports
**WNS +0.119 ns, WHS +0.009 ns**, zero setup/hold/pulse-width failing endpoints,
zero unconstrained internal endpoints, and all specified constraints met.
See [routed timing](reports/overlay_timing_routed.rpt).

Bitstream DRC has **0 errors, 103 warnings and 34 advisories**. The warnings
include DSP pipelining, two DPREG-7 dynamic-OPMODE checks in the vendor floating
add/sub IP, and unused generated-interconnect loads. The methodology report has
one ULMTCS-2 warning: 9,970 control sets exceed the 15% guideline. These warnings
are retained, not waived or represented as a warning-free build. The timing
report finds no combinational loops; NLPC RTL co-simulation passed, but physical
operation remains unverified. See [DRC](reports/overlay_drc_routed.rpt) and
[methodology](reports/overlay_methodology_routed.rpt).

The final build uses the co-simulated IP at
`D:/fpga_build/qwen6_verified/solution1/impl/ip`, and its project is
`D:/fpga_build/q6f`. Log: `D:/fpga_build/qwen6_final_vivado.log`.
Routed timing closure does not replace a board test.

The working OneDrive TCAM folder had two RTL fixes absent from the repository:
forwarding the TDC parameters through the BD wrapper, and explicitly gating TDC
signals when disabled. Those two files were synced into the repository before
the final build. The old wrapper silently failed to pass the intended TDC=0
configuration. The DMA driver files in both locations are identical. Future
Qwen builds reject wrappers that lack the TDC parameter and snapshot their RTL.

## Physical-board and real-checkpoint validation

The user confirmed that the actual connection is the **Linux serial console on
COM17 at 115200 baud**, with no Ethernet use. SSH reachability is therefore not
a prerequisite. Two current COM17 probes opened the port successfully using
8N1/no flow control (including the project's console settings), sent Enter,
and received no bytes. The port was closed after each check. A responsive
Linux console and verification of the board's PYNQ environment are pending.

Not performed. The connection profile in `tcam_rag_zcu106` specifies
`xilinx@192.168.137.99`, COM17 at 115200 baud, and optional local bind address
`192.168.137.137`. SSH was unreachable; that local bind address is not currently
configured. COM14–COM17 provided no usable console response. No board bitstream
was programmed during these checks, and no successful hardware report is claimed.

The local `results/.../QKV_PREFILL_....safetensors` contains Q/K/V activations and
a later hidden state, not checkpoint weights. Saved metadata's checkpoint paths
on another PC are absent here. The exporter requires full BF16/F16/F32 checkpoint
weights; it rejects packed FP4. Physical FPGA accuracy and throughput remain unmeasured.

## Published-checkpoint host validation

A complete block-1 input/weight bundle was downloaded from the same pinned
public Qwen revision used by the visualization:
`70d244cc86ccca08cf5af4e1e306ecf908b1ad5e`. It is explicitly a **reference checkpoint**,
not a verified match to the unavailable historical checkpoint. Only block-0
weights and the required embedding rows were downloaded.

Bundle: `reference/block1_11_tokens.npz`. Provenance and SHA256 hashes of the
BF16 ranges: `reference/block1_11_tokens.source.json`. Both are generated local
artifacts, excluded from Git. The 11 token IDs match the saved prompt exactly.

Host execution used actual 2048/6144-wide Qwen weights, the C++ NLPC kernel and
LPC NumPy test doubles. Output shape: 11 × 2048, all finite. The output matched
the un-tiled INT8 NumPy reference exactly for this case. It differed from a
BF16-rounded NumPy reference by **0.1470 relative L2** (14.70%), maximum absolute
error **0.53125**. This measures block-output quantization error, not language
accuracy or a physical FPGA measurement. No claim of BF16-equivalent accuracy
is made; calibration/quantization quality needs further evaluation.

See [the host report](reports/reference_host_validation.json). Reproduce with:

```text
python -B fpga/qwen6/validate_reference_host.py --bundle fpga/qwen6/reference/block1_11_tokens.npz --report fpga/qwen6/reports/reference_host_validation.json
```

## Reproduce

Local tests:

```text
python -B -m unittest test_ps_int8_quantization test_qwen_fpga6 -v
```

The local test harness compiles the actual NLPC C++ source into a temporary
library. Its LPCs are explicit CPU test doubles. It is **not** FPGA execution.
HLS co-simulation separately compares generated RTL with the C kernel.

Build, bundle export and board commands are in
[the integration guide](../../docs/fpga_block1_prefill_six_engines.md).
