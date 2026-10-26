# X2 Reference Port for ZCU106

This folder uses the `happy_family-main` reference code as the starting point
for a real FPGA path on the ZCU106.

The reference archive provides:

- SystemVerilog X2 AXI-Stream accelerator RTL in
  `reference_code/happy_family-main/hardware/hdl/AXIS_X2_SIMD/srcs`
- Python PYNQ wrappers for MMIO/DMA command packing in
  `reference_code/happy_family-main/main/device`
- INT8 bit-plane compiler/runtime ideas in
  `reference_code/happy_family-main/main/scheduler`

Important mismatch:

- The prebuilt reference `.xsa` files target `tul.com.tw:pynq-z2:1.0`
  / `xc7z020clg400-1`.
- This workspace targets ZCU106 / `xczu7ev-ffvc1156-2-e`.

So the PYNQ-Z2 bitstreams cannot be programmed onto the ZCU106. The portable
pieces are the RTL, command protocol, and bit-sliced runtime strategy.

## Current Status

Implemented and validated on 2026-07-27:

- MLIR-centered TFLite compiler prototype in `x2_zcu106/x2compiler`.
- Two independent signed INT8, batch-32, 32x32 MVM PL engines, each with a
  custom 128-bit AXI DDR reader/writer and AXI-Lite control. The PS runtime
  schedules them together as a batch-64 backend.
- NumPy and PYNQ execution backends sharing the same compiled artifact.
- All 94 Conv2D operators from the real YOLO26n TFLite model compiled.
- The complete 413-operator graph executes without reference-tensor fallback:
  94 Conv2D on X2 semantics, 8 depthwise on CPU, and 311 digital CPU ops.
- Static activation calibration, a 27.9 MB activation arena, class-aware NMS,
  annotated detections, execution tracing, and board packaging are complete.
- XSim regression passed through the maximum 32x32/batch-32 shape with output
  backpressure.
- Direct physical ZCU106/JTAG MVM passed without Linux or PYNQ, including a
  concurrent launch of both full 32x32 PE arrays.
- The clean v0x00020023 dual-PE implementation meets timing at 100 MHz (WNS
  `+1.287 ns`, WHS `+0.010 ns`) with zero routing errors and zero DRC errors.
  It uses 84,805 LUTs (36.81%), 47,643 registers (10.34%), and 2 DSPs.

The original reference-port milestone completed on 2026-07-09:

- Built a ZCU106 block design containing:
  - Zynq UltraScale+ PS
  - AXI DMA
  - `AXIS_X2_SIMD_Wrapper` on the AXI-Stream path
  - PS DDR access through the HPC port
- Generated a real ZCU106 bitstream and XSA:
  - `C:\fpga_build\yolo26n_x2_dma_bd\zcu106_x2_dma.runs\impl_1\zcu106_x2_dma_wrapper.bit`
  - `C:\fpga_build\yolo26n_x2_dma_bd\zcu106_x2_dma.xsa`
- Programmed the ZCU106 PL over JTAG with the generated bitstream.
- Copied deploy artifacts into this workspace:
  - `x2_zcu106/deploy/zcu106_x2_dma.bit`
  - `x2_zcu106/deploy/zcu106_x2_dma.hwh`
  - `x2_zcu106/deploy/zcu106_x2_dma.xsa`
  - `x2_zcu106/deploy/smoke_x2_dma.py`

Implementation results:

- CLB LUTs: 29,034 / 230,400 = 12.60%
- CLB registers: 23,328 / 460,800 = 5.06%
- Block RAM tiles: 3.5 / 312 = 1.12%
- DSPs: 0 / 1,728 = 0.00%
- Timing met at the PS PL clock:
  - WNS: 4.186 ns
  - TNS: 0.000 ns
  - WHS: 0.010 ns

## Rebuild

```powershell
$env:PATH='C:\Windows\System32;C:\Windows'
cmd.exe /d /s /c '""C:\Vivado\2022.2\bin\vivado.bat" -mode batch -source "x2_zcu106\scripts\build_x2_dma_bd.tcl" > "x2_zcu106\build_x2_dma_bd_stdout.log" 2>&1"'
```

## Program PL

```powershell
cmd.exe /d /s /c '""C:\Vitis\2022.2\bin\xsct.bat" "pl_counter\scripts\program_pl.tcl" "C:\fpga_build\yolo26n_x2_dma_bd\zcu106_x2_dma.runs\impl_1\zcu106_x2_dma_wrapper.bit""'
```

## PYNQ Smoke Test

Copy `x2_zcu106/deploy` to a PYNQ/Linux image on the ZCU106, then run:

```bash
cd deploy
python3 smoke_x2_dma.py
```

The smoke test uses the generated hardware's `HEADER_TAG = 0x0DED0000`.
The reference Python config currently uses a different tag, so update the
runtime config before using the reference app with this bitstream.

## YOLO Status

The first TFLite-to-X2 compiler is implemented in `x2_zcu106/x2compiler`.
It imports the actual YOLO26n TFLite model, lowers Conv2D to tiled signed INT8
MVM operations, emits a versioned artifact plus generic MLIR, and runs the same
artifact on NumPy or the ZCU106/PYNQ backend.

Current validation:

- all 94 `CONV_2D` operators in the supplied model compile with zero import
  errors;
- arbitrary M/K/N schedules crossing 32x32 tile boundaries are bit-exact
  against NumPy;
- the first real layer executes as 3,200 batch-32 accelerator calls and
  44,236,800 logical MACs;
- comparison with LiteRT on deterministic random input gives MAE `0.04695`
  against reference absolute mean `7.10355`; this is float-to-INT8 bridge error,
  while the integer accumulator path is bit-exact;
- a physical ZCU106 JTAG transaction passed with result `[40, 46]`, three AXI
  input words, two AXI output words, and no error;
- the first compiled YOLO tile (`K=27`, `N=16`, batch 32) passed physically
  with all 512 INT32 outputs exactly matching NumPy, 332 packed input words
  consumed, 512 outputs written, 24,436 cycles, and no accelerator error;
- both earlier strict physical tests report RTL version `0x00020022`; their records are
  `deploy/jtag_mvm_v22_result.txt` and
  `deploy/jtag_compiled_tile_v22_result.txt`.
- the promoted v0x00020023 bitstream also passed a concurrent PE0+PE1 JTAG
  transaction; its record is `deploy/dual_pe_jtag_result_v23.txt`.
- the full strict NumPy/X2 graph detects the same five objects as LiteRT with
  precision/recall 1.0 at IoU 0.5 and mean matched-box IoU 0.966;
- the board bundle is `deploy/zcu106_hybrid_board.zip`, containing the bit/HWH
  pair, calibrated compiler programs, model, and PYNQ launch script.

The promoted dual-PE bitstream is `deploy/zcu106_axi_x2_mvm.bit` (SHA-256
`f651588a357faf547fc739cde41ee76f36f745597c1c66ddc9fd457b0a183815`).
The matching timing, utilization, routing, and DRC reports are under
`deploy/reports/v23`.

See `x2_zcu106/x2compiler/README.md` for compile and execution commands.

A client-facing encrypted compiler SDK is under `../x2_sdk`. It provides an
installable wheel, public Python/CLI API, AES-256-GCM encrypted compiler core,
Ed25519-signed customer licenses, and per-customer delivery packaging without
shipping plaintext compiler source.

The first scheduler/runtime layer now lives in `x2_zcu106/runtime`:

- `x2_runtime.py` packs the ZCU106 X2 DMA protocol.
- `NumpyBinaryX2Backend` verifies the scheduler locally.
- `PynqDmaBinaryX2Backend` targets the generated `zcu106_x2_dma.bit`.
- `Int8BitPlaneScheduler` maps INT8 matmul, Conv2D, and DepthwiseConv2D onto
  32x32 binary X2 tiles.

This is not a realtime full YOLO accelerator yet. End-to-end orchestration is
implemented, but the current PL protocol requires 99,168 launches per frame
and repeatedly sends the same 2.39 MB of unique weights, creating 722 MB of
physical input traffic. The measured-cycle model estimates 35.8 seconds for
the Conv2D PL work at 100 MHz, before host and CPU overhead. Realtime operation
requires persistent weights, streaming activations, a command queue, and a
depthwise/vector unit as described in `ASIC_MIGRATION.md`.

PYNQ and SSH are not strictly required. They are the fastest route for bring-up
because they provide overlay loading, DMA buffers, and a simple file/control
channel. Alternatives are a Vitis bare-metal A53 app launched by JTAG, a
PetaLinux C/C++ daemon controlled over Ethernet, or another explicit host link.
JTAG alone is useful for programming/debugging the PL but is not a realtime YOLO
data path.
