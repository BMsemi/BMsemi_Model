# FPGA prototype to ASIC architecture

## Frozen functional contract

The proven boundary is a signed INT8 matrix tile:

- up to 32 input rows, 32 output columns, and 32 vectors per launch;
- INT8 activation and weight multiplication;
- INT32 accumulation across K tiles;
- per-channel floating scale and bias outside the primitive;
- deterministic row-major weight and batch-major vector ordering.

The compiler, NumPy backend, RTL simulation, and two physical ZCU106 tests all
agree on this contract. Keep `X2Program` and the generic `x2.mvm_tile` MLIR
operation as the compatibility boundary while replacing the transport.

## Measured prototype bottleneck

The 94 YOLO Conv2D layers contain 2.665 GMAC and 2.39 MB of unique INT8
weights. The current host-driven schedule expands them to:

- 2,458 logical weight tiles;
- 99,168 independent accelerator launches;
- 45,140,912 input protocol words, each occupying a 16-byte DDR line;
- 86,248,000 INT32 output words;
- 722 MB physical input traffic and 345 MB output traffic per frame.

A cycle model fitted to the physical 3x2 and 27x16/batch-32 JTAG results
estimates 3.58 billion PL cycles, or 35.8 seconds at 100 MHz. This is a
transport/scheduling limitation, not a compiler-correctness failure.

## Next FPGA revision

Implement these blocks in order:

1. A descriptor ring in DDR and a hardware command processor. One doorbell
   should launch a complete layer or fused subgraph.
2. Persistent weight storage. Cache each tile once in URAM/BRAM and identify it
   by compiler-assigned weight ID. The ZCU106 can stage weights layer by layer;
   the ASIC should hold the full 2.39 MB model in SRAM.
3. A line-buffered im2col/activation streamer with double-buffered input and
   output SRAM. Eliminate host materialization of every im2col matrix.
4. Wider PE replication and accumulator banks. The existing 8-lane datapath is
   the correctness seed; scale lanes using timing, SRAM bandwidth, and power
   results rather than changing numerical semantics.
5. A depthwise/vector unit for the eight depthwise convolutions and common
   logistic, add, multiply, pooling, resize, transpose, and concatenate work.
6. Fixed-point per-channel requantization fused into output writeback. Freeze
   scales only after representative-dataset calibration and saturation review.

## ASIC handoff requirements

- Register a native X2 MLIR dialect/ODS definition and lower the existing
  generic operations to command descriptors.
- Replace floating scale application with verified integer multiplier/shift
  parameters and specify rounding/saturation exactly.
- Generate SRAM allocation and lifetime tables from `execution_plan.json`.
- Add ECC/parity policy, clock/reset crossing rules, scan/MBIST, performance
  counters, and deterministic fault reporting.
- Run equivalence at three levels: MLIR/NumPy, RTL, and gate-level signed INT8
  vectors. Preserve the physical JTAG cases as permanent regression vectors.
- Calibrate on the full deployment dataset. The included bus-image calibration
  demonstrates the mechanism only.

The recommended stack remains TFLite/LiteRT as the first model frontend, MLIR
as the reusable compiler core, and PyTorch export as a later training frontend.
Framework choice should not leak into the command, SRAM, or arithmetic ABI.
