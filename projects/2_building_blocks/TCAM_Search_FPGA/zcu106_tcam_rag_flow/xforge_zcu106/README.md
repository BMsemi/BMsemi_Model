# XForge ZCU106

This folder consolidates the implemented YOLO26n MLIR-centered compiler and
ZCU106 FPGA execution flow from the `FPGA_acceleration` workspace.

## Implemented Flow

```text
YOLO26n INT8 TFLite model
        |
        v
TFLite frontend
        |
        v
Python Conv2D lowering and 32x32 MVM tiling
        |
        v
program.mlir + manifest.json + constants.npz
        |
        v
Hybrid runtime scheduler
        |
        +-- 94 Conv2D operators --> local emulator or ZCU106 FPGA
        |
        +-- 8 depthwise + 311 digital operators --> CPU
        |
        v
YOLO detection tensor --> NMS --> annotated detections
```

The physical FPGA backend uses two independent signed-INT8 32x32 MVM engines.
Each engine accepts up to 32 vectors per launch, and the runtime schedules them
together as a batch of up to 64 vectors.

## Folder Layout

- `x2_zcu106/x2compiler/`: TFLite frontend, compiler, MLIR emitter, planner, hybrid runtime,
  validation tools, and tests.
- `x2_zcu106/runtime/`: NumPy and PYNQ backends plus the dual-PE AXI/DDR driver.
- `x2_zcu106/rtl/`: accelerator RTL.
- `x2_zcu106/scripts/`: Vivado, XSCT/JTAG, packaging, and validation scripts.
- `x2_zcu106/compiler_artifacts/`: compiled YOLO programs, MLIR files, constants, manifests, and
  calibration data.
- `x2_zcu106/deploy/`: validated bitstreams, XSA/HWH files, reports, board bundle, and
  physical JTAG results.
- `x2_zcu106/board/`: source board launchers.
- `model/`: YOLO26n INT8 TFLite model.
- `YOLO26n-TFLite-FPGA-Partner-Project/`: model, sample image, and local
  neuromorphic engine needed by the compiler regression.
- `docs/`: flow diagram, source project notes, and migration notes.

## MLIR Boundary

The compiler emits a parseable generic `x2.mvm_tile` program for every compiled
Conv2D layer. The current runtime reconstructs the executable `X2Program` from
`manifest.json` and `constants.npz`; it does not invoke `mlir-opt` or parse the
`program.mlir` text at runtime. Native registered X2 dialect operations and MLIR
passes remain future work.

## Local Validation

From the `xforge_zcu106` directory:

```powershell
python -m x2_zcu106.x2compiler.test_compiler
```

The existing complete local graph results are under
`x2_zcu106/deploy/full_hybrid_*`.
They record 413 executed operators, 94 accelerated Conv2D operators, and 99,168
accelerator calls for one frame.

## ZCU106 Board Execution

The self-contained board bundle is:

```text
x2_zcu106/deploy/zcu106_hybrid_board
```

On a ZCU106 PYNQ/Linux image:

```bash
cd x2_zcu106/deploy/zcu106_hybrid_board
python3 -m pip install -r requirements-board.txt
sudo -E python3 run_first_conv_tile0_tdc.py
sudo -E python3 run_on_board.py --image bus.jpg --output board_output
```

The promoted overlay is `x2_zcu106/deploy/zcu106_axi_x2_mvm.bit`, hardware version
`0x00020023`. The repository records bit-exact physical tests for the dual PEs
and the first compiled YOLO tile. A saved result proving completion of the full
94-layer offload on the physical board is not currently included.

## TCAM Status

The standalone TCAM design is not copied into this package because it is not in
the supplied MLIR/X2 flow. It currently has a separate bitstream under
`tcam_rag_zcu106`; combining TCAM and X2 requires one integrated Vivado design,
an `x2.tcam_search` compiler operation, and a corresponding runtime backend.
