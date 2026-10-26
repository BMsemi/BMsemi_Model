# AXI X2/PE MVM Accelerator RTL

This RTL is the intended PL-side flow for the PE/X2 MAC/MVM path. It exposes
AXI-Lite control plus a 128-bit AXI memory master connected to PS DDR. The
top-level core contains its own packed-word reader and result writer; its
legacy AXI-Stream pins are tied inactive in the current integration.

## Intended PS -> PL Flow

1. PS creates physically contiguous input/output buffers.
2. PS writes `CONFIG`, `RD_ADDR`, and `WR_ADDR`.
3. PS writes `CTRL.start = 1`.
4. The custom AXI reader fetches the packed input stream:
   - `ceil(rows * cols / 4)` words of signed INT8 weights, row-major.
   - `batch * ceil(rows / 4)` words of signed INT8 vector data, batch-major.
   Each valid 32-bit input word is placed at the low lane of a 16-byte DDR beat.
5. PL computes batched signed INT8 MVMs with `N_PE` MAC lanes and derives the
   per-column TDC completion timing.
6. The custom writer packs up to four signed INT32 accumulators per 16-byte DDR
   write. PS polls `STATUS`, `RD_COUNT`, and `WR_COUNT`, then reads the output.

## Register Map

Base address is assigned by Vivado Address Editor.

| Offset | Name | Access | Bits |
| --- | --- | --- | --- |
| `0x00` | `CTRL` | R/W | bit0 start pulse, bit1 abort pulse, bit2 IRQ enable, bit8 clear sticky done/error |
| `0x04` | `STATUS` | R | bit0 idle, bit1 done sticky, bit2 error sticky, bits `[15:8]` state flags |
| `0x08` | `CONFIG` | R/W | bits `[7:0]` rows, bits `[15:8]` cols, bits `[23:16]` batch. Zero batch is treated as 1. Valid range: `1..32` |
| `0x0C` | `EXPECTED` | R | expected input stream words |
| `0x10` | `RX_WORDS` | R | input words accepted |
| `0x14` | `TX_WORDS` | R | output words emitted |
| `0x18` | `CYCLES` | R | busy cycle count |
| `0x1C` | `ERROR` | R | `0` none, `1` bad config, `2` early TLAST, `3` missing TLAST |
| `0x20` | `MAGIC` | R | `0x58324D56` (`X2MV`) |
| `0x24` | `VERSION` | R | `0x00020024` |
| `0x70` | `WR_ADDR_LO` | R/W | low 32 bits of 16-byte-aligned result-buffer physical address |
| `0x74` | `WR_ADDR_HI` | R/W | high address bits; current shell uses addresses below 4 GiB |
| `0x78` | `WR_COUNT` | R | number of valid INT32 result words accepted by the writer |
| `0x7C` | `WR_STATUS` | R | writer state, response, error, and beat count |
| `0x80` | `RD_ADDR_LO` | R/W | low 32 bits of 16-byte-aligned packed-input physical address |
| `0x84` | `RD_ADDR_HI` | R/W | high address bits; current shell uses addresses below 4 GiB |
| `0x88` | `RD_COUNT` | R | number of packed 32-bit input words delivered to the MVM core |
| `0x8C` | `RD_STATUS` | R | reader state, response, error, and beat count |
| `0x90` | `TDC_ORDER` | R | bit31 timing valid, bits `[15:8]` last-ready column, bits `[7:0]` first-ready column |
| `0x94` | `TDC_ARRIVAL_MIN` | R | earliest column arrival time in integer ps |
| `0x98` | `TDC_ARRIVAL_MAX` | R | latest column arrival time in integer ps |
| `0x9C` | `TDC_ARRIVAL_SPREAD` | R | latest minus earliest arrival time in integer ps |
| `0xA0` | `TDC_PE_COMPLETION` | R | slowest column arrival, hence PE completion time, in integer ps |
| `0xA4` | `TDC_CONFIG` | R | bit31 timing enable, bits `[15:0]` reference levels |
| `0xA8` | `TDC_TIMING_MIN` | R | configured minimum arrival time in ps |
| `0xAC` | `TDC_TIMING_MAX` | R | configured maximum arrival time in ps |

## TDC Timing Model

The PL and PS models use the same inverse magnitude-to-delay relationship:

```text
arrival = timing_min + (timing_max - timing_min) /
          (1 + abs(column_result) / timing_reference_levels)
```

Larger-magnitude columns complete first and PE completion is the slowest column
arrival. The model has no delay taps, encoder delay, or reset penalty. The
floating-point relationship is rounded to the nearest picosecond only when
reported through the integer PL registers. The AXI output remains batch-major
and column-indexed, so existing DMA consumers do not need to reorder results.
Timing registers describe the final vector in a batch. PL evaluates the two
extreme magnitudes with parallel 64-cycle bit-serial dividers; `CYCLES` includes
this metadata work.

## Stream Packing

All stream words are 32-bit little-endian byte packs.

For `rows=3`, `cols=2`, `batch=1`, weights are sent first:

```text
word0[7:0]   = w00
word0[15:8]  = w01
word0[23:16] = w10
word0[31:24] = w11
word1[7:0]   = w20
word1[15:8]  = w21
word1[31:16] = don't-care
```

Then each vector is packed into `ceil(rows / 4)` words:

```text
word2[7:0]   = a0
word2[15:8]  = a1
word2[23:16] = a2
word2[31:24] = don't-care
```

Weights are row-major: `weights[row * cols + col]`. Outputs are one signed
INT32 accumulator per logical word, batch-major. The writer packs four result
words into each 128-bit DDR line and pads the final line when needed.

## Files

- `axi_x2_mvm_accel.sv`: AXI-Lite + AXI-Stream top, MVM engine, PE MAC lanes.
- `axi_x2_mvm_accel_bd.v`: Verilog wrapper for Vivado block-design module import.
- `rram_int4_programming_accuracy_lut.sv`: PL-side deterministic INT4 RRAM
  programming-accuracy LUT using the preparation-pulse heatmap data.
- `axi_rram_int4_accuracy.sv`: AXI-Lite register wrapper for the PL RRAM
  INT4 model. The PS can start the model and read final conductance, decoded
  INT4 code, pulse count, and correct/incorrect status.
- `../scripts/synth_axi_x2_mvm.tcl`: ZCU106 out-of-context synth check.
- `../scripts/build_axi_x2_mvm_bd.tcl`: ZCU106 PS + two independent 32x32
  accelerator block-design build. PE0 is at `0xA0000000`; PE1 is at
  `0xA0010000`.
- `../scripts/rebuild_axi_x2_mvm_incremental.tcl`: cache-disabled rebuild of
  the corrected accelerator OOC run and full bitstream.
- `../runtime/axi_x2_mvm_driver.py`: PS/PYNQ MMIO + contiguous-buffer runtime.
- `../scripts/run_axi_x2_mvm_jtag_smoke.tcl`: standalone physical JTAG test
  that does not require Linux or PYNQ.

## Vivado Checks

```powershell
$env:PATH='C:\Windows\System32;C:\Windows'
cmd.exe /d /s /c '"C:\Vivado\2022.2\bin\vivado.bat" -mode batch -source "x2_zcu106\scripts\sim_axi_x2_mvm_core.tcl"'
cmd.exe /d /s /c '"C:\Vivado\2022.2\bin\vivado.bat" -mode batch -source "x2_zcu106\scripts\sim_axi_input_word_reader.tcl"'
cmd.exe /d /s /c '"C:\Vivado\2022.2\bin\vivado.bat" -mode batch -source "x2_zcu106\scripts\synth_axi_x2_mvm.tcl"'
```

The focused reader regression requires `ARSIZE=4`, 16-byte address increments,
and exactly one lane-zero word returned per 128-bit AXI response. It catches the
read-width integration bug that stalled the first physical prototype after its
first input word.

## Clean v0.2.34 implementation result

The cache-disabled rebuild of RTL version `0x00020022` completed with:

- WNS `+0.456 ns`, WHS `+0.010 ns`, and no failing timing endpoints;
- 43,815/43,815 routable nets fully routed;
- zero DRC errors (five advisory warnings);
- 44,121 CLB LUTs (19.15%), 26,682 CLB registers (5.79%), one BRAM tile
  (0.32%), and one DSP (0.06%).

The promoted bitstream and reports are in `../deploy` and
`../deploy/reports/v22`. Strict JTAG tests read back version `0x00020022` and
passed both the built-in 3x2 MVM and the generated 27x16, batch-32 compiler
tile.

To validate the full ZCU106 block design without running implementation:

```powershell
$env:AXI_X2_MVM_VALIDATE_ONLY='1'
cmd.exe /d /s /c '"C:\Vivado\2022.2\bin\vivado.bat" -mode batch -source "x2_zcu106\scripts\build_axi_x2_mvm_bd.tcl"'
```

When the RRAM INT4 PL wrapper is included in the block design, its AXI-Lite
base address is `0xA0060000` and the PS-side probe is:

```bash
sudo python3 scripts/probe_rram_int4_accuracy_pl.py \
  --target-code 10 \
  --initial-g-tenth-us 500
```
