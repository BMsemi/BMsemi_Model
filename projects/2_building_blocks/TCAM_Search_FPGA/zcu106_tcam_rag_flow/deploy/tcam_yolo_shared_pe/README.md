# ZCU106 hybrid YOLO package

This package contains the physically validated X2 bitstream, its matching HWH
metadata, the YOLO26n TFLite model, all 94 statically calibrated Conv2D
programs, and the strict CPU/X2 hybrid runtime.

On a ZCU106 PYNQ/Linux image:

```bash
cd zcu106_hybrid_board
python3 -m pip install -r requirements-board.txt
sudo -E python3 run_first_conv_tile0_tdc.py
sudo -E python3 run_on_board.py --image bus.jpg --output board_output
```

`run_first_conv_tile0_tdc.py` sends the validated YOLO26n first-Conv2D
`K=27`, `N=16` weight tile to both PL PEs through coherent PYNQ buffers. It
checks both INT32 outputs and writes the two live
`tdc_pe_completion_time_ps` values to
`board_output/first_conv_tile0_tdc.json`. This is one complete MVM weight tile
for one spatial vector on each PE; the complete first Conv2D layer contains
102,400 spatial vectors.

The driver rejects an old or mismatched overlay and requires the dual-PE
accelerator version `0x00020024` or newer. The PYNQ backend schedules up to 64
vectors per call and starts both independent 32x32 PL arrays before waiting.
`run_on_board.py` does not use LiteRT operator outputs: LiteRT parses the
FlatBuffer and exposes constants only. All 413 graph operators are executed by
the packaged runtime.

The current RTL is a correctness prototype. A full frame requires 99,168
host/FPGA launches and retransmits weight tiles for every spatial batch, so it
is intentionally not a realtime implementation. See
`ASIC_MIGRATION.md` in the source tree for the required streaming architecture.
