# Microscaling driver

Run from the FP4LLM directory:

```powershell
python quantized_user.py
python quantized_user.py --report results/microscaling_demo.json
python -m unittest test_microscaling test_microscaling_crossbar -v
```

Requires NumPy. The default now uses **the local `reram_crossbar.CrossbarArray`
simulator**, with a dedicated single-pair INT5 programming/readout mode. It applies
the existing cell reliability, read-noise, and column I-V models, followed by ADC
rounding/clipping. It does not communicate with a physical board. The old missing
`PE_quantization.py` and `config.ini` are not needed for this integration.

```powershell
# Crossbar simulation with default noise/I-V settings, 4-bit cells, 2-bit DAC, 10-bit ADC
python quantized_user.py --backend crossbar --report results/microscaling_crossbar.json
# Isolate converter behavior from cell noise and I-V effects
python quantized_user.py --disable-noise --disable-iv --adc-bits 10
# Compare against the ideal mathematical backend
python quantized_user.py --backend ideal
# Change the ADC to match your hardware, even if this introduces clipping
python quantized_user.py --adc-bits 8 --report results/microscaling_adc8.json
# Optional existing crossbar model parameters
python quantized_user.py --crossbar-config configs/microscaling_crossbar.json
```

The 10-bit ADC default is a simulation choice, **not a claim about your board**:
16 rows * 15 cell levels * 3 pulse counts = 720, which fits in 10 unsigned bits.
An 8-bit ADC clips bank sums above 255. Larger blocks or DAC widths may need more
ADC range or separately digitized row groups. The ADC is always applied, even with
`--disable-iv`. No physical calibration from amperes/volts is inferred: the simulator
uses one ADC code per integer cell-level times pulse-count unit.

Default dimensions are 32 physical rows and 32 physical columns. A tile has at most
16 logical output columns because two physical columns represent each weight.
Use `--array-rows` and `--physical-columns` to change the simulated geometry. A tile
larger than the available rows/column pairs is rejected. Each scale tile receives
its own persistent array and edge tiles are zero padded; this is not a packing
optimization. Reports include allocated/programmed cells, reads, ADC clips, and
per-array reliability and I-V statistics. Storage includes padded cells. Read counts
are callback operations, not measured hardware cycles or an end-to-end latency estimate.

The optional JSON configuration has `noise`, `reliability`, and `iv` objects,
using the fields of the corresponding dataclasses in `reram_crossbar.py`.
Unknown fields are rejected. CLI disable flags override JSON settings. TDC is not
used by this ADC path. No LLM accuracy, total energy, or physical-board timing is
validated by this demo.

## Custom data

Save floating-point arrays as `.npy`, weights shaped `[input_rows, output_columns]`
and inputs shaped `[input_rows]`. This preserves the original driver's row-input
convention; transpose weights stored as PyTorch Linear `[out, in]`.

```powershell
python quantized_user.py --weights W.npy --inputs x.npy --dac-bits 2 --report results/custom_microscaling.json
python quantized_user.py --weights W.npy --inputs x.npy --input-scale-frac-bits 12 --output-frac-bits 8
```

The deterministic default example includes negative values, multiple blocks with
different ranges, and partial edge tiles. The JSON report stores integer operands,
scale codes, tensor exponent, per-cycle outputs, and scaled results.

## Representation and assumptions

- Weights: symmetric INT5 `[-15,15]`; one E4M3FN scale per 16x16 tile.
- Tensor scale: `2**ceil(log2(max(abs(W))/(15*448)))`, zero tensor uses exponent 0.
  Finite E8M0 exponent range is enforced. E4M3FN scales use nearest-even ties;
  nonzero scales underflowing E4M3FN are clamped to its smallest positive value.
- Inputs: symmetric INT6 `[-31,31]`, one unsigned 16-bit fixed-point scale per
  group of 16. INT6's -32 value is deliberately unused. Scale and integer rounding
  use nearest-even; integers saturate at their symmetric endpoints.
- Input scales are `uint16_code * 2**(-fractional_bits)`. Explicit fractional bits
  represent a calibrated binary point. Without this argument the reference chooses
  one binary point from the current whole input vector. This automatic mode is
  **not** MIRA's streaming, pre-calibrated hardware scale generation. Positive
  scales underflowing this format use one LSB; scale overflow raises an error.
- A signed weight uses two 4-bit cell magnitudes: positive=max(q,0), negative=max(-q,0).
  No extra weight slice is needed for this restricted INT5 range.
- Input signs use two unsigned read passes where needed. Five magnitude bits are
  split into DAC chunks. Each result is weighted by `2**shift` before scale application.
  This pass schedule is an explicit reference assumption, not a hardware timing claim.
- Each tile partial sum receives its weight, tensor, and input scale **before**
  reduction across differently scaled input blocks.
- Scale application and reduction use float64. Optional `--output-frac-bits` rounds
  and saturates the final result to signed INT16. This is not a bit-exact model of
  MIRA's intermediate fixed-point multipliers or per-subarray requantization.
- Partial edge tiles are supported without discarding weights. `--block-size`
  allows experiments; only the default 16 matches the described MIRA grouping.

Every run compares cycle execution to a dense multiply of the reconstructed
quantized operands. Its error against the original float operands is reported
separately; quantization is lossy and does not guarantee improved model accuracy.

## Existing PWM code

`quantized_user_legacy.py` is an unchanged copy of the uploaded file. Run it via
`python quantized_user.py --legacy` once its original `PE_quantization.py`,
`config.ini`, and dependencies are available. The original reads `config.ini`
from the current working directory. The Downloads original remains unchanged.

`src.microscaling.execute` accepts a read callback:

```python
read(positive_cells, negative_cells, unsigned_pulses) -> raw_output_vector
```

The cell matrices have shape `[tile_inputs, tile_outputs]`; pulse values are
unsigned DAC codes. A callback must return one independent differential dot product
per output column, in integer-product units, without applying scales or retaining
prior reads. It may return floating values to represent analog effects. The driver
owns sign subtraction, cycle significance, and scale application.

`src.microscaling_crossbar.MicroscalingCrossbar` implements this contract using
`CrossbarArray.program_int5_weights` and `compute_int5_pulses`. The optional
`prepare_tile(key, positive, negative)` callback programs each tile once and retains
programming faults and random state across reads. Repeated executions reuse the
arrays; changed weights require a new adapter. The driver performs sign handling
and input bit shifts only once; the crossbar returns independent raw column results.
Legacy INT8 programming/readout remains available and unchanged.

A physical-board adapter still requires a board API, calibration into integer
product units, and verification of ADC/read/reset behavior. The simulator's fixed
point scales and float64 digital rescaling are not a synthesized hardware pipeline.
