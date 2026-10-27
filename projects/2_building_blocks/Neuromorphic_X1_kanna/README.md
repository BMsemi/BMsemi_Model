# Neuromorphic 512 × 512 crossbar simulator

This Python simulator runs four 5 × 5 edge filters on one processing element (PE). It models PROGRAM, COMPUTE, STORE, and output-buffer READ transactions, produces feature maps, and reports modeled energy and clock cycles. It performs feature extraction, not digit classification.

## Run

Requires Python and NumPy:

```sh
python user.py
```

`user.py` loads the first MNIST test image labeled 5 and converts its pixels to binary using `pixel > 127`. MNIST is downloaded on the first run and cached in `data/mnist.npz`. The reference label selects the image; it is not a prediction. Generated results are saved in `outputs/`.

For an identity-matrix demonstration without MNIST:

```sh
python chip_demo.py
```

The demo uses `mapping512.m` and checks its output against a NumPy reference. Programming advances one simulated clock tick at a time, so a full-array run can take time to finish.

## Physical array and filter weights

The physical array remains **512 rows × 512 columns**. Eight columns contain the four differential filters; the other 504 columns contain zero weights.

| Physical columns | Filter | Decoded response |
| --- | --- | --- |
| 0–1 | Top edge | `output[0] - output[1]` |
| 2–3 | Bottom edge | `output[2] - output[3]` |
| 4–5 | Left edge | `output[4] - output[5]` |
| 6–7 | Right edge | `output[6] - output[7]` |

`W_manual` contains 20 repeated 25-row filter blocks, occupying rows 0–499. Rows 500–511 are zero. Each patch drives only one block; all other input rows are zero. The scheduler rotates through the 20 blocks and reuses them for later patches. Activating one block at a time keeps different patch responses separate in the shared output columns.

All four filters operate on the same patch in one computation. `CrossbarArray.compute()` performs ideal vector-matrix multiplication:

```text
output[j] = sum(input[i] * weights[i, j])
response[f] = output[2*f] - output[2*f + 1]
```

The crossbar returns raw column sums. Differential subtraction is performed by `user.py` after output-buffer READ. Bottom and right filters reverse the polarity of top and left, respectively.

## Mapping and scheduling

`mapping1.m` declares the physical array separately from the workload:

```text
Hardware { Rows: 512, Columns: 512 }
Stride { X: 1, Y: 1 }
Dimensions { K: 8, C: 1, Y: 28, X: 28, R: 5, S: 5 }

Dataflow {
    SpatialMap(8,8) K;
    TemporalMap(1,1) C;
    TemporalMap(5,5) R;
    TemporalMap(5,1) X;
    TemporalMap(5,5) S;
    TemporalMap(5,1) Y;
}
```

`K=8` is the number of active physical filter columns, not the total array width. One spatial map places these columns inside a single PE. Temporal X/Y traversal visits input-window positions with SAME padding. No Merge directive is needed.

With a 28 × 28 image and stride 1, there are **784 overlapping patches**, each containing 25 values. The schedule programs the complete matrix once, then performs 784 COMPUTE/READ transactions on PE 0. Each transaction produces four decoded responses, giving four 28 × 28 feature maps.

For this supported window policy, the output dimensions are `ceil(Y / stride_y)` by `ceil(X / stride_x)`. The mapper validates full C/R/S tiles and X/Y window sizes and steps against the filter and stride. Arbitrary temporal tiling and loop-order optimization are not implemented. Legacy spatial XY mapping remains available for `mapping512.m`.

## Chip commands and buffers

- **PROGRAM (`0x01`):** transfers staged weights into the physical array. The array becomes usable after programming completes; weights persist across computations.
- **COMPUTE (`0x02`):** transfers the buffered input into the compute FIFO without charging input-transfer cycles, then performs a fixed 288-cycle computation.
- **STORE:** writes all 512 raw column results into the output FIFO. Output is marked valid after the entire vector is stored.
- **READ (`0x03`):** drains output FIFO values into a host result. Partial reads are supported; `user.py` reads all 512 words per patch, including zeros from unused columns.

Commands are rejected while busy. Computation requires programmed weights and a complete input vector. Unread outputs block the next computation or programming command.

```python
import numpy as np
from neuromorphic_x2 import NeuromorphicX2

chip = NeuromorphicX2()
chip.program_weights(np.eye(512))  # PROGRAM and wait for completion.
chip.load_inputs(np.ones(512))    # Untimed input buffering.
chip.command(chip.COMPUTE)
chip.run_until_idle()
output = chip.read_outputs()      # READ and wait for completion.
print(chip.status, chip.stage_cycles)
```

`run_cycle()` advances one simulated clock tick. `run_until_idle()` advances those ticks until completion. The multi-PE wrapper's `compute_pe()` performs COMPUTE followed by READ; `get_output()` returns its cached host result.

## Latency model

Programming is serial at **50 cycles per cell**, including cells programmed to zero. COMPUTE is hardcoded to **288 cycles per patch**, independent of how many columns are active. There is no LOAD latency stage. STORE and READ each transfer one output word per cycle by default.

For one full-array PROGRAM and 784 patches:

| Stage | Formula | Clock cycles |
| --- | --- | ---: |
| PROGRAM | `512 * 512 * 50` | 13,107,200 |
| COMPUTE | `784 * 288` | 225,792 |
| STORE | `784 * 512` | 401,408 |
| FEEDBACK | Disabled: `iterations=0` | 0 |
| READ | `784 * 512` | 401,408 |
| **Total** | **PROGRAM + COMPUTE + STORE + READ** | **14,135,808** |

These are expected counts from the configured model, not a benchmark of measured silicon. Stages do not overlap. Input staging, command submission, and host subtraction have no modeled latency. No clock frequency is assumed; to convert cycles to seconds, divide by the clock frequency in hertz.

`Latency.py` supplies the programming estimate and chip timing configuration. Its `compute_total_latency(iteration_count)` helper covers one initial computation plus feedback computations only; it excludes PROGRAM, STORE, and READ. `user.py` obtains full transaction timing from PE cycle counters.

## Energy model

The three array phases use the same component structure:

```text
E_compute_phase = E_array_compute + E_decoder_driver + E_ADC
E_program_phase = E_array_program + E_decoder_driver + E_ADC
E_read_phase    = E_array_read    + E_decoder_driver + E_ADC
```

The ADC term is included in all three phases by the chosen accounting convention. The phase methods in `energy.py` add decoder/driver and ADC energy exactly once; the original array-only methods remain available.

| Full phase method | Array-only method |
| --- | --- |
| `compute_phase_energy(input_vector, matrix)` | `compute_energy(input_vector, matrix)` |
| `program_phase_energy(matrix)` | `program_energy(matrix)` |
| `read_phase_energy(matrix)` | `read_energy(matrix)` |

For the current binary input and weight matrices, array computation energy is:

```text
E_array_compute = sum over rows where input[i] == 1:
    (number of weight-1 cells in row i) * R_SET
  + (number of weight-0 cells in row i) * R_RESET
```

The existing array PROGRAM and array READ models both use:

```text
E_array_phase = N_ones * E_SET + N_zeros * E_RESET
```

The periphery model currently uses the full physical array dimensions:

```text
E_decoder_driver = matrix.size * decoder_driver_energy_per_operation
E_ADC = matrix.shape[1] * adc_energy_per_column
```

| Constant | Current value |
| --- | ---: |
| `E_SET` | 10.7882 nJ |
| `E_RESET` | 6.994 nJ |
| `R_SET` | 284.484 fJ |
| `R_RESET` | 37.37 fJ |
| Decoder/driver energy per counted operation | 27 pJ |
| ADC energy per column | 2.67 nJ |

Internally these constants and reported totals are in joules. The 288-cycle compute latency is not an extra multiplier on these per-operation energy values.

`user.py` accumulates the phase totals as follows:

```text
E_compute_total = sum(E_compute_phase for every scheduled patch computation)
E_program_total = number_of_PROGRAM_ops * sum(E_program_phase for each PE)
E_array_read_total = sum(E_read_phase for each PE)  # Once per PE.
```

For the current one-PE schedule, this means 784 compute charges, one program charge, and one array-read estimate.

**Array-read energy and output-buffer READ are different quantities.** The array-read estimate is included once per PE; it is not multiplied by 784 output-buffer transactions or by READ clock cycles. Digital output-buffer STORE and READ energy are accounted for separately, using a cost **per full output buffer**, not per word or per cycle.

Configure these two constants near the top of `user.py`, in joules for one full 512-word output buffer:

```python
OUTPUT_BUFFER_STORE_ENERGY_J = None  # Cost of filling one full output buffer.
OUTPUT_BUFFER_READ_ENERGY_J = None   # Cost of draining one full output buffer.
```

Replace `None` with the corresponding hardware energy estimate. Convert a value in pJ to joules by multiplying by `1e-12`. No numerical buffer cost is assumed: `None` is reported as **NOT CONFIGURED**, and the total is labeled **PARTIAL MODELED TOTAL** when an incurred cost is unset. An explicit `0.0` means that the corresponding cost is intentionally zero. Configured costs must be finite and nonnegative.

`NeuromorphicX2.output_buffer_words` counts actual successful FIFO transfers in STORE, host READ, and FEEDBACK. `EnergyCalculator.output_buffer_store_energy(num_words, buffer_size)` and `output_buffer_read_energy(num_words, buffer_size)` convert those counts into energy:

```text
E_buffer_store = (stored_words / buffer_size) * store_energy_per_buffer
E_buffer_read  = (read_words / buffer_size) * read_energy_per_buffer
```

Here `read_words` includes host READ and optional FEEDBACK consumption, using the same configured FIFO read cost. Current feedback is disabled. Partial transfers scale linearly: two half-buffer reads cost one full-buffer read. No fixed per-command overhead is modeled. Changing interface words per clock cycle changes latency but not energy for the same words transferred. Reading the wrapper's cached host result with `get_output()` does not charge another FIFO read.

For the current 784-patch schedule, all 512 physical outputs are stored and read, including zero values in unused columns:

| Buffer access | Words transferred | Full-buffer equivalents | Energy |
| --- | ---: | ---: | --- |
| STORE | 401,408 | 784 | `784 * OUTPUT_BUFFER_STORE_ENERGY_J` |
| READ | 401,408 | 784 | `784 * OUTPUT_BUFFER_READ_ENERGY_J` |

Once both buffer costs are configured, the reported modeled total is:

```text
E_total = E_program_total + E_compute_total + E_array_read_total
        + E_buffer_store + E_buffer_read
```

Buffer costs describe digital FIFO access and any interface activity covered by the supplied estimate. They do not reuse array SET/RESET costs or add the array decoder/driver and ADC terms again. Even with buffer costs configured, this remains a modeled subtotal, not complete system energy; input buffering, host processing, leakage, and other unmodeled circuitry are excluded.

## Output files

All files below are generated in `outputs/`:

| File | Contents |
| --- | --- |
| `feature_map_top.txt` | Top-edge response map |
| `feature_map_bottom.txt` | Bottom-edge response map |
| `feature_map_left.txt` | Left-edge response map |
| `feature_map_right.txt` | Right-edge response map |
| `feature_map.txt` | Top-edge map retained under the original filename |
| `patch_output_reads.csv` | Patch coordinates, active row-block start, 512 raw column outputs, and four decoded responses |
| `pe_outputs_used.txt` | Last host-read output vector for each used PE |
| `pe_outputs_first.txt` | First output vector and coordinates for each used PE |
| `pe_firstcol_details.txt` | First-patch dot-product details for top-edge columns 0 and 1 |

## Modeling limits

Arithmetic uses ideal NumPy floating-point multiplication. Device noise, conductance calibration, quantization, saturation, wire resistance, drift, and programming verification are not modeled. Optional feedback uses softmax without separate normalization latency; the current application disables feedback.

Energy constants and periphery operation counts are uncalibrated assumptions. In particular, the decoder/driver charge is currently per physical cell, and array READ reuses programming SET/RESET costs. The fractional-input energy branch is not consistent with the binary branch, and signed/fractional weights accepted by the arithmetic model are not fully supported by the energy model. The current application uses binary inputs and nonnegative binary physical weights.

