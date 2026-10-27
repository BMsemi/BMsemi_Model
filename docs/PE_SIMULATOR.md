# PE simulator updated from Neuromorphic_X1

Source: BMsemi/Neuromorphic_X1, branch kanna, commit
`ab32cbdd01702ff41e9567c5c1152b2b860cb50f`, directory
`Benchmarking/Neuromorphic_complier`.

The original directory is preserved in
`projects/2_building_blocks/Neuromorphic_X1_kanna`. The runtime files under
`bm_models/vendor/neuromorphic_x1` change only sibling imports to package-relative
imports, avoiding conflicts with the other platform models.

## Canvas and API

Refresh the platform, select **core.pe_simulate** under Core, and run using
**Software simulator**. It is also callable from notebooks with:

```python
from platform_models import run_model
result = await run_model("core.pe_simulate", {
    "rows": 2, "cols": 2,
    "weights": [[1, 0], [0, 1]],
    "input": [1, 0]
})
print(result["output"])  # [1.0, 0.0]
```

API: `POST /v1/run/core.pe_simulate`, using an API key with the **core** module.
The JSON body is `{"params": {"rows": 2, "cols": 2, "input": [1, 0]}}`.

The default is a 512x512 identity matrix with an all-ones input. It executes
13,108,512 modeled clock ticks (13,107,200 PROGRAM + 288 COMPUTE + 512 STORE +
512 READ). Simulation wall time is different from modeled chip time. Smaller
arrays are useful for quick experiments. Dimensions are limited to 1-512 and
requests to 15 million modeled ticks.

Parameters:

| Name | Meaning |
| --- | --- |
| rows, cols | Array dimensions; default 512 each |
| weights | Optional rows-by-cols finite numeric matrix |
| weight_mode | identity (default), ones, or zeros; ignored when weights is supplied |
| input | Input vector with rows values; defaults to all ones |
| iterations | 0-8 feedback passes; feedback requires a square array |
| timing | Optional positive integer overrides for program_cells_per_cycle, program_cycles_per_cell, input_words_per_cycle, output_words_per_cycle, read_words_per_cycle |
| output_buffer_store_energy_J | Joules per full FIFO fill; omitted means unknown |
| output_buffer_read_energy_J | Joules per full FIFO drain; omitted means unknown |

COMPUTE stays fixed at 288 cycles per pass. Weights persist through temporal
feedback; later passes apply the upstream softmax. Output contains the initial
vector, final vector, stage_cycles, total_cycles, output_buffer_words, modeled
energy and the pinned source commit.

Energy is reported for binary weights and inputs without temporal feedback.
Signed/fractional arithmetic is supported, but the upstream energy model does
not fully support those cases, so array energy is reported as unknown. Array
READ is charged once, distinct from FIFO STORE/READ. Missing FIFO costs yield
PARTIAL MODELED TOTAL with a null modeled_total_J; configured_subtotal_J lists
only known costs. Even configured totals are uncalibrated model estimates and
exclude input buffering, host processing, leakage and other circuitry.

This is a behavioral simulator, not a ZCU106 bitstream or driver. The installed
physical board test remains `core.pe_self_test`; the new simulator does not add
FPGA execution for YOLO, TCAM, Deepgate or MNIST. Existing mapping/energy summary
operations and model handlers retain their separate implementations. Use
`core.pe_simulate` for the new upstream PE state machine and energy accounting.
