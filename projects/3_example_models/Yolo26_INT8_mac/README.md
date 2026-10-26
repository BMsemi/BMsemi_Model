# YOLO26n INT8 Crossbar Inference

Diagram-first notes for the current `yolo26n_int8.tflite` hardware flow.

## Current Status

```text
Default command: python user.py
Model source: LiteRT export INT8 model (`yolo26n_int8.tflite`)
Conv/Depthwise on crossbar: 102 / 102
Digital/non-conv ops: 276
Reference fallback ops: 35
Latest bus confidence: 0.834
Latest runtime: about 24 seconds
Layer-0 nonlinear effect scenarios: 14
Latest effect rerun: 2026-07-03; clean PE matches digital reference
All-layer nonlinear audit: 102 layers on 2026-07-06
```

## Run

| Goal | Command |
|---|---|
| Full crossbar conv path | `python user.py` |
| Explicit full crossbar | `python user.py --hardware all` |
| Reference-only check | `python user.py --hardware none` |
| One hardware layer | `python user.py --hardware layers --crossbar-layers Conv2D_96` |
| Slow PE debug path | `$env:YOLO_CROSSBAR_SLOW="1"; python user.py --hardware layers --crossbar-layers Conv2D_96` |

## Top-Level Flow

```mermaid
flowchart LR
    A["bus.jpg"] --> B["Plain resize 640x640"]
    B --> C["yolo26n_int8.tflite"]
    C --> D["TFLite op list"]
    D --> E{"Op type"}
    E -->|"Conv2D / DepthwiseConv2D"| F["Crossbar path"]
    E -->|"Add / Mul / Logistic / Concat / Reshape"| G["Digital path"]
    F --> H["YOLO output tensor 1x84x8400"]
    G --> H
    H --> I["Decode + NMS"]
    I --> J["detected_bus.jpg"]
```

## Hardware Split

```mermaid
flowchart TD
    A["413 graph ops"] --> B["102 Conv/Depthwise ops"]
    A --> C["276 digital ops"]
    A --> D["35 reference fallback ops"]
    B --> E["Crossbar MVM"]
    C --> F["Software digital math"]
    D --> F
    E --> G["Final tensor"]
    F --> G
```

## Neuromorphic Hardware Architecture

The hardware path is split into a compile-time mapping flow and a runtime PE execution flow. The parser extracts convolution geometry from the model, the mapping file describes how K/C/R/S/Y/X are tiled, and the scheduler turns that mapping into PE program and compute commands.

### Model to Mapping Flow

```mermaid
flowchart LR
    A["LiteRT export INT8 model<br/>yolo26n_int8.tflite"]
    A --> A1["ONNX view/export derived from<br/>the same INT8 graph"]
    A1 --> B["ONNX_graph_parser.py"]
    A --> B
    B --> C["Conv / Depthwise specs<br/>K, C, R, S, Y, X, stride"]
    C --> D["MAESTRO-style model spec<br/>Yolo26n_full_model_spec.m"]
    D --> E["AutoMappingOptimizer.py"]
    E --> F["Layer mapping file<br/>Yolo26n_auto_full_mapping.m"]
    F --> G["Mapping report + trace CSV"]
    F --> H["Mapper.py<br/>parse Layer, Dimensions, Dataflow"]
    H --> I["Scheduler.py<br/>PROGRAM_CLUSTER / COMPUTE_CLUSTER_C ops"]
```

The ONNX path in this project is based on the LiteRT-exported INT8 model. In other words, the ONNX/parser flow is used to recover layer geometry and mapping metadata for the quantized CNN graph; the hardware execution still targets signed INT8 weights, unsigned INT8 activation patches, and INT32 accumulation on the crossbar path.

### Hardware Block Diagram

```mermaid
flowchart TB
    A["Input image / previous layer tensor"] --> B["Preprocess + quantize"]
    B --> E["Input buffer<br/>64 unsigned INT8 rows per PE"]

    subgraph CL["PE cluster"]
        direction TB
        L2["Cluster L2 output buffer<br/>partial sums and output channels"]

        subgraph PE0["NeuromorphicX2 PE"]
            direction TB
            IN0["Input FIFO<br/>activation window"]
            L10["L1 / compute buffer<br/>64-row working vector"]
            XB0["CrossbarArray 64x64<br/>pos0,pos1,neg0,neg1 banks"]
            ADC0["ADC / IV-ADC / SAR-TDC sensing"]
            OUT0["Output FIFO<br/>32 logical INT32 outputs"]
            IN0 --> L10 --> XB0 --> ADC0 --> OUT0
        end

        subgraph PE1["NeuromorphicX2 PE"]
            direction TB
            IN1["Input FIFO"]
            L11["L1 / compute buffer"]
            XB1["CrossbarArray 64x64"]
            ADC1["sensing"]
            OUT1["Output FIFO"]
            IN1 --> L11 --> XB1 --> ADC1 --> OUT1
        end

        PE0 --> L2
        PE1 --> L2
    end

    E --> PE0
    E --> PE1
    L2 --> F["Bias / scale / requantize"]
    F --> G["Next layer tensor or YOLO output"]
```

In the hardware view, L2 is the cluster-level output/feature buffer, while each PE has local input and L1/compute buffering around the crossbar. In the current Python simulator, the L1 working vector is represented by the PE FIFO/compute path, and the positive/negative 4-bit storage banks are held inside `CrossbarArray`.

### Mapping File on Crossbar

```mermaid
sequenceDiagram
    participant M as Mapping file
    participant Map as Mapper.py
    participant Sch as Scheduler.py
    participant PE as MultiPENeuromorphicCore
    participant X as CrossbarArray
    participant L2 as L2 accum buffer

    M->>Map: Read Layer block
    Map->>Map: Extract Dimensions K,C,R,S,Y,X and stride
    Map->>Map: Parse SpatialMap / TemporalMap / Cluster
    Map->>Sch: Emit layer dims + PE tiling plan
    Sch->>PE: PROGRAM_CLUSTER for each K chunk and C group
    PE->>X: program_weights_one(weight_tile)
    X->>X: split signed INT8 weights into pos/neg 4-bit banks
    Sch->>PE: COMPUTE_CLUSTER_C for each output Y,X,K,C tile
    PE->>PE: load 64-row activation patch into input/L1 buffers
    PE->>X: compute(vec64)
    X->>X: row-serial INT8 MVM + sensing model
    X-->>PE: 32 logical INT32 outputs
    PE-->>L2: accumulate partial output channels
    L2-->>PE: complete channel tensor for scale/bias
```

The mapping file controls the main hardware decisions:

| Mapping concept | Hardware meaning |
|---|---|
| `SpatialMap(... ) K` | Splits output channels across PE clusters or K chunks. |
| `Cluster(..., P)` | Groups PEs so several channel/input slices can work in parallel. |
| `SpatialMap(... ) C` after `Cluster` | Assigns input-channel groups to PEs inside a cluster. |
| `TemporalMap(... ) R/S/Y/X` | Defines the patch/window traversal order for repeated compute calls. |
| `C*R*S > 64` | Splits the input patch across multiple PE groups or vectorized waves. |

## Crossbar Layer Path

```mermaid
flowchart LR
    A["TFLite weight tensor"] --> B["K,C,R,S layout"]
    B --> C["Logical matrix: C*R*S rows, K cols"]
    C --> D["64-row PE groups"]
    D --> E["INT8 activation patches"]
    E --> F["MVM: activation @ weights"]
    F --> G["INT32 accum"]
    G --> H["Scale + bias"]
    H --> I["Layer output tensor"]
```

## One PE Data Path

```mermaid
flowchart TD
    A["64 input rows"] --> B["Unsigned activation"]
    C["Signed INT8 weights"] --> D["4-bit banks"]
    D --> E["pos0 / pos1 / neg0 / neg1"]
    B --> F["Row-serial equivalent MVM"]
    E --> F
    F --> G{"Optional sensing model"}
    G -->|"cell ADC"| H["INT32 partial sums"]
    G -->|"IV/ADC clamp"| H
    G -->|"SAR-TDC"| H
    H --> I["Output channels"]
```

## Nonlinear Hardware Effects

`test_layer0_pe_effects.py` is the focused nonlinearity, reliability, and noise harness for the layer-0 CNN PE. It loads the layer-0 weights and activation patches, runs a clean `CrossbarArray` baseline, then reruns the same workload with one hardware effect enabled at a time. Each scenario is compared against the clean PE output with MAE, RMSE, max absolute delta, changed-output fraction, effective-weight changed fraction, ECC latency cycles, retention time, endurance failed-cell fraction, IV/ADC saturation counts, and SAR-TDC metadata. Results are written to `layer0_pe_effect_test_results/summary.csv` and `summary.json`.

The PE uses signed INT8 weights and unsigned INT8 activations. Each signed weight is decomposed into positive and negative magnitudes, then each magnitude is split into two 4-bit banks:

```text
w_pos = max(w, 0) = pos0 + 16 * pos1
w_neg = max(-w, 0) = neg0 + 16 * neg1
w = w_pos - w_neg
```

Reliability effects that change stored cell levels are applied during `program_weights()`. Readout noise, IV/ADC clamp, and SAR-TDC sensing effects are applied during `compute()`.

The current CNN hardware-effect surface includes:

```text
reliability: stuck cells, raw BER, ECC estimate, retention, relaxation, conductance spread, endurance
noise: thermal, shot, flicker / 1/f, RTN, readout RMS
sensing: cell ADC, IV/ADC saturation, SAR-TDC differential conversion
```

### Nonlinearity Stress Rerun

Rerun command: `python test_layer0_pe_effects.py` on 2026-07-03. The table reports the layer-0 PE measurement against the clean crossbar baseline as `MAE / max |delta| / changed outputs`.

| Nonlinearity model | Built for | Stress used | Measured in latest rerun |
|---|---|---|---|
| Clean PE execution | Proves the row-serial crossbar PE matches the INT32 digital reference before stressing it. | No injected hardware effect. | `0.000 / 0 / 0.0%`; clean PE matches digital reference. |
| Stuck-at LRS/HRS faults | Permanent ReRAM cell defects after programming. | `2%` stuck-LRS plus `2%` stuck-HRS cells. | `16960.061 / 54720 / 74.5%`; effective weights changed `8.8%`. |
| Raw BER, no ECC | Uncorrected bit errors in 4-bit stored cell levels. | `10000 ppm` raw BER, ECC off. | `7514.537 / 29298 / 90.3%`; effective weights changed `13.4%`. |
| Raw BER with ECC | Residual error after SEC-style correction and its latency cost. | `10000 ppm` raw BER, 38-bit codeword, 32-bit data, 1 encode and 1 decode cycle. | `1896.586 / 15048 / 40.2%`; ECC latency `19264` cycles. |
| Retention drift | Weight-stationary ReRAM drift during mapped YOLO layer residency. | State power-law drift at `125 C`, retention time `0.006144 s`. | `0.000 / 0 / 0.0%`; no changed outputs at this mapped residency. |
| Relaxation drift | Fast post-program settling before inference. | `7200 s` relaxation, `tau=2000 s`, amplitude `0.16 ln(R)`. | `3289.412 / 19608 / 50.0%`; effective weights changed `20.8%`. |
| State/conductance variation | Programming spread and device-to-device conductance variation. | Gaussian log-resistance sigma `0.12`. | `3249.668 / 10716 / 96.9%`; effective weights changed `35.2%`. |
| Endurance limit failure modes | End-of-life cells with stuck and multilevel-loss modes. | `100000` writes at a `100000` cycle limit; `2%` LRS, `2%` HRS, `12%` multilevel loss. | `21647.643 / 69312 / 99.8%`; failed cells `15.4%`. |
| Thermal and shot noise | Current noise under high-temperature, high-bandwidth read. | `400 K`, Fano `5`, bandwidth `5e10 Hz`. | `5722.821 / 31573 / 100.0%`. |
| Flicker / 1/f noise | Low-frequency read noise. | `50 kHz` bandwidth, `0.1 Hz` fmin, `A_LRS=2e-4`, `A_HRS=8e-4`. | `706.691 / 3570 / 99.7%`. |
| RTN noise | Random telegraph noise from state-dependent current jumps. | RTN amplitude fraction `0.08` LRS and `0.18` HRS. | `654.264 / 2686 / 99.7%`. |
| Readout noise | Fixed RMS sensing noise floor. | `4e-7 A rms` readout noise. | `21446.835 / 79831 / 100.0%`. |
| ADC clipping / quantization | IV/ADC nonlinear transfer and column saturation. | Hard clamp with ADC full scale `255` levels. | `1775.969 / 6896 / 50.0%`; `892` upper saturations across `16` columns. |
| SAR-TDC 8b | Differential branch sensing quantized by SAR-TDC. | `8b` SAR-FP output, gain shift `0`, effective coarse/fine bits `4/3`. | `226.828 / 1676 / 50.0%`. |

### Full 102-Layer Aggregate Audit

Rerun command: `python test_all_layers_nonlinearity_effects.py` on 2026-07-06. This audit covers all `102` crossbar-eligible Conv/Depthwise layers using the LiteRT reference activation tensors with the same INT8 bridge used by the crossbar executor. Results are written to `all_layer_nonlinearity_effect_test_results/summary.md`, `summary.json`, and `layer_summary.csv`.

| Effect | Enabled hardware model | Main measured error / stress |
|---|---|---|
| Reliability | 30-day retention drift, 0.02 log-normal state spread, 10 ppm raw BER with ECC | 30,210,165,440 programmed cell reads audited; 29,965,565,094 cell-level reads changed after the retention/state model; 1 raw BER flips after ECC; 808,391,600 effective INT8 weight read entries changed; mean absolute integer branch error 2.687606; max 2,286 |
| Noise | Thermal 4 k T / R, shot 2 q I * fano, flicker/1/f, RTN, readout RMS, 5 kHz bandwidth integration | Mean equivalent branch read-noise sigma 0.026175 conductance levels; max 0.029965 levels |
| IV / ADC | hard_clamp, 4 row groups, full-scale 2,250 product levels before digitization | 29,941,977 upper saturation events; 0 lower clips; 10,293,222 clamped tile-columns; 262,320 saturated tiles; max branch level 14,599 |
| TDC | 9-bit signed-payload SAR-FP TDC, gain_shift=0 | 68,276,800 output values converted; 614,491,200 wire bits; 0 extra SAR range clips after IV limiting; mean absolute integer readout error 23.864615; max 128 |

### Retention Assumptions

YOLO26 uses stationary weights, so the default retention strategy is `weight_stationary`. The layer-0 test estimates retention time from `Yolo26n_auto_layer0_mapping_report.csv` and the configured cycle time:

```text
retention_time_s = mapping_residency_cycles * LAYER0_MAPPING_CYCLE_TIME_S
```

By default `LAYER0_MAPPING_CYCLE_TIME_S=1e-9`, so layer-0 weight-stationary residency is currently about `0.006144 s`. That is intentionally much smaller than a fixed long-term storage value such as 180 days. For long-storage stress tests, override the residency explicitly:

```powershell
$env:LAYER0_WEIGHT_STATIONARY_RETENTION_S = "2592000"
```

### Endurance Assumptions

The current endurance scenario does not count every cell's write history over repeated simulator calls. Instead, it assumes the array has already experienced `endurance_write_cycles`. When that value reaches `endurance_cycles`, the model injects end-of-life failure modes across the programmed cell population:

```text
stuck at LRS
stuck at HRS
analog / multilevel accuracy loss
```

The summary reports how many programmed cells entered those failure modes. A full lifecycle model would add persistent per-bank write counters for `pos0`, `pos1`, `neg0`, and `neg1`, increment them on each program event, and make each cell fail when its own counter reaches the endurance limit.

## Fast vs Slow Hardware Mode

```mermaid
flowchart LR
    A["Crossbar request"] --> B{"YOLO_CROSSBAR_SLOW?"}
    B -->|"not set"| C["Vectorized NumPy MVM"]
    B -->|"1 / true / yes"| D["Per-PE row-serial debug"]
    C --> E["Fast functional hardware math"]
    D --> F["Slow detailed PE calls"]
```

## Vectorized MVM Shape

```mermaid
flowchart LR
    A["Feature map"] --> B["im2col patches"]
    B --> C["patch matrix: N x C*R*S"]
    D["crossbar weights: C*R*S x K"] --> E["matrix multiply"]
    C --> E
    E --> F["output: N x K"]
    F --> G["reshape to H x W x K"]
```

## Depthwise Fix

```mermaid
flowchart TD
    A["DepthwiseConv2D spec"] --> B["Spec K=channels, C=1"]
    B --> C["Use real weight channels"]
    C --> D["Logical C = input channels"]
    D --> E["Diagonal depthwise matrix"]
    E --> F["Correct per-channel crossbar output"]
```

## PE Capacity Handling

```mermaid
flowchart LR
    A["C*R*S > 64"] --> B["Split channels into PE groups"]
    B --> C{"Groups exceed 512 PEs?"}
    C -->|"fast mode"| D["Time-multiplex/vectorized waves"]
    C -->|"slow mode"| E["Reject or run targeted layer"]
    D --> F["All 102 layers can run"]
```

## Detection Output

```mermaid
flowchart TD
    A["Crossbar + digital output"] --> B["Tensor 1x84x8400"]
    B --> C["4 box rows"]
    B --> D["80 class-score rows"]
    C --> E["Box scale to original image"]
    D --> F["Class argmax + confidence"]
    E --> G["NMS"]
    F --> G
    G --> H["detections.csv / detections.json"]
    G --> I["detected_bus.jpg"]
```

## Latest Result

```text
person 0.891  [671,395,809,879]
person 0.841  [48,394,235,905]
person 0.839  [224,399,346,860]
bus    0.834  [1,225,806,746]
person 0.666  [1,556,70,873]
```

## Key Files

```mermaid
flowchart TD
    A["user.py"] --> B["Main CLI + postprocess"]
    C["full_inference_executor.py"] --> D["Model input + full executor"]
    E["hybrid_inference_executor.py"] --> F["Crossbar/digital execution"]
    G["Yolo26n_full_model_spec.m"] --> H["Exact TFLite layer spec"]
    I["ONNX_graph_parser.py"] --> G
    J["Crossbar_array.py"] --> K["INT8 row-serial PE math"]
    L["full_inference_test/"] --> M["Outputs and traces"]
```

## Output Files

| File | Meaning |
|---|---|
| `full_inference_test/detected_bus.jpg` | Annotated image |
| `full_inference_test/detections.csv` | Compact detections |
| `full_inference_test/detections.json` | Full detections |
| `full_inference_test/execution_trace.csv` | Backend per op |
| `full_inference_test/hardware_summary.json` | Crossbar/digital counts |
| `full_inference_test/final_output_float.npy` | YOLO output tensor |

## Architecture Boundary

```mermaid
flowchart LR
    A["Crossbar"] --> B["Conv2D"]
    A --> C["DepthwiseConv2D"]
    D["Digital"] --> E["Logistic"]
    D --> F["Add / Mul"]
    D --> G["Concat / Reshape / Transpose"]
    D --> H["Decode / NMS"]
```

```text
Crossbar scope: MAC-heavy Conv/Depthwise layers.
Digital scope: graph glue, activations, tensor movement, YOLO decode.
```

