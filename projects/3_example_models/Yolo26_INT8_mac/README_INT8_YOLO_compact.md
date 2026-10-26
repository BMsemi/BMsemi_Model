# INT8 x INT8 YOLO Crossbar Flow (Compact)

This README is intentionally short and execution-focused.

## System Target

- Model type: quantized YOLO-lite TFLite
- Arithmetic: INT8 activation x INT8 weight
- PE size: 32 x 32
- Cell precision: 4-bit
- Signed weights: differential banks
- ADC: 8-bit
- Accumulation: INT32

---

## Flow 1: End-to-End Pipeline

![Flow 1 End to End](https://quickchart.io/mermaid?format=png&graph=flowchart%20TD%0A%20%20%20%20A%5BLoad%20TFLite%20model%5D%20--%3E%20B%5BRead%20tensors%20and%20quant%20metadata%5D%0A%20%20%20%20B%20--%3E%20C%5BSelect%20crossbar%20layers%5D%0A%20%20%20%20C%20--%3E%20D%5BConvert%20kernel%20to%20logical%20matrix%5D%0A%20%20%20%20D%20--%3E%20E%5BTile%20matrix%20into%2032%20x%2032%5D%0A%20%20%20%20E%20--%3E%20F%5BProgram%20PE%20banks%5D%0A%20%20%20%20F%20--%3E%20G%5BStream%20activation%20patches%5D%0A%20%20%20%20G%20--%3E%20H%5BCrossbar%20plus%20ADC%20execution%5D%0A%20%20%20%20H%20--%3E%20I%5BSigned%20activation%20correction%5D%0A%20%20%20%20I%20--%3E%20J%5BINT32%20accumulation%5D%0A%20%20%20%20J%20--%3E%20K%5BBias%20add%20plus%20requantize%5D%0A%20%20%20%20K%20--%3E%20L%5BStore%20output%20to%20L2%5D%0A%20%20%20%20L%20--%3E%20M%5BNext%20layer%5D)

---

## Flow 2: Per-Layer Scheduling

![Flow 2 Per Layer](https://quickchart.io/mermaid?format=png&graph=flowchart%20LR%0A%20%20%20%20A%5BLayer%20dims%20C%20R%20S%20K%20Y%20X%5D%20--%3E%20B%5BRows%20equals%20C%20times%20R%20times%20S%5D%0A%20%20%20%20A%20--%3E%20C%5BCols%20equals%20K%5D%0A%20%20%20%20B%20--%3E%20D%5BRow%20folds%20ceil%20rows%20over%2032%5D%0A%20%20%20%20C%20--%3E%20E%5BCol%20chunks%20ceil%20cols%20over%2032%5D%0A%20%20%20%20D%20--%3E%20F%5BTiles%20per%20output%20position%5D%0A%20%20%20%20E%20--%3E%20F%0A%20%20%20%20F%20--%3E%20G%5BFor%20each%20oy%20ox%5D%0A%20%20%20%20G%20--%3E%20H%5BFor%20each%20col%20chunk%5D%0A%20%20%20%20H%20--%3E%20I%5BFor%20each%20row%20fold%5D%0A%20%20%20%20I%20--%3E%20J%5BProgram%20and%20compute%20tile%5D%0A%20%20%20%20J%20--%3E%20K%5BAccumulate%20output%20slice%5D)

---

## Flow 3: One PE Data Path

![Flow 3 PE Data Path](https://quickchart.io/mermaid?format=png&graph=flowchart%20TD%0A%20%20%20%20A%5BInput%20buffer%5D%20--%3E%20B%5BCompute%20buffer%5D%0A%20%20%20%20B%20--%3E%20C%5BCrossbar%20banks%20pos0%20pos1%20neg0%20neg1%5D%0A%20%20%20%20C%20--%3E%20D%5BRow-serial%20PWM%5D%0A%20%20%20%20D%20--%3E%20E%5BADC%5D%0A%20%20%20%20E%20--%3E%20F%5BDigital%20recombination%5D%0A%20%20%20%20F%20--%3E%20G%5BINT32%20accumulator%5D%0A%20%20%20%20G%20--%3E%20H%5BOutput%20buffer%5D)

---

## Flow 4: Software and Hardware Split

![Flow 4 SW HW Split](https://quickchart.io/mermaid?format=png&graph=flowchart%20TD%0A%20%20%20%20A%5BYOLO%20op%20list%5D%20--%3E%20B%7BOperation%20type%7D%0A%20%20%20%20B%20--%3E%7CConv2D%7C%20C%5BMap%20to%20PE%5D%0A%20%20%20%20B%20--%3E%7CDepthwiseConv2D%7C%20C%0A%20%20%20%20B%20--%3E%7CFullyConnected%7C%20C%0A%20%20%20%20B%20--%3E%7CAdd%20Mul%20Reshape%20Transpose%20Concat%20Resize%7C%20D%5BKeep%20digital%5D%0A%20%20%20%20B%20--%3E%7CDecode%20Sigmoid%20NMS%7C%20D%0A%20%20%20%20C%20--%3E%20E%5BCrossbar%20execution%20path%5D%0A%20%20%20%20D%20--%3E%20F%5BPost-processing%20path%5D)

---

## Minimal Execution Steps

1. Generate or refresh `Yolo26n_auto_full_mapping.m` with `AutoMappingOptimizer.py`.
2. Run `full_inference_executor.py` on the local `yolo26n_int8 .tflite` model.
3. Read tensors and quantization metadata from the TFLite interpreter.
4. Convert each mapped kernel to a 2D logical matrix.
5. Tile matrix into 32 x 32 blocks.
6. Convert each tile to 4-bank signed-differential storage.
7. Extract and flatten patches in matching order.
8. Run PE tiles and correct signed activation shift.
9. Accumulate INT32, add bias, requantize, write output.

---

## Tiny Code Snippets

### Kernel to matrix (Conv2D)

```python
W = np.transpose(w_q, (3, 1, 2, 0)).reshape(in_c * kh * kw, out_c)
```

### Tile count

```python
row_folds = math.ceil((C * R * S) / 32)
col_chunks = math.ceil(K / 32)
total_tiles = row_folds * col_chunks
```

### Signed activation correction

```python
acc_corr = acc_unsigned - (128 + act_zero_point) * col_sum_w
```

---

## File Mapping in This Repo

- `Mapper.py`: layer to tile structure
- `Scheduler.py`: program and compute event order
- `FIFO_IOC.py`: input and output streaming
- `Multiple_PE_core.py`: multi-PE execution
- `L2_buffer.py`: inter-layer outputs and partial sums
- `AutoMappingOptimizer.py`: auto-generates mapping files from TFLite/ONNX or `.m` specs
- `full_inference_executor.py`: full-model inference entry point using the auto-generated mapping

---

## Notes

- This README keeps only the core INT8 flow.
- Extended theory and long code examples were intentionally removed.
