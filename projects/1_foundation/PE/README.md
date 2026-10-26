# PE - processing element core

The foundation layer: the processing-element (PE) model every project builds on.
Main files are the **Yolo26_INT8_mac** copies. They are the most complete version: 64-row x 32-signed-column PE, noise,
reliability, IV non-linearity and SAR-TDC models. They import only numpy and each other.

| File | Role |
|---|---|
| `Crossbar_array.py` | Differential RRAM crossbar (`CrossbarArray`) + noise / reliability / IV / TDC configs |
| `neuromorphic_x2.py` | Single PE (`NeuromorphicX2`): program weights, load inputs, run cycle |
| `Multiple_PE_core.py` | Multi-PE core (`MultiPECore`, `MultiPENeuromorphicCore`) |
| `FIFO_IOC.py` | Input/output FIFO |
| `L2_buffer.py` | L2 buffer model |
| `Mapper.py` | Layer -> PE mapping (`SinglePEMapper`, MAESTRO mappers) |
| `Scheduler.py` | Program/compute op scheduling (`MaestroScheduler`, cluster scheduler) |
| `energy.py`, `Latency.py` | Energy and latency estimates |
| `Softmax.py`, `Sigmoid.py` | Digital activation functions |

`variants/` keeps the other two diverged copies unchanged, so nothing is lost while they are merged:
- `variants/compiler_32x32/` - from `Neuromorphic_complier` (original 32x32 PE, smaller crossbar model)
- `variants/deep_gate/` - from `Deep_gate` (its own crossbar, mapper, scheduler, `LatencyObjective.py`)

Use from Python:
```python
import sys; sys.path.insert(0, r"projects/1_foundation/PE")
from Crossbar_array import CrossbarArray
from Multiple_PE_core import MultiPENeuromorphicCore
```
