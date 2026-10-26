# 🧠 Neuromorphic 32×32 Crossbar Simulation

## 📘 Overview
This project simulates a **neuromorphic single-processing-element (PE)** architecture based on a **32×32 analog crossbar array**.  
It performs **vector–matrix multiplications (MVMs)**, models **temporal feedback** between iterations, and evaluates **energy and latency** per computation cycle.

The design is modular, composed of:
- `Crossbar_array.py` – analog-style crossbar model  
- `FIFO_IOC.py` – input/output FIFO structures  
- `Mapper.py` – defines spatial/temporal mapping of computations  
- `Scheduler.py` – builds execution schedule (PROGRAM + COMPUTE ops)  
- `Softmax.py` – normalization module (applied to outputs)  
- `energy.py` – energy model for read/write/compute phases  
- `Latency.py` – cycle-based latency estimator  
- `user.py` – top-level simulation orchestrator  

---

## ⚙️ Architecture Summary

```
=== MAPPING SUMMARY (Single PE, 32x32 crossbar) ===
- Spatial mapping: K=32 outputs across columns of crossbar
- Temporal mapping: iterative feedback across time steps
- Each iteration reuses output FIFO as next input (temporal reuse)
---------------------------------------------------------------
```

### 🔹 Spatial Mapping
Each of the 32 crossbar columns computes one output neuron in parallel.  
This achieves **spatial parallelism** across K=32 outputs.

### 🔹 Temporal Mapping
Across time, outputs are **fed back as inputs** — enabling iterative refinement.  
This models **temporal reuse** and mimics feedback behavior found in neuromorphic systems.

---

## 🧩 Simulation Phases

### 🧱 1. PROGRAM Phase
```
[PROGRAM] Weight-Stationary Phase
          -> Load 32x32 weight matrix into crossbar array
          -> Spatial mapping: K (columns) handled by crossbar width
[PROGRAM] Crossbar weights updated.
[DONE] Crossbar programmed.
```

The crossbar’s weights are programmed once before compute begins, representing a **weight-stationary** dataflow (weights remain fixed, inputs change).

---

### ⚡ 2. TEMPORAL COMPUTE Phase
```
[TEMPORAL] Starting temporal phase: 1 initial compute + 3 iterations.
[LOAD] Loading initial input vector into Input FIFO.
```

- Initial vector (`x_manual`) is loaded into the input FIFO.  
- The system executes one initial computation (non-iterative).  
- Then performs **3 temporal feedback iterations**, each time using the previous output as the new input.

```
[INITIAL] Input FIFO → Compute FIFO → Crossbar → Output FIFO
[INITIAL] Stored 32 results in Output FIFO. Iterations start AFTER this point.
```

---

### 🔄 3. Temporal Iterations
Each iteration reuses the crossbar for new computations:
```
[TEMPORAL] Iteration 1/3
            -> Feedback: Output FIFO → Compute FIFO
            -> Compute FIFO → Crossbar → Output FIFO (stored 32 normalized result)
```

Softmax normalization is applied after each compute to stabilize the dynamic range of outputs.

---

## 🧮 Outputs and Normalization

### Initial output (non-normalized)
```
[INITIAL OUTPUT] (not counted as iteration)
Output [1x32]:
[ 9.  6.  6.  5.  6.  8.  7.  5.  8.  7.  7.  5.  6.  8.  9.  5.  9.  5.
  5.  9.  8. 10.  6.  0.  9.  5.  9.  5.  9.  5.  9.  7.]
```

### Iteration-wise normalized outputs
```
Iteration 1/3: [0. 0. 0. 0. 0. ... 1. 0. 0.]
Iteration 2/3: [0.0173 0.0471 0.0173 ... 0.0471 0.0173]
Iteration 3/3: [0.0292 0.031 0.0316 0.0314 0.0335 ... 0.0339 0.0292 0.0274]
```

### Final normalized output:
```
[OUTPUT] Final output vector (normalized):
[0.0292 0.031 0.0316 0.0314 0.0335 0.0305 0.0316 0.0296 0.032  0.0301
 0.032  0.0349 0.0271 0.0329 0.0284 0.0349 0.0284 0.0349 0.0349 0.0284
 0.0329 0.0446 0.0301 0.0191 0.0292 0.0339 0.0292 0.0339 0.0292 0.0339
 0.0292 0.0274]
```

---

## 🔋 Energy Computation

Energy is modeled per operation type:

| Phase | Description | Formula Used |
|--------|--------------|---------------|
| **Compute Energy** | Per active input × column | Based on binary & non-binary activation |
| **Read Energy** | Read 1024 crossbar cells | `E_SET` / `E_RESET` constants |
| **Program Energy** | Write (SET/RESET) 1024 cells | `E_SET` / `E_RESET` constants |

```
=== ENERGY CALCULATION PER ITERATION ===
[INITIAL] Compute energy (non-iteration): 7.036550e-11 J
[ITERATION 1] Compute energy: 3.706710e-12 J
[ITERATION 2] Compute energy: 3.830207e-12 J
[ITERATION 3] Compute energy: 3.873559e-12 J

=== ENERGY SUMMARY ===
Compute Energy (all iterations): 1.141048e-11 J
Read Energy: 9.284141e-03 J
Program Energy: 9.284141e-03 J
Total Energy: 1.856828e-02 J
---------------------------------------------------------------
```

---

## ⏱ Latency Evaluation

Latency is modeled using constants from `Latency.py`:
- Initial latency: **9 cycles**
- Iteration latency: **6 cycles/iteration**

```
=== Latency CALCULATION PER ITERATION ===
Total Latency for 3 iterations + initial computation: 27 cycles
---------------------------------------------------------------
```

---

## 📊 Final Simulation Summary

```
=== SUMMARY ===
Total temporal steps (iterations): 3
Spatial width (crossbar columns): 32

=== ITERATION HISTORY ===
Iteration 1: [binary output vector]
Iteration 2: [normalized float vector]
Iteration 3: [final stabilized softmax vector]

Simulation complete.
```

---

## 📁 File Descriptions

| File | Purpose |
|------|----------|
| `user.py` | Top-level simulation and orchestration |
| `Crossbar_array.py` | Core 32×32 MVM compute engine |
| `FIFO_IOC.py` | FIFO buffer emulating pipeline communication |
| `Mapper.py` | Defines spatial/temporal dataflow and folding |
| `Scheduler.py` | Builds PROGRAM and COMPUTE_TILE instruction list |
| `Softmax.py` | Normalization applied to iteration outputs |
| `energy.py` | Per-phase energy model (compute, read, program) |
| `Latency.py` | Latency calculator for iteration timing |

---

## 🧩 How to Run

```bash
python user.py
```

**Outputs printed:**
- Mapping Summary
- Temporal Iteration Results
- Energy Report (Compute, Read, Program)
- Latency Report
- Final Normalized Output

---

## 📈 Key Takeaways

| Concept | Description |
|----------|--------------|
| **Spatial Mapping** | Parallel computation across 32 crossbar columns |
| **Temporal Mapping** | Feedback-driven reuse of outputs as new inputs |
| **Weight Stationary** | Weights remain fixed; only inputs evolve |
| **Softmax Normalization** | Ensures stable dynamic range in temporal feedback |
| **Energy Evaluation** | Captures realistic SET/RESET and read costs |
| **Latency Modeling** | Simple cycle-based estimate for each iteration |
