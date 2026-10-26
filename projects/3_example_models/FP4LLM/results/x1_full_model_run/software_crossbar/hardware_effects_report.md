# Software Crossbar Hardware Effects Report

This audit uses the same prompt/decode tensors and W7/A7 quantized weights as the BF16 value run.

| Effect | What was enabled | What happened |
|---|---|---|
| Reliability | 30-day retention drift, state spread, raw BER with ECC | 0 cell writes audited; 0 cell levels changed; 0 INT8 weights changed |
| Noise | Thermal, shot, flicker, RTN, readout model at 5 kHz bandwidth | mean branch sigma 0.000000 levels; max branch sigma 0.000000 levels |
| IV / ADC | hard-clamp branch full-scale before digitization | 0 saturation events; 0 clamped tile-columns; 0 saturated tiles |
| TDC | SAR-FP TDC conversion | 0 output values converted; 0 branch conversions clipped; mean abs integer error 0.000000; max abs integer error 0 |

## By Phase

| Phase | Changed cells | Noise mean sigma | Saturation events | Clamped tile-columns | TDC clipped conversions | TDC mean abs int error |
|---|---:|---:|---:|---:|---:|---:|

## By Module

| Phase | Role | Module / GEMM | Reliability mean / max int error | Noise mean / max sigma | IV saturation events | Clamped tile-columns | TDC mean / max int error |
|---|---|---|---:|---:|---:|---:|---:|
