# Software Crossbar Hardware Effects Report

This audit uses the same prompt/decode tensors and W7/A7 quantized weights as the BF16 value run.

| Effect | What was enabled | What happened |
|---|---|---|
| Reliability | 30-day retention drift, state spread, raw BER with ECC | 404152320 cell writes audited; 402509700 cell levels changed; 16831854 INT8 weights changed |
| Noise | Thermal, shot, flicker, RTN, readout model at 5 kHz bandwidth | mean branch sigma 28.503848 levels; max branch sigma 32.351608 levels |
| IV / ADC | hard-clamp branch full-scale before digitization | 22766 saturation events; 6680 clamped tile-columns; 2745 saturated tiles |
| TDC | SAR-FP TDC conversion | 22101664 output values converted; 0 branch conversions clipped; mean abs integer error 6.663875; max abs integer error 58444 |

## By Phase

| Phase | Changed cells | Noise mean sigma | Saturation events | Clamped tile-columns | TDC clipped conversions | TDC mean abs int error |
|---|---:|---:|---:|---:|---:|---:|
| `decode` | 201258905 | 28.498481 | 3355 | 3217 | 0 | 14.045362 |
| `prefill` | 201250795 | 28.504046 | 19411 | 3463 | 0 | 6.390476 |

## By Module

| Phase | Role | Module / GEMM | Reliability mean / max int error | Noise mean / max sigma | IV saturation events | Clamped tile-columns | TDC mean / max int error |
|---|---|---|---:|---:|---:|---:|---:|
| `prefill` | `q_proj` | `q_proj` | 5.770995 / 1217 | 28.553056 / 29.365542 | 2263 | 356 | 7.627803 / 11188 |
| `prefill` | `k_proj` | `k_proj` | 4.073315 / 266 | 28.573129 / 29.365076 | 852 | 163 | 5.930371 / 4834 |
| `prefill` | `v_proj` | `v_proj` | 11.034617 / 262 | 28.037677 / 29.352443 | 1280 | 420 | 17.504895 / 7588 |
| `prefill` | `out_proj` | `o_proj` | 5.154155 / 266 | 28.518017 / 29.418151 | 680 | 259 | 5.426836 / 7088 |
| `prefill` | `ffn_up` | `gate_proj` | 5.740430 / 270 | 28.529796 / 29.096074 | 5270 | 537 | 6.797378 / 10363 |
| `prefill` | `ffn_up` | `up_proj` | 6.967156 / 258 | 28.458070 / 29.093946 | 162 | 30 | 6.689501 / 3176 |
| `prefill` | `ffn_down` | `down_proj` | 1.034230 / 258 | 28.566672 / 29.501420 | 69 | 69 | 1.210305 / 2375 |
| `decode` | `q_proj` | `q_proj` | 10.904049 / 260 | 28.546814 / 29.419942 | 303 | 296 | 16.602203 / 8898 |
| `decode` | `k_proj` | `k_proj` | 7.637917 / 288 | 28.566912 / 29.421082 | 121 | 121 | 12.339752 / 3849 |
| `decode` | `v_proj` | `v_proj` | 20.784958 / 302 | 28.030811 / 29.408245 | 1096 | 1080 | 43.299286 / 8374 |
| `decode` | `out_proj` | `o_proj` | 14.805687 / 298 | 28.468358 / 29.681793 | 540 | 455 | 19.438004 / 7452 |
| `decode` | `ffn_up` | `gate_proj` | 10.352207 / 282 | 28.530327 / 29.150785 | 468 | 468 | 13.224665 / 10418 |
| `decode` | `ffn_up` | `up_proj` | 12.601044 / 272 | 28.458527 / 29.148623 | 17 | 17 | 11.982290 / 3207 |
| `decode` | `ffn_down` | `down_proj` | 1.235446 / 260 | 28.561569 / 28.709387 | 32 | 32 | 1.476451 / 991 |
| `prefill` | `attn_scores_qkt` | `mapped_attn_scores_qkt` | 0.617948 / 168 | 28.560793 / 30.148166 | 1 | 1 | 4.803712 / 102 |
| `prefill` | `attn_av` | `mapped_attn_av` | 9.916780 / 422 | 28.751050 / 31.214855 | 8834 | 1628 | 257.360279 / 38893 |
| `decode` | `attn_scores_qkt` | `mapped_attn_scores_qkt` | 0.962612 / 147 | 28.589405 / 29.882369 | 1 | 1 | 7.300223 / 348 |
| `decode` | `attn_av` | `mapped_attn_av` | 16.569458 / 513 | 28.985600 / 32.351608 | 777 | 747 | 805.297363 / 58444 |
