# Software Crossbar Hardware Effects Report

This audit uses the same prompt/decode tensors and W7/A7 quantized weights as the BF16 value run.

| Effect | What was enabled | What happened |
|---|---|---|
| Reliability | 30-day retention drift, state spread, raw BER with ECC | 403800064 cell writes audited; 402159902 cell levels changed; 16811044 INT8 weights changed |
| Noise | Thermal, shot, flicker, RTN, readout model at 5 kHz bandwidth | mean branch sigma 28.508057 levels; max branch sigma 31.644411 levels |
| IV / ADC | hard-clamp branch full-scale before digitization | 26090 saturation events; 13369 clamped tile-columns; 4630 saturated tiles |
| TDC | SAR-FP TDC conversion | 9465664 output values converted; 0 branch conversions clipped; mean abs integer error 10.969128; max abs integer error 22354 |

## By Phase

| Phase | Changed cells | Noise mean sigma | Saturation events | Clamped tile-columns | TDC clipped conversions | TDC mean abs int error |
|---|---:|---:|---:|---:|---:|---:|
| `decode` | 201039397 | 28.492191 | 9244 | 8304 | 0 | 25.093422 |
| `prefill` | 201120505 | 28.509498 | 16846 | 5065 | 0 | 9.685622 |

## By Module

| Phase | Role | Module / GEMM | Reliability mean / max int error | Noise mean / max sigma | IV saturation events | Clamped tile-columns | TDC mean / max int error |
|---|---|---|---:|---:|---:|---:|---:|
| `prefill` | `q_proj` | `q_proj` | 6.669034 / 876 | 28.558459 / 29.475432 | 1403 | 440 | 9.546334 / 10777 |
| `prefill` | `k_proj` | `k_proj` | 4.712135 / 278 | 28.578526 / 29.474218 | 645 | 242 | 7.331973 / 4691 |
| `prefill` | `v_proj` | `v_proj` | 12.594046 / 289 | 28.043088 / 29.458715 | 878 | 529 | 20.533317 / 7319 |
| `prefill` | `out_proj` | `o_proj` | 14.298455 / 323 | 28.485005 / 30.142070 | 4475 | 1431 | 17.655223 / 14112 |
| `prefill` | `ffn_up` | `gate_proj` | 7.878613 / 288 | 28.544926 / 29.233143 | 2273 | 581 | 8.753184 / 10233 |
| `prefill` | `ffn_up` | `up_proj` | 9.658814 / 276 | 28.473174 / 29.231485 | 107 | 46 | 9.144779 / 3087 |
| `prefill` | `ffn_down` | `down_proj` | 1.259659 / 258 | 28.565867 / 29.539367 | 73 | 73 | 1.511371 / 2868 |
| `decode` | `q_proj` | `q_proj` | 10.904049 / 260 | 28.546814 / 29.419942 | 303 | 296 | 16.602203 / 8898 |
| `decode` | `k_proj` | `k_proj` | 7.637917 / 288 | 28.566912 / 29.421082 | 121 | 121 | 12.339752 / 3849 |
| `decode` | `v_proj` | `v_proj` | 20.784958 / 302 | 28.030811 / 29.408245 | 1096 | 1080 | 43.299286 / 8374 |
| `decode` | `out_proj` | `o_proj` | 37.448223 / 466 | 28.402114 / 30.373614 | 5851 | 4937 | 83.862717 / 22087 |
| `decode` | `ffn_up` | `gate_proj` | 13.022728 / 310 | 28.530552 / 29.197906 | 432 | 432 | 15.381602 / 9943 |
| `decode` | `ffn_up` | `up_proj` | 15.992308 / 262 | 28.458804 / 29.195391 | 30 | 30 | 15.012629 / 3319 |
| `decode` | `ffn_down` | `down_proj` | 1.809007 / 258 | 28.560528 / 28.932851 | 30 | 30 | 2.173487 / 989 |
| `prefill` | `attn_scores_qkt` | `mapped_attn_scores_qkt` | 0.717136 / 138 | 28.537861 / 30.121354 | 0 | 0 | 5.755940 / 100 |
| `prefill` | `attn_av` | `mapped_attn_av` | 10.674871 / 466 | 28.754470 / 31.644411 | 6992 | 1723 | 550.049494 / 22354 |
| `decode` | `attn_scores_qkt` | `mapped_attn_scores_qkt` | 6.070312 / 212 | 28.523964 / 29.881141 | 5 | 2 | 182.406250 / 3405 |
| `decode` | `attn_av` | `mapped_attn_av` | 13.704590 / 254 | 28.774760 / 29.478807 | 1376 | 1376 | 2426.906250 / 13571 |
