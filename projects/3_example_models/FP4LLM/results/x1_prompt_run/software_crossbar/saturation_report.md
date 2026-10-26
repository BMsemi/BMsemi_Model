# Software Crossbar Saturation Report

- Branch model: `zero_point_corrected_signed_branch`
- IV mode: `hard_clamp`
- Full-scale: 2250.0 product levels
- PE row groups: 4
- Upper saturation events: 27599
- Clamped tile-columns: 14441
- Saturated tiles: 4815

## By Phase

| Phase | Upper saturation events | Clamped tile-columns | Saturated tiles | Max branch level |
|---|---:|---:|---:|---:|
| `decode` | 10207 | 9164 | 2802 | 16129 |
| `prefill` | 17392 | 5277 | 2013 | 24933 |

## By Role

| Phase | Role | Module / GEMM | Upper saturation events | Clamped tile-columns | Saturated tiles | Max branch level |
|---|---|---|---:|---:|---:|---:|
| `prefill` | `q_proj` | `q_proj` | 1444 | 449 | 285 | 13090 |
| `prefill` | `k_proj` | `k_proj` | 667 | 246 | 126 | 7000 |
| `prefill` | `v_proj` | `v_proj` | 944 | 577 | 316 | 9643 |
| `prefill` | `out_proj` | `o_proj` | 4746 | 1545 | 762 | 11504 |
| `prefill` | `ffn_up` | `gate_proj` | 2323 | 602 | 346 | 12539 |
| `prefill` | `ffn_up` | `up_proj` | 111 | 48 | 46 | 5365 |
| `prefill` | `ffn_down` | `down_proj` | 75 | 75 | 68 | 5169 |
| `decode` | `q_proj` | `q_proj` | 314 | 307 | 195 | 11195 |
| `decode` | `k_proj` | `k_proj` | 125 | 125 | 68 | 6247 |
| `decode` | `v_proj` | `v_proj` | 1190 | 1172 | 588 | 10636 |
| `decode` | `out_proj` | `o_proj` | 6688 | 5673 | 1582 | 15713 |
| `decode` | `ffn_up` | `gate_proj` | 446 | 446 | 249 | 12139 |
| `decode` | `ffn_up` | `up_proj` | 33 | 33 | 33 | 5643 |
| `decode` | `ffn_down` | `down_proj` | 30 | 30 | 21 | 3425 |
| `prefill` | `attn_scores_qkt` | `mapped_attn_scores_qkt` | 0 | 0 | 0 | 1789 |
| `prefill` | `attn_av` | `mapped_attn_av` | 7082 | 1735 | 64 | 24933 |
| `decode` | `attn_scores_qkt` | `mapped_attn_scores_qkt` | 5 | 2 | 2 | 5661 |
| `decode` | `attn_av` | `mapped_attn_av` | 1376 | 1376 | 64 | 16129 |
