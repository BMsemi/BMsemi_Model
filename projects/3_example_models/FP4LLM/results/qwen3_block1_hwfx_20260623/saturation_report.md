# Software Crossbar Saturation Report

- Branch model: `zero_point_corrected_signed_branch`
- IV mode: `hard_clamp`
- Full-scale: 2250.0 product levels
- PE row groups: 4
- Upper saturation events: 23448
- Clamped tile-columns: 6917
- Saturated tiles: 2824

## By Phase

| Phase | Upper saturation events | Clamped tile-columns | Saturated tiles | Max branch level |
|---|---:|---:|---:|---:|
| `decode` | 3515 | 3369 | 1576 | 37652 |
| `prefill` | 19933 | 3548 | 1248 | 33450 |

## By Role

| Phase | Role | Module / GEMM | Upper saturation events | Clamped tile-columns | Saturated tiles | Max branch level |
|---|---|---|---:|---:|---:|---:|
| `prefill` | `q_proj` | `q_proj` | 2330 | 367 | 257 | 13492 |
| `prefill` | `k_proj` | `k_proj` | 896 | 166 | 97 | 7122 |
| `prefill` | `v_proj` | `v_proj` | 1345 | 449 | 241 | 10025 |
| `prefill` | `out_proj` | `o_proj` | 714 | 270 | 188 | 9494 |
| `prefill` | `ffn_up` | `gate_proj` | 5379 | 550 | 315 | 12689 |
| `prefill` | `ffn_up` | `up_proj` | 177 | 33 | 32 | 5485 |
| `prefill` | `ffn_down` | `down_proj` | 71 | 71 | 53 | 4677 |
| `decode` | `q_proj` | `q_proj` | 314 | 307 | 195 | 11195 |
| `decode` | `k_proj` | `k_proj` | 125 | 125 | 68 | 6247 |
| `decode` | `v_proj` | `v_proj` | 1190 | 1172 | 588 | 10636 |
| `decode` | `out_proj` | `o_proj` | 572 | 482 | 341 | 9194 |
| `decode` | `ffn_up` | `gate_proj` | 473 | 473 | 282 | 12679 |
| `decode` | `ffn_up` | `up_proj` | 17 | 17 | 17 | 5505 |
| `decode` | `ffn_down` | `down_proj` | 32 | 32 | 22 | 3421 |
| `prefill` | `attn_scores_qkt` | `mapped_attn_scores_qkt` | 1 | 1 | 1 | 2298 |
| `prefill` | `attn_av` | `mapped_attn_av` | 9020 | 1641 | 64 | 33450 |
| `decode` | `attn_scores_qkt` | `mapped_attn_scores_qkt` | 1 | 1 | 1 | 2779 |
| `decode` | `attn_av` | `mapped_attn_av` | 791 | 760 | 62 | 37652 |
