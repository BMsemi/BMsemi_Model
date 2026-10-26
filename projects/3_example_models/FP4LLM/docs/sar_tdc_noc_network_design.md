# SAR-TDC Dynamic-Packing NoC Design

## Goal

The NoC should not know only one fixed output width. It should move PE results as typed packets whose payload width is selected per layer role:

| Role | SAR-TDC format | Payload/value |
|---|---:|---:|
| `gate_proj`, `up_proj` | c4/f3 | 8 bits |
| `down_proj` | c5/f3 | 9 bits |
| `q_proj`, `k_proj`, `v_proj`, `out_proj` | c5/f3 | 9 bits |
| QK scores | c5/f3 | 9 bits |
| AV | c5/f5 | 11 bits |

The rule is simple: reduce row tiles locally, pack the corrected PE output locally, and let the NoC carry opaque compact payloads.

## Hardware Blocks

```text
PE array
  |
  | 32 logical columns of corrected signed PE results
  v
Local row-tile reducer
  |
  | still local, no mesh crossing yet
  v
SAR-TDC packer
  |
  | sign + coarse exponent + fine mantissa
  v
Dynamic packetizer
  |
  | header + bit-packed payload
  v
NoC router fabric
  |
  v
Destination depacketizer / unpacker
  |
  v
Consumer reducer, activation, next PE, SRAM, or output buffer
```

The router only needs to understand packet length, route, virtual channel, and tail mask. It does not need to understand c4/f3 or c5/f5 arithmetic. The endpoint packer/unpacker owns the numerical format.

## Packet Header

A compact universal header can be 64 bits:

| Field | Bits | Purpose |
|---|---:|---|
| `fmt_id` | 4 | `0=raw32`, `1=sar_fp`, `2=int8`, `3=bf16`, future formats |
| `role_id` | 4 | Q/K/V/O, FFN up/down, QK, AV |
| `bits_per_value` | 5 | 8, 9, 11, 16, 32, etc. |
| `coarse_bits` | 3 | 4 or 5 now |
| `fine_bits` | 3 | 3 or 5 now |
| `gain_shift` | 5 | TDC gain exponent bias |
| `value_count` | 10 | number of values in packet |
| `tile_group_id` | 12 | identifies row/column/tile group |
| `reduce_mode` | 3 | none, local row reduce, destination reduce |
| `tail_valid_bits` | 7 | valid bits in final flit |
| `reserved` | 8 | future use |

For a more aggressive implementation, `coarse_bits` and `fine_bits` can be inferred from `role_id`, but keeping them explicit makes the NoC endpoint universal.

## Dynamic Packing

The packetizer takes `(value_count, bits_per_value)` and emits a dense bitstream. For c5/f3, each value is 9 bits:

```text
32 columns x 9 bits = 288 bits = 36 bytes
```

For AV c5/f5:

```text
32 columns x 11 bits = 352 bits = 44 bytes
```

For gate/up c4/f3:

```text
32 columns x 8 bits = 256 bits = 32 bytes
```

With a 128-bit flit and a 64-bit header, one single-tile packet has tail waste. Coalescing 4 tiles per packet amortizes that overhead:

| Role | Payload for 4 tiles | Header+payload | 128-bit flits | Wire bits | 32-bit baseline wire bits | Wire saving |
|---|---:|---:|---:|---:|---:|---:|
| gate/up c4/f3 | 1024 b | 1088 b | 9 | 1152 b | 4224 b | 72.73% |
| c5/f3 roles | 1152 b | 1216 b | 10 | 1280 b | 4224 b | 69.70% |
| AV c5/f5 | 1408 b | 1472 b | 12 | 1536 b | 4224 b | 63.64% |

This is why the NoC should support packet coalescing. Exact payload savings are 75%, 71.875%, and 65.625%, but flit occupancy depends on header and tail alignment.

## Local Reduction Rule

For a 2048-input x 2048-output projection with 64-row tiles:

```text
row_tiles = 2048 / 64 = 32
col_tiles = 2048 / 32 = 64
```

Baseline behavior sends many 32-bit row partials across the mesh:

```text
31 partial rows x 2048 outputs x 32 bits = 2,031,616 bits per token
```

SAR-TDC-aware behavior keeps the 32 row tiles for the same output columns near each other:

```text
local PE island:
  row tile 0
  row tile 1
  ...
  row tile 31
  local reducer
  SAR-TDC packer

NoC egress:
  one compact packed output stream
```

The result is that the mesh carries the final packed PE result instead of all intermediate row partials.

## Universal NoC Interface

Use these rules to make the NoC more universal:

1. Treat payload as opaque variable-width data.
2. Put `bits_per_value`, `value_count`, and `fmt_id` in the header.
3. Use tail-valid bits so the final flit does not need to be fully valid.
4. Allow packet coalescing over multiple tiles with the same destination and format.
5. Keep one unpacker primitive that accepts any `bits_per_value <= 32`.
6. Use virtual channels by traffic class:
   - VC0: activations and residual streams
   - VC1: packed SAR-TDC PE outputs
   - VC2: reductions / collectives
   - VC3: dynamic K/V cache programming
   - VC4: control and descriptors
7. Keep routing independent of numerical format. The router sees only length, destination, VC, and QoS.

## Example: FFN Gate/Up

`gate_proj` and `up_proj` use c4/f3:

```text
sign: 1 bit
coarse exponent: 4 bits
fine mantissa: 3 bits
total: 8 bits/value
```

For one 32-column tile:

```text
baseline output packet: 32 x 32 = 1024 payload bits
SAR-TDC packet:         32 x 8  = 256 payload bits
payload saving:         75%
```

The packet header says:

```text
fmt_id=sar_fp
role_id=ffn_up
bits_per_value=8
coarse_bits=4
fine_bits=3
value_count=32
reduce_mode=local_row_reduce
```

## Example: Q/K/V/O Or Down Projection

These use c5/f3:

```text
sign: 1 bit
coarse exponent: 5 bits
fine mantissa: 3 bits
total: 9 bits/value
```

Pack 32 values:

```text
32 x 9 = 288 bits = 36 bytes
```

A universal unpacker reads 9-bit chunks from the payload and reconstructs sign/coarse/fine fields.

## Example: AV

AV uses c5/f5 because it is after softmax and has no later softmax normalization to hide error:

```text
sign: 1 bit
coarse exponent: 5 bits
fine mantissa: 5 bits
total: 11 bits/value
```

Pack 32 values:

```text
32 x 11 = 352 bits = 44 bytes
payload saving vs 32-bit = 65.625%
```

## Router Behavior

Routers do not unpack. They:

1. Read fixed header.
2. Reserve output VC credits for `ceil((header_bits + payload_bits) / flit_bits)`.
3. Forward body flits.
4. Preserve `tail_valid_bits`.
5. Optionally merge adjacent packets when route, VC, format, and destination match.

This keeps router timing simple and makes numerical formats extensible.

## Endpoint Behavior

Endpoint packer:

```text
input: signed PE result, role policy
1. choose c/f bits from role or descriptor
2. produce SAR mantissa/exponent code
3. bit-pack values into payload stream
4. emit header
```

Endpoint unpacker:

```text
input: header, payload
1. read bits_per_value and value_count
2. unpack opaque values
3. decode by fmt_id and c/f bits
4. reconstruct integer/float-like value for local consumer
```

## Recommended Default

- Flit width: 128 or 256 bits.
- Header: 64 bits fixed.
- Coalesce: 4 or 8 tiles per packet when destination and role match.
- Exact bit packing inside packet payload.
- Byte-aligned mode only for simple router endpoints or debug.
- Keep role policy table programmable in a small CSR table so future layers can choose c4/f4, c5/f2, raw16, or raw32 without changing router RTL.
