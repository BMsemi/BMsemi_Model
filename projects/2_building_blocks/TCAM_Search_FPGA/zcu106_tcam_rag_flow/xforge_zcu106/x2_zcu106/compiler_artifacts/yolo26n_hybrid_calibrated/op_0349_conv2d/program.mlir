module attributes {
  x2.schema_version = 1 : i32,
  x2.name = "tflite_conv2d_op_349",
  x2.kind = "conv2d",
  x2.m = 400 : i64,
  x2.k = 144 : i64,
  x2.n = 16 : i64
} {
  "x2.mvm_tile"() {
    x2.tile_id = 0 : i64,
    x2.k_start = 0 : i64,
    x2.n_start = 0 : i64,
    x2.rows = 32 : i64,
    x2.cols = 16 : i64,
    x2.weight_key = "weight_k0_n0",
    x2.accumulate = false
  } : () -> ()
  "x2.mvm_tile"() {
    x2.tile_id = 1 : i64,
    x2.k_start = 32 : i64,
    x2.n_start = 0 : i64,
    x2.rows = 32 : i64,
    x2.cols = 16 : i64,
    x2.weight_key = "weight_k32_n0",
    x2.accumulate = true
  } : () -> ()
  "x2.mvm_tile"() {
    x2.tile_id = 2 : i64,
    x2.k_start = 64 : i64,
    x2.n_start = 0 : i64,
    x2.rows = 32 : i64,
    x2.cols = 16 : i64,
    x2.weight_key = "weight_k64_n0",
    x2.accumulate = true
  } : () -> ()
  "x2.mvm_tile"() {
    x2.tile_id = 3 : i64,
    x2.k_start = 96 : i64,
    x2.n_start = 0 : i64,
    x2.rows = 32 : i64,
    x2.cols = 16 : i64,
    x2.weight_key = "weight_k96_n0",
    x2.accumulate = true
  } : () -> ()
  "x2.mvm_tile"() {
    x2.tile_id = 4 : i64,
    x2.k_start = 128 : i64,
    x2.n_start = 0 : i64,
    x2.rows = 16 : i64,
    x2.cols = 16 : i64,
    x2.weight_key = "weight_k128_n0",
    x2.accumulate = true
  } : () -> ()
}
