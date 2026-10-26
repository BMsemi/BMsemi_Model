module attributes {
  x2.schema_version = 1 : i32,
  x2.name = "tflite_conv2d_op_383",
  x2.kind = "conv2d",
  x2.m = 1600 : i64,
  x2.k = 80 : i64,
  x2.n = 80 : i64
} {
  "x2.mvm_tile"() {
    x2.tile_id = 0 : i64,
    x2.k_start = 0 : i64,
    x2.n_start = 0 : i64,
    x2.rows = 32 : i64,
    x2.cols = 32 : i64,
    x2.weight_key = "weight_k0_n0",
    x2.accumulate = false
  } : () -> ()
  "x2.mvm_tile"() {
    x2.tile_id = 1 : i64,
    x2.k_start = 32 : i64,
    x2.n_start = 0 : i64,
    x2.rows = 32 : i64,
    x2.cols = 32 : i64,
    x2.weight_key = "weight_k32_n0",
    x2.accumulate = true
  } : () -> ()
  "x2.mvm_tile"() {
    x2.tile_id = 2 : i64,
    x2.k_start = 64 : i64,
    x2.n_start = 0 : i64,
    x2.rows = 16 : i64,
    x2.cols = 32 : i64,
    x2.weight_key = "weight_k64_n0",
    x2.accumulate = true
  } : () -> ()
  "x2.mvm_tile"() {
    x2.tile_id = 3 : i64,
    x2.k_start = 0 : i64,
    x2.n_start = 32 : i64,
    x2.rows = 32 : i64,
    x2.cols = 32 : i64,
    x2.weight_key = "weight_k0_n32",
    x2.accumulate = false
  } : () -> ()
  "x2.mvm_tile"() {
    x2.tile_id = 4 : i64,
    x2.k_start = 32 : i64,
    x2.n_start = 32 : i64,
    x2.rows = 32 : i64,
    x2.cols = 32 : i64,
    x2.weight_key = "weight_k32_n32",
    x2.accumulate = true
  } : () -> ()
  "x2.mvm_tile"() {
    x2.tile_id = 5 : i64,
    x2.k_start = 64 : i64,
    x2.n_start = 32 : i64,
    x2.rows = 16 : i64,
    x2.cols = 32 : i64,
    x2.weight_key = "weight_k64_n32",
    x2.accumulate = true
  } : () -> ()
  "x2.mvm_tile"() {
    x2.tile_id = 6 : i64,
    x2.k_start = 0 : i64,
    x2.n_start = 64 : i64,
    x2.rows = 32 : i64,
    x2.cols = 16 : i64,
    x2.weight_key = "weight_k0_n64",
    x2.accumulate = false
  } : () -> ()
  "x2.mvm_tile"() {
    x2.tile_id = 7 : i64,
    x2.k_start = 32 : i64,
    x2.n_start = 64 : i64,
    x2.rows = 32 : i64,
    x2.cols = 16 : i64,
    x2.weight_key = "weight_k32_n64",
    x2.accumulate = true
  } : () -> ()
  "x2.mvm_tile"() {
    x2.tile_id = 8 : i64,
    x2.k_start = 64 : i64,
    x2.n_start = 64 : i64,
    x2.rows = 16 : i64,
    x2.cols = 16 : i64,
    x2.weight_key = "weight_k64_n64",
    x2.accumulate = true
  } : () -> ()
}
