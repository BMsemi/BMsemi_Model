module attributes {
  x2.schema_version = 1 : i32,
  x2.name = "tflite_conv2d_op_22",
  x2.kind = "conv2d",
  x2.m = 25600 : i64,
  x2.k = 48 : i64,
  x2.n = 64 : i64
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
    x2.rows = 16 : i64,
    x2.cols = 32 : i64,
    x2.weight_key = "weight_k32_n0",
    x2.accumulate = true
  } : () -> ()
  "x2.mvm_tile"() {
    x2.tile_id = 2 : i64,
    x2.k_start = 0 : i64,
    x2.n_start = 32 : i64,
    x2.rows = 32 : i64,
    x2.cols = 32 : i64,
    x2.weight_key = "weight_k0_n32",
    x2.accumulate = false
  } : () -> ()
  "x2.mvm_tile"() {
    x2.tile_id = 3 : i64,
    x2.k_start = 32 : i64,
    x2.n_start = 32 : i64,
    x2.rows = 16 : i64,
    x2.cols = 32 : i64,
    x2.weight_key = "weight_k32_n32",
    x2.accumulate = true
  } : () -> ()
}
