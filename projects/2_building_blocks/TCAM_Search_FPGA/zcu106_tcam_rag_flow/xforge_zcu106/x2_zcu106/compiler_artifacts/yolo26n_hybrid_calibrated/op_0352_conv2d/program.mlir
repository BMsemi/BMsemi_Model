module attributes {
  x2.schema_version = 1 : i32,
  x2.name = "tflite_conv2d_op_352",
  x2.kind = "conv2d",
  x2.m = 400 : i64,
  x2.k = 16 : i64,
  x2.n = 4 : i64
} {
  "x2.mvm_tile"() {
    x2.tile_id = 0 : i64,
    x2.k_start = 0 : i64,
    x2.n_start = 0 : i64,
    x2.rows = 16 : i64,
    x2.cols = 4 : i64,
    x2.weight_key = "weight_k0_n0",
    x2.accumulate = false
  } : () -> ()
}
