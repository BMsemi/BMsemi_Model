module attributes {
  x2.schema_version = 1 : i32,
  x2.name = "tflite_conv2d_op_132",
  x2.kind = "conv2d",
  x2.m = 400 : i64,
  x2.k = 128 : i64,
  x2.n = 256 : i64
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
    x2.rows = 32 : i64,
    x2.cols = 32 : i64,
    x2.weight_key = "weight_k64_n0",
    x2.accumulate = true
  } : () -> ()
  "x2.mvm_tile"() {
    x2.tile_id = 3 : i64,
    x2.k_start = 96 : i64,
    x2.n_start = 0 : i64,
    x2.rows = 32 : i64,
    x2.cols = 32 : i64,
    x2.weight_key = "weight_k96_n0",
    x2.accumulate = true
  } : () -> ()
  "x2.mvm_tile"() {
    x2.tile_id = 4 : i64,
    x2.k_start = 0 : i64,
    x2.n_start = 32 : i64,
    x2.rows = 32 : i64,
    x2.cols = 32 : i64,
    x2.weight_key = "weight_k0_n32",
    x2.accumulate = false
  } : () -> ()
  "x2.mvm_tile"() {
    x2.tile_id = 5 : i64,
    x2.k_start = 32 : i64,
    x2.n_start = 32 : i64,
    x2.rows = 32 : i64,
    x2.cols = 32 : i64,
    x2.weight_key = "weight_k32_n32",
    x2.accumulate = true
  } : () -> ()
  "x2.mvm_tile"() {
    x2.tile_id = 6 : i64,
    x2.k_start = 64 : i64,
    x2.n_start = 32 : i64,
    x2.rows = 32 : i64,
    x2.cols = 32 : i64,
    x2.weight_key = "weight_k64_n32",
    x2.accumulate = true
  } : () -> ()
  "x2.mvm_tile"() {
    x2.tile_id = 7 : i64,
    x2.k_start = 96 : i64,
    x2.n_start = 32 : i64,
    x2.rows = 32 : i64,
    x2.cols = 32 : i64,
    x2.weight_key = "weight_k96_n32",
    x2.accumulate = true
  } : () -> ()
  "x2.mvm_tile"() {
    x2.tile_id = 8 : i64,
    x2.k_start = 0 : i64,
    x2.n_start = 64 : i64,
    x2.rows = 32 : i64,
    x2.cols = 32 : i64,
    x2.weight_key = "weight_k0_n64",
    x2.accumulate = false
  } : () -> ()
  "x2.mvm_tile"() {
    x2.tile_id = 9 : i64,
    x2.k_start = 32 : i64,
    x2.n_start = 64 : i64,
    x2.rows = 32 : i64,
    x2.cols = 32 : i64,
    x2.weight_key = "weight_k32_n64",
    x2.accumulate = true
  } : () -> ()
  "x2.mvm_tile"() {
    x2.tile_id = 10 : i64,
    x2.k_start = 64 : i64,
    x2.n_start = 64 : i64,
    x2.rows = 32 : i64,
    x2.cols = 32 : i64,
    x2.weight_key = "weight_k64_n64",
    x2.accumulate = true
  } : () -> ()
  "x2.mvm_tile"() {
    x2.tile_id = 11 : i64,
    x2.k_start = 96 : i64,
    x2.n_start = 64 : i64,
    x2.rows = 32 : i64,
    x2.cols = 32 : i64,
    x2.weight_key = "weight_k96_n64",
    x2.accumulate = true
  } : () -> ()
  "x2.mvm_tile"() {
    x2.tile_id = 12 : i64,
    x2.k_start = 0 : i64,
    x2.n_start = 96 : i64,
    x2.rows = 32 : i64,
    x2.cols = 32 : i64,
    x2.weight_key = "weight_k0_n96",
    x2.accumulate = false
  } : () -> ()
  "x2.mvm_tile"() {
    x2.tile_id = 13 : i64,
    x2.k_start = 32 : i64,
    x2.n_start = 96 : i64,
    x2.rows = 32 : i64,
    x2.cols = 32 : i64,
    x2.weight_key = "weight_k32_n96",
    x2.accumulate = true
  } : () -> ()
  "x2.mvm_tile"() {
    x2.tile_id = 14 : i64,
    x2.k_start = 64 : i64,
    x2.n_start = 96 : i64,
    x2.rows = 32 : i64,
    x2.cols = 32 : i64,
    x2.weight_key = "weight_k64_n96",
    x2.accumulate = true
  } : () -> ()
  "x2.mvm_tile"() {
    x2.tile_id = 15 : i64,
    x2.k_start = 96 : i64,
    x2.n_start = 96 : i64,
    x2.rows = 32 : i64,
    x2.cols = 32 : i64,
    x2.weight_key = "weight_k96_n96",
    x2.accumulate = true
  } : () -> ()
  "x2.mvm_tile"() {
    x2.tile_id = 16 : i64,
    x2.k_start = 0 : i64,
    x2.n_start = 128 : i64,
    x2.rows = 32 : i64,
    x2.cols = 32 : i64,
    x2.weight_key = "weight_k0_n128",
    x2.accumulate = false
  } : () -> ()
  "x2.mvm_tile"() {
    x2.tile_id = 17 : i64,
    x2.k_start = 32 : i64,
    x2.n_start = 128 : i64,
    x2.rows = 32 : i64,
    x2.cols = 32 : i64,
    x2.weight_key = "weight_k32_n128",
    x2.accumulate = true
  } : () -> ()
  "x2.mvm_tile"() {
    x2.tile_id = 18 : i64,
    x2.k_start = 64 : i64,
    x2.n_start = 128 : i64,
    x2.rows = 32 : i64,
    x2.cols = 32 : i64,
    x2.weight_key = "weight_k64_n128",
    x2.accumulate = true
  } : () -> ()
  "x2.mvm_tile"() {
    x2.tile_id = 19 : i64,
    x2.k_start = 96 : i64,
    x2.n_start = 128 : i64,
    x2.rows = 32 : i64,
    x2.cols = 32 : i64,
    x2.weight_key = "weight_k96_n128",
    x2.accumulate = true
  } : () -> ()
  "x2.mvm_tile"() {
    x2.tile_id = 20 : i64,
    x2.k_start = 0 : i64,
    x2.n_start = 160 : i64,
    x2.rows = 32 : i64,
    x2.cols = 32 : i64,
    x2.weight_key = "weight_k0_n160",
    x2.accumulate = false
  } : () -> ()
  "x2.mvm_tile"() {
    x2.tile_id = 21 : i64,
    x2.k_start = 32 : i64,
    x2.n_start = 160 : i64,
    x2.rows = 32 : i64,
    x2.cols = 32 : i64,
    x2.weight_key = "weight_k32_n160",
    x2.accumulate = true
  } : () -> ()
  "x2.mvm_tile"() {
    x2.tile_id = 22 : i64,
    x2.k_start = 64 : i64,
    x2.n_start = 160 : i64,
    x2.rows = 32 : i64,
    x2.cols = 32 : i64,
    x2.weight_key = "weight_k64_n160",
    x2.accumulate = true
  } : () -> ()
  "x2.mvm_tile"() {
    x2.tile_id = 23 : i64,
    x2.k_start = 96 : i64,
    x2.n_start = 160 : i64,
    x2.rows = 32 : i64,
    x2.cols = 32 : i64,
    x2.weight_key = "weight_k96_n160",
    x2.accumulate = true
  } : () -> ()
  "x2.mvm_tile"() {
    x2.tile_id = 24 : i64,
    x2.k_start = 0 : i64,
    x2.n_start = 192 : i64,
    x2.rows = 32 : i64,
    x2.cols = 32 : i64,
    x2.weight_key = "weight_k0_n192",
    x2.accumulate = false
  } : () -> ()
  "x2.mvm_tile"() {
    x2.tile_id = 25 : i64,
    x2.k_start = 32 : i64,
    x2.n_start = 192 : i64,
    x2.rows = 32 : i64,
    x2.cols = 32 : i64,
    x2.weight_key = "weight_k32_n192",
    x2.accumulate = true
  } : () -> ()
  "x2.mvm_tile"() {
    x2.tile_id = 26 : i64,
    x2.k_start = 64 : i64,
    x2.n_start = 192 : i64,
    x2.rows = 32 : i64,
    x2.cols = 32 : i64,
    x2.weight_key = "weight_k64_n192",
    x2.accumulate = true
  } : () -> ()
  "x2.mvm_tile"() {
    x2.tile_id = 27 : i64,
    x2.k_start = 96 : i64,
    x2.n_start = 192 : i64,
    x2.rows = 32 : i64,
    x2.cols = 32 : i64,
    x2.weight_key = "weight_k96_n192",
    x2.accumulate = true
  } : () -> ()
  "x2.mvm_tile"() {
    x2.tile_id = 28 : i64,
    x2.k_start = 0 : i64,
    x2.n_start = 224 : i64,
    x2.rows = 32 : i64,
    x2.cols = 32 : i64,
    x2.weight_key = "weight_k0_n224",
    x2.accumulate = false
  } : () -> ()
  "x2.mvm_tile"() {
    x2.tile_id = 29 : i64,
    x2.k_start = 32 : i64,
    x2.n_start = 224 : i64,
    x2.rows = 32 : i64,
    x2.cols = 32 : i64,
    x2.weight_key = "weight_k32_n224",
    x2.accumulate = true
  } : () -> ()
  "x2.mvm_tile"() {
    x2.tile_id = 30 : i64,
    x2.k_start = 64 : i64,
    x2.n_start = 224 : i64,
    x2.rows = 32 : i64,
    x2.cols = 32 : i64,
    x2.weight_key = "weight_k64_n224",
    x2.accumulate = true
  } : () -> ()
  "x2.mvm_tile"() {
    x2.tile_id = 31 : i64,
    x2.k_start = 96 : i64,
    x2.n_start = 224 : i64,
    x2.rows = 32 : i64,
    x2.cols = 32 : i64,
    x2.weight_key = "weight_k96_n224",
    x2.accumulate = true
  } : () -> ()
}
