#pragma once
#include <stdint.h>

// ABI 1. All memory words are packed 32-bit: FP32 containers holding BF16
// values, except ADD_I32 which takes/returns signed INT32 bit patterns.
enum QwenOp { PROBE=0, ADD_I32=1, RMS_NORM=2, ROPE=3,
              CAUSAL_SOFTMAX=4, SILU_MUL=5, RESIDUAL_ADD=6 };
enum QwenStatus { OK=0, BAD_SHAPE=1, BAD_OP=2, OVERFLOW=3, NONFINITE=4 };
static const uint32_t QWEN_ABI = 0x514E0001;
static const unsigned MAX_COLS = 6144;
static const unsigned MAX_ROWS = 4096;
void qwen_nlpc(const uint32_t *a, const uint32_t *b, uint32_t *out,
               unsigned rows, unsigned cols, unsigned op, unsigned row_offset,
               float epsilon, float scale, unsigned &status);
