#include "qwen_nlpc.h"
#include <math.h>
#include <limits.h>

static float as_float(uint32_t v) { union {uint32_t u; float f;} x; x.u=v; return x.f; }
static uint32_t as_word(float v) { union {uint32_t u; float f;} x; x.f=v; return x.u; }
static float bf16(float v) {
    uint32_t u=as_word(v);
    // Preserve infinities/NaNs for error reporting; never turn NaN into zero.
    if ((u & 0x7f800000u) != 0x7f800000u)
        u=(u+0x7fffu+((u>>16)&1u)) & 0xffff0000u;
    return as_float(u);
}
static bool finite_value(float v) { return (as_word(v)&0x7f800000u)!=0x7f800000u; }

void qwen_nlpc(const uint32_t *a, const uint32_t *b, uint32_t *out,
               unsigned rows, unsigned cols, unsigned op, unsigned row_offset,
               float epsilon, float scale, unsigned &status) {
// depth sizes the C/RTL testbench memory, not a runtime transfer limit.
// Runtime bounds are validated below; all master addresses are 64-bit.
#pragma HLS INTERFACE m_axi port=a offset=slave bundle=mem depth=8192
#pragma HLS INTERFACE m_axi port=b offset=slave bundle=mem depth=8192
#pragma HLS INTERFACE m_axi port=out offset=slave bundle=mem depth=8192
#pragma HLS INTERFACE s_axilite port=a bundle=control offset=0x10
#pragma HLS INTERFACE s_axilite port=b bundle=control offset=0x1c
#pragma HLS INTERFACE s_axilite port=out bundle=control offset=0x28
#pragma HLS INTERFACE s_axilite port=rows bundle=control offset=0x34
#pragma HLS INTERFACE s_axilite port=cols bundle=control offset=0x3c
#pragma HLS INTERFACE s_axilite port=op bundle=control offset=0x44
#pragma HLS INTERFACE s_axilite port=row_offset bundle=control offset=0x4c
#pragma HLS INTERFACE s_axilite port=epsilon bundle=control offset=0x54
#pragma HLS INTERFACE s_axilite port=scale bundle=control offset=0x5c
#pragma HLS INTERFACE s_axilite port=status bundle=control offset=0x64
#pragma HLS INTERFACE s_axilite port=return bundle=control
    status=OK;
    if (op==PROBE) { out[0]=QWEN_ABI; return; }
    if (op>RESIDUAL_ADD) { status=BAD_OP; return; }
    if (!rows || rows>MAX_ROWS || !cols || cols>MAX_COLS ||
        (op==ROPE && cols%2) ||
        (op==RMS_NORM && (!(epsilon>0) || !finite_value(epsilon))) ||
        (op==CAUSAL_SOFTMAX && (row_offset>=cols || rows>cols-row_offset ||
                               !(scale>0) || !finite_value(scale)))) {
        status=BAD_SHAPE; return;
    }
    float row[MAX_COLS];
#pragma HLS BIND_STORAGE variable=row type=ram_2p impl=bram
    for (unsigned r=0;r<rows;r++) {
        const unsigned base=r*cols;
        if (op==ADD_I32) {
            for (unsigned c=0;c<cols;c++) {
                int64_t sum=(int64_t)(int32_t)a[base+c]+(int64_t)(int32_t)b[base+c];
                if (sum>INT_MAX || sum<INT_MIN) { status=OVERFLOW; return; }
                out[base+c]=(uint32_t)(int32_t)sum;
            }
            continue;
        }
        float sum=0, maximum=-3.402823466e+38F;
        for (unsigned c=0;c<cols;c++) {
            float v=as_float(a[base+c]);
            if (!finite_value(v)) { status=NONFINITE; return; }
            if (op==CAUSAL_SOFTMAX) v=v*scale;
            row[c]=v;
            if (op==RMS_NORM) sum+=v*v;
            if (op==CAUSAL_SOFTMAX && c<=r+row_offset && v>maximum) maximum=v;
        }
        float inv=1;
        if (op==RMS_NORM) inv=1.0f/sqrtf(sum/(float)cols+epsilon);
        if (op==CAUSAL_SOFTMAX) {
            for (unsigned c=0;c<cols;c++) {
                row[c]=(c<=r+row_offset)?expf(row[c]-maximum):0.0f;
                sum+=row[c];
            }
            inv=1.0f/sum;
        }
        for (unsigned c=0;c<cols;c++) {
            float v=row[c];
            if (op==RMS_NORM) {
                // Qwen RMSNorm casts the normalized value before gain multiply.
                v=bf16(v*inv)*as_float(b[c]);
            } else if (op==ROPE) {
                unsigned half=cols/2;
                unsigned other=(c<half)?c+half:c-half;
                float rotated=(c<half)?-row[other]:row[other];
                // b stores [cos(cols), sin(cols)] per row. PyTorch BF16
                // elementwise products round before their final addition.
                v=bf16(v*as_float(b[2*base+c]))+
                  bf16(rotated*as_float(b[2*base+cols+c]));
            } else if (op==CAUSAL_SOFTMAX) v=v*inv;
            else if (op==SILU_MUL) {
                // Stable sigmoid; avoid exp overflow for a large negative gate.
                float e=expf(-fabsf(v));
                float sigmoid=(v>=0)?1.0f/(1.0f+e):e/(1.0f+e);
                v=bf16(v*sigmoid)*as_float(b[base+c]);
            } else if (op==RESIDUAL_ADD) v=v+as_float(b[base+c]);
            v=bf16(v);
            if (!finite_value(v)) { status=NONFINITE; return; }
            out[base+c]=as_word(v);
        }
    }
}
