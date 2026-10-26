#include "qwen_nlpc.h"
#include <cassert>
#include <cmath>
#include <cstdio>
#include <climits>
static uint32_t word(float f) { union{float f;uint32_t u;}x; x.f=f; return x.u; }
static float value(uint32_t u) { union{float f;uint32_t u;}x; x.u=u; return x.f; }
int main() {
    static uint32_t a[8192]={}, b[8192]={}, out[8192]={}; unsigned status=99;
    qwen_nlpc(a,b,out,0,0,PROBE,0,1e-6f,1,status);
    assert(status==OK && out[0]==QWEN_ABI);
    a[0]=(uint32_t)-20; b[0]=7;
    qwen_nlpc(a,b,out,1,1,ADD_I32,0,1e-6f,1,status);
    assert(status==OK && (int32_t)out[0]==-13);
    a[0]=INT_MAX; b[0]=1;
    qwen_nlpc(a,b,out,1,1,ADD_I32,0,1e-6f,1,status); assert(status==OVERFLOW);
    for(unsigned i=0;i<32;i++) {a[i]=word(2); b[i]=word(1);}
    qwen_nlpc(a,b,out,2,4,RMS_NORM,0,1e-6f,1,status);
    assert(status==OK); for(int i=0;i<8;i++) assert(value(out[i])==1);
    for(unsigned i=0;i<8;i++) {a[i]=word(float(i+1)); b[i]=word(i<4?0:1);}
    qwen_nlpc(a,b,out,1,4,ROPE,0,1e-6f,1,status);
    assert(status==OK && value(out[0])==-3 && value(out[1])==-4 && value(out[2])==1 && value(out[3])==2);
    for(unsigned i=0;i<16;i++) a[i]=word(1000);
    qwen_nlpc(a,b,out,4,4,CAUSAL_SOFTMAX,0,1e-6f,1,status);
    assert(status==OK && value(out[0])==1 && value(out[1])==0);
    assert(value(out[4])==.5f && value(out[5])==.5f && value(out[6])==0);
    qwen_nlpc(a,b,out,2,4,CAUSAL_SOFTMAX,2,1e-6f,1,status);
    assert(status==OK && value(out[3])==0 && value(out[4])==.25f);
    a[0]=word(-1000); b[0]=word(1);
    qwen_nlpc(a,b,out,1,1,SILU_MUL,0,1e-6f,1,status);
    assert(status==OK && value(out[0])==0);
    a[0]=word(1.5); b[0]=word(-2);
    qwen_nlpc(a,b,out,1,1,RESIDUAL_ADD,0,1e-6f,1,status);
    assert(status==OK && value(out[0])==-.5f);
    qwen_nlpc(a,b,out,1,3,ROPE,0,1e-6f,1,status); assert(status==BAD_SHAPE);
    qwen_nlpc(a,b,out,1,1,99,0,1e-6f,1,status); assert(status==BAD_OP);
    // Exercise nontrivial exp inputs in generated RTL, not just exp(0).
    for(unsigned i=0;i<16;i++) a[i]=word(float(i%4)-1.0f);
    qwen_nlpc(a,b,out,4,4,CAUSAL_SOFTMAX,0,1e-6f,1,status);
    assert(status==OK && value(out[15])==.64453125f && value(out[3])==0);
    a[0]=word(1); a[1]=word(-1); b[0]=word(2); b[1]=word(2);
    qwen_nlpc(a,b,out,1,2,SILU_MUL,0,1e-6f,1,status);
    assert(status==OK && value(out[0])==1.4609375f && value(out[1])==-.5390625f);
    for(unsigned i=0;i<8;i++) a[i]=word(float(i%4)+1);
    for(unsigned i=0;i<16;i++) b[i]=word(.5f);
    qwen_nlpc(a,b,out,2,4,ROPE,0,1e-6f,1,status);
    assert(status==OK && value(out[4])==-1 && value(out[5])==-1 && value(out[6])==2 && value(out[7])==3);
    for(unsigned i=0;i<6144;i++) { a[i]=word(2); b[i]=word(1); }
    qwen_nlpc(a,b,out,1,6144,RMS_NORM,0,1e-6f,1,status);
    assert(status==OK); for(unsigned i=0;i<6144;i++) assert(value(out[i])==1);
    puts("NLPC C tests passed");
}
