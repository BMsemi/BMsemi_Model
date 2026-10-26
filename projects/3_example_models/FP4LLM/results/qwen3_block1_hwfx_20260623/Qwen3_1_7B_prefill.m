Network Qwen3_1_7B_28L_Seq512_UserPattern_Cluster4P {



Layer L01_Q_Proj {
  Type: CONV
  Stride { X:1, Y:1 }
  Dimensions { K:2048, C:2048, R:1, S:1, Y:512, X:1 }
  Dataflow {
    SpatialMap(1,1) K;
    TemporalMap(32,32) C;
    SpatialMap(256,256) Y;
    TemporalMap(Sz(R),Sz(R)) R;
    TemporalMap(Sz(S),Sz(S)) S;
    TemporalMap(1,1) X;
  }
}

Layer L02_K_Proj {
  Type: CONV
  Stride { X:1, Y:1 }
  Dimensions { K:1024, C:2048, R:1, S:1, Y:512, X:1 }
  Dataflow {
    SpatialMap(1,1) K;
    TemporalMap(32,32) C;
    SpatialMap(256,256) Y;
    TemporalMap(Sz(R),Sz(R)) R;
    TemporalMap(Sz(S),Sz(S)) S;
    TemporalMap(1,1) X;
  }
}

Layer L03_V_Proj {
  Type: CONV
  Stride { X:1, Y:1 }
  Dimensions { K:1024, C:2048, R:1, S:1, Y:512, X:1 }
  Dataflow {
    SpatialMap(1,1) K;
    TemporalMap(32,32) C;
    SpatialMap(256,256) Y;
    TemporalMap(Sz(R),Sz(R)) R;
    TemporalMap(Sz(S),Sz(S)) S;
    TemporalMap(1,1) X;
  }
}

Layer L04_Attn_Scores_QKt {
  Type: CONV
  Stride { X:1, Y:1 }
  Dimensions { K:512, C:128, R:1, S:1, Y:512, X:1 }
  Dataflow {
    SpatialMap(1,1) K;
    TemporalMap(8,8) C;
    TemporalMap(16,16) Y;
    TemporalMap(Sz(R),Sz(R)) R;
    TemporalMap(Sz(S),Sz(S)) S;
    TemporalMap(1,1) X;
  }
}

Layer L05_Attn_AV {
  Type: CONV
  Stride { X:1, Y:1 }
  Dimensions { K:128, C:512, R:1, S:1, Y:512, X:1 }
  Dataflow {
    SpatialMap(1,1) K;
    TemporalMap(16,16) C;
    TemporalMap(8,8) Y;
    TemporalMap(Sz(R),Sz(R)) R;
    TemporalMap(Sz(S),Sz(S)) S;
    TemporalMap(1,1) X;
  }
}

Layer L06_Out_Proj {
  Type: CONV
  Stride { X:1, Y:1 }
  Dimensions { K:2048, C:2048, R:1, S:1, Y:512, X:1 }
  Dataflow {
    SpatialMap(1,1) K;
    TemporalMap(32,32) C;
    SpatialMap(256,256) Y;
    TemporalMap(Sz(R),Sz(R)) R;
    TemporalMap(Sz(S),Sz(S)) S;
    TemporalMap(1,1) X;
  }
}

Layer L07_FFN_1 {
  Type: CONV
  Stride { X:1, Y:1 }
  Dimensions { K:6144, C:2048, R:1, S:1, Y:512, X:1 }
  Dataflow {
    SpatialMap(1,1) K;
    TemporalMap(32,32) C;
    SpatialMap(256,256) Y;
    TemporalMap(Sz(R),Sz(R)) R;
    TemporalMap(Sz(S),Sz(S)) S;
    TemporalMap(1,1) X;
  }
}

Layer L08_FFN_2 {
  Type: CONV
  Stride { X:1, Y:1 }
  Dimensions { K:2048, C:6144, R:1, S:1, Y:512, X:1 }
  Dataflow {
    SpatialMap(1,1) K;
    TemporalMap(32,32) C;
    SpatialMap(256,256) Y;
    TemporalMap(Sz(R),Sz(R)) R;
    TemporalMap(Sz(S),Sz(S)) S;
    TemporalMap(1,1) X;
  }
}

Layer L09_Q_Proj {
  Type: CONV
  Stride { X:1, Y:1 }
  Dimensions { K:2048, C:2048, R:1, S:1, Y:512, X:1 }
  Dataflow {
    SpatialMap(1,1) K;
    TemporalMap(32,32) C;
    SpatialMap(256,256) Y;
    TemporalMap(Sz(R),Sz(R)) R;
    TemporalMap(Sz(S),Sz(S)) S;
    TemporalMap(1,1) X;
  }
}

Layer L10_K_Proj {
  Type: CONV
  Stride { X:1, Y:1 }
  Dimensions { K:1024, C:2048, R:1, S:1, Y:512, X:1 }
  Dataflow {
    SpatialMap(1,1) K;
    TemporalMap(32,32) C;
    SpatialMap(256,256) Y;
    TemporalMap(Sz(R),Sz(R)) R;
    TemporalMap(Sz(S),Sz(S)) S;
    TemporalMap(1,1) X;
  }
}

Layer L11_V_Proj {
  Type: CONV
  Stride { X:1, Y:1 }
  Dimensions { K:1024, C:2048, R:1, S:1, Y:512, X:1 }
  Dataflow {
    SpatialMap(1,1) K;
    TemporalMap(32,32) C;
    SpatialMap(256,256) Y;
    TemporalMap(Sz(R),Sz(R)) R;
    TemporalMap(Sz(S),Sz(S)) S;
    TemporalMap(1,1) X;
  }
}

Layer L12_Attn_Scores_QKt {
  Type: CONV
  Stride { X:1, Y:1 }
  Dimensions { K:512, C:128, R:1, S:1, Y:512, X:1 }
  Dataflow {
    SpatialMap(1,1) K;
    TemporalMap(8,8) C;
    TemporalMap(16,16) Y;
    TemporalMap(Sz(R),Sz(R)) R;
    TemporalMap(Sz(S),Sz(S)) S;
    TemporalMap(1,1) X;
  }
}

Layer L13_Attn_AV {
  Type: CONV
  Stride { X:1, Y:1 }
  Dimensions { K:128, C:512, R:1, S:1, Y:512, X:1 }
  Dataflow {
    SpatialMap(1,1) K;
    TemporalMap(16,16) C;
    TemporalMap(8,8) Y;
    TemporalMap(Sz(R),Sz(R)) R;
    TemporalMap(Sz(S),Sz(S)) S;
    TemporalMap(1,1) X;
  }
}

Layer L14_Out_Proj {
  Type: CONV
  Stride { X:1, Y:1 }
  Dimensions { K:2048, C:2048, R:1, S:1, Y:512, X:1 }
  Dataflow {
    SpatialMap(1,1) K;
    TemporalMap(32,32) C;
    SpatialMap(256,256) Y;
    TemporalMap(Sz(R),Sz(R)) R;
    TemporalMap(Sz(S),Sz(S)) S;
    TemporalMap(1,1) X;
  }
}

Layer L15_FFN_1 {
  Type: CONV
  Stride { X:1, Y:1 }
  Dimensions { K:6144, C:2048, R:1, S:1, Y:512, X:1 }
  Dataflow {
    SpatialMap(1,1) K;
    TemporalMap(32,32) C;
    SpatialMap(256,256) Y;
    TemporalMap(Sz(R),Sz(R)) R;
    TemporalMap(Sz(S),Sz(S)) S;
    TemporalMap(1,1) X;
  }
}

Layer L16_FFN_2 {
  Type: CONV
  Stride { X:1, Y:1 }
  Dimensions { K:2048, C:6144, R:1, S:1, Y:512, X:1 }
  Dataflow {
    SpatialMap(1,1) K;
    TemporalMap(32,32) C;
    SpatialMap(256,256) Y;
    TemporalMap(Sz(R),Sz(R)) R;
    TemporalMap(Sz(S),Sz(S)) S;
    TemporalMap(1,1) X;
  }
}

Layer L17_Q_Proj {
  Type: CONV
  Stride { X:1, Y:1 }
  Dimensions { K:2048, C:2048, R:1, S:1, Y:512, X:1 }
  Dataflow {
    SpatialMap(1,1) K;
    TemporalMap(32,32) C;
    SpatialMap(256,256) Y;
    TemporalMap(Sz(R),Sz(R)) R;
    TemporalMap(Sz(S),Sz(S)) S;
    TemporalMap(1,1) X;
  }
}

Layer L18_K_Proj {
  Type: CONV
  Stride { X:1, Y:1 }
  Dimensions { K:1024, C:2048, R:1, S:1, Y:512, X:1 }
  Dataflow {
    SpatialMap(1,1) K;
    TemporalMap(32,32) C;
    SpatialMap(256,256) Y;
    TemporalMap(Sz(R),Sz(R)) R;
    TemporalMap(Sz(S),Sz(S)) S;
    TemporalMap(1,1) X;
  }
}

Layer L19_V_Proj {
  Type: CONV
  Stride { X:1, Y:1 }
  Dimensions { K:1024, C:2048, R:1, S:1, Y:512, X:1 }
  Dataflow {
    SpatialMap(1,1) K;
    TemporalMap(32,32) C;
    SpatialMap(256,256) Y;
    TemporalMap(Sz(R),Sz(R)) R;
    TemporalMap(Sz(S),Sz(S)) S;
    TemporalMap(1,1) X;
  }
}

Layer L20_Attn_Scores_QKt {
  Type: CONV
  Stride { X:1, Y:1 }
  Dimensions { K:512, C:128, R:1, S:1, Y:512, X:1 }
  Dataflow {
    SpatialMap(1,1) K;
    TemporalMap(8,8) C;
    TemporalMap(16,16) Y;
    TemporalMap(Sz(R),Sz(R)) R;
    TemporalMap(Sz(S),Sz(S)) S;
    TemporalMap(1,1) X;
  }
}

Layer L21_Attn_AV {
  Type: CONV
  Stride { X:1, Y:1 }
  Dimensions { K:128, C:512, R:1, S:1, Y:512, X:1 }
  Dataflow {
    SpatialMap(1,1) K;
    TemporalMap(16,16) C;
    TemporalMap(8,8) Y;
    TemporalMap(Sz(R),Sz(R)) R;
    TemporalMap(Sz(S),Sz(S)) S;
    TemporalMap(1,1) X;
  }
}

Layer L22_Out_Proj {
  Type: CONV
  Stride { X:1, Y:1 }
  Dimensions { K:2048, C:2048, R:1, S:1, Y:512, X:1 }
  Dataflow {
    SpatialMap(1,1) K;
    TemporalMap(32,32) C;
    SpatialMap(256,256) Y;
    TemporalMap(Sz(R),Sz(R)) R;
    TemporalMap(Sz(S),Sz(S)) S;
    TemporalMap(1,1) X;
  }
}

Layer L23_FFN_1 {
  Type: CONV
  Stride { X:1, Y:1 }
  Dimensions { K:6144, C:2048, R:1, S:1, Y:512, X:1 }
  Dataflow {
    SpatialMap(1,1) K;
    TemporalMap(32,32) C;
    SpatialMap(256,256) Y;
    TemporalMap(Sz(R),Sz(R)) R;
    TemporalMap(Sz(S),Sz(S)) S;
    TemporalMap(1,1) X;
  }
}

Layer L24_FFN_2 {
  Type: CONV
  Stride { X:1, Y:1 }
  Dimensions { K:2048, C:6144, R:1, S:1, Y:512, X:1 }
  Dataflow {
    SpatialMap(1,1) K;
    TemporalMap(32,32) C;
    SpatialMap(256,256) Y;
    TemporalMap(Sz(R),Sz(R)) R;
    TemporalMap(Sz(S),Sz(S)) S;
    TemporalMap(1,1) X;
  }
}

Layer L25_Q_Proj {
  Type: CONV
  Stride { X:1, Y:1 }
  Dimensions { K:2048, C:2048, R:1, S:1, Y:512, X:1 }
  Dataflow {
    SpatialMap(1,1) K;
    TemporalMap(32,32) C;
    SpatialMap(256,256) Y;
    TemporalMap(Sz(R),Sz(R)) R;
    TemporalMap(Sz(S),Sz(S)) S;
    TemporalMap(1,1) X;
  }
}

Layer L26_K_Proj {
  Type: CONV
  Stride { X:1, Y:1 }
  Dimensions { K:1024, C:2048, R:1, S:1, Y:512, X:1 }
  Dataflow {
    SpatialMap(1,1) K;
    TemporalMap(32,32) C;
    SpatialMap(256,256) Y;
    TemporalMap(Sz(R),Sz(R)) R;
    TemporalMap(Sz(S),Sz(S)) S;
    TemporalMap(1,1) X;
  }
}

Layer L27_V_Proj {
  Type: CONV
  Stride { X:1, Y:1 }
  Dimensions { K:1024, C:2048, R:1, S:1, Y:512, X:1 }
  Dataflow {
    SpatialMap(1,1) K;
    TemporalMap(32,32) C;
    SpatialMap(256,256) Y;
    TemporalMap(Sz(R),Sz(R)) R;
    TemporalMap(Sz(S),Sz(S)) S;
    TemporalMap(1,1) X;
  }
}

Layer L28_Attn_Scores_QKt {
  Type: CONV
  Stride { X:1, Y:1 }
  Dimensions { K:512, C:128, R:1, S:1, Y:512, X:1 }
  Dataflow {
    SpatialMap(1,1) K;
    TemporalMap(8,8) C;
    TemporalMap(16,16) Y;
    TemporalMap(Sz(R),Sz(R)) R;
    TemporalMap(Sz(S),Sz(S)) S;
    TemporalMap(1,1) X;
  }
}

Layer L29_Attn_AV {
  Type: CONV
  Stride { X:1, Y:1 }
  Dimensions { K:128, C:512, R:1, S:1, Y:512, X:1 }
  Dataflow {
    SpatialMap(1,1) K;
    TemporalMap(16,16) C;
    TemporalMap(8,8) Y;
    TemporalMap(Sz(R),Sz(R)) R;
    TemporalMap(Sz(S),Sz(S)) S;
    TemporalMap(1,1) X;
  }
}

Layer L30_Out_Proj {
  Type: CONV
  Stride { X:1, Y:1 }
  Dimensions { K:2048, C:2048, R:1, S:1, Y:512, X:1 }
  Dataflow {
    SpatialMap(1,1) K;
    TemporalMap(32,32) C;
    SpatialMap(256,256) Y;
    TemporalMap(Sz(R),Sz(R)) R;
    TemporalMap(Sz(S),Sz(S)) S;
    TemporalMap(1,1) X;
  }
}

Layer L31_FFN_1 {
  Type: CONV
  Stride { X:1, Y:1 }
  Dimensions { K:6144, C:2048, R:1, S:1, Y:512, X:1 }
  Dataflow {
    SpatialMap(1,1) K;
    TemporalMap(32,32) C;
    SpatialMap(256,256) Y;
    TemporalMap(Sz(R),Sz(R)) R;
    TemporalMap(Sz(S),Sz(S)) S;
    TemporalMap(1,1) X;
  }
}

Layer L32_FFN_2 {
  Type: CONV
  Stride { X:1, Y:1 }
  Dimensions { K:2048, C:6144, R:1, S:1, Y:512, X:1 }
  Dataflow {
    SpatialMap(1,1) K;
    TemporalMap(32,32) C;
    SpatialMap(256,256) Y;
    TemporalMap(Sz(R),Sz(R)) R;
    TemporalMap(Sz(S),Sz(S)) S;
    TemporalMap(1,1) X;
  }
}

Layer L33_Q_Proj {
  Type: CONV
  Stride { X:1, Y:1 }
  Dimensions { K:2048, C:2048, R:1, S:1, Y:512, X:1 }
  Dataflow {
    SpatialMap(1,1) K;
    TemporalMap(32,32) C;
    SpatialMap(256,256) Y;
    TemporalMap(Sz(R),Sz(R)) R;
    TemporalMap(Sz(S),Sz(S)) S;
    TemporalMap(1,1) X;
  }
}

Layer L34_K_Proj {
  Type: CONV
  Stride { X:1, Y:1 }
  Dimensions { K:1024, C:2048, R:1, S:1, Y:512, X:1 }
  Dataflow {
    SpatialMap(1,1) K;
    TemporalMap(32,32) C;
    SpatialMap(256,256) Y;
    TemporalMap(Sz(R),Sz(R)) R;
    TemporalMap(Sz(S),Sz(S)) S;
    TemporalMap(1,1) X;
  }
}

Layer L35_V_Proj {
  Type: CONV
  Stride { X:1, Y:1 }
  Dimensions { K:1024, C:2048, R:1, S:1, Y:512, X:1 }
  Dataflow {
    SpatialMap(1,1) K;
    TemporalMap(32,32) C;
    SpatialMap(256,256) Y;
    TemporalMap(Sz(R),Sz(R)) R;
    TemporalMap(Sz(S),Sz(S)) S;
    TemporalMap(1,1) X;
  }
}

Layer L36_Attn_Scores_QKt {
  Type: CONV
  Stride { X:1, Y:1 }
  Dimensions { K:512, C:128, R:1, S:1, Y:512, X:1 }
  Dataflow {
    SpatialMap(1,1) K;
    TemporalMap(8,8) C;
    TemporalMap(16,16) Y;
    TemporalMap(Sz(R),Sz(R)) R;
    TemporalMap(Sz(S),Sz(S)) S;
    TemporalMap(1,1) X;
  }
}

Layer L37_Attn_AV {
  Type: CONV
  Stride { X:1, Y:1 }
  Dimensions { K:128, C:512, R:1, S:1, Y:512, X:1 }
  Dataflow {
    SpatialMap(1,1) K;
    TemporalMap(16,16) C;
    TemporalMap(8,8) Y;
    TemporalMap(Sz(R),Sz(R)) R;
    TemporalMap(Sz(S),Sz(S)) S;
    TemporalMap(1,1) X;
  }
}

Layer L38_Out_Proj {
  Type: CONV
  Stride { X:1, Y:1 }
  Dimensions { K:2048, C:2048, R:1, S:1, Y:512, X:1 }
  Dataflow {
    SpatialMap(1,1) K;
    TemporalMap(32,32) C;
    SpatialMap(256,256) Y;
    TemporalMap(Sz(R),Sz(R)) R;
    TemporalMap(Sz(S),Sz(S)) S;
    TemporalMap(1,1) X;
  }
}

Layer L39_FFN_1 {
  Type: CONV
  Stride { X:1, Y:1 }
  Dimensions { K:6144, C:2048, R:1, S:1, Y:512, X:1 }
  Dataflow {
    SpatialMap(1,1) K;
    TemporalMap(32,32) C;
    SpatialMap(256,256) Y;
    TemporalMap(Sz(R),Sz(R)) R;
    TemporalMap(Sz(S),Sz(S)) S;
    TemporalMap(1,1) X;
  }
}

Layer L40_FFN_2 {
  Type: CONV
  Stride { X:1, Y:1 }
  Dimensions { K:2048, C:6144, R:1, S:1, Y:512, X:1 }
  Dataflow {
    SpatialMap(1,1) K;
    TemporalMap(32,32) C;
    SpatialMap(256,256) Y;
    TemporalMap(Sz(R),Sz(R)) R;
    TemporalMap(Sz(S),Sz(S)) S;
    TemporalMap(1,1) X;
  }
}

Layer L41_Q_Proj {
  Type: CONV
  Stride { X:1, Y:1 }
  Dimensions { K:2048, C:2048, R:1, S:1, Y:512, X:1 }
  Dataflow {
    SpatialMap(1,1) K;
    TemporalMap(32,32) C;
    SpatialMap(256,256) Y;
    TemporalMap(Sz(R),Sz(R)) R;
    TemporalMap(Sz(S),Sz(S)) S;
    TemporalMap(1,1) X;
  }
}

Layer L42_K_Proj {
  Type: CONV
  Stride { X:1, Y:1 }
  Dimensions { K:1024, C:2048, R:1, S:1, Y:512, X:1 }
  Dataflow {
    SpatialMap(1,1) K;
    TemporalMap(32,32) C;
    SpatialMap(256,256) Y;
    TemporalMap(Sz(R),Sz(R)) R;
    TemporalMap(Sz(S),Sz(S)) S;
    TemporalMap(1,1) X;
  }
}

Layer L43_V_Proj {
  Type: CONV
  Stride { X:1, Y:1 }
  Dimensions { K:1024, C:2048, R:1, S:1, Y:512, X:1 }
  Dataflow {
    SpatialMap(1,1) K;
    TemporalMap(32,32) C;
    SpatialMap(256,256) Y;
    TemporalMap(Sz(R),Sz(R)) R;
    TemporalMap(Sz(S),Sz(S)) S;
    TemporalMap(1,1) X;
  }
}

Layer L44_Attn_Scores_QKt {
  Type: CONV
  Stride { X:1, Y:1 }
  Dimensions { K:512, C:128, R:1, S:1, Y:512, X:1 }
  Dataflow {
    SpatialMap(1,1) K;
    TemporalMap(8,8) C;
    TemporalMap(16,16) Y;
    TemporalMap(Sz(R),Sz(R)) R;
    TemporalMap(Sz(S),Sz(S)) S;
    TemporalMap(1,1) X;
  }
}

Layer L45_Attn_AV {
  Type: CONV
  Stride { X:1, Y:1 }
  Dimensions { K:128, C:512, R:1, S:1, Y:512, X:1 }
  Dataflow {
    SpatialMap(1,1) K;
    TemporalMap(16,16) C;
    TemporalMap(8,8) Y;
    TemporalMap(Sz(R),Sz(R)) R;
    TemporalMap(Sz(S),Sz(S)) S;
    TemporalMap(1,1) X;
  }
}

Layer L46_Out_Proj {
  Type: CONV
  Stride { X:1, Y:1 }
  Dimensions { K:2048, C:2048, R:1, S:1, Y:512, X:1 }
  Dataflow {
    SpatialMap(1,1) K;
    TemporalMap(32,32) C;
    SpatialMap(256,256) Y;
    TemporalMap(Sz(R),Sz(R)) R;
    TemporalMap(Sz(S),Sz(S)) S;
    TemporalMap(1,1) X;
  }
}

Layer L47_FFN_1 {
  Type: CONV
  Stride { X:1, Y:1 }
  Dimensions { K:6144, C:2048, R:1, S:1, Y:512, X:1 }
  Dataflow {
    SpatialMap(1,1) K;
    TemporalMap(32,32) C;
    SpatialMap(256,256) Y;
    TemporalMap(Sz(R),Sz(R)) R;
    TemporalMap(Sz(S),Sz(S)) S;
    TemporalMap(1,1) X;
  }
}

Layer L48_FFN_2 {
  Type: CONV
  Stride { X:1, Y:1 }
  Dimensions { K:2048, C:6144, R:1, S:1, Y:512, X:1 }
  Dataflow {
    SpatialMap(1,1) K;
    TemporalMap(32,32) C;
    SpatialMap(256,256) Y;
    TemporalMap(Sz(R),Sz(R)) R;
    TemporalMap(Sz(S),Sz(S)) S;
    TemporalMap(1,1) X;
  }
}

Layer L49_Q_Proj {
  Type: CONV
  Stride { X:1, Y:1 }
  Dimensions { K:2048, C:2048, R:1, S:1, Y:512, X:1 }
  Dataflow {
    SpatialMap(1,1) K;
    TemporalMap(32,32) C;
    SpatialMap(256,256) Y;
    TemporalMap(Sz(R),Sz(R)) R;
    TemporalMap(Sz(S),Sz(S)) S;
    TemporalMap(1,1) X;
  }
}

Layer L50_K_Proj {
  Type: CONV
  Stride { X:1, Y:1 }
  Dimensions { K:1024, C:2048, R:1, S:1, Y:512, X:1 }
  Dataflow {
    SpatialMap(1,1) K;
    TemporalMap(32,32) C;
    SpatialMap(256,256) Y;
    TemporalMap(Sz(R),Sz(R)) R;
    TemporalMap(Sz(S),Sz(S)) S;
    TemporalMap(1,1) X;
  }
}

Layer L51_V_Proj {
  Type: CONV
  Stride { X:1, Y:1 }
  Dimensions { K:1024, C:2048, R:1, S:1, Y:512, X:1 }
  Dataflow {
    SpatialMap(1,1) K;
    TemporalMap(32,32) C;
    SpatialMap(256,256) Y;
    TemporalMap(Sz(R),Sz(R)) R;
    TemporalMap(Sz(S),Sz(S)) S;
    TemporalMap(1,1) X;
  }
}

Layer L52_Attn_Scores_QKt {
  Type: CONV
  Stride { X:1, Y:1 }
  Dimensions { K:512, C:128, R:1, S:1, Y:512, X:1 }
  Dataflow {
    SpatialMap(1,1) K;
    TemporalMap(8,8) C;
    TemporalMap(16,16) Y;
    TemporalMap(Sz(R),Sz(R)) R;
    TemporalMap(Sz(S),Sz(S)) S;
    TemporalMap(1,1) X;
  }
}

Layer L53_Attn_AV {
  Type: CONV
  Stride { X:1, Y:1 }
  Dimensions { K:128, C:512, R:1, S:1, Y:512, X:1 }
  Dataflow {
    SpatialMap(1,1) K;
    TemporalMap(16,16) C;
    TemporalMap(8,8) Y;
    TemporalMap(Sz(R),Sz(R)) R;
    TemporalMap(Sz(S),Sz(S)) S;
    TemporalMap(1,1) X;
  }
}

Layer L54_Out_Proj {
  Type: CONV
  Stride { X:1, Y:1 }
  Dimensions { K:2048, C:2048, R:1, S:1, Y:512, X:1 }
  Dataflow {
    SpatialMap(1,1) K;
    TemporalMap(32,32) C;
    SpatialMap(256,256) Y;
    TemporalMap(Sz(R),Sz(R)) R;
    TemporalMap(Sz(S),Sz(S)) S;
    TemporalMap(1,1) X;
  }
}

Layer L55_FFN_1 {
  Type: CONV
  Stride { X:1, Y:1 }
  Dimensions { K:6144, C:2048, R:1, S:1, Y:512, X:1 }
  Dataflow {
    SpatialMap(1,1) K;
    TemporalMap(32,32) C;
    SpatialMap(256,256) Y;
    TemporalMap(Sz(R),Sz(R)) R;
    TemporalMap(Sz(S),Sz(S)) S;
    TemporalMap(1,1) X;
  }
}

Layer L56_FFN_2 {
  Type: CONV
  Stride { X:1, Y:1 }
  Dimensions { K:2048, C:6144, R:1, S:1, Y:512, X:1 }
  Dataflow {
    SpatialMap(1,1) K;
    TemporalMap(32,32) C;
    SpatialMap(256,256) Y;
    TemporalMap(Sz(R),Sz(R)) R;
    TemporalMap(Sz(S),Sz(S)) S;
    TemporalMap(1,1) X;
  }
}

Layer L57_Q_Proj {
  Type: CONV
  Stride { X:1, Y:1 }
  Dimensions { K:2048, C:2048, R:1, S:1, Y:512, X:1 }
  Dataflow {
    SpatialMap(1,1) K;
    TemporalMap(32,32) C;
    SpatialMap(256,256) Y;
    TemporalMap(Sz(R),Sz(R)) R;
    TemporalMap(Sz(S),Sz(S)) S;
    TemporalMap(1,1) X;
  }
}

Layer L58_K_Proj {
  Type: CONV
  Stride { X:1, Y:1 }
  Dimensions { K:1024, C:2048, R:1, S:1, Y:512, X:1 }
  Dataflow {
    SpatialMap(1,1) K;
    TemporalMap(32,32) C;
    SpatialMap(256,256) Y;
    TemporalMap(Sz(R),Sz(R)) R;
    TemporalMap(Sz(S),Sz(S)) S;
    TemporalMap(1,1) X;
  }
}

Layer L59_V_Proj {
  Type: CONV
  Stride { X:1, Y:1 }
  Dimensions { K:1024, C:2048, R:1, S:1, Y:512, X:1 }
  Dataflow {
    SpatialMap(1,1) K;
    TemporalMap(32,32) C;
    SpatialMap(256,256) Y;
    TemporalMap(Sz(R),Sz(R)) R;
    TemporalMap(Sz(S),Sz(S)) S;
    TemporalMap(1,1) X;
  }
}

Layer L60_Attn_Scores_QKt {
  Type: CONV
  Stride { X:1, Y:1 }
  Dimensions { K:512, C:128, R:1, S:1, Y:512, X:1 }
  Dataflow {
    SpatialMap(1,1) K;
    TemporalMap(8,8) C;
    TemporalMap(16,16) Y;
    TemporalMap(Sz(R),Sz(R)) R;
    TemporalMap(Sz(S),Sz(S)) S;
    TemporalMap(1,1) X;
  }
}

Layer L61_Attn_AV {
  Type: CONV
  Stride { X:1, Y:1 }
  Dimensions { K:128, C:512, R:1, S:1, Y:512, X:1 }
  Dataflow {
    SpatialMap(1,1) K;
    TemporalMap(16,16) C;
    TemporalMap(8,8) Y;
    TemporalMap(Sz(R),Sz(R)) R;
    TemporalMap(Sz(S),Sz(S)) S;
    TemporalMap(1,1) X;
  }
}

Layer L62_Out_Proj {
  Type: CONV
  Stride { X:1, Y:1 }
  Dimensions { K:2048, C:2048, R:1, S:1, Y:512, X:1 }
  Dataflow {
    SpatialMap(1,1) K;
    TemporalMap(32,32) C;
    SpatialMap(256,256) Y;
    TemporalMap(Sz(R),Sz(R)) R;
    TemporalMap(Sz(S),Sz(S)) S;
    TemporalMap(1,1) X;
  }
}

Layer L63_FFN_1 {
  Type: CONV
  Stride { X:1, Y:1 }
  Dimensions { K:6144, C:2048, R:1, S:1, Y:512, X:1 }
  Dataflow {
    SpatialMap(1,1) K;
    TemporalMap(32,32) C;
    SpatialMap(256,256) Y;
    TemporalMap(Sz(R),Sz(R)) R;
    TemporalMap(Sz(S),Sz(S)) S;
    TemporalMap(1,1) X;
  }
}

Layer L64_FFN_2 {
  Type: CONV
  Stride { X:1, Y:1 }
  Dimensions { K:2048, C:6144, R:1, S:1, Y:512, X:1 }
  Dataflow {
    SpatialMap(1,1) K;
    TemporalMap(32,32) C;
    SpatialMap(256,256) Y;
    TemporalMap(Sz(R),Sz(R)) R;
    TemporalMap(Sz(S),Sz(S)) S;
    TemporalMap(1,1) X;
  }
}

Layer L65_Q_Proj {
  Type: CONV
  Stride { X:1, Y:1 }
  Dimensions { K:2048, C:2048, R:1, S:1, Y:512, X:1 }
  Dataflow {
    SpatialMap(1,1) K;
    TemporalMap(32,32) C;
    SpatialMap(256,256) Y;
    TemporalMap(Sz(R),Sz(R)) R;
    TemporalMap(Sz(S),Sz(S)) S;
    TemporalMap(1,1) X;
  }
}

Layer L66_K_Proj {
  Type: CONV
  Stride { X:1, Y:1 }
  Dimensions { K:1024, C:2048, R:1, S:1, Y:512, X:1 }
  Dataflow {
    SpatialMap(1,1) K;
    TemporalMap(32,32) C;
    SpatialMap(256,256) Y;
    TemporalMap(Sz(R),Sz(R)) R;
    TemporalMap(Sz(S),Sz(S)) S;
    TemporalMap(1,1) X;
  }
}

Layer L67_V_Proj {
  Type: CONV
  Stride { X:1, Y:1 }
  Dimensions { K:1024, C:2048, R:1, S:1, Y:512, X:1 }
  Dataflow {
    SpatialMap(1,1) K;
    TemporalMap(32,32) C;
    SpatialMap(256,256) Y;
    TemporalMap(Sz(R),Sz(R)) R;
    TemporalMap(Sz(S),Sz(S)) S;
    TemporalMap(1,1) X;
  }
}

Layer L68_Attn_Scores_QKt {
  Type: CONV
  Stride { X:1, Y:1 }
  Dimensions { K:512, C:128, R:1, S:1, Y:512, X:1 }
  Dataflow {
    SpatialMap(1,1) K;
    TemporalMap(8,8) C;
    TemporalMap(16,16) Y;
    TemporalMap(Sz(R),Sz(R)) R;
    TemporalMap(Sz(S),Sz(S)) S;
    TemporalMap(1,1) X;
  }
}

Layer L69_Attn_AV {
  Type: CONV
  Stride { X:1, Y:1 }
  Dimensions { K:128, C:512, R:1, S:1, Y:512, X:1 }
  Dataflow {
    SpatialMap(1,1) K;
    TemporalMap(16,16) C;
    TemporalMap(8,8) Y;
    TemporalMap(Sz(R),Sz(R)) R;
    TemporalMap(Sz(S),Sz(S)) S;
    TemporalMap(1,1) X;
  }
}

Layer L70_Out_Proj {
  Type: CONV
  Stride { X:1, Y:1 }
  Dimensions { K:2048, C:2048, R:1, S:1, Y:512, X:1 }
  Dataflow {
    SpatialMap(1,1) K;
    TemporalMap(32,32) C;
    SpatialMap(256,256) Y;
    TemporalMap(Sz(R),Sz(R)) R;
    TemporalMap(Sz(S),Sz(S)) S;
    TemporalMap(1,1) X;
  }
}

Layer L71_FFN_1 {
  Type: CONV
  Stride { X:1, Y:1 }
  Dimensions { K:6144, C:2048, R:1, S:1, Y:512, X:1 }
  Dataflow {
    SpatialMap(1,1) K;
    TemporalMap(32,32) C;
    SpatialMap(256,256) Y;
    TemporalMap(Sz(R),Sz(R)) R;
    TemporalMap(Sz(S),Sz(S)) S;
    TemporalMap(1,1) X;
  }
}

Layer L72_FFN_2 {
  Type: CONV
  Stride { X:1, Y:1 }
  Dimensions { K:2048, C:6144, R:1, S:1, Y:512, X:1 }
  Dataflow {
    SpatialMap(1,1) K;
    TemporalMap(32,32) C;
    SpatialMap(256,256) Y;
    TemporalMap(Sz(R),Sz(R)) R;
    TemporalMap(Sz(S),Sz(S)) S;
    TemporalMap(1,1) X;
  }
}

Layer L73_Q_Proj {
  Type: CONV
  Stride { X:1, Y:1 }
  Dimensions { K:2048, C:2048, R:1, S:1, Y:512, X:1 }
  Dataflow {
    SpatialMap(1,1) K;
    TemporalMap(32,32) C;
    SpatialMap(256,256) Y;
    TemporalMap(Sz(R),Sz(R)) R;
    TemporalMap(Sz(S),Sz(S)) S;
    TemporalMap(1,1) X;
  }
}

Layer L74_K_Proj {
  Type: CONV
  Stride { X:1, Y:1 }
  Dimensions { K:1024, C:2048, R:1, S:1, Y:512, X:1 }
  Dataflow {
    SpatialMap(1,1) K;
    TemporalMap(32,32) C;
    SpatialMap(256,256) Y;
    TemporalMap(Sz(R),Sz(R)) R;
    TemporalMap(Sz(S),Sz(S)) S;
    TemporalMap(1,1) X;
  }
}

Layer L75_V_Proj {
  Type: CONV
  Stride { X:1, Y:1 }
  Dimensions { K:1024, C:2048, R:1, S:1, Y:512, X:1 }
  Dataflow {
    SpatialMap(1,1) K;
    TemporalMap(32,32) C;
    SpatialMap(256,256) Y;
    TemporalMap(Sz(R),Sz(R)) R;
    TemporalMap(Sz(S),Sz(S)) S;
    TemporalMap(1,1) X;
  }
}

Layer L76_Attn_Scores_QKt {
  Type: CONV
  Stride { X:1, Y:1 }
  Dimensions { K:512, C:128, R:1, S:1, Y:512, X:1 }
  Dataflow {
    SpatialMap(1,1) K;
    TemporalMap(8,8) C;
    TemporalMap(16,16) Y;
    TemporalMap(Sz(R),Sz(R)) R;
    TemporalMap(Sz(S),Sz(S)) S;
    TemporalMap(1,1) X;
  }
}

Layer L77_Attn_AV {
  Type: CONV
  Stride { X:1, Y:1 }
  Dimensions { K:128, C:512, R:1, S:1, Y:512, X:1 }
  Dataflow {
    SpatialMap(1,1) K;
    TemporalMap(16,16) C;
    TemporalMap(8,8) Y;
    TemporalMap(Sz(R),Sz(R)) R;
    TemporalMap(Sz(S),Sz(S)) S;
    TemporalMap(1,1) X;
  }
}

Layer L78_Out_Proj {
  Type: CONV
  Stride { X:1, Y:1 }
  Dimensions { K:2048, C:2048, R:1, S:1, Y:512, X:1 }
  Dataflow {
    SpatialMap(1,1) K;
    TemporalMap(32,32) C;
    SpatialMap(256,256) Y;
    TemporalMap(Sz(R),Sz(R)) R;
    TemporalMap(Sz(S),Sz(S)) S;
    TemporalMap(1,1) X;
  }
}

Layer L79_FFN_1 {
  Type: CONV
  Stride { X:1, Y:1 }
  Dimensions { K:6144, C:2048, R:1, S:1, Y:512, X:1 }
  Dataflow {
    SpatialMap(1,1) K;
    TemporalMap(32,32) C;
    SpatialMap(256,256) Y;
    TemporalMap(Sz(R),Sz(R)) R;
    TemporalMap(Sz(S),Sz(S)) S;
    TemporalMap(1,1) X;
  }
}

Layer L80_FFN_2 {
  Type: CONV
  Stride { X:1, Y:1 }
  Dimensions { K:2048, C:6144, R:1, S:1, Y:512, X:1 }
  Dataflow {
    SpatialMap(1,1) K;
    TemporalMap(32,32) C;
    SpatialMap(256,256) Y;
    TemporalMap(Sz(R),Sz(R)) R;
    TemporalMap(Sz(S),Sz(S)) S;
    TemporalMap(1,1) X;
  }
}

Layer L81_Q_Proj {
  Type: CONV
  Stride { X:1, Y:1 }
  Dimensions { K:2048, C:2048, R:1, S:1, Y:512, X:1 }
  Dataflow {
    SpatialMap(1,1) K;
    TemporalMap(32,32) C;
    SpatialMap(256,256) Y;
    TemporalMap(Sz(R),Sz(R)) R;
    TemporalMap(Sz(S),Sz(S)) S;
    TemporalMap(1,1) X;
  }
}

Layer L82_K_Proj {
  Type: CONV
  Stride { X:1, Y:1 }
  Dimensions { K:1024, C:2048, R:1, S:1, Y:512, X:1 }
  Dataflow {
    SpatialMap(1,1) K;
    TemporalMap(32,32) C;
    SpatialMap(256,256) Y;
    TemporalMap(Sz(R),Sz(R)) R;
    TemporalMap(Sz(S),Sz(S)) S;
    TemporalMap(1,1) X;
  }
}

Layer L83_V_Proj {
  Type: CONV
  Stride { X:1, Y:1 }
  Dimensions { K:1024, C:2048, R:1, S:1, Y:512, X:1 }
  Dataflow {
    SpatialMap(1,1) K;
    TemporalMap(32,32) C;
    SpatialMap(256,256) Y;
    TemporalMap(Sz(R),Sz(R)) R;
    TemporalMap(Sz(S),Sz(S)) S;
    TemporalMap(1,1) X;
  }
}

Layer L84_Attn_Scores_QKt {
  Type: CONV
  Stride { X:1, Y:1 }
  Dimensions { K:512, C:128, R:1, S:1, Y:512, X:1 }
  Dataflow {
    SpatialMap(1,1) K;
    TemporalMap(8,8) C;
    TemporalMap(16,16) Y;
    TemporalMap(Sz(R),Sz(R)) R;
    TemporalMap(Sz(S),Sz(S)) S;
    TemporalMap(1,1) X;
  }
}

Layer L85_Attn_AV {
  Type: CONV
  Stride { X:1, Y:1 }
  Dimensions { K:128, C:512, R:1, S:1, Y:512, X:1 }
  Dataflow {
    SpatialMap(1,1) K;
    TemporalMap(16,16) C;
    TemporalMap(8,8) Y;
    TemporalMap(Sz(R),Sz(R)) R;
    TemporalMap(Sz(S),Sz(S)) S;
    TemporalMap(1,1) X;
  }
}

Layer L86_Out_Proj {
  Type: CONV
  Stride { X:1, Y:1 }
  Dimensions { K:2048, C:2048, R:1, S:1, Y:512, X:1 }
  Dataflow {
    SpatialMap(1,1) K;
    TemporalMap(32,32) C;
    SpatialMap(256,256) Y;
    TemporalMap(Sz(R),Sz(R)) R;
    TemporalMap(Sz(S),Sz(S)) S;
    TemporalMap(1,1) X;
  }
}

Layer L87_FFN_1 {
  Type: CONV
  Stride { X:1, Y:1 }
  Dimensions { K:6144, C:2048, R:1, S:1, Y:512, X:1 }
  Dataflow {
    SpatialMap(1,1) K;
    TemporalMap(32,32) C;
    SpatialMap(256,256) Y;
    TemporalMap(Sz(R),Sz(R)) R;
    TemporalMap(Sz(S),Sz(S)) S;
    TemporalMap(1,1) X;
  }
}

Layer L88_FFN_2 {
  Type: CONV
  Stride { X:1, Y:1 }
  Dimensions { K:2048, C:6144, R:1, S:1, Y:512, X:1 }
  Dataflow {
    SpatialMap(1,1) K;
    TemporalMap(32,32) C;
    SpatialMap(256,256) Y;
    TemporalMap(Sz(R),Sz(R)) R;
    TemporalMap(Sz(S),Sz(S)) S;
    TemporalMap(1,1) X;
  }
}

Layer L89_Q_Proj {
  Type: CONV
  Stride { X:1, Y:1 }
  Dimensions { K:2048, C:2048, R:1, S:1, Y:512, X:1 }
  Dataflow {
    SpatialMap(1,1) K;
    TemporalMap(32,32) C;
    SpatialMap(256,256) Y;
    TemporalMap(Sz(R),Sz(R)) R;
    TemporalMap(Sz(S),Sz(S)) S;
    TemporalMap(1,1) X;
  }
}

Layer L90_K_Proj {
  Type: CONV
  Stride { X:1, Y:1 }
  Dimensions { K:1024, C:2048, R:1, S:1, Y:512, X:1 }
  Dataflow {
    SpatialMap(1,1) K;
    TemporalMap(32,32) C;
    SpatialMap(256,256) Y;
    TemporalMap(Sz(R),Sz(R)) R;
    TemporalMap(Sz(S),Sz(S)) S;
    TemporalMap(1,1) X;
  }
}

Layer L91_V_Proj {
  Type: CONV
  Stride { X:1, Y:1 }
  Dimensions { K:1024, C:2048, R:1, S:1, Y:512, X:1 }
  Dataflow {
    SpatialMap(1,1) K;
    TemporalMap(32,32) C;
    SpatialMap(256,256) Y;
    TemporalMap(Sz(R),Sz(R)) R;
    TemporalMap(Sz(S),Sz(S)) S;
    TemporalMap(1,1) X;
  }
}

Layer L92_Attn_Scores_QKt {
  Type: CONV
  Stride { X:1, Y:1 }
  Dimensions { K:512, C:128, R:1, S:1, Y:512, X:1 }
  Dataflow {
    SpatialMap(1,1) K;
    TemporalMap(8,8) C;
    TemporalMap(16,16) Y;
    TemporalMap(Sz(R),Sz(R)) R;
    TemporalMap(Sz(S),Sz(S)) S;
    TemporalMap(1,1) X;
  }
}

Layer L93_Attn_AV {
  Type: CONV
  Stride { X:1, Y:1 }
  Dimensions { K:128, C:512, R:1, S:1, Y:512, X:1 }
  Dataflow {
    SpatialMap(1,1) K;
    TemporalMap(16,16) C;
    TemporalMap(8,8) Y;
    TemporalMap(Sz(R),Sz(R)) R;
    TemporalMap(Sz(S),Sz(S)) S;
    TemporalMap(1,1) X;
  }
}

Layer L94_Out_Proj {
  Type: CONV
  Stride { X:1, Y:1 }
  Dimensions { K:2048, C:2048, R:1, S:1, Y:512, X:1 }
  Dataflow {
    SpatialMap(1,1) K;
    TemporalMap(32,32) C;
    SpatialMap(256,256) Y;
    TemporalMap(Sz(R),Sz(R)) R;
    TemporalMap(Sz(S),Sz(S)) S;
    TemporalMap(1,1) X;
  }
}

Layer L95_FFN_1 {
  Type: CONV
  Stride { X:1, Y:1 }
  Dimensions { K:6144, C:2048, R:1, S:1, Y:512, X:1 }
  Dataflow {
    SpatialMap(1,1) K;
    TemporalMap(32,32) C;
    SpatialMap(256,256) Y;
    TemporalMap(Sz(R),Sz(R)) R;
    TemporalMap(Sz(S),Sz(S)) S;
    TemporalMap(1,1) X;
  }
}

Layer L96_FFN_2 {
  Type: CONV
  Stride { X:1, Y:1 }
  Dimensions { K:2048, C:6144, R:1, S:1, Y:512, X:1 }
  Dataflow {
    SpatialMap(1,1) K;
    TemporalMap(32,32) C;
    SpatialMap(256,256) Y;
    TemporalMap(Sz(R),Sz(R)) R;
    TemporalMap(Sz(S),Sz(S)) S;
    TemporalMap(1,1) X;
  }
}

Layer L97_Q_Proj {
  Type: CONV
  Stride { X:1, Y:1 }
  Dimensions { K:2048, C:2048, R:1, S:1, Y:512, X:1 }
  Dataflow {
    SpatialMap(1,1) K;
    TemporalMap(32,32) C;
    SpatialMap(256,256) Y;
    TemporalMap(Sz(R),Sz(R)) R;
    TemporalMap(Sz(S),Sz(S)) S;
    TemporalMap(1,1) X;
  }
}

Layer L98_K_Proj {
  Type: CONV
  Stride { X:1, Y:1 }
  Dimensions { K:1024, C:2048, R:1, S:1, Y:512, X:1 }
  Dataflow {
    SpatialMap(1,1) K;
    TemporalMap(32,32) C;
    SpatialMap(256,256) Y;
    TemporalMap(Sz(R),Sz(R)) R;
    TemporalMap(Sz(S),Sz(S)) S;
    TemporalMap(1,1) X;
  }
}

Layer L99_V_Proj {
  Type: CONV
  Stride { X:1, Y:1 }
  Dimensions { K:1024, C:2048, R:1, S:1, Y:512, X:1 }
  Dataflow {
    SpatialMap(1,1) K;
    TemporalMap(32,32) C;
    SpatialMap(256,256) Y;
    TemporalMap(Sz(R),Sz(R)) R;
    TemporalMap(Sz(S),Sz(S)) S;
    TemporalMap(1,1) X;
  }
}

Layer L100_Attn_Scores_QKt {
  Type: CONV
  Stride { X:1, Y:1 }
  Dimensions { K:512, C:128, R:1, S:1, Y:512, X:1 }
  Dataflow {
    SpatialMap(1,1) K;
    TemporalMap(8,8) C;
    TemporalMap(16,16) Y;
    TemporalMap(Sz(R),Sz(R)) R;
    TemporalMap(Sz(S),Sz(S)) S;
    TemporalMap(1,1) X;
  }
}

Layer L101_Attn_AV {
  Type: CONV
  Stride { X:1, Y:1 }
  Dimensions { K:128, C:512, R:1, S:1, Y:512, X:1 }
  Dataflow {
    SpatialMap(1,1) K;
    TemporalMap(16,16) C;
    TemporalMap(8,8) Y;
    TemporalMap(Sz(R),Sz(R)) R;
    TemporalMap(Sz(S),Sz(S)) S;
    TemporalMap(1,1) X;
  }
}

Layer L102_Out_Proj {
  Type: CONV
  Stride { X:1, Y:1 }
  Dimensions { K:2048, C:2048, R:1, S:1, Y:512, X:1 }
  Dataflow {
    SpatialMap(1,1) K;
    TemporalMap(32,32) C;
    SpatialMap(256,256) Y;
    TemporalMap(Sz(R),Sz(R)) R;
    TemporalMap(Sz(S),Sz(S)) S;
    TemporalMap(1,1) X;
  }
}

Layer L103_FFN_1 {
  Type: CONV
  Stride { X:1, Y:1 }
  Dimensions { K:6144, C:2048, R:1, S:1, Y:512, X:1 }
  Dataflow {
    SpatialMap(1,1) K;
    TemporalMap(32,32) C;
    SpatialMap(256,256) Y;
    TemporalMap(Sz(R),Sz(R)) R;
    TemporalMap(Sz(S),Sz(S)) S;
    TemporalMap(1,1) X;
  }
}

Layer L104_FFN_2 {
  Type: CONV
  Stride { X:1, Y:1 }
  Dimensions { K:2048, C:6144, R:1, S:1, Y:512, X:1 }
  Dataflow {
    SpatialMap(1,1) K;
    TemporalMap(32,32) C;
    SpatialMap(256,256) Y;
    TemporalMap(Sz(R),Sz(R)) R;
    TemporalMap(Sz(S),Sz(S)) S;
    TemporalMap(1,1) X;
  }
}

Layer L105_Q_Proj {
  Type: CONV
  Stride { X:1, Y:1 }
  Dimensions { K:2048, C:2048, R:1, S:1, Y:512, X:1 }
  Dataflow {
    SpatialMap(1,1) K;
    TemporalMap(32,32) C;
    SpatialMap(256,256) Y;
    TemporalMap(Sz(R),Sz(R)) R;
    TemporalMap(Sz(S),Sz(S)) S;
    TemporalMap(1,1) X;
  }
}

Layer L106_K_Proj {
  Type: CONV
  Stride { X:1, Y:1 }
  Dimensions { K:1024, C:2048, R:1, S:1, Y:512, X:1 }
  Dataflow {
    SpatialMap(1,1) K;
    TemporalMap(32,32) C;
    SpatialMap(256,256) Y;
    TemporalMap(Sz(R),Sz(R)) R;
    TemporalMap(Sz(S),Sz(S)) S;
    TemporalMap(1,1) X;
  }
}

Layer L107_V_Proj {
  Type: CONV
  Stride { X:1, Y:1 }
  Dimensions { K:1024, C:2048, R:1, S:1, Y:512, X:1 }
  Dataflow {
    SpatialMap(1,1) K;
    TemporalMap(32,32) C;
    SpatialMap(256,256) Y;
    TemporalMap(Sz(R),Sz(R)) R;
    TemporalMap(Sz(S),Sz(S)) S;
    TemporalMap(1,1) X;
  }
}

Layer L108_Attn_Scores_QKt {
  Type: CONV
  Stride { X:1, Y:1 }
  Dimensions { K:512, C:128, R:1, S:1, Y:512, X:1 }
  Dataflow {
    SpatialMap(1,1) K;
    TemporalMap(8,8) C;
    TemporalMap(16,16) Y;
    TemporalMap(Sz(R),Sz(R)) R;
    TemporalMap(Sz(S),Sz(S)) S;
    TemporalMap(1,1) X;
  }
}

Layer L109_Attn_AV {
  Type: CONV
  Stride { X:1, Y:1 }
  Dimensions { K:128, C:512, R:1, S:1, Y:512, X:1 }
  Dataflow {
    SpatialMap(1,1) K;
    TemporalMap(16,16) C;
    TemporalMap(8,8) Y;
    TemporalMap(Sz(R),Sz(R)) R;
    TemporalMap(Sz(S),Sz(S)) S;
    TemporalMap(1,1) X;
  }
}

Layer L110_Out_Proj {
  Type: CONV
  Stride { X:1, Y:1 }
  Dimensions { K:2048, C:2048, R:1, S:1, Y:512, X:1 }
  Dataflow {
    SpatialMap(1,1) K;
    TemporalMap(32,32) C;
    SpatialMap(256,256) Y;
    TemporalMap(Sz(R),Sz(R)) R;
    TemporalMap(Sz(S),Sz(S)) S;
    TemporalMap(1,1) X;
  }
}

Layer L111_FFN_1 {
  Type: CONV
  Stride { X:1, Y:1 }
  Dimensions { K:6144, C:2048, R:1, S:1, Y:512, X:1 }
  Dataflow {
    SpatialMap(1,1) K;
    TemporalMap(32,32) C;
    SpatialMap(256,256) Y;
    TemporalMap(Sz(R),Sz(R)) R;
    TemporalMap(Sz(S),Sz(S)) S;
    TemporalMap(1,1) X;
  }
}

Layer L112_FFN_2 {
  Type: CONV
  Stride { X:1, Y:1 }
  Dimensions { K:2048, C:6144, R:1, S:1, Y:512, X:1 }
  Dataflow {
    SpatialMap(1,1) K;
    TemporalMap(32,32) C;
    SpatialMap(256,256) Y;
    TemporalMap(Sz(R),Sz(R)) R;
    TemporalMap(Sz(S),Sz(S)) S;
    TemporalMap(1,1) X;
  }
}

Layer L113_Q_Proj {
  Type: CONV
  Stride { X:1, Y:1 }
  Dimensions { K:2048, C:2048, R:1, S:1, Y:512, X:1 }
  Dataflow {
    SpatialMap(1,1) K;
    TemporalMap(32,32) C;
    SpatialMap(256,256) Y;
    TemporalMap(Sz(R),Sz(R)) R;
    TemporalMap(Sz(S),Sz(S)) S;
    TemporalMap(1,1) X;
  }
}

Layer L114_K_Proj {
  Type: CONV
  Stride { X:1, Y:1 }
  Dimensions { K:1024, C:2048, R:1, S:1, Y:512, X:1 }
  Dataflow {
    SpatialMap(1,1) K;
    TemporalMap(32,32) C;
    SpatialMap(256,256) Y;
    TemporalMap(Sz(R),Sz(R)) R;
    TemporalMap(Sz(S),Sz(S)) S;
    TemporalMap(1,1) X;
  }
}

Layer L115_V_Proj {
  Type: CONV
  Stride { X:1, Y:1 }
  Dimensions { K:1024, C:2048, R:1, S:1, Y:512, X:1 }
  Dataflow {
    SpatialMap(1,1) K;
    TemporalMap(32,32) C;
    SpatialMap(256,256) Y;
    TemporalMap(Sz(R),Sz(R)) R;
    TemporalMap(Sz(S),Sz(S)) S;
    TemporalMap(1,1) X;
  }
}

Layer L116_Attn_Scores_QKt {
  Type: CONV
  Stride { X:1, Y:1 }
  Dimensions { K:512, C:128, R:1, S:1, Y:512, X:1 }
  Dataflow {
    SpatialMap(1,1) K;
    TemporalMap(8,8) C;
    TemporalMap(16,16) Y;
    TemporalMap(Sz(R),Sz(R)) R;
    TemporalMap(Sz(S),Sz(S)) S;
    TemporalMap(1,1) X;
  }
}

Layer L117_Attn_AV {
  Type: CONV
  Stride { X:1, Y:1 }
  Dimensions { K:128, C:512, R:1, S:1, Y:512, X:1 }
  Dataflow {
    SpatialMap(1,1) K;
    TemporalMap(16,16) C;
    TemporalMap(8,8) Y;
    TemporalMap(Sz(R),Sz(R)) R;
    TemporalMap(Sz(S),Sz(S)) S;
    TemporalMap(1,1) X;
  }
}

Layer L118_Out_Proj {
  Type: CONV
  Stride { X:1, Y:1 }
  Dimensions { K:2048, C:2048, R:1, S:1, Y:512, X:1 }
  Dataflow {
    SpatialMap(1,1) K;
    TemporalMap(32,32) C;
    SpatialMap(256,256) Y;
    TemporalMap(Sz(R),Sz(R)) R;
    TemporalMap(Sz(S),Sz(S)) S;
    TemporalMap(1,1) X;
  }
}

Layer L119_FFN_1 {
  Type: CONV
  Stride { X:1, Y:1 }
  Dimensions { K:6144, C:2048, R:1, S:1, Y:512, X:1 }
  Dataflow {
    SpatialMap(1,1) K;
    TemporalMap(32,32) C;
    SpatialMap(256,256) Y;
    TemporalMap(Sz(R),Sz(R)) R;
    TemporalMap(Sz(S),Sz(S)) S;
    TemporalMap(1,1) X;
  }
}

Layer L120_FFN_2 {
  Type: CONV
  Stride { X:1, Y:1 }
  Dimensions { K:2048, C:6144, R:1, S:1, Y:512, X:1 }
  Dataflow {
    SpatialMap(1,1) K;
    TemporalMap(32,32) C;
    SpatialMap(256,256) Y;
    TemporalMap(Sz(R),Sz(R)) R;
    TemporalMap(Sz(S),Sz(S)) S;
    TemporalMap(1,1) X;
  }
}

Layer L121_Q_Proj {
  Type: CONV
  Stride { X:1, Y:1 }
  Dimensions { K:2048, C:2048, R:1, S:1, Y:512, X:1 }
  Dataflow {
    SpatialMap(1,1) K;
    TemporalMap(32,32) C;
    SpatialMap(256,256) Y;
    TemporalMap(Sz(R),Sz(R)) R;
    TemporalMap(Sz(S),Sz(S)) S;
    TemporalMap(1,1) X;
  }
}

Layer L122_K_Proj {
  Type: CONV
  Stride { X:1, Y:1 }
  Dimensions { K:1024, C:2048, R:1, S:1, Y:512, X:1 }
  Dataflow {
    SpatialMap(1,1) K;
    TemporalMap(32,32) C;
    SpatialMap(256,256) Y;
    TemporalMap(Sz(R),Sz(R)) R;
    TemporalMap(Sz(S),Sz(S)) S;
    TemporalMap(1,1) X;
  }
}

Layer L123_V_Proj {
  Type: CONV
  Stride { X:1, Y:1 }
  Dimensions { K:1024, C:2048, R:1, S:1, Y:512, X:1 }
  Dataflow {
    SpatialMap(1,1) K;
    TemporalMap(32,32) C;
    SpatialMap(256,256) Y;
    TemporalMap(Sz(R),Sz(R)) R;
    TemporalMap(Sz(S),Sz(S)) S;
    TemporalMap(1,1) X;
  }
}

Layer L124_Attn_Scores_QKt {
  Type: CONV
  Stride { X:1, Y:1 }
  Dimensions { K:512, C:128, R:1, S:1, Y:512, X:1 }
  Dataflow {
    SpatialMap(1,1) K;
    TemporalMap(8,8) C;
    TemporalMap(16,16) Y;
    TemporalMap(Sz(R),Sz(R)) R;
    TemporalMap(Sz(S),Sz(S)) S;
    TemporalMap(1,1) X;
  }
}

Layer L125_Attn_AV {
  Type: CONV
  Stride { X:1, Y:1 }
  Dimensions { K:128, C:512, R:1, S:1, Y:512, X:1 }
  Dataflow {
    SpatialMap(1,1) K;
    TemporalMap(16,16) C;
    TemporalMap(8,8) Y;
    TemporalMap(Sz(R),Sz(R)) R;
    TemporalMap(Sz(S),Sz(S)) S;
    TemporalMap(1,1) X;
  }
}

Layer L126_Out_Proj {
  Type: CONV
  Stride { X:1, Y:1 }
  Dimensions { K:2048, C:2048, R:1, S:1, Y:512, X:1 }
  Dataflow {
    SpatialMap(1,1) K;
    TemporalMap(32,32) C;
    SpatialMap(256,256) Y;
    TemporalMap(Sz(R),Sz(R)) R;
    TemporalMap(Sz(S),Sz(S)) S;
    TemporalMap(1,1) X;
  }
}

Layer L127_FFN_1 {
  Type: CONV
  Stride { X:1, Y:1 }
  Dimensions { K:6144, C:2048, R:1, S:1, Y:512, X:1 }
  Dataflow {
    SpatialMap(1,1) K;
    TemporalMap(32,32) C;
    SpatialMap(256,256) Y;
    TemporalMap(Sz(R),Sz(R)) R;
    TemporalMap(Sz(S),Sz(S)) S;
    TemporalMap(1,1) X;
  }
}

Layer L128_FFN_2 {
  Type: CONV
  Stride { X:1, Y:1 }
  Dimensions { K:2048, C:6144, R:1, S:1, Y:512, X:1 }
  Dataflow {
    SpatialMap(1,1) K;
    TemporalMap(32,32) C;
    SpatialMap(256,256) Y;
    TemporalMap(Sz(R),Sz(R)) R;
    TemporalMap(Sz(S),Sz(S)) S;
    TemporalMap(1,1) X;
  }
}

Layer L129_Q_Proj {
  Type: CONV
  Stride { X:1, Y:1 }
  Dimensions { K:2048, C:2048, R:1, S:1, Y:512, X:1 }
  Dataflow {
    SpatialMap(1,1) K;
    TemporalMap(32,32) C;
    SpatialMap(256,256) Y;
    TemporalMap(Sz(R),Sz(R)) R;
    TemporalMap(Sz(S),Sz(S)) S;
    TemporalMap(1,1) X;
  }
}

Layer L130_K_Proj {
  Type: CONV
  Stride { X:1, Y:1 }
  Dimensions { K:1024, C:2048, R:1, S:1, Y:512, X:1 }
  Dataflow {
    SpatialMap(1,1) K;
    TemporalMap(32,32) C;
    SpatialMap(256,256) Y;
    TemporalMap(Sz(R),Sz(R)) R;
    TemporalMap(Sz(S),Sz(S)) S;
    TemporalMap(1,1) X;
  }
}

Layer L131_V_Proj {
  Type: CONV
  Stride { X:1, Y:1 }
  Dimensions { K:1024, C:2048, R:1, S:1, Y:512, X:1 }
  Dataflow {
    SpatialMap(1,1) K;
    TemporalMap(32,32) C;
    SpatialMap(256,256) Y;
    TemporalMap(Sz(R),Sz(R)) R;
    TemporalMap(Sz(S),Sz(S)) S;
    TemporalMap(1,1) X;
  }
}

Layer L132_Attn_Scores_QKt {
  Type: CONV
  Stride { X:1, Y:1 }
  Dimensions { K:512, C:128, R:1, S:1, Y:512, X:1 }
  Dataflow {
    SpatialMap(1,1) K;
    TemporalMap(8,8) C;
    TemporalMap(16,16) Y;
    TemporalMap(Sz(R),Sz(R)) R;
    TemporalMap(Sz(S),Sz(S)) S;
    TemporalMap(1,1) X;
  }
}

Layer L133_Attn_AV {
  Type: CONV
  Stride { X:1, Y:1 }
  Dimensions { K:128, C:512, R:1, S:1, Y:512, X:1 }
  Dataflow {
    SpatialMap(1,1) K;
    TemporalMap(16,16) C;
    TemporalMap(8,8) Y;
    TemporalMap(Sz(R),Sz(R)) R;
    TemporalMap(Sz(S),Sz(S)) S;
    TemporalMap(1,1) X;
  }
}

Layer L134_Out_Proj {
  Type: CONV
  Stride { X:1, Y:1 }
  Dimensions { K:2048, C:2048, R:1, S:1, Y:512, X:1 }
  Dataflow {
    SpatialMap(1,1) K;
    TemporalMap(32,32) C;
    SpatialMap(256,256) Y;
    TemporalMap(Sz(R),Sz(R)) R;
    TemporalMap(Sz(S),Sz(S)) S;
    TemporalMap(1,1) X;
  }
}

Layer L135_FFN_1 {
  Type: CONV
  Stride { X:1, Y:1 }
  Dimensions { K:6144, C:2048, R:1, S:1, Y:512, X:1 }
  Dataflow {
    SpatialMap(1,1) K;
    TemporalMap(32,32) C;
    SpatialMap(256,256) Y;
    TemporalMap(Sz(R),Sz(R)) R;
    TemporalMap(Sz(S),Sz(S)) S;
    TemporalMap(1,1) X;
  }
}

Layer L136_FFN_2 {
  Type: CONV
  Stride { X:1, Y:1 }
  Dimensions { K:2048, C:6144, R:1, S:1, Y:512, X:1 }
  Dataflow {
    SpatialMap(1,1) K;
    TemporalMap(32,32) C;
    SpatialMap(256,256) Y;
    TemporalMap(Sz(R),Sz(R)) R;
    TemporalMap(Sz(S),Sz(S)) S;
    TemporalMap(1,1) X;
  }
}

Layer L137_Q_Proj {
  Type: CONV
  Stride { X:1, Y:1 }
  Dimensions { K:2048, C:2048, R:1, S:1, Y:512, X:1 }
  Dataflow {
    SpatialMap(1,1) K;
    TemporalMap(32,32) C;
    SpatialMap(256,256) Y;
    TemporalMap(Sz(R),Sz(R)) R;
    TemporalMap(Sz(S),Sz(S)) S;
    TemporalMap(1,1) X;
  }
}

Layer L138_K_Proj {
  Type: CONV
  Stride { X:1, Y:1 }
  Dimensions { K:1024, C:2048, R:1, S:1, Y:512, X:1 }
  Dataflow {
    SpatialMap(1,1) K;
    TemporalMap(32,32) C;
    SpatialMap(256,256) Y;
    TemporalMap(Sz(R),Sz(R)) R;
    TemporalMap(Sz(S),Sz(S)) S;
    TemporalMap(1,1) X;
  }
}

Layer L139_V_Proj {
  Type: CONV
  Stride { X:1, Y:1 }
  Dimensions { K:1024, C:2048, R:1, S:1, Y:512, X:1 }
  Dataflow {
    SpatialMap(1,1) K;
    TemporalMap(32,32) C;
    SpatialMap(256,256) Y;
    TemporalMap(Sz(R),Sz(R)) R;
    TemporalMap(Sz(S),Sz(S)) S;
    TemporalMap(1,1) X;
  }
}

Layer L140_Attn_Scores_QKt {
  Type: CONV
  Stride { X:1, Y:1 }
  Dimensions { K:512, C:128, R:1, S:1, Y:512, X:1 }
  Dataflow {
    SpatialMap(1,1) K;
    TemporalMap(8,8) C;
    TemporalMap(16,16) Y;
    TemporalMap(Sz(R),Sz(R)) R;
    TemporalMap(Sz(S),Sz(S)) S;
    TemporalMap(1,1) X;
  }
}

Layer L141_Attn_AV {
  Type: CONV
  Stride { X:1, Y:1 }
  Dimensions { K:128, C:512, R:1, S:1, Y:512, X:1 }
  Dataflow {
    SpatialMap(1,1) K;
    TemporalMap(16,16) C;
    TemporalMap(8,8) Y;
    TemporalMap(Sz(R),Sz(R)) R;
    TemporalMap(Sz(S),Sz(S)) S;
    TemporalMap(1,1) X;
  }
}

Layer L142_Out_Proj {
  Type: CONV
  Stride { X:1, Y:1 }
  Dimensions { K:2048, C:2048, R:1, S:1, Y:512, X:1 }
  Dataflow {
    SpatialMap(1,1) K;
    TemporalMap(32,32) C;
    SpatialMap(256,256) Y;
    TemporalMap(Sz(R),Sz(R)) R;
    TemporalMap(Sz(S),Sz(S)) S;
    TemporalMap(1,1) X;
  }
}

Layer L143_FFN_1 {
  Type: CONV
  Stride { X:1, Y:1 }
  Dimensions { K:6144, C:2048, R:1, S:1, Y:512, X:1 }
  Dataflow {
    SpatialMap(1,1) K;
    TemporalMap(32,32) C;
    SpatialMap(256,256) Y;
    TemporalMap(Sz(R),Sz(R)) R;
    TemporalMap(Sz(S),Sz(S)) S;
    TemporalMap(1,1) X;
  }
}

Layer L144_FFN_2 {
  Type: CONV
  Stride { X:1, Y:1 }
  Dimensions { K:2048, C:6144, R:1, S:1, Y:512, X:1 }
  Dataflow {
    SpatialMap(1,1) K;
    TemporalMap(32,32) C;
    SpatialMap(256,256) Y;
    TemporalMap(Sz(R),Sz(R)) R;
    TemporalMap(Sz(S),Sz(S)) S;
    TemporalMap(1,1) X;
  }
}

Layer L145_Q_Proj {
  Type: CONV
  Stride { X:1, Y:1 }
  Dimensions { K:2048, C:2048, R:1, S:1, Y:512, X:1 }
  Dataflow {
    SpatialMap(1,1) K;
    TemporalMap(32,32) C;
    SpatialMap(256,256) Y;
    TemporalMap(Sz(R),Sz(R)) R;
    TemporalMap(Sz(S),Sz(S)) S;
    TemporalMap(1,1) X;
  }
}

Layer L146_K_Proj {
  Type: CONV
  Stride { X:1, Y:1 }
  Dimensions { K:1024, C:2048, R:1, S:1, Y:512, X:1 }
  Dataflow {
    SpatialMap(1,1) K;
    TemporalMap(32,32) C;
    SpatialMap(256,256) Y;
    TemporalMap(Sz(R),Sz(R)) R;
    TemporalMap(Sz(S),Sz(S)) S;
    TemporalMap(1,1) X;
  }
}

Layer L147_V_Proj {
  Type: CONV
  Stride { X:1, Y:1 }
  Dimensions { K:1024, C:2048, R:1, S:1, Y:512, X:1 }
  Dataflow {
    SpatialMap(1,1) K;
    TemporalMap(32,32) C;
    SpatialMap(256,256) Y;
    TemporalMap(Sz(R),Sz(R)) R;
    TemporalMap(Sz(S),Sz(S)) S;
    TemporalMap(1,1) X;
  }
}

Layer L148_Attn_Scores_QKt {
  Type: CONV
  Stride { X:1, Y:1 }
  Dimensions { K:512, C:128, R:1, S:1, Y:512, X:1 }
  Dataflow {
    SpatialMap(1,1) K;
    TemporalMap(8,8) C;
    TemporalMap(16,16) Y;
    TemporalMap(Sz(R),Sz(R)) R;
    TemporalMap(Sz(S),Sz(S)) S;
    TemporalMap(1,1) X;
  }
}

Layer L149_Attn_AV {
  Type: CONV
  Stride { X:1, Y:1 }
  Dimensions { K:128, C:512, R:1, S:1, Y:512, X:1 }
  Dataflow {
    SpatialMap(1,1) K;
    TemporalMap(16,16) C;
    TemporalMap(8,8) Y;
    TemporalMap(Sz(R),Sz(R)) R;
    TemporalMap(Sz(S),Sz(S)) S;
    TemporalMap(1,1) X;
  }
}

Layer L150_Out_Proj {
  Type: CONV
  Stride { X:1, Y:1 }
  Dimensions { K:2048, C:2048, R:1, S:1, Y:512, X:1 }
  Dataflow {
    SpatialMap(1,1) K;
    TemporalMap(32,32) C;
    SpatialMap(256,256) Y;
    TemporalMap(Sz(R),Sz(R)) R;
    TemporalMap(Sz(S),Sz(S)) S;
    TemporalMap(1,1) X;
  }
}

Layer L151_FFN_1 {
  Type: CONV
  Stride { X:1, Y:1 }
  Dimensions { K:6144, C:2048, R:1, S:1, Y:512, X:1 }
  Dataflow {
    SpatialMap(1,1) K;
    TemporalMap(32,32) C;
    SpatialMap(256,256) Y;
    TemporalMap(Sz(R),Sz(R)) R;
    TemporalMap(Sz(S),Sz(S)) S;
    TemporalMap(1,1) X;
  }
}

Layer L152_FFN_2 {
  Type: CONV
  Stride { X:1, Y:1 }
  Dimensions { K:2048, C:6144, R:1, S:1, Y:512, X:1 }
  Dataflow {
    SpatialMap(1,1) K;
    TemporalMap(32,32) C;
    SpatialMap(256,256) Y;
    TemporalMap(Sz(R),Sz(R)) R;
    TemporalMap(Sz(S),Sz(S)) S;
    TemporalMap(1,1) X;
  }
}

Layer L153_Q_Proj {
  Type: CONV
  Stride { X:1, Y:1 }
  Dimensions { K:2048, C:2048, R:1, S:1, Y:512, X:1 }
  Dataflow {
    SpatialMap(1,1) K;
    TemporalMap(32,32) C;
    SpatialMap(256,256) Y;
    TemporalMap(Sz(R),Sz(R)) R;
    TemporalMap(Sz(S),Sz(S)) S;
    TemporalMap(1,1) X;
  }
}

Layer L154_K_Proj {
  Type: CONV
  Stride { X:1, Y:1 }
  Dimensions { K:1024, C:2048, R:1, S:1, Y:512, X:1 }
  Dataflow {
    SpatialMap(1,1) K;
    TemporalMap(32,32) C;
    SpatialMap(256,256) Y;
    TemporalMap(Sz(R),Sz(R)) R;
    TemporalMap(Sz(S),Sz(S)) S;
    TemporalMap(1,1) X;
  }
}

Layer L155_V_Proj {
  Type: CONV
  Stride { X:1, Y:1 }
  Dimensions { K:1024, C:2048, R:1, S:1, Y:512, X:1 }
  Dataflow {
    SpatialMap(1,1) K;
    TemporalMap(32,32) C;
    SpatialMap(256,256) Y;
    TemporalMap(Sz(R),Sz(R)) R;
    TemporalMap(Sz(S),Sz(S)) S;
    TemporalMap(1,1) X;
  }
}

Layer L156_Attn_Scores_QKt {
  Type: CONV
  Stride { X:1, Y:1 }
  Dimensions { K:512, C:128, R:1, S:1, Y:512, X:1 }
  Dataflow {
    SpatialMap(1,1) K;
    TemporalMap(8,8) C;
    TemporalMap(16,16) Y;
    TemporalMap(Sz(R),Sz(R)) R;
    TemporalMap(Sz(S),Sz(S)) S;
    TemporalMap(1,1) X;
  }
}

Layer L157_Attn_AV {
  Type: CONV
  Stride { X:1, Y:1 }
  Dimensions { K:128, C:512, R:1, S:1, Y:512, X:1 }
  Dataflow {
    SpatialMap(1,1) K;
    TemporalMap(16,16) C;
    TemporalMap(8,8) Y;
    TemporalMap(Sz(R),Sz(R)) R;
    TemporalMap(Sz(S),Sz(S)) S;
    TemporalMap(1,1) X;
  }
}

Layer L158_Out_Proj {
  Type: CONV
  Stride { X:1, Y:1 }
  Dimensions { K:2048, C:2048, R:1, S:1, Y:512, X:1 }
  Dataflow {
    SpatialMap(1,1) K;
    TemporalMap(32,32) C;
    SpatialMap(256,256) Y;
    TemporalMap(Sz(R),Sz(R)) R;
    TemporalMap(Sz(S),Sz(S)) S;
    TemporalMap(1,1) X;
  }
}

Layer L159_FFN_1 {
  Type: CONV
  Stride { X:1, Y:1 }
  Dimensions { K:6144, C:2048, R:1, S:1, Y:512, X:1 }
  Dataflow {
    SpatialMap(1,1) K;
    TemporalMap(32,32) C;
    SpatialMap(256,256) Y;
    TemporalMap(Sz(R),Sz(R)) R;
    TemporalMap(Sz(S),Sz(S)) S;
    TemporalMap(1,1) X;
  }
}

Layer L160_FFN_2 {
  Type: CONV
  Stride { X:1, Y:1 }
  Dimensions { K:2048, C:6144, R:1, S:1, Y:512, X:1 }
  Dataflow {
    SpatialMap(1,1) K;
    TemporalMap(32,32) C;
    SpatialMap(256,256) Y;
    TemporalMap(Sz(R),Sz(R)) R;
    TemporalMap(Sz(S),Sz(S)) S;
    TemporalMap(1,1) X;
  }
}

Layer L161_Q_Proj {
  Type: CONV
  Stride { X:1, Y:1 }
  Dimensions { K:2048, C:2048, R:1, S:1, Y:512, X:1 }
  Dataflow {
    SpatialMap(1,1) K;
    TemporalMap(32,32) C;
    SpatialMap(256,256) Y;
    TemporalMap(Sz(R),Sz(R)) R;
    TemporalMap(Sz(S),Sz(S)) S;
    TemporalMap(1,1) X;
  }
}

Layer L162_K_Proj {
  Type: CONV
  Stride { X:1, Y:1 }
  Dimensions { K:1024, C:2048, R:1, S:1, Y:512, X:1 }
  Dataflow {
    SpatialMap(1,1) K;
    TemporalMap(32,32) C;
    SpatialMap(256,256) Y;
    TemporalMap(Sz(R),Sz(R)) R;
    TemporalMap(Sz(S),Sz(S)) S;
    TemporalMap(1,1) X;
  }
}

Layer L163_V_Proj {
  Type: CONV
  Stride { X:1, Y:1 }
  Dimensions { K:1024, C:2048, R:1, S:1, Y:512, X:1 }
  Dataflow {
    SpatialMap(1,1) K;
    TemporalMap(32,32) C;
    SpatialMap(256,256) Y;
    TemporalMap(Sz(R),Sz(R)) R;
    TemporalMap(Sz(S),Sz(S)) S;
    TemporalMap(1,1) X;
  }
}

Layer L164_Attn_Scores_QKt {
  Type: CONV
  Stride { X:1, Y:1 }
  Dimensions { K:512, C:128, R:1, S:1, Y:512, X:1 }
  Dataflow {
    SpatialMap(1,1) K;
    TemporalMap(8,8) C;
    TemporalMap(16,16) Y;
    TemporalMap(Sz(R),Sz(R)) R;
    TemporalMap(Sz(S),Sz(S)) S;
    TemporalMap(1,1) X;
  }
}

Layer L165_Attn_AV {
  Type: CONV
  Stride { X:1, Y:1 }
  Dimensions { K:128, C:512, R:1, S:1, Y:512, X:1 }
  Dataflow {
    SpatialMap(1,1) K;
    TemporalMap(16,16) C;
    TemporalMap(8,8) Y;
    TemporalMap(Sz(R),Sz(R)) R;
    TemporalMap(Sz(S),Sz(S)) S;
    TemporalMap(1,1) X;
  }
}

Layer L166_Out_Proj {
  Type: CONV
  Stride { X:1, Y:1 }
  Dimensions { K:2048, C:2048, R:1, S:1, Y:512, X:1 }
  Dataflow {
    SpatialMap(1,1) K;
    TemporalMap(32,32) C;
    SpatialMap(256,256) Y;
    TemporalMap(Sz(R),Sz(R)) R;
    TemporalMap(Sz(S),Sz(S)) S;
    TemporalMap(1,1) X;
  }
}

Layer L167_FFN_1 {
  Type: CONV
  Stride { X:1, Y:1 }
  Dimensions { K:6144, C:2048, R:1, S:1, Y:512, X:1 }
  Dataflow {
    SpatialMap(1,1) K;
    TemporalMap(32,32) C;
    SpatialMap(256,256) Y;
    TemporalMap(Sz(R),Sz(R)) R;
    TemporalMap(Sz(S),Sz(S)) S;
    TemporalMap(1,1) X;
  }
}

Layer L168_FFN_2 {
  Type: CONV
  Stride { X:1, Y:1 }
  Dimensions { K:2048, C:6144, R:1, S:1, Y:512, X:1 }
  Dataflow {
    SpatialMap(1,1) K;
    TemporalMap(32,32) C;
    SpatialMap(256,256) Y;
    TemporalMap(Sz(R),Sz(R)) R;
    TemporalMap(Sz(S),Sz(S)) S;
    TemporalMap(1,1) X;
  }
}

Layer L169_Q_Proj {
  Type: CONV
  Stride { X:1, Y:1 }
  Dimensions { K:2048, C:2048, R:1, S:1, Y:512, X:1 }
  Dataflow {
    SpatialMap(1,1) K;
    TemporalMap(32,32) C;
    SpatialMap(256,256) Y;
    TemporalMap(Sz(R),Sz(R)) R;
    TemporalMap(Sz(S),Sz(S)) S;
    TemporalMap(1,1) X;
  }
}

Layer L170_K_Proj {
  Type: CONV
  Stride { X:1, Y:1 }
  Dimensions { K:1024, C:2048, R:1, S:1, Y:512, X:1 }
  Dataflow {
    SpatialMap(1,1) K;
    TemporalMap(32,32) C;
    SpatialMap(256,256) Y;
    TemporalMap(Sz(R),Sz(R)) R;
    TemporalMap(Sz(S),Sz(S)) S;
    TemporalMap(1,1) X;
  }
}

Layer L171_V_Proj {
  Type: CONV
  Stride { X:1, Y:1 }
  Dimensions { K:1024, C:2048, R:1, S:1, Y:512, X:1 }
  Dataflow {
    SpatialMap(1,1) K;
    TemporalMap(32,32) C;
    SpatialMap(256,256) Y;
    TemporalMap(Sz(R),Sz(R)) R;
    TemporalMap(Sz(S),Sz(S)) S;
    TemporalMap(1,1) X;
  }
}

Layer L172_Attn_Scores_QKt {
  Type: CONV
  Stride { X:1, Y:1 }
  Dimensions { K:512, C:128, R:1, S:1, Y:512, X:1 }
  Dataflow {
    SpatialMap(1,1) K;
    TemporalMap(8,8) C;
    TemporalMap(16,16) Y;
    TemporalMap(Sz(R),Sz(R)) R;
    TemporalMap(Sz(S),Sz(S)) S;
    TemporalMap(1,1) X;
  }
}

Layer L173_Attn_AV {
  Type: CONV
  Stride { X:1, Y:1 }
  Dimensions { K:128, C:512, R:1, S:1, Y:512, X:1 }
  Dataflow {
    SpatialMap(1,1) K;
    TemporalMap(16,16) C;
    TemporalMap(8,8) Y;
    TemporalMap(Sz(R),Sz(R)) R;
    TemporalMap(Sz(S),Sz(S)) S;
    TemporalMap(1,1) X;
  }
}

Layer L174_Out_Proj {
  Type: CONV
  Stride { X:1, Y:1 }
  Dimensions { K:2048, C:2048, R:1, S:1, Y:512, X:1 }
  Dataflow {
    SpatialMap(1,1) K;
    TemporalMap(32,32) C;
    SpatialMap(256,256) Y;
    TemporalMap(Sz(R),Sz(R)) R;
    TemporalMap(Sz(S),Sz(S)) S;
    TemporalMap(1,1) X;
  }
}

Layer L175_FFN_1 {
  Type: CONV
  Stride { X:1, Y:1 }
  Dimensions { K:6144, C:2048, R:1, S:1, Y:512, X:1 }
  Dataflow {
    SpatialMap(1,1) K;
    TemporalMap(32,32) C;
    SpatialMap(256,256) Y;
    TemporalMap(Sz(R),Sz(R)) R;
    TemporalMap(Sz(S),Sz(S)) S;
    TemporalMap(1,1) X;
  }
}

Layer L176_FFN_2 {
  Type: CONV
  Stride { X:1, Y:1 }
  Dimensions { K:2048, C:6144, R:1, S:1, Y:512, X:1 }
  Dataflow {
    SpatialMap(1,1) K;
    TemporalMap(32,32) C;
    SpatialMap(256,256) Y;
    TemporalMap(Sz(R),Sz(R)) R;
    TemporalMap(Sz(S),Sz(S)) S;
    TemporalMap(1,1) X;
  }
}

Layer L177_Q_Proj {
  Type: CONV
  Stride { X:1, Y:1 }
  Dimensions { K:2048, C:2048, R:1, S:1, Y:512, X:1 }
  Dataflow {
    SpatialMap(1,1) K;
    TemporalMap(32,32) C;
    SpatialMap(256,256) Y;
    TemporalMap(Sz(R),Sz(R)) R;
    TemporalMap(Sz(S),Sz(S)) S;
    TemporalMap(1,1) X;
  }
}

Layer L178_K_Proj {
  Type: CONV
  Stride { X:1, Y:1 }
  Dimensions { K:1024, C:2048, R:1, S:1, Y:512, X:1 }
  Dataflow {
    SpatialMap(1,1) K;
    TemporalMap(32,32) C;
    SpatialMap(256,256) Y;
    TemporalMap(Sz(R),Sz(R)) R;
    TemporalMap(Sz(S),Sz(S)) S;
    TemporalMap(1,1) X;
  }
}

Layer L179_V_Proj {
  Type: CONV
  Stride { X:1, Y:1 }
  Dimensions { K:1024, C:2048, R:1, S:1, Y:512, X:1 }
  Dataflow {
    SpatialMap(1,1) K;
    TemporalMap(32,32) C;
    SpatialMap(256,256) Y;
    TemporalMap(Sz(R),Sz(R)) R;
    TemporalMap(Sz(S),Sz(S)) S;
    TemporalMap(1,1) X;
  }
}

Layer L180_Attn_Scores_QKt {
  Type: CONV
  Stride { X:1, Y:1 }
  Dimensions { K:512, C:128, R:1, S:1, Y:512, X:1 }
  Dataflow {
    SpatialMap(1,1) K;
    TemporalMap(8,8) C;
    TemporalMap(16,16) Y;
    TemporalMap(Sz(R),Sz(R)) R;
    TemporalMap(Sz(S),Sz(S)) S;
    TemporalMap(1,1) X;
  }
}

Layer L181_Attn_AV {
  Type: CONV
  Stride { X:1, Y:1 }
  Dimensions { K:128, C:512, R:1, S:1, Y:512, X:1 }
  Dataflow {
    SpatialMap(1,1) K;
    TemporalMap(16,16) C;
    TemporalMap(8,8) Y;
    TemporalMap(Sz(R),Sz(R)) R;
    TemporalMap(Sz(S),Sz(S)) S;
    TemporalMap(1,1) X;
  }
}

Layer L182_Out_Proj {
  Type: CONV
  Stride { X:1, Y:1 }
  Dimensions { K:2048, C:2048, R:1, S:1, Y:512, X:1 }
  Dataflow {
    SpatialMap(1,1) K;
    TemporalMap(32,32) C;
    SpatialMap(256,256) Y;
    TemporalMap(Sz(R),Sz(R)) R;
    TemporalMap(Sz(S),Sz(S)) S;
    TemporalMap(1,1) X;
  }
}

Layer L183_FFN_1 {
  Type: CONV
  Stride { X:1, Y:1 }
  Dimensions { K:6144, C:2048, R:1, S:1, Y:512, X:1 }
  Dataflow {
    SpatialMap(1,1) K;
    TemporalMap(32,32) C;
    SpatialMap(256,256) Y;
    TemporalMap(Sz(R),Sz(R)) R;
    TemporalMap(Sz(S),Sz(S)) S;
    TemporalMap(1,1) X;
  }
}

Layer L184_FFN_2 {
  Type: CONV
  Stride { X:1, Y:1 }
  Dimensions { K:2048, C:6144, R:1, S:1, Y:512, X:1 }
  Dataflow {
    SpatialMap(1,1) K;
    TemporalMap(32,32) C;
    SpatialMap(256,256) Y;
    TemporalMap(Sz(R),Sz(R)) R;
    TemporalMap(Sz(S),Sz(S)) S;
    TemporalMap(1,1) X;
  }
}

Layer L185_Q_Proj {
  Type: CONV
  Stride { X:1, Y:1 }
  Dimensions { K:2048, C:2048, R:1, S:1, Y:512, X:1 }
  Dataflow {
    SpatialMap(1,1) K;
    TemporalMap(32,32) C;
    SpatialMap(256,256) Y;
    TemporalMap(Sz(R),Sz(R)) R;
    TemporalMap(Sz(S),Sz(S)) S;
    TemporalMap(1,1) X;
  }
}

Layer L186_K_Proj {
  Type: CONV
  Stride { X:1, Y:1 }
  Dimensions { K:1024, C:2048, R:1, S:1, Y:512, X:1 }
  Dataflow {
    SpatialMap(1,1) K;
    TemporalMap(32,32) C;
    SpatialMap(256,256) Y;
    TemporalMap(Sz(R),Sz(R)) R;
    TemporalMap(Sz(S),Sz(S)) S;
    TemporalMap(1,1) X;
  }
}

Layer L187_V_Proj {
  Type: CONV
  Stride { X:1, Y:1 }
  Dimensions { K:1024, C:2048, R:1, S:1, Y:512, X:1 }
  Dataflow {
    SpatialMap(1,1) K;
    TemporalMap(32,32) C;
    SpatialMap(256,256) Y;
    TemporalMap(Sz(R),Sz(R)) R;
    TemporalMap(Sz(S),Sz(S)) S;
    TemporalMap(1,1) X;
  }
}

Layer L188_Attn_Scores_QKt {
  Type: CONV
  Stride { X:1, Y:1 }
  Dimensions { K:512, C:128, R:1, S:1, Y:512, X:1 }
  Dataflow {
    SpatialMap(1,1) K;
    TemporalMap(8,8) C;
    TemporalMap(16,16) Y;
    TemporalMap(Sz(R),Sz(R)) R;
    TemporalMap(Sz(S),Sz(S)) S;
    TemporalMap(1,1) X;
  }
}

Layer L189_Attn_AV {
  Type: CONV
  Stride { X:1, Y:1 }
  Dimensions { K:128, C:512, R:1, S:1, Y:512, X:1 }
  Dataflow {
    SpatialMap(1,1) K;
    TemporalMap(16,16) C;
    TemporalMap(8,8) Y;
    TemporalMap(Sz(R),Sz(R)) R;
    TemporalMap(Sz(S),Sz(S)) S;
    TemporalMap(1,1) X;
  }
}

Layer L190_Out_Proj {
  Type: CONV
  Stride { X:1, Y:1 }
  Dimensions { K:2048, C:2048, R:1, S:1, Y:512, X:1 }
  Dataflow {
    SpatialMap(1,1) K;
    TemporalMap(32,32) C;
    SpatialMap(256,256) Y;
    TemporalMap(Sz(R),Sz(R)) R;
    TemporalMap(Sz(S),Sz(S)) S;
    TemporalMap(1,1) X;
  }
}

Layer L191_FFN_1 {
  Type: CONV
  Stride { X:1, Y:1 }
  Dimensions { K:6144, C:2048, R:1, S:1, Y:512, X:1 }
  Dataflow {
    SpatialMap(1,1) K;
    TemporalMap(32,32) C;
    SpatialMap(256,256) Y;
    TemporalMap(Sz(R),Sz(R)) R;
    TemporalMap(Sz(S),Sz(S)) S;
    TemporalMap(1,1) X;
  }
}

Layer L192_FFN_2 {
  Type: CONV
  Stride { X:1, Y:1 }
  Dimensions { K:2048, C:6144, R:1, S:1, Y:512, X:1 }
  Dataflow {
    SpatialMap(1,1) K;
    TemporalMap(32,32) C;
    SpatialMap(256,256) Y;
    TemporalMap(Sz(R),Sz(R)) R;
    TemporalMap(Sz(S),Sz(S)) S;
    TemporalMap(1,1) X;
  }
}

Layer L193_Q_Proj {
  Type: CONV
  Stride { X:1, Y:1 }
  Dimensions { K:2048, C:2048, R:1, S:1, Y:512, X:1 }
  Dataflow {
    SpatialMap(1,1) K;
    TemporalMap(32,32) C;
    SpatialMap(256,256) Y;
    TemporalMap(Sz(R),Sz(R)) R;
    TemporalMap(Sz(S),Sz(S)) S;
    TemporalMap(1,1) X;
  }
}

Layer L194_K_Proj {
  Type: CONV
  Stride { X:1, Y:1 }
  Dimensions { K:1024, C:2048, R:1, S:1, Y:512, X:1 }
  Dataflow {
    SpatialMap(1,1) K;
    TemporalMap(32,32) C;
    SpatialMap(256,256) Y;
    TemporalMap(Sz(R),Sz(R)) R;
    TemporalMap(Sz(S),Sz(S)) S;
    TemporalMap(1,1) X;
  }
}

Layer L195_V_Proj {
  Type: CONV
  Stride { X:1, Y:1 }
  Dimensions { K:1024, C:2048, R:1, S:1, Y:512, X:1 }
  Dataflow {
    SpatialMap(1,1) K;
    TemporalMap(32,32) C;
    SpatialMap(256,256) Y;
    TemporalMap(Sz(R),Sz(R)) R;
    TemporalMap(Sz(S),Sz(S)) S;
    TemporalMap(1,1) X;
  }
}

Layer L196_Attn_Scores_QKt {
  Type: CONV
  Stride { X:1, Y:1 }
  Dimensions { K:512, C:128, R:1, S:1, Y:512, X:1 }
  Dataflow {
    SpatialMap(1,1) K;
    TemporalMap(8,8) C;
    TemporalMap(16,16) Y;
    TemporalMap(Sz(R),Sz(R)) R;
    TemporalMap(Sz(S),Sz(S)) S;
    TemporalMap(1,1) X;
  }
}

Layer L197_Attn_AV {
  Type: CONV
  Stride { X:1, Y:1 }
  Dimensions { K:128, C:512, R:1, S:1, Y:512, X:1 }
  Dataflow {
    SpatialMap(1,1) K;
    TemporalMap(16,16) C;
    TemporalMap(8,8) Y;
    TemporalMap(Sz(R),Sz(R)) R;
    TemporalMap(Sz(S),Sz(S)) S;
    TemporalMap(1,1) X;
  }
}

Layer L198_Out_Proj {
  Type: CONV
  Stride { X:1, Y:1 }
  Dimensions { K:2048, C:2048, R:1, S:1, Y:512, X:1 }
  Dataflow {
    SpatialMap(1,1) K;
    TemporalMap(32,32) C;
    SpatialMap(256,256) Y;
    TemporalMap(Sz(R),Sz(R)) R;
    TemporalMap(Sz(S),Sz(S)) S;
    TemporalMap(1,1) X;
  }
}

Layer L199_FFN_1 {
  Type: CONV
  Stride { X:1, Y:1 }
  Dimensions { K:6144, C:2048, R:1, S:1, Y:512, X:1 }
  Dataflow {
    SpatialMap(1,1) K;
    TemporalMap(32,32) C;
    SpatialMap(256,256) Y;
    TemporalMap(Sz(R),Sz(R)) R;
    TemporalMap(Sz(S),Sz(S)) S;
    TemporalMap(1,1) X;
  }
}

Layer L200_FFN_2 {
  Type: CONV
  Stride { X:1, Y:1 }
  Dimensions { K:2048, C:6144, R:1, S:1, Y:512, X:1 }
  Dataflow {
    SpatialMap(1,1) K;
    TemporalMap(32,32) C;
    SpatialMap(256,256) Y;
    TemporalMap(Sz(R),Sz(R)) R;
    TemporalMap(Sz(S),Sz(S)) S;
    TemporalMap(1,1) X;
  }
}

Layer L201_Q_Proj {
  Type: CONV
  Stride { X:1, Y:1 }
  Dimensions { K:2048, C:2048, R:1, S:1, Y:512, X:1 }
  Dataflow {
    SpatialMap(1,1) K;
    TemporalMap(32,32) C;
    SpatialMap(256,256) Y;
    TemporalMap(Sz(R),Sz(R)) R;
    TemporalMap(Sz(S),Sz(S)) S;
    TemporalMap(1,1) X;
  }
}

Layer L202_K_Proj {
  Type: CONV
  Stride { X:1, Y:1 }
  Dimensions { K:1024, C:2048, R:1, S:1, Y:512, X:1 }
  Dataflow {
    SpatialMap(1,1) K;
    TemporalMap(32,32) C;
    SpatialMap(256,256) Y;
    TemporalMap(Sz(R),Sz(R)) R;
    TemporalMap(Sz(S),Sz(S)) S;
    TemporalMap(1,1) X;
  }
}

Layer L203_V_Proj {
  Type: CONV
  Stride { X:1, Y:1 }
  Dimensions { K:1024, C:2048, R:1, S:1, Y:512, X:1 }
  Dataflow {
    SpatialMap(1,1) K;
    TemporalMap(32,32) C;
    SpatialMap(256,256) Y;
    TemporalMap(Sz(R),Sz(R)) R;
    TemporalMap(Sz(S),Sz(S)) S;
    TemporalMap(1,1) X;
  }
}

Layer L204_Attn_Scores_QKt {
  Type: CONV
  Stride { X:1, Y:1 }
  Dimensions { K:512, C:128, R:1, S:1, Y:512, X:1 }
  Dataflow {
    SpatialMap(1,1) K;
    TemporalMap(8,8) C;
    TemporalMap(16,16) Y;
    TemporalMap(Sz(R),Sz(R)) R;
    TemporalMap(Sz(S),Sz(S)) S;
    TemporalMap(1,1) X;
  }
}

Layer L205_Attn_AV {
  Type: CONV
  Stride { X:1, Y:1 }
  Dimensions { K:128, C:512, R:1, S:1, Y:512, X:1 }
  Dataflow {
    SpatialMap(1,1) K;
    TemporalMap(16,16) C;
    TemporalMap(8,8) Y;
    TemporalMap(Sz(R),Sz(R)) R;
    TemporalMap(Sz(S),Sz(S)) S;
    TemporalMap(1,1) X;
  }
}

Layer L206_Out_Proj {
  Type: CONV
  Stride { X:1, Y:1 }
  Dimensions { K:2048, C:2048, R:1, S:1, Y:512, X:1 }
  Dataflow {
    SpatialMap(1,1) K;
    TemporalMap(32,32) C;
    SpatialMap(256,256) Y;
    TemporalMap(Sz(R),Sz(R)) R;
    TemporalMap(Sz(S),Sz(S)) S;
    TemporalMap(1,1) X;
  }
}

Layer L207_FFN_1 {
  Type: CONV
  Stride { X:1, Y:1 }
  Dimensions { K:6144, C:2048, R:1, S:1, Y:512, X:1 }
  Dataflow {
    SpatialMap(1,1) K;
    TemporalMap(32,32) C;
    SpatialMap(256,256) Y;
    TemporalMap(Sz(R),Sz(R)) R;
    TemporalMap(Sz(S),Sz(S)) S;
    TemporalMap(1,1) X;
  }
}

Layer L208_FFN_2 {
  Type: CONV
  Stride { X:1, Y:1 }
  Dimensions { K:2048, C:6144, R:1, S:1, Y:512, X:1 }
  Dataflow {
    SpatialMap(1,1) K;
    TemporalMap(32,32) C;
    SpatialMap(256,256) Y;
    TemporalMap(Sz(R),Sz(R)) R;
    TemporalMap(Sz(S),Sz(S)) S;
    TemporalMap(1,1) X;
  }
}

Layer L209_Q_Proj {
  Type: CONV
  Stride { X:1, Y:1 }
  Dimensions { K:2048, C:2048, R:1, S:1, Y:512, X:1 }
  Dataflow {
    SpatialMap(1,1) K;
    TemporalMap(32,32) C;
    SpatialMap(256,256) Y;
    TemporalMap(Sz(R),Sz(R)) R;
    TemporalMap(Sz(S),Sz(S)) S;
    TemporalMap(1,1) X;
  }
}

Layer L210_K_Proj {
  Type: CONV
  Stride { X:1, Y:1 }
  Dimensions { K:1024, C:2048, R:1, S:1, Y:512, X:1 }
  Dataflow {
    SpatialMap(1,1) K;
    TemporalMap(32,32) C;
    SpatialMap(256,256) Y;
    TemporalMap(Sz(R),Sz(R)) R;
    TemporalMap(Sz(S),Sz(S)) S;
    TemporalMap(1,1) X;
  }
}

Layer L211_V_Proj {
  Type: CONV
  Stride { X:1, Y:1 }
  Dimensions { K:1024, C:2048, R:1, S:1, Y:512, X:1 }
  Dataflow {
    SpatialMap(1,1) K;
    TemporalMap(32,32) C;
    SpatialMap(256,256) Y;
    TemporalMap(Sz(R),Sz(R)) R;
    TemporalMap(Sz(S),Sz(S)) S;
    TemporalMap(1,1) X;
  }
}

Layer L212_Attn_Scores_QKt {
  Type: CONV
  Stride { X:1, Y:1 }
  Dimensions { K:512, C:128, R:1, S:1, Y:512, X:1 }
  Dataflow {
    SpatialMap(1,1) K;
    TemporalMap(8,8) C;
    TemporalMap(16,16) Y;
    TemporalMap(Sz(R),Sz(R)) R;
    TemporalMap(Sz(S),Sz(S)) S;
    TemporalMap(1,1) X;
  }
}

Layer L213_Attn_AV {
  Type: CONV
  Stride { X:1, Y:1 }
  Dimensions { K:128, C:512, R:1, S:1, Y:512, X:1 }
  Dataflow {
    SpatialMap(1,1) K;
    TemporalMap(16,16) C;
    TemporalMap(8,8) Y;
    TemporalMap(Sz(R),Sz(R)) R;
    TemporalMap(Sz(S),Sz(S)) S;
    TemporalMap(1,1) X;
  }
}

Layer L214_Out_Proj {
  Type: CONV
  Stride { X:1, Y:1 }
  Dimensions { K:2048, C:2048, R:1, S:1, Y:512, X:1 }
  Dataflow {
    SpatialMap(1,1) K;
    TemporalMap(32,32) C;
    SpatialMap(256,256) Y;
    TemporalMap(Sz(R),Sz(R)) R;
    TemporalMap(Sz(S),Sz(S)) S;
    TemporalMap(1,1) X;
  }
}

Layer L215_FFN_1 {
  Type: CONV
  Stride { X:1, Y:1 }
  Dimensions { K:6144, C:2048, R:1, S:1, Y:512, X:1 }
  Dataflow {
    SpatialMap(1,1) K;
    TemporalMap(32,32) C;
    SpatialMap(256,256) Y;
    TemporalMap(Sz(R),Sz(R)) R;
    TemporalMap(Sz(S),Sz(S)) S;
    TemporalMap(1,1) X;
  }
}

Layer L216_FFN_2 {
  Type: CONV
  Stride { X:1, Y:1 }
  Dimensions { K:2048, C:6144, R:1, S:1, Y:512, X:1 }
  Dataflow {
    SpatialMap(1,1) K;
    TemporalMap(32,32) C;
    SpatialMap(256,256) Y;
    TemporalMap(Sz(R),Sz(R)) R;
    TemporalMap(Sz(S),Sz(S)) S;
    TemporalMap(1,1) X;
  }
}

Layer L217_Q_Proj {
  Type: CONV
  Stride { X:1, Y:1 }
  Dimensions { K:2048, C:2048, R:1, S:1, Y:512, X:1 }
  Dataflow {
    SpatialMap(1,1) K;
    TemporalMap(32,32) C;
    SpatialMap(256,256) Y;
    TemporalMap(Sz(R),Sz(R)) R;
    TemporalMap(Sz(S),Sz(S)) S;
    TemporalMap(1,1) X;
  }
}

Layer L218_K_Proj {
  Type: CONV
  Stride { X:1, Y:1 }
  Dimensions { K:1024, C:2048, R:1, S:1, Y:512, X:1 }
  Dataflow {
    SpatialMap(1,1) K;
    TemporalMap(32,32) C;
    SpatialMap(256,256) Y;
    TemporalMap(Sz(R),Sz(R)) R;
    TemporalMap(Sz(S),Sz(S)) S;
    TemporalMap(1,1) X;
  }
}

Layer L219_V_Proj {
  Type: CONV
  Stride { X:1, Y:1 }
  Dimensions { K:1024, C:2048, R:1, S:1, Y:512, X:1 }
  Dataflow {
    SpatialMap(1,1) K;
    TemporalMap(32,32) C;
    SpatialMap(256,256) Y;
    TemporalMap(Sz(R),Sz(R)) R;
    TemporalMap(Sz(S),Sz(S)) S;
    TemporalMap(1,1) X;
  }
}

Layer L220_Attn_Scores_QKt {
  Type: CONV
  Stride { X:1, Y:1 }
  Dimensions { K:512, C:128, R:1, S:1, Y:512, X:1 }
  Dataflow {
    SpatialMap(1,1) K;
    TemporalMap(8,8) C;
    TemporalMap(16,16) Y;
    TemporalMap(Sz(R),Sz(R)) R;
    TemporalMap(Sz(S),Sz(S)) S;
    TemporalMap(1,1) X;
  }
}

Layer L221_Attn_AV {
  Type: CONV
  Stride { X:1, Y:1 }
  Dimensions { K:128, C:512, R:1, S:1, Y:512, X:1 }
  Dataflow {
    SpatialMap(1,1) K;
    TemporalMap(16,16) C;
    TemporalMap(8,8) Y;
    TemporalMap(Sz(R),Sz(R)) R;
    TemporalMap(Sz(S),Sz(S)) S;
    TemporalMap(1,1) X;
  }
}

Layer L222_Out_Proj {
  Type: CONV
  Stride { X:1, Y:1 }
  Dimensions { K:2048, C:2048, R:1, S:1, Y:512, X:1 }
  Dataflow {
    SpatialMap(1,1) K;
    TemporalMap(32,32) C;
    SpatialMap(256,256) Y;
    TemporalMap(Sz(R),Sz(R)) R;
    TemporalMap(Sz(S),Sz(S)) S;
    TemporalMap(1,1) X;
  }
}

Layer L223_FFN_1 {
  Type: CONV
  Stride { X:1, Y:1 }
  Dimensions { K:6144, C:2048, R:1, S:1, Y:512, X:1 }
  Dataflow {
    SpatialMap(1,1) K;
    TemporalMap(32,32) C;
    SpatialMap(256,256) Y;
    TemporalMap(Sz(R),Sz(R)) R;
    TemporalMap(Sz(S),Sz(S)) S;
    TemporalMap(1,1) X;
  }
}

Layer L224_FFN_2 {
  Type: CONV
  Stride { X:1, Y:1 }
  Dimensions { K:2048, C:6144, R:1, S:1, Y:512, X:1 }
  Dataflow {
    SpatialMap(1,1) K;
    TemporalMap(32,32) C;
    SpatialMap(256,256) Y;
    TemporalMap(Sz(R),Sz(R)) R;
    TemporalMap(Sz(S),Sz(S)) S;
    TemporalMap(1,1) X;
  }
}

}

