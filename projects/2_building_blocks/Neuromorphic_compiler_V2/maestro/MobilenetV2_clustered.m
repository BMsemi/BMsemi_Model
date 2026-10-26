Network TinyMobileNetV2_Digits {
Layer features_0_0 {
  Type: CONV
  Dimensions { K: 8, C: 1, Y: 4, X: 4, R: 3, S: 3 }
  Stride { Y: 2, X: 2 }
  Meta { module: "features.0.0", groups: 1, expected_tiles: 1 }
  Dataflow {
    SpatialMap(8, 8) K;
    TemporalMap(1, 1) C;
    TemporalMap(3, 3) R;
    TemporalMap(3, 3) S;
    Cluster(1, P);
  }
}
Layer features_1_conv_0_0 {
  Type: DEPTHWISE_CONV
  Dimensions { K: 8, C: 8, Y: 4, X: 4, R: 3, S: 3 }
  Stride { Y: 1, X: 1 }
  Meta { module: "features.1.conv.0.0", groups: 8, expected_tiles: 1 }
  Dataflow {
    SpatialMap(8, 8) K;
    TemporalMap(1, 1) C;
    TemporalMap(3, 3) R;
    TemporalMap(3, 3) S;
    Cluster(1, P);
  }
}
Layer features_1_conv_1 {
  Type: POINTWISE_CONV
  Dimensions { K: 8, C: 8, Y: 4, X: 4, R: 1, S: 1 }
  Stride { Y: 1, X: 1 }
  Meta { module: "features.1.conv.1", groups: 1, expected_tiles: 1 }
  Dataflow {
    SpatialMap(8, 8) K;
    TemporalMap(8, 8) C;
    TemporalMap(1, 1) R;
    TemporalMap(1, 1) S;
    Cluster(1, P);
  }
}
Layer features_2_conv_0_0 {
  Type: POINTWISE_CONV
  Dimensions { K: 16, C: 8, Y: 4, X: 4, R: 1, S: 1 }
  Stride { Y: 1, X: 1 }
  Meta { module: "features.2.conv.0.0", groups: 1, expected_tiles: 1 }
  Dataflow {
    SpatialMap(16, 16) K;
    TemporalMap(8, 8) C;
    TemporalMap(1, 1) R;
    TemporalMap(1, 1) S;
    Cluster(1, P);
  }
}
Layer features_2_conv_1_0 {
  Type: DEPTHWISE_CONV
  Dimensions { K: 16, C: 16, Y: 2, X: 2, R: 3, S: 3 }
  Stride { Y: 2, X: 2 }
  Meta { module: "features.2.conv.1.0", groups: 16, expected_tiles: 1 }
  Dataflow {
    SpatialMap(16, 16) K;
    TemporalMap(1, 1) C;
    TemporalMap(3, 3) R;
    TemporalMap(3, 3) S;
    Cluster(1, P);
  }
}
Layer features_2_conv_2 {
  Type: POINTWISE_CONV
  Dimensions { K: 12, C: 16, Y: 2, X: 2, R: 1, S: 1 }
  Stride { Y: 1, X: 1 }
  Meta { module: "features.2.conv.2", groups: 1, expected_tiles: 1 }
  Dataflow {
    SpatialMap(12, 12) K;
    TemporalMap(16, 16) C;
    TemporalMap(1, 1) R;
    TemporalMap(1, 1) S;
    Cluster(1, P);
  }
}
Layer features_3_conv_0_0 {
  Type: POINTWISE_CONV
  Dimensions { K: 24, C: 12, Y: 2, X: 2, R: 1, S: 1 }
  Stride { Y: 1, X: 1 }
  Meta { module: "features.3.conv.0.0", groups: 1, expected_tiles: 2 }
  Dataflow {
    SpatialMap(16, 16) K;
    TemporalMap(12, 12) C;
    TemporalMap(1, 1) R;
    TemporalMap(1, 1) S;
    Cluster(2, P);
  }
}
Layer features_3_conv_1_0 {
  Type: DEPTHWISE_CONV
  Dimensions { K: 24, C: 24, Y: 2, X: 2, R: 3, S: 3 }
  Stride { Y: 1, X: 1 }
  Meta { module: "features.3.conv.1.0", groups: 24, expected_tiles: 2 }
  Dataflow {
    SpatialMap(16, 16) K;
    TemporalMap(1, 1) C;
    TemporalMap(3, 3) R;
    TemporalMap(3, 3) S;
    Cluster(2, P);
  }
}
Layer features_3_conv_2 {
  Type: POINTWISE_CONV
  Dimensions { K: 12, C: 24, Y: 2, X: 2, R: 1, S: 1 }
  Stride { Y: 1, X: 1 }
  Meta { module: "features.3.conv.2", groups: 1, expected_tiles: 1 }
  Dataflow {
    SpatialMap(12, 12) K;
    TemporalMap(16, 16) C;
    TemporalMap(1, 1) R;
    TemporalMap(1, 1) S;
    Cluster(1, P);
  }
}
Layer features_4_conv_0_0 {
  Type: POINTWISE_CONV
  Dimensions { K: 24, C: 12, Y: 2, X: 2, R: 1, S: 1 }
  Stride { Y: 1, X: 1 }
  Meta { module: "features.4.conv.0.0", groups: 1, expected_tiles: 2 }
  Dataflow {
    SpatialMap(16, 16) K;
    TemporalMap(12, 12) C;
    TemporalMap(1, 1) R;
    TemporalMap(1, 1) S;
    Cluster(2, P);
  }
}
Layer features_4_conv_1_0 {
  Type: DEPTHWISE_CONV
  Dimensions { K: 24, C: 24, Y: 1, X: 1, R: 3, S: 3 }
  Stride { Y: 2, X: 2 }
  Meta { module: "features.4.conv.1.0", groups: 24, expected_tiles: 2 }
  Dataflow {
    SpatialMap(16, 16) K;
    TemporalMap(1, 1) C;
    TemporalMap(3, 3) R;
    TemporalMap(3, 3) S;
    Cluster(2, P);
  }
}
Layer features_4_conv_2 {
  Type: POINTWISE_CONV
  Dimensions { K: 16, C: 24, Y: 1, X: 1, R: 1, S: 1 }
  Stride { Y: 1, X: 1 }
  Meta { module: "features.4.conv.2", groups: 1, expected_tiles: 1 }
  Dataflow {
    SpatialMap(16, 16) K;
    TemporalMap(16, 16) C;
    TemporalMap(1, 1) R;
    TemporalMap(1, 1) S;
    Cluster(1, P);
  }
}
Layer features_5_conv_0_0 {
  Type: POINTWISE_CONV
  Dimensions { K: 32, C: 16, Y: 1, X: 1, R: 1, S: 1 }
  Stride { Y: 1, X: 1 }
  Meta { module: "features.5.conv.0.0", groups: 1, expected_tiles: 2 }
  Dataflow {
    SpatialMap(16, 16) K;
    TemporalMap(16, 16) C;
    TemporalMap(1, 1) R;
    TemporalMap(1, 1) S;
    Cluster(2, P);
  }
}
Layer features_5_conv_1_0 {
  Type: DEPTHWISE_CONV
  Dimensions { K: 32, C: 32, Y: 1, X: 1, R: 3, S: 3 }
  Stride { Y: 1, X: 1 }
  Meta { module: "features.5.conv.1.0", groups: 32, expected_tiles: 2 }
  Dataflow {
    SpatialMap(16, 16) K;
    TemporalMap(1, 1) C;
    TemporalMap(3, 3) R;
    TemporalMap(3, 3) S;
    Cluster(2, P);
  }
}
Layer features_5_conv_2 {
  Type: POINTWISE_CONV
  Dimensions { K: 16, C: 32, Y: 1, X: 1, R: 1, S: 1 }
  Stride { Y: 1, X: 1 }
  Meta { module: "features.5.conv.2", groups: 1, expected_tiles: 1 }
  Dataflow {
    SpatialMap(16, 16) K;
    TemporalMap(16, 16) C;
    TemporalMap(1, 1) R;
    TemporalMap(1, 1) S;
    Cluster(1, P);
  }
}
Layer features_6_0 {
  Type: POINTWISE_CONV
  Dimensions { K: 32, C: 16, Y: 1, X: 1, R: 1, S: 1 }
  Stride { Y: 1, X: 1 }
  Meta { module: "features.6.0", groups: 1, expected_tiles: 2 }
  Dataflow {
    SpatialMap(16, 16) K;
    TemporalMap(16, 16) C;
    TemporalMap(1, 1) R;
    TemporalMap(1, 1) S;
    Cluster(2, P);
  }
}
Layer classifier {
  Type: GEMM
  Dimensions { K: 10, C: 32, Y: 1, X: 1, R: 1, S: 1 }
  Stride { Y: 1, X: 1 }
  Meta { module: "classifier", groups: 1, expected_tiles: 1 }
  Dataflow {
    SpatialMap(10, 10) K;
    TemporalMap(16, 16) C;
    TemporalMap(1, 1) R;
    TemporalMap(1, 1) S;
    Cluster(1, P);
  }
}
}
