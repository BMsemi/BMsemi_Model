Network Yolo26nano_Full {
    Layer Conv2D {
        Type: Conv
        Stride { X: 2, Y: 2 }
        Dimensions { K: 16, C: 3, R: 3, S: 3, Y: 640, X: 640 }
        Dataflow {
            // Parsed from CONV_2D op 2.
            SpatialMap(8,8) K;
            TemporalMap(1,1) C;
            TemporalMap(3,3) R;
            TemporalMap(3,3) S;
            TemporalMap(3,1) Y;
            TemporalMap(3,1) X;
        }
    }

    Layer Conv2D_1 {
        Type: Conv
        Stride { X: 2, Y: 2 }
        Dimensions { K: 32, C: 16, R: 3, S: 3, Y: 320, X: 320 }
        Dataflow {
            // Parsed from CONV_2D op 6.
            SpatialMap(8,8) K;
            TemporalMap(1,1) C;
            TemporalMap(3,3) R;
            TemporalMap(3,3) S;
            TemporalMap(3,1) Y;
            TemporalMap(3,1) X;
        }
    }

    Layer Conv2D_2 {
        Type: Conv
        Stride { X: 1, Y: 1 }
        Dimensions { K: 32, C: 32, R: 1, S: 1, Y: 160, X: 160 }
        Dataflow {
            // Parsed from CONV_2D op 9.
            SpatialMap(8,8) K;
            TemporalMap(1,1) C;
            TemporalMap(1,1) Y;
            TemporalMap(1,1) X;
            TemporalMap(1,1) R;
            TemporalMap(1,1) S;
        }
    }

    Layer Conv2D_3 {
        Type: Conv
        Stride { X: 1, Y: 1 }
        Dimensions { K: 8, C: 16, R: 3, S: 3, Y: 160, X: 160 }
        Dataflow {
            // Parsed from CONV_2D op 13.
            SpatialMap(8,8) K;
            TemporalMap(1,1) C;
            TemporalMap(3,3) R;
            TemporalMap(3,3) S;
            TemporalMap(3,1) Y;
            TemporalMap(3,1) X;
        }
    }

    Layer Conv2D_4 {
        Type: Conv
        Stride { X: 1, Y: 1 }
        Dimensions { K: 16, C: 8, R: 3, S: 3, Y: 160, X: 160 }
        Dataflow {
            // Parsed from CONV_2D op 16.
            SpatialMap(8,8) K;
            TemporalMap(1,1) C;
            TemporalMap(3,3) R;
            TemporalMap(3,3) S;
            TemporalMap(3,1) Y;
            TemporalMap(3,1) X;
        }
    }

    Layer Conv2D_5 {
        Type: Conv
        Stride { X: 1, Y: 1 }
        Dimensions { K: 64, C: 48, R: 1, S: 1, Y: 160, X: 160 }
        Dataflow {
            // Parsed from CONV_2D op 22.
            SpatialMap(8,8) K;
            TemporalMap(1,1) C;
            TemporalMap(1,1) Y;
            TemporalMap(1,1) X;
            TemporalMap(1,1) R;
            TemporalMap(1,1) S;
        }
    }

    Layer Conv2D_6 {
        Type: Conv
        Stride { X: 2, Y: 2 }
        Dimensions { K: 64, C: 64, R: 3, S: 3, Y: 160, X: 160 }
        Dataflow {
            // Parsed from CONV_2D op 26.
            SpatialMap(8,8) K;
            TemporalMap(1,1) C;
            TemporalMap(3,3) R;
            TemporalMap(3,3) S;
            TemporalMap(3,1) Y;
            TemporalMap(3,1) X;
        }
    }

    Layer Conv2D_7 {
        Type: Conv
        Stride { X: 1, Y: 1 }
        Dimensions { K: 64, C: 64, R: 1, S: 1, Y: 80, X: 80 }
        Dataflow {
            // Parsed from CONV_2D op 29.
            SpatialMap(8,8) K;
            TemporalMap(1,1) C;
            TemporalMap(1,1) Y;
            TemporalMap(1,1) X;
            TemporalMap(1,1) R;
            TemporalMap(1,1) S;
        }
    }

    Layer Conv2D_8 {
        Type: Conv
        Stride { X: 1, Y: 1 }
        Dimensions { K: 16, C: 32, R: 3, S: 3, Y: 80, X: 80 }
        Dataflow {
            // Parsed from CONV_2D op 33.
            SpatialMap(8,8) K;
            TemporalMap(1,1) C;
            TemporalMap(3,3) R;
            TemporalMap(3,3) S;
            TemporalMap(3,1) Y;
            TemporalMap(3,1) X;
        }
    }

    Layer Conv2D_9 {
        Type: Conv
        Stride { X: 1, Y: 1 }
        Dimensions { K: 32, C: 16, R: 3, S: 3, Y: 80, X: 80 }
        Dataflow {
            // Parsed from CONV_2D op 36.
            SpatialMap(8,8) K;
            TemporalMap(1,1) C;
            TemporalMap(3,3) R;
            TemporalMap(3,3) S;
            TemporalMap(3,1) Y;
            TemporalMap(3,1) X;
        }
    }

    Layer Conv2D_10 {
        Type: Conv
        Stride { X: 1, Y: 1 }
        Dimensions { K: 128, C: 96, R: 1, S: 1, Y: 80, X: 80 }
        Dataflow {
            // Parsed from CONV_2D op 42.
            SpatialMap(8,8) K;
            TemporalMap(1,1) C;
            TemporalMap(1,1) Y;
            TemporalMap(1,1) X;
            TemporalMap(1,1) R;
            TemporalMap(1,1) S;
        }
    }

    Layer Conv2D_11 {
        Type: Conv
        Stride { X: 2, Y: 2 }
        Dimensions { K: 128, C: 128, R: 3, S: 3, Y: 80, X: 80 }
        Dataflow {
            // Parsed from CONV_2D op 46.
            SpatialMap(8,8) K;
            TemporalMap(1,1) C;
            TemporalMap(3,3) R;
            TemporalMap(3,3) S;
            TemporalMap(3,1) Y;
            TemporalMap(3,1) X;
        }
    }

    Layer Conv2D_12 {
        Type: Conv
        Stride { X: 1, Y: 1 }
        Dimensions { K: 128, C: 128, R: 1, S: 1, Y: 40, X: 40 }
        Dataflow {
            // Parsed from CONV_2D op 49.
            SpatialMap(8,8) K;
            TemporalMap(1,1) C;
            TemporalMap(1,1) Y;
            TemporalMap(1,1) X;
            TemporalMap(1,1) R;
            TemporalMap(1,1) S;
        }
    }

    Layer Conv2D_13 {
        Type: Conv
        Stride { X: 1, Y: 1 }
        Dimensions { K: 32, C: 64, R: 1, S: 1, Y: 40, X: 40 }
        Dataflow {
            // Parsed from CONV_2D op 53.
            SpatialMap(8,8) K;
            TemporalMap(1,1) C;
            TemporalMap(1,1) Y;
            TemporalMap(1,1) X;
            TemporalMap(1,1) R;
            TemporalMap(1,1) S;
        }
    }

    Layer Conv2D_14 {
        Type: Conv
        Stride { X: 1, Y: 1 }
        Dimensions { K: 32, C: 32, R: 3, S: 3, Y: 40, X: 40 }
        Dataflow {
            // Parsed from CONV_2D op 56.
            SpatialMap(8,8) K;
            TemporalMap(1,1) C;
            TemporalMap(3,3) R;
            TemporalMap(3,3) S;
            TemporalMap(3,1) Y;
            TemporalMap(3,1) X;
        }
    }

    Layer Conv2D_15 {
        Type: Conv
        Stride { X: 1, Y: 1 }
        Dimensions { K: 32, C: 32, R: 3, S: 3, Y: 40, X: 40 }
        Dataflow {
            // Parsed from CONV_2D op 59.
            SpatialMap(8,8) K;
            TemporalMap(1,1) C;
            TemporalMap(3,3) R;
            TemporalMap(3,3) S;
            TemporalMap(3,1) Y;
            TemporalMap(3,1) X;
        }
    }

    Layer Conv2D_16 {
        Type: Conv
        Stride { X: 1, Y: 1 }
        Dimensions { K: 32, C: 32, R: 3, S: 3, Y: 40, X: 40 }
        Dataflow {
            // Parsed from CONV_2D op 63.
            SpatialMap(8,8) K;
            TemporalMap(1,1) C;
            TemporalMap(3,3) R;
            TemporalMap(3,3) S;
            TemporalMap(3,1) Y;
            TemporalMap(3,1) X;
        }
    }

    Layer Conv2D_17 {
        Type: Conv
        Stride { X: 1, Y: 1 }
        Dimensions { K: 32, C: 32, R: 3, S: 3, Y: 40, X: 40 }
        Dataflow {
            // Parsed from CONV_2D op 66.
            SpatialMap(8,8) K;
            TemporalMap(1,1) C;
            TemporalMap(3,3) R;
            TemporalMap(3,3) S;
            TemporalMap(3,1) Y;
            TemporalMap(3,1) X;
        }
    }

    Layer Conv2D_18 {
        Type: Conv
        Stride { X: 1, Y: 1 }
        Dimensions { K: 32, C: 64, R: 1, S: 1, Y: 40, X: 40 }
        Dataflow {
            // Parsed from CONV_2D op 70.
            SpatialMap(8,8) K;
            TemporalMap(1,1) C;
            TemporalMap(1,1) Y;
            TemporalMap(1,1) X;
            TemporalMap(1,1) R;
            TemporalMap(1,1) S;
        }
    }

    Layer Conv2D_19 {
        Type: Conv
        Stride { X: 1, Y: 1 }
        Dimensions { K: 64, C: 64, R: 1, S: 1, Y: 40, X: 40 }
        Dataflow {
            // Parsed from CONV_2D op 74.
            SpatialMap(8,8) K;
            TemporalMap(1,1) C;
            TemporalMap(1,1) Y;
            TemporalMap(1,1) X;
            TemporalMap(1,1) R;
            TemporalMap(1,1) S;
        }
    }

    Layer Conv2D_20 {
        Type: Conv
        Stride { X: 1, Y: 1 }
        Dimensions { K: 128, C: 192, R: 1, S: 1, Y: 40, X: 40 }
        Dataflow {
            // Parsed from CONV_2D op 79.
            SpatialMap(8,8) K;
            TemporalMap(1,1) C;
            TemporalMap(1,1) Y;
            TemporalMap(1,1) X;
            TemporalMap(1,1) R;
            TemporalMap(1,1) S;
        }
    }

    Layer Conv2D_21 {
        Type: Conv
        Stride { X: 2, Y: 2 }
        Dimensions { K: 256, C: 128, R: 3, S: 3, Y: 40, X: 40 }
        Dataflow {
            // Parsed from CONV_2D op 83.
            SpatialMap(8,8) K;
            TemporalMap(1,1) C;
            TemporalMap(3,3) R;
            TemporalMap(3,3) S;
            TemporalMap(3,1) Y;
            TemporalMap(3,1) X;
        }
    }

    Layer Conv2D_22 {
        Type: Conv
        Stride { X: 1, Y: 1 }
        Dimensions { K: 256, C: 256, R: 1, S: 1, Y: 20, X: 20 }
        Dataflow {
            // Parsed from CONV_2D op 86.
            SpatialMap(8,8) K;
            TemporalMap(1,1) C;
            TemporalMap(1,1) Y;
            TemporalMap(1,1) X;
            TemporalMap(1,1) R;
            TemporalMap(1,1) S;
        }
    }

    Layer Conv2D_23 {
        Type: Conv
        Stride { X: 1, Y: 1 }
        Dimensions { K: 64, C: 128, R: 1, S: 1, Y: 20, X: 20 }
        Dataflow {
            // Parsed from CONV_2D op 90.
            SpatialMap(8,8) K;
            TemporalMap(1,1) C;
            TemporalMap(1,1) Y;
            TemporalMap(1,1) X;
            TemporalMap(1,1) R;
            TemporalMap(1,1) S;
        }
    }

    Layer Conv2D_24 {
        Type: Conv
        Stride { X: 1, Y: 1 }
        Dimensions { K: 64, C: 64, R: 3, S: 3, Y: 20, X: 20 }
        Dataflow {
            // Parsed from CONV_2D op 93.
            SpatialMap(8,8) K;
            TemporalMap(1,1) C;
            TemporalMap(3,3) R;
            TemporalMap(3,3) S;
            TemporalMap(3,1) Y;
            TemporalMap(3,1) X;
        }
    }

    Layer Conv2D_25 {
        Type: Conv
        Stride { X: 1, Y: 1 }
        Dimensions { K: 64, C: 64, R: 3, S: 3, Y: 20, X: 20 }
        Dataflow {
            // Parsed from CONV_2D op 96.
            SpatialMap(8,8) K;
            TemporalMap(1,1) C;
            TemporalMap(3,3) R;
            TemporalMap(3,3) S;
            TemporalMap(3,1) Y;
            TemporalMap(3,1) X;
        }
    }

    Layer Conv2D_26 {
        Type: Conv
        Stride { X: 1, Y: 1 }
        Dimensions { K: 64, C: 64, R: 3, S: 3, Y: 20, X: 20 }
        Dataflow {
            // Parsed from CONV_2D op 100.
            SpatialMap(8,8) K;
            TemporalMap(1,1) C;
            TemporalMap(3,3) R;
            TemporalMap(3,3) S;
            TemporalMap(3,1) Y;
            TemporalMap(3,1) X;
        }
    }

    Layer Conv2D_27 {
        Type: Conv
        Stride { X: 1, Y: 1 }
        Dimensions { K: 64, C: 64, R: 3, S: 3, Y: 20, X: 20 }
        Dataflow {
            // Parsed from CONV_2D op 103.
            SpatialMap(8,8) K;
            TemporalMap(1,1) C;
            TemporalMap(3,3) R;
            TemporalMap(3,3) S;
            TemporalMap(3,1) Y;
            TemporalMap(3,1) X;
        }
    }

    Layer Conv2D_28 {
        Type: Conv
        Stride { X: 1, Y: 1 }
        Dimensions { K: 64, C: 128, R: 1, S: 1, Y: 20, X: 20 }
        Dataflow {
            // Parsed from CONV_2D op 107.
            SpatialMap(8,8) K;
            TemporalMap(1,1) C;
            TemporalMap(1,1) Y;
            TemporalMap(1,1) X;
            TemporalMap(1,1) R;
            TemporalMap(1,1) S;
        }
    }

    Layer Conv2D_29 {
        Type: Conv
        Stride { X: 1, Y: 1 }
        Dimensions { K: 128, C: 128, R: 1, S: 1, Y: 20, X: 20 }
        Dataflow {
            // Parsed from CONV_2D op 111.
            SpatialMap(8,8) K;
            TemporalMap(1,1) C;
            TemporalMap(1,1) Y;
            TemporalMap(1,1) X;
            TemporalMap(1,1) R;
            TemporalMap(1,1) S;
        }
    }

    Layer Conv2D_30 {
        Type: Conv
        Stride { X: 1, Y: 1 }
        Dimensions { K: 256, C: 384, R: 1, S: 1, Y: 20, X: 20 }
        Dataflow {
            // Parsed from CONV_2D op 116.
            SpatialMap(8,8) K;
            TemporalMap(1,1) C;
            TemporalMap(1,1) Y;
            TemporalMap(1,1) X;
            TemporalMap(1,1) R;
            TemporalMap(1,1) S;
        }
    }

    Layer Conv2D_31 {
        Type: Conv
        Stride { X: 1, Y: 1 }
        Dimensions { K: 128, C: 256, R: 1, S: 1, Y: 20, X: 20 }
        Dataflow {
            // Parsed from CONV_2D op 119.
            SpatialMap(8,8) K;
            TemporalMap(1,1) C;
            TemporalMap(1,1) Y;
            TemporalMap(1,1) X;
            TemporalMap(1,1) R;
            TemporalMap(1,1) S;
        }
    }

    Layer Conv2D_32 {
        Type: Conv
        Stride { X: 1, Y: 1 }
        Dimensions { K: 256, C: 512, R: 1, S: 1, Y: 20, X: 20 }
        Dataflow {
            // Parsed from CONV_2D op 124.
            SpatialMap(8,8) K;
            TemporalMap(1,1) C;
            TemporalMap(1,1) Y;
            TemporalMap(1,1) X;
            TemporalMap(1,1) R;
            TemporalMap(1,1) S;
        }
    }

    Layer Conv2D_33 {
        Type: Conv
        Stride { X: 1, Y: 1 }
        Dimensions { K: 256, C: 256, R: 1, S: 1, Y: 20, X: 20 }
        Dataflow {
            // Parsed from CONV_2D op 128.
            SpatialMap(8,8) K;
            TemporalMap(1,1) C;
            TemporalMap(1,1) Y;
            TemporalMap(1,1) X;
            TemporalMap(1,1) R;
            TemporalMap(1,1) S;
        }
    }

    Layer Conv2D_34 {
        Type: Conv
        Stride { X: 1, Y: 1 }
        Dimensions { K: 256, C: 128, R: 1, S: 1, Y: 20, X: 20 }
        Dataflow {
            // Parsed from CONV_2D op 132.
            SpatialMap(8,8) K;
            TemporalMap(1,1) C;
            TemporalMap(1,1) Y;
            TemporalMap(1,1) X;
            TemporalMap(1,1) R;
            TemporalMap(1,1) S;
        }
    }

    Layer DepthwiseConv2D_35 {
        Type: DepthwiseConv
        Stride { X: 1, Y: 1 }
        Dimensions { K: 128, C: 1, R: 3, S: 3, Y: 20, X: 20 }
        Dataflow {
            // Parsed from DEPTHWISE_CONV_2D op 152.
            SpatialMap(8,8) K;
            TemporalMap(1,1) C;
            TemporalMap(3,3) R;
            TemporalMap(3,3) S;
            TemporalMap(3,1) Y;
            TemporalMap(3,1) X;
        }
    }

    Layer Conv2D_36 {
        Type: Conv
        Stride { X: 1, Y: 1 }
        Dimensions { K: 128, C: 128, R: 1, S: 1, Y: 20, X: 20 }
        Dataflow {
            // Parsed from CONV_2D op 156.
            SpatialMap(8,8) K;
            TemporalMap(1,1) C;
            TemporalMap(1,1) Y;
            TemporalMap(1,1) X;
            TemporalMap(1,1) R;
            TemporalMap(1,1) S;
        }
    }

    Layer Conv2D_37 {
        Type: Conv
        Stride { X: 1, Y: 1 }
        Dimensions { K: 256, C: 128, R: 1, S: 1, Y: 20, X: 20 }
        Dataflow {
            // Parsed from CONV_2D op 158.
            SpatialMap(8,8) K;
            TemporalMap(1,1) C;
            TemporalMap(1,1) Y;
            TemporalMap(1,1) X;
            TemporalMap(1,1) R;
            TemporalMap(1,1) S;
        }
    }

    Layer Conv2D_38 {
        Type: Conv
        Stride { X: 1, Y: 1 }
        Dimensions { K: 128, C: 256, R: 1, S: 1, Y: 20, X: 20 }
        Dataflow {
            // Parsed from CONV_2D op 161.
            SpatialMap(8,8) K;
            TemporalMap(1,1) C;
            TemporalMap(1,1) Y;
            TemporalMap(1,1) X;
            TemporalMap(1,1) R;
            TemporalMap(1,1) S;
        }
    }

    Layer Conv2D_39 {
        Type: Conv
        Stride { X: 1, Y: 1 }
        Dimensions { K: 256, C: 256, R: 1, S: 1, Y: 20, X: 20 }
        Dataflow {
            // Parsed from CONV_2D op 165.
            SpatialMap(8,8) K;
            TemporalMap(1,1) C;
            TemporalMap(1,1) Y;
            TemporalMap(1,1) X;
            TemporalMap(1,1) R;
            TemporalMap(1,1) S;
        }
    }

    Layer Conv2D_40 {
        Type: Conv
        Stride { X: 1, Y: 1 }
        Dimensions { K: 128, C: 384, R: 1, S: 1, Y: 40, X: 40 }
        Dataflow {
            // Parsed from CONV_2D op 170.
            SpatialMap(8,8) K;
            TemporalMap(1,1) C;
            TemporalMap(1,1) Y;
            TemporalMap(1,1) X;
            TemporalMap(1,1) R;
            TemporalMap(1,1) S;
        }
    }

    Layer Conv2D_41 {
        Type: Conv
        Stride { X: 1, Y: 1 }
        Dimensions { K: 32, C: 64, R: 1, S: 1, Y: 40, X: 40 }
        Dataflow {
            // Parsed from CONV_2D op 174.
            SpatialMap(8,8) K;
            TemporalMap(1,1) C;
            TemporalMap(1,1) Y;
            TemporalMap(1,1) X;
            TemporalMap(1,1) R;
            TemporalMap(1,1) S;
        }
    }

    Layer Conv2D_42 {
        Type: Conv
        Stride { X: 1, Y: 1 }
        Dimensions { K: 32, C: 32, R: 3, S: 3, Y: 40, X: 40 }
        Dataflow {
            // Parsed from CONV_2D op 177.
            SpatialMap(8,8) K;
            TemporalMap(1,1) C;
            TemporalMap(3,3) R;
            TemporalMap(3,3) S;
            TemporalMap(3,1) Y;
            TemporalMap(3,1) X;
        }
    }

    Layer Conv2D_43 {
        Type: Conv
        Stride { X: 1, Y: 1 }
        Dimensions { K: 32, C: 32, R: 3, S: 3, Y: 40, X: 40 }
        Dataflow {
            // Parsed from CONV_2D op 180.
            SpatialMap(8,8) K;
            TemporalMap(1,1) C;
            TemporalMap(3,3) R;
            TemporalMap(3,3) S;
            TemporalMap(3,1) Y;
            TemporalMap(3,1) X;
        }
    }

    Layer Conv2D_44 {
        Type: Conv
        Stride { X: 1, Y: 1 }
        Dimensions { K: 32, C: 32, R: 3, S: 3, Y: 40, X: 40 }
        Dataflow {
            // Parsed from CONV_2D op 184.
            SpatialMap(8,8) K;
            TemporalMap(1,1) C;
            TemporalMap(3,3) R;
            TemporalMap(3,3) S;
            TemporalMap(3,1) Y;
            TemporalMap(3,1) X;
        }
    }

    Layer Conv2D_45 {
        Type: Conv
        Stride { X: 1, Y: 1 }
        Dimensions { K: 32, C: 32, R: 3, S: 3, Y: 40, X: 40 }
        Dataflow {
            // Parsed from CONV_2D op 187.
            SpatialMap(8,8) K;
            TemporalMap(1,1) C;
            TemporalMap(3,3) R;
            TemporalMap(3,3) S;
            TemporalMap(3,1) Y;
            TemporalMap(3,1) X;
        }
    }

    Layer Conv2D_46 {
        Type: Conv
        Stride { X: 1, Y: 1 }
        Dimensions { K: 32, C: 64, R: 1, S: 1, Y: 40, X: 40 }
        Dataflow {
            // Parsed from CONV_2D op 191.
            SpatialMap(8,8) K;
            TemporalMap(1,1) C;
            TemporalMap(1,1) Y;
            TemporalMap(1,1) X;
            TemporalMap(1,1) R;
            TemporalMap(1,1) S;
        }
    }

    Layer Conv2D_47 {
        Type: Conv
        Stride { X: 1, Y: 1 }
        Dimensions { K: 64, C: 64, R: 1, S: 1, Y: 40, X: 40 }
        Dataflow {
            // Parsed from CONV_2D op 195.
            SpatialMap(8,8) K;
            TemporalMap(1,1) C;
            TemporalMap(1,1) Y;
            TemporalMap(1,1) X;
            TemporalMap(1,1) R;
            TemporalMap(1,1) S;
        }
    }

    Layer Conv2D_48 {
        Type: Conv
        Stride { X: 1, Y: 1 }
        Dimensions { K: 128, C: 192, R: 1, S: 1, Y: 40, X: 40 }
        Dataflow {
            // Parsed from CONV_2D op 200.
            SpatialMap(8,8) K;
            TemporalMap(1,1) C;
            TemporalMap(1,1) Y;
            TemporalMap(1,1) X;
            TemporalMap(1,1) R;
            TemporalMap(1,1) S;
        }
    }

    Layer Conv2D_49 {
        Type: Conv
        Stride { X: 1, Y: 1 }
        Dimensions { K: 64, C: 256, R: 1, S: 1, Y: 80, X: 80 }
        Dataflow {
            // Parsed from CONV_2D op 205.
            SpatialMap(8,8) K;
            TemporalMap(1,1) C;
            TemporalMap(1,1) Y;
            TemporalMap(1,1) X;
            TemporalMap(1,1) R;
            TemporalMap(1,1) S;
        }
    }

    Layer Conv2D_50 {
        Type: Conv
        Stride { X: 1, Y: 1 }
        Dimensions { K: 16, C: 32, R: 1, S: 1, Y: 80, X: 80 }
        Dataflow {
            // Parsed from CONV_2D op 209.
            SpatialMap(8,8) K;
            TemporalMap(1,1) C;
            TemporalMap(1,1) Y;
            TemporalMap(1,1) X;
            TemporalMap(1,1) R;
            TemporalMap(1,1) S;
        }
    }

    Layer Conv2D_51 {
        Type: Conv
        Stride { X: 1, Y: 1 }
        Dimensions { K: 16, C: 16, R: 3, S: 3, Y: 80, X: 80 }
        Dataflow {
            // Parsed from CONV_2D op 212.
            SpatialMap(8,8) K;
            TemporalMap(1,1) C;
            TemporalMap(3,3) R;
            TemporalMap(3,3) S;
            TemporalMap(3,1) Y;
            TemporalMap(3,1) X;
        }
    }

    Layer Conv2D_52 {
        Type: Conv
        Stride { X: 1, Y: 1 }
        Dimensions { K: 16, C: 16, R: 3, S: 3, Y: 80, X: 80 }
        Dataflow {
            // Parsed from CONV_2D op 215.
            SpatialMap(8,8) K;
            TemporalMap(1,1) C;
            TemporalMap(3,3) R;
            TemporalMap(3,3) S;
            TemporalMap(3,1) Y;
            TemporalMap(3,1) X;
        }
    }

    Layer Conv2D_53 {
        Type: Conv
        Stride { X: 1, Y: 1 }
        Dimensions { K: 16, C: 16, R: 3, S: 3, Y: 80, X: 80 }
        Dataflow {
            // Parsed from CONV_2D op 219.
            SpatialMap(8,8) K;
            TemporalMap(1,1) C;
            TemporalMap(3,3) R;
            TemporalMap(3,3) S;
            TemporalMap(3,1) Y;
            TemporalMap(3,1) X;
        }
    }

    Layer Conv2D_54 {
        Type: Conv
        Stride { X: 1, Y: 1 }
        Dimensions { K: 16, C: 16, R: 3, S: 3, Y: 80, X: 80 }
        Dataflow {
            // Parsed from CONV_2D op 222.
            SpatialMap(8,8) K;
            TemporalMap(1,1) C;
            TemporalMap(3,3) R;
            TemporalMap(3,3) S;
            TemporalMap(3,1) Y;
            TemporalMap(3,1) X;
        }
    }

    Layer Conv2D_55 {
        Type: Conv
        Stride { X: 1, Y: 1 }
        Dimensions { K: 16, C: 32, R: 1, S: 1, Y: 80, X: 80 }
        Dataflow {
            // Parsed from CONV_2D op 226.
            SpatialMap(8,8) K;
            TemporalMap(1,1) C;
            TemporalMap(1,1) Y;
            TemporalMap(1,1) X;
            TemporalMap(1,1) R;
            TemporalMap(1,1) S;
        }
    }

    Layer Conv2D_56 {
        Type: Conv
        Stride { X: 1, Y: 1 }
        Dimensions { K: 32, C: 32, R: 1, S: 1, Y: 80, X: 80 }
        Dataflow {
            // Parsed from CONV_2D op 230.
            SpatialMap(8,8) K;
            TemporalMap(1,1) C;
            TemporalMap(1,1) Y;
            TemporalMap(1,1) X;
            TemporalMap(1,1) R;
            TemporalMap(1,1) S;
        }
    }

    Layer Conv2D_57 {
        Type: Conv
        Stride { X: 1, Y: 1 }
        Dimensions { K: 64, C: 96, R: 1, S: 1, Y: 80, X: 80 }
        Dataflow {
            // Parsed from CONV_2D op 235.
            SpatialMap(8,8) K;
            TemporalMap(1,1) C;
            TemporalMap(1,1) Y;
            TemporalMap(1,1) X;
            TemporalMap(1,1) R;
            TemporalMap(1,1) S;
        }
    }

    Layer Conv2D_58 {
        Type: Conv
        Stride { X: 2, Y: 2 }
        Dimensions { K: 64, C: 64, R: 3, S: 3, Y: 80, X: 80 }
        Dataflow {
            // Parsed from CONV_2D op 239.
            SpatialMap(8,8) K;
            TemporalMap(1,1) C;
            TemporalMap(3,3) R;
            TemporalMap(3,3) S;
            TemporalMap(3,1) Y;
            TemporalMap(3,1) X;
        }
    }

    Layer Conv2D_59 {
        Type: Conv
        Stride { X: 1, Y: 1 }
        Dimensions { K: 128, C: 192, R: 1, S: 1, Y: 40, X: 40 }
        Dataflow {
            // Parsed from CONV_2D op 243.
            SpatialMap(8,8) K;
            TemporalMap(1,1) C;
            TemporalMap(1,1) Y;
            TemporalMap(1,1) X;
            TemporalMap(1,1) R;
            TemporalMap(1,1) S;
        }
    }

    Layer Conv2D_60 {
        Type: Conv
        Stride { X: 1, Y: 1 }
        Dimensions { K: 32, C: 64, R: 1, S: 1, Y: 40, X: 40 }
        Dataflow {
            // Parsed from CONV_2D op 247.
            SpatialMap(8,8) K;
            TemporalMap(1,1) C;
            TemporalMap(1,1) Y;
            TemporalMap(1,1) X;
            TemporalMap(1,1) R;
            TemporalMap(1,1) S;
        }
    }

    Layer Conv2D_61 {
        Type: Conv
        Stride { X: 1, Y: 1 }
        Dimensions { K: 32, C: 32, R: 3, S: 3, Y: 40, X: 40 }
        Dataflow {
            // Parsed from CONV_2D op 250.
            SpatialMap(8,8) K;
            TemporalMap(1,1) C;
            TemporalMap(3,3) R;
            TemporalMap(3,3) S;
            TemporalMap(3,1) Y;
            TemporalMap(3,1) X;
        }
    }

    Layer Conv2D_62 {
        Type: Conv
        Stride { X: 1, Y: 1 }
        Dimensions { K: 32, C: 32, R: 3, S: 3, Y: 40, X: 40 }
        Dataflow {
            // Parsed from CONV_2D op 253.
            SpatialMap(8,8) K;
            TemporalMap(1,1) C;
            TemporalMap(3,3) R;
            TemporalMap(3,3) S;
            TemporalMap(3,1) Y;
            TemporalMap(3,1) X;
        }
    }

    Layer Conv2D_63 {
        Type: Conv
        Stride { X: 1, Y: 1 }
        Dimensions { K: 32, C: 32, R: 3, S: 3, Y: 40, X: 40 }
        Dataflow {
            // Parsed from CONV_2D op 257.
            SpatialMap(8,8) K;
            TemporalMap(1,1) C;
            TemporalMap(3,3) R;
            TemporalMap(3,3) S;
            TemporalMap(3,1) Y;
            TemporalMap(3,1) X;
        }
    }

    Layer Conv2D_64 {
        Type: Conv
        Stride { X: 1, Y: 1 }
        Dimensions { K: 32, C: 32, R: 3, S: 3, Y: 40, X: 40 }
        Dataflow {
            // Parsed from CONV_2D op 260.
            SpatialMap(8,8) K;
            TemporalMap(1,1) C;
            TemporalMap(3,3) R;
            TemporalMap(3,3) S;
            TemporalMap(3,1) Y;
            TemporalMap(3,1) X;
        }
    }

    Layer Conv2D_65 {
        Type: Conv
        Stride { X: 1, Y: 1 }
        Dimensions { K: 32, C: 64, R: 1, S: 1, Y: 40, X: 40 }
        Dataflow {
            // Parsed from CONV_2D op 264.
            SpatialMap(8,8) K;
            TemporalMap(1,1) C;
            TemporalMap(1,1) Y;
            TemporalMap(1,1) X;
            TemporalMap(1,1) R;
            TemporalMap(1,1) S;
        }
    }

    Layer Conv2D_66 {
        Type: Conv
        Stride { X: 1, Y: 1 }
        Dimensions { K: 64, C: 64, R: 1, S: 1, Y: 40, X: 40 }
        Dataflow {
            // Parsed from CONV_2D op 268.
            SpatialMap(8,8) K;
            TemporalMap(1,1) C;
            TemporalMap(1,1) Y;
            TemporalMap(1,1) X;
            TemporalMap(1,1) R;
            TemporalMap(1,1) S;
        }
    }

    Layer Conv2D_67 {
        Type: Conv
        Stride { X: 1, Y: 1 }
        Dimensions { K: 128, C: 192, R: 1, S: 1, Y: 40, X: 40 }
        Dataflow {
            // Parsed from CONV_2D op 273.
            SpatialMap(8,8) K;
            TemporalMap(1,1) C;
            TemporalMap(1,1) Y;
            TemporalMap(1,1) X;
            TemporalMap(1,1) R;
            TemporalMap(1,1) S;
        }
    }

    Layer Conv2D_68 {
        Type: Conv
        Stride { X: 2, Y: 2 }
        Dimensions { K: 128, C: 128, R: 3, S: 3, Y: 40, X: 40 }
        Dataflow {
            // Parsed from CONV_2D op 277.
            SpatialMap(8,8) K;
            TemporalMap(1,1) C;
            TemporalMap(3,3) R;
            TemporalMap(3,3) S;
            TemporalMap(3,1) Y;
            TemporalMap(3,1) X;
        }
    }

    Layer Conv2D_69 {
        Type: Conv
        Stride { X: 1, Y: 1 }
        Dimensions { K: 256, C: 384, R: 1, S: 1, Y: 20, X: 20 }
        Dataflow {
            // Parsed from CONV_2D op 281.
            SpatialMap(8,8) K;
            TemporalMap(1,1) C;
            TemporalMap(1,1) Y;
            TemporalMap(1,1) X;
            TemporalMap(1,1) R;
            TemporalMap(1,1) S;
        }
    }

    Layer Conv2D_70 {
        Type: Conv
        Stride { X: 1, Y: 1 }
        Dimensions { K: 64, C: 128, R: 3, S: 3, Y: 20, X: 20 }
        Dataflow {
            // Parsed from CONV_2D op 285.
            SpatialMap(8,8) K;
            TemporalMap(1,1) C;
            TemporalMap(3,3) R;
            TemporalMap(3,3) S;
            TemporalMap(3,1) Y;
            TemporalMap(3,1) X;
        }
    }

    Layer Conv2D_71 {
        Type: Conv
        Stride { X: 1, Y: 1 }
        Dimensions { K: 128, C: 64, R: 3, S: 3, Y: 20, X: 20 }
        Dataflow {
            // Parsed from CONV_2D op 288.
            SpatialMap(8,8) K;
            TemporalMap(1,1) C;
            TemporalMap(3,3) R;
            TemporalMap(3,3) S;
            TemporalMap(3,1) Y;
            TemporalMap(3,1) X;
        }
    }

    Layer Conv2D_72 {
        Type: Conv
        Stride { X: 1, Y: 1 }
        Dimensions { K: 256, C: 128, R: 1, S: 1, Y: 20, X: 20 }
        Dataflow {
            // Parsed from CONV_2D op 292.
            SpatialMap(8,8) K;
            TemporalMap(1,1) C;
            TemporalMap(1,1) Y;
            TemporalMap(1,1) X;
            TemporalMap(1,1) R;
            TemporalMap(1,1) S;
        }
    }

    Layer DepthwiseConv2D_73 {
        Type: DepthwiseConv
        Stride { X: 1, Y: 1 }
        Dimensions { K: 128, C: 1, R: 3, S: 3, Y: 20, X: 20 }
        Dataflow {
            // Parsed from DEPTHWISE_CONV_2D op 312.
            SpatialMap(8,8) K;
            TemporalMap(1,1) C;
            TemporalMap(3,3) R;
            TemporalMap(3,3) S;
            TemporalMap(3,1) Y;
            TemporalMap(3,1) X;
        }
    }

    Layer Conv2D_74 {
        Type: Conv
        Stride { X: 1, Y: 1 }
        Dimensions { K: 128, C: 128, R: 1, S: 1, Y: 20, X: 20 }
        Dataflow {
            // Parsed from CONV_2D op 316.
            SpatialMap(8,8) K;
            TemporalMap(1,1) C;
            TemporalMap(1,1) Y;
            TemporalMap(1,1) X;
            TemporalMap(1,1) R;
            TemporalMap(1,1) S;
        }
    }

    Layer Conv2D_75 {
        Type: Conv
        Stride { X: 1, Y: 1 }
        Dimensions { K: 256, C: 128, R: 1, S: 1, Y: 20, X: 20 }
        Dataflow {
            // Parsed from CONV_2D op 318.
            SpatialMap(8,8) K;
            TemporalMap(1,1) C;
            TemporalMap(1,1) Y;
            TemporalMap(1,1) X;
            TemporalMap(1,1) R;
            TemporalMap(1,1) S;
        }
    }

    Layer Conv2D_76 {
        Type: Conv
        Stride { X: 1, Y: 1 }
        Dimensions { K: 128, C: 256, R: 1, S: 1, Y: 20, X: 20 }
        Dataflow {
            // Parsed from CONV_2D op 321.
            SpatialMap(8,8) K;
            TemporalMap(1,1) C;
            TemporalMap(1,1) Y;
            TemporalMap(1,1) X;
            TemporalMap(1,1) R;
            TemporalMap(1,1) S;
        }
    }

    Layer Conv2D_77 {
        Type: Conv
        Stride { X: 1, Y: 1 }
        Dimensions { K: 256, C: 384, R: 1, S: 1, Y: 20, X: 20 }
        Dataflow {
            // Parsed from CONV_2D op 325.
            SpatialMap(8,8) K;
            TemporalMap(1,1) C;
            TemporalMap(1,1) Y;
            TemporalMap(1,1) X;
            TemporalMap(1,1) R;
            TemporalMap(1,1) S;
        }
    }

    Layer Conv2D_78 {
        Type: Conv
        Stride { X: 1, Y: 1 }
        Dimensions { K: 16, C: 64, R: 3, S: 3, Y: 80, X: 80 }
        Dataflow {
            // Parsed from CONV_2D op 328.
            SpatialMap(8,8) K;
            TemporalMap(1,1) C;
            TemporalMap(3,3) R;
            TemporalMap(3,3) S;
            TemporalMap(3,1) Y;
            TemporalMap(3,1) X;
        }
    }

    Layer Conv2D_79 {
        Type: Conv
        Stride { X: 1, Y: 1 }
        Dimensions { K: 16, C: 16, R: 3, S: 3, Y: 80, X: 80 }
        Dataflow {
            // Parsed from CONV_2D op 331.
            SpatialMap(8,8) K;
            TemporalMap(1,1) C;
            TemporalMap(3,3) R;
            TemporalMap(3,3) S;
            TemporalMap(3,1) Y;
            TemporalMap(3,1) X;
        }
    }

    Layer Conv2D_80 {
        Type: Conv
        Stride { X: 1, Y: 1 }
        Dimensions { K: 4, C: 16, R: 1, S: 1, Y: 80, X: 80 }
        Dataflow {
            // Parsed from CONV_2D op 334.
            SpatialMap(4,4) K;
            TemporalMap(1,1) C;
            TemporalMap(1,1) Y;
            TemporalMap(1,1) X;
            TemporalMap(1,1) R;
            TemporalMap(1,1) S;
        }
    }

    Layer Conv2D_81 {
        Type: Conv
        Stride { X: 1, Y: 1 }
        Dimensions { K: 16, C: 128, R: 3, S: 3, Y: 40, X: 40 }
        Dataflow {
            // Parsed from CONV_2D op 337.
            SpatialMap(8,8) K;
            TemporalMap(1,1) C;
            TemporalMap(3,3) R;
            TemporalMap(3,3) S;
            TemporalMap(3,1) Y;
            TemporalMap(3,1) X;
        }
    }

    Layer Conv2D_82 {
        Type: Conv
        Stride { X: 1, Y: 1 }
        Dimensions { K: 16, C: 16, R: 3, S: 3, Y: 40, X: 40 }
        Dataflow {
            // Parsed from CONV_2D op 340.
            SpatialMap(8,8) K;
            TemporalMap(1,1) C;
            TemporalMap(3,3) R;
            TemporalMap(3,3) S;
            TemporalMap(3,1) Y;
            TemporalMap(3,1) X;
        }
    }

    Layer Conv2D_83 {
        Type: Conv
        Stride { X: 1, Y: 1 }
        Dimensions { K: 4, C: 16, R: 1, S: 1, Y: 40, X: 40 }
        Dataflow {
            // Parsed from CONV_2D op 343.
            SpatialMap(4,4) K;
            TemporalMap(1,1) C;
            TemporalMap(1,1) Y;
            TemporalMap(1,1) X;
            TemporalMap(1,1) R;
            TemporalMap(1,1) S;
        }
    }

    Layer Conv2D_84 {
        Type: Conv
        Stride { X: 1, Y: 1 }
        Dimensions { K: 16, C: 256, R: 3, S: 3, Y: 20, X: 20 }
        Dataflow {
            // Parsed from CONV_2D op 346.
            SpatialMap(8,8) K;
            TemporalMap(1,1) C;
            TemporalMap(3,3) R;
            TemporalMap(3,3) S;
            TemporalMap(3,1) Y;
            TemporalMap(3,1) X;
        }
    }

    Layer Conv2D_85 {
        Type: Conv
        Stride { X: 1, Y: 1 }
        Dimensions { K: 16, C: 16, R: 3, S: 3, Y: 20, X: 20 }
        Dataflow {
            // Parsed from CONV_2D op 349.
            SpatialMap(8,8) K;
            TemporalMap(1,1) C;
            TemporalMap(3,3) R;
            TemporalMap(3,3) S;
            TemporalMap(3,1) Y;
            TemporalMap(3,1) X;
        }
    }

    Layer Conv2D_86 {
        Type: Conv
        Stride { X: 1, Y: 1 }
        Dimensions { K: 4, C: 16, R: 1, S: 1, Y: 20, X: 20 }
        Dataflow {
            // Parsed from CONV_2D op 352.
            SpatialMap(4,4) K;
            TemporalMap(1,1) C;
            TemporalMap(1,1) Y;
            TemporalMap(1,1) X;
            TemporalMap(1,1) R;
            TemporalMap(1,1) S;
        }
    }

    Layer DepthwiseConv2D_87 {
        Type: DepthwiseConv
        Stride { X: 1, Y: 1 }
        Dimensions { K: 64, C: 1, R: 3, S: 3, Y: 80, X: 80 }
        Dataflow {
            // Parsed from DEPTHWISE_CONV_2D op 356.
            SpatialMap(8,8) K;
            TemporalMap(1,1) C;
            TemporalMap(3,3) R;
            TemporalMap(3,3) S;
            TemporalMap(3,1) Y;
            TemporalMap(3,1) X;
        }
    }

    Layer Conv2D_88 {
        Type: Conv
        Stride { X: 1, Y: 1 }
        Dimensions { K: 80, C: 64, R: 1, S: 1, Y: 80, X: 80 }
        Dataflow {
            // Parsed from CONV_2D op 359.
            SpatialMap(8,8) K;
            TemporalMap(1,1) C;
            TemporalMap(1,1) Y;
            TemporalMap(1,1) X;
            TemporalMap(1,1) R;
            TemporalMap(1,1) S;
        }
    }

    Layer DepthwiseConv2D_89 {
        Type: DepthwiseConv
        Stride { X: 1, Y: 1 }
        Dimensions { K: 80, C: 1, R: 3, S: 3, Y: 80, X: 80 }
        Dataflow {
            // Parsed from DEPTHWISE_CONV_2D op 362.
            SpatialMap(8,8) K;
            TemporalMap(1,1) C;
            TemporalMap(3,3) R;
            TemporalMap(3,3) S;
            TemporalMap(3,1) Y;
            TemporalMap(3,1) X;
        }
    }

    Layer Conv2D_90 {
        Type: Conv
        Stride { X: 1, Y: 1 }
        Dimensions { K: 80, C: 80, R: 1, S: 1, Y: 80, X: 80 }
        Dataflow {
            // Parsed from CONV_2D op 365.
            SpatialMap(8,8) K;
            TemporalMap(1,1) C;
            TemporalMap(1,1) Y;
            TemporalMap(1,1) X;
            TemporalMap(1,1) R;
            TemporalMap(1,1) S;
        }
    }

    Layer Conv2D_91 {
        Type: Conv
        Stride { X: 1, Y: 1 }
        Dimensions { K: 80, C: 80, R: 1, S: 1, Y: 80, X: 80 }
        Dataflow {
            // Parsed from CONV_2D op 368.
            SpatialMap(8,8) K;
            TemporalMap(1,1) C;
            TemporalMap(1,1) Y;
            TemporalMap(1,1) X;
            TemporalMap(1,1) R;
            TemporalMap(1,1) S;
        }
    }

    Layer DepthwiseConv2D_92 {
        Type: DepthwiseConv
        Stride { X: 1, Y: 1 }
        Dimensions { K: 128, C: 1, R: 3, S: 3, Y: 40, X: 40 }
        Dataflow {
            // Parsed from DEPTHWISE_CONV_2D op 371.
            SpatialMap(8,8) K;
            TemporalMap(1,1) C;
            TemporalMap(3,3) R;
            TemporalMap(3,3) S;
            TemporalMap(3,1) Y;
            TemporalMap(3,1) X;
        }
    }

    Layer Conv2D_93 {
        Type: Conv
        Stride { X: 1, Y: 1 }
        Dimensions { K: 80, C: 128, R: 1, S: 1, Y: 40, X: 40 }
        Dataflow {
            // Parsed from CONV_2D op 374.
            SpatialMap(8,8) K;
            TemporalMap(1,1) C;
            TemporalMap(1,1) Y;
            TemporalMap(1,1) X;
            TemporalMap(1,1) R;
            TemporalMap(1,1) S;
        }
    }

    Layer DepthwiseConv2D_94 {
        Type: DepthwiseConv
        Stride { X: 1, Y: 1 }
        Dimensions { K: 80, C: 1, R: 3, S: 3, Y: 40, X: 40 }
        Dataflow {
            // Parsed from DEPTHWISE_CONV_2D op 377.
            SpatialMap(8,8) K;
            TemporalMap(1,1) C;
            TemporalMap(3,3) R;
            TemporalMap(3,3) S;
            TemporalMap(3,1) Y;
            TemporalMap(3,1) X;
        }
    }

    Layer Conv2D_95 {
        Type: Conv
        Stride { X: 1, Y: 1 }
        Dimensions { K: 80, C: 80, R: 1, S: 1, Y: 40, X: 40 }
        Dataflow {
            // Parsed from CONV_2D op 380.
            SpatialMap(8,8) K;
            TemporalMap(1,1) C;
            TemporalMap(1,1) Y;
            TemporalMap(1,1) X;
            TemporalMap(1,1) R;
            TemporalMap(1,1) S;
        }
    }

    Layer Conv2D_96 {
        Type: Conv
        Stride { X: 1, Y: 1 }
        Dimensions { K: 80, C: 80, R: 1, S: 1, Y: 40, X: 40 }
        Dataflow {
            // Parsed from CONV_2D op 383.
            SpatialMap(8,8) K;
            TemporalMap(1,1) C;
            TemporalMap(1,1) Y;
            TemporalMap(1,1) X;
            TemporalMap(1,1) R;
            TemporalMap(1,1) S;
        }
    }

    Layer DepthwiseConv2D_97 {
        Type: DepthwiseConv
        Stride { X: 1, Y: 1 }
        Dimensions { K: 256, C: 1, R: 3, S: 3, Y: 20, X: 20 }
        Dataflow {
            // Parsed from DEPTHWISE_CONV_2D op 386.
            SpatialMap(8,8) K;
            TemporalMap(1,1) C;
            TemporalMap(3,3) R;
            TemporalMap(3,3) S;
            TemporalMap(3,1) Y;
            TemporalMap(3,1) X;
        }
    }

    Layer Conv2D_98 {
        Type: Conv
        Stride { X: 1, Y: 1 }
        Dimensions { K: 80, C: 256, R: 1, S: 1, Y: 20, X: 20 }
        Dataflow {
            // Parsed from CONV_2D op 389.
            SpatialMap(8,8) K;
            TemporalMap(1,1) C;
            TemporalMap(1,1) Y;
            TemporalMap(1,1) X;
            TemporalMap(1,1) R;
            TemporalMap(1,1) S;
        }
    }

    Layer DepthwiseConv2D_99 {
        Type: DepthwiseConv
        Stride { X: 1, Y: 1 }
        Dimensions { K: 80, C: 1, R: 3, S: 3, Y: 20, X: 20 }
        Dataflow {
            // Parsed from DEPTHWISE_CONV_2D op 392.
            SpatialMap(8,8) K;
            TemporalMap(1,1) C;
            TemporalMap(3,3) R;
            TemporalMap(3,3) S;
            TemporalMap(3,1) Y;
            TemporalMap(3,1) X;
        }
    }

    Layer Conv2D_100 {
        Type: Conv
        Stride { X: 1, Y: 1 }
        Dimensions { K: 80, C: 80, R: 1, S: 1, Y: 20, X: 20 }
        Dataflow {
            // Parsed from CONV_2D op 395.
            SpatialMap(8,8) K;
            TemporalMap(1,1) C;
            TemporalMap(1,1) Y;
            TemporalMap(1,1) X;
            TemporalMap(1,1) R;
            TemporalMap(1,1) S;
        }
    }

    Layer Conv2D_101 {
        Type: Conv
        Stride { X: 1, Y: 1 }
        Dimensions { K: 80, C: 80, R: 1, S: 1, Y: 20, X: 20 }
        Dataflow {
            // Parsed from CONV_2D op 398.
            SpatialMap(8,8) K;
            TemporalMap(1,1) C;
            TemporalMap(1,1) Y;
            TemporalMap(1,1) X;
            TemporalMap(1,1) R;
            TemporalMap(1,1) S;
        }
    }

}