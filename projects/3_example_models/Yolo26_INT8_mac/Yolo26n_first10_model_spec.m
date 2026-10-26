Network Yolo26nano_First10 {
    Layer Conv2D {
        Type: Conv
        Stride { X: 2, Y: 2 }
        Dimensions { K: 16, C: 3, R: 3, S: 3, Y: 640, X: 640 }
        Dataflow {
            // Parsed from CONV_2D op 1.
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
            // Parsed from CONV_2D op 5.
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
            // Parsed from CONV_2D op 8.
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
            // Parsed from CONV_2D op 12.
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
            // Parsed from CONV_2D op 15.
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
            // Parsed from CONV_2D op 21.
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
            // Parsed from CONV_2D op 25.
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
            // Parsed from CONV_2D op 28.
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

}