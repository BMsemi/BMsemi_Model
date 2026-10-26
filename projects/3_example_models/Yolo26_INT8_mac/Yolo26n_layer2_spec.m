Network Yolo26nano_Layer2_Input {
    Layer Conv2D_2 {
        Type: Conv
        Stride { X: 1, Y: 1 }
        Dimensions { K: 32, C: 32, R: 1, S: 1, Y: 160, X: 160 }
        Dataflow {
            SpatialMap(8,8) K;
            TemporalMap(1,1) C;
            TemporalMap(1,1) Y;
            TemporalMap(1,1) X;
            TemporalMap(1,1) R;
            TemporalMap(1,1) S;
        }
    }
}
