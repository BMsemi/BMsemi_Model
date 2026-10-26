Network Yolo26nano_Layer1_Input {
    Layer Conv2D_1 {
        Type: Conv
        Stride { X: 2, Y: 2 }
        Dimensions { K: 32, C: 16, R: 3, S: 3, Y: 320, X: 320 }
        Dataflow {
            SpatialMap(8,8) K;
            TemporalMap(1,1) C;
            TemporalMap(3,3) R;
            TemporalMap(3,3) S;
            TemporalMap(3,1) Y;
            TemporalMap(3,1) X;
        }
    }
}
