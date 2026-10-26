Network Yolo26nano_Layer0 {
    Layer Conv2D {
        Type: Conv
        Stride { X: 2, Y: 2 }
        Dimensions { K: 16, C: 3, R: 3, S: 3, Y: 640, X: 640 }
        Dataflow {
            // K is the outer spatial loop. One PE computes up to 8 output neurons.
            SpatialMap(8,8) K;

            // C is next. For layer 0, C*R*S = 3*3*3 = 27, so this is one fold.
            TemporalMap(3,3) C;

            // R,S > 1, so use RSYX loop ordering.
            TemporalMap(3,3) R;
            TemporalMap(3,3) S;
            TemporalMap(3,1) Y;
            TemporalMap(3,1) X;
        }
    }
}
