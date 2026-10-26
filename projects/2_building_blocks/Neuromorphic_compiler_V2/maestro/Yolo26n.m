Network Yolo26nano {
    Layer Conv2D {
        Type: Conv
        Stride { X: 2, Y: 2}
        Dimensions { K: 16, C: 3, R: 3, S: 3, Y: 640, X: 640}
        Dataflow {
            SpatialMap(8,8) K;
            TemporalMap(1,1) C;
            TemporalMap(Sz(R),Sz(R)) R;
            TemporalMap(Sz(S),Sz(S)) S;
            TemporalMap(Sz(R),1) Y;
            TemporalMap(Sz(S),1) X;
        }
    }
}
