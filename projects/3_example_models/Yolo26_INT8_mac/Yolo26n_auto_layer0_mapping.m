Network Yolo26nano_AutoMapped {
    Layer Conv2D {
        Type: Conv
        Stride { X: 2, Y: 2 }
        Dimensions { K: 16, C: 3, R: 3, S: 3, Y: 640, X: 640 }
        Dataflow {
            // Optimized cost: 6144000 cycles; loop order K->C->RSYX.
            // K groups=2, C folds=1, C waves=1, row fold=27.
            SpatialMap(8,8) K;
            TemporalMap(3,3) C;
            TemporalMap(3,3) R;
            TemporalMap(3,3) S;
            TemporalMap(3,1) Y;
            TemporalMap(3,1) X;
        }
    }

}