Network Yolo26nano_Layer2_Input_AutoMapped {
    Layer Conv2D_2 {
        Type: Conv
        Stride { X: 1, Y: 1 }
        Dimensions { K: 32, C: 32, R: 1, S: 1, Y: 160, X: 160 }
        Dataflow {
            // Optimized cost: 614400 cycles; loop order K->C->YXRS.
            // K groups=4, C folds=4, C waves=1, row fold=8.
            SpatialMap(8,8) K;
            TemporalMap(8,8) C;
            TemporalMap(1,1) Y;
            TemporalMap(1,1) X;
            TemporalMap(1,1) R;
            TemporalMap(1,1) S;

            // Cluster C folds across 4 PEs to reduce latency.
            Cluster(4,P);
            SpatialMap(8,8) C;
            TemporalMap(1,1) Y;
            TemporalMap(1,1) X;
            TemporalMap(1,1) R;
            TemporalMap(1,1) S;
        }
    }

}