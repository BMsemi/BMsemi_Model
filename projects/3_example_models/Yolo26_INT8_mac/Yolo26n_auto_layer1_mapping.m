Network Yolo26nano_Layer1_Input_AutoMapped {
    Layer Conv2D_1 {
        Type: Conv
        Stride { X: 2, Y: 2 }
        Dimensions { K: 32, C: 16, R: 3, S: 3, Y: 320, X: 320 }
        Dataflow {
            // Optimized cost: 2048000 cycles; loop order K->C->RSYX.
            // K groups=4, C folds=4, C waves=1, row fold=36.
            SpatialMap(8,8) K;
            TemporalMap(4,4) C;
            TemporalMap(3,3) R;
            TemporalMap(3,3) S;
            TemporalMap(3,1) Y;
            TemporalMap(3,1) X;

            // Cluster C folds across 4 PEs to reduce latency.
            Cluster(4,P);
            SpatialMap(4,4) C;
            TemporalMap(3,3) R;
            TemporalMap(3,3) S;
            TemporalMap(3,1) Y;
            TemporalMap(3,1) X;
        }
    }

}