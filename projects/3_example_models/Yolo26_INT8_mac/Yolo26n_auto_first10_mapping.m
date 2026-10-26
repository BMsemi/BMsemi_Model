Network Yolo26nano_First10_AutoMapped {
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

    Layer Conv2D_3 {
        Type: Conv
        Stride { X: 1, Y: 1 }
        Dimensions { K: 8, C: 16, R: 3, S: 3, Y: 160, X: 160 }
        Dataflow {
            // Optimized cost: 7756800 cycles; loop order K->C->RSYX.
            // K groups=1, C folds=4, C waves=4, row fold=36.
            SpatialMap(8,8) K;
            TemporalMap(4,4) C;
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
            // Optimized cost: 691200 cycles; loop order K->C->RSYX.
            // K groups=2, C folds=8, C waves=1, row fold=9.
            SpatialMap(8,8) K;
            TemporalMap(1,1) C;
            TemporalMap(3,3) R;
            TemporalMap(3,3) S;
            TemporalMap(3,1) Y;
            TemporalMap(3,1) X;

            // Cluster C folds across 8 PEs to reduce latency.
            Cluster(8,P);
            SpatialMap(1,1) C;
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
            // Optimized cost: 1408000 cycles; loop order K->C->YXRS.
            // K groups=8, C folds=2, C waves=1, row fold=24.
            SpatialMap(8,8) K;
            TemporalMap(24,24) C;
            TemporalMap(1,1) Y;
            TemporalMap(1,1) X;
            TemporalMap(1,1) R;
            TemporalMap(1,1) S;

            // Cluster C folds across 2 PEs to reduce latency.
            Cluster(2,P);
            SpatialMap(24,24) C;
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
            // Optimized cost: 3910400 cycles; loop order K->C->RSYX.
            // K groups=8, C folds=16, C waves=8, row fold=36.
            SpatialMap(8,8) K;
            TemporalMap(4,4) C;
            TemporalMap(3,3) R;
            TemporalMap(3,3) S;
            TemporalMap(3,1) Y;
            TemporalMap(3,1) X;

            // Cluster C folds across 2 PEs to reduce latency.
            Cluster(2,P);
            SpatialMap(4,4) C;
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
            // Optimized cost: 454400 cycles; loop order K->C->YXRS.
            // K groups=8, C folds=2, C waves=1, row fold=32.
            SpatialMap(8,8) K;
            TemporalMap(32,32) C;
            TemporalMap(1,1) Y;
            TemporalMap(1,1) X;
            TemporalMap(1,1) R;
            TemporalMap(1,1) S;

            // Cluster C folds across 2 PEs to reduce latency.
            Cluster(2,P);
            SpatialMap(32,32) C;
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
            // Optimized cost: 518400 cycles; loop order K->C->RSYX.
            // K groups=2, C folds=8, C waves=1, row fold=36.
            SpatialMap(8,8) K;
            TemporalMap(4,4) C;
            TemporalMap(3,3) R;
            TemporalMap(3,3) S;
            TemporalMap(3,1) Y;
            TemporalMap(3,1) X;

            // Cluster C folds across 8 PEs to reduce latency.
            Cluster(8,P);
            SpatialMap(4,4) C;
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
            // Optimized cost: 512000 cycles; loop order K->C->RSYX.
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