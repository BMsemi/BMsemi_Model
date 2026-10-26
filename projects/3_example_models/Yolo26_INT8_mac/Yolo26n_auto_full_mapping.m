Network Yolo26nano_Full_AutoMapped {
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

    Layer Conv2D_10 {
        Type: Conv
        Stride { X: 1, Y: 1 }
        Dimensions { K: 128, C: 96, R: 1, S: 1, Y: 80, X: 80 }
        Dataflow {
            // Optimized cost: 1286400 cycles; loop order K->C->YXRS.
            // K groups=16, C folds=2, C waves=2, row fold=48.
            SpatialMap(8,8) K;
            TemporalMap(48,48) C;
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
            // Optimized cost: 3844800 cycles; loop order K->C->RSYX.
            // K groups=16, C folds=32, C waves=32, row fold=36.
            SpatialMap(8,8) K;
            TemporalMap(4,4) C;
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
            // Optimized cost: 424000 cycles; loop order K->C->YXRS.
            // K groups=16, C folds=2, C waves=2, row fold=64.
            SpatialMap(8,8) K;
            TemporalMap(64,64) C;
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
            // Optimized cost: 64000 cycles; loop order K->C->YXRS.
            // K groups=4, C folds=4, C waves=1, row fold=16.
            SpatialMap(8,8) K;
            TemporalMap(16,16) C;
            TemporalMap(1,1) Y;
            TemporalMap(1,1) X;
            TemporalMap(1,1) R;
            TemporalMap(1,1) S;

            // Cluster C folds across 4 PEs to reduce latency.
            Cluster(4,P);
            SpatialMap(16,16) C;
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
            // Optimized cost: 251200 cycles; loop order K->C->RSYX.
            // K groups=4, C folds=8, C waves=2, row fold=36.
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

    Layer Conv2D_15 {
        Type: Conv
        Stride { X: 1, Y: 1 }
        Dimensions { K: 32, C: 32, R: 3, S: 3, Y: 40, X: 40 }
        Dataflow {
            // Optimized cost: 251200 cycles; loop order K->C->RSYX.
            // K groups=4, C folds=8, C waves=2, row fold=36.
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

    Layer Conv2D_16 {
        Type: Conv
        Stride { X: 1, Y: 1 }
        Dimensions { K: 32, C: 32, R: 3, S: 3, Y: 40, X: 40 }
        Dataflow {
            // Optimized cost: 251200 cycles; loop order K->C->RSYX.
            // K groups=4, C folds=8, C waves=2, row fold=36.
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

    Layer Conv2D_17 {
        Type: Conv
        Stride { X: 1, Y: 1 }
        Dimensions { K: 32, C: 32, R: 3, S: 3, Y: 40, X: 40 }
        Dataflow {
            // Optimized cost: 251200 cycles; loop order K->C->RSYX.
            // K groups=4, C folds=8, C waves=2, row fold=36.
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

    Layer Conv2D_18 {
        Type: Conv
        Stride { X: 1, Y: 1 }
        Dimensions { K: 32, C: 64, R: 1, S: 1, Y: 40, X: 40 }
        Dataflow {
            // Optimized cost: 64000 cycles; loop order K->C->YXRS.
            // K groups=4, C folds=4, C waves=1, row fold=16.
            SpatialMap(8,8) K;
            TemporalMap(16,16) C;
            TemporalMap(1,1) Y;
            TemporalMap(1,1) X;
            TemporalMap(1,1) R;
            TemporalMap(1,1) S;

            // Cluster C folds across 4 PEs to reduce latency.
            Cluster(4,P);
            SpatialMap(16,16) C;
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
            // Optimized cost: 113600 cycles; loop order K->C->YXRS.
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

    Layer Conv2D_20 {
        Type: Conv
        Stride { X: 1, Y: 1 }
        Dimensions { K: 128, C: 192, R: 1, S: 1, Y: 40, X: 40 }
        Dataflow {
            // Optimized cost: 633600 cycles; loop order K->C->YXRS.
            // K groups=16, C folds=3, C waves=3, row fold=64.
            SpatialMap(8,8) K;
            TemporalMap(64,64) C;
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
            // Optimized cost: 961200 cycles; loop order K->C->RSYX.
            // K groups=32, C folds=32, C waves=32, row fold=36.
            SpatialMap(8,8) K;
            TemporalMap(4,4) C;
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
            // Optimized cost: 210800 cycles; loop order K->C->YXRS.
            // K groups=32, C folds=4, C waves=4, row fold=64.
            SpatialMap(8,8) K;
            TemporalMap(64,64) C;
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
            // Optimized cost: 54000 cycles; loop order K->C->YXRS.
            // K groups=8, C folds=2, C waves=1, row fold=64.
            SpatialMap(8,8) K;
            TemporalMap(64,64) C;
            TemporalMap(1,1) Y;
            TemporalMap(1,1) X;
            TemporalMap(1,1) R;
            TemporalMap(1,1) S;

            // Cluster C folds across 2 PEs to reduce latency.
            Cluster(2,P);
            SpatialMap(64,64) C;
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
            // Optimized cost: 244400 cycles; loop order K->C->RSYX.
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

    Layer Conv2D_25 {
        Type: Conv
        Stride { X: 1, Y: 1 }
        Dimensions { K: 64, C: 64, R: 3, S: 3, Y: 20, X: 20 }
        Dataflow {
            // Optimized cost: 244400 cycles; loop order K->C->RSYX.
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

    Layer Conv2D_26 {
        Type: Conv
        Stride { X: 1, Y: 1 }
        Dimensions { K: 64, C: 64, R: 3, S: 3, Y: 20, X: 20 }
        Dataflow {
            // Optimized cost: 244400 cycles; loop order K->C->RSYX.
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

    Layer Conv2D_27 {
        Type: Conv
        Stride { X: 1, Y: 1 }
        Dimensions { K: 64, C: 64, R: 3, S: 3, Y: 20, X: 20 }
        Dataflow {
            // Optimized cost: 244400 cycles; loop order K->C->RSYX.
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

    Layer Conv2D_28 {
        Type: Conv
        Stride { X: 1, Y: 1 }
        Dimensions { K: 64, C: 128, R: 1, S: 1, Y: 20, X: 20 }
        Dataflow {
            // Optimized cost: 54000 cycles; loop order K->C->YXRS.
            // K groups=8, C folds=2, C waves=1, row fold=64.
            SpatialMap(8,8) K;
            TemporalMap(64,64) C;
            TemporalMap(1,1) Y;
            TemporalMap(1,1) X;
            TemporalMap(1,1) R;
            TemporalMap(1,1) S;

            // Cluster C folds across 2 PEs to reduce latency.
            Cluster(2,P);
            SpatialMap(64,64) C;
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
            // Optimized cost: 106000 cycles; loop order K->C->YXRS.
            // K groups=16, C folds=2, C waves=2, row fold=64.
            SpatialMap(8,8) K;
            TemporalMap(64,64) C;
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
            // Optimized cost: 315600 cycles; loop order K->C->YXRS.
            // K groups=32, C folds=6, C waves=6, row fold=64.
            SpatialMap(8,8) K;
            TemporalMap(64,64) C;
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
            // Optimized cost: 210800 cycles; loop order K->C->YXRS.
            // K groups=16, C folds=4, C waves=4, row fold=64.
            SpatialMap(8,8) K;
            TemporalMap(64,64) C;
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
            // Optimized cost: 420400 cycles; loop order K->C->YXRS.
            // K groups=32, C folds=8, C waves=8, row fold=64.
            SpatialMap(8,8) K;
            TemporalMap(64,64) C;
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
            // Optimized cost: 210800 cycles; loop order K->C->YXRS.
            // K groups=32, C folds=4, C waves=4, row fold=64.
            SpatialMap(8,8) K;
            TemporalMap(64,64) C;
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
            // Optimized cost: 106000 cycles; loop order K->C->YXRS.
            // K groups=32, C folds=2, C waves=2, row fold=64.
            SpatialMap(8,8) K;
            TemporalMap(64,64) C;
            TemporalMap(1,1) Y;
            TemporalMap(1,1) X;
            TemporalMap(1,1) R;
            TemporalMap(1,1) S;
        }
    }

    Layer Conv2D_35 {
        Type: Conv
        Stride { X: 1, Y: 1 }
        Dimensions { K: 128, C: 1, R: 3, S: 3, Y: 20, X: 20 }
        Dataflow {
            // Optimized cost: 9600 cycles; loop order K->C->RSYX.
            // K groups=16, C folds=1, C waves=1, row fold=9.
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
            // Optimized cost: 106000 cycles; loop order K->C->YXRS.
            // K groups=16, C folds=2, C waves=2, row fold=64.
            SpatialMap(8,8) K;
            TemporalMap(64,64) C;
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
            // Optimized cost: 106000 cycles; loop order K->C->YXRS.
            // K groups=32, C folds=2, C waves=2, row fold=64.
            SpatialMap(8,8) K;
            TemporalMap(64,64) C;
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
            // Optimized cost: 210800 cycles; loop order K->C->YXRS.
            // K groups=16, C folds=4, C waves=4, row fold=64.
            SpatialMap(8,8) K;
            TemporalMap(64,64) C;
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
            // Optimized cost: 210800 cycles; loop order K->C->YXRS.
            // K groups=32, C folds=4, C waves=4, row fold=64.
            SpatialMap(8,8) K;
            TemporalMap(64,64) C;
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
            // Optimized cost: 1262400 cycles; loop order K->C->YXRS.
            // K groups=16, C folds=6, C waves=6, row fold=64.
            SpatialMap(8,8) K;
            TemporalMap(64,64) C;
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
            // Optimized cost: 64000 cycles; loop order K->C->YXRS.
            // K groups=4, C folds=4, C waves=1, row fold=16.
            SpatialMap(8,8) K;
            TemporalMap(16,16) C;
            TemporalMap(1,1) Y;
            TemporalMap(1,1) X;
            TemporalMap(1,1) R;
            TemporalMap(1,1) S;

            // Cluster C folds across 4 PEs to reduce latency.
            Cluster(4,P);
            SpatialMap(16,16) C;
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
            // Optimized cost: 251200 cycles; loop order K->C->RSYX.
            // K groups=4, C folds=8, C waves=2, row fold=36.
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

    Layer Conv2D_43 {
        Type: Conv
        Stride { X: 1, Y: 1 }
        Dimensions { K: 32, C: 32, R: 3, S: 3, Y: 40, X: 40 }
        Dataflow {
            // Optimized cost: 251200 cycles; loop order K->C->RSYX.
            // K groups=4, C folds=8, C waves=2, row fold=36.
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

    Layer Conv2D_44 {
        Type: Conv
        Stride { X: 1, Y: 1 }
        Dimensions { K: 32, C: 32, R: 3, S: 3, Y: 40, X: 40 }
        Dataflow {
            // Optimized cost: 251200 cycles; loop order K->C->RSYX.
            // K groups=4, C folds=8, C waves=2, row fold=36.
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

    Layer Conv2D_45 {
        Type: Conv
        Stride { X: 1, Y: 1 }
        Dimensions { K: 32, C: 32, R: 3, S: 3, Y: 40, X: 40 }
        Dataflow {
            // Optimized cost: 251200 cycles; loop order K->C->RSYX.
            // K groups=4, C folds=8, C waves=2, row fold=36.
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

    Layer Conv2D_46 {
        Type: Conv
        Stride { X: 1, Y: 1 }
        Dimensions { K: 32, C: 64, R: 1, S: 1, Y: 40, X: 40 }
        Dataflow {
            // Optimized cost: 64000 cycles; loop order K->C->YXRS.
            // K groups=4, C folds=4, C waves=1, row fold=16.
            SpatialMap(8,8) K;
            TemporalMap(16,16) C;
            TemporalMap(1,1) Y;
            TemporalMap(1,1) X;
            TemporalMap(1,1) R;
            TemporalMap(1,1) S;

            // Cluster C folds across 4 PEs to reduce latency.
            Cluster(4,P);
            SpatialMap(16,16) C;
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
            // Optimized cost: 113600 cycles; loop order K->C->YXRS.
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

    Layer Conv2D_48 {
        Type: Conv
        Stride { X: 1, Y: 1 }
        Dimensions { K: 128, C: 192, R: 1, S: 1, Y: 40, X: 40 }
        Dataflow {
            // Optimized cost: 633600 cycles; loop order K->C->YXRS.
            // K groups=16, C folds=3, C waves=3, row fold=64.
            SpatialMap(8,8) K;
            TemporalMap(64,64) C;
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
            // Optimized cost: 1708800 cycles; loop order K->C->YXRS.
            // K groups=8, C folds=4, C waves=2, row fold=64.
            SpatialMap(8,8) K;
            TemporalMap(64,64) C;
            TemporalMap(1,1) Y;
            TemporalMap(1,1) X;
            TemporalMap(1,1) R;
            TemporalMap(1,1) S;

            // Cluster C folds across 2 PEs to reduce latency.
            Cluster(2,P);
            SpatialMap(64,64) C;
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
            // Optimized cost: 108800 cycles; loop order K->C->YXRS.
            // K groups=2, C folds=8, C waves=1, row fold=4.
            SpatialMap(8,8) K;
            TemporalMap(4,4) C;
            TemporalMap(1,1) Y;
            TemporalMap(1,1) X;
            TemporalMap(1,1) R;
            TemporalMap(1,1) S;

            // Cluster C folds across 8 PEs to reduce latency.
            Cluster(8,P);
            SpatialMap(4,4) C;
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
            // Optimized cost: 288000 cycles; loop order K->C->RSYX.
            // K groups=2, C folds=8, C waves=1, row fold=18.
            SpatialMap(8,8) K;
            TemporalMap(2,2) C;
            TemporalMap(3,3) R;
            TemporalMap(3,3) S;
            TemporalMap(3,1) Y;
            TemporalMap(3,1) X;

            // Cluster C folds across 8 PEs to reduce latency.
            Cluster(8,P);
            SpatialMap(2,2) C;
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
            // Optimized cost: 288000 cycles; loop order K->C->RSYX.
            // K groups=2, C folds=8, C waves=1, row fold=18.
            SpatialMap(8,8) K;
            TemporalMap(2,2) C;
            TemporalMap(3,3) R;
            TemporalMap(3,3) S;
            TemporalMap(3,1) Y;
            TemporalMap(3,1) X;

            // Cluster C folds across 8 PEs to reduce latency.
            Cluster(8,P);
            SpatialMap(2,2) C;
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
            // Optimized cost: 288000 cycles; loop order K->C->RSYX.
            // K groups=2, C folds=8, C waves=1, row fold=18.
            SpatialMap(8,8) K;
            TemporalMap(2,2) C;
            TemporalMap(3,3) R;
            TemporalMap(3,3) S;
            TemporalMap(3,1) Y;
            TemporalMap(3,1) X;

            // Cluster C folds across 8 PEs to reduce latency.
            Cluster(8,P);
            SpatialMap(2,2) C;
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
            // Optimized cost: 288000 cycles; loop order K->C->RSYX.
            // K groups=2, C folds=8, C waves=1, row fold=18.
            SpatialMap(8,8) K;
            TemporalMap(2,2) C;
            TemporalMap(3,3) R;
            TemporalMap(3,3) S;
            TemporalMap(3,1) Y;
            TemporalMap(3,1) X;

            // Cluster C folds across 8 PEs to reduce latency.
            Cluster(8,P);
            SpatialMap(2,2) C;
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
            // Optimized cost: 108800 cycles; loop order K->C->YXRS.
            // K groups=2, C folds=8, C waves=1, row fold=4.
            SpatialMap(8,8) K;
            TemporalMap(4,4) C;
            TemporalMap(1,1) Y;
            TemporalMap(1,1) X;
            TemporalMap(1,1) R;
            TemporalMap(1,1) S;

            // Cluster C folds across 8 PEs to reduce latency.
            Cluster(8,P);
            SpatialMap(4,4) C;
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
            // Optimized cost: 153600 cycles; loop order K->C->YXRS.
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

    Layer Conv2D_57 {
        Type: Conv
        Stride { X: 1, Y: 1 }
        Dimensions { K: 64, C: 96, R: 1, S: 1, Y: 80, X: 80 }
        Dataflow {
            // Optimized cost: 659200 cycles; loop order K->C->YXRS.
            // K groups=8, C folds=2, C waves=1, row fold=48.
            SpatialMap(8,8) K;
            TemporalMap(48,48) C;
            TemporalMap(1,1) Y;
            TemporalMap(1,1) X;
            TemporalMap(1,1) R;
            TemporalMap(1,1) S;

            // Cluster C folds across 2 PEs to reduce latency.
            Cluster(2,P);
            SpatialMap(48,48) C;
            TemporalMap(1,1) Y;
            TemporalMap(1,1) X;
            TemporalMap(1,1) R;
            TemporalMap(1,1) S;
        }
    }

    Layer DepthwiseConv2D_58 {
        Type: DepthwiseConv
        Stride { X: 1, Y: 1 }
        Dimensions { K: 64, C: 1, R: 3, S: 3, Y: 80, X: 80 }
        Dataflow {
            // Optimized cost: 153600 cycles; loop order K->C->RSYX.
            // K groups=8, C folds=1, C waves=1, row fold=9.
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
        Dimensions { K: 80, C: 64, R: 1, S: 1, Y: 80, X: 80 }
        Dataflow {
            // Optimized cost: 857600 cycles; loop order K->C->YXRS.
            // K groups=10, C folds=1, C waves=1, row fold=64.
            SpatialMap(8,8) K;
            TemporalMap(64,64) C;
            TemporalMap(1,1) Y;
            TemporalMap(1,1) X;
            TemporalMap(1,1) R;
            TemporalMap(1,1) S;
        }
    }

    Layer DepthwiseConv2D_60 {
        Type: DepthwiseConv
        Stride { X: 1, Y: 1 }
        Dimensions { K: 80, C: 1, R: 3, S: 3, Y: 80, X: 80 }
        Dataflow {
            // Optimized cost: 153600 cycles; loop order K->C->RSYX.
            // K groups=10, C folds=1, C waves=1, row fold=9.
            SpatialMap(8,8) K;
            TemporalMap(1,1) C;
            TemporalMap(3,3) R;
            TemporalMap(3,3) S;
            TemporalMap(3,1) Y;
            TemporalMap(3,1) X;
        }
    }

    Layer Conv2D_61 {
        Type: Conv
        Stride { X: 1, Y: 1 }
        Dimensions { K: 80, C: 80, R: 1, S: 1, Y: 80, X: 80 }
        Dataflow {
            // Optimized cost: 1081600 cycles; loop order K->C->YXRS.
            // K groups=10, C folds=2, C waves=2, row fold=40.
            SpatialMap(8,8) K;
            TemporalMap(40,40) C;
            TemporalMap(1,1) Y;
            TemporalMap(1,1) X;
            TemporalMap(1,1) R;
            TemporalMap(1,1) S;
        }
    }

    Layer Conv2D_62 {
        Type: Conv
        Stride { X: 1, Y: 1 }
        Dimensions { K: 80, C: 80, R: 1, S: 1, Y: 80, X: 80 }
        Dataflow {
            // Optimized cost: 1081600 cycles; loop order K->C->YXRS.
            // K groups=10, C folds=2, C waves=2, row fold=40.
            SpatialMap(8,8) K;
            TemporalMap(40,40) C;
            TemporalMap(1,1) Y;
            TemporalMap(1,1) X;
            TemporalMap(1,1) R;
            TemporalMap(1,1) S;
        }
    }

    Layer Conv2D_63 {
        Type: Conv
        Stride { X: 2, Y: 2 }
        Dimensions { K: 64, C: 64, R: 3, S: 3, Y: 80, X: 80 }
        Dataflow {
            // Optimized cost: 977600 cycles; loop order K->C->RSYX.
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

    Layer Conv2D_64 {
        Type: Conv
        Stride { X: 1, Y: 1 }
        Dimensions { K: 128, C: 192, R: 1, S: 1, Y: 40, X: 40 }
        Dataflow {
            // Optimized cost: 633600 cycles; loop order K->C->YXRS.
            // K groups=16, C folds=3, C waves=3, row fold=64.
            SpatialMap(8,8) K;
            TemporalMap(64,64) C;
            TemporalMap(1,1) Y;
            TemporalMap(1,1) X;
            TemporalMap(1,1) R;
            TemporalMap(1,1) S;
        }
    }

    Layer Conv2D_65 {
        Type: Conv
        Stride { X: 1, Y: 1 }
        Dimensions { K: 32, C: 64, R: 1, S: 1, Y: 40, X: 40 }
        Dataflow {
            // Optimized cost: 64000 cycles; loop order K->C->YXRS.
            // K groups=4, C folds=4, C waves=1, row fold=16.
            SpatialMap(8,8) K;
            TemporalMap(16,16) C;
            TemporalMap(1,1) Y;
            TemporalMap(1,1) X;
            TemporalMap(1,1) R;
            TemporalMap(1,1) S;

            // Cluster C folds across 4 PEs to reduce latency.
            Cluster(4,P);
            SpatialMap(16,16) C;
            TemporalMap(1,1) Y;
            TemporalMap(1,1) X;
            TemporalMap(1,1) R;
            TemporalMap(1,1) S;
        }
    }

    Layer Conv2D_66 {
        Type: Conv
        Stride { X: 1, Y: 1 }
        Dimensions { K: 32, C: 32, R: 3, S: 3, Y: 40, X: 40 }
        Dataflow {
            // Optimized cost: 251200 cycles; loop order K->C->RSYX.
            // K groups=4, C folds=8, C waves=2, row fold=36.
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

    Layer Conv2D_67 {
        Type: Conv
        Stride { X: 1, Y: 1 }
        Dimensions { K: 32, C: 32, R: 3, S: 3, Y: 40, X: 40 }
        Dataflow {
            // Optimized cost: 251200 cycles; loop order K->C->RSYX.
            // K groups=4, C folds=8, C waves=2, row fold=36.
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

    Layer Conv2D_68 {
        Type: Conv
        Stride { X: 1, Y: 1 }
        Dimensions { K: 32, C: 32, R: 3, S: 3, Y: 40, X: 40 }
        Dataflow {
            // Optimized cost: 251200 cycles; loop order K->C->RSYX.
            // K groups=4, C folds=8, C waves=2, row fold=36.
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

    Layer Conv2D_69 {
        Type: Conv
        Stride { X: 1, Y: 1 }
        Dimensions { K: 32, C: 32, R: 3, S: 3, Y: 40, X: 40 }
        Dataflow {
            // Optimized cost: 251200 cycles; loop order K->C->RSYX.
            // K groups=4, C folds=8, C waves=2, row fold=36.
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

    Layer Conv2D_70 {
        Type: Conv
        Stride { X: 1, Y: 1 }
        Dimensions { K: 32, C: 64, R: 1, S: 1, Y: 40, X: 40 }
        Dataflow {
            // Optimized cost: 64000 cycles; loop order K->C->YXRS.
            // K groups=4, C folds=4, C waves=1, row fold=16.
            SpatialMap(8,8) K;
            TemporalMap(16,16) C;
            TemporalMap(1,1) Y;
            TemporalMap(1,1) X;
            TemporalMap(1,1) R;
            TemporalMap(1,1) S;

            // Cluster C folds across 4 PEs to reduce latency.
            Cluster(4,P);
            SpatialMap(16,16) C;
            TemporalMap(1,1) Y;
            TemporalMap(1,1) X;
            TemporalMap(1,1) R;
            TemporalMap(1,1) S;
        }
    }

    Layer Conv2D_71 {
        Type: Conv
        Stride { X: 1, Y: 1 }
        Dimensions { K: 64, C: 64, R: 1, S: 1, Y: 40, X: 40 }
        Dataflow {
            // Optimized cost: 113600 cycles; loop order K->C->YXRS.
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

    Layer Conv2D_72 {
        Type: Conv
        Stride { X: 1, Y: 1 }
        Dimensions { K: 128, C: 192, R: 1, S: 1, Y: 40, X: 40 }
        Dataflow {
            // Optimized cost: 633600 cycles; loop order K->C->YXRS.
            // K groups=16, C folds=3, C waves=3, row fold=64.
            SpatialMap(8,8) K;
            TemporalMap(64,64) C;
            TemporalMap(1,1) Y;
            TemporalMap(1,1) X;
            TemporalMap(1,1) R;
            TemporalMap(1,1) S;
        }
    }

    Layer DepthwiseConv2D_73 {
        Type: DepthwiseConv
        Stride { X: 1, Y: 1 }
        Dimensions { K: 128, C: 1, R: 3, S: 3, Y: 40, X: 40 }
        Dataflow {
            // Optimized cost: 38400 cycles; loop order K->C->RSYX.
            // K groups=16, C folds=1, C waves=1, row fold=9.
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
        Dimensions { K: 80, C: 128, R: 1, S: 1, Y: 40, X: 40 }
        Dataflow {
            // Optimized cost: 424000 cycles; loop order K->C->YXRS.
            // K groups=10, C folds=2, C waves=2, row fold=64.
            SpatialMap(8,8) K;
            TemporalMap(64,64) C;
            TemporalMap(1,1) Y;
            TemporalMap(1,1) X;
            TemporalMap(1,1) R;
            TemporalMap(1,1) S;
        }
    }

    Layer DepthwiseConv2D_75 {
        Type: DepthwiseConv
        Stride { X: 1, Y: 1 }
        Dimensions { K: 80, C: 1, R: 3, S: 3, Y: 40, X: 40 }
        Dataflow {
            // Optimized cost: 38400 cycles; loop order K->C->RSYX.
            // K groups=10, C folds=1, C waves=1, row fold=9.
            SpatialMap(8,8) K;
            TemporalMap(1,1) C;
            TemporalMap(3,3) R;
            TemporalMap(3,3) S;
            TemporalMap(3,1) Y;
            TemporalMap(3,1) X;
        }
    }

    Layer Conv2D_76 {
        Type: Conv
        Stride { X: 1, Y: 1 }
        Dimensions { K: 80, C: 80, R: 1, S: 1, Y: 40, X: 40 }
        Dataflow {
            // Optimized cost: 270400 cycles; loop order K->C->YXRS.
            // K groups=10, C folds=2, C waves=2, row fold=40.
            SpatialMap(8,8) K;
            TemporalMap(40,40) C;
            TemporalMap(1,1) Y;
            TemporalMap(1,1) X;
            TemporalMap(1,1) R;
            TemporalMap(1,1) S;
        }
    }

    Layer Conv2D_77 {
        Type: Conv
        Stride { X: 1, Y: 1 }
        Dimensions { K: 80, C: 80, R: 1, S: 1, Y: 40, X: 40 }
        Dataflow {
            // Optimized cost: 270400 cycles; loop order K->C->YXRS.
            // K groups=10, C folds=2, C waves=2, row fold=40.
            SpatialMap(8,8) K;
            TemporalMap(40,40) C;
            TemporalMap(1,1) Y;
            TemporalMap(1,1) X;
            TemporalMap(1,1) R;
            TemporalMap(1,1) S;
        }
    }

    Layer Conv2D_78 {
        Type: Conv
        Stride { X: 2, Y: 2 }
        Dimensions { K: 128, C: 128, R: 3, S: 3, Y: 40, X: 40 }
        Dataflow {
            // Optimized cost: 961200 cycles; loop order K->C->RSYX.
            // K groups=16, C folds=32, C waves=32, row fold=36.
            SpatialMap(8,8) K;
            TemporalMap(4,4) C;
            TemporalMap(3,3) R;
            TemporalMap(3,3) S;
            TemporalMap(3,1) Y;
            TemporalMap(3,1) X;
        }
    }

    Layer Conv2D_79 {
        Type: Conv
        Stride { X: 1, Y: 1 }
        Dimensions { K: 256, C: 384, R: 1, S: 1, Y: 20, X: 20 }
        Dataflow {
            // Optimized cost: 315600 cycles; loop order K->C->YXRS.
            // K groups=32, C folds=6, C waves=6, row fold=64.
            SpatialMap(8,8) K;
            TemporalMap(64,64) C;
            TemporalMap(1,1) Y;
            TemporalMap(1,1) X;
            TemporalMap(1,1) R;
            TemporalMap(1,1) S;
        }
    }

    Layer Conv2D_80 {
        Type: Conv
        Stride { X: 1, Y: 1 }
        Dimensions { K: 64, C: 128, R: 3, S: 3, Y: 20, X: 20 }
        Dataflow {
            // Optimized cost: 487600 cycles; loop order K->C->RSYX.
            // K groups=8, C folds=32, C waves=16, row fold=36.
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

    Layer Conv2D_81 {
        Type: Conv
        Stride { X: 1, Y: 1 }
        Dimensions { K: 128, C: 64, R: 3, S: 3, Y: 20, X: 20 }
        Dataflow {
            // Optimized cost: 481200 cycles; loop order K->C->RSYX.
            // K groups=16, C folds=16, C waves=16, row fold=36.
            SpatialMap(8,8) K;
            TemporalMap(4,4) C;
            TemporalMap(3,3) R;
            TemporalMap(3,3) S;
            TemporalMap(3,1) Y;
            TemporalMap(3,1) X;
        }
    }

    Layer Conv2D_82 {
        Type: Conv
        Stride { X: 1, Y: 1 }
        Dimensions { K: 256, C: 128, R: 1, S: 1, Y: 20, X: 20 }
        Dataflow {
            // Optimized cost: 106000 cycles; loop order K->C->YXRS.
            // K groups=32, C folds=2, C waves=2, row fold=64.
            SpatialMap(8,8) K;
            TemporalMap(64,64) C;
            TemporalMap(1,1) Y;
            TemporalMap(1,1) X;
            TemporalMap(1,1) R;
            TemporalMap(1,1) S;
        }
    }

    Layer Conv2D_83 {
        Type: Conv
        Stride { X: 1, Y: 1 }
        Dimensions { K: 128, C: 1, R: 3, S: 3, Y: 20, X: 20 }
        Dataflow {
            // Optimized cost: 9600 cycles; loop order K->C->RSYX.
            // K groups=16, C folds=1, C waves=1, row fold=9.
            SpatialMap(8,8) K;
            TemporalMap(1,1) C;
            TemporalMap(3,3) R;
            TemporalMap(3,3) S;
            TemporalMap(3,1) Y;
            TemporalMap(3,1) X;
        }
    }

    Layer Conv2D_84 {
        Type: Conv
        Stride { X: 1, Y: 1 }
        Dimensions { K: 128, C: 128, R: 1, S: 1, Y: 20, X: 20 }
        Dataflow {
            // Optimized cost: 106000 cycles; loop order K->C->YXRS.
            // K groups=16, C folds=2, C waves=2, row fold=64.
            SpatialMap(8,8) K;
            TemporalMap(64,64) C;
            TemporalMap(1,1) Y;
            TemporalMap(1,1) X;
            TemporalMap(1,1) R;
            TemporalMap(1,1) S;
        }
    }

    Layer Conv2D_85 {
        Type: Conv
        Stride { X: 1, Y: 1 }
        Dimensions { K: 256, C: 128, R: 1, S: 1, Y: 20, X: 20 }
        Dataflow {
            // Optimized cost: 106000 cycles; loop order K->C->YXRS.
            // K groups=32, C folds=2, C waves=2, row fold=64.
            SpatialMap(8,8) K;
            TemporalMap(64,64) C;
            TemporalMap(1,1) Y;
            TemporalMap(1,1) X;
            TemporalMap(1,1) R;
            TemporalMap(1,1) S;
        }
    }

    Layer Conv2D_86 {
        Type: Conv
        Stride { X: 1, Y: 1 }
        Dimensions { K: 128, C: 256, R: 1, S: 1, Y: 20, X: 20 }
        Dataflow {
            // Optimized cost: 210800 cycles; loop order K->C->YXRS.
            // K groups=16, C folds=4, C waves=4, row fold=64.
            SpatialMap(8,8) K;
            TemporalMap(64,64) C;
            TemporalMap(1,1) Y;
            TemporalMap(1,1) X;
            TemporalMap(1,1) R;
            TemporalMap(1,1) S;
        }
    }

    Layer Conv2D_87 {
        Type: Conv
        Stride { X: 1, Y: 1 }
        Dimensions { K: 256, C: 384, R: 1, S: 1, Y: 20, X: 20 }
        Dataflow {
            // Optimized cost: 315600 cycles; loop order K->C->YXRS.
            // K groups=32, C folds=6, C waves=6, row fold=64.
            SpatialMap(8,8) K;
            TemporalMap(64,64) C;
            TemporalMap(1,1) Y;
            TemporalMap(1,1) X;
            TemporalMap(1,1) R;
            TemporalMap(1,1) S;
        }
    }

    Layer DepthwiseConv2D_88 {
        Type: DepthwiseConv
        Stride { X: 1, Y: 1 }
        Dimensions { K: 256, C: 1, R: 3, S: 3, Y: 20, X: 20 }
        Dataflow {
            // Optimized cost: 9600 cycles; loop order K->C->RSYX.
            // K groups=32, C folds=1, C waves=1, row fold=9.
            SpatialMap(8,8) K;
            TemporalMap(1,1) C;
            TemporalMap(3,3) R;
            TemporalMap(3,3) S;
            TemporalMap(3,1) Y;
            TemporalMap(3,1) X;
        }
    }

    Layer Conv2D_89 {
        Type: Conv
        Stride { X: 1, Y: 1 }
        Dimensions { K: 80, C: 256, R: 1, S: 1, Y: 20, X: 20 }
        Dataflow {
            // Optimized cost: 210800 cycles; loop order K->C->YXRS.
            // K groups=10, C folds=4, C waves=4, row fold=64.
            SpatialMap(8,8) K;
            TemporalMap(64,64) C;
            TemporalMap(1,1) Y;
            TemporalMap(1,1) X;
            TemporalMap(1,1) R;
            TemporalMap(1,1) S;
        }
    }

    Layer DepthwiseConv2D_90 {
        Type: DepthwiseConv
        Stride { X: 1, Y: 1 }
        Dimensions { K: 80, C: 1, R: 3, S: 3, Y: 20, X: 20 }
        Dataflow {
            // Optimized cost: 9600 cycles; loop order K->C->RSYX.
            // K groups=10, C folds=1, C waves=1, row fold=9.
            SpatialMap(8,8) K;
            TemporalMap(1,1) C;
            TemporalMap(3,3) R;
            TemporalMap(3,3) S;
            TemporalMap(3,1) Y;
            TemporalMap(3,1) X;
        }
    }

    Layer Conv2D_91 {
        Type: Conv
        Stride { X: 1, Y: 1 }
        Dimensions { K: 80, C: 80, R: 1, S: 1, Y: 20, X: 20 }
        Dataflow {
            // Optimized cost: 67600 cycles; loop order K->C->YXRS.
            // K groups=10, C folds=2, C waves=2, row fold=40.
            SpatialMap(8,8) K;
            TemporalMap(40,40) C;
            TemporalMap(1,1) Y;
            TemporalMap(1,1) X;
            TemporalMap(1,1) R;
            TemporalMap(1,1) S;
        }
    }

    Layer Conv2D_92 {
        Type: Conv
        Stride { X: 1, Y: 1 }
        Dimensions { K: 80, C: 80, R: 1, S: 1, Y: 20, X: 20 }
        Dataflow {
            // Optimized cost: 67600 cycles; loop order K->C->YXRS.
            // K groups=10, C folds=2, C waves=2, row fold=40.
            SpatialMap(8,8) K;
            TemporalMap(40,40) C;
            TemporalMap(1,1) Y;
            TemporalMap(1,1) X;
            TemporalMap(1,1) R;
            TemporalMap(1,1) S;
        }
    }

    Layer Conv2D_93 {
        Type: Conv
        Stride { X: 1, Y: 1 }
        Dimensions { K: 16, C: 256, R: 3, S: 3, Y: 20, X: 20 }
        Dataflow {
            // Optimized cost: 250800 cycles; loop order K->C->RSYX.
            // K groups=2, C folds=64, C waves=8, row fold=36.
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

    Layer Conv2D_94 {
        Type: Conv
        Stride { X: 1, Y: 1 }
        Dimensions { K: 16, C: 16, R: 3, S: 3, Y: 20, X: 20 }
        Dataflow {
            // Optimized cost: 18000 cycles; loop order K->C->RSYX.
            // K groups=2, C folds=8, C waves=1, row fold=18.
            SpatialMap(8,8) K;
            TemporalMap(2,2) C;
            TemporalMap(3,3) R;
            TemporalMap(3,3) S;
            TemporalMap(3,1) Y;
            TemporalMap(3,1) X;

            // Cluster C folds across 8 PEs to reduce latency.
            Cluster(8,P);
            SpatialMap(2,2) C;
            TemporalMap(3,3) R;
            TemporalMap(3,3) S;
            TemporalMap(3,1) Y;
            TemporalMap(3,1) X;
        }
    }

    Layer Conv2D_95 {
        Type: Conv
        Stride { X: 1, Y: 1 }
        Dimensions { K: 4, C: 16, R: 1, S: 1, Y: 20, X: 20 }
        Dataflow {
            // Optimized cost: 15200 cycles; loop order K->C->YXRS.
            // K groups=1, C folds=1, C waves=1, row fold=16.
            SpatialMap(4,4) K;
            TemporalMap(16,16) C;
            TemporalMap(1,1) Y;
            TemporalMap(1,1) X;
            TemporalMap(1,1) R;
            TemporalMap(1,1) S;
        }
    }

    Layer Conv2D_96 {
        Type: Conv
        Stride { X: 1, Y: 1 }
        Dimensions { K: 16, C: 128, R: 3, S: 3, Y: 40, X: 40 }
        Dataflow {
            // Optimized cost: 504000 cycles; loop order K->C->RSYX.
            // K groups=2, C folds=32, C waves=4, row fold=36.
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

    Layer Conv2D_97 {
        Type: Conv
        Stride { X: 1, Y: 1 }
        Dimensions { K: 16, C: 16, R: 3, S: 3, Y: 40, X: 40 }
        Dataflow {
            // Optimized cost: 72000 cycles; loop order K->C->RSYX.
            // K groups=2, C folds=8, C waves=1, row fold=18.
            SpatialMap(8,8) K;
            TemporalMap(2,2) C;
            TemporalMap(3,3) R;
            TemporalMap(3,3) S;
            TemporalMap(3,1) Y;
            TemporalMap(3,1) X;

            // Cluster C folds across 8 PEs to reduce latency.
            Cluster(8,P);
            SpatialMap(2,2) C;
            TemporalMap(3,3) R;
            TemporalMap(3,3) S;
            TemporalMap(3,1) Y;
            TemporalMap(3,1) X;
        }
    }

    Layer Conv2D_98 {
        Type: Conv
        Stride { X: 1, Y: 1 }
        Dimensions { K: 4, C: 16, R: 1, S: 1, Y: 40, X: 40 }
        Dataflow {
            // Optimized cost: 60800 cycles; loop order K->C->YXRS.
            // K groups=1, C folds=1, C waves=1, row fold=16.
            SpatialMap(4,4) K;
            TemporalMap(16,16) C;
            TemporalMap(1,1) Y;
            TemporalMap(1,1) X;
            TemporalMap(1,1) R;
            TemporalMap(1,1) S;
        }
    }

    Layer Conv2D_99 {
        Type: Conv
        Stride { X: 1, Y: 1 }
        Dimensions { K: 16, C: 64, R: 3, S: 3, Y: 80, X: 80 }
        Dataflow {
            // Optimized cost: 1017600 cycles; loop order K->C->RSYX.
            // K groups=2, C folds=16, C waves=2, row fold=36.
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

    Layer Conv2D_100 {
        Type: Conv
        Stride { X: 1, Y: 1 }
        Dimensions { K: 16, C: 16, R: 3, S: 3, Y: 80, X: 80 }
        Dataflow {
            // Optimized cost: 288000 cycles; loop order K->C->RSYX.
            // K groups=2, C folds=8, C waves=1, row fold=18.
            SpatialMap(8,8) K;
            TemporalMap(2,2) C;
            TemporalMap(3,3) R;
            TemporalMap(3,3) S;
            TemporalMap(3,1) Y;
            TemporalMap(3,1) X;

            // Cluster C folds across 8 PEs to reduce latency.
            Cluster(8,P);
            SpatialMap(2,2) C;
            TemporalMap(3,3) R;
            TemporalMap(3,3) S;
            TemporalMap(3,1) Y;
            TemporalMap(3,1) X;
        }
    }

    Layer Conv2D_101 {
        Type: Conv
        Stride { X: 1, Y: 1 }
        Dimensions { K: 4, C: 16, R: 1, S: 1, Y: 80, X: 80 }
        Dataflow {
            // Optimized cost: 243200 cycles; loop order K->C->YXRS.
            // K groups=1, C folds=1, C waves=1, row fold=16.
            SpatialMap(4,4) K;
            TemporalMap(16,16) C;
            TemporalMap(1,1) Y;
            TemporalMap(1,1) X;
            TemporalMap(1,1) R;
            TemporalMap(1,1) S;
        }
    }

    // K-spatial PE count summary:
    // For SpatialMap(KTile,KTile) K, KSpatialPEs = ceil(K / KTile).
    // Example: SpatialMap(8,8) K with K=16 uses 2 PEs.
    // Layers counted: 102 (Conv2D through Conv2D_101).
    // Total KSpatialPEs across all layers: 1098.
    // Total ActivePEs including Cluster(...,P) parallelism: 1686.

}
