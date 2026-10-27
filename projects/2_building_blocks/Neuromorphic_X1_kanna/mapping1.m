Network MNIST_CrossbarPE_Array {
  # Simulator extension: physical capacity, independent of workload dimensions.
  Hardware { Rows: 512, Columns: 512 }

  Layer Conv5x5 {
    Type: CONV
    Stride { X: 1, Y: 1 }

    # One active 25-row block; eight columns encode four differential edge filters.
    Dimensions { K: 8, C: 1, Y: 28, X: 28, R: 5, S: 5 }

    # Intended single-PE window traversal: 5x5 windows, stride 1.
    # X/Y sizes are input extents; SAME-padded windows traverse at Stride.
    Dataflow {
      SpatialMap(8,8) K;
      TemporalMap(1,1) C;
      TemporalMap(5,5) R;
      TemporalMap(5,1) X;
      TemporalMap(5,5) S;
      TemporalMap(5,1) Y;
    }
  }
}

