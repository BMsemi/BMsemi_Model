# Full-capacity single-PE workload for chip_demo.py.
Network Crossbar512 {
  Hardware { Rows: 512, Columns: 512 }

  Layer Dense512 {
    Type: CONV
    Stride { X: 1, Y: 1 }
    Dimensions { K: 512, C: 512, Y: 1, X: 1, R: 1, S: 1 }
    # XY positions are derived from Y * X in row-major order.
    Dataflow {
      SpatialMap(1,1) XY;
      TemporalMap(512,512) K;
      TemporalMap(512,512) C;
      TemporalMap(1,1) R;
      TemporalMap(1,1) S;
    }
  }
}

