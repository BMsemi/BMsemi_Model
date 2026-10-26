Network MNIST_CrossbarPE_Array {
  Layer Conv3x3 {
    Type: CONV
    Stride { X: 1, Y: 1 }

    Dimensions { K: 2, C: 1, Y: 28, X: 28, R: 5, S: 5 }

    Dataflow {
      Merge(XY, Y, X);              # Flatten 28x28 = 784 positions

      SpatialMap(25,25) XY;           # Map each pixel to a different PE

      TemporalMap(1,1) K;
      TemporalMap(1,1) C;

      TemporalMap(5,5) R;           # Do the 5x5 dot product inside the PE over time
      TemporalMap(5,5) S;
    }
  }
}