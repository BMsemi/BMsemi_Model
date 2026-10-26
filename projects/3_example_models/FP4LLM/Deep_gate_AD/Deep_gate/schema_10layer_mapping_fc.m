Network Schema_10Layer_MLP {

	// QuantLinear 640->128, activation=relu, first layer
	Layer FC0 {
		Type: FC
		Stride { X: 1, Y: 1 }
		Dimensions { K: 128, C: 640, R: 1, S: 1, Y: 1, X: 1 }
		Dataflow {
			// Map output channels spatially, fold input channels temporally
			SpatialMap(8,8) K;
			TemporalMap(32,32) C;
			TemporalMap(1,1) Y;
			TemporalMap(1,1) X;
			TemporalMap(1,1) R;
			TemporalMap(1,1) S;

		}
	}

	// QuantLinear 128->128, activation=relu
	Layer FC1 {
		Type: FC
		Stride { X: 1, Y: 1 }
		Dimensions { K: 128, C: 128, R: 1, S: 1, Y: 1, X: 1 }
		Dataflow {
		    SpatialMap(8,8) K;
			TemporalMap(32,32) C;
			TemporalMap(1,1) Y;
			TemporalMap(1,1) X;
			TemporalMap(1,1) R;
			TemporalMap(1,1) S;
		}
	}

	// QuantLinear 128->128, activation=relu
	Layer FC2 {
		Type: FC
		Stride { X: 1, Y: 1 }
		Dimensions { K: 128, C: 128, R: 1, S: 1, Y: 1, X: 1 }
		Dataflow {
			SpatialMap(8,8) K;
			TemporalMap(32,32) C;
			TemporalMap(1,1) Y;
			TemporalMap(1,1) X;
			TemporalMap(1,1) R;
			TemporalMap(1,1) S;
		}
	}

	// QuantLinear 128->128, activation=relu
	Layer FC3 {
		Type: FC
		Stride { X: 1, Y: 1 }
		Dimensions { K: 128, C: 128, R: 1, S: 1, Y: 1, X: 1 }
		Dataflow {
			SpatialMap(8,8) K;
			TemporalMap(32,32) C;
			TemporalMap(1,1) Y;
			TemporalMap(1,1) X;
			TemporalMap(1,1) R;
			TemporalMap(1,1) S;
		}
	}

	// QuantLinear 128->8, activation=relu
	Layer FC4 {
		Type: FC
		Stride { X: 1, Y: 1 }
		Dimensions { K: 8, C: 128, R: 1, S: 1, Y: 1, X: 1 }
		Dataflow {
			SpatialMap(8,8) K;
			TemporalMap(32,32) C;
			TemporalMap(1,1) Y;
			TemporalMap(1,1) X;
			TemporalMap(1,1) R;
			TemporalMap(1,1) S;
		}
	}

	// QuantLinear 8->128, activation=relu
	Layer FC5 {
		Type: FC
		Stride { X: 1, Y: 1 }
		Dimensions { K: 128, C: 8, R: 1, S: 1, Y: 1, X: 1 }
		Dataflow {
			SpatialMap(8,8) K;
			TemporalMap(32,32) C;
			TemporalMap(1,1) Y;
			TemporalMap(1,1) X;
			TemporalMap(1,1) R;
			TemporalMap(1,1) S;
		}
	}

	// QuantLinear 128->128, activation=relu
	Layer FC6 {
		Type: FC
		Stride { X: 1, Y: 1 }
		Dimensions { K: 128, C: 128, R: 1, S: 1, Y: 1, X: 1 }
		Dataflow {
			SpatialMap(8,8) K;
			TemporalMap(32,32) C;
			TemporalMap(1,1) Y;
			TemporalMap(1,1) X;
			TemporalMap(1,1) R;
			TemporalMap(1,1) S;
		}
	}

	// QuantLinear 128->128, activation=relu
	Layer FC7 {
		Type: FC
		Stride { X: 1, Y: 1 }
		Dimensions { K: 128, C: 128, R: 1, S: 1, Y: 1, X: 1 }
		Dataflow {
			SpatialMap(8,8) K;
			TemporalMap(32,32) C;
			TemporalMap(1,1) Y;
			TemporalMap(1,1) X;
			TemporalMap(1,1) R;
			TemporalMap(1,1) S;
		}
	}

	// QuantLinear 128->128, activation=relu
	Layer FC8 {
		Type: FC
		Stride { X: 1, Y: 1 }
		Dimensions { K: 128, C: 128, R: 1, S: 1, Y: 1, X: 1 }
		Dataflow {
			SpatialMap(8,8) K;
			TemporalMap(32,32) C;
			TemporalMap(1,1) Y;
			TemporalMap(1,1) X;
			TemporalMap(1,1) R;
			TemporalMap(1,1) S;
		}
	}

	// QuantLinear 128->640, activation=None, output layer
	Layer FC9 {
		Type: FC
		Stride { X: 1, Y: 1 }
		Dimensions { K: 640, C: 128, R: 1, S: 1, Y: 1, X: 1 }
		Dataflow {
			SpatialMap(8,8) K;
			TemporalMap(32,32) C;
			TemporalMap(1,1) Y;
			TemporalMap(1,1) X;
			TemporalMap(1,1) R;
			TemporalMap(1,1) S;
		}
	}

}