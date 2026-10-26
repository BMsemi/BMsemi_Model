import numpy as np
class CrossbarArray:
    """32x32 crossbar performing analog-style vector-matrix multiplication."""
    def __init__(self, input_size=32, output_size=32):
        self.input_size = input_size
        self.output_size = output_size
        self.weights = np.zeros((input_size, output_size))

    def program_weights(self, weight_matrix):
        if weight_matrix.shape != (self.input_size, self.output_size):
            raise ValueError("Weight matrix size mismatch.")
        self.weights = weight_matrix
        print("[PROGRAM] Crossbar weights updated.\n")

    def compute(self, input_vector):
        if len(input_vector) != self.input_size:
            raise ValueError("Input vector size mismatch.")
        output_vector = np.dot(input_vector, self.weights)
        return output_vector