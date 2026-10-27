import numpy as np


class CrossbarArray:
    """Ideal crossbar arithmetic; transaction timing is handled by NeuromorphicX2."""

    def __init__(self, input_size=512, output_size=512):
        if input_size <= 0 or output_size <= 0:
            raise ValueError("Array dimensions must be positive.")
        self.input_size = input_size
        self.output_size = output_size
        self.weights = np.zeros((input_size, output_size))
        self.programmed = False

    def program_weights(self, weight_matrix):
        matrix = np.asarray(weight_matrix, dtype=float)
        if matrix.shape != (self.input_size, self.output_size):
            raise ValueError("Weight matrix size mismatch.")
        if not np.isfinite(matrix).all():
            raise ValueError("Weights must be finite.")
        self.weights = matrix.copy()
        self.programmed = True

    def compute(self, input_vector):
        if not self.programmed:
            raise RuntimeError("Crossbar must be programmed before computing.")
        vector = np.asarray(input_vector, dtype=float)
        if vector.shape != (self.input_size,):
            raise ValueError("Input vector size mismatch.")
        return np.dot(vector, self.weights)

