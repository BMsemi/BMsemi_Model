import numpy as np
from neuromorphic_x2 import NeuromorphicX2

class MultiPECore:
    """Manager for multiple CrossbarArray PEs."""
    def __init__(self, num_pes=512, input_size=512, output_size=512, timing=None):
        self.num_pes = num_pes
        self.input_size = input_size
        self.output_size = output_size
        self.pes = [NeuromorphicX2(input_size=input_size, output_size=output_size, timing=timing) for _ in range(num_pes)]
        self.crossbars = [pe.crossbar for pe in self.pes]
        # Optional buffers per PE
        self.inputs = [np.zeros(input_size) for _ in range(num_pes)]
        self.outputs = [np.zeros(output_size) for _ in range(num_pes)]

    def program_weights_all(self, weight_matrices):
        """
        Program weights for all PEs.
        weight_matrices can be:
          - A single (512x512) np.array to broadcast to all PEs
          - A list of length num_pes with per-PE (512x512) matrices
        """
        if isinstance(weight_matrices, np.ndarray):
            # Broadcast one matrix to all
            for pe in range(self.num_pes):
                self.pes[pe].program_weights(weight_matrices)
        else:
            if len(weight_matrices) != self.num_pes:
                raise ValueError("weight_matrices must be length num_pes.")
            for pe in range(self.num_pes):
                self.pes[pe].program_weights(weight_matrices[pe])

    def load_inputs_all(self, input_vectors):
        """
        Load inputs for all PEs.
        input_vectors can be:
          - A single (512,) np.array to broadcast to all PEs
          - A list of length num_pes with per-PE (512,) vectors
        """
        if isinstance(input_vectors, np.ndarray):
            for pe in range(self.num_pes):
                if input_vectors.shape[0] != self.input_size:
                    raise ValueError("Input vector size mismatch.")
                self.inputs[pe] = input_vectors.copy()
        else:
            if len(input_vectors) != self.num_pes:
                raise ValueError("input_vectors must be length num_pes.")
            for pe in range(self.num_pes):
                vec = input_vectors[pe]
                if vec.shape[0] != self.input_size:
                    raise ValueError("Input vector size mismatch.")
                self.inputs[pe] = vec.copy()

    def compute_pe(self, pe_idx):
        """Compute output for a single PE."""
        pe = self.pes[pe_idx]
        pe.load_inputs(self.inputs[pe_idx])
        pe.command(pe.COMPUTE)
        pe.run_until_idle()
        out = pe.read_outputs()
        self.outputs[pe_idx] = out
        return out

    def compute_all(self):
        """Compute outputs for all PEs."""
        for pe in range(self.num_pes):
            self.compute_pe(pe)
        return self.outputs

    def get_output(self, pe_idx):
        return self.outputs[pe_idx]
    
class MultiPENeuromorphicCore:
    """
    Multi-PE wrapper where EACH PE is a NeuromorphicX2 instance.
    Use this for mapping1.m-driven scheduling so the PE micro-architecture is preserved.
    """
    def __init__(self, num_pes: int, input_size: int = 512, output_size: int = 512, iterations: int = 0, timing=None):
        self.num_pes = int(num_pes)
        self.input_size = int(input_size)
        self.output_size = int(output_size)
        self.iterations = int(iterations)

        self.pes = [NeuromorphicX2(iterations=self.iterations, input_size=self.input_size, output_size=self.output_size, timing=timing) for _ in range(self.num_pes)]
        self.outputs = [np.zeros(self.output_size, dtype=float) for _ in range(self.num_pes)]

    def program_weights_all(self, weight_tile: np.ndarray):
        """Program the same 512x512 tile into every PE crossbar."""
        weight_tile = np.asarray(weight_tile)
        for pe in self.pes:
            pe.program_weights(weight_tile)

    def compute_pe(self, pe_idx: int, input_vec512: np.ndarray) -> np.ndarray:
        """Run one compute on a single PE using NeuromorphicX2 flow."""
        pe = self.pes[int(pe_idx)]
        v = np.asarray(input_vec512, dtype=float)

        if v.shape[0] != self.input_size:
            raise ValueError(f"input_vec512 must be length {self.input_size}, got {v.shape}")

        pe.load_inputs(v)
        pe.command(0x02)   # COMPUTE
        pe.run_until_idle()

        out = pe.read_outputs()
        self.outputs[int(pe_idx)] = out
        return out

    def get_output(self, pe_idx: int) -> np.ndarray:
        return self.outputs[int(pe_idx)]
