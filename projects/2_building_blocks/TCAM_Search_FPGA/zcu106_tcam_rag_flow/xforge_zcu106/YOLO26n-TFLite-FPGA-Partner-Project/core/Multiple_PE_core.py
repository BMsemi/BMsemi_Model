import numpy as np
from .Crossbar_array import CrossbarArray
from .neuromorphic_x2 import NeuromorphicX2

class MultiPECore:
    """Manager for multiple CrossbarArray PEs."""
    def __init__(self, num_pes=512, input_size=64, output_size=64):
        self.num_pes = num_pes
        self.input_size = input_size
        self.output_size = output_size
        self.crossbars = [CrossbarArray(input_size, output_size) for _ in range(num_pes)]
        # Optional buffers per PE
        self.inputs = [np.zeros(input_size) for _ in range(num_pes)]
        self.outputs = [np.zeros(output_size) for _ in range(num_pes)]

    def program_weights_all(self, weight_matrices):
        """
        Program weights for all PEs.
        weight_matrices can be:
          - A single (64x64) np.array to broadcast to all PEs
          - A list of length num_pes with per-PE (64x64) matrices
        """
        if isinstance(weight_matrices, np.ndarray):
            # Broadcast one matrix to all
            for pe in range(self.num_pes):
                self.crossbars[pe].program_weights(weight_matrices)
        else:
            if len(weight_matrices) != self.num_pes:
                raise ValueError("weight_matrices must be length num_pes.")
            for pe in range(self.num_pes):
                self.crossbars[pe].program_weights(weight_matrices[pe])

    def load_inputs_all(self, input_vectors):
        """
        Load inputs for all PEs.
        input_vectors can be:
          - A single (64,) np.array to broadcast to all PEs
          - A list of length num_pes with per-PE (64,) vectors
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
        out = self.crossbars[pe_idx].compute(self.inputs[pe_idx])
        self.outputs[pe_idx] = out
        return out

    def compute_all(self):
        """Compute outputs for all PEs."""
        for pe in range(self.num_pes):
            self.outputs[pe] = self.crossbars[pe].compute(self.inputs[pe])
        return self.outputs

    def get_output(self, pe_idx):
        return self.outputs[pe_idx]
    
class MultiPENeuromorphicCore:
    """
    Multi-PE wrapper where EACH PE is a NeuromorphicX2 instance.
    Use this for mapping1.m-driven scheduling so the PE micro-architecture is preserved.
    """
    def __init__(self, num_pes: int, input_size: int = 64, output_size: int = 64, iterations: int = 0):
        self.num_pes = int(num_pes)
        self.input_size = int(input_size)
        self.output_size = int(output_size)
        self.iterations = int(iterations)

        self.pes = [NeuromorphicX2(iterations=self.iterations) for _ in range(self.num_pes)]
        self.outputs = [np.zeros(self.output_size, dtype=float) for _ in range(self.num_pes)]

    def program_weights_all(self, weight_tile: np.ndarray):
        """Program the same 64x64 tile into every PE crossbar."""
        weight_tile = np.asarray(weight_tile)
        for pe in self.pes:
            pe.crossbar.program_weights(weight_tile)


    def program_weights_one(self, pe_idx: int, weight_tile: np.ndarray):
        """Program a 64x64 tile into a single PE crossbar."""
        weight_tile = np.asarray(weight_tile)
        self.pes[int(pe_idx)].crossbar.program_weights(weight_tile)

    def program_weights_range(self, start_pe: int, end_pe: int, weight_tile: np.ndarray):
        """
        Program a 64x64 tile into PEs in [start_pe, end_pe).
        Useful for Cluster(...) scheduling.
        """
        weight_tile = np.asarray(weight_tile)
        s = int(start_pe)
        e = int(end_pe)
        if s < 0 or e > self.num_pes or s > e:
            raise ValueError(f"Invalid PE range [{s}, {e}) for num_pes={self.num_pes}")
        for pe_id in range(s, e):
            self.pes[pe_id].crossbar.program_weights(weight_tile)

    def compute_pe(self, pe_idx: int, input_vec64: np.ndarray) -> np.ndarray:
        """Run one compute on a single PE using NeuromorphicX2 flow."""
        pe = self.pes[int(pe_idx)]
        # Select dtype based on PE mode
        mode = getattr(pe.crossbar, "mode", None)
        if mode == getattr(CrossbarArray, "MODE_PWM_INT8_ROW_SERIAL", "PWM_INT8_ROW_SERIAL"):
            # As per PWM row-serial INT8 method: activations are unsigned [0,255]
            v = np.asarray(input_vec64, dtype=np.uint8)
        else:
            v = np.asarray(input_vec64, dtype=float)

        if v.shape[0] != self.input_size:
            raise ValueError(f"input_vec64 must be length {self.input_size}, got {v.shape}")

        # clear FIFOs and run one compute
        pe.input_fifo.clear()
        pe.compute_fifo.clear()
        pe.output_fifo.clear()

        pe.load_inputs(v)
        pe.command(0x02)   # COMPUTE
        pe.run_cycle()

        # Preserve output dtype (PWM mode returns INT32, legacy modes return float)
        out = np.asarray(pe.output_fifo.buffer).astype(np.int32 if getattr(pe.crossbar, "mode", None) == getattr(CrossbarArray, "MODE_PWM_INT8_ROW_SERIAL", "PWM_INT8_ROW_SERIAL") else float)
        self.outputs[int(pe_idx)] = out
        return out


    def compute_pe_fast(self, pe_idx: int, input_vec64: np.ndarray) -> np.ndarray:
        """
        Fast/silent path: directly performs the MVM on the PE crossbar without
        going through FIFO + controller prints.

        This is useful for large clustered schedules (e.g., MobileNetV2),
        where the "full" NeuromorphicX2 flow can be too verbose/slow.
        """
        pe = self.pes[int(pe_idx)]
        # Select dtype based on PE mode
        mode = getattr(pe.crossbar, "mode", None)
        if mode == getattr(CrossbarArray, "MODE_PWM_INT8_ROW_SERIAL", "PWM_INT8_ROW_SERIAL"):
            # As per PWM row-serial INT8 method: activations are unsigned [0,255]
            v = np.asarray(input_vec64, dtype=np.uint8)
        else:
            v = np.asarray(input_vec64, dtype=float)
        if v.shape[0] != self.input_size:
            raise ValueError(f"input_vec64 must be length {self.input_size}, got {v.shape}")

        # Preserve output dtype (PWM mode returns INT32, legacy modes return float)
        out_raw = pe.crossbar.compute(v)
        out = np.asarray(out_raw).reshape(-1)
        self.outputs[int(pe_idx)] = out
        return out

    def get_output(self, pe_idx: int) -> np.ndarray:
        return self.outputs[int(pe_idx)]