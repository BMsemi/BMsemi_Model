from numbers import Integral, Real
import numpy as np

class EnergyCalculator:
    def __init__(self, *, output_buffer_store_energy_per_buffer=None,
                 output_buffer_read_energy_per_buffer=None):
        # Joules for one full output FIFO fill/drain, supplied separately from array costs.
        # None means unconfigured, not zero energy.
        for name, value in (
            ("output_buffer_store_energy_per_buffer", output_buffer_store_energy_per_buffer),
            ("output_buffer_read_energy_per_buffer", output_buffer_read_energy_per_buffer),
        ):
            if value is not None and (
                isinstance(value, bool) or not isinstance(value, Real)
                or not np.isfinite(value) or value < 0
            ):
                raise ValueError(f"{name} must be a finite nonnegative energy in joules, or None.")
            setattr(self, name, None if value is None else float(value))
        # All energy constants are stored in joules.
        self.E_SET = 10.7882 * 1e-9  # During program mode and read mode in nj 130nm node
        self.E_RESET =6.994  * 1e-9   # During program mode and read mode in nj 130nm node
        self.R_SET = 284.484 * 1e-15  # Read energy after set during the compute mode fj
        self.R_RESET = 37.37 * 1e-15  # Read energy after reset during the compute mode f
        self.Decoder_driver_energy = 27 * 1e-12  # per operation energy for decoder and driver in pj
        self.adc_energy_per_column = 2.67 * 1e-9  # per column adc energy in nj
        # self.E_SET = 44.95 * 1e-12  # During program mode and read mode in pj 7nm node
        # self.E_RESET = 29.14  * 1e-12   # During program mode and read mode in pj 7nm node
        # self.R_SET = 47.41 * 1e-15  # Read energy after set during the compute mode fj
        # self.R_RESET = 6.23 * 1e-15  # Read energy after reset during the compute mode fj
        # self.Decoder_driver_energy = 4.59 * 1e-12  # per operation energy for decoder and driver in pj 
        # self.adc_energy_per_column = 8.5 * 1e-12   # per column adc energy in pj 


        
    def compute_energy(self, input_vector, weight_matrix):
       """
       Unified compute energy function:
       - Handles both binary and non-binary input vectors in one place.
       - For binary inputs (exact 0/1):
            Only rows with input == 1 contribute energy:
                E_row = (#1s * R_SET) + (#0s * R_RESET)
       - For non-binary inputs (e.g., analog or softmax values):
            Each cell energy:
                E_cell = R_RESET + (R_SET - R_RESET) * input_val * weight_bit
       - Input values are clipped to [0,1] to maintain physical meaning.
       """

       x = np.asarray(input_vector, dtype=float).reshape(-1)
       W = np.asarray(weight_matrix, dtype=float)

       if W.shape[0] != x.size:
          raise ValueError(f"Input length {x.size} must match weight rows {W.shape[0]}")

       # Check if input is strictly binary {0,1}
       is_binary = np.all(np.isin(x, [0, 1]))

       total_energy = 0.0

       if is_binary:
           # --------------------------
           # 🔹 Binary Energy Calculation
           # --------------------------
           active_rows = np.where(x == 1)[0]# this for ignoring non zero  input vector

           for row_idx in active_rows:
               row = W[row_idx]
               num_set = np.sum(row == 1)
               num_reset = len(row) - num_set
               row_energy = (num_set * self.R_SET) + (num_reset * self.R_RESET) 
               total_energy += row_energy 

       else:
           # --------------------------
           # 🔹 Non-Binary Energy Calculation
           # --------------------------
           # Clip to [0,1] range for analog values
           x = np.clip(x, 0.0, 1.0)

           # Effective drive: element-wise multiply input and weight
           drive = x[:, None] * W

           # Interpolated energy model
           E_cell =  (self.R_SET - self.R_RESET) * drive

           total_energy = float(np.sum(E_cell))

       return total_energy

    def read_energy(self, matrix):
        """Calculate energy for 262144 read operations"""
        total_energy = 0
    
        for row in matrix:
            for cell in row:
                if cell == 1:
                    total_energy += self.E_SET
                else:
                    total_energy += self.E_RESET
        return total_energy

    def program_energy(self, matrix):
        """Calculate energy for 262144 program operations"""
        total_energy = 0

        for row in matrix:
            for cell in row:
                if cell == 1:
                    total_energy += self.E_SET
                else:
                    total_energy += self.E_RESET
        return total_energy
    
    def decoder_driver_energy(self, num_cells=262144):
        """Calculate decoder and driver energy for given number of operations"""
        no_of_loops = num_cells
        return no_of_loops * self.Decoder_driver_energy
    
    def adc_energy(self, num_columns=512):
        """Calculate ADC energy for 512 columns"""
        no_of_columns = num_columns
        return no_of_columns * self.adc_energy_per_column

    def periphery_energy(self, matrix):
        """Decoder/driver plus ADC energy for the physical array dimensions."""
        matrix = np.asarray(matrix)
        if matrix.ndim != 2:
            raise ValueError("Energy accounting requires a 2D weight matrix.")
        return self.decoder_driver_energy(matrix.size) + self.adc_energy(matrix.shape[1])

    def compute_phase_energy(self, input_vector, matrix):
        """One computation: array + decoder/driver + ADC."""
        return self.compute_energy(input_vector, matrix) + self.periphery_energy(matrix)

    def program_phase_energy(self, matrix):
        """One array programming operation: array + decoder/driver + ADC."""
        return self.program_energy(matrix) + self.periphery_energy(matrix)

    def read_phase_energy(self, matrix):
        """One array-read estimate: array + decoder/driver + ADC (not FIFO read)."""
        return self.read_energy(matrix) + self.periphery_energy(matrix)

    @staticmethod
    def _output_buffer_energy(num_words, buffer_size, energy_per_buffer):
        """Scale a full-buffer cost by words transferred; partial transfers scale linearly."""
        for name, value, minimum in (("num_words", num_words, 0), ("buffer_size", buffer_size, 1)):
            if isinstance(value, bool) or not isinstance(value, Integral) or value < minimum:
                raise ValueError(f"{name} must be an integer >= {minimum}.")
        if num_words == 0:
            return 0.0
        if energy_per_buffer is None:
            return None
        return (num_words / buffer_size) * energy_per_buffer

    def output_buffer_store_energy(self, num_words, buffer_size):
        """Digital FIFO STORE energy in joules, or None when its cost is unset."""
        return self._output_buffer_energy(
            num_words, buffer_size, self.output_buffer_store_energy_per_buffer
        )

    def output_buffer_read_energy(self, num_words, buffer_size):
        """Digital FIFO READ energy in joules, or None when its cost is unset."""
        return self._output_buffer_energy(
            num_words, buffer_size, self.output_buffer_read_energy_per_buffer
        )
    
    def read_energy_first_row(self, matrix):
        """Calculate read energy for only the first row (row index 0) using E_SET/E_RESET."""
        total_energy = 0.0
        first_row = matrix[0]
        for cell in first_row:
            total_energy += self.E_SET if cell == 1 else self.E_RESET
        return total_energy
    
    def read_energy_first_cell(self, matrix):
        """Return read energy for only cell (0,0) using E_SET/E_RESET. No periphery."""
        val = matrix[0][0]
        return self.E_SET if val == 1 else self.E_RESET

    # def print_energy_report(self, input_vector, weight_matrix):
    #     """Print detailed energy report"""
    #     compute_e = self.compute_energy(input_vector, weight_matrix)
    #     read_e = self.read_energy(weight_matrix)
    #     program_e = self.program_energy(weight_matrix)
        
    #     print("\n=== ENERGY REPORT ===")
    #     print(f"Active input rows: {np.sum(input_vector == 1)} of {len(input_vector)}")
    #     print(f"Compute Energy: {compute_e:.6e} J")
    #     print(f"Read Energy: {read_e:.6e} J")
    #     print(f"Program Energy: {program_e:.6e} J")
    #     print(f"Total Energy: {compute_e + read_e + program_e:.6e} J")
    #     print("===================\n")

