import numpy as np

class EnergyCalculator:
    def __init__(self):
        # Energy constants in femtojoules (fJ)
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
           active_rows = np.where(x == 1)[0]

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
        """Calculate energy for 1024 read operations"""
        total_energy = 0
    
        for row in matrix:
            for cell in row:
                if cell == 1:
                    total_energy += self.E_SET
                else:
                    total_energy += self.E_RESET
        return total_energy

    def program_energy(self, matrix):
        """Calculate energy for 1024 program operations"""
        total_energy = 0

        for row in matrix:
            for cell in row:
                if cell == 1:
                    total_energy += self.E_SET
                else:
                    total_energy += self.E_RESET
        return total_energy
    
    def decoder_driver_energy(self):
        """Calculate decoder and driver energy for given number of operations"""
        no_of_loops = 1024
        return no_of_loops * self.Decoder_driver_energy
    
    def adc_energy(self):
        """Calculate ADC energy for 32 columns"""
        no_of_columns = 32
        return no_of_columns * self.adc_energy_per_column
    
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