from pathlib import Path
from PE_quantization import QuantizedPWMSystem
import numpy as np
import math

def print_weight_matrix(weight_matrix, weights_per_row):
    """Pretty print the weight matrix"""
    for row_idx, row in enumerate(weight_matrix):
        print(f"\nRow {row_idx}:")
        for col_idx, weight in enumerate(row[:weights_per_row]):
            print(f"Weight[{col_idx}]: 0x{weight:04X}", end="  ")
        print()

def split_input_to_chunks(input_value, total_bits, chunk_size):
    """Split input into chunks from LSB to MSB"""
    chunks = []
    for i in range(0, total_bits, chunk_size):
        # Extract chunk_size bits starting from LSB
        mask = (1 << chunk_size) - 1
        chunk = (input_value >> i) & mask
        chunks.append(chunk)
    return chunks

if __name__ == "__main__":
    config_path = Path("config.ini")
    print(f"Absolute config path: {config_path.absolute()}")
    
    # Initialize the quantized system
    system = QuantizedPWMSystem(config_path)
    
    # Get configuration limits from correct sections
    num_rows = system.config.getint('Crossbar', 'array_rows')
    num_columns = system.config.getint('Crossbar', 'array_columns')
    weight_bits = system.config.getint('Quantization', 'weight_bits')
    cell_bits = system.config.getint('Crossbar', 'cell_weight_bits')
    input_bits = system.config.getint('Quantization', 'input_bits')
    dac_bits = system.config.getint('DAC', 'num_bits')
    
    # Calculate maximum values
    max_weight = (1 << weight_bits) - 1
    max_input = (1 << input_bits) - 1
    
    # Calculate capacity and weight distribution
    cells_per_weight = math.ceil(weight_bits / cell_bits)
    weights_per_row = num_columns // (2 * cells_per_weight)
    
    print(f"\nSystem Configuration:")
    print(f"Input bits: {input_bits}, DAC bits per cycle: {dac_bits}")
    print(f"Number of cycles needed: {math.ceil(input_bits/dac_bits)}")
    
    # Example weight matrix
    weight_matrix = [
        
        # [0x1C11, 0x2D22, 0x3E33, 0x4F44],  # Row 0 weights (for 48 bit weights in a row)
        # [0x5525, 0x6636, 0x7747, 0x8858],  # Row 1 weights (for 48 bit weights in a row)
        # [0x9991, 0xAAA2, 0xBBB3, 0xCCC4],  # Row 2 weights (for 48 bit weights in a row)

        [0x1C11, 0x2D22, 0x3E33],  # Row 0 weights
        [0x5525, 0x6636, 0x7747],  # Row 1 weights
        [0x9991, 0xAAA2, 0xBBB3],  # Row 2 weights
    ]
    
    # Convert weight matrix to list and validate
    weights = []
    active_rows = []
    
    print("\nProcessing Weight Matrix:")
    for row_idx, row_weights in enumerate(weight_matrix):
        if len(row_weights) > weights_per_row:
            print(f"Warning: Row {row_idx} has {len(row_weights)} weights, truncating to {weights_per_row}")
            row_weights = row_weights[:weights_per_row]
        
        for col_idx, weight in enumerate(row_weights):
            if weight > max_weight:
                raise ValueError(f"Weight at row {row_idx}, position {col_idx} (0x{weight:X}) exceeds maximum (0x{max_weight:X})")
            weights.append(weight)
        active_rows.append(row_idx)
    
    # Set the weights in the system
    system.set_weights(weights)
    
    # Define different inputs for each row (16-bit examples)
    row_inputs = {
        0: 0xABCD,  # Row 0 input
        1: 0x1234,  # Row 1 input
        2: 0x5678   # Row 2 input
    }
    
    # Validate inputs
    for row, input_value in row_inputs.items():
        if input_value > max_input:
            raise ValueError(f"Input for row {row} (0x{input_value:X}) exceeds maximum (0x{max_input:X})")
    
    # Process inputs cycle by cycle
    num_cycles = math.ceil(input_bits / dac_bits)
    all_cycle_outputs = []
    
    print("\nProcessing inputs cycle by cycle:")
    for cycle in range(num_cycles):
        print(f"\nCycle {cycle + 1}:")
        cycle_inputs = []
        cycle_rows = []
        
        # Prepare inputs for this cycle
        for row in active_rows:
            if row in row_inputs:
                input_chunks = split_input_to_chunks(row_inputs[row], input_bits, dac_bits)
                if cycle < len(input_chunks):
                    chunk = input_chunks[cycle]
                    print(f"Row {row}: Processing chunk 0x{chunk:X}")
                    cycle_inputs.append(chunk)
                    cycle_rows.append(row)
        
        if cycle_inputs:  # Only process if we have inputs for this cycle
            outputs, metrics = system.process_inputs(cycle_inputs, cycle_rows)
            all_cycle_outputs.append((outputs, metrics))
            print(f"Cycle {cycle + 1} outputs: {outputs}")
    
    # Print final results
    print("\nFinal Results:")
    print(f"Configuration:")
    print(f"Array Dimensions: {num_rows}x{num_columns}")
    print(f"Weight Bits: {weight_bits} (max: {max_weight:X}h)")
    print(f"Cell Bits: {cell_bits} (max: {(1 << cell_bits) - 1})")
    print(f"Input Bits: {input_bits} (max: {max_input:X}h)")
    
    print("\nInputs processed:")
    for row, input_val in row_inputs.items():
        print(f"Row {row}: 0x{input_val:04X}")
    
    print("\nCycle-by-cycle outputs:")
    for cycle, (outputs, metrics) in enumerate(all_cycle_outputs):
        print(f"Cycle {cycle + 1}:")
        print(f"  Outputs: {outputs}")
        print(f"  Metrics: {metrics}")