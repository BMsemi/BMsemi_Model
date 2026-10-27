"""Execute mapping512.m through PROGRAM, COMPUTE, and output READ."""
from pathlib import Path
import numpy as np
from Mapper import MaestroMapper
from Scheduler import MaestroScheduler
from neuromorphic_x2 import NeuromorphicX2


def main():
    dims, mapping = MaestroMapper().map_from_file(Path(__file__).with_name("mapping512.m"))
    weights = np.eye(dims.C, dims.K)
    inputs = np.arange(dims.C, dtype=float) % 2
    ops = MaestroScheduler().build(dims, mapping, lambda k0, kN: weights[:, k0:kN])
    chip = NeuromorphicX2(input_size=mapping.array_rows, output_size=mapping.array_cols)
    print(f"Physical array: {mapping.array_rows} x {mapping.array_cols}; "
          f"active rows: {mapping.patch_len}; active columns: {dims.K}")
    for op in ops:
        if op.op == "PROGRAM":
            chip.load_weights(op.info["weight_tile"])
            chip.command(chip.PROGRAM)
            chip.run_until_idle()
            print("PROGRAM complete:", chip.status)
        elif op.op == "COMPUTE_TILE_CRS_FOLD":
            chip.load_inputs(inputs)
            chip.command(chip.COMPUTE)
            chip.run_until_idle()
            print("COMPUTE complete:", chip.status)
            output = chip.read_outputs()
            np.testing.assert_array_equal(output, inputs @ weights)
            print("READ complete:", chip.status)
            print("First 16 outputs:", output[:16])
    print("Stage cycles:", chip.stage_cycles)
    print("PASS: full 512 x 512 mapping matches the reference output.")


if __name__ == '__main__':
    main()

