"""Architectural cycle estimates shared with the chip controller."""
from .neuromorphic_x2 import ChipTiming


class latencycalculator:
    def __init__(self, program_latency=50):
        self._positive(program_latency)
        self.program_latency = program_latency  # cycles per serially programmed cell
        self.compute_latency = ChipTiming.compute_cycles  # fixed at 288 cycles

    @staticmethod
    def _positive(value):
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError("Dimensions and latency parameters must be positive integers.")

    def program_cycles(self, rows, columns):
        self._positive(rows)
        self._positive(columns)
        return rows * columns * self.program_latency

    def compute_cycles(self, columns=512):
        self._positive(columns)
        return ChipTiming.compute_cycles

    def chip_timing(self, columns=512):
        self._positive(columns)
        return ChipTiming(program_cells_per_cycle=1,
                          program_cycles_per_cell=self.program_latency)

    def compute_total_latency(self, iteration_count, columns=512):
        """Initial computation plus feedback computations; excludes PROGRAM and FIFO transfers."""
        if isinstance(iteration_count, bool) or not isinstance(iteration_count, int) or iteration_count < 0:
            raise ValueError("iteration_count must be a nonnegative integer.")
        return (1 + iteration_count) * self.compute_cycles(columns)

