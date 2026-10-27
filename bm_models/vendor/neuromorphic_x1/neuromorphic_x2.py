"""Cycle-stepped behavioral PE model (ideal arithmetic, configurable timing)."""
from dataclasses import dataclass
from typing import ClassVar
import numpy as np
from .FIFO_IOC import FIFO
from .Crossbar_array import CrossbarArray
from .Softmax import softmax


@dataclass(frozen=True)
class ChipTiming:
    # Illustrative architecture parameters, not measured silicon timings.
    program_cells_per_cycle: int = 1
    input_words_per_cycle: int = 1
    output_words_per_cycle: int = 1
    read_words_per_cycle: int = 1
    compute_cycles: ClassVar[int] = 288  # Fixed latency per computation, independent of column count.
    program_cycles_per_cell: int = 50

    def __post_init__(self):
        for value in vars(self).values():
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError("Timing parameters must be positive integers.")


class NeuromorphicX2:
    PROGRAM, COMPUTE, READ = 0x01, 0x02, 0x03

    def __init__(self, iterations=0, input_size=512, output_size=512, timing=None):
        if not isinstance(iterations, int) or iterations < 0:
            raise ValueError("iterations must be a nonnegative integer.")
        if iterations and input_size != output_size:
            raise ValueError("Temporal feedback requires a square array.")
        self.crossbar = CrossbarArray(input_size, output_size)
        self.input_fifo = FIFO(size=input_size)
        self.compute_fifo = FIFO(size=input_size)
        self.output_fifo = FIFO(size=output_size)
        self.timing = timing or ChipTiming()
        self.mode = 'IDLE'
        self.cycle = 0
        self.stage_cycles = {name: 0 for name in ('PROGRAM', 'COMPUTE', 'STORE', 'FEEDBACK', 'READ')}
        # Actual FIFO word transfers, independent of interface words per clock cycle.
        # FEEDBACK counts output-buffer reads consumed internally instead of by the host.
        self.output_buffer_words = {name: 0 for name in ('STORE', 'READ', 'FEEDBACK')}
        self.iteration_count = iterations
        self.current_iteration = 0
        self.history_log = []
        self.initial_output_vector = None
        self.softmax_unit = softmax()
        self.output_valid = False
        self.read_data = np.array([], dtype=float)
        self._staged_weights = None
        self._read_values = []

    @property
    def busy(self):
        return self.mode != 'IDLE'

    @property
    def status(self):
        return dict(state=self.mode, busy=self.busy, cycle=self.cycle,
                    programmed=self.crossbar.programmed, output_valid=self.output_valid,
                    input_words=len(self.input_fifo), output_words=len(self.output_fifo))

    def _require_idle(self):
        if self.busy:
            raise RuntimeError(f"Chip is busy in {self.mode}.")

    def load_weights(self, weights):
        """Stage a host payload; PROGRAM transfers it to the array on clock ticks."""
        self._require_idle()
        matrix = np.asarray(weights, dtype=float)
        if matrix.shape != self.crossbar.weights.shape or not np.isfinite(matrix).all():
            raise ValueError("Expected a finite matrix matching the array dimensions.")
        self._staged_weights = matrix.copy()

    def load_inputs(self, data):
        """Enqueue a complete host vector with no modeled input-transfer latency."""
        self._require_idle()
        vector = np.asarray(data, dtype=float)
        if vector.shape != (self.crossbar.input_size,) or not np.isfinite(vector).all():
            raise ValueError("Expected a finite vector matching the input dimension.")
        if len(self.input_fifo):
            raise RuntimeError("Input FIFO already contains an unconsumed vector.")
        for value in vector:
            self.input_fifo.push(value)

    def command(self, mode, *, read_count=None):
        self._require_idle()
        if mode == self.PROGRAM:
            if self._staged_weights is None:
                raise RuntimeError("Load a weight payload before PROGRAM.")
            if len(self.output_fifo):
                raise RuntimeError("Read pending outputs before reprogramming.")
            self.crossbar.programmed = False
            self._program_offset = 0
            self._program_remaining = self.timing.program_cycles_per_cell
            self.mode = 'PROGRAM'
        elif mode == self.COMPUTE:
            if not self.crossbar.programmed:
                raise RuntimeError("PROGRAM must complete before COMPUTE.")
            if len(self.input_fifo) != self.crossbar.input_size:
                raise RuntimeError("A complete input vector is required.")
            if len(self.output_fifo):
                raise RuntimeError("Read pending outputs before COMPUTE.")
            self.current_iteration = 0
            self.history_log = []
            self.initial_output_vector = None
            self.output_valid = False
            # Input transfer is untimed; computation starts on the next clock tick.
            while not self.input_fifo.is_empty():
                self.compute_fifo.push(self.input_fifo.pop())
            self._start_compute()
        elif mode == self.READ:
            count = len(self.output_fifo) if read_count is None else read_count
            if not isinstance(count, int) or count <= 0 or count > len(self.output_fifo):
                raise ValueError("READ count must fit the available output words.")
            if not self.output_valid:
                raise RuntimeError("Output is not ready.")
            self._read_remaining = count
            self._read_values = []
            self.read_data = np.array([], dtype=float)
            self.mode = 'READ'
        else:
            raise ValueError(f"Unknown command: {mode!r}")

    def _start_compute(self):
        self._compute_remaining = ChipTiming.compute_cycles
        self.mode = 'COMPUTE'

    def run_cycle(self):
        """Advance exactly one clock tick. No command implicitly runs to completion."""
        self.cycle += 1
        if not self.busy:
            return self.status
        self.stage_cycles[self.mode] += 1
        if self.mode == 'PROGRAM':
            self._program_remaining -= 1
            if self._program_remaining > 0:
                return self.status
            end = min(self._program_offset + self.timing.program_cells_per_cycle,
                      self.crossbar.weights.size)
            self.crossbar.weights.flat[self._program_offset:end] = self._staged_weights.flat[self._program_offset:end]
            self._program_offset = end
            self._program_remaining = self.timing.program_cycles_per_cell
            if end == self.crossbar.weights.size:
                self.crossbar.programmed = True
                self._staged_weights = None
                self.mode = 'IDLE'
        elif self.mode == 'FEEDBACK':
            source = self.output_fifo
            for _ in range(min(self.timing.input_words_per_cycle, len(source))):
                self.compute_fifo.push(source.pop())
                self.output_buffer_words['FEEDBACK'] += 1
            if source.is_empty():
                self._start_compute()
        elif self.mode == 'COMPUTE':
            self._compute_remaining -= 1
            if self._compute_remaining == 0:
                vector = np.array([self.compute_fifo.pop() for _ in range(self.crossbar.input_size)])
                self._result = self.crossbar.compute(vector)
                if self.current_iteration:
                    self._result = self.softmax_unit.softmax(self._result)
                    self.history_log.append(self._result.copy())
                else:
                    self.initial_output_vector = self._result.copy()
                self._store_offset = 0
                self.mode = 'STORE'
        elif self.mode == 'STORE':
            end = min(self._store_offset + self.timing.output_words_per_cycle, self.crossbar.output_size)
            for value in self._result[self._store_offset:end]:
                self.output_fifo.push(value)
                self.output_buffer_words['STORE'] += 1
            self._store_offset = end
            if end == self.crossbar.output_size:
                if self.current_iteration < self.iteration_count:
                    self.current_iteration += 1
                    self.mode = 'FEEDBACK'
                else:
                    self.output_valid = True
                    self.mode = 'IDLE'
        elif self.mode == 'READ':
            count = min(self.timing.read_words_per_cycle, self._read_remaining)
            for _ in range(count):
                self._read_values.append(self.output_fifo.pop())
                self.output_buffer_words['READ'] += 1
            self._read_remaining -= count
            if self._read_remaining == 0:
                self.read_data = np.asarray(self._read_values, dtype=float)
                self.output_valid = bool(len(self.output_fifo))
                self.mode = 'IDLE'
        return self.status

    def run_until_idle(self, max_cycles=100_000_000):
        """Blocking host helper; elapsed simulation time still advances by ticks."""
        for _ in range(max_cycles):
            if not self.busy:
                return self.status
            self.run_cycle()
        if self.busy:
            raise TimeoutError("Chip did not complete within max_cycles.")
        return self.status

    def program_weights(self, weights):
        self.load_weights(weights)
        self.command(self.PROGRAM)
        self.run_until_idle()

    def read_outputs(self, count=None):
        self.command(self.READ, read_count=count)
        self.run_until_idle()
        return self.read_data.copy()

