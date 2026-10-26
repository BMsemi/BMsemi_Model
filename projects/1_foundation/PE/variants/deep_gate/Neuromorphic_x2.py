import numpy as np
import time
from FIFO_IOC import FIFO
from Crossbar_array import CrossbarArray


class NeuromorphicX2:
    """
    Neuromorphic PE core with a distinct digital processing stage.

    Data-path stages
    ----------------
    1. Input FIFO      : receives raw activation chunks (INT8/float)
    2. Compute FIFO    : staging buffer → crossbar input
    3. CrossbarArray   : analog PWM MAC → partial INT32 sums (num_neurons wide)
    4. DIGITAL STAGE   :
         Registers     : latch the per-iteration partial MAC result (INT32)
         Accumulators  : running INT32 sum across all temporal iterations
                         (TemporalMap(32,32) C from schema mapping)
    5. Output FIFO     : holds final accumulated INT32 output for READ

    State machine modes
    -------------------
    0x01  PROGRAM  : load weights into crossbar; reset digital stage
    0x02  COMPUTE  : iterate input chunks through crossbar + digital stage
    0x03  READ     : drain Output FIFO

    Schema alignment
    ----------------
    - input_size  = 32   (TemporalMap(32,32) C — one chunk per iteration)
    - output_size = 32   (physical crossbar columns)
    - num_neurons = crossbar.num_neurons = 8  (SpatialMap(8,8) K)
    - iterations  = total_input_channels / input_size  (e.g. 640/32 = 20)
    """

    def __init__(self, iterations=20, input_size=32, output_size=32):
        self.input_size      = int(input_size)
        self.output_size     = int(output_size)
        self.iteration_count = int(iterations)

        # FIFOs
        self.input_fifo   = FIFO(size=2048)
        self.compute_fifo = FIFO(size=input_size)
        self.output_fifo  = FIFO(size=output_size)

        # CrossbarArray (32×32 physical; num_neurons=8 in PWM mode)
        self.crossbar = CrossbarArray(input_size=input_size, output_size=output_size)

        # ------------------------------------------------------------
        # Digital processing state
        # _num_neurons is resolved after program_weights() sets mode;
        # pre-allocate at output_size and resize on PROGRAM.
        # ------------------------------------------------------------
        self._num_neurons  = self.output_size
        self._registers    = np.zeros(self._num_neurons, dtype=np.int32)
        self._accumulators = np.zeros(self._num_neurons, dtype=np.int32)

        self.mode              = 'IDLE'
        self.current_iteration = 0
        self.history_log       = []

        print(f"[INIT] NeuromorphicX2: {input_size}×{output_size} crossbar")
        print(f"       iterations={iterations}  (one {input_size}-element chunk per iteration)")
        print(f"       Digital stage: {self._num_neurons} registers + "
              f"{self._num_neurons} accumulators (resized on PROGRAM)\n")

    # ------------------------------------------------------------------
    # Command / mode control
    # ------------------------------------------------------------------
    def command(self, mode):
        modes = {0x01: 'PROGRAM', 0x02: 'COMPUTE', 0x03: 'READ'}
        self.mode = modes.get(mode, 'IDLE')
        print(f"[CMD] Mode set to: {self.mode}\n")

    # ------------------------------------------------------------------
    # Weight loading  (wraps crossbar + resizes digital stage)
    # ------------------------------------------------------------------
    def program_weights(self, weight_matrix):
        """
        Program the crossbar and resize the digital stage to num_neurons.
        Resets registers and accumulators.
        """
        self.crossbar.program_weights(weight_matrix)

        if self.crossbar.mode == CrossbarArray.MODE_PWM_INT8_ROW_SERIAL:
            self._num_neurons = self.crossbar.num_neurons
        else:
            self._num_neurons = self.output_size

        self._registers    = np.zeros(self._num_neurons, dtype=np.int32)
        self._accumulators = np.zeros(self._num_neurons, dtype=np.int32)

        print(f"[PROGRAM] Digital stage resized: "
              f"{self._num_neurons} registers + {self._num_neurons} accumulators\n")

    # ------------------------------------------------------------------
    # Input loading
    # ------------------------------------------------------------------
    def load_inputs(self, data):
        """Push a flat array of activation values into the Input FIFO."""
        for val in data:
            self.input_fifo.push(val)
        print(f"[LOAD] Loaded {len(data)} elements into Input FIFO.\n")

    # ------------------------------------------------------------------
    # Digital-stage helpers
    # ------------------------------------------------------------------
    def _reset_digital_stage(self):
        """Clear registers and accumulators before a new inference pass."""
        self._registers.fill(0)
        self._accumulators.fill(0)
        print(f"[DIGITAL] {self._num_neurons} registers + "
              f"{self._num_neurons} accumulators cleared\n")

    def _digital_step(self, partial_mac: np.ndarray):
        """
        One digital processing step per temporal iteration:
          1. Latch partial MAC into registers
          2. Add registers into accumulators
        """
        self._registers[:]  = partial_mac.astype(np.int32)
        self._accumulators += self._registers

    # ------------------------------------------------------------------
    # Main execution cycle
    # ------------------------------------------------------------------
    def run_cycle(self):
        """
        Execute a full inference pass over all temporal iterations.

        Flow per iteration
        ------------------
        Input FIFO → pop input_size elements → Compute FIFO
            → Crossbar MAC (analog) → partial INT32 (num_neurons)
            → Registers  (latch)
            → Accumulators (add)

        After all iterations: Accumulators → Output FIFO
        """
        if self.mode == 'PROGRAM':
            print("[MODE: PROGRAM] Crossbar programmed — no computation.\n")
            return

        if self.mode != 'COMPUTE':
            print("[MODE: IDLE] Waiting for COMPUTE command.\n")
            return

        if self.crossbar.mode is None:
            raise RuntimeError("Crossbar weights not programmed. "
                               "Call program_weights() before run_cycle().")

        self.history_log       = []
        self.current_iteration = 0

        # Reset digital stage before new inference pass
        self._reset_digital_stage()

        print(f"[START] {self.iteration_count} temporal iterations "
              f"({self.input_size} activations per iteration)")
        print(f"        Crossbar: {self.input_size}×{self.output_size}  "
              f"num_neurons={self._num_neurons}\n")

        # ------------------------------------------------------------------
        # Temporal iterations  (TemporalMap(32,32) C)
        # ------------------------------------------------------------------
        while self.current_iteration < self.iteration_count:
            iter_idx = self.current_iteration + 1

            # Transfer one chunk: Input FIFO → Compute FIFO
            moved = 0
            while not self.input_fifo.is_empty() and moved < self.input_size:
                self.compute_fifo.push(self.input_fifo.pop())
                moved += 1

            if len(self.compute_fifo.buffer) < self.input_size:
                print(f"[WARN] Iteration {iter_idx}: only "
                      f"{len(self.compute_fifo.buffer)} elements available "
                      f"(expected {self.input_size}). Stopping early.")
                break

            chunk = np.array(
                [self.compute_fifo.pop() for _ in range(self.input_size)],
                dtype=np.int32
            )

            # Analog crossbar MAC
            if self.crossbar.mode == CrossbarArray.MODE_PWM_INT8_ROW_SERIAL:
                partial = self.crossbar._compute_pwm_row_serial_int8(chunk)
            else:
                partial = self.crossbar.compute(chunk)

            # Digital processing stage: Registers → Accumulators
            self._digital_step(partial)

            print(f"[ITER {iter_idx:>3}/{self.iteration_count}] "
                  f"Crossbar MAC → Registers latched → Accumulators updated  "
                  f"| acc_sum={int(self._accumulators.sum())}")

            self.history_log.append({
                'iteration':    iter_idx,
                'registers':    self._registers.copy(),
                'accumulators': self._accumulators.copy(),
            })

            self.current_iteration += 1

        # Flush accumulators → Output FIFO
        self.output_fifo.clear()
        for v in self._accumulators:
            self.output_fifo.push(int(v))

        final = np.array(list(self.output_fifo.buffer), dtype=np.int32)
        print(f"\n[DIGITAL] All {self.current_iteration} iterations complete.")
        print(f"[DIGITAL] Accumulators → Output FIFO  ({self._num_neurons} values)")
        print(f"[OUTPUT]  Final accumulated INT32 output:\n{final}\n")

    # ------------------------------------------------------------------
    # READ mode — drain Output FIFO
    # ------------------------------------------------------------------
    def read_output(self) -> np.ndarray:
        """Return the accumulated output and clear the Output FIFO."""
        if self.mode != 'READ':
            print("[WARN] read_output() called outside READ mode.")
        result = np.array(list(self.output_fifo.buffer), dtype=np.int32)
        self.output_fifo.clear()
        return result

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------
    def get_digital_state(self) -> dict:
        """Return a snapshot of the current digital stage state."""
        return {
            'num_neurons':  self._num_neurons,
            'registers':    self._registers.copy(),
            'accumulators': self._accumulators.copy(),
            'iteration':    self.current_iteration,
        }

