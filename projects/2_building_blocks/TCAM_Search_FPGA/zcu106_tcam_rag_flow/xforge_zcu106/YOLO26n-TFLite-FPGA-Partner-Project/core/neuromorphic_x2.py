import numpy as np
import time
from .FIFO_IOC import FIFO
from .Crossbar_array import CrossbarArray
from .Softmax import softmax
class NeuromorphicX2:
    """
    Modified Neuromorphic core with:
    Input FIFO -> Compute FIFO -> Crossbar (64×64) -> Output FIFO
    Feedback loop: Output FIFO -> Compute FIFO (for temporal iterations)
    """
    def __init__(self, iterations=3):
        self.input_fifo = FIFO(size=2048)  # Increased for 64-element vectors
        self.compute_fifo = FIFO(size=64)  # Changed from 32 to 64
        self.output_fifo = FIFO(size=64)   # Changed from 32 to 64
        self.crossbar = CrossbarArray()    # 64×64 by default
        self.mode = 'IDLE'
        self.iteration_count = iterations
        self.current_iteration = 0
        self.history_log = []
        self.softmax_unit = softmax()

    def command(self, mode):
        modes = {0x01: 'PROGRAM', 0x02: 'COMPUTE', 0x03: 'READ'}
        self.mode = modes.get(mode, 'IDLE')
        print(f"[CMD] Mode set to: {self.mode}\n")

    def load_inputs(self, data):
        for val in data:
            self.input_fifo.push(val)
        print(f"[LOAD] Loaded {len(data)} elements into Input FIFO.\n")

    def run_cycle(self):
        if self.mode == 'PROGRAM':
            print("[MODE: PROGRAM] Crossbar programmed — no computation.\n")
            return

        if self.mode != 'COMPUTE':
            print("[MODE: IDLE] Waiting for COMPUTE command.\n")
            return

        # Reset per-run logs
        self.history_log = []             # will store ONLY the outputs of temporal iterations
        self.current_iteration = 0
        self.initial_output_vector = None # store the first compute result (not counted as an iteration)

        print(f"[START] Beginning computation: 1 initial compute + {self.iteration_count} temporal iterations.")
        print(f"        Crossbar size: 64×64\n")

        # -----------------------------
        # Initial compute (NOT iteration)
        # -----------------------------
        # Move input -> compute
        while not self.input_fifo.is_empty():
            self.compute_fifo.push(self.input_fifo.pop())

        print("[INITIAL] Input FIFO → Compute FIFO → Crossbar (64×64) → Output FIFO")

        if len(self.compute_fifo.buffer) < 64:
            raise ValueError("Compute FIFO does not have 64 elements for initial computation.")

        init_vector = np.array([self.compute_fifo.pop() for _ in range(64)])
        init_result  = self.crossbar.compute(init_vector)

        #init_result = self.softmax_unit.softmax(init_result)

        # Write initial result to Output FIFO
        self.output_fifo.clear()    
        for v in init_result:
            self.output_fifo.push(v)

        self.initial_output_vector = np.array(init_result, copy=True)

        print("[INITIAL] Stored 64 results in Output FIFO. Iterations start AFTER this point.\n")

        # -----------------------------------
        # Temporal iterations (feedback passes)
        # -----------------------------------
        while self.current_iteration < self.iteration_count:
            iter_idx = self.current_iteration + 1
            print(f"[TEMPORAL] Iteration {iter_idx}/{self.iteration_count}")
            print("            -> Feedback: Output FIFO → Compute FIFO")
            # feedback: output -> compute
            while not self.output_fifo.is_empty():
                self.compute_fifo.push(self.output_fifo.pop())

            # compute
            if len(self.compute_fifo.buffer) < 64:
                raise ValueError("Compute FIFO does not have 64 elements for temporal computation.")

            in_vec = np.array([self.compute_fifo.pop() for _ in range(64)])
            out_vec = self.crossbar.compute(in_vec)

            # apply softmax normalization
            out_vec = self.softmax_unit.softmax(out_vec)

            # write to Output FIFO
            self.output_fifo.clear()
            for v in out_vec:
                self.output_fifo.push(v)

            # log The normalized outptut
            self.history_log.append(np.array(out_vec, copy=True))
            print("            -> Compute FIFO → Crossbar (64×64) → Output FIFO (stored 64 normalized results)\n")

            self.current_iteration += 1

        # After all iterations, Output FIFO already holds final results
        final_output = list(self.output_fifo.buffer)
        print("[FINAL] All iterations complete. Final Output FIFO ready.\n")
        print("[OUTPUT] Final output vector(normalized):")
        print(np.round(final_output, 4), "\n")

