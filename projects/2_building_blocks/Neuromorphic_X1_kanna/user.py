# user.py
# Uses the scheduler: PROGRAM once, then NUM_ITERS COMPUTE_TILE ops.
# Manual 512x512 weights; 1x512 input; feedback: x_{k+1} = y_k.

from pathlib import Path
import numpy as np
from neuromorphic_x2 import NeuromorphicX2

# ADD: Maestro scheduler/mapper
from Scheduler import SinglePEScheduler, SchedOp, MaestroScheduler
from Mapper import Dims, SinglePEMapper, MaestroMapper

from energy import EnergyCalculator
from Latency import latencycalculator

# ADD: Neuromorphic multi-PE core
from Multiple_PE_core import MultiPECore, MultiPENeuromorphicCore

# Joules per full 512-word output buffer STORE / READ, not per word or clock cycle.
# Supply hardware estimates here. None reports an unset cost; zero explicitly disables it.
OUTPUT_BUFFER_STORE_ENERGY_J = None
OUTPUT_BUFFER_READ_ENERGY_J = None

import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

# --- Manual 512x512 weights: repeated 5x5 filters: top 0-1, bottom 2-3, left 4-5, right 6-7 ---
W_manual = np.array([
    # Filter block 0: top(+,-), bottom(+,-), left(+,-), right(+,-)
    [0, 1, 1, 0, 0, 1, 1, 0],
    [0, 1, 1, 0, 0, 1, 1, 0],
    [0, 1, 1, 0, 0, 0, 0, 0],
    [0, 1, 1, 0, 1, 0, 0, 1],
    [0, 1, 1, 0, 1, 0, 0, 1],
    [0, 1, 1, 0, 0, 1, 1, 0],
    [0, 1, 1, 0, 0, 1, 1, 0],
    [0, 1, 1, 0, 0, 0, 0, 0],
    [0, 1, 1, 0, 1, 0, 0, 1],
    [0, 1, 1, 0, 1, 0, 0, 1],
    [0, 0, 0, 0, 0, 1, 1, 0],
    [0, 0, 0, 0, 0, 1, 1, 0],
    [0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 1, 0, 0, 1],
    [0, 0, 0, 0, 1, 0, 0, 1],
    [1, 0, 0, 1, 0, 1, 1, 0],
    [1, 0, 0, 1, 0, 1, 1, 0],
    [1, 0, 0, 1, 0, 0, 0, 0],
    [1, 0, 0, 1, 1, 0, 0, 1],
    [1, 0, 0, 1, 1, 0, 0, 1],
    [1, 0, 0, 1, 0, 1, 1, 0],
    [1, 0, 0, 1, 0, 1, 1, 0],
    [1, 0, 0, 1, 0, 0, 0, 0],
    [1, 0, 0, 1, 1, 0, 0, 1],
    [1, 0, 0, 1, 1, 0, 0, 1],
    # Filter block 1: top(+,-), bottom(+,-), left(+,-), right(+,-)
    [0, 1, 1, 0, 0, 1, 1, 0],
    [0, 1, 1, 0, 0, 1, 1, 0],
    [0, 1, 1, 0, 0, 0, 0, 0],
    [0, 1, 1, 0, 1, 0, 0, 1],
    [0, 1, 1, 0, 1, 0, 0, 1],
    [0, 1, 1, 0, 0, 1, 1, 0],
    [0, 1, 1, 0, 0, 1, 1, 0],
    [0, 1, 1, 0, 0, 0, 0, 0],
    [0, 1, 1, 0, 1, 0, 0, 1],
    [0, 1, 1, 0, 1, 0, 0, 1],
    [0, 0, 0, 0, 0, 1, 1, 0],
    [0, 0, 0, 0, 0, 1, 1, 0],
    [0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 1, 0, 0, 1],
    [0, 0, 0, 0, 1, 0, 0, 1],
    [1, 0, 0, 1, 0, 1, 1, 0],
    [1, 0, 0, 1, 0, 1, 1, 0],
    [1, 0, 0, 1, 0, 0, 0, 0],
    [1, 0, 0, 1, 1, 0, 0, 1],
    [1, 0, 0, 1, 1, 0, 0, 1],
    [1, 0, 0, 1, 0, 1, 1, 0],
    [1, 0, 0, 1, 0, 1, 1, 0],
    [1, 0, 0, 1, 0, 0, 0, 0],
    [1, 0, 0, 1, 1, 0, 0, 1],
    [1, 0, 0, 1, 1, 0, 0, 1],
    # Filter block 2: top(+,-), bottom(+,-), left(+,-), right(+,-)
    [0, 1, 1, 0, 0, 1, 1, 0],
    [0, 1, 1, 0, 0, 1, 1, 0],
    [0, 1, 1, 0, 0, 0, 0, 0],
    [0, 1, 1, 0, 1, 0, 0, 1],
    [0, 1, 1, 0, 1, 0, 0, 1],
    [0, 1, 1, 0, 0, 1, 1, 0],
    [0, 1, 1, 0, 0, 1, 1, 0],
    [0, 1, 1, 0, 0, 0, 0, 0],
    [0, 1, 1, 0, 1, 0, 0, 1],
    [0, 1, 1, 0, 1, 0, 0, 1],
    [0, 0, 0, 0, 0, 1, 1, 0],
    [0, 0, 0, 0, 0, 1, 1, 0],
    [0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 1, 0, 0, 1],
    [0, 0, 0, 0, 1, 0, 0, 1],
    [1, 0, 0, 1, 0, 1, 1, 0],
    [1, 0, 0, 1, 0, 1, 1, 0],
    [1, 0, 0, 1, 0, 0, 0, 0],
    [1, 0, 0, 1, 1, 0, 0, 1],
    [1, 0, 0, 1, 1, 0, 0, 1],
    [1, 0, 0, 1, 0, 1, 1, 0],
    [1, 0, 0, 1, 0, 1, 1, 0],
    [1, 0, 0, 1, 0, 0, 0, 0],
    [1, 0, 0, 1, 1, 0, 0, 1],
    [1, 0, 0, 1, 1, 0, 0, 1],
    # Filter block 3: top(+,-), bottom(+,-), left(+,-), right(+,-)
    [0, 1, 1, 0, 0, 1, 1, 0],
    [0, 1, 1, 0, 0, 1, 1, 0],
    [0, 1, 1, 0, 0, 0, 0, 0],
    [0, 1, 1, 0, 1, 0, 0, 1],
    [0, 1, 1, 0, 1, 0, 0, 1],
    [0, 1, 1, 0, 0, 1, 1, 0],
    [0, 1, 1, 0, 0, 1, 1, 0],
    [0, 1, 1, 0, 0, 0, 0, 0],
    [0, 1, 1, 0, 1, 0, 0, 1],
    [0, 1, 1, 0, 1, 0, 0, 1],
    [0, 0, 0, 0, 0, 1, 1, 0],
    [0, 0, 0, 0, 0, 1, 1, 0],
    [0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 1, 0, 0, 1],
    [0, 0, 0, 0, 1, 0, 0, 1],
    [1, 0, 0, 1, 0, 1, 1, 0],
    [1, 0, 0, 1, 0, 1, 1, 0],
    [1, 0, 0, 1, 0, 0, 0, 0],
    [1, 0, 0, 1, 1, 0, 0, 1],
    [1, 0, 0, 1, 1, 0, 0, 1],
    [1, 0, 0, 1, 0, 1, 1, 0],
    [1, 0, 0, 1, 0, 1, 1, 0],
    [1, 0, 0, 1, 0, 0, 0, 0],
    [1, 0, 0, 1, 1, 0, 0, 1],
    [1, 0, 0, 1, 1, 0, 0, 1],
    # Filter block 4: top(+,-), bottom(+,-), left(+,-), right(+,-)
    [0, 1, 1, 0, 0, 1, 1, 0],
    [0, 1, 1, 0, 0, 1, 1, 0],
    [0, 1, 1, 0, 0, 0, 0, 0],
    [0, 1, 1, 0, 1, 0, 0, 1],
    [0, 1, 1, 0, 1, 0, 0, 1],
    [0, 1, 1, 0, 0, 1, 1, 0],
    [0, 1, 1, 0, 0, 1, 1, 0],
    [0, 1, 1, 0, 0, 0, 0, 0],
    [0, 1, 1, 0, 1, 0, 0, 1],
    [0, 1, 1, 0, 1, 0, 0, 1],
    [0, 0, 0, 0, 0, 1, 1, 0],
    [0, 0, 0, 0, 0, 1, 1, 0],
    [0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 1, 0, 0, 1],
    [0, 0, 0, 0, 1, 0, 0, 1],
    [1, 0, 0, 1, 0, 1, 1, 0],
    [1, 0, 0, 1, 0, 1, 1, 0],
    [1, 0, 0, 1, 0, 0, 0, 0],
    [1, 0, 0, 1, 1, 0, 0, 1],
    [1, 0, 0, 1, 1, 0, 0, 1],
    [1, 0, 0, 1, 0, 1, 1, 0],
    [1, 0, 0, 1, 0, 1, 1, 0],
    [1, 0, 0, 1, 0, 0, 0, 0],
    [1, 0, 0, 1, 1, 0, 0, 1],
    [1, 0, 0, 1, 1, 0, 0, 1],
    # Filter block 5: top(+,-), bottom(+,-), left(+,-), right(+,-)
    [0, 1, 1, 0, 0, 1, 1, 0],
    [0, 1, 1, 0, 0, 1, 1, 0],
    [0, 1, 1, 0, 0, 0, 0, 0],
    [0, 1, 1, 0, 1, 0, 0, 1],
    [0, 1, 1, 0, 1, 0, 0, 1],
    [0, 1, 1, 0, 0, 1, 1, 0],
    [0, 1, 1, 0, 0, 1, 1, 0],
    [0, 1, 1, 0, 0, 0, 0, 0],
    [0, 1, 1, 0, 1, 0, 0, 1],
    [0, 1, 1, 0, 1, 0, 0, 1],
    [0, 0, 0, 0, 0, 1, 1, 0],
    [0, 0, 0, 0, 0, 1, 1, 0],
    [0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 1, 0, 0, 1],
    [0, 0, 0, 0, 1, 0, 0, 1],
    [1, 0, 0, 1, 0, 1, 1, 0],
    [1, 0, 0, 1, 0, 1, 1, 0],
    [1, 0, 0, 1, 0, 0, 0, 0],
    [1, 0, 0, 1, 1, 0, 0, 1],
    [1, 0, 0, 1, 1, 0, 0, 1],
    [1, 0, 0, 1, 0, 1, 1, 0],
    [1, 0, 0, 1, 0, 1, 1, 0],
    [1, 0, 0, 1, 0, 0, 0, 0],
    [1, 0, 0, 1, 1, 0, 0, 1],
    [1, 0, 0, 1, 1, 0, 0, 1],
    # Filter block 6: top(+,-), bottom(+,-), left(+,-), right(+,-)
    [0, 1, 1, 0, 0, 1, 1, 0],
    [0, 1, 1, 0, 0, 1, 1, 0],
    [0, 1, 1, 0, 0, 0, 0, 0],
    [0, 1, 1, 0, 1, 0, 0, 1],
    [0, 1, 1, 0, 1, 0, 0, 1],
    [0, 1, 1, 0, 0, 1, 1, 0],
    [0, 1, 1, 0, 0, 1, 1, 0],
    [0, 1, 1, 0, 0, 0, 0, 0],
    [0, 1, 1, 0, 1, 0, 0, 1],
    [0, 1, 1, 0, 1, 0, 0, 1],
    [0, 0, 0, 0, 0, 1, 1, 0],
    [0, 0, 0, 0, 0, 1, 1, 0],
    [0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 1, 0, 0, 1],
    [0, 0, 0, 0, 1, 0, 0, 1],
    [1, 0, 0, 1, 0, 1, 1, 0],
    [1, 0, 0, 1, 0, 1, 1, 0],
    [1, 0, 0, 1, 0, 0, 0, 0],
    [1, 0, 0, 1, 1, 0, 0, 1],
    [1, 0, 0, 1, 1, 0, 0, 1],
    [1, 0, 0, 1, 0, 1, 1, 0],
    [1, 0, 0, 1, 0, 1, 1, 0],
    [1, 0, 0, 1, 0, 0, 0, 0],
    [1, 0, 0, 1, 1, 0, 0, 1],
    [1, 0, 0, 1, 1, 0, 0, 1],
    # Filter block 7: top(+,-), bottom(+,-), left(+,-), right(+,-)
    [0, 1, 1, 0, 0, 1, 1, 0],
    [0, 1, 1, 0, 0, 1, 1, 0],
    [0, 1, 1, 0, 0, 0, 0, 0],
    [0, 1, 1, 0, 1, 0, 0, 1],
    [0, 1, 1, 0, 1, 0, 0, 1],
    [0, 1, 1, 0, 0, 1, 1, 0],
    [0, 1, 1, 0, 0, 1, 1, 0],
    [0, 1, 1, 0, 0, 0, 0, 0],
    [0, 1, 1, 0, 1, 0, 0, 1],
    [0, 1, 1, 0, 1, 0, 0, 1],
    [0, 0, 0, 0, 0, 1, 1, 0],
    [0, 0, 0, 0, 0, 1, 1, 0],
    [0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 1, 0, 0, 1],
    [0, 0, 0, 0, 1, 0, 0, 1],
    [1, 0, 0, 1, 0, 1, 1, 0],
    [1, 0, 0, 1, 0, 1, 1, 0],
    [1, 0, 0, 1, 0, 0, 0, 0],
    [1, 0, 0, 1, 1, 0, 0, 1],
    [1, 0, 0, 1, 1, 0, 0, 1],
    [1, 0, 0, 1, 0, 1, 1, 0],
    [1, 0, 0, 1, 0, 1, 1, 0],
    [1, 0, 0, 1, 0, 0, 0, 0],
    [1, 0, 0, 1, 1, 0, 0, 1],
    [1, 0, 0, 1, 1, 0, 0, 1],
    # Filter block 8: top(+,-), bottom(+,-), left(+,-), right(+,-)
    [0, 1, 1, 0, 0, 1, 1, 0],
    [0, 1, 1, 0, 0, 1, 1, 0],
    [0, 1, 1, 0, 0, 0, 0, 0],
    [0, 1, 1, 0, 1, 0, 0, 1],
    [0, 1, 1, 0, 1, 0, 0, 1],
    [0, 1, 1, 0, 0, 1, 1, 0],
    [0, 1, 1, 0, 0, 1, 1, 0],
    [0, 1, 1, 0, 0, 0, 0, 0],
    [0, 1, 1, 0, 1, 0, 0, 1],
    [0, 1, 1, 0, 1, 0, 0, 1],
    [0, 0, 0, 0, 0, 1, 1, 0],
    [0, 0, 0, 0, 0, 1, 1, 0],
    [0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 1, 0, 0, 1],
    [0, 0, 0, 0, 1, 0, 0, 1],
    [1, 0, 0, 1, 0, 1, 1, 0],
    [1, 0, 0, 1, 0, 1, 1, 0],
    [1, 0, 0, 1, 0, 0, 0, 0],
    [1, 0, 0, 1, 1, 0, 0, 1],
    [1, 0, 0, 1, 1, 0, 0, 1],
    [1, 0, 0, 1, 0, 1, 1, 0],
    [1, 0, 0, 1, 0, 1, 1, 0],
    [1, 0, 0, 1, 0, 0, 0, 0],
    [1, 0, 0, 1, 1, 0, 0, 1],
    [1, 0, 0, 1, 1, 0, 0, 1],
    # Filter block 9: top(+,-), bottom(+,-), left(+,-), right(+,-)
    [0, 1, 1, 0, 0, 1, 1, 0],
    [0, 1, 1, 0, 0, 1, 1, 0],
    [0, 1, 1, 0, 0, 0, 0, 0],
    [0, 1, 1, 0, 1, 0, 0, 1],
    [0, 1, 1, 0, 1, 0, 0, 1],
    [0, 1, 1, 0, 0, 1, 1, 0],
    [0, 1, 1, 0, 0, 1, 1, 0],
    [0, 1, 1, 0, 0, 0, 0, 0],
    [0, 1, 1, 0, 1, 0, 0, 1],
    [0, 1, 1, 0, 1, 0, 0, 1],
    [0, 0, 0, 0, 0, 1, 1, 0],
    [0, 0, 0, 0, 0, 1, 1, 0],
    [0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 1, 0, 0, 1],
    [0, 0, 0, 0, 1, 0, 0, 1],
    [1, 0, 0, 1, 0, 1, 1, 0],
    [1, 0, 0, 1, 0, 1, 1, 0],
    [1, 0, 0, 1, 0, 0, 0, 0],
    [1, 0, 0, 1, 1, 0, 0, 1],
    [1, 0, 0, 1, 1, 0, 0, 1],
    [1, 0, 0, 1, 0, 1, 1, 0],
    [1, 0, 0, 1, 0, 1, 1, 0],
    [1, 0, 0, 1, 0, 0, 0, 0],
    [1, 0, 0, 1, 1, 0, 0, 1],
    [1, 0, 0, 1, 1, 0, 0, 1],
    # Filter block 10: top(+,-), bottom(+,-), left(+,-), right(+,-)
    [0, 1, 1, 0, 0, 1, 1, 0],
    [0, 1, 1, 0, 0, 1, 1, 0],
    [0, 1, 1, 0, 0, 0, 0, 0],
    [0, 1, 1, 0, 1, 0, 0, 1],
    [0, 1, 1, 0, 1, 0, 0, 1],
    [0, 1, 1, 0, 0, 1, 1, 0],
    [0, 1, 1, 0, 0, 1, 1, 0],
    [0, 1, 1, 0, 0, 0, 0, 0],
    [0, 1, 1, 0, 1, 0, 0, 1],
    [0, 1, 1, 0, 1, 0, 0, 1],
    [0, 0, 0, 0, 0, 1, 1, 0],
    [0, 0, 0, 0, 0, 1, 1, 0],
    [0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 1, 0, 0, 1],
    [0, 0, 0, 0, 1, 0, 0, 1],
    [1, 0, 0, 1, 0, 1, 1, 0],
    [1, 0, 0, 1, 0, 1, 1, 0],
    [1, 0, 0, 1, 0, 0, 0, 0],
    [1, 0, 0, 1, 1, 0, 0, 1],
    [1, 0, 0, 1, 1, 0, 0, 1],
    [1, 0, 0, 1, 0, 1, 1, 0],
    [1, 0, 0, 1, 0, 1, 1, 0],
    [1, 0, 0, 1, 0, 0, 0, 0],
    [1, 0, 0, 1, 1, 0, 0, 1],
    [1, 0, 0, 1, 1, 0, 0, 1],
    # Filter block 11: top(+,-), bottom(+,-), left(+,-), right(+,-)
    [0, 1, 1, 0, 0, 1, 1, 0],
    [0, 1, 1, 0, 0, 1, 1, 0],
    [0, 1, 1, 0, 0, 0, 0, 0],
    [0, 1, 1, 0, 1, 0, 0, 1],
    [0, 1, 1, 0, 1, 0, 0, 1],
    [0, 1, 1, 0, 0, 1, 1, 0],
    [0, 1, 1, 0, 0, 1, 1, 0],
    [0, 1, 1, 0, 0, 0, 0, 0],
    [0, 1, 1, 0, 1, 0, 0, 1],
    [0, 1, 1, 0, 1, 0, 0, 1],
    [0, 0, 0, 0, 0, 1, 1, 0],
    [0, 0, 0, 0, 0, 1, 1, 0],
    [0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 1, 0, 0, 1],
    [0, 0, 0, 0, 1, 0, 0, 1],
    [1, 0, 0, 1, 0, 1, 1, 0],
    [1, 0, 0, 1, 0, 1, 1, 0],
    [1, 0, 0, 1, 0, 0, 0, 0],
    [1, 0, 0, 1, 1, 0, 0, 1],
    [1, 0, 0, 1, 1, 0, 0, 1],
    [1, 0, 0, 1, 0, 1, 1, 0],
    [1, 0, 0, 1, 0, 1, 1, 0],
    [1, 0, 0, 1, 0, 0, 0, 0],
    [1, 0, 0, 1, 1, 0, 0, 1],
    [1, 0, 0, 1, 1, 0, 0, 1],
    # Filter block 12: top(+,-), bottom(+,-), left(+,-), right(+,-)
    [0, 1, 1, 0, 0, 1, 1, 0],
    [0, 1, 1, 0, 0, 1, 1, 0],
    [0, 1, 1, 0, 0, 0, 0, 0],
    [0, 1, 1, 0, 1, 0, 0, 1],
    [0, 1, 1, 0, 1, 0, 0, 1],
    [0, 1, 1, 0, 0, 1, 1, 0],
    [0, 1, 1, 0, 0, 1, 1, 0],
    [0, 1, 1, 0, 0, 0, 0, 0],
    [0, 1, 1, 0, 1, 0, 0, 1],
    [0, 1, 1, 0, 1, 0, 0, 1],
    [0, 0, 0, 0, 0, 1, 1, 0],
    [0, 0, 0, 0, 0, 1, 1, 0],
    [0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 1, 0, 0, 1],
    [0, 0, 0, 0, 1, 0, 0, 1],
    [1, 0, 0, 1, 0, 1, 1, 0],
    [1, 0, 0, 1, 0, 1, 1, 0],
    [1, 0, 0, 1, 0, 0, 0, 0],
    [1, 0, 0, 1, 1, 0, 0, 1],
    [1, 0, 0, 1, 1, 0, 0, 1],
    [1, 0, 0, 1, 0, 1, 1, 0],
    [1, 0, 0, 1, 0, 1, 1, 0],
    [1, 0, 0, 1, 0, 0, 0, 0],
    [1, 0, 0, 1, 1, 0, 0, 1],
    [1, 0, 0, 1, 1, 0, 0, 1],
    # Filter block 13: top(+,-), bottom(+,-), left(+,-), right(+,-)
    [0, 1, 1, 0, 0, 1, 1, 0],
    [0, 1, 1, 0, 0, 1, 1, 0],
    [0, 1, 1, 0, 0, 0, 0, 0],
    [0, 1, 1, 0, 1, 0, 0, 1],
    [0, 1, 1, 0, 1, 0, 0, 1],
    [0, 1, 1, 0, 0, 1, 1, 0],
    [0, 1, 1, 0, 0, 1, 1, 0],
    [0, 1, 1, 0, 0, 0, 0, 0],
    [0, 1, 1, 0, 1, 0, 0, 1],
    [0, 1, 1, 0, 1, 0, 0, 1],
    [0, 0, 0, 0, 0, 1, 1, 0],
    [0, 0, 0, 0, 0, 1, 1, 0],
    [0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 1, 0, 0, 1],
    [0, 0, 0, 0, 1, 0, 0, 1],
    [1, 0, 0, 1, 0, 1, 1, 0],
    [1, 0, 0, 1, 0, 1, 1, 0],
    [1, 0, 0, 1, 0, 0, 0, 0],
    [1, 0, 0, 1, 1, 0, 0, 1],
    [1, 0, 0, 1, 1, 0, 0, 1],
    [1, 0, 0, 1, 0, 1, 1, 0],
    [1, 0, 0, 1, 0, 1, 1, 0],
    [1, 0, 0, 1, 0, 0, 0, 0],
    [1, 0, 0, 1, 1, 0, 0, 1],
    [1, 0, 0, 1, 1, 0, 0, 1],
    # Filter block 14: top(+,-), bottom(+,-), left(+,-), right(+,-)
    [0, 1, 1, 0, 0, 1, 1, 0],
    [0, 1, 1, 0, 0, 1, 1, 0],
    [0, 1, 1, 0, 0, 0, 0, 0],
    [0, 1, 1, 0, 1, 0, 0, 1],
    [0, 1, 1, 0, 1, 0, 0, 1],
    [0, 1, 1, 0, 0, 1, 1, 0],
    [0, 1, 1, 0, 0, 1, 1, 0],
    [0, 1, 1, 0, 0, 0, 0, 0],
    [0, 1, 1, 0, 1, 0, 0, 1],
    [0, 1, 1, 0, 1, 0, 0, 1],
    [0, 0, 0, 0, 0, 1, 1, 0],
    [0, 0, 0, 0, 0, 1, 1, 0],
    [0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 1, 0, 0, 1],
    [0, 0, 0, 0, 1, 0, 0, 1],
    [1, 0, 0, 1, 0, 1, 1, 0],
    [1, 0, 0, 1, 0, 1, 1, 0],
    [1, 0, 0, 1, 0, 0, 0, 0],
    [1, 0, 0, 1, 1, 0, 0, 1],
    [1, 0, 0, 1, 1, 0, 0, 1],
    [1, 0, 0, 1, 0, 1, 1, 0],
    [1, 0, 0, 1, 0, 1, 1, 0],
    [1, 0, 0, 1, 0, 0, 0, 0],
    [1, 0, 0, 1, 1, 0, 0, 1],
    [1, 0, 0, 1, 1, 0, 0, 1],
    # Filter block 15: top(+,-), bottom(+,-), left(+,-), right(+,-)
    [0, 1, 1, 0, 0, 1, 1, 0],
    [0, 1, 1, 0, 0, 1, 1, 0],
    [0, 1, 1, 0, 0, 0, 0, 0],
    [0, 1, 1, 0, 1, 0, 0, 1],
    [0, 1, 1, 0, 1, 0, 0, 1],
    [0, 1, 1, 0, 0, 1, 1, 0],
    [0, 1, 1, 0, 0, 1, 1, 0],
    [0, 1, 1, 0, 0, 0, 0, 0],
    [0, 1, 1, 0, 1, 0, 0, 1],
    [0, 1, 1, 0, 1, 0, 0, 1],
    [0, 0, 0, 0, 0, 1, 1, 0],
    [0, 0, 0, 0, 0, 1, 1, 0],
    [0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 1, 0, 0, 1],
    [0, 0, 0, 0, 1, 0, 0, 1],
    [1, 0, 0, 1, 0, 1, 1, 0],
    [1, 0, 0, 1, 0, 1, 1, 0],
    [1, 0, 0, 1, 0, 0, 0, 0],
    [1, 0, 0, 1, 1, 0, 0, 1],
    [1, 0, 0, 1, 1, 0, 0, 1],
    [1, 0, 0, 1, 0, 1, 1, 0],
    [1, 0, 0, 1, 0, 1, 1, 0],
    [1, 0, 0, 1, 0, 0, 0, 0],
    [1, 0, 0, 1, 1, 0, 0, 1],
    [1, 0, 0, 1, 1, 0, 0, 1],
    # Filter block 16: top(+,-), bottom(+,-), left(+,-), right(+,-)
    [0, 1, 1, 0, 0, 1, 1, 0],
    [0, 1, 1, 0, 0, 1, 1, 0],
    [0, 1, 1, 0, 0, 0, 0, 0],
    [0, 1, 1, 0, 1, 0, 0, 1],
    [0, 1, 1, 0, 1, 0, 0, 1],
    [0, 1, 1, 0, 0, 1, 1, 0],
    [0, 1, 1, 0, 0, 1, 1, 0],
    [0, 1, 1, 0, 0, 0, 0, 0],
    [0, 1, 1, 0, 1, 0, 0, 1],
    [0, 1, 1, 0, 1, 0, 0, 1],
    [0, 0, 0, 0, 0, 1, 1, 0],
    [0, 0, 0, 0, 0, 1, 1, 0],
    [0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 1, 0, 0, 1],
    [0, 0, 0, 0, 1, 0, 0, 1],
    [1, 0, 0, 1, 0, 1, 1, 0],
    [1, 0, 0, 1, 0, 1, 1, 0],
    [1, 0, 0, 1, 0, 0, 0, 0],
    [1, 0, 0, 1, 1, 0, 0, 1],
    [1, 0, 0, 1, 1, 0, 0, 1],
    [1, 0, 0, 1, 0, 1, 1, 0],
    [1, 0, 0, 1, 0, 1, 1, 0],
    [1, 0, 0, 1, 0, 0, 0, 0],
    [1, 0, 0, 1, 1, 0, 0, 1],
    [1, 0, 0, 1, 1, 0, 0, 1],
    # Filter block 17: top(+,-), bottom(+,-), left(+,-), right(+,-)
    [0, 1, 1, 0, 0, 1, 1, 0],
    [0, 1, 1, 0, 0, 1, 1, 0],
    [0, 1, 1, 0, 0, 0, 0, 0],
    [0, 1, 1, 0, 1, 0, 0, 1],
    [0, 1, 1, 0, 1, 0, 0, 1],
    [0, 1, 1, 0, 0, 1, 1, 0],
    [0, 1, 1, 0, 0, 1, 1, 0],
    [0, 1, 1, 0, 0, 0, 0, 0],
    [0, 1, 1, 0, 1, 0, 0, 1],
    [0, 1, 1, 0, 1, 0, 0, 1],
    [0, 0, 0, 0, 0, 1, 1, 0],
    [0, 0, 0, 0, 0, 1, 1, 0],
    [0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 1, 0, 0, 1],
    [0, 0, 0, 0, 1, 0, 0, 1],
    [1, 0, 0, 1, 0, 1, 1, 0],
    [1, 0, 0, 1, 0, 1, 1, 0],
    [1, 0, 0, 1, 0, 0, 0, 0],
    [1, 0, 0, 1, 1, 0, 0, 1],
    [1, 0, 0, 1, 1, 0, 0, 1],
    [1, 0, 0, 1, 0, 1, 1, 0],
    [1, 0, 0, 1, 0, 1, 1, 0],
    [1, 0, 0, 1, 0, 0, 0, 0],
    [1, 0, 0, 1, 1, 0, 0, 1],
    [1, 0, 0, 1, 1, 0, 0, 1],
    # Filter block 18: top(+,-), bottom(+,-), left(+,-), right(+,-)
    [0, 1, 1, 0, 0, 1, 1, 0],
    [0, 1, 1, 0, 0, 1, 1, 0],
    [0, 1, 1, 0, 0, 0, 0, 0],
    [0, 1, 1, 0, 1, 0, 0, 1],
    [0, 1, 1, 0, 1, 0, 0, 1],
    [0, 1, 1, 0, 0, 1, 1, 0],
    [0, 1, 1, 0, 0, 1, 1, 0],
    [0, 1, 1, 0, 0, 0, 0, 0],
    [0, 1, 1, 0, 1, 0, 0, 1],
    [0, 1, 1, 0, 1, 0, 0, 1],
    [0, 0, 0, 0, 0, 1, 1, 0],
    [0, 0, 0, 0, 0, 1, 1, 0],
    [0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 1, 0, 0, 1],
    [0, 0, 0, 0, 1, 0, 0, 1],
    [1, 0, 0, 1, 0, 1, 1, 0],
    [1, 0, 0, 1, 0, 1, 1, 0],
    [1, 0, 0, 1, 0, 0, 0, 0],
    [1, 0, 0, 1, 1, 0, 0, 1],
    [1, 0, 0, 1, 1, 0, 0, 1],
    [1, 0, 0, 1, 0, 1, 1, 0],
    [1, 0, 0, 1, 0, 1, 1, 0],
    [1, 0, 0, 1, 0, 0, 0, 0],
    [1, 0, 0, 1, 1, 0, 0, 1],
    [1, 0, 0, 1, 1, 0, 0, 1],
    # Filter block 19: top(+,-), bottom(+,-), left(+,-), right(+,-)
    [0, 1, 1, 0, 0, 1, 1, 0],
    [0, 1, 1, 0, 0, 1, 1, 0],
    [0, 1, 1, 0, 0, 0, 0, 0],
    [0, 1, 1, 0, 1, 0, 0, 1],
    [0, 1, 1, 0, 1, 0, 0, 1],
    [0, 1, 1, 0, 0, 1, 1, 0],
    [0, 1, 1, 0, 0, 1, 1, 0],
    [0, 1, 1, 0, 0, 0, 0, 0],
    [0, 1, 1, 0, 1, 0, 0, 1],
    [0, 1, 1, 0, 1, 0, 0, 1],
    [0, 0, 0, 0, 0, 1, 1, 0],
    [0, 0, 0, 0, 0, 1, 1, 0],
    [0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 1, 0, 0, 1],
    [0, 0, 0, 0, 1, 0, 0, 1],
    [1, 0, 0, 1, 0, 1, 1, 0],
    [1, 0, 0, 1, 0, 1, 1, 0],
    [1, 0, 0, 1, 0, 0, 0, 0],
    [1, 0, 0, 1, 1, 0, 0, 1],
    [1, 0, 0, 1, 1, 0, 0, 1],
    [1, 0, 0, 1, 0, 1, 1, 0],
    [1, 0, 0, 1, 0, 1, 1, 0],
    [1, 0, 0, 1, 0, 0, 0, 0],
    [1, 0, 0, 1, 1, 0, 0, 1],
    [1, 0, 0, 1, 1, 0, 0, 1],
    # Remaining 12 rows are unused.
    [0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0],
], dtype=int)

# Keep the eight manual filter columns; the other 504 physical columns are zero.
W_manual = np.pad(W_manual, ((0, 512 - W_manual.shape[0]), (0, 512 - W_manual.shape[1])))
FILTER_NAMES = ("top", "bottom", "left", "right")
assert W_manual.shape == (512, 512), f"Expected (512,512) weights, got {W_manual.shape}"


x_manual = np.array([1,0,1,0,1,0,1,0,0,0,1,1,0,0,1,0,
                    1,1,0,0,1,0,0,1,0,1,0,1,1,0,0,0], dtype=int)

x_manual = np.pad(x_manual, (0, 512 - x_manual.size))

# MNIST image stored as a matrix; mapping file uses (Y,X) = (rows,cols)
img5_bin = None   # 28x28 binary matrix (0/1)
img_Y = 28        # rows
img_X = 28        # cols
img_label = None  # should be 5

try:
    # Keep dataset loading independent of any classifier implementation.
    dataset_path = Path(__file__).with_name("data") / "mnist.npz"
    if not dataset_path.exists():
        from urllib.request import urlopen
        dataset_path.parent.mkdir(exist_ok=True)
        with urlopen("https://storage.googleapis.com/tensorflow/tf-keras-datasets/mnist.npz", timeout=120) as response:
            payload = response.read()
        temporary_path = dataset_path.with_suffix(".tmp")
        temporary_path.write_bytes(payload)
        temporary_path.replace(dataset_path)
    with np.load(dataset_path, allow_pickle=False) as dataset:
        test_images = dataset["x_test"]
        test_labels = dataset["y_test"]
    idx = int(np.flatnonzero(test_labels == 5)[0])
    img5_bin = (test_images[idx] > 127).astype(int)
    img_label = int(test_labels[idx])
    img_Y, img_X = img5_bin.shape
    print(f"[MNIST] Loaded held-out test image {idx}; reference label {img_label}.")
except Exception as e:
    print(f"[MNIST] Failed to load test image: {e}")

def _extract_patch_same_2d(img2d: np.ndarray, oy: int, ox: int, R: int, S: int) -> np.ndarray:
    pad_y = R // 2
    pad_x = S // 2
    padded = np.pad(img2d, ((pad_y, pad_y), (pad_x, pad_x)), mode="constant", constant_values=0)
    yy = oy + pad_y
    xx = ox + pad_x
    patch = padded[yy - pad_y : yy - pad_y + R, xx - pad_x : xx - pad_x + S]
    return patch.reshape(-1).astype(float)

def _write_pe_outputs(path: str, pe_ids: list[int], outputs_getter):
    """
    outputs_getter(pe_id) -> 1D array-like of length 512
    """
    with open(path, "w", encoding="utf-8") as f:
        f.write(f"=== Used PE Outputs (count={len(pe_ids)}) (each line: 512 values) ===\n")
        for pe in pe_ids:
            out = np.asarray(outputs_getter(pe), dtype=float).reshape(-1)
            line = " ".join(f"{v:.6f}" for v in out[:512])
            f.write(f"PE {pe}: {line}\n")

def _write_pe_outputs_with_coords(path: str, pe_ids: list[int], outputs_getter, coords_getter):
    """
    outputs_getter(pe_id) -> 1D array-like of length 512
    coords_getter(pe_id) -> tuple(y, x)
    """
    with open(path, "w", encoding="utf-8") as f:
        f.write(f"=== PE Outputs with coords (count={len(pe_ids)}) (each line: 512 values) ===\n")
        for pe in pe_ids:
            out = np.asarray(outputs_getter(pe), dtype=float).reshape(-1)
            yx = coords_getter(pe)
            if yx is None:
                y, x = -1, -1
            else:
                y, x = yx
            line = " ".join(f"{v:.6f}" for v in out[:512])
            f.write(f"PE {pe} (y={y}, x={x}): {line}\n")



# =====================================================
# Main
# =====================================================
def main():
    """
    Run according to mapping1.m:
      - Mapper reads Dimensions/Stride/Dataflow from mapping file
      - Scheduler traverses patches temporally on one PE
      - Each PE is a NeuromorphicX2 (via MultiPENeuromorphicCore)
      - Energy/latency are computed from the scheduled ops
    """

    # ----------------------------
    # 0) Preconditions
    # ----------------------------
    mapping_file = Path(__file__).with_name("mapping1.m")

    if img5_bin is None:
        print("[ERROR] MNIST digit 5 image not loaded (img5_bin is None).")
        print("        Check dataset download access, or provide the image manually.")
        return

    # Validate the saved manual array without changing its weights.
    W2 = np.asarray(W_manual)
    if W2.ndim == 3 and W2.shape[0] == 1:
        W2 = np.squeeze(W2, axis=0)
    if W2.shape != (512, 512):
        raise ValueError(f"W_manual must be (512,512). Got {W2.shape}.")

    # ----------------------------
    # 1) Mapping-file driven dims + mapping
    # ----------------------------
    mm = MaestroMapper()
    mdims, mmapping = mm.map_from_file(mapping_file)

    print("\n=== MAESTRO MAPPING (from mapping1.m) ===")
    print(f"Dims: K={mdims.K}, C={mdims.C}, R={mdims.R}, S={mdims.S}, Y={mdims.Y}, X={mdims.X}")
    print(f"Stride: stride_y={mdims.stride_y}, stride_x={mdims.stride_x}")
    print(f"Physical array: {mmapping.array_rows} x {mmapping.array_cols}")
    print(f"XY_size={mmapping.xy_size}")
    print(f"Single-PE temporal window traversal: PEs={mmapping.num_pes_needed}")
    print(f"patch_len(C*R*S)={mmapping.patch_len}, fold_steps={mmapping.fold_steps}, fold_sizes={mmapping.fold_sizes}")
    print("---------------------------------------------------------------\n")

    # This script currently assumes single-channel MNIST
    if mdims.C != 1:
        raise ValueError("This user.py currently supports only C=1 (single-channel) input.")

    # ----------------------------
    # 2) Weight provider (shape: patch_len x Kchunk)
    # ----------------------------
    patch_len = int(mmapping.patch_len)
    K = int(mdims.K)
    if patch_len > 512 or K != 8:
        raise ValueError("This run requires patch_len<=512 and K=8 for the 512x512 crossbar.")

    # Column-differential strategy uses two physical output columns (0 and 1).
    if K < 2:
        raise ValueError(f"Column-differential needs K>=2, got K={K}.")

    if (mmapping.array_rows, mmapping.array_cols) != W2.shape:
        raise ValueError("Mapping Hardware dimensions must match W_manual.")
    if not mmapping.single_pe_windows:
        raise ValueError("This manual repeated-filter run requires SpatialMap K and temporal X/Y.")

    def maestro_weight_provider(k0: int, kN: int) -> np.ndarray:
        # Program all manually supplied row blocks once, without slicing to 25 rows.
        # K describes active filter columns, but PROGRAM covers the full physical array.
        return W2

    # ----------------------------
    # 3) Build schedule per mapping1.m
    # ----------------------------
    ms = MaestroScheduler()
    mops = ms.build(mdims, mmapping, maestro_weight_provider, schedule_crs_temporally=True)

    program_ops = [op for op in mops if op.op == "PROGRAM"]
    compute_ops = [op for op in mops if op.op == "COMPUTE_TILE_CRS_FOLD"]

    used_pe_ids = sorted({int(op.info["pe_id"]) for op in compute_ops})

    NUM_PES = int(mmapping.num_pes_needed)
    print(f"[SCHEDULE] PROGRAM ops: {len(program_ops)}")
    print(f"[SCHEDULE] COMPUTE_TILE_CRS_FOLD ops: {len(compute_ops)}")
    print(f"[SCHEDULE] PEs used: {NUM_PES}\n")
    print(f"[SCHEDULE] PEs actually used: {len(used_pe_ids)} -> {used_pe_ids}\n")
    # ----------------------------
    # 4) Multi-PE core where each PE is NeuromorphicX2
    # ----------------------------
    latency_calc = latencycalculator()
    mpe = MultiPENeuromorphicCore(num_pes=NUM_PES, input_size=512, output_size=512,
                                 iterations=0, timing=latency_calc.chip_timing(512))
    print(f"[TIMING] COMPUTE: {latency_calc.compute_cycles(512)} cycles per patch (fixed)")

    # ----------------------------
    # 5) Execute ops + accumulate outputs
    # ----------------------------
    output_y = (mdims.Y + mdims.stride_y - 1) // mdims.stride_y
    output_x = (mdims.X + mdims.stride_x - 1) // mdims.stride_x
    out = np.zeros((output_y, output_x, len(FILTER_NAMES)), dtype=float)

    energy_calc = EnergyCalculator(
        output_buffer_store_energy_per_buffer=OUTPUT_BUFFER_STORE_ENERGY_J,
        output_buffer_read_energy_per_buffer=OUTPUT_BUFFER_READ_ENERGY_J,
    )

    compute_energy_total = 0.0
    num_compute_ops = 0
    num_program_ops = 0

     # FIX: count only for used PEs
    pe_compute_counts = {pe: 0 for pe in used_pe_ids}

    print("=== EXECUTION (PROGRAM + COMPUTE as per mapping1.m) ===")

    # Track first-seen output per PE and its (y,x) for verification/debug
    first_output_per_pe: dict[int, np.ndarray] = {}
    first_xy_per_pe: dict[int, tuple[int, int]] = {}
    # Track first input vector per PE used for the compute (length 512)
    first_vec_per_pe: dict[int, np.ndarray] = {}
    patch_read_log = []
    for op in mops:
        if op.op == "PROGRAM":
            mpe.program_weights_all(op.info["weight_tile"])
            num_program_ops += 1
            continue

        if op.op != "COMPUTE_TILE_CRS_FOLD":
            continue

        pe_id = int(op.info["pe_id"])
        oy = int(op.info["y"])
        ox = int(op.info["x"])
        k0 = int(op.info["k0"])
        kN = int(op.info["kN"])
        crs0 = int(op.info["crs0"])
        crsN = int(op.info["crsN"])

        # SAME-padded patch (length R*S, and since C=1, CRS == RS here)
        patch = _extract_patch_same_2d(img5_bin, op.info["input_y"], op.info["input_x"], mdims.R, mdims.S)

        # Temporal CRS fold: only drive [crs0:crsN] rows this op
        vec512 = np.zeros(512, dtype=float)
        # Only one filter block is driven at a time, so patches do not sum together.
        row_start = int(op.info["row_start"])
        vec512[row_start:row_start + patch_len] = patch

        # Compute on this PE (NeuromorphicX2 micro-architecture)
        out_full = mpe.compute_pe(pe_id, vec512)  # COMPUTE then READ: 512 physical outputs; columns 8..511 are zero
        responses = out_full[0:8:2] - out_full[1:8:2]
        out[oy, ox, :] = responses
        patch_read_log.append((oy, ox, row_start, *out_full, *responses))

        # Energy accounting per scheduled compute op
        compute_energy_total += energy_calc.compute_phase_energy(
            vec512, mpe.pes[pe_id].crossbar.weights
        )

        num_compute_ops += 1
        if pe_id in pe_compute_counts:
            pe_compute_counts[pe_id] += 1

        # Record first tile/output for this PE (matches the baseline verification behavior)
        if pe_id not in first_output_per_pe:
            first_output_per_pe[pe_id] = out_full.copy()
            first_xy_per_pe[pe_id] = (oy, ox)
            first_vec_per_pe[pe_id] = vec512.copy()

    print(f"[DONE] PROGRAM ops executed: {num_program_ops}")
    print(f"[DONE] COMPUTE ops executed: {num_compute_ops}\n")

    output_dir = Path(__file__).with_name("outputs")
    output_dir.mkdir(exist_ok=True)
    # Retain the original filename for the top-edge map.
    np.savetxt(output_dir / "feature_map.txt", out[:, :, 0], fmt="%.6f")
    for filter_id, name in enumerate(FILTER_NAMES):
        np.savetxt(output_dir / f"feature_map_{name}.txt", out[:, :, filter_id], fmt="%.6f")
    read_header = "y,x,row_start," + ",".join(f"column_{i}" for i in range(512))
    read_header += "," + ",".join(f"{name}_response" for name in FILTER_NAMES)
    np.savetxt(output_dir / "patch_output_reads.csv", patch_read_log, delimiter=",",
               header=read_header, comments="", fmt="%.6f")
    print(f"[SAVE] Feature map: {output_dir / 'feature_map.txt'}")
    print(f"[SAVE] Every output-buffer read: {output_dir / 'patch_output_reads.csv'}")

    # Dump the host results returned by output-buffer reads.
    pe_out_path = output_dir / "pe_outputs_used.txt"
    _write_pe_outputs(pe_out_path, used_pe_ids, lambda pe: mpe.get_output(pe))
    print(f"[SAVE] Wrote used-PE outputs only: {pe_out_path}")

    # Also save the first outputs per PE (with their (y,x) coords)
    pe_first_path = output_dir / "pe_outputs_first.txt"
    _write_pe_outputs_with_coords(
        pe_first_path,
        used_pe_ids,
        lambda pe: first_output_per_pe.get(pe, np.zeros(512)),
        lambda pe: first_xy_per_pe.get(pe, None),
    )
    print(f"[SAVE] Wrote first-PE outputs with coords: {pe_first_path}")

    # Detailed column-differential dot-product breakdown per PE (for the first nonzero patch it processed)
    pe_firstcol_details = output_dir / "pe_firstcol_details.txt"
    with open(pe_firstcol_details, "w", encoding="utf-8") as f:
        f.write("=== Column-Differential Dot-Product Details (first tile per PE) ===\n")
        f.write("Format: For each PE, show (y,x), input vec512 nonzeros, weights[:,0]/weights[:,1], contributions, and (sum0-sum1).\n\n")
        for pe in used_pe_ids:
            yx = first_xy_per_pe.get(pe)
            vec = first_vec_per_pe.get(pe)
            if yx is None or vec is None:
                continue
            y, x = yx
            W = mpe.pes[pe].crossbar.weights
            wcol0 = np.asarray(W)[:, 0].astype(float)
            wcol1 = np.asarray(W)[:, 1].astype(float)

            # SAME-padded 5x5 patch (for clarity)
            patch = _extract_patch_same_2d(img5_bin, int(y) * mdims.stride_y, int(x) * mdims.stride_x, mdims.R, mdims.S)

            # Element-wise contributions for differential columns
            contrib0 = vec * wcol0
            contrib1 = vec * wcol1
            sum0 = float(np.dot(vec, wcol0))
            sum1 = float(np.dot(vec, wcol1))
            diff = sum0 - sum1

            f.write(f"PE {pe} (y={y}, x={x})\n")
            f.write("Input vec512 (indices:val where nonzero): ")
            nz = [(i, float(vec[i])) for i in range(512) if vec[i] != 0]
            f.write(" ".join([f"{i}:{v:.0f}" for i, v in nz]) + "\n")

            f.write("Weights[:,0] (diff col0): ")
            f.write(" ".join(f"{float(v):.0f}" for v in wcol0) + "\n")

            f.write("Weights[:,1] (diff col1): ")
            f.write(" ".join(f"{float(v):.0f}" for v in wcol1) + "\n")

            f.write("Contrib col0 (i:vec[i]*w[i,0] where nonzero): ")
            nzc0 = [(i, float(contrib0[i])) for i in range(512) if contrib0[i] != 0]
            if nzc0:
                f.write(" ".join([f"{i}:{v:.0f}" for i, v in nzc0]) + "\n")
            else:
                f.write("(none)\n")

            f.write("Contrib col1 (i:vec[i]*w[i,1] where nonzero): ")
            nzc1 = [(i, float(contrib1[i])) for i in range(512) if contrib1[i] != 0]
            if nzc1:
                f.write(" ".join([f"{i}:{v:.0f}" for i, v in nzc1]) + "\n")
            else:
                f.write("(none)\n")

            f.write(f"Dot sum col0: {sum0:.0f}\n")
            f.write(f"Dot sum col1: {sum1:.0f}\n")
            f.write(f"Differential (col0-col1): {diff:.0f}\n")
            # Show 5x5 patch matrix for reference
            f.write("Patch 5x5 (SAME at this (y,x)):\n")
            R = int(mdims.R)
            S = int(mdims.S)
            for r in range(R):
                row = patch[r*S:(r+1)*S]
                f.write(" ".join(str(int(v)) for v in row) + "\n")
            f.write("\n")
    print(f"[SAVE] Wrote first-column dot-product details: {pe_firstcol_details}")

    # # ----------------------------
    # # 6) Save output (K=1 convenience)
    # # ----------------------------
    # if mdims.K == 1:
    #     out2d = out[:, :, 0]
    #     out_path = r"c:\Users\KannaSridhar\Desktop\Behavioral_py\maestro_out_28x28.txt"
    #     np.savetxt(out_path, out2d, fmt="%.6f", delimiter=" ")
    #     print(f"[SAVE] Output saved: {out_path}\n")

    # ----------------------------
    # 7) Energy summary (mapping-driven scaling)
    # ----------------------------
    # Array phases include array energy + decoder/driver energy + ADC energy.
    # PROGRAM: each scheduled program per PE. Array READ estimate: once per PE.
    program_energy_total = num_program_ops * sum(
        energy_calc.program_phase_energy(pe.crossbar.weights) for pe in mpe.pes
    )
    array_read_energy_total = sum(
        energy_calc.read_phase_energy(pe.crossbar.weights) for pe in mpe.pes
    )
    total_energy = program_energy_total + compute_energy_total + array_read_energy_total

    print("=== ENERGY SUMMARY (uncalibrated estimates) ===")
    print(f"PEs used: {NUM_PES}")
    print(f"PROGRAM energy (scaled): {program_energy_total:.6e} J")
    print(f"Array-read energy estimate (once per PE): {array_read_energy_total:.6e} J")
    print(f"COMPUTE energy (ops):    {compute_energy_total:.6e} J")
    missing_buffer_costs = []
    for phase, counter_names, energy_method in (
        ("STORE", ("STORE",), energy_calc.output_buffer_store_energy),
        ("READ", ("READ", "FEEDBACK"), energy_calc.output_buffer_read_energy),
    ):
        counts = [sum(pe.output_buffer_words[name] for name in counter_names) for pe in mpe.pes]
        full_buffers = sum(count / pe.output_fifo.size for count, pe in zip(counts, mpe.pes))
        energies = [energy_method(count, pe.output_fifo.size) for count, pe in zip(counts, mpe.pes)]
        print(f"Output-buffer {phase}: {sum(counts)} words = {full_buffers:g} full-buffer equivalents")
        if any(energy is None for energy in energies):
            missing_buffer_costs.append(phase)
            print(f"Output-buffer {phase} energy: NOT CONFIGURED (set OUTPUT_BUFFER_{phase}_ENERGY_J)")
        else:
            buffer_energy = sum(energies)
            total_energy += buffer_energy
            print(f"Output-buffer {phase} energy: {buffer_energy:.6e} J")
    if missing_buffer_costs:
        print(f"PARTIAL MODELED TOTAL (excludes unset buffer {'/'.join(missing_buffer_costs)} costs): {total_energy:.6e} J")
    else:
        print(f"MODELED TOTAL (PROGRAM+COMPUTE+ARRAY_READ+BUFFER_STORE+BUFFER_READ): {total_energy:.6e} J")
    print("This estimate does not include complete system energy.")
    print("---------------------------------------------------------------\n")

    # ----------------------------
    # 8) Latency summary (derived from scheduled ops)
    # ----------------------------
    parallel_cycles = max((pe.cycle for pe in mpe.pes), default=0)
    sequential_cycles = sum(pe.cycle for pe in mpe.pes)

    print("=== LATENCY SUMMARY (mapping1.m) ===")
    print(f"Independent-PE latency estimate: {parallel_cycles} cycles (no shared-bus contention)")
    print(f"Serialized PE transactions: {sequential_cycles} cycles (PROGRAM + COMPUTE + STORE + READ)")
    print("---------------------------------------------------------------\n")

    print(f"[CHIP] Stage cycles: {mpe.pes[0].stage_cycles}")
    print("Simulation complete.\n")

if __name__ == "__main__":
    main()

