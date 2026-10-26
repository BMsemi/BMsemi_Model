"""Building blocks: schedule summary + energy/latency estimate for a mapped layer.

Pure re-implementation of the counting logic in Neuromorphic_compiler
(Mapper.MaestroMapper / Scheduler.MaestroScheduler / energy.EnergyCalculator /
Latency.latencycalculator).  No file or weight data is needed, so these run
anywhere.  Equivalence with the original Mapper/Scheduler/EnergyCalculator was checked
by hand on 6 layer shapes (op counts, PEs, folds; energy to 1e-13 relative).
Note the original Scheduler itself rejects C*R*S > 32 weight tiles; here the
fold count for such layers is reported from the Mapper rule only.

Layer params (same names as the .m Dimensions block):
  K, C, R, S, Y, X        defaults 32,1,1,1,1,1
  xy_tile, xy_stride      SpatialMap(tile, stride) XY; defaults tile=1, stride=tile
                          (mapping1.m uses 25,25 for a 28x28 image)
  pe_rows/pe_cols         crossbar size (default 32/32)
"""
from __future__ import annotations

import math

# energy.py constants (J)
E_SET, E_RESET = 10.7882e-9, 6.994e-9
R_SET, R_RESET = 284.484e-15, 37.37e-15
DECODER_PER_OP = 27e-12
ADC_PER_COL = 2.67e-9
# Latency.py
INIT_LAT, ITER_LAT = 9, 6

MAX_DIM = 4096


def _int(p: dict, k: str, d: int, lo=1, hi=MAX_DIM) -> int:
    v = p.get(k, d)
    if isinstance(v, bool) or int(v) != v or not lo <= int(v) <= hi:
        raise ValueError(f"{k} must be an integer in [{lo},{hi}]")
    return int(v)


def _layer(p: dict) -> dict:
    rows, cols = _int(p, "pe_rows", 32, 1, 1024), _int(p, "pe_cols", 32, 1, 1024)
    K, C, R, S = (_int(p, k, d) for k, d in (("K", 32), ("C", 1), ("R", 1), ("S", 1)))
    Y, X = _int(p, "Y", 1), _int(p, "X", 1)
    xy = Y * X
    tile = _int(p, "xy_tile", 1, 1, xy)
    stride = _int(p, "xy_stride", tile, 1, xy)
    patch = C * R * S
    k_chunks = math.ceil(K / cols)
    folds = [min(rows, patch - i) for i in range(0, patch, rows)]
    pes = math.ceil(xy / stride)
    # positions per PE: PE i owns xy in [i*stride, min(i*stride+tile, xy))
    per_pe = [max(0, min(i * stride + tile, xy) - i * stride) for i in range(pes)]
    return dict(K=K, C=C, R=R, S=S, Y=Y, X=X, xy=xy, tile=tile, stride=stride, patch=patch,
                rows=rows, cols=cols, k_chunks=k_chunks, folds=folds, pes=pes, per_pe=per_pe)


def schedule_layer(p: dict) -> dict:
    L = _layer(p)
    program_ops = L["k_chunks"]
    compute_ops = L["k_chunks"] * sum(L["per_pe"]) * len(L["folds"])
    per_pe_ops = [n * len(L["folds"]) * L["k_chunks"] for n in L["per_pe"]]
    res = {
        "dims": {k: L[k] for k in ("K", "C", "R", "S", "Y", "X")},
        "spatial_map": {"xy_size": L["xy"], "tile": L["tile"], "stride": L["stride"]},
        "pes_needed": L["pes"], "k_chunks": L["k_chunks"],
        "fold_sizes": L["folds"], "patch_len": L["patch"],
        "program_ops": program_ops, "compute_ops": compute_ops,
        "max_compute_ops_per_pe": max(per_pe_ops), "loop_order": ["K", "XY", "CRS_fold"],
        "positions_covered": sum(L["per_pe"]),
    }
    if L["tile"] > L["stride"]:
        res["warning"] = "tile > stride: PEs overlap and some positions are computed more than once"
    if sum(L["per_pe"]) < L["xy"]:
        res["warning"] = "tile < stride: some output positions are never computed"
    return res


def energy_estimate(p: dict) -> dict:
    """Expected-value energy for binary weights/inputs of given density.

    Follows user.py's accounting: PROGRAM = program_ops x PEs x (cell program +
    periphery); READ = PEs x (cell read + periphery); COMPUTE = per compute op
    (active rows x row read energy + periphery).
    """
    L = _layer(p)
    wd, idn = float(p.get("weight_density", 0.5)), float(p.get("input_density", 0.5))
    if not (0 <= wd <= 1 and 0 <= idn <= 1):
        raise ValueError("weight_density and input_density must be in [0,1]")
    cells = L["rows"] * L["cols"]
    periph = 1024 * DECODER_PER_OP + L["cols"] * ADC_PER_COL      # decoder_driver_energy + adc_energy
    cell_e = cells * (wd * E_SET + (1 - wd) * E_RESET)
    prog = L["k_chunks"] * L["pes"] * (cell_e + periph)
    read = L["pes"] * (cell_e + periph)
    row_e = L["cols"] * (wd * R_SET + (1 - wd) * R_RESET)
    ops = L["k_chunks"] * sum(L["per_pe"])
    compute = 0.0
    for f in L["folds"]:
        compute += ops * (idn * f * row_e + periph)
    sched = schedule_layer(p)
    par, seq = sched["max_compute_ops_per_pe"], sched["compute_ops"]
    res = {"program_J": prog, "read_J": read, "compute_J": compute,
           "total_program_plus_compute_J": prog + compute,
           "assumptions": {"weight_density": wd, "input_density": idn, "basis": "expected value, binary 0/1"},
           "parallel_cycles": par, "sequential_cycles": seq, "pes": L["pes"]}
    if "temporal_iterations" in p:
        n = _int(p, "temporal_iterations", 0, 0, 10**6)
        res["temporal_latency_cycles"] = INIT_LAT + ITER_LAT * n
    return res
