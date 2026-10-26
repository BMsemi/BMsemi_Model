# user.py
# Uses the scheduler: PROGRAM once, then NUM_ITERS COMPUTE_TILE ops.
# Manual 32x32 weights; 1x32 input; feedback: x_{k+1} = y_k.

import numpy as np
from neuromorphic_x2 import NeuromorphicX2

# ADD: Maestro scheduler/mapper
from Scheduler import SinglePEScheduler, SchedOp, MaestroScheduler
from Mapper import Dims, SinglePEMapper, MaestroMapper

from energy import EnergyCalculator
from Latency import latencycalculator

# ADD: Neuromorphic multi-PE core
from Multiple_PE_core import MultiPECore, MultiPENeuromorphicCore

import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

# --- Weights (32x32) ---with Top edge filter using column diffrential strategy using column 0 and column 1
W_manual = np.array([

    [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  
    [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  
    [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  
    [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  
    [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  
    [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  
    [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  
    [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  
    [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  
    [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  
    [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  
    [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  
    [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  
    [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  
    [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  
    [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  
    [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  
    [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],   
    [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  
    [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 


    
    # Edge boundary kernel 
], dtype=int)
assert W_manual.shape == (32, 32), f"Expected (32,32) weights, got {W_manual.shape}"


x_manual = np.array([1,0,1,0,1,0,1,0,0,0,1,1,0,0,1,0,
                    1,1,0,0,1,0,0,1,0,1,0,1,1,0,0,0], dtype=int)

# MNIST image stored as a matrix; mapping file uses (Y,X) = (rows,cols)
img5_bin = None   # 28x28 binary matrix (0/1)
img_Y = 28        # rows
img_X = 28        # cols
img_label = None  # should be 5

try:
    from sklearn.datasets import fetch_openml

    mnist = fetch_openml("mnist_784", version=1, parser="auto")
    X_data = mnist.data.to_numpy() if hasattr(mnist.data, "to_numpy") else np.asarray(mnist.data)
    y_labels = mnist.target.astype(int)

    idx = int(np.where(y_labels == 5)[0][0])
    img = X_data[idx].reshape(28, 28)        # grayscale 0..255

    img5_bin = (img > 127).astype(int)       # binary 0/1 matrix
    img_label = 5
    img_Y, img_X = img5_bin.shape            # should be (28, 28)

    print(f"[MNIST] Loaded digit {img_label} as binary matrix: Y={img_Y}, X={img_X}")

    # Optional: save matrix for Maestro to read/flatten
    np.savetxt(
        r"c:\Users\KannaSridhar\Desktop\Behavioral_py\mnist9_bin_28x28.txt",
        img5_bin,
        fmt="%d",
        delimiter=" "
    )

except ImportError:
    print("[MNIST] scikit-learn not installed. Run: pip install scikit-learn")
except Exception as e:
    print(f"[MNIST] Failed to load MNIST digit 5: {e}")

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
    outputs_getter(pe_id) -> 1D array-like of length 32
    """
    with open(path, "w", encoding="utf-8") as f:
        f.write(f"=== Used PE Outputs (count={len(pe_ids)}) (each line: 32 values) ===\n")
        for pe in pe_ids:
            out = np.asarray(outputs_getter(pe), dtype=float).reshape(-1)
            line = " ".join(f"{v:.6f}" for v in out[:32])
            f.write(f"PE {pe}: {line}\n")

def _write_pe_outputs_with_coords(path: str, pe_ids: list[int], outputs_getter, coords_getter):
    """
    outputs_getter(pe_id) -> 1D array-like of length 32
    coords_getter(pe_id) -> tuple(y, x)
    """
    with open(path, "w", encoding="utf-8") as f:
        f.write(f"=== PE Outputs with coords (count={len(pe_ids)}) (each line: 32 values) ===\n")
        for pe in pe_ids:
            out = np.asarray(outputs_getter(pe), dtype=float).reshape(-1)
            yx = coords_getter(pe)
            if yx is None:
                y, x = -1, -1
            else:
                y, x = yx
            line = " ".join(f"{v:.6f}" for v in out[:32])
            f.write(f"PE {pe} (y={y}, x={x}): {line}\n")



# =====================================================
# Main
# =====================================================
def main():
    """
    Run according to mapping1.m:
      - Mapper reads Dimensions/Stride/Dataflow from mapping file
      - Scheduler assigns XY tiles to 32 PEs (for 28*28 with SpatialMap(25,25))
      - Each PE is a NeuromorphicX2 (via MultiPENeuromorphicCore)
      - Energy/latency are computed from the scheduled ops
    """

    # ----------------------------
    # 0) Preconditions
    # ----------------------------
    mapping_file = r"c:\Users\KannaSridhar\Desktop\Behavioral_py\mapping1.m"

    if img5_bin is None:
        print("[ERROR] MNIST digit 5 image not loaded (img5_bin is None).")
        print("        Install scikit-learn and re-run, or provide the image manually.")
        return

    # Make sure weights are 2D (your W_manual currently becomes (1,32,32) due to extra brackets)
    W2 = np.asarray(W_manual)
    if W2.ndim == 3 and W2.shape[0] == 1:
        W2 = np.squeeze(W2, axis=0)
    if W2.shape != (32, 32):
        raise ValueError(f"W_manual must be (32,32). Got {W2.shape}.")

    # ----------------------------
    # 1) Mapping-file driven dims + mapping
    # ----------------------------
    mm = MaestroMapper()
    mdims, mmapping = mm.map_from_file(mapping_file)

    print("\n=== MAESTRO MAPPING (from mapping1.m) ===")
    print(f"Dims: K={mdims.K}, C={mdims.C}, R={mdims.R}, S={mdims.S}, Y={mdims.Y}, X={mdims.X}")
    print(f"Stride: stride_y={mdims.stride_y}, stride_x={mdims.stride_x}")
    print(f"XY_size={mmapping.xy_size}")
    print(f"SpatialMap(tile={mmapping.xy_tile}, stride={mmapping.xy_stride}) -> num_pes_needed={mmapping.num_pes_needed}")
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
    if patch_len > 32 or K > 32:
        raise ValueError("This run assumes patch_len<=32 and K<=32 to fit one 32x32 crossbar tile.")

    # Column-differential strategy uses two physical output columns (0 and 1).
    if K < 2:
        raise ValueError(f"Column-differential needs K>=2, got K={K}.")

    # Reuse your existing 32x32 weights by slicing the CRS rows that are used by the conv patch.
    # (Replace this later with a real conv kernel if needed.)
    W_conv = W2[:patch_len, :K]  # (patch_len, K)

    def maestro_weight_provider(k0: int, kN: int) -> np.ndarray:
        return W_conv[:, k0:kN]

    # ----------------------------
    # 3) Build schedule per mapping1.m
    # ----------------------------
    ms = MaestroScheduler()
    mops = ms.build(mdims, mmapping, maestro_weight_provider, schedule_crs_temporally=True)

    program_ops = [op for op in mops if op.op == "PROGRAM"]
    compute_ops = [op for op in mops if op.op == "COMPUTE_TILE_CRS_FOLD"]

    used_pe_ids = sorted({int(op.info["pe_id"]) for op in compute_ops})

    NUM_PES = int(mmapping.num_pes_needed)  # for 28x28 with SpatialMap(25,25) => 32
    print(f"[SCHEDULE] PROGRAM ops: {len(program_ops)}")
    print(f"[SCHEDULE] COMPUTE_TILE_CRS_FOLD ops: {len(compute_ops)}")
    print(f"[SCHEDULE] PEs used: {NUM_PES}\n")
    print(f"[SCHEDULE] PEs actually used: {len(used_pe_ids)} -> {used_pe_ids}\n")
    # ----------------------------
    # 4) Multi-PE core where each PE is NeuromorphicX2
    # ----------------------------
    mpe = MultiPENeuromorphicCore(num_pes=NUM_PES, input_size=32, output_size=32, iterations=0)

    # ----------------------------
    # 5) Execute ops + accumulate outputs
    # ----------------------------
    out = np.zeros((mdims.Y, mdims.X, mdims.K), dtype=float)

    # Column-differential accumulation happens during COMPUTE only:
    # store + and - columns separately, then subtract at the end.
    out_pos = np.zeros((mdims.Y, mdims.X), dtype=float)  # column 0 (+)
    out_neg = np.zeros((mdims.Y, mdims.X), dtype=float)  # column 1 (-)

    energy_calc = EnergyCalculator()
    per_compute_periphery = energy_calc.decoder_driver_energy() + energy_calc.adc_energy()

    compute_energy_total = 0.0
    num_compute_ops = 0
    num_program_ops = 0

     # FIX: count only for used PEs
    pe_compute_counts = {pe: 0 for pe in used_pe_ids}

    print("=== EXECUTION (PROGRAM + COMPUTE as per mapping1.m) ===")

    # Track first-seen output per PE and its (y,x) for verification/debug
    first_output_per_pe: dict[int, np.ndarray] = {}
    first_xy_per_pe: dict[int, tuple[int, int]] = {}
    # Track first input vector per PE used for the compute (length 32)
    first_vec_per_pe: dict[int, np.ndarray] = {}
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
        patch = _extract_patch_same_2d(img5_bin, oy, ox, mdims.R, mdims.S)

        # Temporal CRS fold: only drive [crs0:crsN] rows this op
        vec32 = np.zeros(32, dtype=float)
        vec32[crs0:crsN] = patch[crs0:crsN]

        # Compute on this PE (NeuromorphicX2 micro-architecture)
        out_full = mpe.compute_pe(pe_id, vec32)  # length 32

        # Column-differential accumulation (happens during computation)
        # Robust to K-chunking: columns 0 and 1 may arrive in different ops.
        if k0 <= 0 < kN:
            out_pos[oy, ox] += float(out_full[0])
        if k0 <= 1 < kN:
            out_neg[oy, ox] += float(out_full[1])

        # Energy accounting per scheduled compute op
        compute_energy_total += energy_calc.compute_energy(vec32, mpe.pes[pe_id].crossbar.weights)
        compute_energy_total += per_compute_periphery

        num_compute_ops += 1
        if pe_id in pe_compute_counts:
            pe_compute_counts[pe_id] += 1

        # Record first tile/output for this PE (matches the baseline verification behavior)
        if pe_id not in first_output_per_pe:
            first_output_per_pe[pe_id] = out_full.copy()
            first_xy_per_pe[pe_id] = (oy, ox)
            first_vec_per_pe[pe_id] = vec32.copy()

    # Finalize differential output into channel 0
    out[:, :, 0] = out_pos - out_neg

    print(f"[DONE] PROGRAM ops executed: {num_program_ops}")
    print(f"[DONE] COMPUTE ops executed: {num_compute_ops}\n")

    # FIX: dump only used PEs *after* execution, now mpe exists and has outputs
    pe_out_path = r"c:\Users\KannaSridhar\Desktop\Behavioral_py\pe_outputs_used.txt"
    _write_pe_outputs(pe_out_path, used_pe_ids, lambda pe: mpe.get_output(pe))
    print(f"[SAVE] Wrote used-PE outputs only: {pe_out_path}")

    # Also save the first outputs per PE (with their (y,x) coords)
    pe_first_path = r"c:\Users\KannaSridhar\Desktop\Behavioral_py\pe_outputs_first.txt"
    _write_pe_outputs_with_coords(
        pe_first_path,
        used_pe_ids,
        lambda pe: first_output_per_pe.get(pe, np.zeros(32)),
        lambda pe: first_xy_per_pe.get(pe, None),
    )
    print(f"[SAVE] Wrote first-PE outputs with coords: {pe_first_path}")

    # Detailed column-differential dot-product breakdown per PE (for the first nonzero patch it processed)
    pe_firstcol_details = r"c:\Users\KannaSridhar\Desktop\Behavioral_py\pe_firstcol_details.txt"
    with open(pe_firstcol_details, "w", encoding="utf-8") as f:
        f.write("=== Column-Differential Dot-Product Details (first tile per PE) ===\n")
        f.write("Format: For each PE, show (y,x), input vec32 nonzeros, weights[:,0]/weights[:,1], contributions, and (sum0-sum1).\n\n")
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
            patch = _extract_patch_same_2d(img5_bin, int(y), int(x), mdims.R, mdims.S)

            # Element-wise contributions for differential columns
            contrib0 = vec * wcol0
            contrib1 = vec * wcol1
            sum0 = float(np.dot(vec, wcol0))
            sum1 = float(np.dot(vec, wcol1))
            diff = sum0 - sum1

            f.write(f"PE {pe} (y={y}, x={x})\n")
            f.write("Input vec32 (indices:val where nonzero): ")
            nz = [(i, float(vec[i])) for i in range(32) if vec[i] != 0]
            f.write(" ".join([f"{i}:{v:.0f}" for i, v in nz]) + "\n")

            f.write("Weights[:,0] (diff col0): ")
            f.write(" ".join(f"{float(v):.0f}" for v in wcol0) + "\n")

            f.write("Weights[:,1] (diff col1): ")
            f.write(" ".join(f"{float(v):.0f}" for v in wcol1) + "\n")

            f.write("Contrib col0 (i:vec[i]*w[i,0] where nonzero): ")
            nzc0 = [(i, float(contrib0[i])) for i in range(32) if contrib0[i] != 0]
            if nzc0:
                f.write(" ".join([f"{i}:{v:.0f}" for i, v in nzc0]) + "\n")
            else:
                f.write("(none)\n")

            f.write("Contrib col1 (i:vec[i]*w[i,1] where nonzero): ")
            nzc1 = [(i, float(contrib1[i])) for i in range(32) if contrib1[i] != 0]
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
    # Program energy is paid each time PROGRAM happens, per PE.
    # Read/program models in your EnergyCalculator are per-array, so scale by NUM_PES.
    program_energy_one = energy_calc.program_energy(mpe.pes[0].crossbar.weights)
    read_energy_one = energy_calc.read_energy(mpe.pes[0].crossbar.weights)

    program_energy_total = num_program_ops * NUM_PES * (
        program_energy_one + energy_calc.decoder_driver_energy() + energy_calc.adc_energy()
    )
    # Keep read as "once per run per PE" (matches your earlier style)
    read_energy_total = NUM_PES * (read_energy_one + energy_calc.decoder_driver_energy() + energy_calc.adc_energy())

    print("=== ENERGY SUMMARY (mapping1.m, NeuromorphicX2 PEs) ===")
    print(f"PEs used: {NUM_PES}")
    print(f"PROGRAM energy (scaled): {program_energy_total:.6e} J")
    print(f"READ energy (scaled):    {read_energy_total:.6e} J")
    print(f"COMPUTE energy (ops):    {compute_energy_total:.6e} J")
    print(f"TOTAL (PROGRAM+COMPUTE): {(program_energy_total + compute_energy_total):.6e} J")
    print("---------------------------------------------------------------\n")

    # ----------------------------
    # 8) Latency summary (derived from scheduled ops)
    # ----------------------------
    parallel_cycles = max(pe_compute_counts.values()) if pe_compute_counts else 0
    sequential_cycles = num_compute_ops

    print("=== LATENCY SUMMARY (mapping1.m) ===")
    print(f"Parallel latency (PEs_used={len(used_pe_ids)}): {parallel_cycles} cycles (approx)")
    print(f"Sequential latency (1 PE):           {sequential_cycles} cycles (approx)")
    print("---------------------------------------------------------------\n")

    print("Simulation complete.\n")

if __name__ == "__main__":
    main()