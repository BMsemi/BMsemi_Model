from __future__ import annotations
# scheduler.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable
import numpy as np


@dataclass
class SchedOp:
    """
    op:
      - "PROGRAM": program a (32x32) weight tile
      - "COMPUTE_TILE": execute compute for a tile
    info: metadata required by the driver (k0,kN, tile coords, fold sizes, etc.)
    """
    op: str
    info: dict


class SinglePEScheduler:
    """
    Existing scheduler (kept as-is logically):
      - PROGRAM once per K-chunk
      - COMPUTE_TILE per (y,x) position
    """

    def make_weight_tile(self, W_chunk: np.ndarray) -> np.ndarray:
        W_chunk = np.asarray(W_chunk)
        if W_chunk.ndim != 2:
            raise ValueError("weight_provider must return a 2D array (rows, cols).")

        rows, cols = W_chunk.shape
        if rows > 32 or cols > 32:
            raise ValueError(f"weight tile must be <= 32x32, got {W_chunk.shape}.")

        tile = np.zeros((32, 32), dtype=float)
        tile[:rows, :cols] = W_chunk.astype(float, copy=False)
        return tile

    def build(
        self,
        dims: Any,
        mapping: Any,
        weight_provider: Callable[[int, int], np.ndarray],
    ) -> list[SchedOp]:
        ops: list[SchedOp] = []

        for (k0, kN) in mapping.k_chunks:
            W = weight_provider(k0, kN)
            tile = self.make_weight_tile(W)
            ops.append(SchedOp("PROGRAM", {"k0": k0, "kN": kN, "weight_tile": tile}))

            for yy in range(0, dims.Y, getattr(dims, "stride_y", 1)):
                for xx in range(0, dims.X, getattr(dims, "stride_x", 1)):
                    ops.append(
                        SchedOp(
                            "COMPUTE_TILE",
                            {
                                "y": yy,
                                "x": xx,
                                "k0": k0,
                                "kN": kN,
                                "fold_sizes": list(mapping.fold_sizes),
                            },
                        )
                    )
        return ops


# ...existing code...
class MaestroScheduler:
    """
    Scheduler that follows mapping1.m semantics for XY distribution across PEs,
    and (optionally) schedules CRS (C*R*S) temporally inside each PE.

    Emitted ops:
      - PROGRAM once per K-chunk
      - COMPUTE_TILE_CRS_FOLD for each (pe_id, xy_index, fold_id)
        where fold_id iterates over CRS folds (<=32 each)
    """

    def make_weight_tile(self, W_chunk: np.ndarray) -> np.ndarray:
        W_chunk = np.asarray(W_chunk)
        if W_chunk.ndim != 2:
            raise ValueError("weight_provider must return a 2D array (rows, cols).")

        rows, cols = W_chunk.shape
        if rows > 32 or cols > 32:
            raise ValueError(f"weight tile must be <= 32x32, got {W_chunk.shape}.")

        tile = np.zeros((32, 32), dtype=float)
        tile[:rows, :cols] = W_chunk.astype(float, copy=False)
        return tile

    @staticmethod
    def xy_to_yx(xy: int, X_dim: int) -> tuple[int, int]:
        """Row-major unflatten: xy = y*X + x."""
        return xy // X_dim, xy % X_dim

    @staticmethod
    def crs_to_c_r_s(crs: int, C: int, R: int, S: int) -> tuple[int, int, int]:
        """
        Flatten order: crs = c*(R*S) + r*S + s
        Returns (c,r,s).
        """
        rs = R * S
        c = crs // rs
        rem = crs % rs
        r = rem // S
        s = rem % S
        if c >= C:
            raise ValueError("crs index out of range")
        return c, r, s

    def build(
        self,
        dims: Any,
        mapping: Any,
        weight_provider: Callable[[int, int], np.ndarray],
        *,
        schedule_crs_temporally: bool = True,
    ) -> list[SchedOp]:
        """
        Expects mapping to have at least:
          mapping.k_chunks, mapping.fold_sizes,
          mapping.xy_size, mapping.xy_tile, mapping.xy_stride, mapping.num_pes_needed

        If schedule_crs_temporally=True:
          emits COMPUTE_TILE_CRS_FOLD ops (per xy_index, per fold)
        Else:
          emits one COMPUTE_TILE per PE with the whole xy_indices list (older behavior).
        """
        ops: list[SchedOp] = []

        XY_size = int(mapping.xy_size)
        T = int(mapping.xy_tile)
        S_stride = int(mapping.xy_stride)
        num_pes = int(mapping.num_pes_needed)

        if XY_size <= 0:
            raise ValueError("mapping.xy_size must be > 0")
        if T <= 0:
            raise ValueError("mapping.xy_tile must be > 0")
        if S_stride <= 0:
            raise ValueError("mapping.xy_stride must be > 0")

        X_dim = int(getattr(dims, "X", 1))
        C_dim = int(getattr(dims, "C", 1))
        R_dim = int(getattr(dims, "R", 1))
        S_dim = int(getattr(dims, "S", 1))

        patch_len = C_dim * R_dim * S_dim
        # mapping.fold_sizes should sum to patch_len
        if sum(mapping.fold_sizes) != patch_len:
            raise ValueError(
                f"fold_sizes sum ({sum(mapping.fold_sizes)}) != C*R*S ({patch_len}). "
                "Check mapper vs mapping1.m dimensions."
            )

        for (k0, kN) in mapping.k_chunks:
            # PROGRAM once per K-chunk
            W = weight_provider(k0, kN)
            tile = self.make_weight_tile(W)
            ops.append(SchedOp("PROGRAM", {"k0": k0, "kN": kN, "weight_tile": tile}))

            # SpatialMap(T, S_stride) XY across PEs
            for pe_id in range(num_pes):
                xy0 = pe_id * S_stride
                if xy0 >= XY_size:
                    break
                xyN = min(xy0 + T, XY_size)

                xy_indices = list(range(xy0, xyN))

                if not schedule_crs_temporally:
                    # Older coarse op (kept for compatibility)
                    yx_coords = [self.xy_to_yx(xyi, X_dim) for xyi in xy_indices]
                    ops.append(
                        SchedOp(
                            "COMPUTE_TILE",
                            {
                                "pe_id": pe_id,
                                "xy0": xy0,
                                "xyN": xyN,
                                "xy_indices": xy_indices,
                                "yx_coords": yx_coords,
                                "k0": k0,
                                "kN": kN,
                                "fold_sizes": list(mapping.fold_sizes),
                            },
                        )
                    )
                    continue

                # New: schedule CRS temporally inside each PE
                # For each XY position assigned to this PE, emit ops per CRS fold.
                fold_sizes = list(mapping.fold_sizes)
                for xy_idx in xy_indices:
                    out_y, out_x = self.xy_to_yx(xy_idx, X_dim)

                    crs_cursor = 0
                    for fold_id, fold_sz in enumerate(fold_sizes):
                        crs0 = crs_cursor
                        crsN = crs_cursor + int(fold_sz)
                        crs_cursor = crsN

                        # Optional: provide explicit (c,r,s) list for this fold (useful for debugging)
                        crs_coords = [
                            self.crs_to_c_r_s(crs, C_dim, R_dim, S_dim) for crs in range(crs0, crsN)
                        ]

                        ops.append(
                            SchedOp(
                                "COMPUTE_TILE_CRS_FOLD",
                                {
                                    "pe_id": pe_id,
                                    "xy": xy_idx,
                                    "y": out_y,
                                    "x": out_x,
                                    "k0": k0,
                                    "kN": kN,
                                    "fold_id": fold_id,
                                    "crs0": crs0,
                                    "crsN": crsN,          # exclusive
                                    "crs_coords": crs_coords,  # list[(c,r,s)] for this fold
                                    "patch_len": patch_len,
                                },
                            )
                        )

        return ops


# -------------------------------------------------------
# Cluster(...) directive scheduler (for MobileNetV2 CONV1)
# -------------------------------------------------------

class MaestroClusterScheduler:
    """
    Scheduler for a clustered mapping (Cluster(...) directive).

    This scheduler is intentionally minimal:
      - We treat the outer SpatialMap before Cluster as mapping the K dimension across clusters.
      - We treat the first SpatialMap after Cluster as mapping the C dimension across PEs inside each cluster.
      - Y and X are iterated temporally (with stride_x/stride_y).

    Emitted ops (streaming via generator to avoid huge lists):
      - "PROGRAM_CLUSTER": program a weight tile for a given (cluster_id, k_local)
      - "COMPUTE_CLUSTER_C": compute one partial contribution for (cluster_id, c0:cN) at (y_in,x_in)

    Expected mapping fields (from Mapper.MaestroClusteredMapResult):
      - cluster_size, num_clusters
      - outer_spatial_dim/outer_tile/outer_stride (typically K)
      - inner_spatial_dim/inner_tile/inner_stride (typically C)
      - k_chunks, fold_sizes, patch_len
    """

    def make_weight_tile(self, W_chunk: np.ndarray) -> np.ndarray:
        W_chunk = np.asarray(W_chunk)
        if W_chunk.ndim != 2:
            raise ValueError("weight_provider must return a 2D array (rows, cols).")

        rows, cols = W_chunk.shape
        if rows > 32 or cols > 32:
            raise ValueError(f"weight tile must be <= 32x32, got {W_chunk.shape}.")

        tile = np.zeros((32, 32), dtype=float)
        tile[:rows, :cols] = W_chunk.astype(float, copy=False)
        return tile

    def iter_ops(
        self,
        dims: Any,
        mapping: Any,
        weight_provider: Callable[[int, int], np.ndarray],
        *,
        max_out_y: int | None = None,
        max_out_x: int | None = None,
    ):
        """
        Generator of SchedOp objects.
        
        UPDATED: Now accounts for K spatial parallelism (e.g., SpatialMap(32,32) K).
        Each PE operation produces K_SPATIAL_PARALLELISM output channels (not just 1).

        max_out_y / max_out_x:
          If set, only schedules the top-left (max_out_y, max_out_x) outputs.
          This is strongly recommended for bring-up/debug since full MobileNetV2 CONV1
          is extremely large (112x112 outputs for stride=2).
        """
        cluster_size = int(mapping.cluster_size)
        num_clusters = int(mapping.num_clusters)
        
        # Extract K spatial parallelism (how many output channels per operation)
        k_spatial_parallelism = int(getattr(mapping, 'k_spatial_parallelism', 1))

        # Outer map -> K across clusters
        outer_dim = str(mapping.outer_spatial_dim).upper()
        if outer_dim != "K":
            # We can still support others, but the current driver assumes K.
            pass
        outer_tile = int(mapping.outer_tile)
        outer_stride = int(mapping.outer_stride)

        # Inner map -> C across PEs inside cluster - KEY FIX FOR C DISTRIBUTION
        inner_dim = getattr(mapping, "inner_spatial_dim", None)
        inner_tile = getattr(mapping, "inner_tile", None)
        inner_stride = getattr(mapping, "inner_stride", None)
        inner_dim = (str(inner_dim).upper() if inner_dim is not None else None)
        inner_tile = (int(inner_tile) if inner_tile is not None else None)
        inner_stride = (int(inner_stride) if inner_stride is not None else None)
        
        # Default to single-channel processing if not specified
        if inner_tile is None:
            inner_tile = 1
        if inner_stride is None:
            inner_stride = 1

        stride_y = int(getattr(dims, "stride_y", 1))
        stride_x = int(getattr(dims, "stride_x", 1))
        if stride_y <= 0 or stride_x <= 0:
            raise ValueError("stride_x/stride_y must be >0")

        # Derive output grid size from input grid + stride.
        out_Y = int((int(dims.Y) + stride_y - 1) // stride_y)
        out_X = int((int(dims.X) + stride_x - 1) // stride_x)

        if max_out_y is not None:
            out_Y = min(out_Y, int(max_out_y))
        if max_out_x is not None:
            out_X = min(out_X, int(max_out_x))
        
        # Extract dimensions for C distribution
        C_dim = int(getattr(dims, "C", 1))
        R_dim = int(getattr(dims, "R", 1))
        S_dim = int(getattr(dims, "S", 1))
        K_dim = int(getattr(dims, "K", 1))
        patch_len = C_dim * R_dim * S_dim

        # For each K chunk, program cluster weights, then schedule compute
        for (k0, kN) in mapping.k_chunks:
            W = weight_provider(k0, kN)  # shape: (patch_len, kN-k0)
            W = np.asarray(W, dtype=float)
            if W.shape[1] != (kN - k0):
                raise ValueError(f"weight_provider returned cols={W.shape[1]} but expected {kN-k0}")
            if W.shape[0] != int(mapping.patch_len):
                raise ValueError(f"weight_provider returned rows={W.shape[0]} but expected patch_len={mapping.patch_len}")

            # =====================================================================
            # WEIGHT PROGRAMMING: One PROGRAM_CLUSTER per PE with ALL K columns.
            #
            # BUG FIX: The previous loop iterated kk=0..outer_tile and issued a
            # separate PROGRAM_CLUSTER per k channel, each overwriting the same PE.
            # Only the last kk's weights survived when compute began, so channels
            # 0..(outer_tile-2) always produced zero.
            #
            # Fix: build ONE tile per PE that has every applicable k channel
            # loaded into its own column (k_local = kk - k0).  A single yield
            # per PE replaces the entire inner kk loop.
            # =====================================================================
            for cluster_id in range(num_clusters):
                k_cluster_start = cluster_id * outer_stride
                if k_cluster_start >= K_dim:
                    break

                # All K channels for this cluster that fall inside the current chunk
                k_range = [
                    kk for kk in range(k_cluster_start,
                                       min(k_cluster_start + outer_tile, K_dim))
                    if k0 <= kk < kN
                ]
                if not k_range:
                    continue

                num_pes_for_c = int(np.ceil(C_dim / inner_stride))
                num_pes_for_c = min(num_pes_for_c, cluster_size)

                for pe_offset in range(num_pes_for_c):
                    c_start = pe_offset * inner_stride
                    c_end = min(c_start + inner_tile, C_dim)

                    if c_start >= C_dim:
                        break

                    pe_id = cluster_id * cluster_size + pe_offset

                    # Build tile with ALL applicable K channels in their columns.
                    # Physical crossbar is 32×32: rows = input channels (C-fold),
                    # cols = output neurons (logical K channels for this chunk).
                    # In PWM mode the 4-way weight split is handled inside the
                    # crossbar storage banks, not by explicit 4-column expansion
                    # in this scheduler tile layout.
                    tile = np.zeros((32, 32), dtype=float)
                    row_idx = 0
                    for c in range(c_start, c_end):
                        for r in range(R_dim):
                            for s in range(S_dim):
                                if row_idx >= 32:
                                    break
                                crs_idx = c * (R_dim * S_dim) + r * S_dim + s
                                if crs_idx < patch_len:
                                    for kk in k_range:
                                        k_local = kk - k0
                                        if k_local < 32:
                                            tile[row_idx, k_local] = W[crs_idx, k_local]
                                row_idx += 1

                    yield SchedOp(
                        "PROGRAM_CLUSTER",
                        {
                            "k0": k0,
                            "kN": kN,
                            "k_global": k_cluster_start,
                            "k_local": 0,
                            "cluster_id": cluster_id,
                            "cluster_size": cluster_size,
                            "pe_id": pe_id,
                            "pe_offset": pe_offset,
                            "c_start": c_start,
                            "c_end": c_end,
                            "weight_tile": tile,
                        },
                    )

            # =====================================================================
            # COMPUTE OPERATIONS: Always output-stationary (Y → X → K → C).
            #
            # BUG FIX: The previous filter-stationary branch (rs_before_yx=True)
            # iterated over every (r, s) separately.  Each op extracted a full
            # R×S patch from a SHIFTED position and dotted it against the full
            # weight matrix, then accumulated — giving a wrong 2-D correlation
            # instead of the correct 3×3 convolution.
            #
            # Now we always use output-stationary: one COMPUTE_CLUSTER_C per
            # (oy, ox, k_group, pe_offset).  Each PE extracts only its assigned
            # C channels' R×S patch from position (oy*stride, ox*stride) in the
            # pre-padded input — matching how the tile was programmed.
            # =====================================================================
            for oy in range(out_Y):
                for ox in range(out_X):
                    for k_group_start in range(0, K_dim, k_spatial_parallelism):
                        k_group_end = min(k_group_start + k_spatial_parallelism, K_dim)

                        cluster_id = (k_group_start // outer_stride) % num_clusters

                        if k_group_end <= k0 or k_group_start >= kN:
                            continue

                        num_pes_for_c = int(np.ceil(C_dim / inner_stride))
                        num_pes_for_c = min(num_pes_for_c, cluster_size)

                        for pe_offset in range(num_pes_for_c):
                            c_start = pe_offset * inner_stride
                            c_end = min(c_start + inner_tile, C_dim)

                            if c_start >= C_dim:
                                break

                            pe_id = cluster_id * cluster_size + pe_offset

                            # Top-left corner of the R×S window in the pre-padded input
                            y_in = oy * stride_y
                            x_in = ox * stride_x

                            yield SchedOp(
                                "COMPUTE_CLUSTER_C",
                                {
                                    "pe_id": pe_id,
                                    "cluster_id": cluster_id,
                                    "pe_offset": pe_offset,
                                    "c_start": c_start,
                                    "c_end": c_end,
                                    "k_global": k_group_start,
                                    "k_end": k_group_end,
                                    "k_count": k_group_end - k_group_start,
                                    "y_in": y_in,
                                    "x_in": x_in,
                                    "oy": oy,
                                    "ox": ox,
                                    "r": 0,
                                    "s": 0,
                                    "loop_order": "YXKC",
                                },
                            )
