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
