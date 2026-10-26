"""MNIST edge-filter demo (Neuromorphic_compiler/user.py, mapping1.m) as an operation.

One 5x5 SAME-padded window per output pixel is dotted with two crossbar columns
(column 0 = '+', column 1 = '-'); output = col0 - col1 (column-differential).
The default kernels are exactly the 0/1 weights used in user.py's W_manual
(rows 16..24 on col 0, rows 0..10 on col 1 of the flattened 5x5 patch).

The window/padding convention was checked against the bundled
pe_firstcol_details.txt (32/32 PE dot sums match).

params:
  digit   5 | 9          bundled binary 28x28 digit            (default 5)
  image   28x28 of 0/1   OR your own binary image
  kernel_pos, kernel_neg 5x5 numbers, optional overrides
  output  "map" (default) returns the full 28x28 map; "summary" omits it
"""
from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

import numpy as np

DEFAULT_DIR = Path(__file__).resolve().parents[2] / "projects" / "2_building_blocks" / "Neuromorphic_compiler"
R = S = 5


def _dir() -> Path:
    return DEFAULT_DIR


def default_kernels() -> tuple[np.ndarray, np.ndarray]:
    pos = np.zeros(R * S); pos[16:25] = 1
    neg = np.zeros(R * S); neg[0:11] = 1
    return pos.reshape(R, S), neg.reshape(R, S)


@lru_cache(maxsize=4)
def _digit(d: int) -> np.ndarray:
    return np.loadtxt(_dir() / f"mnist{d}_bin_28x28.txt", dtype=np.int64)


def _kernel(v, name):
    k = np.asarray(v, dtype=np.float64)
    if k.shape != (R, S) or not np.all(np.isfinite(k)):
        raise ValueError(f"{name} must be a {R}x{S} array of numbers")
    return k


def edge_filter(params: dict) -> dict:
    if "image" in params:
        img = np.asarray(params["image"])
        if img.ndim != 2 or img.size == 0 or max(img.shape) > 128 or not np.isin(img, (0, 1)).all():
            raise ValueError("image must be a 2-D array of 0/1 (up to 128x128)")
        img = img.astype(np.int64)
    else:
        d = int(params.get("digit", 5))
        if d not in (5, 9):
            raise ValueError("digit must be 5 or 9")
        img = _digit(d)
    kp, kn = default_kernels()
    if "kernel_pos" in params:
        kp = _kernel(params["kernel_pos"], "kernel_pos")
    if "kernel_neg" in params:
        kn = _kernel(params["kernel_neg"], "kernel_neg")

    H, W = img.shape
    p = np.pad(img, (R // 2, S // 2))
    win = np.lib.stride_tricks.sliding_window_view(p, (R, S))        # (H, W, R, S)
    pos = np.einsum("yxrs,rs->yx", win, kp)
    neg = np.einsum("yxrs,rs->yx", win, kn)
    out = pos - neg
    res = {"shape": [H, W], "min": float(out.min()), "max": float(out.max()),
           "nonzero": int(np.count_nonzero(out)), "image_ones": int(img.sum()),
           "pe_tiles_32x32": 1, "windows": H * W}
    if params.get("output", "map") != "summary":
        res["map"] = out.astype(int).tolist() if np.all(out == np.rint(out)) else out.tolist()
    return res
