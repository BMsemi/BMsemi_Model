"""Layer 0 - foundational mapping.

One parameterised mapper replacing the diverged SinglePEMapper / MaestroMapper
copies (Neuromorphic_complier, Deep_gate, Yolo26_INT8_mac).  PE geometry is an
argument, not a module constant, so the 32x32 (X1/compiler) and 64-row x 32
signed-column (YOLO26 / FP4LLM) PEs share one code path.

Tiling rule (matches the documented YOLO26 mapping): a layer is a matrix of
K output channels by C*R*S input taps.  Outputs split into ceil(K/pe_cols)
column chunks; inputs fold into ceil(C*R*S/pe_rows) row folds.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
from math import ceil


@dataclass(frozen=True)
class PEGeometry:
    rows: int = 64   # input rows per MVM
    cols: int = 32   # signed logical output columns per MVM

    def __post_init__(self):
        if self.rows < 1 or self.cols < 1:
            raise ValueError("PE rows/cols must be >= 1")


@dataclass(frozen=True)
class LayerDims:
    K: int          # output channels
    C: int          # input channels
    R: int = 1      # kernel height
    S: int = 1      # kernel width
    Y: int = 1      # output height
    X: int = 1      # output width

    def __post_init__(self):
        for name, v in asdict(self).items():
            if not isinstance(v, int) or v < 1:
                raise ValueError(f"{name} must be a positive integer")


@dataclass(frozen=True)
class MapResult:
    k_chunks: list          # [(k0, kN)] column chunks
    fold_sizes: list        # row-fold sizes, sum == patch_len
    patch_len: int          # C*R*S
    pe_tiles: int           # weight tiles = len(k_chunks) * len(fold_sizes)
    mvm_ops: int            # MVMs for the full output map = pe_tiles * Y * X
    utilization: float      # used cells / allocated cells across all tiles

    def to_dict(self) -> dict:
        return asdict(self)


def map_layer(dims: LayerDims, pe: PEGeometry = PEGeometry()) -> MapResult:
    patch_len = dims.C * dims.R * dims.S
    k_chunks = [(k0, min(k0 + pe.cols, dims.K)) for k0 in range(0, dims.K, pe.cols)]
    n_folds = ceil(patch_len / pe.rows)
    fold_sizes = [min(pe.rows, patch_len - i * pe.rows) for i in range(n_folds)]
    tiles = len(k_chunks) * len(fold_sizes)
    used = dims.K * patch_len
    return MapResult(
        k_chunks=k_chunks,
        fold_sizes=fold_sizes,
        patch_len=patch_len,
        pe_tiles=tiles,
        mvm_ops=tiles * dims.Y * dims.X,
        utilization=used / (tiles * pe.rows * pe.cols),
    )
