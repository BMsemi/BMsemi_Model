# mapper.py
from __future__ import annotations

from dataclasses import dataclass
import math
import re
from pathlib import Path

PE_IN_ROWS = 32   # crossbar rows (inputs per MVM)
PE_OUT_COLS = 32  # crossbar cols (outputs per MVM)


@dataclass
class Dims:
    K: int = 32
    C: int = 32
    R: int = 1
    S: int = 1
    Y: int = 1
    X: int = 1
    stride_x: int = 1
    stride_y: int = 1


@dataclass
class MapResult:
    k_chunks: list[tuple[int, int]]  # [(k0, kN)], kN ≤ 32
    fold_steps: int                  # CRS folds
    fold_sizes: list[int]            # e.g., [32]
    patch_len: int                   # C*R*S
    loop_order: list[str]            # e.g., ["K","Y","X","CRS_fold"]


class SinglePEMapper:
    """
    Fixed policy for your Dot32 layer on one PE:
      - Specialized for K=C=32, R=S=Y=X=1
    """
    def map(self, dims: Dims | None = None) -> MapResult:
        if dims is None:
            dims = Dims()

        if not (dims.K == 32 and dims.C == 32 and dims.R == 1 and dims.S == 1 and dims.Y == 1 and dims.X == 1):
            raise ValueError("This mapper is specialized for K=C=32, R=S=Y=X=1.")

        k_chunks = [(0, min(PE_OUT_COLS, dims.K))]  # [(0,32)]
        patch_len = dims.C * dims.R * dims.S        # 32
        if patch_len != PE_IN_ROWS:
            raise ValueError("C*R*S must equal 32 for this fixed mapper.")

        return MapResult(
            k_chunks=k_chunks,
            fold_steps=1,
            fold_sizes=[PE_IN_ROWS],
            patch_len=patch_len,
            loop_order=["K", "Y", "X", "CRS_fold"],
        )


# -----------------------------
# Maestro (mapping1.m) support
# -----------------------------

@dataclass
class MaestroDataflow:
    merge_xy: bool
    xy_tile: int
    xy_stride: int


@dataclass
class MaestroMapResult:
    # weight/compute shape mapping (still uses your 32x32 PE limits)
    k_chunks: list[tuple[int, int]]   # [(k0,kN)], kN <= 32
    fold_steps: int                   # CRS folds
    fold_sizes: list[int]             # e.g., [25]
    patch_len: int                    # C*R*S
    loop_order: list[str]             # e.g., ["K","XY","CRS_fold"]

    # spatial work distribution implied by Maestro
    xy_size: int                      # Y*X
    xy_tile: int                      # SpatialMap(tile, stride) XY -> tile
    xy_stride: int                    # SpatialMap(tile, stride) XY -> stride
    num_pes_needed: int               # ceil(xy_size/xy_stride)


class MaestroMapper:
    """
    Reads Maestro-like .m mapping file and produces mapping that matches:
      Merge(XY, Y, X)
      SpatialMap(T, S) XY
    """

    def _parse_block(self, text: str, block_name: str) -> str:
        m = re.search(rf"\b{re.escape(block_name)}\s*\{{", text)
        if not m:
            raise ValueError(f"Missing '{block_name} {{...}}' block")

        i = m.end()
        depth = 1
        start = i
        while i < len(text) and depth > 0:
            if text[i] == "{":
                depth += 1
            elif text[i] == "}":
                depth -= 1
            i += 1
        if depth != 0:
            raise ValueError(f"Unbalanced braces in '{block_name}' block")

        return text[start : i - 1]

    def _parse_keyvals(self, block_text: str) -> dict[str, int]:
        out: dict[str, int] = {}
        for key, val in re.findall(r"([A-Za-z_]+)\s*:\s*([0-9]+)", block_text):
            out[key.upper()] = int(val)
        return out

    def _parse_dataflow(self, block_text: str) -> MaestroDataflow:
        # Remove '#' comments
        lines: list[str] = []
        for ln in block_text.splitlines():
            ln = ln.split("#", 1)[0].strip()
            if ln:
                lines.append(ln)

        merge_xy = any(re.search(r"\bMerge\s*\(\s*XY\s*,\s*Y\s*,\s*X\s*\)\s*;", ln) for ln in lines)

        # SpatialMap(tile,stride) XY;
        m = None
        for ln in lines:
            mm = re.search(r"\bSpatialMap\s*\(\s*([0-9]+)\s*,\s*([0-9]+)\s*\)\s*XY\s*;", ln)
            if mm:
                m = mm
                break
        if not m:
            raise ValueError("mapping file must include: SpatialMap(tile,stride) XY;")

        return MaestroDataflow(
            merge_xy=merge_xy,
            xy_tile=int(m.group(1)),
            xy_stride=int(m.group(2)),
        )

    def map_from_file(self, mapping_path: str | Path) -> tuple[Dims, MaestroMapResult]:
        mapping_path = Path(mapping_path)
        text = mapping_path.read_text(encoding="utf-8", errors="ignore")

        # Strip // comments
        text = re.sub(r"//.*", "", text)

        stride_block = self._parse_block(text, "Stride")
        dim_block = self._parse_block(text, "Dimensions")
        df_block = self._parse_block(text, "Dataflow")

        stride_kv = self._parse_keyvals(stride_block)
        dim_kv = self._parse_keyvals(dim_block)
        df = self._parse_dataflow(df_block)

        dims = Dims(
            K=dim_kv.get("K", 1),
            C=dim_kv.get("C", 1),
            R=dim_kv.get("R", 1),
            S=dim_kv.get("S", 1),
            Y=dim_kv.get("Y", 1),
            X=dim_kv.get("X", 1),
            stride_x=stride_kv.get("X", 1),
            stride_y=stride_kv.get("Y", 1),
        )

        # K chunks (<=32 cols)
        k_chunks: list[tuple[int, int]] = []
        k0 = 0
        while k0 < dims.K:
            kN = min(k0 + PE_OUT_COLS, dims.K)
            k_chunks.append((k0, kN))
            k0 = kN

        # CRS folding (<=32 rows)
        patch_len = dims.C * dims.R * dims.S
        if patch_len <= 0:
            raise ValueError("Invalid patch_len = C*R*S")

        fold_sizes: list[int] = []
        remaining = patch_len
        while remaining > 0:
            take = min(PE_IN_ROWS, remaining)
            fold_sizes.append(take)
            remaining -= take

        if not df.merge_xy:
            raise ValueError("Expected Merge(XY, Y, X); in Dataflow")

        xy_size = dims.Y * dims.X
        if df.xy_stride <= 0:
            raise ValueError("XY stride must be > 0")
        if df.xy_tile <= 0:
            raise ValueError("XY tile must be > 0")

        # SpatialMap(tile,stride) XY across PEs
        num_pes_needed = int(math.ceil(xy_size / df.xy_stride))

        return dims, MaestroMapResult(
            k_chunks=k_chunks,
            fold_steps=len(fold_sizes),
            fold_sizes=fold_sizes,
            patch_len=patch_len,
            loop_order=["K", "XY", "CRS_fold"],
            xy_size=xy_size,
            xy_tile=df.xy_tile,
            xy_stride=df.xy_stride,
            num_pes_needed=num_pes_needed,
        )