# mapper.py
from __future__ import annotations

from dataclasses import dataclass
import math
import re
from pathlib import Path

PE_IN_ROWS = 512   # crossbar rows (inputs per MVM)
PE_OUT_COLS = 512  # crossbar cols (outputs per MVM)


@dataclass
class Dims:
    K: int = 512
    C: int = 512
    R: int = 1
    S: int = 1
    Y: int = 1
    X: int = 1
    stride_x: int = 1
    stride_y: int = 1


@dataclass
class MapResult:
    k_chunks: list[tuple[int, int]]  # [(k0, kN)], kN ≤ 512
    fold_steps: int                  # CRS folds
    fold_sizes: list[int]            # e.g., [512]
    patch_len: int                   # C*R*S
    loop_order: list[str]            # e.g., ["K","Y","X","CRS_fold"]


class SinglePEMapper:
    """
    Fixed policy for your Dot512 layer on one PE:
      - Specialized for K=C=512, R=S=Y=X=1
    """
    def map(self, dims: Dims | None = None) -> MapResult:
        if dims is None:
            dims = Dims()

        if not (dims.K == 512 and dims.C == 512 and dims.R == 1 and dims.S == 1 and dims.Y == 1 and dims.X == 1):
            raise ValueError("This mapper is specialized for K=C=512, R=S=Y=X=1.")

        k_chunks = [(0, min(PE_OUT_COLS, dims.K))]  # [(0,512)]
        patch_len = dims.C * dims.R * dims.S        # 512
        if patch_len != PE_IN_ROWS:
            raise ValueError("C*R*S must equal 512 for this fixed mapper.")

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
    xy_tile: int
    xy_stride: int
    spatial_dimension: str = "XY"


@dataclass
class MaestroMapResult:
    # weight/compute shape mapping (still uses your 512x512 PE limits)
    k_chunks: list[tuple[int, int]]   # [(k0,kN)], kN <= 512
    fold_steps: int                   # CRS folds
    fold_sizes: list[int]             # e.g., [25]
    patch_len: int                    # C*R*S
    loop_order: list[str]             # e.g., ["K","XY","CRS_fold"]

    # spatial work distribution implied by Maestro
    xy_size: int                      # Y*X
    xy_tile: int                      # SpatialMap(tile, stride) XY -> tile
    xy_stride: int                    # SpatialMap(tile, stride) XY -> stride
    num_pes_needed: int               # ceil(xy_size/xy_stride)
    array_rows: int = PE_IN_ROWS
    array_cols: int = PE_OUT_COLS
    single_pe_windows: bool = False


class MaestroMapper:
    """
    Reads Maestro-like .m mapping file and produces mapping that matches:
      SpatialMap(T, S) XY
    XY positions are derived implicitly from Dimensions Y and X.
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

        spatial = re.findall(r"SpatialMap\s*\(\s*(\d+)\s*,\s*(\d+)\s*\)\s*(\w+)\s*;", block_text)
        if len(spatial) != 1:
            raise ValueError("Exactly one SpatialMap is required.")
        if spatial[0][2] == "K":
            return MaestroDataflow(int(spatial[0][0]), int(spatial[0][1]), "K")

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
            xy_tile=int(m.group(1)),
            xy_stride=int(m.group(2)),
        )

    def map_from_file(self, mapping_path: str | Path) -> tuple[Dims, MaestroMapResult]:
        mapping_path = Path(mapping_path)
        text = mapping_path.read_text(encoding="utf-8", errors="ignore")

        # Strip comments before parsing any blocks or dimension keys.
        text = re.sub(r"(?m)(//|#).*", "", text)

        array_rows, array_cols = PE_IN_ROWS, PE_OUT_COLS
        if re.search(r"\bHardware\s*\{", text):
            hardware = self._parse_keyvals(self._parse_block(text, "Hardware"))
            array_rows, array_cols = hardware.get("ROWS", 0), hardware.get("COLUMNS", 0)
            if array_rows <= 0 or array_cols <= 0:
                raise ValueError("Hardware requires positive Rows and Columns.")

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

        if any(value <= 0 for value in vars(dims).values()):
            raise ValueError("All dimensions and strides must be positive.")

        if df.spatial_dimension == "K":
            # Supported policy: full CRS windows on one PE, SAME padding.
            temporal_items = re.findall(r"TemporalMap\s*\(\s*(\d+)\s*,\s*(\d+)\s*\)\s*(\w+)\s*;", df_block)
            temporal = {name: (int(tile), int(step)) for tile, step, name in temporal_items}
            expected = {"C": (dims.C, dims.C), "R": (dims.R, dims.R),
                        "S": (dims.S, dims.S), "X": (dims.S, dims.stride_x),
                        "Y": (dims.R, dims.stride_y)}
            if temporal != expected or len(temporal_items) != len(expected):
                raise ValueError(f"Single-PE window policy requires TemporalMap entries {expected}.")
            if (df.xy_tile, df.xy_stride) != (dims.K, dims.K) or dims.K > array_cols:
                raise ValueError("SpatialMap K must cover K columns within one array.")
            patch_len = dims.C * dims.R * dims.S
            if patch_len > array_rows:
                raise ValueError("A full patch must fit in the physical array rows.")
            xy_size = math.ceil(dims.Y / dims.stride_y) * math.ceil(dims.X / dims.stride_x)
            return dims, MaestroMapResult(
                [(0, dims.K)], 1, [patch_len], patch_len,
                ["K", "Y", "X", "CRS_fold"], xy_size, xy_size, xy_size, 1,
                array_rows, array_cols, True)

        # K chunks (<=512 cols)
        k_chunks: list[tuple[int, int]] = []
        k0 = 0
        while k0 < dims.K:
            kN = min(k0 + array_cols, dims.K)
            k_chunks.append((k0, kN))
            k0 = kN

        # CRS folding (<=512 rows)
        patch_len = dims.C * dims.R * dims.S
        if patch_len <= 0:
            raise ValueError("Invalid patch_len = C*R*S")

        fold_sizes: list[int] = []
        remaining = patch_len
        while remaining > 0:
            take = min(array_rows, remaining)
            fold_sizes.append(take)
            remaining -= take

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
            array_rows=array_rows, array_cols=array_cols,
        )
