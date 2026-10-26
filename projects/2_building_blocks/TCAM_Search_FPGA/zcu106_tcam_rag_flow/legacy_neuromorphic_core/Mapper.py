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

# ----------------------------------------
# Maestro Cluster(...) directive (MobileNetV2_clustered.m) support
# ----------------------------------------

@dataclass
class MaestroClusteredMapResult:
    """
    Minimal clustered mapping information extracted from a MAESTRO-style mapping file
    that uses the Cluster(...) directive.

    We currently use this primarily for MobileNetV2 CONV1 where:
      - Outer SpatialMap maps K across clusters
      - Cluster(size, P) defines physical cluster size
      - Inner SpatialMap maps C within each cluster
    """
    # weight/compute shape mapping (still uses your 32x32 PE limits)
    k_chunks: list[tuple[int, int]]    # [(k0,kN)], kN <= 32
    fold_steps: int                    # CRS folds
    fold_sizes: list[int]              # e.g., [27]
    patch_len: int                     # C*R*S
    loop_order: list[str]              # e.g., ["K","Y","X","Cluster","C"]

    # cluster directive
    cluster_size: int                  # e.g., 16
    cluster_dim: str                   # usually "P"

    # outer SpatialMap (before Cluster) -- typically K
    outer_spatial_dim: str             # e.g., "K"
    outer_tile: int                    # SpatialMap(tile, stride)
    outer_stride: int

    # inner SpatialMap (after Cluster) -- typically C
    inner_spatial_dim: str | None      # e.g., "C"
    inner_tile: int | None
    inner_stride: int | None

    # derived
    num_clusters: int                  # number of clusters implied by outer spatial mapping
    num_pes_total: int                 # num_clusters * cluster_size
    num_active_pes_total: int          # approximate "used" PEs (based on inner spatial mapping)
    
    # K spatial parallelism (outputs per PE operation)
    k_spatial_parallelism: int         # e.g., 32 from SpatialMap(32,32) K
    
    # Loop order analysis (NEW)
    temporal_loop_order: list[tuple[str, str, str]]  # [(dim, tile, stride), ...] in order
    has_rs_loops: bool                 # True if R>1 or S>1
    rs_before_yx: bool                 # True if R/S loops come before Y/X (filter-stationary)


class MaestroClusterMapper:
    """
    Reads a MAESTRO-like .m mapping file with per-layer blocks and Cluster(...) directive,
    and produces a minimal mapping usable by our simulator.

    Supported patterns (per layer):
      - Stride { X: a, Y: b }
      - Dimensions { K:..., C:..., R:..., S:..., Y:..., X:... }
      - Dataflow { ... SpatialMap(...) <DIM>; ... Cluster(<N>, <dim>); ... SpatialMap(...) <DIM>; ... }

    Notes:
      - We intentionally ignore TemporalMap(...) details for now.
      - We detect:
          * first SpatialMap before Cluster  -> outer map
          * first Cluster(...)              -> cluster config
          * first SpatialMap after Cluster  -> inner map
    """

    def _strip_comments(self, text: str) -> str:
        # Strip // ... end-of-line comments
        text = re.sub(r"//.*", "", text)
        # Strip # ... end-of-line comments (sometimes used in mapping files)
        text = re.sub(r"#.*", "", text)
        # Strip MATLAB-style % comments
        text = re.sub(r"%.*", "", text)
        return text

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

    def _parse_named_block(self, text: str, keyword: str, name: str) -> str:
        """
        Extract a named block like:
           Layer CONV1 { ... }
        """
        # Case 1: braced form -> Layer NAME { ... }
        m = re.search(rf"\b{re.escape(keyword)}\s+{re.escape(name)}\s*\{{", text)
        if m:
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
                raise ValueError(f"Unbalanced braces in '{keyword} {name}' block")
            return text[start : i - 1]

        # Case 2: unbraced MAESTRO style -> Layer NAME ... until next Layer
        m = re.search(rf"\b{re.escape(keyword)}\s+{re.escape(name)}\b", text)
        if not m:
            raise ValueError(f"Missing '{keyword} {name} {{...}}' block")

        start = m.end()
        rest = text[start:]
        m_next = re.search(rf"\b{re.escape(keyword)}\s+[A-Za-z_][A-Za-z0-9_]*\b", rest)
        end = start + m_next.start() if m_next else len(text)
        return text[start:end]

    def _parse_keyvals(self, block_text: str) -> dict[str, int]:
        out: dict[str, int] = {}
        for key, val in re.findall(r"([A-Za-z_]+)\s*:\s*([0-9]+)", block_text):
            out[key.upper()] = int(val)
        return out

    def _parse_temporal_loop_order(self, df_directives: list) -> list[tuple[str, str, str]]:
        """
        Extract TemporalMap sequence to determine loop nesting order.
        
        Returns:
            List of (dim, tile, stride) in order from outermost to innermost.
        """
        temporal_loops = []
        
        for kind, payload in df_directives:
            if kind == "TemporalMap":
                tile_s, stride_s, dim = payload
                temporal_loops.append((dim, tile_s, stride_s))
        
        return temporal_loops
    
    def _analyze_loop_structure(self, temporal_loops: list[tuple[str, str, str]], dims) -> tuple[bool, bool]:
        """
        Analyze whether R/S loops exist and their position relative to Y/X.
        
        Returns:
            (has_rs_loops, rs_before_yx)
        """
        # Check if R or S have size > 1
        has_rs = (dims.R > 1 or dims.S > 1)
        
        # Find positions of R, S, Y, X in loop order
        positions = {}
        for idx, (dim, _, _) in enumerate(temporal_loops):
            if dim.upper() in ['R', 'S', 'Y', 'X']:
                if dim.upper() not in positions:  # Keep first occurrence
                    positions[dim.upper()] = idx
        
        # Determine if R/S come before Y/X
        rs_before_yx = False
        if has_rs:
            r_pos = positions.get('R', float('inf'))
            s_pos = positions.get('S', float('inf'))
            y_pos = positions.get('Y', float('inf'))
            x_pos = positions.get('X', float('inf'))
            
            # If any R/S appears before any Y/X, it's filter-stationary
            min_rs_pos = min(r_pos, s_pos)
            min_yx_pos = min(y_pos, x_pos)
            rs_before_yx = (min_rs_pos < min_yx_pos)
        
        return has_rs, rs_before_yx
    
    def _parse_dataflow_directives(self, block_text: str) -> list[tuple[str, tuple]]:
        """
        Return a list of directives in-order.

        Each directive is:
          ("SpatialMap", (tile, stride, dim))
          ("TemporalMap", (tile, stride, dim))   # parsed but not used
          ("Cluster", (size, dim))
          ("Other", (raw_line,))
        """
        directives: list[tuple[str, tuple]] = []

        # Split by ';' so we can parse directives even if multiple are on a line
        cleaned = []
        for ln in block_text.splitlines():
            ln = ln.strip()
            if not ln:
                continue
            cleaned.append(ln)
        text = " ".join(cleaned)
        parts = [p.strip() for p in text.split(";") if p.strip()]

        for p in parts:
            # SpatialMap(tile,stride) DIM
            m = re.match(r"SpatialMap\s*\(\s*([0-9]+)\s*,\s*([0-9]+)\s*\)\s*([A-Za-z_]+)\s*$", p)
            if m:
                tile = int(m.group(1))
                stride = int(m.group(2))
                dim = m.group(3).upper()
                directives.append(("SpatialMap", (tile, stride, dim)))
                continue

            # TemporalMap(tile,stride) DIM  (tile/stride may be Sz(X) etc; we ignore those)
            m = re.match(r"TemporalMap\s*\(\s*([^,]+)\s*,\s*([^)]+)\)\s*([A-Za-z_]+)\s*$", p)
            if m:
                tile_s = m.group(1).strip()
                stride_s = m.group(2).strip()
                dim = m.group(3).upper()
                directives.append(("TemporalMap", (tile_s, stride_s, dim)))
                continue

            # Cluster(size, P)
            m = re.match(r"Cluster\s*\(\s*([0-9]+)\s*,\s*([A-Za-z_]+)\s*\)\s*$", p)
            if m:
                size = int(m.group(1))
                dim = m.group(2).upper()
                directives.append(("Cluster", (size, dim)))
                continue

            directives.append(("Other", (p,)))

        return directives

    def map_layer_from_file(self, mapping_path: str | Path, *, layer_name: str) -> tuple[Dims, MaestroClusteredMapResult]:
        mapping_path = Path(mapping_path)
        text = mapping_path.read_text(encoding="utf-8", errors="ignore")
        text = self._strip_comments(text)

        layer_text = self._parse_named_block(text, "Layer", layer_name)

        stride_block = self._parse_block(layer_text, "Stride")
        dim_block = self._parse_block(layer_text, "Dimensions")
        df_block = self._parse_block(layer_text, "Dataflow")

        stride_kv = self._parse_keyvals(stride_block)
        dim_kv = self._parse_keyvals(dim_block)
        df_directives = self._parse_dataflow_directives(df_block)

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

        # Find first Cluster(...) and spatial maps around it
        cluster_size = None
        cluster_dim = None
        outer_spatial = None
        inner_spatial = None

        seen_cluster = False
        for kind, payload in df_directives:
            if kind == "Cluster" and cluster_size is None:
                cluster_size, cluster_dim = payload
                seen_cluster = True
                continue
            if kind == "SpatialMap":
                tile, stride, dim = payload
                if not seen_cluster and outer_spatial is None:
                    outer_spatial = (dim, tile, stride)
                elif seen_cluster and inner_spatial is None:
                    inner_spatial = (dim, tile, stride)

        if outer_spatial is None:
            raise ValueError(
                f"Layer {layer_name}: expected at least one SpatialMap(...) before Cluster(...) in Dataflow block."
            )

        # Fallback for unclustered layers: emulate a minimal cluster model so
        # the existing clustered scheduler path can still execute unchanged.
        if cluster_size is None or cluster_dim is None:
            # Keep the synthetic cluster bounded while still covering all C channels.
            cluster_size = 1
            cluster_dim = "P"

            if inner_spatial is None:
                # Map C in contiguous chunks across synthetic PEs.
                channels_per_pe = int(math.ceil(int(dims.C) / float(cluster_size)))
                inner_spatial = ("C", channels_per_pe, channels_per_pe)

        outer_dim, outer_tile, outer_stride = outer_spatial
        if outer_stride <= 0:
            raise ValueError("outer SpatialMap stride must be >0")

        # Derive num_clusters from outer spatial map
        outer_dim_size = int(getattr(dims, outer_dim, 1))
        num_clusters = int(math.ceil(outer_dim_size / outer_stride))

        # Approximate active PEs from inner spatial map if present
        num_active_pes_total = num_clusters * min(cluster_size, dims.C)  # default assumption
        inner_dim = None
        inner_tile = None
        inner_stride = None
        if inner_spatial is not None:
            inner_dim, inner_tile, inner_stride = inner_spatial
            inner_dim_size = int(getattr(dims, inner_dim, 1))
            if inner_stride is None or inner_stride <= 0:
                inner_stride = 1
            # number of inner spatial units used per cluster
            inner_units = int(math.ceil(inner_dim_size / inner_stride))
            num_active_pes_total = num_clusters * min(cluster_size, inner_units)

        num_pes_total = num_clusters * int(cluster_size)
        
        # Extract K spatial parallelism from outer spatial map
        # SpatialMap(32,32) K means 32 output channels per operation
        k_spatial_parallelism = outer_tile if outer_dim == "K" else 1
        
        # Parse temporal loop order and analyze R/S vs Y/X ordering
        temporal_loop_order = self._parse_temporal_loop_order(df_directives)
        has_rs_loops, rs_before_yx = self._analyze_loop_structure(temporal_loop_order, dims)

        return dims, MaestroClusteredMapResult(
            k_chunks=k_chunks,
            fold_steps=len(fold_sizes),
            fold_sizes=fold_sizes,
            patch_len=patch_len,
            loop_order=["K", "Y", "X", "Cluster", "C"],
            cluster_size=int(cluster_size),
            cluster_dim=str(cluster_dim),
            outer_spatial_dim=str(outer_dim),
            outer_tile=int(outer_tile),
            outer_stride=int(outer_stride),
            inner_spatial_dim=(str(inner_dim) if inner_dim is not None else None),
            inner_tile=(int(inner_tile) if inner_tile is not None else None),
            inner_stride=(int(inner_stride) if inner_stride is not None else None),
            num_clusters=int(num_clusters),
            num_pes_total=int(num_pes_total),
            num_active_pes_total=int(num_active_pes_total),
            k_spatial_parallelism=int(k_spatial_parallelism),
            temporal_loop_order=temporal_loop_order,
            has_rs_loops=has_rs_loops,
            rs_before_yx=rs_before_yx,
        )
