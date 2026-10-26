from __future__ import annotations

from dataclasses import asdict, dataclass, field
import json
from pathlib import Path
from typing import Any

import numpy as np


SCHEMA_VERSION = 1


def _json_ready(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    return value


@dataclass(frozen=True)
class X2Target:
    name: str = "zcu106-axi-x2-mvm-v0.2.34"
    max_rows: int = 32
    max_cols: int = 32
    max_batch: int = 32
    accumulator_dtype: str = "int32"
    weight_dtype: str = "int8"
    activation_dtype: str = "int8"

    def validate(self) -> None:
        if self.max_rows != 32 or self.max_cols != 32 or self.max_batch != 32:
            raise ValueError(
                "The implemented RTL contract is fixed at max_rows=max_cols=max_batch=32."
            )
        if self.accumulator_dtype != "int32":
            raise ValueError("The implemented RTL accumulator type is int32.")
        if self.weight_dtype != "int8" or self.activation_dtype != "int8":
            raise ValueError("The implemented RTL weight and activation types are int8.")


@dataclass(frozen=True)
class X2WeightTile:
    tile_id: int
    k_start: int
    n_start: int
    rows: int
    cols: int
    weight_key: str
    accumulate: bool

    def validate(self, target: X2Target, k_dim: int, n_dim: int) -> None:
        if self.tile_id < 0:
            raise ValueError("tile_id must be non-negative.")
        if not 0 <= self.k_start < k_dim or not 0 <= self.n_start < n_dim:
            raise ValueError(f"Tile {self.tile_id} starts outside the logical matrix.")
        if not 1 <= self.rows <= target.max_rows:
            raise ValueError(f"Tile {self.tile_id} rows={self.rows} exceed the target.")
        if not 1 <= self.cols <= target.max_cols:
            raise ValueError(f"Tile {self.tile_id} cols={self.cols} exceed the target.")
        if self.k_start + self.rows > k_dim or self.n_start + self.cols > n_dim:
            raise ValueError(f"Tile {self.tile_id} extends outside the logical matrix.")
        if not self.weight_key:
            raise ValueError(f"Tile {self.tile_id} has an empty weight key.")


@dataclass
class X2Program:
    name: str
    kind: str
    m_dim: int
    k_dim: int
    n_dim: int
    tiles: list[X2WeightTile]
    constants: dict[str, np.ndarray]
    target: X2Target = field(default_factory=X2Target)
    metadata: dict[str, Any] = field(default_factory=dict)

    def validate(self) -> None:
        self.target.validate()
        if self.kind not in {"matmul", "conv2d"}:
            raise ValueError(f"Unsupported X2 program kind: {self.kind}")
        if min(self.m_dim, self.k_dim, self.n_dim) < 1:
            raise ValueError(
                f"Logical dimensions must be positive, got MKN={self.m_dim,self.k_dim,self.n_dim}."
            )
        if not self.tiles:
            raise ValueError("An X2 program must contain at least one weight tile.")

        covered: set[tuple[int, int]] = set()
        for tile in self.tiles:
            tile.validate(self.target, self.k_dim, self.n_dim)
            if tile.weight_key not in self.constants:
                raise ValueError(f"Missing constant {tile.weight_key!r}.")
            weights = np.asarray(self.constants[tile.weight_key])
            if weights.dtype != np.dtype(np.int8):
                raise ValueError(
                    f"Tile {tile.tile_id} weights must be int8, got {weights.dtype}."
                )
            if weights.shape != (tile.rows, tile.cols):
                raise ValueError(
                    f"Tile {tile.tile_id} constant has shape {weights.shape}; "
                    f"expected {(tile.rows, tile.cols)}."
                )
            key = (tile.k_start, tile.n_start)
            if key in covered:
                raise ValueError(f"Duplicate logical tile start {key}.")
            covered.add(key)

    def manifest(self) -> dict[str, Any]:
        self.validate()
        return {
            "schema_version": SCHEMA_VERSION,
            "name": self.name,
            "kind": self.kind,
            "logical_shape": {
                "m": self.m_dim,
                "k": self.k_dim,
                "n": self.n_dim,
            },
            "target": asdict(self.target),
            "tiles": [asdict(tile) for tile in self.tiles],
            "constants": {
                key: {
                    "shape": list(np.asarray(value).shape),
                    "dtype": str(np.asarray(value).dtype),
                }
                for key, value in sorted(self.constants.items())
            },
            "metadata": _json_ready(self.metadata),
        }

    def emit_mlir(self) -> str:
        """
        Emit parseable generic MLIR for the prototype ``x2`` dialect.

        The operations intentionally use MLIR's generic syntax. This preserves a
        stable compiler boundary now; registering ODS operations and C++ rewrite
        passes can replace the Python lowering without changing program meaning.
        """

        self.validate()
        lines = [
            "module attributes {",
            f"  x2.schema_version = {SCHEMA_VERSION} : i32,",
            f"  x2.name = {json.dumps(self.name)},",
            f"  x2.kind = {json.dumps(self.kind)},",
            f"  x2.m = {self.m_dim} : i64,",
            f"  x2.k = {self.k_dim} : i64,",
            f"  x2.n = {self.n_dim} : i64",
            "} {",
        ]
        for tile in self.tiles:
            lines.extend(
                [
                    '  "x2.mvm_tile"() {',
                    f"    x2.tile_id = {tile.tile_id} : i64,",
                    f"    x2.k_start = {tile.k_start} : i64,",
                    f"    x2.n_start = {tile.n_start} : i64,",
                    f"    x2.rows = {tile.rows} : i64,",
                    f"    x2.cols = {tile.cols} : i64,",
                    f"    x2.weight_key = {json.dumps(tile.weight_key)},",
                    f"    x2.accumulate = {'true' if tile.accumulate else 'false'}",
                    "  } : () -> ()",
                ]
            )
        lines.append("}")
        return "\n".join(lines) + "\n"

    def save(self, directory: str | Path) -> Path:
        output = Path(directory)
        output.mkdir(parents=True, exist_ok=True)
        (output / "manifest.json").write_text(
            json.dumps(self.manifest(), indent=2),
            encoding="utf-8",
        )
        np.savez_compressed(output / "constants.npz", **self.constants)
        (output / "program.mlir").write_text(self.emit_mlir(), encoding="utf-8")
        return output

    @classmethod
    def load(cls, directory: str | Path) -> "X2Program":
        source = Path(directory)
        manifest = json.loads((source / "manifest.json").read_text(encoding="utf-8"))
        if int(manifest.get("schema_version", -1)) != SCHEMA_VERSION:
            raise ValueError(
                f"Unsupported X2 schema {manifest.get('schema_version')}; "
                f"expected {SCHEMA_VERSION}."
            )
        with np.load(source / "constants.npz", allow_pickle=False) as archive:
            constants = {key: np.array(archive[key], copy=True) for key in archive.files}
        shape = manifest["logical_shape"]
        program = cls(
            name=str(manifest["name"]),
            kind=str(manifest["kind"]),
            m_dim=int(shape["m"]),
            k_dim=int(shape["k"]),
            n_dim=int(shape["n"]),
            tiles=[X2WeightTile(**tile) for tile in manifest["tiles"]],
            constants=constants,
            target=X2Target(**manifest["target"]),
            metadata=dict(manifest.get("metadata", {})),
        )
        program.validate()
        return program
