from __future__ import annotations

import math
import json
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Iterable

from .isa import CsrOp, compute_tile, wait_init
from .mapping import LayerSpec, ceil_div
from .noc import role_policy


@dataclass(frozen=True)
class CompilerConfig:
    model_name: str = "Qwen3-1.7B"
    fp4llm_tile_rows: int = 64
    fp4llm_tile_cols: int = 32
    x1_macro_rows: int = 32
    x1_macro_cols: int = 32
    x1_macros_per_endpoint: int = 4
    stored_cell_banks_per_weight: int = 4
    default_vc: int = 1
    default_dst: int = 0
    default_gain_shift: int = 0
    max_materialized_tasks: int | None = 20000

    @property
    def row_shards_per_fp4llm_tile(self) -> int:
        return ceil_div(self.fp4llm_tile_rows, self.x1_macro_rows)

    def as_dict(self) -> dict[str, object]:
        out = asdict(self)
        out["row_shards_per_fp4llm_tile"] = self.row_shards_per_fp4llm_tile
        return out


@dataclass(frozen=True)
class TileTask:
    phase: str
    layer_name: str
    role: str
    tile_group: int
    token_block: int
    row_block: int
    col_block: int
    input_start: int
    input_end: int
    output_start: int
    output_end: int
    token_start: int
    token_end: int
    macro_lanes: tuple[int, ...]

    def as_dict(self) -> dict[str, object]:
        out = asdict(self)
        out["macro_lanes"] = list(self.macro_lanes)
        return out

    @classmethod
    def from_dict(cls, data: dict[str, object]) -> "TileTask":
        item = dict(data)
        item["macro_lanes"] = tuple(int(v) for v in item.get("macro_lanes", ()))
        return cls(**item)


@dataclass(frozen=True)
class LayerPlan:
    layer: dict[str, object]
    role_policy: dict[str, object]
    token_blocks: int
    row_tiles: int
    col_tiles: int
    fp4llm_tile_tasks: int
    x1_compute_groups: int
    x1_compute_commands: int
    x1_noc_flits: int
    weight_program_cells: int
    weight_program_commands: int
    materialized_task_start: int
    materialized_task_count: int

    @classmethod
    def from_layer(
        cls,
        layer: LayerSpec,
        cfg: CompilerConfig | None = None,
        *,
        materialized_task_start: int = 0,
        materialized_task_count: int = 0,
    ) -> "LayerPlan":
        """Compute a plan directly from one normalized layer specification."""

        return _layer_plan(
            layer,
            cfg or CompilerConfig(),
            materialized_start=int(materialized_task_start),
            materialized_count=int(materialized_task_count),
        )

    def as_dict(self) -> dict[str, object]:
        return asdict(self)

    @property
    def name(self) -> str:
        return str(self.layer["name"])

    @property
    def phase(self) -> str:
        return str(self.layer["phase"])

    @property
    def role(self) -> str:
        return str(self.layer["role"])

    @property
    def in_dim(self) -> int:
        return int(self.layer["in_dim"])

    @property
    def out_dim(self) -> int:
        return int(self.layer["out_dim"])

    @property
    def tokens(self) -> int:
        return int(self.layer["tokens"])

    @property
    def total_tiles(self) -> int:
        return int(self.token_blocks) * int(self.row_tiles) * int(self.col_tiles)

    @classmethod
    def from_dict(cls, data: dict[str, object]) -> "LayerPlan":
        return cls(**dict(data))


@dataclass(frozen=True)
class ProgramManifest:
    model_name: str
    networks: dict[str, str]
    compiler_config: dict[str, object]
    layers: list[LayerPlan]
    tasks: list[TileTask] = field(default_factory=list)
    bootstrap_ops: list[CsrOp] = field(default_factory=list)
    sample_ops: list[CsrOp] = field(default_factory=list)
    truncated_tasks: bool = False

    @property
    def summary(self) -> dict[str, object]:
        totals = {
            "layers": len(self.layers),
            "materialized_tasks": len(self.tasks),
            "fp4llm_tile_tasks": sum(layer.fp4llm_tile_tasks for layer in self.layers),
            "x1_compute_commands": sum(layer.x1_compute_commands for layer in self.layers),
            "x1_noc_flits": sum(layer.x1_noc_flits for layer in self.layers),
            "weight_program_cells": sum(layer.weight_program_cells for layer in self.layers),
            "weight_program_commands": sum(layer.weight_program_commands for layer in self.layers),
            "truncated_tasks": self.truncated_tasks,
        }
        phase_counts: dict[str, int] = {}
        role_counts: dict[str, int] = {}
        for layer in self.layers:
            phase = str(layer.layer["phase"])
            role = str(layer.layer["role"])
            phase_counts[phase] = phase_counts.get(phase, 0) + 1
            role_counts[role] = role_counts.get(role, 0) + 1
        totals["phase_layer_counts"] = phase_counts
        totals["role_layer_counts"] = role_counts
        return totals

    def as_dict(self) -> dict[str, object]:
        return {
            "model_name": self.model_name,
            "networks": dict(self.networks),
            "compiler_config": dict(self.compiler_config),
            "summary": self.summary,
            "truncated_tasks": self.truncated_tasks,
            "layers": [layer.as_dict() for layer in self.layers],
            "tasks": [task.as_dict() for task in self.tasks],
            "bootstrap_ops": [op.as_dict() for op in self.bootstrap_ops],
            "sample_ops": [op.as_dict() for op in self.sample_ops],
        }

    @classmethod
    def from_dict(cls, data: dict[str, object]) -> "ProgramManifest":
        summary = data.get("summary", {})
        if not isinstance(summary, dict):
            summary = {}
        return cls(
            model_name=str(data["model_name"]),
            networks={str(k): str(v) for k, v in dict(data.get("networks", {})).items()},
            compiler_config=dict(data.get("compiler_config", {})),
            layers=[LayerPlan.from_dict(item) for item in data.get("layers", [])],
            tasks=[TileTask.from_dict(item) for item in data.get("tasks", [])],
            bootstrap_ops=[CsrOp(**dict(item)) for item in data.get("bootstrap_ops", [])],
            sample_ops=[CsrOp(**dict(item)) for item in data.get("sample_ops", [])],
            truncated_tasks=bool(data.get("truncated_tasks", summary.get("truncated_tasks", False))),
        )

    def write_json(self, path: str | Path) -> None:
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(self.as_dict(), indent=2) + "\n", encoding="utf-8")

    @classmethod
    def read_json(cls, path: str | Path) -> "ProgramManifest":
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))


def config_from_manifest(manifest: ProgramManifest) -> CompilerConfig:
    valid_names = {item.name for item in fields(CompilerConfig)}
    cfg_data = {
        key: value
        for key, value in manifest.compiler_config.items()
        if key in valid_names
    }
    return CompilerConfig(**cfg_data)


def _token_tile(layer: LayerSpec) -> int:
    return max(1, int(layer.y_tile or layer.tokens or 1))


def _iter_layer_tasks(layer: LayerSpec, cfg: CompilerConfig, start_tile_group: int) -> Iterable[TileTask]:
    token_tile = _token_tile(layer)
    token_blocks = ceil_div(layer.tokens, token_tile)
    row_tiles = ceil_div(layer.in_dim, cfg.fp4llm_tile_rows)
    col_tiles = ceil_div(layer.out_dim, cfg.fp4llm_tile_cols)
    group = int(start_tile_group)
    for token_block in range(token_blocks):
        token_start = token_block * token_tile
        token_end = min(token_start + token_tile, layer.tokens)
        for row_block in range(row_tiles):
            input_start = row_block * cfg.fp4llm_tile_rows
            input_end = min(input_start + cfg.fp4llm_tile_rows, layer.in_dim)
            for col_block in range(col_tiles):
                output_start = col_block * cfg.fp4llm_tile_cols
                output_end = min(output_start + cfg.fp4llm_tile_cols, layer.out_dim)
                yield TileTask(
                    phase=layer.phase,
                    layer_name=layer.name,
                    role=layer.role,
                    tile_group=group & 0xFFF,
                    token_block=token_block,
                    row_block=row_block,
                    col_block=col_block,
                    input_start=input_start,
                    input_end=input_end,
                    output_start=output_start,
                    output_end=output_end,
                    token_start=token_start,
                    token_end=token_end,
                    macro_lanes=tuple(range(cfg.x1_macros_per_endpoint)),
                )
                group += 1


def _layer_plan(
    layer: LayerSpec,
    cfg: CompilerConfig,
    materialized_start: int,
    materialized_count: int,
) -> LayerPlan:
    token_blocks = ceil_div(layer.tokens, _token_tile(layer))
    row_tiles = ceil_div(layer.in_dim, cfg.fp4llm_tile_rows)
    col_tiles = ceil_div(layer.out_dim, cfg.fp4llm_tile_cols)
    fp4llm_tile_tasks = token_blocks * row_tiles * col_tiles
    row_shards = cfg.row_shards_per_fp4llm_tile
    x1_groups_per_tile = max(1, ceil_div(row_shards, cfg.x1_macros_per_endpoint))
    x1_compute_commands = fp4llm_tile_tasks * x1_groups_per_tile
    if layer.static_weights:
        weight_program_cells = layer.weight_elements * cfg.stored_cell_banks_per_weight
        x1_weight_tiles = ceil_div(layer.in_dim, cfg.x1_macro_rows) * ceil_div(layer.out_dim, cfg.x1_macro_cols)
        weight_program_commands = x1_weight_tiles * cfg.stored_cell_banks_per_weight
    else:
        weight_program_cells = 0
        weight_program_commands = 0
    return LayerPlan(
        layer=layer.as_dict(),
        role_policy=role_policy(layer.role).as_dict(),
        token_blocks=token_blocks,
        row_tiles=row_tiles,
        col_tiles=col_tiles,
        fp4llm_tile_tasks=fp4llm_tile_tasks,
        x1_compute_groups=x1_groups_per_tile,
        x1_compute_commands=x1_compute_commands,
        x1_noc_flits=x1_compute_commands,
        weight_program_cells=weight_program_cells,
        weight_program_commands=weight_program_commands,
        materialized_task_start=materialized_start,
        materialized_task_count=materialized_count,
    )


def plan_layer(
    layer: LayerSpec,
    cfg: CompilerConfig | None = None,
    *,
    materialized_task_start: int = 0,
    materialized_task_count: int = 0,
) -> LayerPlan:
    """Build the layer-level runtime plan for one :class:`LayerSpec`.

    This is the public single-layer entry point for runtime inference.  Use
    :func:`compile_model` when a complete prefill/decode manifest and task
    list are needed.
    """

    return LayerPlan.from_layer(
        layer,
        cfg,
        materialized_task_start=materialized_task_start,
        materialized_task_count=materialized_task_count,
    )


def iter_layer_tasks(
    layer: LayerSpec,
    cfg: CompilerConfig | None = None,
    *,
    start_tile_group: int = 0,
) -> Iterable[TileTask]:
    """Yield runtime tile tasks for one layer without compiling a manifest."""

    return _iter_layer_tasks(layer, cfg or CompilerConfig(), int(start_tile_group))


def lower_task_to_csr_ops(task: TileTask, cfg: CompilerConfig) -> list[CsrOp]:
    return compute_tile(
        task.role,
        row=task.input_start // cfg.x1_macro_rows,
        col=task.output_start // cfg.x1_macro_cols,
        gain_shift=cfg.default_gain_shift,
        vc=cfg.default_vc,
        dst=cfg.default_dst,
        tile_group=task.tile_group,
        full_row=False,
    )


def compile_model(
    phase_layers: dict[str, tuple[str, list[LayerSpec]]],
    cfg: CompilerConfig | None = None,
) -> ProgramManifest:
    config = cfg or CompilerConfig()
    networks = {phase: network for phase, (network, _layers) in phase_layers.items()}
    layers: list[LayerPlan] = []
    tasks: list[TileTask] = []
    tile_group = 0
    truncated = False

    for _phase, (_network, phase_layer_list) in phase_layers.items():
        for layer in phase_layer_list:
            materialized_start = len(tasks)
            materialized_count = 0
            for task in _iter_layer_tasks(layer, config, tile_group):
                if config.max_materialized_tasks is None or len(tasks) < config.max_materialized_tasks:
                    tasks.append(task)
                    materialized_count += 1
                else:
                    truncated = True
                tile_group += 1
            layers.append(
                plan_layer(
                    layer,
                    config,
                    materialized_task_start=materialized_start,
                    materialized_task_count=materialized_count,
                )
            )

    bootstrap_ops = wait_init()
    sample_ops: list[CsrOp] = []
    if tasks:
        sample_ops = lower_task_to_csr_ops(tasks[0], config)

    return ProgramManifest(
        model_name=config.model_name,
        networks=networks,
        compiler_config=config.as_dict(),
        layers=layers,
        tasks=tasks,
        bootstrap_ops=bootstrap_ops,
        sample_ops=sample_ops,
        truncated_tasks=truncated,
    )
