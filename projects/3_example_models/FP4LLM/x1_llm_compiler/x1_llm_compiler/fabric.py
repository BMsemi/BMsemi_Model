from __future__ import annotations

import csv
import json
from dataclasses import asdict, dataclass
from enum import IntEnum
from pathlib import Path

from .mapping import ceil_div
from .noc import role_policy
from .planner import ProgramManifest, TileTask


class FabricSpace(IntEnum):
    HOST = 0
    PERIPHERAL_MEMORY = 1
    DIE_SRAM = 2
    X1_ARRAY = 3
    PEER_DIE = 4
    NOC_OUTPUT = 5


class DescriptorOp(IntEnum):
    NOP = 0
    DMA_LOAD = 1
    DMA_STORE = 2
    PROGRAM_X1_WINDOW = 3
    EXECUTE_TILE = 4
    NOC_SEND = 5
    REDUCE = 6
    EVICT_WINDOW = 7
    FENCE = 8
    PEER_LOAD = 9


SPACE_NAMES = {space.name.lower(): space for space in FabricSpace}
OP_NAMES = {op.name.lower(): op for op in DescriptorOp}
DESCRIPTOR_MAGIC = 0xA1
DESCRIPTOR_VERSION = 1
DESCRIPTOR_WORDS = 16
DESCRIPTOR_BYTES = DESCRIPTOR_WORDS * 4


@dataclass(frozen=True)
class FabricConfig:
    die_count: int = 1
    x1_macros_per_die: int = 20000
    x1_macros_per_endpoint: int = 4
    x1_rows: int = 32
    x1_cols: int = 32
    stored_cell_banks_per_weight: int = 4
    peripheral_memory_bytes: int = 8 * 1024**3
    dma_burst_bytes: int = 4096
    max_descriptors: int = 4096
    max_execute_descriptors_per_window: int = 2
    result_buffer_base: int = 0x4000_0000
    weight_store_base: int = 0x8000_0000
    x1_array_base: int = 0
    unified_protocol: str = "X1UP/1"
    resident_weights: bool | None = None
    weight_window_cells: int | None = None
    noc_coalescing_tiles: int = 1
    reduction_placement: str = "local"

    @property
    def die_cell_capacity(self) -> int:
        return int(self.x1_macros_per_die) * int(self.x1_rows) * int(self.x1_cols)

    @property
    def total_cell_capacity(self) -> int:
        return int(self.die_count) * self.die_cell_capacity

    @property
    def endpoints_per_die(self) -> int:
        return max(1, int(self.x1_macros_per_die) // max(1, int(self.x1_macros_per_endpoint)))

    @property
    def cells_per_window(self) -> int:
        if self.weight_window_cells is None:
            return max(1, self.die_cell_capacity)
        return max(1, min(int(self.weight_window_cells), self.die_cell_capacity))

    def as_dict(self) -> dict[str, object]:
        out = asdict(self)
        out.update(
            {
                "die_cell_capacity": self.die_cell_capacity,
                "total_cell_capacity": self.total_cell_capacity,
                "endpoints_per_die": self.endpoints_per_die,
                "cells_per_window": self.cells_per_window,
                "descriptor_words": DESCRIPTOR_WORDS,
                "descriptor_bytes": DESCRIPTOR_BYTES,
            }
        )
        return out


@dataclass(frozen=True)
class FabricEndpoint:
    space: str
    die: int = 0
    bank: int = 0
    address: int = 0
    port: int = 0

    def space_id(self) -> int:
        key = str(self.space).strip().lower()
        if key not in SPACE_NAMES:
            raise ValueError(f"Unknown fabric space {self.space!r}.")
        return int(SPACE_NAMES[key])

    def endpoint_word(self) -> int:
        return (
            ((self.space_id() & 0xF) << 28)
            | ((int(self.die) & 0xFF) << 20)
            | ((int(self.bank) & 0xFF) << 12)
            | (int(self.port) & 0xFFF)
        )

    def as_dict(self) -> dict[str, object]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, object]) -> "FabricEndpoint":
        return cls(**dict(data))


@dataclass(frozen=True)
class FabricDescriptor:
    op: str
    src: FabricEndpoint
    dst: FabricEndpoint
    byte_count: int
    layer_id: int = 0
    window_id: int = 0
    tile_group: int = 0
    token_block: int = 0
    row_block: int = 0
    col_block: int = 0
    role: str = "unknown"
    vc: int = 1
    route_dst: int = 0
    dependency: int = 0
    flags: int = 0
    aux0: int = 0
    aux1: int = 0

    def op_id(self) -> int:
        key = str(self.op).strip().lower()
        if key not in OP_NAMES:
            raise ValueError(f"Unknown descriptor op {self.op!r}.")
        return int(OP_NAMES[key])

    def role_id(self) -> int:
        if self.role == "unknown":
            return 0
        return role_policy(self.role).role_id

    def pack_words(self) -> list[int]:
        words = [0] * DESCRIPTOR_WORDS
        words[0] = (
            ((DESCRIPTOR_MAGIC & 0xFF) << 24)
            | ((DESCRIPTOR_VERSION & 0xF) << 20)
            | ((self.op_id() & 0xFF) << 12)
            | (int(self.flags) & 0xFFF)
        )
        words[1] = self.src.endpoint_word()
        words[2] = int(self.src.address) & 0xFFFF_FFFF
        words[3] = (int(self.src.address) >> 32) & 0xFFFF_FFFF
        words[4] = self.dst.endpoint_word()
        words[5] = int(self.dst.address) & 0xFFFF_FFFF
        words[6] = (int(self.dst.address) >> 32) & 0xFFFF_FFFF
        words[7] = int(self.byte_count) & 0xFFFF_FFFF
        words[8] = ((int(self.layer_id) & 0xFFFF) << 16) | (int(self.window_id) & 0xFFFF)
        words[9] = ((int(self.tile_group) & 0xFFFF) << 16) | (int(self.token_block) & 0xFFFF)
        words[10] = ((int(self.row_block) & 0xFFFF) << 16) | (int(self.col_block) & 0xFFFF)
        words[11] = (
            ((self.role_id() & 0xFF) << 24)
            | ((int(self.vc) & 0xFF) << 16)
            | ((int(self.route_dst) & 0xFF) << 8)
            | (int(self.dependency) & 0xFF)
        )
        words[12] = int(self.aux0) & 0xFFFF_FFFF
        words[13] = int(self.aux1) & 0xFFFF_FFFF
        words[14] = descriptor_checksum(words[:14])
        words[15] = 0
        return words

    def as_dict(self) -> dict[str, object]:
        out = asdict(self)
        out["src"] = self.src.as_dict()
        out["dst"] = self.dst.as_dict()
        out["words"] = [f"0x{word:08x}" for word in self.pack_words()]
        return out

    @classmethod
    def from_dict(cls, data: dict[str, object]) -> "FabricDescriptor":
        item = dict(data)
        item.pop("words", None)
        item["src"] = FabricEndpoint.from_dict(dict(item["src"]))
        item["dst"] = FabricEndpoint.from_dict(dict(item["dst"]))
        return cls(**item)


@dataclass(frozen=True)
class WeightWindow:
    layer_id: int
    window_id: int
    layer_name: str
    role: str
    phase: str
    weight_element_start: int
    weight_element_count: int
    programmed_cells: int
    byte_count: int
    assigned_die: int
    residency: str
    source: FabricEndpoint
    target: FabricEndpoint

    def as_dict(self) -> dict[str, object]:
        out = asdict(self)
        out["source"] = self.source.as_dict()
        out["target"] = self.target.as_dict()
        return out


@dataclass(frozen=True)
class StreamPlan:
    model_name: str
    execution_mode: str
    fabric: dict[str, object]
    summary: dict[str, object]
    windows: list[WeightWindow]
    descriptors: list[FabricDescriptor]

    def as_dict(self) -> dict[str, object]:
        return {
            "model_name": self.model_name,
            "execution_mode": self.execution_mode,
            "fabric": dict(self.fabric),
            "summary": dict(self.summary),
            "windows": [window.as_dict() for window in self.windows],
            "descriptors": [descriptor.as_dict() for descriptor in self.descriptors],
        }

    def write_json(self, path: str | Path) -> None:
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(self.as_dict(), indent=2) + "\n", encoding="utf-8")

    def write_csv(self, path: str | Path) -> None:
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=[
                    "idx",
                    "op",
                    "layer_id",
                    "window_id",
                    "role",
                    "src_space",
                    "src_die",
                    "src_addr",
                    "dst_space",
                    "dst_die",
                    "dst_addr",
                    "byte_count",
                    "tile_group",
                    "dependency",
                ],
            )
            writer.writeheader()
            for idx, descriptor in enumerate(self.descriptors):
                writer.writerow(
                    {
                        "idx": idx,
                        "op": descriptor.op,
                        "layer_id": descriptor.layer_id,
                        "window_id": descriptor.window_id,
                        "role": descriptor.role,
                        "src_space": descriptor.src.space,
                        "src_die": descriptor.src.die,
                        "src_addr": descriptor.src.address,
                        "dst_space": descriptor.dst.space,
                        "dst_die": descriptor.dst.die,
                        "dst_addr": descriptor.dst.address,
                        "byte_count": descriptor.byte_count,
                        "tile_group": descriptor.tile_group,
                        "dependency": descriptor.dependency,
                    }
                )

    @classmethod
    def read_json(cls, path: str | Path) -> "StreamPlan":
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(
            model_name=str(data["model_name"]),
            execution_mode=str(data["execution_mode"]),
            fabric=dict(data["fabric"]),
            summary=dict(data["summary"]),
            windows=[WeightWindow(**{**dict(item), "source": FabricEndpoint.from_dict(dict(item["source"])), "target": FabricEndpoint.from_dict(dict(item["target"]))}) for item in data["windows"]],
            descriptors=[FabricDescriptor.from_dict(dict(item)) for item in data["descriptors"]],
        )


def descriptor_checksum(words: list[int]) -> int:
    value = 0x5831_5550
    for word in words:
        value = ((value << 5) | (value >> 27)) & 0xFFFF_FFFF
        value ^= int(word) & 0xFFFF_FFFF
    return value & 0xFFFF_FFFF


def verify_descriptor_words(words: list[int]) -> bool:
    if len(words) != DESCRIPTOR_WORDS:
        return False
    if ((int(words[0]) >> 24) & 0xFF) != DESCRIPTOR_MAGIC:
        return False
    return descriptor_checksum([int(word) for word in words[:14]]) == int(words[14])


def _unique_static_layers(manifest: ProgramManifest) -> list[tuple[int, object]]:
    out: list[tuple[int, object]] = []
    seen: set[tuple[str, str, int, int]] = set()
    for idx, layer in enumerate(manifest.layers):
        item = dict(layer.layer)
        if not bool(item.get("static_weights", False)):
            continue
        key = (
            str(item["name"]),
            str(item["role"]),
            int(item["in_dim"]),
            int(item["out_dim"]),
        )
        if key in seen:
            continue
        seen.add(key)
        out.append((idx, layer))
    return out


def _execution_mode(total_cells: int, fabric: FabricConfig) -> str:
    if fabric.resident_weights is True:
        if total_cells <= fabric.die_cell_capacity:
            return "single_die_resident"
        if fabric.die_count > 1 and total_cells <= fabric.total_cell_capacity:
            return "multi_die_resident"
        return "resident_infeasible"
    if fabric.resident_weights is False:
        return "peripheral_streaming"
    if total_cells <= fabric.die_cell_capacity:
        return "single_die_resident"
    if fabric.die_count > 1 and total_cells <= fabric.total_cell_capacity:
        return "multi_die_resident"
    return "peripheral_streaming"


def _tasks_for_layer(manifest: ProgramManifest, layer_name: str, limit: int) -> list[TileTask]:
    tasks = [task for task in manifest.tasks if task.layer_name == layer_name]
    return tasks[: max(0, int(limit))]


def build_stream_plan(manifest: ProgramManifest, fabric: FabricConfig | None = None) -> StreamPlan:
    cfg = fabric or FabricConfig(
        x1_macros_per_endpoint=int(manifest.compiler_config.get("x1_macros_per_endpoint", 4)),
        x1_rows=int(manifest.compiler_config.get("x1_macro_rows", 32)),
        x1_cols=int(manifest.compiler_config.get("x1_macro_cols", 32)),
        stored_cell_banks_per_weight=int(manifest.compiler_config.get("stored_cell_banks_per_weight", 4)),
    )
    static_layers = _unique_static_layers(manifest)
    total_cells = sum(int(layer.weight_program_cells) for _idx, layer in static_layers)
    mode = _execution_mode(total_cells, cfg)
    windows: list[WeightWindow] = []
    descriptors: list[FabricDescriptor] = []
    next_weight_addr = int(cfg.weight_store_base)
    next_result_addr = int(cfg.result_buffer_base)
    dependency = 0

    for layer_id, layer_plan in static_layers:
        layer = dict(layer_plan.layer)
        cells_remaining = int(layer_plan.weight_program_cells)
        weight_elements_remaining = int(layer["weight_elements"])
        window_idx = 0
        while cells_remaining > 0:
            cells = min(cells_remaining, cfg.cells_per_window)
            weight_elements = min(
                weight_elements_remaining,
                ceil_div(cells, max(1, cfg.stored_cell_banks_per_weight)),
            )
            assigned_die = len(windows) % max(1, cfg.die_count)
            byte_count = cells
            src = FabricEndpoint(
                "peripheral_memory",
                die=0,
                bank=assigned_die,
                address=next_weight_addr,
            )
            dst_space = "x1_array" if mode != "peripheral_streaming" else "die_sram"
            dst = FabricEndpoint(
                dst_space,
                die=assigned_die,
                bank=0,
                address=cfg.x1_array_base + (window_idx * cfg.cells_per_window),
            )
            residency = "resident" if mode in {"single_die_resident", "multi_die_resident"} else "streamed"
            window = WeightWindow(
                layer_id=layer_id,
                window_id=window_idx,
                layer_name=str(layer["name"]),
                role=str(layer["role"]),
                phase=str(layer["phase"]),
                weight_element_start=int(layer["weight_elements"]) - weight_elements_remaining,
                weight_element_count=weight_elements,
                programmed_cells=cells,
                byte_count=byte_count,
                assigned_die=assigned_die,
                residency=residency,
                source=src,
                target=dst,
            )
            windows.append(window)

            load_op = "peer_load" if src.space == "peer_die" else "dma_load"
            descriptors.append(
                FabricDescriptor(
                    load_op,
                    src=src,
                    dst=FabricEndpoint("die_sram", die=assigned_die, bank=0, address=window_idx * cfg.dma_burst_bytes),
                    byte_count=byte_count,
                    layer_id=layer_id,
                    window_id=window_idx,
                    role=str(layer["role"]),
                    route_dst=assigned_die,
                    dependency=dependency,
                    flags=1 if residency == "resident" else 0,
                )
            )
            dependency = min(255, dependency + 1)
            descriptors.append(
                FabricDescriptor(
                    "program_x1_window",
                    src=FabricEndpoint("die_sram", die=assigned_die, bank=0, address=window_idx * cfg.dma_burst_bytes),
                    dst=FabricEndpoint("x1_array", die=assigned_die, bank=0, address=window.target.address),
                    byte_count=byte_count,
                    layer_id=layer_id,
                    window_id=window_idx,
                    role=str(layer["role"]),
                    route_dst=assigned_die,
                    dependency=dependency,
                    aux0=int(layer["in_dim"]),
                    aux1=int(layer["out_dim"]),
                )
            )
            dependency = min(255, dependency + 1)

            tasks_for_window = _tasks_for_layer(
                manifest,
                str(layer["name"]),
                cfg.max_execute_descriptors_per_window,
            )
            coalescing = max(1, int(cfg.noc_coalescing_tiles))
            for group_start in range(0, len(tasks_for_window), coalescing):
                task_group = tasks_for_window[group_start : group_start + coalescing]
                if not task_group:
                    continue
                first_task = task_group[0]
                group_bytes = 16 * len(task_group)
                group_result_addr = next_result_addr
                for group_offset, task in enumerate(task_group):
                    descriptors.append(
                        FabricDescriptor(
                            "execute_tile",
                            src=FabricEndpoint("x1_array", die=assigned_die, bank=0, address=window.target.address),
                            dst=FabricEndpoint(
                                "noc_output",
                                die=assigned_die,
                                bank=0,
                                address=group_result_addr + group_offset * 16,
                            ),
                            byte_count=16,
                            layer_id=layer_id,
                            window_id=window_idx,
                            tile_group=task.tile_group,
                            token_block=task.token_block,
                            row_block=task.row_block,
                            col_block=task.col_block,
                            role=task.role,
                            vc=int(manifest.compiler_config.get("default_vc", 1)),
                            route_dst=int(manifest.compiler_config.get("default_dst", assigned_die)),
                            dependency=dependency,
                        )
                    )
                    dependency = min(255, dependency + 1)
                descriptors.append(
                    FabricDescriptor(
                        "noc_send",
                        src=FabricEndpoint("noc_output", die=assigned_die, bank=0, address=group_result_addr),
                        dst=FabricEndpoint("die_sram", die=assigned_die, bank=1, address=group_result_addr),
                        byte_count=group_bytes,
                        layer_id=layer_id,
                        window_id=window_idx,
                        tile_group=first_task.tile_group,
                        token_block=first_task.token_block,
                        row_block=first_task.row_block,
                        col_block=first_task.col_block,
                        role=first_task.role,
                        vc=int(manifest.compiler_config.get("default_vc", 1)),
                        route_dst=int(manifest.compiler_config.get("default_dst", assigned_die)),
                        dependency=dependency,
                        flags=2 if len(task_group) > 1 else 0,
                        aux0=len(task_group),
                    )
                )
                next_result_addr += group_bytes
                dependency = min(255, dependency + 1)
                if str(cfg.reduction_placement).strip().lower() == "destination":
                    descriptors.append(
                        FabricDescriptor(
                            "reduce",
                            src=FabricEndpoint("die_sram", die=assigned_die, bank=1, address=group_result_addr),
                            dst=FabricEndpoint("die_sram", die=assigned_die, bank=1, address=group_result_addr),
                            byte_count=group_bytes,
                            layer_id=layer_id,
                            window_id=window_idx,
                            tile_group=first_task.tile_group,
                            token_block=first_task.token_block,
                            row_block=first_task.row_block,
                            col_block=first_task.col_block,
                            role=first_task.role,
                            vc=2,
                            route_dst=int(manifest.compiler_config.get("default_dst", assigned_die)),
                            dependency=dependency,
                            flags=2 if len(task_group) > 1 else 0,
                            aux0=len(task_group),
                        )
                    )
                    dependency = min(255, dependency + 1)

            if residency == "streamed":
                descriptors.append(
                    FabricDescriptor(
                        "evict_window",
                        src=FabricEndpoint("x1_array", die=assigned_die, bank=0, address=window.target.address),
                        dst=FabricEndpoint("peripheral_memory", die=0, bank=assigned_die, address=next_weight_addr),
                        byte_count=0,
                        layer_id=layer_id,
                        window_id=window_idx,
                        role=str(layer["role"]),
                        route_dst=assigned_die,
                        dependency=dependency,
                    )
                )
                dependency = min(255, dependency + 1)

            descriptors.append(
                FabricDescriptor(
                    "fence",
                    src=FabricEndpoint("die_sram", die=assigned_die, bank=0, address=0),
                    dst=FabricEndpoint("die_sram", die=assigned_die, bank=0, address=0),
                    byte_count=0,
                    layer_id=layer_id,
                    window_id=window_idx,
                    role=str(layer["role"]),
                    route_dst=assigned_die,
                    dependency=dependency,
                )
            )
            dependency = min(255, dependency + 1)
            next_weight_addr += byte_count
            cells_remaining -= cells
            weight_elements_remaining -= weight_elements
            window_idx += 1
            if len(descriptors) >= cfg.max_descriptors:
                break
        if len(descriptors) >= cfg.max_descriptors:
            break

    truncated = len(descriptors) >= cfg.max_descriptors
    return StreamPlan(
        model_name=manifest.model_name,
        execution_mode=mode,
        fabric=cfg.as_dict(),
        summary={
            "total_static_programmed_cells": total_cells,
            "die_cell_capacity": cfg.die_cell_capacity,
            "total_cell_capacity": cfg.total_cell_capacity,
            "windows": len(windows),
            "descriptors": len(descriptors),
            "truncated_descriptors": truncated,
            "requires_dma": mode == "peripheral_streaming" or len(windows) > 1,
            "requires_peer_die_protocol": mode == "multi_die_resident" or cfg.die_count > 1,
            "resident_feasible": mode != "resident_infeasible",
            "noc_coalescing_tiles": max(1, int(cfg.noc_coalescing_tiles)),
            "reduction_placement": str(cfg.reduction_placement),
            "cells_per_window": cfg.cells_per_window,
            "descriptor_bytes": len(descriptors) * DESCRIPTOR_BYTES,
        },
        windows=windows,
        descriptors=descriptors[: cfg.max_descriptors],
    )


def write_stream_plan(
    manifest: ProgramManifest,
    out_json: str | Path,
    out_csv: str | Path | None = None,
    fabric: FabricConfig | None = None,
) -> StreamPlan:
    plan = build_stream_plan(manifest, fabric)
    plan.write_json(out_json)
    if out_csv is not None:
        plan.write_csv(out_csv)
    return plan
