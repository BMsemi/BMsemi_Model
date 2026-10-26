from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Any

import numpy as np

from .tflite_ops import TensorMeta


@dataclass(frozen=True)
class TensorAllocation:
    tensor_index: int
    name: str
    offset: int
    size: int
    allocated_size: int
    first_op: int
    last_op: int
    shape: tuple[int, ...]
    dtype: str


@dataclass(frozen=True)
class Partition:
    first_op: int
    last_op: int
    backend: str
    operator_count: int


def _align(value: int, alignment: int) -> int:
    return ((int(value) + alignment - 1) // alignment) * alignment


def _merge_blocks(blocks: list[tuple[int, int]]) -> list[tuple[int, int]]:
    merged: list[tuple[int, int]] = []
    for offset, size in sorted(blocks):
        if size <= 0:
            continue
        if merged and merged[-1][0] + merged[-1][1] == offset:
            previous_offset, previous_size = merged[-1]
            merged[-1] = (previous_offset, previous_size + size)
        else:
            merged.append((offset, size))
    return merged


def build_memory_plan(
    ops: list[dict[str, Any]],
    tensors: dict[int, TensorMeta],
    constant_indices: set[int],
    output_indices: set[int],
    *,
    alignment: int = 64,
) -> dict[str, Any]:
    """Build an aligned first-fit activation arena from tensor lifetimes."""

    if alignment < 1 or alignment & (alignment - 1):
        raise ValueError("Arena alignment must be a positive power of two.")
    producer: dict[int, int] = {}
    last_use: dict[int, int] = {}
    for ordinal, op in enumerate(ops):
        for tensor_index in op.get("outputs", []):
            producer[int(tensor_index)] = ordinal
        for tensor_index in op.get("inputs", []):
            index = int(tensor_index)
            if index >= 0:
                last_use[index] = ordinal
    graph_end = len(ops)
    for index in output_indices:
        last_use[int(index)] = graph_end

    intervals: list[tuple[int, int, int, int]] = []
    for index, meta in tensors.items():
        if index in constant_indices:
            continue
        if index not in producer and index not in last_use:
            continue
        first = producer.get(index, 0)
        last = last_use.get(index, first)
        size = max(meta.nbytes, 1)
        intervals.append((first, last, index, size))
    intervals.sort(key=lambda item: (item[0], -item[3], item[2]))

    arena_end = 0
    active: list[tuple[int, int, int, int]] = []
    free_blocks: list[tuple[int, int]] = []
    allocations: list[TensorAllocation] = []
    for first, last, index, size in intervals:
        still_active: list[tuple[int, int, int, int]] = []
        for active_last, active_offset, active_size, active_index in active:
            if active_last < first:
                free_blocks.append((active_offset, active_size))
            else:
                still_active.append(
                    (active_last, active_offset, active_size, active_index)
                )
        active = still_active
        free_blocks = _merge_blocks(free_blocks)

        allocated_size = _align(size, alignment)
        offset = -1
        for block_number, (block_offset, block_size) in enumerate(free_blocks):
            if block_size >= allocated_size:
                offset = block_offset
                remainder = block_size - allocated_size
                if remainder:
                    free_blocks[block_number] = (
                        block_offset + allocated_size,
                        remainder,
                    )
                else:
                    del free_blocks[block_number]
                break
        if offset < 0:
            offset = _align(arena_end, alignment)
            arena_end = offset + allocated_size

        meta = tensors[index]
        allocations.append(
            TensorAllocation(
                tensor_index=index,
                name=meta.name,
                offset=offset,
                size=size,
                allocated_size=allocated_size,
                first_op=first,
                last_op=last,
                shape=meta.shape,
                dtype=str(meta.dtype),
            )
        )
        active.append((last, offset, allocated_size, index))

    naive_bytes = sum(item[3] for item in intervals)
    return {
        "alignment_bytes": alignment,
        "arena_bytes": arena_end,
        "naive_activation_bytes": naive_bytes,
        "bytes_saved": naive_bytes - arena_end,
        "reduction_percent": (
            100.0 * (naive_bytes - arena_end) / naive_bytes
            if naive_bytes
            else 0.0
        ),
        "allocation_count": len(allocations),
        "allocations": [
            {
                **asdict(allocation),
                "shape": list(allocation.shape),
            }
            for allocation in sorted(
                allocations,
                key=lambda item: item.tensor_index,
            )
        ],
    }


def build_partition_plan(
    ops: list[dict[str, Any]],
    compiled_conv_indices: set[int],
) -> dict[str, Any]:
    assignments: list[dict[str, Any]] = []
    for op in ops:
        index = int(op["index"])
        name = str(op["op_name"])
        if name == "CONV_2D" and index in compiled_conv_indices:
            backend = "x2"
        elif name == "DEPTHWISE_CONV_2D":
            backend = "cpu_depthwise"
        else:
            backend = "cpu_digital"
        assignments.append(
            {
                "op_index": index,
                "op_name": name,
                "backend": backend,
            }
        )

    partitions: list[Partition] = []
    for assignment in assignments:
        if partitions and partitions[-1].backend == assignment["backend"]:
            previous = partitions[-1]
            partitions[-1] = Partition(
                first_op=previous.first_op,
                last_op=int(assignment["op_index"]),
                backend=previous.backend,
                operator_count=previous.operator_count + 1,
            )
        else:
            partitions.append(
                Partition(
                    first_op=int(assignment["op_index"]),
                    last_op=int(assignment["op_index"]),
                    backend=str(assignment["backend"]),
                    operator_count=1,
                )
            )

    counts = Counter(item["backend"] for item in assignments)
    return {
        "operator_count": len(assignments),
        "backend_counts": dict(sorted(counts.items())),
        "partition_count": len(partitions),
        "partitions": [asdict(partition) for partition in partitions],
        "assignments": assignments,
    }


def plan_tflite_model(
    model_path: str | Path,
    artifact_directory: str | Path,
    *,
    alignment: int = 64,
) -> dict[str, Any]:
    import ai_edge_litert.interpreter as litert

    kwargs: dict[str, Any] = {
        "model_path": str(Path(model_path).resolve()),
        "experimental_preserve_all_tensors": True,
    }
    resolver = getattr(litert, "OpResolverType", None)
    if resolver is not None:
        kwargs["experimental_op_resolver_type"] = (
            resolver.BUILTIN_WITHOUT_DEFAULT_DELEGATES
        )
    interpreter = litert.Interpreter(**kwargs)
    interpreter.allocate_tensors()
    ops = list(interpreter._get_ops_details())  # noqa: SLF001
    tensors = {
        int(detail["index"]): TensorMeta.from_detail(detail)
        for detail in interpreter.get_tensor_details()
    }
    produced = {
        int(index)
        for op in ops
        for index in op.get("outputs", [])
        if int(index) >= 0
    }
    input_indices = {
        int(detail["index"]) for detail in interpreter.get_input_details()
    }
    constant_indices = set(tensors) - produced - input_indices
    output_indices = {
        int(detail["index"]) for detail in interpreter.get_output_details()
    }

    manifest_path = Path(artifact_directory) / "model_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    compiled = {
        int(item["op_index"]) for item in manifest["compiled_programs"]
    }
    return {
        "schema_version": 1,
        "model": str(Path(model_path).resolve()),
        "artifacts": str(Path(artifact_directory).resolve()),
        "partition": build_partition_plan(ops, compiled),
        "memory": build_memory_plan(
            ops,
            tensors,
            constant_indices,
            output_indices,
            alignment=alignment,
        ),
        "constant_tensor_count": len(constant_indices),
        "constant_bytes": sum(tensors[index].nbytes for index in constant_indices),
    }

