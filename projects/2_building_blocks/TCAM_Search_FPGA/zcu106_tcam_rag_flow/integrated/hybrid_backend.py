from __future__ import annotations

from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
from typing import Any

import numpy as np

from .arbiter import SharedPEArbiter


class Layer0HardwareRemainingNeuromorphicBackend:
    """Route first Conv2D to FPGA YOLO PEs and later Conv2Ds to the emulator."""

    max_batch = 32
    diagnostics_key = "hybrid_dispatch"

    def __init__(
        self,
        hardware_backend: Any,
        neuromorphic_backend: Any,
        *,
        layer0_op_index: int = 2,
        arbiter: SharedPEArbiter | None = None,
    ) -> None:
        self.hardware_backend = hardware_backend
        self.neuromorphic_backend = neuromorphic_backend
        self.layer0_op_index = int(layer0_op_index)
        self.arbiter = arbiter or SharedPEArbiter()
        self.report_name = "uninitialized"
        self.dispatch_counts: Counter[str] = Counter()
        self.dispatched_ops: list[dict[str, Any]] = []

    @staticmethod
    def _op_index(program: Any) -> int:
        source = dict(program.metadata.get("source", {}))
        if "op_index" not in source:
            raise ValueError(f"Program {program.name!r} has no source op_index.")
        return int(source["op_index"])

    @staticmethod
    def _run_tiled(program: Any, inputs: np.ndarray, backend: Any) -> np.ndarray:
        vectors = np.asarray(inputs, dtype=np.int8)
        output = np.zeros((program.m_dim, program.n_dim), dtype=np.int32)
        batch_limit = int(backend.max_batch)
        for tile in program.tiles:
            weights = np.asarray(program.constants[tile.weight_key], dtype=np.int8)
            for start in range(0, program.m_dim, batch_limit):
                stop = min(start + batch_limit, program.m_dim)
                partial = backend.run_batch(
                    vectors[start:stop, tile.k_start : tile.k_start + tile.rows],
                    weights,
                )
                output[start:stop, tile.n_start : tile.n_start + tile.cols] += np.asarray(
                    partial, dtype=np.int32
                )
        return output

    @staticmethod
    def _backend_engine_count(backend: Any) -> int | None:
        if hasattr(backend, "pe_count"):
            return int(getattr(backend, "pe_count"))
        engines = getattr(backend, "engines", None)
        if engines is not None:
            return len(tuple(engines))
        wrapped = getattr(backend, "driver", None)
        engines = getattr(wrapped, "engines", None)
        if engines is not None:
            return len(tuple(engines))
        return None

    def run_program(self, program: Any, inputs: np.ndarray) -> np.ndarray:
        op_index = self._op_index(program)
        if op_index == self.layer0_op_index:
            engine_count = self._backend_engine_count(self.hardware_backend)
            route = (
                f"fpga_{engine_count}_pe_layer0"
                if engine_count is not None
                else "fpga_pe_layer0"
            )
            self.report_name = route
            with self.arbiter.reserve("yolo_layer0"):
                output = self._run_tiled(program, inputs, self.hardware_backend)
        else:
            route = "neuromorphic_emulator"
            self.report_name = route
            run_program = getattr(self.neuromorphic_backend, "run_program", None)
            if callable(run_program):
                output = run_program(program, inputs)
            else:
                output = self._run_tiled(program, inputs, self.neuromorphic_backend)
        self.dispatch_counts[route] += 1
        self.dispatched_ops.append({"op_index": op_index, "route": route})
        return np.asarray(output, dtype=np.int32)

    def run_batch(self, vectors: np.ndarray, weights: np.ndarray) -> np.ndarray:
        raise RuntimeError("Layer-routed execution requires the X2 run_program contract.")

    def diagnostics(self) -> dict[str, Any]:
        result: dict[str, Any] = {
            "block_name": (
                "Hybrid Runtime Scheduler + TCAM Hit/Miss + Image Fingerprint"
            ),
            "policy": (
                "TCAM hit/miss before MLIR execution; Engine 0/1 keep TCAM "
                "resident and Engines 2/3/4/5 run YOLO layers on FPGA on miss. Remaining "
                "Conv2D layers run on the neuromorphic emulator/software path."
            ),
            "scheduler_responsibilities": {
                "fingerprint": "generate image fingerprint before inference dispatch",
                "tcam_lookup": "batch queries and route them to Engine 0/1 first",
                "hit_path": "return cached detections and skip MLIR",
                "miss_path": "run YOLO/MLIR and update key -> chunk ID after miss",
                "miss_latency_policy": (
                    "avoid erasing TCAM cells by keeping Engine 0/1 resident and "
                    "sending YOLO Layers 0/1/2/3 to Engines 2/3/4/5"
                ),
                "tcam_capacity_path": (
                    "search additional TCAM banks/pages with Engine 0/1; the "
                    "neuromorphic emulator is reserved for remaining YOLO layers"
                ),
            },
            "engine_partition": {
                "Engine 0": "dedicated 32x32 TCAM slots 0-15 with TDC enabled",
                "Engine 1": "dedicated 32x32 TCAM slots 16-31 with TDC enabled",
                "Engine 2": "dedicated YOLO layer on FPGA on TCAM miss, 4 lanes, TDC disabled",
                "Engine 3": "dedicated YOLO Layer 1 on TCAM miss, 4 lanes, TDC disabled",
                "Engine 4": "dedicated YOLO Layer 2 on TCAM miss, 4 lanes, TDC disabled",
                "Engine 5": "dedicated YOLO Layer 3 on TCAM miss, 4 lanes, TDC disabled",
                "encoding": "q/qbar rows, A/B columns, B-A + TDC on Engine 0/1",
                "tcam_internal_lanes_per_engine": 8,
                "yolo_layer0_internal_lanes_per_engine": 4,
            },
            "layer0_op_index": self.layer0_op_index,
            "dispatch_counts": dict(sorted(self.dispatch_counts.items())),
            "dispatched_ops": list(self.dispatched_ops),
            "pe_arbiter": vars(self.arbiter.snapshot()),
            "unsupported_on_neuromorphic": (
                "Depthwise and digital graph operators remain on CPU in the XForge executor."
            ),
        }
        for name, backend in (
            ("hardware", self.hardware_backend),
            ("neuromorphic", self.neuromorphic_backend),
        ):
            diagnostics = getattr(backend, "diagnostics", None)
            if callable(diagnostics):
                result[name] = diagnostics()
        return result


class EarlyLayerMappedHardwareBackend:
    """Route early MLIR Conv2D ops to channel-tiled FPGA engines."""

    max_batch = 32
    diagnostics_key = "early_layer_mapped_hybrid_dispatch"

    def __init__(
        self,
        hardware_backends_by_engine: dict[int, Any],
        neuromorphic_backend: Any,
        *,
        fpga_op_engine_map: dict[int, int] | None = None,
        op_layer_map: dict[int, int] | None = None,
        channel_tile_size: int = 16,
        channel_tile_engines: tuple[int, ...] = (2, 3, 4, 5),
        embedding_directory: str | Path | None = None,
        arbiter: SharedPEArbiter | None = None,
    ) -> None:
        self.hardware_backends_by_engine = {
            int(engine_id): backend
            for engine_id, backend in hardware_backends_by_engine.items()
        }
        self.neuromorphic_backend = neuromorphic_backend
        self.fpga_op_engine_map = {
            int(op_index): int(engine_id)
            for op_index, engine_id in (
                fpga_op_engine_map or {2: 2, 6: 3, 9: 4, 13: 5}
            ).items()
        }
        self.op_layer_map = {
            int(op_index): int(layer_id)
            for op_index, layer_id in (
                op_layer_map or {2: 0, 6: 1, 9: 2, 13: 3}
            ).items()
        }
        self.channel_tile_size = int(channel_tile_size)
        if self.channel_tile_size < 1:
            raise ValueError("channel_tile_size must be positive.")
        self.channel_tile_engines = tuple(int(engine) for engine in channel_tile_engines)
        if not self.channel_tile_engines:
            raise ValueError("At least one channel-tile engine is required.")
        missing = sorted(
            {
                engine_id
                for engine_id in set(self.fpga_op_engine_map.values())
                | set(self.channel_tile_engines)
                if engine_id not in self.hardware_backends_by_engine
            }
        )
        if missing:
            raise ValueError(f"No hardware backend was provided for engines {missing}.")

        self.embedding_directory = (
            Path(embedding_directory) if embedding_directory is not None else None
        )
        self.arbiter = arbiter or SharedPEArbiter()
        self.report_name = "uninitialized"
        self.dispatch_counts: Counter[str] = Counter()
        self.dispatched_ops: list[dict[str, Any]] = []
        self.embedding_records: list[dict[str, Any]] = []
        self.channel_tile_records: list[dict[str, Any]] = []

    @staticmethod
    def _op_index(program: Any) -> int:
        return Layer0HardwareRemainingNeuromorphicBackend._op_index(program)

    @staticmethod
    def _backend_engine_count(backend: Any) -> int | None:
        return Layer0HardwareRemainingNeuromorphicBackend._backend_engine_count(backend)

    @staticmethod
    def _engine_batch_limit(backend: Any) -> int:
        backend_limit = int(getattr(backend, "max_batch", 32))
        engine_count = EarlyLayerMappedHardwareBackend._backend_engine_count(backend)
        if engine_count == 1:
            return min(backend_limit, 32)
        return min(backend_limit, 32 * int(engine_count or 1))

    @staticmethod
    def _run_tiled(program: Any, inputs: np.ndarray, backend: Any) -> np.ndarray:
        vectors = np.asarray(inputs, dtype=np.int8)
        output = np.zeros((int(program.m_dim), int(program.n_dim)), dtype=np.int32)
        batch_limit = EarlyLayerMappedHardwareBackend._engine_batch_limit(backend)
        for tile in program.tiles:
            weights = np.asarray(program.constants[tile.weight_key], dtype=np.int8)
            for start in range(0, int(program.m_dim), batch_limit):
                stop = min(start + batch_limit, int(program.m_dim))
                partial = backend.run_batch(
                    vectors[start:stop, tile.k_start : tile.k_start + tile.rows],
                    weights,
                )
                output[start:stop, tile.n_start : tile.n_start + tile.cols] += np.asarray(
                    partial, dtype=np.int32
                )
        return output

    @staticmethod
    def build_channel_tile_schedule(
        *,
        op_index: int,
        layer_id: int,
        output_channels: int,
        channel_tile_size: int = 16,
        engine_ids: tuple[int, ...] = (2, 3, 4, 5),
    ) -> list[dict[str, Any]]:
        if output_channels < 1:
            raise ValueError("output_channels must be positive.")
        if channel_tile_size < 1:
            raise ValueError("channel_tile_size must be positive.")
        if not engine_ids:
            raise ValueError("engine_ids must not be empty.")

        schedule: list[dict[str, Any]] = []
        tile_id = 0
        for oc_start in range(0, output_channels, channel_tile_size):
            engine_slot = tile_id % len(engine_ids)
            time_step = tile_id // len(engine_ids)
            oc_end = min(oc_start + channel_tile_size, output_channels)
            schedule.append(
                {
                    "tile_id": tile_id,
                    "op_index": int(op_index),
                    "logical_layer": int(layer_id),
                    "time_step": int(time_step),
                    "engine": int(engine_ids[engine_slot]),
                    "oc_start": int(oc_start),
                    "oc_end": int(oc_end),
                    "channel_count": int(oc_end - oc_start),
                    "output_offset": int(oc_start),
                }
            )
            tile_id += 1
        return schedule

    @staticmethod
    def _run_channel_slice(
        program: Any,
        inputs: np.ndarray,
        backend: Any,
        *,
        oc_start: int,
        oc_end: int,
    ) -> np.ndarray:
        vectors = np.asarray(inputs, dtype=np.int8)
        width = int(oc_end) - int(oc_start)
        output = np.zeros((int(program.m_dim), width), dtype=np.int32)
        batch_limit = EarlyLayerMappedHardwareBackend._engine_batch_limit(backend)
        for tile in program.tiles:
            tile_n_start = int(tile.n_start)
            tile_n_end = tile_n_start + int(tile.cols)
            slice_start = max(int(oc_start), tile_n_start)
            slice_end = min(int(oc_end), tile_n_end)
            if slice_start >= slice_end:
                continue

            local_start = slice_start - tile_n_start
            local_end = slice_end - tile_n_start
            output_start = slice_start - int(oc_start)
            output_end = slice_end - int(oc_start)
            weights = np.asarray(
                program.constants[tile.weight_key],
                dtype=np.int8,
            )[:, local_start:local_end]

            for start in range(0, int(program.m_dim), batch_limit):
                stop = min(start + batch_limit, int(program.m_dim))
                partial = backend.run_batch(
                    vectors[start:stop, int(tile.k_start) : int(tile.k_start) + int(tile.rows)],
                    weights,
                )
                output[start:stop, output_start:output_end] += np.asarray(
                    partial,
                    dtype=np.int32,
                )
        return output

    def _channel_schedule(self, program: Any, op_index: int, layer_id: int) -> list[dict[str, Any]]:
        metadata_schedule = (
            dict(getattr(program, "metadata", {}))
            .get("spatial_temporal_channel_schedule", {})
        )
        tiles = metadata_schedule.get("tiles") if isinstance(metadata_schedule, dict) else None
        if tiles:
            return [dict(tile) for tile in tiles]
        return self.build_channel_tile_schedule(
            op_index=op_index,
            layer_id=layer_id,
            output_channels=int(program.n_dim),
            channel_tile_size=self.channel_tile_size,
            engine_ids=self.channel_tile_engines,
        )

    def _run_channel_tiled(
        self,
        program: Any,
        inputs: np.ndarray,
        *,
        op_index: int,
        layer_id: int,
    ) -> tuple[np.ndarray, list[dict[str, Any]]]:
        output = np.zeros((int(program.m_dim), int(program.n_dim)), dtype=np.int32)
        schedule = self._channel_schedule(program, op_index, layer_id)
        schedule_by_step: dict[int, list[dict[str, Any]]] = {}
        for tile in schedule:
            schedule_by_step.setdefault(int(tile["time_step"]), []).append(tile)

        executed: list[dict[str, Any]] = []
        for time_step in sorted(schedule_by_step):
            tiles = sorted(
                schedule_by_step[time_step],
                key=lambda item: (int(item["engine"]), int(item["oc_start"])),
            )
            with ThreadPoolExecutor(max_workers=len(tiles)) as pool:
                futures = []
                for tile in tiles:
                    engine_id = int(tile["engine"])
                    backend = self.hardware_backends_by_engine[engine_id]
                    futures.append(
                        (
                            tile,
                            pool.submit(
                                self._run_channel_slice,
                                program,
                                inputs,
                                backend,
                                oc_start=int(tile["oc_start"]),
                                oc_end=int(tile["oc_end"]),
                            ),
                        )
                    )
                for tile, future in futures:
                    partial = future.result()
                    oc_start = int(tile["oc_start"])
                    oc_end = int(tile["oc_end"])
                    output[:, oc_start:oc_end] = partial
                    executed.append(
                        {
                            **dict(tile),
                            "shape": list(partial.shape),
                            "dtype": str(partial.dtype),
                        }
                    )
        return output, executed

    def set_embedding_directory(self, directory: str | Path) -> None:
        self.embedding_directory = Path(directory)
        self.embedding_directory.mkdir(parents=True, exist_ok=True)
        self.embedding_records.clear()

    def _store_fpga_layer_embedding(
        self,
        *,
        op_index: int,
        layer_id: int,
        tensor: np.ndarray,
        program: Any,
        engine_id: int | None = None,
        tile_schedule: list[dict[str, Any]] | None = None,
    ) -> None:
        tensor_i32 = np.asarray(tensor, dtype=np.int32)
        write_layer = getattr(self.neuromorphic_backend, "write_layer_tensor", None)
        if callable(write_layer):
            source = (
                f"fpga_engine_{engine_id}"
                if engine_id is not None
                else "fpga_channel_tiled_engines_2_3_4_5"
            )
            write_layer(layer_id, tensor_i32, source=source)

        if self.embedding_directory is None:
            return

        self.embedding_directory.mkdir(parents=True, exist_ok=True)
        stem = (
            f"layer_{layer_id}_op_{op_index}_engine_{engine_id}"
            if engine_id is not None
            else f"layer_{layer_id}_op_{op_index}_channel_tiled"
        )
        tensor_path = self.embedding_directory / f"{stem}.npy"
        meta_path = self.embedding_directory / f"{stem}.json"
        np.save(tensor_path, tensor_i32)
        record = {
            "layer_id": int(layer_id),
            "op_index": int(op_index),
            "physical_engine": int(engine_id) if engine_id is not None else None,
            "physical_engines": sorted(
                {
                    int(tile["engine"])
                    for tile in (tile_schedule or [])
                }
            )
            if tile_schedule
            else ([int(engine_id)] if engine_id is not None else []),
            "tensor_file": str(tensor_path),
            "shape": list(tensor_i32.shape),
            "dtype": str(tensor_i32.dtype),
            "program_name": str(getattr(program, "name", f"op_{op_index}")),
            "tile_count": len(list(getattr(program, "tiles", []))),
            "channel_tile_schedule": list(tile_schedule or []),
            "storage": "fpga_layer_embeddings plus mirrored DDR/L2_Buffer",
        }
        meta_path.write_text(json.dumps(record, indent=2), encoding="utf-8")
        record["metadata_file"] = str(meta_path)
        self.embedding_records.append(record)

    def run_program(self, program: Any, inputs: np.ndarray) -> np.ndarray:
        op_index = self._op_index(program)
        if op_index in self.fpga_op_engine_map:
            layer_id = self.op_layer_map.get(op_index, op_index)
            route = f"fpga_channel_tiled_layer_{layer_id}"
            self.report_name = route
            with self.arbiter.reserve(f"yolo_layer_{layer_id}_channel_tiles"):
                output, tile_schedule = self._run_channel_tiled(
                    program,
                    inputs,
                    op_index=op_index,
                    layer_id=layer_id,
                )
            self.channel_tile_records.extend(tile_schedule)
            self._store_fpga_layer_embedding(
                op_index=op_index,
                layer_id=layer_id,
                tensor=output,
                program=program,
                tile_schedule=tile_schedule,
            )
        else:
            route = "neuromorphic_emulator"
            self.report_name = route
            run_program = getattr(self.neuromorphic_backend, "run_program", None)
            if callable(run_program):
                output = run_program(program, inputs)
            else:
                output = self._run_tiled(program, inputs, self.neuromorphic_backend)

        self.dispatch_counts[route] += 1
        self.dispatched_ops.append(
            {
                "op_index": int(op_index),
                "logical_layer": self.op_layer_map.get(op_index),
                "route": route,
                "physical_engine": self.fpga_op_engine_map.get(op_index),
                "channel_tiled": op_index in self.fpga_op_engine_map,
            }
        )
        return np.asarray(output, dtype=np.int32)

    def run_batch(self, vectors: np.ndarray, weights: np.ndarray) -> np.ndarray:
        raise RuntimeError("Layer-routed execution requires the X2 run_program contract.")

    def embedding_manifest(self) -> list[dict[str, Any]]:
        return list(self.embedding_records)

    def diagnostics(self) -> dict[str, Any]:
        result: dict[str, Any] = {
            "block_name": (
                "Hybrid Runtime Scheduler + TCAM Hit/Miss + Image Fingerprint"
            ),
            "policy": (
                "TCAM hit/miss before MLIR execution; Engine 0/1 keep TCAM "
                "resident. On a miss, the first four YOLO Conv2D layers are "
                "split into output-channel tiles, scheduled temporally, and run "
                "on Engine 2/3/4/5. Remaining YOLO layers run on the legacy "
                "neuromorphic emulator/CPU."
            ),
            "scheduler_responsibilities": {
                "fingerprint": "generate image fingerprint before inference dispatch",
                "tcam_lookup": "batch queries and route them to Engine 0/1 first",
                "hit_path": "return cached detections and skip MLIR",
                "miss_path": (
                    "run early YOLO/MLIR layers as output-channel tiles on "
                    "Engine 2/3/4/5, store each merged FPGA layer tensor in "
                    "DDR/L2, then run remaining layers on the legacy "
                    "neuromorphic emulator/CPU"
                ),
                "embedding_storage": (
                    "write layer_0..layer_3 INT32 tensors to fpga_layer_embeddings "
                    "and mirror them into L2_Buffer"
                ),
                "miss_latency_policy": (
                    "avoid erasing TCAM cells by keeping Engine 0/1 resident and "
                    "using Engine 2/3/4/5 for YOLO miss-path layers"
                ),
            },
            "engine_partition": {
                "Engine 0": "dedicated 32x32 TCAM slots 0-15 with TDC enabled",
                "Engine 1": "dedicated 32x32 TCAM slots 16-31 with TDC enabled",
                "Engine 2": "YOLO output-channel tile engine on TCAM miss, TDC disabled",
                "Engine 3": "YOLO output-channel tile engine on TCAM miss, TDC disabled",
                "Engine 4": "YOLO output-channel tile engine on TCAM miss, TDC disabled",
                "Engine 5": "YOLO output-channel tile engine on TCAM miss, TDC disabled",
                "encoding": "q/qbar rows, A/B columns, B-A + TDC on Engine 0/1",
            },
            "spatial_temporal_channel_policy": {
                "fpga_layers": "logical layer 0/1/2/3 only",
                "channel_tile_size": self.channel_tile_size,
                "engines": list(self.channel_tile_engines),
                "spatial_rule": "engine = engines[(tile_id mod engine_count)]",
                "temporal_rule": "time_step = floor(tile_id / engine_count)",
                "remaining_layers": "legacy neuromorphic emulator/CPU",
            },
            "fpga_op_engine_map": {
                str(op_index): engine_id
                for op_index, engine_id in sorted(self.fpga_op_engine_map.items())
            },
            "op_layer_map": {
                str(op_index): layer_id
                for op_index, layer_id in sorted(self.op_layer_map.items())
            },
            "channel_tile_records": list(self.channel_tile_records),
            "dispatch_counts": dict(sorted(self.dispatch_counts.items())),
            "dispatched_ops": list(self.dispatched_ops),
            "fpga_layer_embeddings": self.embedding_manifest(),
            "pe_arbiter": vars(self.arbiter.snapshot()),
            "unsupported_on_neuromorphic": (
                "Depthwise and digital graph operators remain on CPU in the XForge executor."
            ),
        }
        for engine_id, backend in sorted(self.hardware_backends_by_engine.items()):
            diagnostics = getattr(backend, "diagnostics", None)
            if callable(diagnostics):
                result[f"hardware_engine_{engine_id}"] = diagnostics()
        diagnostics = getattr(self.neuromorphic_backend, "diagnostics", None)
        if callable(diagnostics):
            result["neuromorphic"] = diagnostics()
        return result
