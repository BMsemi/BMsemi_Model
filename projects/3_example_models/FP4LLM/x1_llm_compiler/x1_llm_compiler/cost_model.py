"""Analytical latency and energy models for compiled X1 execution plans.

The model consumes the existing :class:`ProgramManifest` and
:class:`StreamPlan`.  Its constants are deliberately configurable: the
defaults are a useful architectural estimate, not a substitute for measured
RTL, FPGA, or silicon data.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from math import ceil
from pathlib import Path

from .fabric import StreamPlan
from .mapping import ceil_div
from .planner import LayerPlan, ProgramManifest


def _positive(value: int | float, name: str) -> int | float:
    if isinstance(value, bool) or float(value) <= 0:
        raise ValueError(f"{name} must be positive")
    return value


@dataclass(frozen=True)
class CostModelConfig:
    """Hardware cost assumptions used by both engines.

    Time values are in cycles. Energy values are in picojoules. Replace these
    defaults with technology-specific measurements before using absolute
    numbers for a design decision.
    """

    programming_word_bytes: int = 4
    flit_bits: int = 128
    noc_header_bits: int = 64
    noc_flits_per_cycle: int = 1
    noc_hops: int = 1
    noc_startup_cycles: int = 2

    dma_setup_cycles: int = 16
    dma_burst_setup_cycles: int = 4
    dma_burst_bytes: int = 4096
    dma_bytes_per_cycle: int = 16

    x1_program_setup_cycles: int = 8
    x1_program_cells_per_cycle: int = 4

    pe_compute_setup_cycles: int = 1
    pe_macs_per_cycle_per_macro: int = 1024
    reduction_setup_cycles: int = 2
    reduction_values_per_cycle: int = 32
    sar_tdc_setup_cycles: int = 1
    sar_tdc_values_per_cycle: int = 32

    memory_read_energy_pj_per_byte: float = 0.20
    memory_write_energy_pj_per_byte: float = 0.25
    x1_program_energy_pj_per_cell: float = 1.00
    compute_read_energy_pj_per_mac: float = 0.05
    reduction_energy_pj_per_value: float = 0.01
    adc_energy_pj_per_value: float = 0.20
    noc_energy_pj_per_bit: float = 0.001

    # A spatial mapping can expose extra parallelism only if the target
    # hardware supports it. Keep this at 1.0 until that gain is calibrated.
    spatial_map_speedup: float = 1.0

    def __post_init__(self) -> None:
        integer_fields = (
            "programming_word_bytes",
            "flit_bits",
            "noc_header_bits",
            "noc_flits_per_cycle",
            "noc_hops",
            "noc_startup_cycles",
            "dma_setup_cycles",
            "dma_burst_setup_cycles",
            "dma_burst_bytes",
            "dma_bytes_per_cycle",
            "x1_program_setup_cycles",
            "x1_program_cells_per_cycle",
            "pe_compute_setup_cycles",
            "pe_macs_per_cycle_per_macro",
            "reduction_setup_cycles",
            "reduction_values_per_cycle",
            "sar_tdc_setup_cycles",
            "sar_tdc_values_per_cycle",
        )
        for name in integer_fields:
            _positive(getattr(self, name), name)
        for name in (
            "memory_read_energy_pj_per_byte",
            "memory_write_energy_pj_per_byte",
            "x1_program_energy_pj_per_cell",
            "compute_read_energy_pj_per_mac",
            "reduction_energy_pj_per_value",
            "adc_energy_pj_per_value",
            "noc_energy_pj_per_bit",
            "spatial_map_speedup",
        ):
            _positive(getattr(self, name), name)

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class LatencyBreakdown:
    dma_load_cycles: int
    x1_programming_cycles: int
    pe_compute_cycles: int
    reduction_cycles: int
    sar_tdc_conversion_cycles: int
    noc_transfer_cycles: int

    @property
    def total_cycles(self) -> int:
        return sum(
            (
                self.dma_load_cycles,
                self.x1_programming_cycles,
                self.pe_compute_cycles,
                self.reduction_cycles,
                self.sar_tdc_conversion_cycles,
                self.noc_transfer_cycles,
            )
        )

    def as_dict(self) -> dict[str, object]:
        out = asdict(self)
        out["total_cycles"] = self.total_cycles
        return out


@dataclass(frozen=True)
class EnergyBreakdown:
    memory_read_pj: float
    memory_write_pj: float
    x1_programming_pj: float
    compute_read_pj: float
    reduction_pj: float
    adc_sar_tdc_pj: float
    noc_pj: float

    @property
    def total_pj(self) -> float:
        return sum(
            (
                self.memory_read_pj,
                self.memory_write_pj,
                self.x1_programming_pj,
                self.compute_read_pj,
                self.reduction_pj,
                self.adc_sar_tdc_pj,
                self.noc_pj,
            )
        )

    def as_dict(self) -> dict[str, object]:
        out = asdict(self)
        out["total_pj"] = self.total_pj
        return out


@dataclass(frozen=True)
class CostEstimate:
    latency: LatencyBreakdown
    energy: EnergyBreakdown
    counters: dict[str, int | float] = field(default_factory=dict)

    def as_dict(self) -> dict[str, object]:
        return {
            "latency": self.latency.as_dict(),
            "energy": self.energy.as_dict(),
            "counters": dict(self.counters),
        }

    def write_json(self, path: str | Path) -> None:
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(self.as_dict(), indent=2) + "\n", encoding="utf-8")


@dataclass
class _PlanCounters:
    weight_cells: int = 0
    weight_windows: int = 0
    weight_bytes: int = 0
    mac_operations: int = 0
    compute_groups: int = 0
    compute_cycles: int = 0
    reduction_values: int = 0
    reduction_groups: int = 0
    converted_values: int = 0
    noc_values: int = 0
    noc_payload_bits: int = 0
    noc_packets: int = 0
    noc_flits: int = 0
    noc_wire_bits: int = 0


def _unique_static_layers(manifest: ProgramManifest) -> list[LayerPlan]:
    layers: list[LayerPlan] = []
    seen: set[tuple[str, str, int, int]] = set()
    for layer_plan in manifest.layers:
        layer = layer_plan.layer
        if not bool(layer.get("static_weights", False)):
            continue
        key = (
            str(layer["name"]),
            str(layer["role"]),
            int(layer["in_dim"]),
            int(layer["out_dim"]),
        )
        if key not in seen:
            seen.add(key)
            layers.append(layer_plan)
    return layers


def _transfer_cycles(
    byte_count: int,
    *,
    setup_cycles: int,
    burst_setup_cycles: int,
    burst_bytes: int,
    bytes_per_cycle: int,
) -> int:
    if byte_count <= 0:
        return 0
    bursts = ceil_div(byte_count, burst_bytes)
    return (
        int(setup_cycles)
        + ceil_div(byte_count, bytes_per_cycle)
        + max(0, bursts - 1) * int(burst_setup_cycles)
    )


def _map_speedup(layer: LayerPlan, cfg: CostModelConfig) -> float:
    layer_data = layer.layer
    spatial_axes = sum(
        str(layer_data.get(axis, "")).lower() == "spatialmap"
        for axis in ("k_map", "c_map", "y_map")
    )
    return max(1.0, float(cfg.spatial_map_speedup) ** spatial_axes)


def _analyze(manifest: ProgramManifest, stream_plan: StreamPlan, cfg: CostModelConfig) -> _PlanCounters:
    counters = _PlanCounters()
    compiler = manifest.compiler_config
    fabric = stream_plan.fabric
    fp4_rows = int(compiler.get("fp4llm_tile_rows", 64))
    fp4_cols = int(compiler.get("fp4llm_tile_cols", 32))
    x1_rows = int(compiler.get("x1_macro_rows", fabric.get("x1_rows", 32)))
    cluster = int(fabric.get("x1_macros_per_endpoint", compiler.get("x1_macros_per_endpoint", 4)))
    cluster = max(1, cluster)
    coalescing = max(1, int(fabric.get("noc_coalescing_tiles", 1)))
    reduction_placement = str(fabric.get("reduction_placement", "local")).lower()
    if reduction_placement not in {"local", "destination"}:
        raise ValueError("reduction_placement must be 'local' or 'destination'")

    window_cells = int(fabric.get("cells_per_window", fabric.get("die_cell_capacity", 1)))
    window_cells = max(1, window_cells)
    for layer in _unique_static_layers(manifest):
        cells = int(layer.weight_program_cells)
        counters.weight_cells += cells
        counters.weight_windows += ceil_div(cells, window_cells)
    counters.weight_bytes = counters.weight_cells * int(cfg.programming_word_bytes)

    for layer in manifest.layers:
        layer_data = layer.layer
        tokens = max(1, int(layer.tokens))
        token_tile = max(1, int(layer_data.get("y_tile", 0) or ceil_div(tokens, layer.token_blocks)))
        row_shards_total = 0
        output_values = 0
        for token_block in range(int(layer.token_blocks)):
            token_start = token_block * token_tile
            token_count = min(token_tile, tokens - token_start)
            for col_block in range(int(layer.col_tiles)):
                output_start = col_block * fp4_cols
                active_cols = min(fp4_cols, int(layer.out_dim) - output_start)
                if active_cols <= 0:
                    continue
                output_values += token_count * active_cols
                for row_block in range(int(layer.row_tiles)):
                    input_start = row_block * fp4_rows
                    active_rows = min(fp4_rows, int(layer.in_dim) - input_start)
                    if active_rows <= 0:
                        continue
                    row_shards = ceil_div(active_rows, x1_rows)
                    row_shards_total += row_shards
                    groups = max(1, ceil_div(row_shards, cluster))
                    counters.compute_groups += groups
                    speedup = _map_speedup(layer, cfg)
                    for group in range(groups):
                        shard_start = group * cluster
                        shard_count = min(cluster, row_shards - shard_start)
                        group_rows = min(active_rows, shard_count * x1_rows)
                        macs = token_count * group_rows * active_cols
                        counters.mac_operations += macs
                        counters.compute_cycles += (
                            ceil(macs / (int(cfg.pe_macs_per_cycle_per_macro) * shard_count * speedup))
                            + int(cfg.pe_compute_setup_cycles)
                        )

        partial_segments = max(1, row_shards_total // max(1, int(layer.token_blocks) * int(layer.col_tiles)))
        reduction_values = output_values * max(0, partial_segments - 1)
        counters.reduction_values += reduction_values
        counters.reduction_groups += int(layer.token_blocks) * int(layer.col_tiles) if reduction_values else 0
        counters.converted_values += output_values if reduction_placement == "local" else output_values * partial_segments
        counters.noc_values += output_values if reduction_placement == "local" else output_values * partial_segments

        bits_per_value = int(layer.role_policy.get("bits_per_value", 9))
        counters.noc_payload_bits += (
            output_values if reduction_placement == "local" else output_values * partial_segments
        ) * bits_per_value
        packet_units = int(layer.token_blocks) * int(layer.col_tiles)
        if reduction_placement == "destination":
            packet_units *= partial_segments
        counters.noc_packets += ceil_div(packet_units, coalescing)

    counters.noc_wire_bits = counters.noc_payload_bits + counters.noc_packets * int(cfg.noc_header_bits)
    counters.noc_flits = ceil_div(counters.noc_wire_bits, int(cfg.flit_bits))
    return counters


class LatencyEngine:
    """Estimate cycle cost for a manifest and its stream plan."""

    def __init__(self, config: CostModelConfig | None = None):
        self.config = config or CostModelConfig()

    def estimate(self, manifest: ProgramManifest, stream_plan: StreamPlan) -> LatencyBreakdown:
        cfg = self.config
        counters = _analyze(manifest, stream_plan, cfg)
        dma_cycles = _transfer_cycles(
            counters.weight_bytes,
            setup_cycles=cfg.dma_setup_cycles * counters.weight_windows,
            burst_setup_cycles=cfg.dma_burst_setup_cycles,
            burst_bytes=cfg.dma_burst_bytes,
            bytes_per_cycle=cfg.dma_bytes_per_cycle,
        )
        program_cycles = counters.weight_windows * cfg.x1_program_setup_cycles + ceil_div(
            counters.weight_cells, cfg.x1_program_cells_per_cycle
        )
        reduction_cycles = 0
        if counters.reduction_values:
            reduction_cycles = (
                counters.reduction_groups * cfg.reduction_setup_cycles
                + ceil_div(counters.reduction_values, cfg.reduction_values_per_cycle)
            )
        sar_cycles = 0
        if counters.converted_values:
            sar_cycles = (
                counters.noc_packets * cfg.sar_tdc_setup_cycles
                + ceil_div(counters.converted_values, cfg.sar_tdc_values_per_cycle)
            )
        noc_cycles = (
            counters.noc_packets * cfg.noc_startup_cycles
            + ceil_div(counters.noc_flits, cfg.noc_flits_per_cycle)
            + counters.noc_flits * cfg.noc_hops
        )
        return LatencyBreakdown(
            dma_load_cycles=int(dma_cycles),
            x1_programming_cycles=int(program_cycles),
            pe_compute_cycles=int(counters.compute_cycles),
            reduction_cycles=int(reduction_cycles),
            sar_tdc_conversion_cycles=int(sar_cycles),
            noc_transfer_cycles=int(noc_cycles),
        )


class EnergyEngine:
    """Estimate energy cost for a manifest and its stream plan."""

    def __init__(self, config: CostModelConfig | None = None):
        self.config = config or CostModelConfig()

    def estimate(self, manifest: ProgramManifest, stream_plan: StreamPlan) -> EnergyBreakdown:
        cfg = self.config
        counters = _analyze(manifest, stream_plan, cfg)
        output_bytes = ceil(counters.noc_payload_bits / 8)
        return EnergyBreakdown(
            memory_read_pj=counters.weight_bytes * cfg.memory_read_energy_pj_per_byte,
            memory_write_pj=output_bytes * cfg.memory_write_energy_pj_per_byte,
            x1_programming_pj=counters.weight_cells * cfg.x1_program_energy_pj_per_cell,
            compute_read_pj=counters.mac_operations * cfg.compute_read_energy_pj_per_mac,
            reduction_pj=counters.reduction_values * cfg.reduction_energy_pj_per_value,
            adc_sar_tdc_pj=counters.converted_values * cfg.adc_energy_pj_per_value,
            noc_pj=counters.noc_wire_bits * cfg.noc_energy_pj_per_bit,
        )


class LatencyEnergyEngine:
    """Convenience facade returning both cost breakdowns and raw counters."""

    def __init__(self, config: CostModelConfig | None = None):
        self.config = config or CostModelConfig()
        self.latency = LatencyEngine(self.config)
        self.energy = EnergyEngine(self.config)

    def estimate(self, manifest: ProgramManifest, stream_plan: StreamPlan) -> CostEstimate:
        counters = _analyze(manifest, stream_plan, self.config)
        latency = self.latency.estimate(manifest, stream_plan)
        energy = self.energy.estimate(manifest, stream_plan)
        return CostEstimate(
            latency=latency,
            energy=energy,
            counters=asdict(counters),
        )


__all__ = [
    "CostEstimate",
    "CostModelConfig",
    "EnergyBreakdown",
    "EnergyEngine",
    "LatencyBreakdown",
    "LatencyEnergyEngine",
    "LatencyEngine",
]
