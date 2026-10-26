from __future__ import annotations

from dataclasses import dataclass
from math import ceil
from typing import Any


@dataclass(frozen=True)
class LatencyModelConfig:
    """
    Cycle-level latency model for the current DeepGate mapping flow.

    The simulator uses a 32x32 PE in PWM_INT8_ROW_SERIAL mode. One logical
    compute fold streams input rows through the PE; each row has two PWM passes
    because INT8 activations are split into two 4-bit nibbles.
    """

    pe_rows: int = 32
    k_parallelism_default: int = 8
    program_tile_cycles: int = 1
    program_ports: int = 512
    pwm_pass_cycles_per_row: int = 2
    digital_accumulate_cycles_per_fold: int = 1
    input_buffer_read_cycles_per_fold: int = 1
    output_buffer_write_cycles: int = 1
    requant_cycles_per_k_group: int = 1
    activation_cycles_per_k_group: int = 1
    reduction_cycles_per_level: int = 1
    loop_overhead_cycles: int = 0
    dynamic_weight_reprogramming: bool = False


@dataclass(frozen=True)
class LatencyEstimate:
    """
    Result of the latency objective.

    latency_objective is intentionally the raw cycle count. Optimizers can
    minimize it directly, or use normalized_latency for comparing layers with
    different dense MAC counts.
    """

    latency_objective: int
    normalized_latency: float
    program_cycles: int
    compute_cycles: int
    finalize_cycles: int
    serial_latency_cycles: int
    dynamic_sequence_cycles: int
    dense_mac_reference_cycles: int
    output_positions: int
    k_groups: int
    k_waves: int
    c_groups: int
    c_waves: int
    reduction_cycles: int
    fold_steps: int
    fold_compute_cycles: int
    fold_program_cycles: int
    fold_cycles: int
    loop_order: str
    logical_compute_events: int
    program_events: int
    program_iterations: int
    dynamic_weight_reprogramming: bool

    def as_dict(self) -> dict[str, int | float | bool]:
        return {
            "latency_objective": self.latency_objective,
            "normalized_latency": self.normalized_latency,
            "program_cycles": self.program_cycles,
            "compute_cycles": self.compute_cycles,
            "finalize_cycles": self.finalize_cycles,
            "serial_latency_cycles": self.serial_latency_cycles,
            "dynamic_sequence_cycles": self.dynamic_sequence_cycles,
            "dense_mac_reference_cycles": self.dense_mac_reference_cycles,
            "output_positions": self.output_positions,
            "k_groups": self.k_groups,
            "k_waves": self.k_waves,
            "c_groups": self.c_groups,
            "c_waves": self.c_waves,
            "reduction_cycles": self.reduction_cycles,
            "fold_steps": self.fold_steps,
            "fold_compute_cycles": self.fold_compute_cycles,
            "fold_program_cycles": self.fold_program_cycles,
            "fold_cycles": self.fold_cycles,
            "loop_order": self.loop_order,
            "logical_compute_events": self.logical_compute_events,
            "program_events": self.program_events,
            "program_iterations": self.program_iterations,
            "dynamic_weight_reprogramming": self.dynamic_weight_reprogramming,
        }


DEFAULT_LATENCY_CONFIG = LatencyModelConfig()


def _ceil_div(num: int, den: int) -> int:
    return int(ceil(int(num) / max(1, int(den))))


def _fold_sizes_from_patch_len(patch_len: int, pe_rows: int) -> list[int]:
    remaining = int(patch_len)
    out: list[int] = []
    while remaining > 0:
        take = min(int(pe_rows), remaining)
        out.append(take)
        remaining -= take
    return out or [0]


def _output_positions(dims: Any) -> int:
    y = int(getattr(dims, "Y", 1))
    x = int(getattr(dims, "X", 1))
    stride_y = max(1, int(getattr(dims, "stride_y", 1)))
    stride_x = max(1, int(getattr(dims, "stride_x", 1)))
    return _ceil_div(y, stride_y) * _ceil_div(x, stride_x)


def _reduction_tree_levels(groups: int) -> int:
    levels = 0
    width = max(1, int(groups))
    while width > 1:
        width = _ceil_div(width, 2)
        levels += 1
    return levels


def estimate_latency_objective(
    dims: Any,
    mapping: Any,
    *,
    total_pes: int,
    config: LatencyModelConfig = DEFAULT_LATENCY_CONFIG,
) -> LatencyEstimate:
    """
    Define the latency objective for a mapped layer.

    Objective:
        minimize J_latency = L_program + L_compute + L_finalize

    where:
        loop order = K -> C -> (RSYX if R,S > 1 else YXRS)
        L_program = ceil(program_events / program_ports) * program_tile_cycles
        L_compute = output_positions * k_waves * c_waves * sum_fold(
            input_buffer_read + fold_size * pwm_pass_cycles_per_row
            + digital_accumulate
        )
        L_finalize = output_positions * k_waves * (
            reduction_tree + output_buffer_write + requant + activation
        )

    If the mapping spatializes C inside a Cluster, each PE only pays the local
    C tile latency and the objective adds a small reduction-tree term for the
    partial sums.

    For dynamic weights, set dynamic_weight_reprogramming=True in the config.
    Then each C fold charges program latency before its inference latency:
        fold_dynamic = program_fold + input/read + PWM row-serial + accumulate
    K groups within the same spatial wave run in parallel, so the C-fold latency
    is multiplied by k_waves, not by the number of K groups.
    """

    k_dim = int(getattr(dims, "K", 1))
    c_dim = int(getattr(dims, "C", 1))
    r_dim = int(getattr(dims, "R", 1))
    s_dim = int(getattr(dims, "S", 1))
    patch_len = int(getattr(mapping, "patch_len", c_dim * r_dim * s_dim))
    output_positions = _output_positions(dims)
    loop_order = "K->C->RSYX" if r_dim > 1 and s_dim > 1 else "K->C->YXRS"

    k_chunks = list(getattr(mapping, "k_chunks", []))
    if k_chunks:
        k_groups = len(k_chunks)
    else:
        k_parallelism = int(
            getattr(
                mapping,
                "k_spatial_parallelism",
                getattr(mapping, "outer_tile", config.k_parallelism_default),
            )
        )
        k_groups = _ceil_div(k_dim, k_parallelism)

    inner_dim = str(getattr(mapping, "inner_spatial_dim", "") or "").upper()
    inner_tile = getattr(mapping, "inner_tile", None)
    inner_stride = getattr(mapping, "inner_stride", None)
    cluster_size = int(getattr(mapping, "cluster_size", 1) or 1)

    c_groups = 1
    local_patch_len = patch_len
    if inner_dim == "C" and inner_tile is not None and inner_stride is not None:
        c_groups = _ceil_div(c_dim, int(inner_stride))
        c_waves = _ceil_div(c_groups, cluster_size)
        local_c = min(c_dim, int(inner_tile))
        local_patch_len = max(1, local_c * r_dim * s_dim)
    else:
        c_waves = 1

    fold_sizes = list(getattr(mapping, "fold_sizes", []))
    if c_groups > 1:
        fold_sizes = _fold_sizes_from_patch_len(local_patch_len, config.pe_rows)
    if not fold_sizes:
        fold_sizes = _fold_sizes_from_patch_len(patch_len, config.pe_rows)

    fold_compute_cycles = sum(
        int(config.input_buffer_read_cycles_per_fold)
        + int(fold_size) * int(config.pwm_pass_cycles_per_row)
        + int(config.digital_accumulate_cycles_per_fold)
        + int(config.loop_overhead_cycles)
        for fold_size in fold_sizes
    )

    pes_per_k_group = min(max(1, cluster_size), max(1, c_groups))
    active_k_parallelism = min(
        max(1, k_groups),
        max(1, int(total_pes) // pes_per_k_group),
    )
    k_waves = _ceil_div(k_groups, active_k_parallelism)
    program_events = k_groups * c_groups * len(fold_sizes) * output_positions
    static_program_cycles = (
        _ceil_div(k_groups * c_groups, config.program_ports)
        * int(config.program_tile_cycles)
    )

    static_compute_cycles = int(output_positions * k_waves * c_waves * fold_compute_cycles)
    reduction_cycles = (
        _reduction_tree_levels(c_groups)
        * int(config.reduction_cycles_per_level)
    )
    finalize_per_wave = (
        reduction_cycles
        + int(config.output_buffer_write_cycles)
        + int(config.requant_cycles_per_k_group)
        + int(config.activation_cycles_per_k_group)
    )
    static_finalize_cycles = int(output_positions * k_waves * finalize_per_wave)

    program_iterations = int(output_positions * k_waves * c_waves * len(fold_sizes))
    fold_program_cycles = (
        _ceil_div(k_groups * c_groups, config.program_ports)
        * int(config.program_tile_cycles)
    )
    dynamic_program_cycles = int(program_iterations * fold_program_cycles)
    dynamic_compute_cycles = int(output_positions * k_waves * c_waves * fold_compute_cycles)
    dynamic_finalize_cycles = int(output_positions * k_waves * finalize_per_wave)
    dynamic_sequence_cycles = int(
        dynamic_program_cycles
        + dynamic_compute_cycles
        + dynamic_finalize_cycles
    )

    if config.dynamic_weight_reprogramming:
        program_cycles = dynamic_program_cycles
        compute_cycles = dynamic_compute_cycles
        finalize_cycles = dynamic_finalize_cycles
        latency_cycles = dynamic_sequence_cycles
    else:
        program_cycles = static_program_cycles
        compute_cycles = static_compute_cycles
        finalize_cycles = static_finalize_cycles
        latency_cycles = int(program_cycles + compute_cycles + finalize_cycles)

    logical_compute_events = int(output_positions * k_groups * c_groups * len(fold_sizes))
    serial_latency_cycles = int(
        program_events * int(config.program_tile_cycles)
        + output_positions * k_groups * c_groups * fold_compute_cycles
        + output_positions * k_groups * finalize_per_wave
    )
    dense_mac_reference_cycles = max(1, int(output_positions * k_dim * patch_len))
    normalized_latency = float(latency_cycles / dense_mac_reference_cycles)

    return LatencyEstimate(
        latency_objective=latency_cycles,
        normalized_latency=normalized_latency,
        program_cycles=program_cycles,
        compute_cycles=compute_cycles,
        finalize_cycles=finalize_cycles,
        serial_latency_cycles=serial_latency_cycles,
        dynamic_sequence_cycles=dynamic_sequence_cycles,
        dense_mac_reference_cycles=dense_mac_reference_cycles,
        output_positions=output_positions,
        k_groups=k_groups,
        k_waves=k_waves,
        c_groups=c_groups,
        c_waves=c_waves,
        reduction_cycles=reduction_cycles,
        fold_steps=len(fold_sizes),
        fold_compute_cycles=fold_compute_cycles,
        fold_program_cycles=fold_program_cycles if config.dynamic_weight_reprogramming else 0,
        fold_cycles=(
            fold_compute_cycles
            + (len(fold_sizes) * fold_program_cycles if config.dynamic_weight_reprogramming else 0)
        ),
        loop_order=loop_order,
        logical_compute_events=logical_compute_events,
        program_events=program_events,
        program_iterations=program_iterations,
        dynamic_weight_reprogramming=bool(config.dynamic_weight_reprogramming),
    )


def estimate_fc_latency_objective(
    *,
    shape_in: int,
    shape_out: int,
    total_pes: int,
    k_parallelism: int = DEFAULT_LATENCY_CONFIG.k_parallelism_default,
    config: LatencyModelConfig = DEFAULT_LATENCY_CONFIG,
) -> LatencyEstimate:
    """
    Fallback latency objective for a dense FC layer when no mapping is present.
    """

    class _Dims:
        K = int(shape_out)
        C = int(shape_in)
        R = 1
        S = 1
        Y = 1
        X = 1
        stride_y = 1
        stride_x = 1

    class _Mapping:
        patch_len = int(shape_in)
        fold_sizes = _fold_sizes_from_patch_len(int(shape_in), config.pe_rows)
        k_chunks = [
            (k0, min(k0 + int(k_parallelism), int(shape_out)))
            for k0 in range(0, int(shape_out), int(k_parallelism))
        ]
        k_spatial_parallelism = int(k_parallelism)

    return estimate_latency_objective(
        _Dims(),
        _Mapping(),
        total_pes=total_pes,
        config=config,
    )
