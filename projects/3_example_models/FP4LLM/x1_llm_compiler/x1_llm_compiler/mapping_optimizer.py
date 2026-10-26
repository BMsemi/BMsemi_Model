"""Search mapping choices using the latency and energy engines."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, replace
from itertools import combinations, product
from pathlib import Path
from typing import Iterable

from .cost_model import CostEstimate, CostModelConfig, LatencyEnergyEngine
from .fabric import FabricConfig, StreamPlan, build_stream_plan
from .mapping import LayerSpec
from .planner import CompilerConfig, ProgramManifest, compile_model


def _map_name(value: str) -> str:
    normalized = str(value).strip().lower()
    if normalized == "spatialmap":
        return "SpatialMap"
    if normalized == "temporalmap":
        return "TemporalMap"
    raise ValueError("mapping choice must be SpatialMap or TemporalMap")


@dataclass(frozen=True)
class MappingCandidate:
    """One complete compiler/fabric mapping choice."""

    fp4llm_tile_rows: int = 64
    fp4llm_tile_cols: int = 32
    token_block_size: int | None = None
    pe_cluster_size: int = 4
    k_map: str = "SpatialMap"
    c_map: str = "TemporalMap"
    y_map: str = "SpatialMap"
    resident_weights: bool | None = None
    noc_coalescing_tiles: int = 1
    dma_window_cells: int | None = None
    reduction_placement: str = "local"

    def __post_init__(self) -> None:
        for name in ("fp4llm_tile_rows", "fp4llm_tile_cols", "pe_cluster_size", "noc_coalescing_tiles"):
            value = int(getattr(self, name))
            if value <= 0:
                raise ValueError(f"{name} must be positive")
            object.__setattr__(self, name, value)
        if self.token_block_size is not None and int(self.token_block_size) <= 0:
            raise ValueError("token_block_size must be positive when provided")
        for name in ("k_map", "c_map", "y_map"):
            object.__setattr__(self, name, _map_name(getattr(self, name)))
        if self.dma_window_cells is not None and int(self.dma_window_cells) <= 0:
            raise ValueError("dma_window_cells must be positive when provided")
        reduction = str(self.reduction_placement).strip().lower()
        if reduction not in {"local", "destination"}:
            raise ValueError("reduction_placement must be local or destination")
        object.__setattr__(self, "reduction_placement", reduction)

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class MappingSearchSpace:
    """Candidate values explored by :class:`MappingOptimizer`."""

    tile_rows: tuple[int, ...] = (32, 64, 128)
    tile_cols: tuple[int, ...] = (16, 32, 64)
    token_block_sizes: tuple[int | None, ...] = (None, 128, 256, 512)
    pe_cluster_sizes: tuple[int, ...] = (2, 4, 8)
    k_maps: tuple[str, ...] = ("SpatialMap", "TemporalMap")
    c_maps: tuple[str, ...] = ("TemporalMap", "SpatialMap")
    y_maps: tuple[str, ...] = ("SpatialMap", "TemporalMap")
    resident_weights: tuple[bool | None, ...] = (None, False, True)
    noc_coalescing_tiles: tuple[int, ...] = (1, 4, 8)
    dma_window_cells: tuple[int | None, ...] = (None, 65536, 262144)
    reduction_placements: tuple[str, ...] = ("local", "destination")

    def _axis_values(self) -> list[tuple[str, tuple[object, ...], object]]:
        return [
            ("fp4llm_tile_rows", tuple(self.tile_rows), 64),
            ("fp4llm_tile_cols", tuple(self.tile_cols), 32),
            ("token_block_size", tuple(self.token_block_sizes), None),
            ("pe_cluster_size", tuple(self.pe_cluster_sizes), 4),
            ("k_map", tuple(self.k_maps), "SpatialMap"),
            ("c_map", tuple(self.c_maps), "TemporalMap"),
            ("y_map", tuple(self.y_maps), "SpatialMap"),
            ("resident_weights", tuple(self.resident_weights), None),
            ("noc_coalescing_tiles", tuple(self.noc_coalescing_tiles), 1),
            ("dma_window_cells", tuple(self.dma_window_cells), None),
            ("reduction_placement", tuple(self.reduction_placements), "local"),
        ]

    def candidates(self, max_candidates: int | None = 64) -> Iterable[MappingCandidate]:
        """Yield deterministic baseline, single-axis, then pairwise choices.

        Passing ``None`` performs the full Cartesian product. The bounded
        default explores all individual knobs and then useful pairwise
        combinations without accidentally launching thousands of compiles.
        """

        axes = self._axis_values()
        baseline_values = {name: default for name, _values, default in axes}
        baseline = MappingCandidate(**baseline_values)
        yielded: set[MappingCandidate] = set()

        def emit(candidate: MappingCandidate) -> MappingCandidate | None:
            if candidate in yielded:
                return None
            yielded.add(candidate)
            return candidate

        first = emit(baseline)
        if first is not None:
            yield first
        if max_candidates is not None and len(yielded) >= int(max_candidates):
            return

        if max_candidates is None:
            choices = [tuple(values) if values else (default,) for _name, values, default in axes]
            names = [name for name, _values, _default in axes]
            for values in product(*choices):
                candidate = MappingCandidate(**dict(zip(names, values)))
                item = emit(candidate)
                if item is not None:
                    yield item
            return

        limit = max(1, int(max_candidates))
        for name, values, _default in axes:
            for value in values:
                candidate = MappingCandidate(**{**baseline_values, name: value})
                item = emit(candidate)
                if item is not None:
                    yield item
                if len(yielded) >= limit:
                    return
        for (name_a, values_a, _default_a), (name_b, values_b, _default_b) in combinations(axes, 2):
            for value_a in values_a:
                for value_b in values_b:
                    candidate = MappingCandidate(
                        **{**baseline_values, name_a: value_a, name_b: value_b}
                    )
                    item = emit(candidate)
                    if item is not None:
                        yield item
                    if len(yielded) >= limit:
                        return


@dataclass(frozen=True)
class OptimizationConfig:
    latency_weight: float = 1.0
    energy_weight: float = 1.0
    latency_scale_cycles: float = 1_000_000.0
    energy_scale_pj: float = 1_000_000_000.0
    max_candidates: int | None = 64

    def __post_init__(self) -> None:
        for name in ("latency_weight", "energy_weight", "latency_scale_cycles", "energy_scale_pj"):
            if float(getattr(self, name)) <= 0:
                raise ValueError(f"{name} must be positive")
        if self.max_candidates is not None and int(self.max_candidates) <= 0:
            raise ValueError("max_candidates must be positive when provided")

    def score(self, cost: CostEstimate) -> float:
        return (
            float(self.latency_weight) * cost.latency.total_cycles / float(self.latency_scale_cycles)
            + float(self.energy_weight) * cost.energy.total_pj / float(self.energy_scale_pj)
        )


@dataclass(frozen=True)
class CandidateEvaluation:
    candidate: MappingCandidate
    score: float | None
    cost: CostEstimate | None
    execution_mode: str
    valid: bool
    reason: str = ""

    def as_dict(self) -> dict[str, object]:
        return {
            "candidate": self.candidate.as_dict(),
            "score": self.score,
            "cost": None if self.cost is None else self.cost.as_dict(),
            "execution_mode": self.execution_mode,
            "valid": self.valid,
            "reason": self.reason,
        }


@dataclass(frozen=True)
class OptimizationResult:
    best_candidate: MappingCandidate
    best_cost: CostEstimate
    best_manifest: ProgramManifest
    best_stream_plan: StreamPlan
    evaluations: list[CandidateEvaluation]

    @property
    def best_score(self) -> float:
        return min(float(item.score) for item in self.evaluations if item.valid and item.score is not None)

    def as_dict(self) -> dict[str, object]:
        return {
            "best_candidate": self.best_candidate.as_dict(),
            "best_score": self.best_score,
            "best_cost": self.best_cost.as_dict(),
            "manifest_summary": self.best_manifest.summary,
            "stream_summary": dict(self.best_stream_plan.summary),
            "evaluations": [item.as_dict() for item in self.evaluations],
        }

    def write_json(self, path: str | Path) -> None:
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(self.as_dict(), indent=2) + "\n", encoding="utf-8")


def apply_mapping_candidate(
    phase_layers: dict[str, tuple[str, list[LayerSpec]]],
    candidate: MappingCandidate,
) -> dict[str, tuple[str, list[LayerSpec]]]:
    """Apply candidate K/C/Y map choices without mutating source layers."""

    return {
        phase: (
            network,
            [
                replace(
                    layer,
                    k_map=candidate.k_map,
                    c_map=candidate.c_map,
                    y_map=candidate.y_map,
                    y_tile=(
                        layer.y_tile
                        if candidate.token_block_size is None
                        else min(int(candidate.token_block_size), int(layer.tokens))
                    ),
                    y_stride=(
                        layer.y_stride
                        if candidate.token_block_size is None
                        else min(int(candidate.token_block_size), int(layer.tokens))
                    ),
                )
                for layer in layers
            ],
        )
        for phase, (network, layers) in phase_layers.items()
    }


def render_mapping_file(
    phase_layers: tuple[str, list[LayerSpec]],
    *,
    candidate: MappingCandidate | None = None,
) -> str:
    """Render normalized layer specs as a MAESTRO-style ``.m`` file."""

    network, layers = phase_layers
    lines = [
        "# Generated by x1_llm_compiler MappingOptimizer.",
    ]
    if candidate is not None:
        lines.append(f"# Best candidate: {json.dumps(candidate.as_dict(), sort_keys=True)}")
    lines.extend(("", f"Network {network} {{", ""))

    for layer in layers:
        lines.extend(
            (
                f"Layer {layer.name} {{",
                f"  Type: {layer.layer_type}",
                f"  Stride {{ X:{int(layer.stride_x)}, Y:{int(layer.stride_y)} }}",
                (
                    "  Dimensions { "
                    f"K:{int(layer.k)}, C:{int(layer.c)}, R:{int(layer.r)}, "
                    f"S:{int(layer.s)}, Y:{int(layer.y)}, X:{int(layer.x)} "
                    "}"
                ),
                "  Dataflow {",
                f"    {layer.k_map}({int(layer.k_tile)},{int(layer.k_stride)}) K;",
                f"    {layer.c_map}({int(layer.c_tile)},{int(layer.c_stride)}) C;",
                f"    {layer.y_map}({int(layer.y_tile)},{int(layer.y_stride)}) Y;",
                "    TemporalMap(Sz(R),Sz(R)) R;",
                "    TemporalMap(Sz(S),Sz(S)) S;",
                "    TemporalMap(1,1) X;",
                "  }",
                "}",
                "",
            )
        )

    lines.append("}")
    return "\n".join(lines) + "\n"


def write_mapping_file(
    phase_layers: tuple[str, list[LayerSpec]],
    path: str | Path,
    *,
    candidate: MappingCandidate | None = None,
) -> None:
    """Write a normalized MAESTRO-style mapping file to ``path``."""

    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(render_mapping_file(phase_layers, candidate=candidate), encoding="utf-8")


class MappingOptimizer:
    """Compile, stream-plan, cost, and rank mapping candidates."""

    def __init__(
        self,
        *,
        cost_model: CostModelConfig | None = None,
        optimization: OptimizationConfig | None = None,
    ):
        self.cost_model = cost_model or CostModelConfig()
        self.optimization = optimization or OptimizationConfig()
        self.engine = LatencyEnergyEngine(self.cost_model)

    def optimize(
        self,
        phase_layers: dict[str, tuple[str, list[LayerSpec]]],
        *,
        compiler_config: CompilerConfig | None = None,
        fabric_config: FabricConfig | None = None,
        search_space: MappingSearchSpace | None = None,
    ) -> OptimizationResult:
        base_compiler = compiler_config or CompilerConfig()
        base_fabric = fabric_config or FabricConfig()
        space = search_space or MappingSearchSpace()
        evaluations: list[CandidateEvaluation] = []
        best: tuple[float, MappingCandidate, CostEstimate, ProgramManifest, StreamPlan] | None = None

        for candidate in space.candidates(self.optimization.max_candidates):
            candidate_compiler = replace(
                base_compiler,
                fp4llm_tile_rows=candidate.fp4llm_tile_rows,
                fp4llm_tile_cols=candidate.fp4llm_tile_cols,
                x1_macros_per_endpoint=candidate.pe_cluster_size,
            )
            candidate_fabric = replace(
                base_fabric,
                x1_macros_per_endpoint=candidate.pe_cluster_size,
                resident_weights=candidate.resident_weights,
                weight_window_cells=candidate.dma_window_cells,
                noc_coalescing_tiles=candidate.noc_coalescing_tiles,
                reduction_placement=candidate.reduction_placement,
            )
            try:
                mapped_layers = apply_mapping_candidate(phase_layers, candidate)
                manifest = compile_model(mapped_layers, candidate_compiler)
                stream_plan = build_stream_plan(manifest, candidate_fabric)
                if stream_plan.execution_mode == "resident_infeasible":
                    evaluations.append(
                        CandidateEvaluation(
                            candidate=candidate,
                            score=None,
                            cost=None,
                            execution_mode=stream_plan.execution_mode,
                            valid=False,
                            reason="resident weight choice does not fit available X1 capacity",
                        )
                    )
                    continue
                cost = self.engine.estimate(manifest, stream_plan)
                score = self.optimization.score(cost)
                evaluation = CandidateEvaluation(
                    candidate=candidate,
                    score=score,
                    cost=cost,
                    execution_mode=stream_plan.execution_mode,
                    valid=True,
                )
                evaluations.append(evaluation)
                if best is None or score < best[0]:
                    best = (score, candidate, cost, manifest, stream_plan)
            except (TypeError, ValueError, KeyError) as exc:
                evaluations.append(
                    CandidateEvaluation(
                        candidate=candidate,
                        score=None,
                        cost=None,
                        execution_mode="invalid",
                        valid=False,
                        reason=str(exc),
                    )
                )

        if best is None:
            raise RuntimeError("No valid mapping candidates were produced")
        _score, candidate, cost, manifest, stream_plan = best
        return OptimizationResult(
            best_candidate=candidate,
            best_cost=cost,
            best_manifest=manifest,
            best_stream_plan=stream_plan,
            evaluations=evaluations,
        )


__all__ = [
    "CandidateEvaluation",
    "MappingCandidate",
    "MappingOptimizer",
    "MappingSearchSpace",
    "OptimizationConfig",
    "OptimizationResult",
    "apply_mapping_candidate",
    "render_mapping_file",
    "write_mapping_file",
]
