from __future__ import annotations

import argparse
import math
import re
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class HardwareConfig:
    total_pes: int = 512
    pe_rows: int = 64
    neurons_per_pe: int = 8
    program_ports: int = 512
    max_cluster_size: int = 16
    max_cluster_pes: int = 16
    min_cluster_c: int = 8
    min_cluster_k: int = 16
    latency_weight: float = 1.0
    area_weight: float = 0.001
    program_cycles: int = 1
    input_read_cycles: int = 1
    pwm_pass_cycles_per_row: int = 2
    accumulate_cycles: int = 1
    output_write_cycles: int = 1
    requant_cycles: int = 1
    activation_cycles: int = 1
    reduction_cycles_per_level: int = 1


@dataclass(frozen=True)
class LayerSpec:
    name: str
    layer_type: str
    k: int
    c: int
    r: int
    s: int
    y: int
    x: int
    stride_x: int
    stride_y: int

    @property
    def output_y(self) -> int:
        return ceil_div(self.y, self.stride_y)

    @property
    def output_x(self) -> int:
        return ceil_div(self.x, self.stride_x)

    @property
    def output_positions(self) -> int:
        return self.output_y * self.output_x

    @property
    def patch_len(self) -> int:
        return self.c * self.r * self.s

    @property
    def weight_count(self) -> int:
        return self.k * self.patch_len

    @property
    def macs(self) -> int:
        return self.output_positions * self.weight_count

    @property
    def uses_rsyx(self) -> bool:
        return self.r > 1 and self.s > 1

    @property
    def loop_order(self) -> str:
        return "K->C->RSYX" if self.uses_rsyx else "K->C->YXRS"


@dataclass(frozen=True)
class MappingCandidate:
    k_tile: int
    c_tile: int
    uses_cluster: bool
    cluster_size: int
    k_groups: int
    k_waves: int
    c_folds: int
    c_waves: int
    row_fold: int
    fold_program_cycles: int
    fold_compute_cycles: int
    reduction_cycles: int
    finalize_cycles: int
    latency_cycles: int
    area_cost: int
    active_pes: int
    utilization_pct: float
    score: float


def ceil_div(num: int, den: int) -> int:
    return int(math.ceil(int(num) / max(1, int(den))))


def strip_line_comments(text: str) -> str:
    out = []
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("%"):
            continue
        line = re.sub(r"//.*", "", line)
        if line.strip():
            out.append(line)
    return "\n".join(out)


def extract_braced_block(text: str, start: int) -> tuple[str, int]:
    open_idx = text.find("{", start)
    if open_idx < 0:
        raise ValueError("Expected '{' for braced block.")

    depth = 1
    i = open_idx + 1
    while i < len(text) and depth > 0:
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
        i += 1

    if depth != 0:
        raise ValueError("Unbalanced braces in mapping file.")

    return text[open_idx + 1 : i - 1], i


def parse_keyvals(block_text: str) -> dict[str, int]:
    return {
        key.upper(): int(val)
        for key, val in re.findall(r"([A-Za-z_]+)\s*:\s*([0-9]+)", block_text)
    }


def parse_layers(path: Path) -> tuple[str, list[LayerSpec]]:
    text = strip_line_comments(path.read_text(encoding="utf-8", errors="ignore"))
    network_match = re.search(r"\bNetwork\s+([A-Za-z_][A-Za-z0-9_]*)\s*\{", text)
    network_name = network_match.group(1) if network_match else "OptimizedNetwork"

    layers: list[LayerSpec] = []
    pos = 0
    while True:
        match = re.search(r"\bLayer\s+([A-Za-z_][A-Za-z0-9_]*)\s*\{", text[pos:])
        if not match:
            break

        layer_name = match.group(1)
        layer_start = pos + match.start()
        layer_body, layer_end = extract_braced_block(text, layer_start)
        pos = layer_end

        type_match = re.search(r"\bType\s*:\s*([A-Za-z_][A-Za-z0-9_]*)", layer_body)
        layer_type = type_match.group(1) if type_match else "Conv"

        stride_match = re.search(r"\bStride\s*\{", layer_body)
        dim_match = re.search(r"\bDimensions\s*\{", layer_body)
        if not stride_match or not dim_match:
            continue

        stride_body, _ = extract_braced_block(layer_body, stride_match.start())
        dim_body, _ = extract_braced_block(layer_body, dim_match.start())
        stride = parse_keyvals(stride_body)
        dims = parse_keyvals(dim_body)

        layers.append(
            LayerSpec(
                name=layer_name,
                layer_type=layer_type,
                k=dims.get("K", 1),
                c=dims.get("C", 1),
                r=dims.get("R", 1),
                s=dims.get("S", 1),
                y=dims.get("Y", 1),
                x=dims.get("X", 1),
                stride_x=stride.get("X", 1),
                stride_y=stride.get("Y", 1),
            )
        )

    if not layers:
        raise ValueError(f"No layers found in {path}")

    return network_name, layers


def is_model_input(path: Path) -> bool:
    return path.suffix.lower() in {".onnx", ".tflite", ".lite", ".pb"}


def graph_layers_to_layer_specs(graph_layers) -> list[LayerSpec]:
    return [
        LayerSpec(
            name=layer.name,
            layer_type=layer.layer_type,
            k=layer.k,
            c=layer.c,
            r=layer.r,
            s=layer.s,
            y=layer.y,
            x=layer.x,
            stride_x=layer.stride_x,
            stride_y=layer.stride_y,
        )
        for layer in graph_layers
    ]


def legal_k_tiles(layer: LayerSpec, hw: HardwareConfig) -> list[int]:
    max_tile = min(layer.k, hw.neurons_per_pe)
    return [tile for tile in range(1, max_tile + 1) if layer.k % tile == 0 or tile == max_tile]


def legal_c_tiles(layer: LayerSpec, hw: HardwareConfig) -> list[int]:
    max_c_tile = max(1, min(layer.c, hw.pe_rows // max(1, layer.r * layer.s)))
    candidates = {1, max_c_tile}
    for tile in range(1, max_c_tile + 1):
        if layer.c % tile == 0:
            candidates.add(tile)
    return sorted(candidates)


def reduction_tree_levels(groups: int) -> int:
    levels = 0
    width = max(1, int(groups))
    while width > 1:
        width = ceil_div(width, 2)
        levels += 1
    return levels


def legal_cluster_sizes(layer: LayerSpec, hw: HardwareConfig, k_groups: int, c_folds: int) -> list[int]:
    if (
        c_folds <= 1
        or layer.c < hw.min_cluster_c
        or layer.k < hw.min_cluster_k
    ):
        return [1]

    max_by_total_pes = max(1, hw.total_pes // max(1, k_groups))
    max_by_cluster_budget = max(1, hw.max_cluster_pes // max(1, k_groups))
    max_by_pes = min(max_by_total_pes, max_by_cluster_budget)
    max_cluster = min(c_folds, hw.max_cluster_size, max_by_pes)
    sizes = {1, max_cluster}
    for size in range(2, max_cluster + 1):
        if c_folds % size == 0:
            sizes.add(size)
    return sorted(sizes)


def estimate_candidate(
    layer: LayerSpec,
    hw: HardwareConfig,
    k_tile: int,
    c_tile: int,
    cluster_size: int = 1,
) -> MappingCandidate:
    row_fold = c_tile * layer.r * layer.s
    if row_fold > hw.pe_rows:
        raise ValueError("C tile does not fit in PE rows.")

    k_groups = ceil_div(layer.k, k_tile)
    c_folds = ceil_div(layer.c, c_tile)
    cluster_size = max(1, min(int(cluster_size), c_folds))
    uses_cluster = cluster_size > 1

    k_parallel_groups = max(1, hw.total_pes // cluster_size)
    k_waves = ceil_div(k_groups, k_parallel_groups)
    c_waves = ceil_div(c_folds, cluster_size)

    active_c_lanes = min(cluster_size, c_folds)
    active_program_lanes = min(k_groups, k_parallel_groups) * active_c_lanes
    fold_program = ceil_div(active_program_lanes, hw.program_ports) * hw.program_cycles
    fold_compute = (
        hw.input_read_cycles
        + row_fold * hw.pwm_pass_cycles_per_row
        + hw.accumulate_cycles
    )
    reduction = (
        reduction_tree_levels(active_c_lanes) * hw.reduction_cycles_per_level
        if uses_cluster
        else 0
    )
    finalize = hw.output_write_cycles + hw.requant_cycles + hw.activation_cycles

    latency = layer.output_positions * k_waves * c_waves * (
        fold_program + fold_compute + reduction
    )
    latency += layer.output_positions * k_waves * finalize

    active_pes = min(hw.total_pes, k_groups * active_c_lanes)
    utilization = 100.0 * active_pes / hw.total_pes if hw.total_pes else 0.0

    area_cost = active_pes
    score = float(hw.latency_weight) * float(latency) + float(hw.area_weight) * float(area_cost)

    return MappingCandidate(
        k_tile=k_tile,
        c_tile=c_tile,
        uses_cluster=uses_cluster,
        cluster_size=cluster_size,
        k_groups=k_groups,
        k_waves=k_waves,
        c_folds=c_folds,
        c_waves=c_waves,
        row_fold=row_fold,
        fold_program_cycles=fold_program,
        fold_compute_cycles=fold_compute,
        reduction_cycles=reduction,
        finalize_cycles=finalize,
        latency_cycles=int(latency),
        area_cost=area_cost,
        active_pes=active_pes,
        utilization_pct=utilization,
        score=score,
    )


def enumerate_candidates(layer: LayerSpec, hw: HardwareConfig) -> list[MappingCandidate]:
    candidates: list[MappingCandidate] = []
    for k_tile in legal_k_tiles(layer, hw):
        for c_tile in legal_c_tiles(layer, hw):
            k_groups = ceil_div(layer.k, k_tile)
            c_folds = ceil_div(layer.c, c_tile)
            for cluster_size in legal_cluster_sizes(layer, hw, k_groups, c_folds):
                candidates.append(estimate_candidate(layer, hw, k_tile, c_tile, cluster_size))
    return candidates


def optimize_layer(layer: LayerSpec, hw: HardwareConfig) -> tuple[MappingCandidate, list[MappingCandidate]]:
    candidates = enumerate_candidates(layer, hw)
    return min(
        candidates,
        key=lambda candidate: (
            candidate.score,
            candidate.area_cost,
            candidate.latency_cycles,
            -int(candidate.uses_cluster),
            -candidate.cluster_size,
            -candidate.k_tile,
            -candidate.c_tile,
        ),
    ), candidates


def render_layer(layer: LayerSpec, candidate: MappingCandidate) -> str:
    lines = [
        f"    Layer {layer.name} {{",
        f"        Type: {layer.layer_type}",
        f"        Stride {{ X: {layer.stride_x}, Y: {layer.stride_y} }}",
        f"        Dimensions {{ K: {layer.k}, C: {layer.c}, R: {layer.r}, S: {layer.s}, Y: {layer.y}, X: {layer.x} }}",
        "        Dataflow {",
        f"            // Optimized cost: {candidate.latency_cycles} cycles; loop order {layer.loop_order}.",
        f"            // K groups={candidate.k_groups}, C folds={candidate.c_folds}, C waves={candidate.c_waves}, row fold={candidate.row_fold}.",
        f"            SpatialMap({candidate.k_tile},{candidate.k_tile}) K;",
        f"            TemporalMap({candidate.c_tile},{candidate.c_tile}) C;",
    ]

    if layer.uses_rsyx:
        lines.extend(
            [
                f"            TemporalMap({layer.r},{layer.r}) R;",
                f"            TemporalMap({layer.s},{layer.s}) S;",
                f"            TemporalMap({layer.r},1) Y;",
                f"            TemporalMap({layer.s},1) X;",
            ]
        )
    else:
        lines.extend(
            [
                "            TemporalMap(1,1) Y;",
                "            TemporalMap(1,1) X;",
                f"            TemporalMap({layer.r},{layer.r}) R;",
                f"            TemporalMap({layer.s},{layer.s}) S;",
            ]
        )

    if candidate.uses_cluster:
        lines.extend(
            [
                "",
                f"            // Cluster C folds across {candidate.cluster_size} PEs to reduce latency.",
                f"            Cluster({candidate.cluster_size},P);",
                f"            SpatialMap({candidate.c_tile},{candidate.c_tile}) C;",
            ]
        )
        if layer.uses_rsyx:
            lines.extend(
                [
                    f"            TemporalMap({layer.r},{layer.r}) R;",
                    f"            TemporalMap({layer.s},{layer.s}) S;",
                    f"            TemporalMap({layer.r},1) Y;",
                    f"            TemporalMap({layer.s},1) X;",
                ]
            )
        else:
            lines.extend(
                [
                    "            TemporalMap(1,1) Y;",
                    "            TemporalMap(1,1) X;",
                    f"            TemporalMap({layer.r},{layer.r}) R;",
                    f"            TemporalMap({layer.s},{layer.s}) S;",
                ]
            )

    lines.extend(["        }", "    }"])
    return "\n".join(lines)


def render_mapping(network_name: str, layers: list[LayerSpec], selected: list[MappingCandidate]) -> str:
    out = [f"Network {network_name}_AutoMapped {{"]
    for layer, candidate in zip(layers, selected):
        out.append(render_layer(layer, candidate))
        out.append("")
    out.append("}")
    return "\n".join(out)


def render_report(layers: list[LayerSpec], selected: list[MappingCandidate], hw: HardwareConfig) -> str:
    rows = [
        "Layer,Optimizer,LatencyWeight,AreaWeight,LoopOrder,K,C,R,S,Y,X,StrideY,StrideX,KTile,CTile,UsesCluster,ClusterSize,KGroups,Kwaves,Cfolds,Cwaves,RowFold,FoldProgram,FoldCompute,Reduction,Finalize,LatencyCycles,AreaCost,Score,ActivePEs,UtilizationPct",
    ]
    for layer, candidate in zip(layers, selected):
        rows.append(
            ",".join(
                [
                    layer.name,
                    "exhaustive_discrete_search",
                    str(hw.latency_weight),
                    str(hw.area_weight),
                    layer.loop_order,
                    str(layer.k),
                    str(layer.c),
                    str(layer.r),
                    str(layer.s),
                    str(layer.y),
                    str(layer.x),
                    str(layer.stride_y),
                    str(layer.stride_x),
                    str(candidate.k_tile),
                    str(candidate.c_tile),
                    str(candidate.uses_cluster),
                    str(candidate.cluster_size),
                    str(candidate.k_groups),
                    str(candidate.k_waves),
                    str(candidate.c_folds),
                    str(candidate.c_waves),
                    str(candidate.row_fold),
                    str(candidate.fold_program_cycles),
                    str(candidate.fold_compute_cycles),
                    str(candidate.reduction_cycles),
                    str(candidate.finalize_cycles),
                    str(candidate.latency_cycles),
                    str(candidate.area_cost),
                    f"{candidate.score:.6f}",
                    str(candidate.active_pes),
                    f"{candidate.utilization_pct:.4f}",
                ]
            )
        )
    return "\n".join(rows) + "\n"


def render_trace(layers: list[LayerSpec], all_candidates: list[list[MappingCandidate]], hw: HardwareConfig) -> str:
    rows = [
        "Layer,CandidateRank,Selected,Optimizer,LatencyWeight,AreaWeight,KTile,CTile,UsesCluster,ClusterSize,KGroups,Kwaves,Cfolds,Cwaves,RowFold,FoldProgram,FoldCompute,Reduction,Finalize,LatencyCycles,AreaCost,Score,ActivePEs,UtilizationPct",
    ]
    for layer, candidates in zip(layers, all_candidates):
        ranked = sorted(
            candidates,
            key=lambda candidate: (
                candidate.score,
                candidate.area_cost,
                candidate.latency_cycles,
                -int(candidate.uses_cluster),
                -candidate.cluster_size,
                -candidate.k_tile,
                -candidate.c_tile,
            ),
        )
        best = ranked[0]
        for rank, candidate in enumerate(ranked, start=1):
            rows.append(
                ",".join(
                    [
                        layer.name,
                        str(rank),
                        str(candidate == best),
                        "exhaustive_discrete_search",
                        str(hw.latency_weight),
                        str(hw.area_weight),
                        str(candidate.k_tile),
                        str(candidate.c_tile),
                        str(candidate.uses_cluster),
                        str(candidate.cluster_size),
                        str(candidate.k_groups),
                        str(candidate.k_waves),
                        str(candidate.c_folds),
                        str(candidate.c_waves),
                        str(candidate.row_fold),
                        str(candidate.fold_program_cycles),
                        str(candidate.fold_compute_cycles),
                        str(candidate.reduction_cycles),
                        str(candidate.finalize_cycles),
                        str(candidate.latency_cycles),
                        str(candidate.area_cost),
                        f"{candidate.score:.6f}",
                        str(candidate.active_pes),
                        f"{candidate.utilization_pct:.4f}",
                    ]
                )
            )
    return "\n".join(rows) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="Auto-generate legal MAESTRO mapping files from a latency cost function.")
    parser.add_argument("--input", default="yolo26n_int8 .tflite", help="Input MAESTRO .m, ONNX, or TFLite model file.")
    parser.add_argument("--output", default="Yolo26n_auto_full_mapping.m", help="Generated optimized mapping file.")
    parser.add_argument("--report", default="Yolo26n_auto_full_mapping_report.csv", help="CSV report for selected mappings.")
    parser.add_argument("--trace", default="Yolo26n_auto_full_mapping_trace.csv", help="CSV trace of every evaluated candidate.")
    parser.add_argument("--parsed-spec", default=None, help="Optional parsed .m spec when input is ONNX/TFLite.")
    parser.add_argument("--network-name", default=None, help="Override parsed network name for ONNX/TFLite inputs.")
    parser.add_argument("--skip-depthwise", action="store_true", help="Skip depthwise convolution layers from model inputs.")
    parser.add_argument("--layer-index", type=int, default=None, help="Optimize only one layer index.")
    parser.add_argument("--max-layers", type=int, default=None, help="Optimize only the first N parsed layers.")
    parser.add_argument("--total-pes", type=int, default=512)
    parser.add_argument("--pe-rows", type=int, default=64)
    parser.add_argument("--neurons-per-pe", type=int, default=8)
    parser.add_argument("--program-ports", type=int, default=512)
    parser.add_argument("--max-cluster-size", type=int, default=16)
    parser.add_argument("--max-cluster-pes", type=int, default=16)
    parser.add_argument("--min-cluster-c", type=int, default=8)
    parser.add_argument("--min-cluster-k", type=int, default=16)
    parser.add_argument("--latency-weight", type=float, default=1.0)
    parser.add_argument("--area-weight", type=float, default=0.001)
    args = parser.parse_args()
    if args.layer_index is not None and args.max_layers is not None:
        parser.error("Use either --layer-index or --max-layers, not both.")
    if args.max_layers is not None and args.max_layers < 1:
        parser.error("--max-layers must be at least 1.")

    hw = HardwareConfig(
        total_pes=args.total_pes,
        pe_rows=args.pe_rows,
        neurons_per_pe=args.neurons_per_pe,
        program_ports=args.program_ports,
        max_cluster_size=args.max_cluster_size,
        max_cluster_pes=args.max_cluster_pes,
        min_cluster_c=args.min_cluster_c,
        min_cluster_k=args.min_cluster_k,
        latency_weight=args.latency_weight,
        area_weight=args.area_weight,
    )

    input_path = Path(args.input)
    if is_model_input(input_path):
        from ONNX_graph_parser import parse_model, render_maestro_spec

        network_name, graph_layers = parse_model(
            input_path,
            network_name=args.network_name,
            include_depthwise=not args.skip_depthwise,
        )
        if args.layer_index is not None:
            graph_layers = [graph_layers[args.layer_index]]
        elif args.max_layers is not None:
            graph_layers = graph_layers[: args.max_layers]
        if args.parsed_spec:
            Path(args.parsed_spec).write_text(render_maestro_spec(network_name, graph_layers), encoding="utf-8")
        layers = graph_layers_to_layer_specs(graph_layers)
    else:
        network_name, layers = parse_layers(input_path)
        if args.layer_index is not None:
            layers = [layers[args.layer_index]]
        elif args.max_layers is not None:
            layers = layers[: args.max_layers]

    optimization_results = [optimize_layer(layer, hw) for layer in layers]
    selected = [result[0] for result in optimization_results]
    all_candidates = [result[1] for result in optimization_results]
    Path(args.output).write_text(render_mapping(network_name, layers, selected), encoding="utf-8")
    Path(args.report).write_text(render_report(layers, selected, hw), encoding="utf-8")
    Path(args.trace).write_text(render_trace(layers, all_candidates, hw), encoding="utf-8")

    for layer, candidate in zip(layers, selected):
        cluster_text = f"Cluster({candidate.cluster_size},P), " if candidate.uses_cluster else ""
        print(
            f"{layer.name}: {layer.loop_order}, "
            f"SpatialMap({candidate.k_tile},{candidate.k_tile}) K, "
            f"TemporalMap({candidate.c_tile},{candidate.c_tile}) C, "
            f"{cluster_text}"
            f"latency={candidate.latency_cycles} cycles, "
            f"area={candidate.area_cost} active PEs, "
            f"score={candidate.score:.6f}"
        )


if __name__ == "__main__":
    main()
