from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch.nn as nn

from compiler.onnx_ir import ONNXIR
from compiler.op_tiler import DEFAULT_TILE_WIDTH
from compiler.mapping_assign import NUM_PES
from utils.checkpoints import CheckpointLogger


def _safe_name(module_name: str) -> str:
    return module_name.replace('.', '_').replace('/', '_')


def _output_hw(shape: Optional[List[int]]) -> tuple[int, int]:
    if shape is None:
        return 1, 1
    if len(shape) >= 4:
        return int(shape[-2]), int(shape[-1])
    return 1, 1


def _conv_layer_kind(module: nn.Conv2d) -> str:
    if module.groups == module.in_channels == module.out_channels:
        return "DEPTHWISE_CONV"
    if module.kernel_size == (1, 1):
        return "POINTWISE_CONV"
    return "CONV"


def _build_directives(kind: str, k_dim: int, c_dim: int, r_dim: int, s_dim: int, tile_width: int) -> List[Dict[str, Any]]:
    if kind == "DEPTHWISE_CONV":
        channel_tile = 1
        return [
            {"directive": "SpatialMap", "size": min(tile_width, k_dim), "offset": min(tile_width, k_dim), "dim": "K"},
            {"directive": "TemporalMap", "size": channel_tile, "offset": channel_tile, "dim": "C"},
            {"directive": "TemporalMap", "size": r_dim, "offset": r_dim, "dim": "R"},
            {"directive": "TemporalMap", "size": s_dim, "offset": s_dim, "dim": "S"},
            {"directive": "Cluster", "size": min(NUM_PES, max(1, (k_dim + tile_width - 1) // tile_width)), "dim": "P"},
        ]
    c_tile = min(max(1, c_dim), DEFAULT_TILE_WIDTH)
    return [
        {"directive": "SpatialMap", "size": min(tile_width, k_dim), "offset": min(tile_width, k_dim), "dim": "K"},
        {"directive": "TemporalMap", "size": c_tile, "offset": c_tile, "dim": "C"},
        {"directive": "TemporalMap", "size": r_dim, "offset": r_dim, "dim": "R"},
        {"directive": "TemporalMap", "size": s_dim, "offset": s_dim, "dim": "S"},
        {"directive": "Cluster", "size": min(NUM_PES, max(1, (k_dim + tile_width - 1) // tile_width)), "dim": "P"},
    ]


def build_layer_dataflows(ir: ONNXIR, logger: Optional[CheckpointLogger] = None) -> List[Dict[str, Any]]:
    descriptors: List[Dict[str, Any]] = []
    compute_modules = list(ir.iter_compute_modules())
    compute_count = len(compute_modules)

    for op_index, (module_name, module) in enumerate(compute_modules):
        output_shape = ir.module_output_shape(module_name)
        layer_name = _safe_name(module_name)
        if isinstance(module, nn.Conv2d):
            y_dim, x_dim = _output_hw(output_shape)
            k_dim = int(module.out_channels)
            c_dim = int(module.in_channels)
            r_dim, s_dim = int(module.kernel_size[0]), int(module.kernel_size[1])
            stride_y, stride_x = int(module.stride[0]), int(module.stride[1])
            tile_width = min(DEFAULT_TILE_WIDTH, k_dim)
            layer_kind = _conv_layer_kind(module)
            dimensions = {
                "K": k_dim,
                "C": c_dim,
                "R": r_dim,
                "S": s_dim,
                "Y": y_dim,
                "X": x_dim,
            }
            extra = {
                "groups": int(module.groups),
                "stride_y": stride_y,
                "stride_x": stride_x,
                "padding_y": int(module.padding[0]),
                "padding_x": int(module.padding[1]),
            }
        elif isinstance(module, nn.Linear):
            layer_kind = "GEMM"
            k_dim = int(module.out_features)
            c_dim = int(module.in_features)
            r_dim = s_dim = 1
            y_dim = x_dim = 1
            tile_width = min(DEFAULT_TILE_WIDTH, k_dim)
            dimensions = {
                "K": k_dim,
                "C": c_dim,
                "R": 1,
                "S": 1,
                "Y": 1,
                "X": 1,
            }
            extra = {"groups": 1, "stride_y": 1, "stride_x": 1, "padding_y": 0, "padding_x": 0}
        else:  # pragma: no cover - iter_compute_modules already filters types
            continue

        descriptor: Dict[str, Any] = {
            "layer_name": layer_name,
            "module_name": module_name,
            "op_index": op_index,
            "layer_type": layer_kind,
            "dimensions": dimensions,
            "tile_width": tile_width,
            "expected_tile_count": (k_dim + tile_width - 1) // tile_width,
            "directives": _build_directives(layer_kind, k_dim, c_dim, r_dim, s_dim, tile_width),
            "output_shape": output_shape,
            **extra,
        }
        descriptors.append(descriptor)
        if logger is not None:
            logger.check(
                name=f"maestro_layer_dimensions_valid::{module_name}",
                condition=k_dim > 0 and c_dim > 0 and y_dim > 0 and x_dim > 0,
                details=f"Built MAESTRO per-layer dataflow descriptor for {module_name}.",
                payload={
                    "layer_name": layer_name,
                    "layer_type": layer_kind,
                    **dimensions,
                    "tile_width": tile_width,
                    "expected_tile_count": descriptor["expected_tile_count"],
                },
            )

    ir.maestro_dataflows = descriptors
    if logger is not None:
        logger.check(
            name="maestro_descriptor_count_matches_compute_modules",
            condition=len(descriptors) == compute_count,
            details="Built one MAESTRO descriptor per compute layer.",
            payload={"num_descriptors": len(descriptors), "compute_modules": compute_count},
        )
    return descriptors


def render_maestro_network(descriptors: List[Dict[str, Any]], network_name: str = "TinyMobileNetV2_Digits") -> str:
    lines: List[str] = []
    lines.append(f"Network {network_name} {{")
    for desc in descriptors:
        dims = desc["dimensions"]
        lines.append(f"Layer {desc['layer_name']} {{")
        lines.append(f"  Type: {desc['layer_type']}")
        lines.append(
            "  Dimensions { "
            f"K: {dims['K']}, C: {dims['C']}, Y: {dims['Y']}, X: {dims['X']}, R: {dims['R']}, S: {dims['S']} "
            "}"
        )
        lines.append(
            "  Stride { "
            f"Y: {desc['stride_y']}, X: {desc['stride_x']} "
            "}"
        )
        lines.append(
            "  Meta { "
            f"module: \"{desc['module_name']}\", groups: {desc['groups']}, expected_tiles: {desc['expected_tile_count']} "
            "}"
        )
        lines.append("  Dataflow {")
        for directive in desc["directives"]:
            if directive["directive"] == "Cluster":
                lines.append(f"    Cluster({directive['size']}, {directive['dim']});")
            else:
                lines.append(
                    f"    {directive['directive']}({directive['size']}, {directive['offset']}) {directive['dim']};"
                )
        lines.append("  }")
        lines.append("}")
    lines.append("}")
    return "\n".join(lines) + "\n"


def export_maestro_descriptions(
    descriptors: List[Dict[str, Any]],
    export_dir: str | Path,
    repo_maestro_file: str | Path,
    repo_json_reference: str | Path | None = None,
    logger: Optional[CheckpointLogger] = None,
) -> Dict[str, Any]:
    export_dir = Path(export_dir)
    export_dir.mkdir(parents=True, exist_ok=True)
    repo_maestro_file = Path(repo_maestro_file)
    repo_json_reference = Path(repo_json_reference) if repo_json_reference is not None else repo_maestro_file.with_name("layer_dataflow_reference.json")

    json_path = export_dir / "maestro_layer_dataflow.json"
    json_path.write_text(json.dumps({"layers": descriptors}, indent=2, sort_keys=True))

    network_text = render_maestro_network(descriptors)
    generated_m_path = export_dir / "MobilenetV2_clustered_generated.m"
    generated_m_path.write_text(network_text)

    repo_text = repo_maestro_file.read_text() if repo_maestro_file.exists() else ""
    reference_matches = repo_text.strip() == network_text.strip()
    generated_json_payload = {"layers": descriptors}
    repo_json_payload = None
    if repo_json_reference.exists():
        try:
            repo_json_payload = json.loads(repo_json_reference.read_text())
        except Exception:
            repo_json_payload = None
    json_reference_matches = repo_json_payload == generated_json_payload

    manifest = {
        "num_layers": len(descriptors),
        "json_path": str(json_path),
        "generated_m_path": str(generated_m_path),
        "repo_reference_path": str(repo_maestro_file),
        "repo_json_reference_path": str(repo_json_reference),
        "reference_matches_generated": reference_matches,
        "json_reference_matches_generated": json_reference_matches,
    }
    manifest_path = export_dir / "maestro_integration_status.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True))

    if logger is not None:
        logger.check(
            name="maestro_artifacts_written",
            condition=json_path.exists() and generated_m_path.exists() and manifest_path.exists(),
            details="Wrote MAESTRO per-layer JSON and .m dataflow artifacts.",
            payload={
                "json_path": str(json_path),
                "generated_m_path": str(generated_m_path),
                "manifest_path": str(manifest_path),
            },
        )
        logger.check(
            name="maestro_repo_reference_matches_generated",
            condition=reference_matches,
            details="Repo MAESTRO reference file matches generated per-layer dataflow description.",
            payload={
                "repo_reference_path": str(repo_maestro_file),
                "generated_m_path": str(generated_m_path),
            },
        )
        logger.check(
            name="maestro_repo_json_reference_matches_generated",
            condition=json_reference_matches,
            details="Repo MAESTRO JSON reference matches generated per-layer dataflow description.",
            payload={
                "repo_json_reference_path": str(repo_json_reference),
                "json_path": str(json_path),
            },
        )
    return manifest
