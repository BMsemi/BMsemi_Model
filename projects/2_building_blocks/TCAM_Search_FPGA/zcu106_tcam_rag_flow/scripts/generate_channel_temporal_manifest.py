from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_ARTIFACTS = (
    PROJECT_ROOT
    / "xforge_zcu106"
    / "x2_zcu106"
    / "compiler_artifacts"
    / "yolo26n_hybrid_calibrated"
)
DEFAULT_OP_LAYER_MAP = {2: 0, 6: 1, 9: 2, 13: 3}
DEFAULT_ENGINES = (2, 3, 4, 5)


def _load_manifest(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _schedule(
    *,
    op_index: int,
    layer_id: int,
    output_channels: int,
    channel_tile_size: int,
    engines: tuple[int, ...],
) -> list[dict[str, Any]]:
    tiles: list[dict[str, Any]] = []
    tile_id = 0
    for oc_start in range(0, output_channels, channel_tile_size):
        engine_slot = tile_id % len(engines)
        oc_end = min(oc_start + channel_tile_size, output_channels)
        tiles.append(
            {
                "tile_id": tile_id,
                "op_index": int(op_index),
                "logical_layer": int(layer_id),
                "time_step": int(tile_id // len(engines)),
                "engine": int(engines[engine_slot]),
                "oc_start": int(oc_start),
                "oc_end": int(oc_end),
                "channel_count": int(oc_end - oc_start),
                "output_offset": int(oc_start),
            }
        )
        tile_id += 1
    return tiles


def _annotate_op_manifest(
    *,
    artifacts: Path,
    op_index: int,
    layer_id: int,
    channel_tile_size: int,
    engines: tuple[int, ...],
) -> dict[str, Any]:
    op_dir = artifacts / f"op_{op_index:04d}_conv2d"
    manifest_path = op_dir / "manifest.json"
    manifest = _load_manifest(manifest_path)
    output_channels = int(manifest["logical_shape"]["n"])
    tiles = _schedule(
        op_index=op_index,
        layer_id=layer_id,
        output_channels=output_channels,
        channel_tile_size=channel_tile_size,
        engines=engines,
    )
    schedule = {
        "strategy": "spatial_temporal_output_channel_tiling",
        "scope": "first_four_yolo_conv2d_layers_on_fpga",
        "op_index": int(op_index),
        "logical_layer": int(layer_id),
        "output_channels": output_channels,
        "channel_tile_size": int(channel_tile_size),
        "engine_ids": list(engines),
        "time_steps": max(int(tile["time_step"]) for tile in tiles) + 1,
        "spatial_rule": "engine = engine_ids[tile_id mod engine_count]",
        "temporal_rule": "time_step = floor(tile_id / engine_count)",
        "tiles": tiles,
        "merge_rule": "write each tile into output[:, :, :, oc_start:oc_end]",
        "next_layer_rule": "next layer starts only after all tiles are merged in DDR/L2",
    }
    manifest.setdefault("metadata", {})[
        "spatial_temporal_channel_schedule"
    ] = schedule
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return schedule


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Annotate YOLO X2 artifacts with PE channel-tile scheduling."
    )
    parser.add_argument("--artifacts", type=Path, default=DEFAULT_ARTIFACTS)
    parser.add_argument("--channel-tile-size", type=int, default=16)
    parser.add_argument(
        "--engines",
        type=int,
        nargs="+",
        default=list(DEFAULT_ENGINES),
        help="YOLO PE engines used for output-channel tiles.",
    )
    args = parser.parse_args()

    artifacts = args.artifacts.resolve()
    if not artifacts.exists():
        raise FileNotFoundError(f"Artifact directory not found: {artifacts}")
    if args.channel_tile_size < 1:
        raise ValueError("--channel-tile-size must be positive")

    engines = tuple(int(engine) for engine in args.engines)
    if not engines:
        raise ValueError("--engines must contain at least one engine")

    schedules = [
        _annotate_op_manifest(
            artifacts=artifacts,
            op_index=op_index,
            layer_id=layer_id,
            channel_tile_size=args.channel_tile_size,
            engines=engines,
        )
        for op_index, layer_id in DEFAULT_OP_LAYER_MAP.items()
    ]
    top_level = {
        "schema_version": 1,
        "name": "yolo_first_four_layer_channel_temporal_schedule",
        "target": "zcu106 six-engine shared PE fabric",
        "tcam_engines": [0, 1],
        "yolo_tile_engines": list(engines),
        "channel_tile_size": int(args.channel_tile_size),
        "layers_on_fpga": [
            {
                "logical_layer": item["logical_layer"],
                "op_index": item["op_index"],
                "output_channels": item["output_channels"],
                "time_steps": item["time_steps"],
                "tiles": item["tiles"],
            }
            for item in schedules
        ],
        "remaining_layers": "legacy neuromorphic simulator / CPU path",
        "contract": (
            "TCAM hit skips MLIR/YOLO. TCAM miss runs the first four Conv2D "
            "layers using spatial output-channel tiles and temporal rounds, "
            "then stores merged layer embeddings in DDR/L2 for the simulator."
        ),
    }
    output = artifacts / "channel_temporal_manifest.json"
    output.write_text(json.dumps(top_level, indent=2), encoding="utf-8")
    print(output)


if __name__ == "__main__":
    sys.path.insert(0, str(PROJECT_ROOT))
    main()
