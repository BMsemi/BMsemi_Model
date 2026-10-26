from __future__ import annotations

from typing import Dict, List

import torch.nn as nn

from .onnx_ir import ONNXIR


DEFAULT_TILE_WIDTH = 16


def tile(ir: ONNXIR) -> ONNXIR:
    tiles: List[Dict[str, int | str]] = []
    op_index = 0
    for module_name, module in ir.iter_compute_modules():
        if isinstance(module, nn.Conv2d):
            total_channels = int(module.out_channels)
            kind = "conv"
        elif isinstance(module, nn.Linear):
            total_channels = int(module.out_features)
            kind = "linear"
        else:
            continue

        tile_width = min(DEFAULT_TILE_WIDTH, total_channels)
        module_shape = ir.module_output_shape(module_name)
        for tile_index, start in enumerate(range(0, total_channels, tile_width)):
            end = min(start + tile_width, total_channels)
            tiles.append(
                {
                    "op_index": op_index,
                    "module_name": module_name,
                    "kind": kind,
                    "tile_index": tile_index,
                    "channel_start": start,
                    "channel_end": end,
                    "tile_channels": end - start,
                    "output_shape": module_shape,
                }
            )
        op_index += 1

    ir.tiles = tiles
    if ir.logger is not None:
        ir.logger.check(
            name="op_tiler_tiles_created",
            condition=len(tiles) > 0,
            details="Partitioned compute ops into hardware-sized tiles.",
            payload={"num_tiles": len(tiles), "tile_width": DEFAULT_TILE_WIDTH},
        )
    return ir
