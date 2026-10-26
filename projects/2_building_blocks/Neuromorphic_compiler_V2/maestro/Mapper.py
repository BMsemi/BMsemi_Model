from __future__ import annotations

from collections import defaultdict
from typing import Any, Dict, List, Optional

from utils.checkpoints import CheckpointLogger


class Mapper:
    """Translate compiler tile assignments into MAESTRO-style work items.

    The mapper cross-checks that every compiler tile belongs to a valid
    per-layer MAESTRO descriptor and that tile coverage matches the descriptor's
    output-channel K dimension.
    """

    def __init__(
        self,
        mapping: Dict[str, List[Dict[str, Any]]],
        layer_dataflows: List[Dict[str, Any]],
        logger: Optional[CheckpointLogger] = None,
    ) -> None:
        self.mapping = mapping
        self.layer_dataflows = layer_dataflows
        self.logger = logger
        self.descriptor_by_module = {item["module_name"]: item for item in layer_dataflows}
        num_tiles = len(self.mapping.get("tiles", []))
        if self.logger is not None:
            self.logger.info(
                name="maestro_mapper_init",
                details="Initialized MAESTRO mapper with tile assignments and per-layer descriptors.",
                payload={"num_tiles": num_tiles, "num_descriptors": len(self.layer_dataflows)},
            )
        else:
            print(f"CHECKPOINT [INFO] maestro_mapper_init: received {num_tiles} tile assignments")

    def map(self) -> List[Dict[str, Any]]:
        mapped: List[Dict[str, Any]] = []
        coverage = defaultdict(int)
        missing_modules: List[str] = []

        for tile in self.mapping.get("tiles", []):
            descriptor = self.descriptor_by_module.get(tile["module_name"])
            if descriptor is None:
                missing_modules.append(tile["module_name"])
                continue
            coverage[tile["module_name"]] += int(tile["tile_channels"])
            mapped.append(
                {
                    "layer_name": descriptor["layer_name"],
                    "module_name": tile["module_name"],
                    "layer_type": descriptor["layer_type"],
                    "dimensions": descriptor["dimensions"],
                    "directives": descriptor["directives"],
                    "tile_index": tile["tile_index"],
                    "pe": tile["pe"],
                    "bank": tile["bank"],
                    "estimated_cycles": int(tile["estimated_work_units"]),
                    "tile_channels": int(tile["tile_channels"]),
                    "op_index": int(tile["op_index"]),
                }
            )

        if self.logger is not None:
            self.logger.check(
                name="maestro_mapper_all_tiles_have_descriptors",
                condition=not missing_modules,
                details="Every compiler tile matched a MAESTRO per-layer descriptor.",
                payload={"missing_modules": sorted(set(missing_modules))},
            )
            for module_name, descriptor in self.descriptor_by_module.items():
                self.logger.check(
                    name=f"maestro_tile_coverage_matches_k::{module_name}",
                    condition=coverage[module_name] == int(descriptor["dimensions"]["K"]),
                    details=f"Tile channel coverage matches K dimension for {module_name}.",
                    payload={
                        "covered_channels": coverage[module_name],
                        "expected_k": int(descriptor["dimensions"]["K"]),
                        "expected_tile_count": int(descriptor["expected_tile_count"]),
                    },
                )
            self.logger.check(
                name="maestro_mapper_output_nonempty",
                condition=len(mapped) > 0,
                details="Mapped compiler tiles into MAESTRO work items with attached dataflow directives.",
                payload={"num_mapped": len(mapped)},
            )
        else:
            print(f"CHECKPOINT [PASS] maestro_mapper_output_nonempty: mapped {len(mapped)} work items")
        return mapped
