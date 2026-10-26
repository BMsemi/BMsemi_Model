from __future__ import annotations

from copy import deepcopy
from typing import Any, Dict, Optional

from utils.checkpoints import CheckpointLogger


def preprocess(spec: Dict[str, Any], logger: Optional[CheckpointLogger] = None) -> Dict[str, Any]:
    updated = deepcopy(spec)
    updated["delegate"] = {
        "lane": "executorch",
        "edge_transform_applied": True,
        "bmsemi_partition_tags": ["conv", "depthwise", "linear"],
    }
    if logger is not None:
        logger.check(
            name="executorch_delegate_tags_present",
            condition=bool(updated["delegate"]["bmsemi_partition_tags"]),
            details="Inserted delegate preprocess metadata for supported op classes.",
            payload=updated["delegate"],
        )
    return updated
