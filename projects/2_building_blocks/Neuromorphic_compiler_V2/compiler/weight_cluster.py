from __future__ import annotations

from typing import Dict, List

import torch

from .onnx_ir import ONNXIR


def _cluster_summary(weight: torch.Tensor, num_clusters: int = 8) -> Dict[str, List[float]]:
    flat = weight.flatten().float()
    if flat.numel() == 0:
        return {"centroids": [], "counts": []}
    quantiles = torch.linspace(0.0, 1.0, steps=num_clusters)
    centroids = torch.quantile(flat, quantiles).tolist()
    bins = torch.bucketize(flat, torch.tensor(centroids[1:-1], dtype=flat.dtype))
    counts = torch.bincount(bins, minlength=num_clusters).tolist()
    return {
        "centroids": [float(x) for x in centroids],
        "counts": [int(x) for x in counts],
    }


def cluster(ir: ONNXIR) -> ONNXIR:
    summaries: Dict[str, Dict[str, List[float]]] = {}
    for name, module in ir.iter_compute_modules():
        summary = _cluster_summary(module.weight.detach().cpu())
        summaries[name] = summary
        if ir.logger is not None:
            ir.logger.check(
                name=f"weight_clusters_nonempty::{name}",
                condition=len(summary["centroids"]) > 0,
                details=f"Generated weight cluster summary for {name}.",
                payload={
                    "num_clusters": len(summary["centroids"]),
                    "first_centroid": summary["centroids"][0] if summary["centroids"] else None,
                },
            )
    ir.weight_clusters = summaries
    return ir
