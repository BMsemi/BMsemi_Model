from __future__ import annotations

from dataclasses import asdict
import json
from pathlib import Path
import time
from typing import Any, Callable

import numpy as np

from .cache import PersistentRagCache
from .fingerprint import identify_image


class XForgeYoloInference:
    """Execute the existing XForge compiled YOLO graph with routed backends."""

    def __init__(
        self,
        model: str | Path,
        artifacts: str | Path,
        backend: Any,
        *,
        confidence: float = 0.25,
        iou: float = 0.45,
        max_detections: int = 100,
        preprocess: str = "resize",
    ) -> None:
        from x2_zcu106.x2compiler.hybrid_runtime import HybridTFLiteExecutor

        self.backend = backend
        self.executor = HybridTFLiteExecutor(
            model,
            artifacts,
            backend,
            offload_conv_indices=None,
        )
        self.confidence = float(confidence)
        self.iou = float(iou)
        self.max_detections = int(max_detections)
        self.preprocess = preprocess

    def __call__(self, image: Path, output_directory: Path) -> dict[str, Any]:
        from x2_zcu106.x2compiler.hybrid_runtime import save_trace
        from x2_zcu106.x2compiler.postprocess import (
            decode_yolo,
            save_detection_artifacts,
        )

        output_directory.mkdir(parents=True, exist_ok=True)
        embedding_directory = output_directory / "fpga_layer_embeddings"
        set_embedding_directory = getattr(self.backend, "set_embedding_directory", None)
        if callable(set_embedding_directory):
            set_embedding_directory(embedding_directory)
        input_tensor, original_bgr, preprocess_meta = self.executor.preprocess_image(
            image, mode=self.preprocess
        )
        result = self.executor.run(input_tensor)
        raw_output = output_directory / "raw_output.npy"
        np.save(raw_output, result.output)
        trace_path = output_directory / "execution_trace.json"
        save_trace(trace_path, result)
        detections = decode_yolo(
            result.output,
            preprocess_meta,
            confidence_threshold=self.confidence,
            iou_threshold=self.iou,
            max_detections=self.max_detections,
        )
        artifacts = save_detection_artifacts(
            output_directory,
            original_bgr,
            detections,
            metadata={"preprocess": preprocess_meta},
        )
        embedding_manifest = getattr(self.backend, "embedding_manifest", None)
        fpga_layer_embeddings = (
            embedding_manifest() if callable(embedding_manifest) else []
        )
        return {
            "schema_version": 1,
            "image": image.name,
            "preprocess": preprocess_meta,
            "execution": result.summary(),
            "detections": [asdict(detection) for detection in detections],
            "detection_count": len(detections),
            "artifacts": {
                "raw_output": str(raw_output),
                "execution_trace": str(trace_path),
                "fpga_layer_embedding_dir": str(embedding_directory),
                **{name: str(path) for name, path in artifacts.items()},
            },
            "fpga_layer_embeddings": fpga_layer_embeddings,
            "dispatch": self.backend.diagnostics(),
        }


class IntegratedTcamYoloPipeline:
    """Lookup an image in TCAM RAG, otherwise run YOLO and populate the cache."""

    def __init__(
        self,
        cache: PersistentRagCache,
        inference: Callable[[Path, Path], dict[str, Any]],
        output_directory: str | Path,
        *,
        max_hamming_distance: int = 0,
    ) -> None:
        self.cache = cache
        self.inference = inference
        self.output_directory = Path(output_directory)
        self.output_directory.mkdir(parents=True, exist_ok=True)
        self.max_hamming_distance = int(max_hamming_distance)

    def run(self, image: str | Path) -> dict[str, Any]:
        image_path = Path(image).resolve()
        identity = identify_image(image_path)
        lookup = self.cache.lookup(
            identity,
            max_hamming_distance=self.max_hamming_distance,
        )
        tcam = {
            "visual_key": f"0x{identity.visual_key:04X}",
            "sha256": identity.sha256,
            "hit": lookup.hit,
            "reason": lookup.reason,
            "candidate_count": lookup.candidate_count,
            "match": asdict(lookup.match) if lookup.match is not None else None,
        }
        flow_blocks = {
            "constants_manifest": self.cache.manifest_metadata(),
            "hybrid_runtime_scheduler": {
                "block_name": (
                    "Hybrid Runtime Scheduler + TCAM Hit/Miss + Image Fingerprint"
                ),
                "image_fingerprint": f"0x{identity.visual_key:04X}",
                "decision": "TCAM HIT: cached detections, skip MLIR"
                if lookup.hit
                else "TCAM MISS: run YOLO/MLIR",
                "query_routing": {
                    "tcam_primary": "Engine 0/1 dedicated TCAM search",
                    "yolo_early_layers_on_miss": (
                        "Engine 2 -> Layer 0, Engine 3 -> Layer 1, "
                        "Engine 4 -> Layer 2, Engine 5 -> Layer 3"
                    ),
                    "fpga_embedding_storage": (
                        "each FPGA layer tensor is stored under fpga_layer_embeddings "
                        "and mirrored into DDR/L2"
                    ),
                    "tcam_capacity": (
                        "extra TCAM entries are searched as additional banks/pages "
                        "on Engine 0/1, not routed to the neuromorphic emulator"
                    ),
                    "miss_latency_policy": (
                        "TCAM cells are not erased on miss; YOLO early layers are "
                        "routed to Engine 2/3/4/5."
                    ),
                },
            },
        }
        if lookup.hit:
            return {
                "schema_version": 1,
                "path": "tcam_cache_hit",
                "yolo_skipped": True,
                "flow_blocks": flow_blocks,
                "tcam": tcam,
                "result": lookup.payload,
                "cache": self.cache.diagnostics(),
            }

        run_directory = self.output_directory / (
            f"{image_path.stem}_{identity.sha256[:12]}_{time.time_ns()}"
        )
        result = self.inference(image_path, run_directory)
        entry = self.cache.insert(identity, result, image_name=image_path.name)
        report = {
            "schema_version": 1,
            "path": "tcam_miss_then_yolo",
            "yolo_skipped": False,
            "flow_blocks": flow_blocks,
            "tcam": tcam,
            "inserted": asdict(entry),
            "result": result,
            "cache": self.cache.diagnostics(),
        }
        report_path = run_directory / "integrated_report.json"
        report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
        report["report"] = str(report_path)
        report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
        return report
