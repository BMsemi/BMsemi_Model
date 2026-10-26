from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

from integrated.arbiter import SharedPEArbiter
from integrated.cache import PersistentRagCache
from integrated.hybrid_backend import EarlyLayerMappedHardwareBackend
from integrated.legacy_emulator import LegacyNeuromorphicEmulatorBackend
from integrated.pe_tcam import DualPETcamSearch, NumpyMvmEngine
from integrated.pipeline import IntegratedTcamYoloPipeline, XForgeYoloInference


PROJECT_ROOT = Path(__file__).resolve().parent
ENGINE0_BASE = 0xA0000000
ENGINE_STRIDE = 0x00010000
RESERVED_BUFFER_BASE = 0x74000000
RESERVED_BUFFER_STRIDE = 0x00020000


def _default_xforge() -> Path:
    return PROJECT_ROOT / "xforge_zcu106"


def _default_cache_dir() -> Path:
    return PROJECT_ROOT / "rag_cache"


def _default_bitstream(xforge: Path) -> Path:
    local_overlay = (
        PROJECT_ROOT
        / "deploy"
        / "tcam_yolo_shared_pe"
        / "overlay"
        / "zcu106_axi_x2_mvm.bit"
    )
    if local_overlay.exists():
        return local_overlay
    return xforge / "x2_zcu106" / "deploy" / "zcu106_axi_x2_mvm.bit"


def _yolo_core_dir(xforge: Path) -> Path:
    candidates = (
        xforge / "YOLO26n-TFLite-FPGA-Partner-Project" / "core",
        PROJECT_ROOT / "YOLO26n-TFLite-FPGA-Partner-Project" / "core",
    )
    for candidate in candidates:
        if (candidate / "pl_dma_backend.py").exists():
            return candidate
    return candidates[0]


class CmaMappedX2Backend:
    """X2 compiler backend for an already-programmed ZCU106 PE engine."""

    max_batch = 32
    pe_count = 1

    def __init__(self, engine_id: int, core_dir: Path) -> None:
        self.engine_id = int(engine_id)
        self.mmio_base = ENGINE0_BASE + (self.engine_id * ENGINE_STRIDE)
        self.reserved_base = RESERVED_BUFFER_BASE + (
            self.engine_id * RESERVED_BUFFER_STRIDE
        )
        sys.path.insert(0, str(core_dir))
        from pl_dma_backend import AxiX2MvmCmaBackend

        self.driver = AxiX2MvmCmaBackend(
            lib_path=core_dir / f"libaxi_x2_mvm_cma_engine{self.engine_id}.so",
            source_path=core_dir / "axi_x2_mvm_cma.c",
            accel_base=self.mmio_base,
            reserved_base=self.reserved_base,
        )

    def run_batch(self, vectors, weights):
        return self.driver.run_mvm_i8_batch(vectors, weights)

    def run_mvm(self, vector, weights):
        return self.driver.run_mvm_i8(vector, weights)

    def diagnostics(self) -> dict[str, Any]:
        return {
            "backend": "cma_devmem_preprogrammed_pl",
            "engine_id": self.engine_id,
            "mmio_base": f"0x{self.mmio_base:08X}",
            "reserved_buffer_base": f"0x{self.reserved_base:08X}",
            "library": str(self.driver.lib_path),
            "stats": vars(self.driver.stats),
        }


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Run TCAM RAG lookup on Engine 0/1 and channel-tiled YOLO layers "
            "0/1/2/3 on Engine 2/3/4/5 after a miss."
        )
    )
    parser.add_argument("image", type=Path)
    parser.add_argument("--xforge", type=Path, default=_default_xforge())
    parser.add_argument("--mode", choices=("hardware", "software"), default="software")
    parser.add_argument("--model", type=Path, default=None)
    parser.add_argument("--artifacts", type=Path, default=None)
    parser.add_argument("--bitstream", type=Path, default=None)
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=_default_cache_dir(),
        help=(
            "Persistent TCAM/RAG cache directory. Reuse this directory across "
            "runs to make the second launch hit immediately."
        ),
    )
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "integrated_output")
    parser.add_argument("--max-hamming-distance", type=int, default=0)
    parser.add_argument("--preprocess", choices=("resize", "letterbox"), default="resize")
    parser.add_argument("--confidence", type=float, default=0.25)
    parser.add_argument("--iou", type=float, default=0.45)
    parser.add_argument(
        "--channel-tile-size",
        type=int,
        default=16,
        help=(
            "Maximum logical output channels assigned to one YOLO PE tile. "
            "Temporal rounds are created when a layer has more channels than "
            "engines * channel_tile_size."
        ),
    )
    parser.add_argument(
        "--no-overlay-download",
        action="store_true",
        help=(
            "In hardware mode, attach to an already-programmed PL overlay "
            "instead of re-downloading the bitstream from Linux/PYNQ."
        ),
    )
    args = parser.parse_args()

    xforge = args.xforge.resolve()
    sys.path.insert(0, str(xforge))
    from x2_zcu106.x2compiler.runtime import (
        PEArrayX2Backend,
        PynqX2Backend,
    )

    model = args.model or xforge / "model" / "yolo26n_int8.tflite"
    artifacts = (
        args.artifacts
        or xforge / "x2_zcu106" / "compiler_artifacts" / "yolo26n_hybrid_calibrated"
    )
    bitstream = args.bitstream or _default_bitstream(xforge)
    arbiter = SharedPEArbiter()

    if args.mode == "hardware":
        if args.no_overlay_download:
            core_dir = _yolo_core_dir(xforge)
            tcam_engines = (
                CmaMappedX2Backend(0, core_dir),
                CmaMappedX2Backend(1, core_dir),
            )
            yolo_layer_backends = {
                engine_id: CmaMappedX2Backend(engine_id, core_dir)
                for engine_id in (2, 3, 4, 5)
            }
        else:
            from x2_zcu106.runtime.axi_x2_mvm_driver import DualAxiX2MvmPynq
            from pynq import Overlay  # type: ignore

            overlay = Overlay(str(bitstream), download=True)
            tcam_pe = DualAxiX2MvmPynq(
                overlay=overlay,
                accel_names=("axi_x2_mvm", "axi_x2_mvm_1"),
                download=False,
            )
            tcam_engines = tcam_pe.engines
            yolo_layer_backends = {
                engine_id: PynqX2Backend(
                    DualAxiX2MvmPynq(
                        overlay=overlay,
                        accel_names=(f"axi_x2_mvm_{engine_id}",),
                        download=False,
                    )
                )
                for engine_id in (2, 3, 4, 5)
            }
    else:
        tcam_engines = (NumpyMvmEngine("pe0"), NumpyMvmEngine("pe1"))
        yolo_layer_backends = {
            2: PEArrayX2Backend(pe_count=1),
            3: PEArrayX2Backend(pe_count=1),
            4: PEArrayX2Backend(pe_count=1),
            5: PEArrayX2Backend(pe_count=1),
        }

    searcher = DualPETcamSearch(tcam_engines, arbiter=arbiter)
    cache = PersistentRagCache(args.cache_dir, searcher)
    inference_runner: XForgeYoloInference | None = None

    def inference(image: Path, output: Path):
        nonlocal inference_runner
        if inference_runner is None:
            routed_backend = EarlyLayerMappedHardwareBackend(
                yolo_layer_backends,
                LegacyNeuromorphicEmulatorBackend(),
                fpga_op_engine_map={2: 2, 6: 3, 9: 4, 13: 5},
                op_layer_map={2: 0, 6: 1, 9: 2, 13: 3},
                channel_tile_size=args.channel_tile_size,
                channel_tile_engines=(2, 3, 4, 5),
                arbiter=arbiter,
            )
            inference_runner = XForgeYoloInference(
                model,
                artifacts,
                routed_backend,
                confidence=args.confidence,
                iou=args.iou,
                preprocess=args.preprocess,
            )
        return inference_runner(image, output)

    pipeline = IntegratedTcamYoloPipeline(
        cache,
        inference,
        args.output_dir,
        max_hamming_distance=args.max_hamming_distance,
    )
    print(json.dumps(pipeline.run(args.image), indent=2))


if __name__ == "__main__":
    main()
