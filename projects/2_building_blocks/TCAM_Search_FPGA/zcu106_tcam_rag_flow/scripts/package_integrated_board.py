from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Package shared-PE TCAM RAG plus XForge YOLO for ZCU106."
    )
    parser.add_argument(
        "--xforge",
        type=Path,
        default=Path(__file__).resolve().parent.parent / "xforge_zcu106",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).resolve().parent.parent / "deploy" / "tcam_yolo_shared_pe",
    )
    args = parser.parse_args()
    root = Path(__file__).resolve().parent.parent
    xforge = args.xforge.resolve()
    source_bundle = xforge / "x2_zcu106" / "deploy" / "zcu106_hybrid_board"
    output = args.output.resolve()
    if not source_bundle.is_dir():
        raise FileNotFoundError(f"Missing XForge board bundle: {source_bundle}")

    output.mkdir(parents=True, exist_ok=True)
    shutil.copytree(source_bundle, output, dirs_exist_ok=True)
    for package_name in ("x2compiler", "runtime"):
        shutil.copytree(
            xforge / "x2_zcu106" / package_name,
            output / "x2_zcu106" / package_name,
            dirs_exist_ok=True,
            ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
        )
    shutil.copy2(
        xforge / "x2_zcu106" / "__init__.py",
        output / "x2_zcu106" / "__init__.py",
    )
    shutil.copytree(
        root / "integrated",
        output / "integrated",
        dirs_exist_ok=True,
        ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
    )
    shutil.copy2(root / "run_tcam_yolo.py", output / "run_tcam_yolo.py")
    output_scripts = output / "scripts"
    output_scripts.mkdir(parents=True, exist_ok=True)
    for script_name in (
        "model_rram_int4_programming_accuracy.py",
        "run_rram_int4_accuracy_ps.py",
        "probe_rram_int4_accuracy_pl.py",
    ):
        script_path = root / "scripts" / script_name
        if script_path.exists():
            shutil.copy2(script_path, output_scripts / script_name)
    output_rtl = output / "rtl"
    output_rtl.mkdir(parents=True, exist_ok=True)
    for rtl_name in (
        "rram_int4_programming_accuracy_lut.sv",
        "axi_rram_int4_accuracy.sv",
    ):
        rtl_path = root / "rtl" / rtl_name
        if rtl_path.exists():
            shutil.copy2(rtl_path, output_rtl / rtl_name)

    source_core = xforge / "YOLO26n-TFLite-FPGA-Partner-Project" / "core"
    output_core = output / "YOLO26n-TFLite-FPGA-Partner-Project" / "core"
    output_core.mkdir(parents=True, exist_ok=True)
    for core_file in (
        "Crossbar_array.py",
        "pl_dma_backend.py",
        "axi_x2_mvm_cma.c",
        "libxlnk_cma.h",
    ):
        source_file = source_core / core_file
        if source_file.exists():
            shutil.copy2(source_file, output_core / core_file)
    manifest = {
        "schema_version": 1,
        "platform": "ZCU106",
        "pipeline": (
            "TCAM Engine0/1 -> Layer0 Engine2/3/4/5 on miss -> remaining Conv2D emulator"
        ),
        "entrypoint": "run_tcam_yolo.py",
        "overlay": "overlay/zcu106_axi_x2_mvm.bit",
        "overlay_sha256": hashlib.sha256(
            (output / "overlay" / "zcu106_axi_x2_mvm.bit").read_bytes()
        ).hexdigest(),
        "accelerator_version_minimum": "0x00020024",
        "tcam_capacity": 32,
        "tcam_physical_shape_per_pe": [32, 32],
        "tcam_transaction_shape": [32, 16],
        "tcam_column_passes": 2,
        "tcam_engine_count": 2,
        "yolo_layer0_engine_count": 4,
        "total_engine_count": 6,
        "tcam_internal_lanes_per_engine": 8,
        "yolo_layer0_internal_lanes_per_engine": 4,
        "constants_manifest_additions": {
            "block_name": "Constants & Manifest + TCAM Cache Index",
            "stores": [
                "MLIR tile constants",
                "TCAM keys",
                "care masks",
                "chunk IDs",
                "slot -> chunk mapping",
            ],
        },
        "hybrid_runtime_scheduler_additions": {
            "block_name": (
                "Hybrid Runtime Scheduler + TCAM Hit/Miss + Image Fingerprint"
            ),
            "responsibilities": [
                "generate image fingerprint",
                "batch TCAM queries",
                "route fitting TCAM queries to Engine 0/1",
                "route YOLO Layer 0/1/2/3 miss-path work to Engine 2/3/4/5",
                "search extra TCAM banks/pages on Engine 0/1",
                "skip YOLO/MLIR on TCAM hit",
                "run YOLO/MLIR on TCAM miss",
                "store key -> chunk ID after miss",
            ],
        },
        "rram_int4_accuracy_model": {
            "ps_runner": "scripts/run_rram_int4_accuracy_ps.py",
            "ps_model": "scripts/model_rram_int4_programming_accuracy.py",
            "pl_probe": "scripts/probe_rram_int4_accuracy_pl.py",
            "pl_rtl": [
                "rtl/rram_int4_programming_accuracy_lut.sv",
                "rtl/axi_rram_int4_accuracy.sv",
            ],
            "register_base_when_integrated": "0xA0060000",
            "role": (
                "Models 4-bit RRAM weight programming accuracy from the "
                "diagnostic preparation-pulse heatmap. PS uses Monte Carlo; "
                "PL uses a deterministic LUT/verify-pulse model."
            ),
        },
        "engine_partition": {
            "Engine 0": "dedicated 32x32 TCAM slots 0-15 with TDC enabled",
            "Engine 1": "dedicated 32x32 TCAM slots 16-31 with TDC enabled",
            "Engine 2": "dedicated YOLO Layer 0 on TCAM miss, 4 lanes, TDC disabled",
            "Engine 3": "dedicated YOLO Layer 1 on TCAM miss, 4 lanes, TDC disabled",
            "Engine 4": "dedicated YOLO Layer 2 on TCAM miss, 4 lanes, TDC disabled",
            "Engine 5": "dedicated YOLO Layer 3 on TCAM miss, 4 lanes, TDC disabled",
            "encoding": "q/qbar rows, A/B columns, B-A + TDC on Engine 0/1",
            "miss_latency_policy": (
                "TCAM cells remain resident; miss-path YOLO does not erase/reprogram "
                "TCAM Engine 0/1."
            ),
        },
    }
    (output / "integrated_package_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    print(f"PACKAGED {output}")


if __name__ == "__main__":
    main()
