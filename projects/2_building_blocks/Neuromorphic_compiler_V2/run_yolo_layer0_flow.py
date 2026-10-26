#!/usr/bin/env python3
from __future__ import annotations

import argparse
import ast
import json
import shutil
from pathlib import Path
from typing import Any, Dict

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, TensorDataset

from compiler.activation_quant import quantize as quantize_activations
from compiler.graph_level_optimizer import optimize as graph_optimize
from compiler.mapping_assign import assign as assign_mapping
from compiler.onnx_ir import ONNXIR
from compiler.op_tiler import tile as tile_ops
from compiler.preprocess_onnx import simplify as preprocess_onnx
from compiler.weight_cluster import cluster as cluster_weights
from compiler.weight_legalization import legalize as legalize_weights
from compiler.weight_quant import quantize as quantize_weights
from deployment.export_tflite import export_quantized_tflite
from executorch.delegate_preprocess import preprocess as delegate_preprocess
from executorch.executorch_compile_spec import get_spec as get_executorch_spec
from executorch.lower_to_execu_runtime import lower as lower_to_execu
from maestro.Mapper import Mapper
from maestro.Scheduler import Scheduler
from maestro.dataflow_builder import build_layer_dataflows, export_maestro_descriptions
from models.yolo_layer0 import YoloLayer0
from runtime.dispatch import RuntimeDispatcher
from runtime.global_memory_planner import GlobalMemoryPlanner
from utils.checkpoints import CheckpointLogger


DEFAULT_RUN_NAME = "yolo_layer0"



def save_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str), encoding="utf-8")



def _load_channel_txt(path: Path) -> np.ndarray:
    """Load a channel text file from the original YOLO example.

    The original files may include a short header. Any line whose first
    non-whitespace character is not a digit or '-' is ignored.
    """
    lines = path.read_text(encoding="utf-8").splitlines()
    data_lines = []
    for line in lines:
        stripped = line.strip()
        if not stripped:
            continue
        if stripped[0].isdigit() or stripped[0] == "-":
            data_lines.append(stripped)
    if not data_lines:
        raise ValueError(f"No numeric rows found in channel file: {path}")
    rows = [list(map(int, row.split())) for row in data_lines]
    arr = np.asarray(rows, dtype=np.float32)
    return arr



def load_rgb_example(source_dir: Path) -> torch.Tensor:
    """Load the sample RGB image used by the YOLO layer-0 example.

    Preference order:
      1. pre-generated channel text files
      2. ``bus.jpg`` resized to 640x640 and split into RGB channels
    """
    channel_files = {
        "R": source_dir / "bus_R_channel_640x640.txt",
        "G": source_dir / "bus_G_channel_640x640.txt",
        "B": source_dir / "bus_B_channel_640x640.txt",
    }
    if all(path.exists() and path.stat().st_size > 0 for path in channel_files.values()):
        channels = [_load_channel_txt(channel_files[name]) for name in ("R", "G", "B")]
        stacked = np.stack(channels, axis=0)
        return torch.tensor(stacked / 255.0, dtype=torch.float32)

    image_path = source_dir / "bus.jpg"
    if not image_path.exists():
        raise FileNotFoundError(
            "Could not find either channel text files or bus.jpg in YOLO source dir: "
            f"{source_dir}"
        )
    image = Image.open(image_path).convert("RGB").resize((640, 640))
    arr = np.asarray(image, dtype=np.float32) / 255.0
    chw = np.transpose(arr, (2, 0, 1))
    return torch.tensor(chw, dtype=torch.float32)



def build_calibration_loader(example_input: torch.Tensor) -> DataLoader:
    """Build a minimal calibration loader for activation quantization."""
    repeated = example_input.repeat(8, 1, 1, 1)
    dummy_labels = torch.zeros(repeated.shape[0], dtype=torch.long)
    dataset = TensorDataset(repeated, dummy_labels)
    return DataLoader(dataset, batch_size=2, shuffle=False)



def default_yolo_source_dir(repo_root: Path) -> Path:
    candidate = repo_root.parent / "Yolo26_INT8_mac"
    return candidate



def export_maestro_with_fallback(
    descriptors: Any,
    artifact_dir: Path,
    repo_m_path: Path,
    repo_json_path: Path,
    logger: CheckpointLogger,
    run_name: str,
) -> Dict[str, Any]:
    """Use the repo exporter when possible, otherwise fall back to simple dumps.

    This keeps the integration usable even before the committed reference JSON is
    updated for the first time.
    """
    try:
        return export_maestro_descriptions(
            descriptors,
            artifact_dir,
            repo_m_path,
            repo_json_path,
            logger,
        )
    except Exception as exc:  # pragma: no cover - defensive fallback for repo-specific helpers
        generated_json_path = artifact_dir / f"{run_name}_maestro_layer_dataflow.json"
        generated_m_path = artifact_dir / f"{run_name}_generated.m"
        integration_status_path = artifact_dir / f"{run_name}_maestro_integration_status.json"

        generated_json_path.write_text(
            json.dumps(descriptors, indent=2, default=str),
            encoding="utf-8",
        )
        shutil.copyfile(repo_m_path, generated_m_path)

        status = {
            "json_path": str(generated_json_path),
            "generated_m_path": str(generated_m_path),
            "repo_reference_path": str(repo_m_path),
            "repo_json_reference_path": str(repo_json_path),
            "reference_matches_generated": False,
            "json_reference_matches_generated": False,
            "num_layers": len(descriptors) if hasattr(descriptors, "__len__") else 0,
            "export_mode": "fallback",
            "export_error": str(exc),
            "status_path": str(integration_status_path),
        }
        save_json(integration_status_path, status)
        logger.info(
            name="maestro_export_fallback",
            details="MAESTRO export helper failed; wrote fallback artifacts instead.",
            payload=status,
        )
        return status



def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the Neuromorphic Compiler V2 flow on YOLO layer-0.")
    parser.add_argument(
        "--yolo-source-dir",
        type=Path,
        default=None,
        help=(
            "Directory containing the original YOLO one-layer assets: "
            "layer0_weights.txt, bias.txt, Layer0_padding.txt, and either "
            "bus_R/G/B_channel_640x640.txt files or bus.jpg. "
            "Defaults to Benchmarking/Yolo26_INT8_mac if present."
        ),
    )
    parser.add_argument(
        "--artifacts-subdir",
        type=str,
        default=DEFAULT_RUN_NAME,
        help="Name prefix/subdir used for generated artifacts.",
    )
    return parser.parse_args()



def main() -> None:
    args = parse_args()
    repo_root = Path(__file__).resolve().parent
    yolo_source_dir = args.yolo_source_dir or default_yolo_source_dir(repo_root)
    artifact_dir = repo_root / "artifacts"
    artifact_dir.mkdir(parents=True, exist_ok=True)

    logger = CheckpointLogger()
    logger.info(
        name="yolo_source_dir",
        details="Using YOLO source assets for one-layer integration.",
        payload={"yolo_source_dir": str(yolo_source_dir)},
    )

    model = YoloLayer0(yolo_source_dir)
    model.eval()

    rgb_example = load_rgb_example(yolo_source_dir)
    example_input = rgb_example.unsqueeze(0)
    calibration_loader = build_calibration_loader(example_input)

    with torch.no_grad():
        example_output = model(example_input)

    output_shape_ok = tuple(example_output.shape) == (1, 16, 320, 320)
    logger.check(
        name="yolo_layer0_forward",
        condition=output_shape_ok,
        details="Executed YOLO layer-0 forward pass and checked the expected output shape.",
        payload={
            "input_shape": list(example_input.shape),
            "output_shape": list(example_output.shape),
            "expected_output_shape": [1, 16, 320, 320],
        },
    )

    execu_spec = get_executorch_spec(model, example_input, logger)
    execu_spec = delegate_preprocess(execu_spec, logger)
    execu_spec = lower_to_execu(execu_spec, logger)
    execu_spec_path = artifact_dir / f"{args.artifacts_subdir}_executorch_compile_spec.json"
    save_json(execu_spec_path, execu_spec)

    fx_bundle = preprocess_onnx(
        model,
        example_input,
        artifact_dir / f"{args.artifacts_subdir}_fx_graph.json",
        logger,
        args.artifacts_subdir,
    )
    ir = ONNXIR(
        bundle=fx_bundle,
        model=model,
        example_input=example_input,
        logger=logger,
        calibration_loader=calibration_loader,
    )
    ir = graph_optimize(ir)
    ir = legalize_weights(ir)
    ir = cluster_weights(ir)
    ir = quantize_weights(ir)
    ir = quantize_activations(ir)
    ir = tile_ops(ir)

    maestro_descriptors = build_layer_dataflows(ir, logger)
    maestro_status = export_maestro_with_fallback(
        maestro_descriptors,
        artifact_dir,
        repo_root / "maestro" / "Yolo26n.m",
        repo_root / "maestro" / "yolo_layer0_layer_dataflow_reference.json",
        logger,
        args.artifacts_subdir,
    )

    maestro_json_alias = artifact_dir / f"{args.artifacts_subdir}_maestro_layer_dataflow.json"
    maestro_m_alias = artifact_dir / f"{args.artifacts_subdir}_generated.m"
    maestro_json_src = Path(str(maestro_status.get("json_path", ""))) if maestro_status.get("json_path") else None
    maestro_m_src = Path(str(maestro_status.get("generated_m_path", ""))) if maestro_status.get("generated_m_path") else None
    if maestro_json_src and maestro_json_src.exists() and maestro_json_src != maestro_json_alias:
        shutil.copyfile(maestro_json_src, maestro_json_alias)
    if maestro_m_src and maestro_m_src.exists() and maestro_m_src != maestro_m_alias:
        shutil.copyfile(maestro_m_src, maestro_m_alias)
    maestro_status["json_alias_path"] = str(maestro_json_alias)
    maestro_status["generated_m_alias_path"] = str(maestro_m_alias)

    ir.maestro_status = maestro_status

    mapping = assign_mapping(ir)
    maestro_mapper = Mapper(mapping, maestro_descriptors, logger)
    mapped = maestro_mapper.map()
    scheduler = Scheduler(mapped, maestro_descriptors, logger)
    schedule = scheduler.schedule()

    maestro_schedule_path = artifact_dir / f"{args.artifacts_subdir}_maestro_schedule.json"
    save_json(
        maestro_schedule_path,
        {
            "schedule": schedule,
            "num_entries": len(schedule),
            "maestro_layers": len(maestro_descriptors),
        },
    )
    logger.check(
        name="maestro_schedule_written",
        condition=maestro_schedule_path.exists(),
        details="Generated MAESTRO schedule for YOLO layer-0.",
        payload={"maestro_schedule_path": str(maestro_schedule_path), "num_entries": len(schedule)},
    )

    planner = GlobalMemoryPlanner(schedule)
    plan = planner.plan()
    logger.check(
        name="memory_plan_matches_schedule",
        condition=plan.get("num_entries") == len(schedule),
        details="Verified that memory plan entry count matches the schedule length.",
        payload={"plan_num_entries": plan.get("num_entries"), "schedule_entries": len(schedule)},
    )

    dispatcher = RuntimeDispatcher(plan, schedule)
    dispatch_report = dispatcher.dispatch()

    tflite_status = export_quantized_tflite(model, example_input, artifact_dir, logger)

    checkpoint_report_path = artifact_dir / f"{args.artifacts_subdir}_checkpoint_report.json"
    logger.dump_json(checkpoint_report_path)

    summary = {
        "model_name": "YoloLayer0",
        "source_dir": str(yolo_source_dir),
        "input_shape": list(example_input.shape),
        "output_shape": list(example_output.shape),
        "executorch_spec_path": str(execu_spec_path),
        "fx_graph_path": fx_bundle["export_path"],
        "compiler_graph_path": fx_bundle["compiler_graph_path"],
        "onnx_status": fx_bundle["onnx_status"],
        "onnx_status_path": fx_bundle["onnx_status_path"],
        "maestro_status": maestro_status,
        "maestro_schedule_path": str(maestro_schedule_path),
        "schedule_entries": len(schedule),
        "dispatch_report": dispatch_report,
        "tflite_status": tflite_status,
        "checkpoint_report_path": str(checkpoint_report_path),
        "checkpoint_counts": logger.summary().get("counts", {}),
    }
    summary_path = artifact_dir / f"{args.artifacts_subdir}_run_summary.json"
    save_json(summary_path, summary)

    print("\nFINAL SUMMARY")
    print(json.dumps(summary, indent=2, sort_keys=True, default=str))


if __name__ == "__main__":
    main()
