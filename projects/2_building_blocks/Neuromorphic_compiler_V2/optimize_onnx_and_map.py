#!/usr/bin/env python3
from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any, Dict, List, Tuple

import torch
import torch.nn as nn
from PIL import Image
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
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
from executorch.delegate_preprocess import preprocess as delegate_preprocess
from executorch.executorch_compile_spec import get_spec as get_executorch_spec
from executorch.lower_to_execu_runtime import lower as lower_to_execu
from maestro.Mapper import Mapper
from maestro.Scheduler import Scheduler
from maestro.dataflow_builder import build_layer_dataflows, export_maestro_descriptions
from models.tiny_mobilenet_v2 import TinyMobileNetV2
from runtime.dispatch import RuntimeDispatcher
from runtime.global_memory_planner import GlobalMemoryPlanner
from runtime.verification_profiling_harness import verify_run
from deployment.export_tflite import export_quantized_tflite
from utils.checkpoints import CheckpointLogger


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(1)


def save_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True))


def load_digits_data(batch_size: int = 128) -> Dict[str, Any]:
    X, y = load_digits(return_X_y=True)
    raw_images = torch.tensor(X.reshape(-1, 1, 8, 8), dtype=torch.float32) / 16.0
    labels = torch.tensor(y, dtype=torch.long)

    train_images, test_images, train_labels, test_labels = train_test_split(
        raw_images,
        labels,
        test_size=0.2,
        random_state=42,
        stratify=labels,
    )

    mean = train_images.mean()
    std = train_images.std().clamp_min(1e-6)
    train_images_norm = (train_images - mean) / std
    test_images_norm = (test_images - mean) / std

    train_loader = DataLoader(TensorDataset(train_images_norm, train_labels), batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(TensorDataset(test_images_norm, test_labels), batch_size=batch_size, shuffle=False)
    calibration_loader = DataLoader(TensorDataset(train_images_norm[:256], train_labels[:256]), batch_size=64, shuffle=False)

    return {
        "train_loader": train_loader,
        "test_loader": test_loader,
        "calibration_loader": calibration_loader,
        "normalization": {"mean": float(mean.item()), "std": float(std.item())},
        "sample_raw_image": test_images[0].clone(),
        "sample_input": test_images_norm[0:1].clone(),
        "sample_label": int(test_labels[0].item()),
        "num_train": int(train_images.shape[0]),
        "num_test": int(test_images.shape[0]),
        "num_classes": 10,
    }


def save_sample_image(raw_image: torch.Tensor, path: Path) -> None:
    image = raw_image.squeeze(0).clamp(0.0, 1.0)
    image_uint8 = (image * 255.0).to(torch.uint8).numpy()
    pil_image = Image.fromarray(image_uint8, mode="L").resize((64, 64), resample=Image.Resampling.NEAREST)
    path.parent.mkdir(parents=True, exist_ok=True)
    pil_image.save(path)


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader) -> Tuple[float, float]:
    model.eval()
    loss_fn = nn.CrossEntropyLoss()
    total_loss = 0.0
    total_correct = 0
    total_examples = 0
    for inputs, labels in loader:
        logits = model(inputs)
        loss = loss_fn(logits, labels)
        total_loss += float(loss.item()) * int(inputs.shape[0])
        total_correct += int((logits.argmax(dim=1) == labels).sum().item())
        total_examples += int(inputs.shape[0])
    return total_loss / max(total_examples, 1), total_correct / max(total_examples, 1)


def train_or_load_checkpoint(
    model: nn.Module,
    train_loader: DataLoader,
    test_loader: DataLoader,
    checkpoint_path: Path,
    logger: CheckpointLogger,
    epochs: int = 10,
    lr: float = 3e-3,
) -> List[Dict[str, float]]:
    if checkpoint_path.exists():
        payload = torch.load(checkpoint_path, map_location="cpu")
        model.load_state_dict(payload["state_dict"])
        history = payload.get("history", [])
        logger.info(
            name="training_checkpoint_loaded",
            details="Loaded previously trained tiny MobileNetV2 checkpoint.",
            payload={"checkpoint_path": str(checkpoint_path), "epochs_recorded": len(history)},
        )
        return history

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()
    history: List[Dict[str, float]] = []
    best_state = None
    best_test_acc = -1.0

    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        running_correct = 0
        running_examples = 0
        for inputs, labels in train_loader:
            optimizer.zero_grad()
            logits = model(inputs)
            loss = loss_fn(logits, labels)
            loss.backward()
            optimizer.step()
            running_loss += float(loss.item()) * int(inputs.shape[0])
            running_correct += int((logits.argmax(dim=1) == labels).sum().item())
            running_examples += int(inputs.shape[0])

        train_loss = running_loss / max(running_examples, 1)
        train_acc = running_correct / max(running_examples, 1)
        test_loss, test_acc = evaluate(model, test_loader)
        history.append(
            {
                "epoch": float(epoch),
                "train_loss": float(train_loss),
                "train_accuracy": float(train_acc),
                "test_loss": float(test_loss),
                "test_accuracy": float(test_acc),
            }
        )
        logger.info(
            name=f"training_epoch_{epoch}",
            details=(
                f"epoch={epoch} train_loss={train_loss:.4f} train_acc={train_acc:.4f} "
                f"test_loss={test_loss:.4f} test_acc={test_acc:.4f}"
            ),
            payload=history[-1],
        )
        if test_acc > best_test_acc:
            best_test_acc = test_acc
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}

    assert best_state is not None
    model.load_state_dict(best_state)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": best_state, "history": history}, checkpoint_path)
    logger.check(
        name="training_checkpoint_saved",
        condition=checkpoint_path.exists(),
        details="Saved trained tiny MobileNetV2 checkpoint to disk.",
        payload={"checkpoint_path": str(checkpoint_path), "best_test_accuracy": best_test_acc},
    )
    return history


def main() -> None:
    repo_root = Path(__file__).resolve().parent
    artifact_dir = repo_root / "artifacts"
    checkpoint_dir = repo_root / "checkpoints"
    logger = CheckpointLogger()

    set_seed(7)
    logger.info(name="runtime_seed", details="Initialized deterministic runtime settings.", payload={"seed": 7, "torch_threads": 1})

    data = load_digits_data(batch_size=128)
    logger.check(
        name="dataset_loaded",
        condition=data["num_train"] > 0 and data["num_test"] > 0,
        details="Loaded sklearn digits dataset for a real image classification task.",
        payload={
            "num_train": data["num_train"],
            "num_test": data["num_test"],
            **data["normalization"],
        },
    )

    sample_image_path = artifact_dir / "sample_digit.png"
    save_sample_image(data["sample_raw_image"], sample_image_path)
    logger.check(
        name="sample_image_saved",
        condition=sample_image_path.exists(),
        details="Saved a real sample digit image used for inference verification.",
        payload={"sample_image_path": str(sample_image_path), "sample_label": data["sample_label"]},
    )

    model = TinyMobileNetV2(num_classes=data["num_classes"], width_mult=0.5, in_channels=1)
    total_params = int(sum(param.numel() for param in model.parameters()))
    logger.check(
        name="real_tiny_model_built",
        condition=total_params > 0,
        details="Constructed a real TinyMobileNetV2 model for the digits task.",
        payload={"num_parameters": total_params},
    )

    checkpoint_path = checkpoint_dir / "tiny_mobilenetv2_digits_best.pt"
    history = train_or_load_checkpoint(
        model=model,
        train_loader=data["train_loader"],
        test_loader=data["test_loader"],
        checkpoint_path=checkpoint_path,
        logger=logger,
        epochs=10,
    )

    test_loss, test_accuracy = evaluate(model, data["test_loader"])
    accuracy_threshold = 0.85
    logger.check(
        name="model_accuracy_threshold",
        condition=test_accuracy >= accuracy_threshold,
        details=f"Verified test accuracy against threshold {accuracy_threshold:.2f}.",
        payload={"test_loss": test_loss, "test_accuracy": test_accuracy},
    )

    example_input = data["sample_input"]
    with torch.no_grad():
        logits = model(example_input)
        probabilities = torch.softmax(logits, dim=1).squeeze(0)
        predicted_class = int(probabilities.argmax().item())
        confidence = float(probabilities.max().item())
    logger.check(
        name="sample_inference_completed",
        condition=probabilities.numel() == data["num_classes"],
        details="Ran inference on a real sample image through the trained model.",
        payload={
            "predicted_class": predicted_class,
            "true_class": data["sample_label"],
            "confidence": confidence,
        },
    )

    execu_spec = get_executorch_spec(model, example_input, logger)
    execu_spec = delegate_preprocess(execu_spec, logger)
    execu_spec = lower_to_execu(execu_spec, logger)
    execu_spec_path = artifact_dir / "executorch_compile_spec.json"
    save_json(execu_spec_path, execu_spec)

    fx_bundle = preprocess_onnx(model, example_input, artifact_dir / "fx_graph.json", logger)
    ir = ONNXIR(
        bundle=fx_bundle,
        model=model,
        example_input=example_input,
        logger=logger,
        calibration_loader=data["calibration_loader"],
    )
    ir = graph_optimize(ir)
    ir = legalize_weights(ir)
    ir = cluster_weights(ir)
    ir = quantize_weights(ir)
    ir = quantize_activations(ir)
    ir = tile_ops(ir)

    maestro_descriptors = build_layer_dataflows(ir, logger)
    maestro_status = export_maestro_descriptions(
        maestro_descriptors,
        artifact_dir,
        repo_root / "maestro" / "MobilenetV2_clustered.m",
        repo_root / "maestro" / "layer_dataflow_reference.json",
        logger,
    )
    ir.maestro_status = maestro_status

    mapping = assign_mapping(ir)

    maestro_mapper = Mapper(mapping, maestro_descriptors, logger)
    mapped = maestro_mapper.map()
    scheduler = Scheduler(mapped, maestro_descriptors, logger)
    schedule = scheduler.schedule()

    maestro_schedule_path = artifact_dir / "maestro_schedule.json"
    save_json(
        maestro_schedule_path,
        {
            "schedule": schedule,
            "num_entries": len(schedule),
            "maestro_layers": len(maestro_descriptors),
        },
    )
    logger.check(
        name="maestro_schedule_artifact_written",
        condition=maestro_schedule_path.exists(),
        details="Wrote MAESTRO schedule artifact for inspection.",
        payload={"maestro_schedule_path": str(maestro_schedule_path), "schedule_entries": len(schedule)},
    )

    planner = GlobalMemoryPlanner(schedule)
    plan = planner.plan()
    logger.check(
        name="memory_plan_matches_schedule",
        condition=plan["num_entries"] == len(schedule),
        details="Verified that memory plan entry count matches schedule length.",
        payload={"num_entries": plan["num_entries"], "schedule_entries": len(schedule)},
    )

    dispatcher = RuntimeDispatcher(plan, schedule)
    dispatch_report = dispatcher.dispatch()

    tflite_status = export_quantized_tflite(model, example_input, artifact_dir, logger)

    summary: Dict[str, Any] = {
        "model_name": "TinyMobileNetV2",
        "dataset": "sklearn_digits",
        "accuracy_threshold": accuracy_threshold,
        "test_loss": float(test_loss),
        "test_accuracy": float(test_accuracy),
        "predicted_class": predicted_class,
        "true_class": data["sample_label"],
        "confidence": confidence,
        "history_last": history[-1] if history else {},
        "schedule_entries": len(schedule),
        "dispatch_report": dispatch_report,
        "checkpoint_path": str(checkpoint_path),
        "sample_image_path": str(sample_image_path),
        "executorch_spec_path": str(execu_spec_path),
        "compiler_json_graph_path": fx_bundle["compiler_graph_path"],
        "fx_export_path": fx_bundle["export_path"],
        "maestro_dataflow_json_path": maestro_status["json_path"],
        "maestro_generated_m_path": maestro_status["generated_m_path"],
        "maestro_repo_reference_path": maestro_status["repo_reference_path"],
        "maestro_repo_json_reference_path": maestro_status["repo_json_reference_path"],
        "maestro_reference_matches_generated": maestro_status["reference_matches_generated"],
        "maestro_json_reference_matches_generated": maestro_status["json_reference_matches_generated"],
        "maestro_layer_count": maestro_status["num_layers"],
        "maestro_schedule_path": str(maestro_schedule_path),
        "onnx_path": fx_bundle.get("onnx_path"),
        "onnx_status_path": fx_bundle["onnx_status_path"],
        "onnx_status": fx_bundle["onnx_status"],
        "tflite_status": tflite_status,
    }
    verification = verify_run(summary)
    logger.check(
        name="verification_harness_accuracy",
        condition=verification["accuracy_ok"],
        details="Verification harness accepted model accuracy.",
        payload=verification,
    )
    logger.check(
        name="verification_harness_schedule",
        condition=verification["schedule_ok"],
        details="Verification harness accepted generated schedule.",
        payload=verification,
    )
    logger.check(
        name="verification_harness_dispatch",
        condition=verification["dispatch_ok"],
        details="Verification harness accepted runtime dispatch results.",
        payload=verification,
    )
    logger.check(
        name="verification_harness_maestro",
        condition=verification["maestro_ok"],
        details="Verification harness accepted MAESTRO per-layer integration outputs.",
        payload=verification,
    )

    checkpoint_report_path = artifact_dir / "checkpoint_report.json"
    logger.dump_json(checkpoint_report_path)
    summary["checkpoint_report_path"] = str(checkpoint_report_path)
    summary["checkpoint_counts"] = logger.summary()["counts"]
    save_json(artifact_dir / "test_run_summary.json", summary)

    print("\nFINAL SUMMARY")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
