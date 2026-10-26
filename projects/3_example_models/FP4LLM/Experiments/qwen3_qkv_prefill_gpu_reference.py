from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import platform
import time
from datetime import datetime
from pathlib import Path
from typing import Any

os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import torch
from huggingface_hub import snapshot_download
from transformers import AutoModelForCausalLM, AutoTokenizer


DEFAULT_MODEL = "Qwen/Qwen3-1.7B"
DEFAULT_PROMPT = (
    "GPU reference prefill run for Qwen3-1.7B. Capture Q, K, and V tensors "
    "from the first three transformer layers for later crossbar verification."
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Capture Q/K/V prefill tensors for the first Qwen3 decoder layers on GPU."
    )
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--layers", type=int, default=3)
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--cache-dir", default=os.environ.get("HF_HUB_CACHE", str(Path.home() / ".cache" / "huggingface" / "hub")))
    parser.add_argument("--local-root", default=os.environ.get("HF_MODEL_ROOT", str(Path.home() / "hf_models")))
    parser.add_argument("--dtype", choices=("auto", "bfloat16", "float16", "float32"), default="auto")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max-input-tokens", type=int, default=128)
    parser.add_argument("--seed", type=int, default=1234)
    return parser.parse_args()


def select_dtype(dtype_name: str, device: torch.device) -> torch.dtype:
    if dtype_name == "bfloat16":
        return torch.bfloat16
    if dtype_name == "float16":
        return torch.float16
    if dtype_name == "float32":
        return torch.float32
    if device.type == "cuda" and torch.cuda.is_bf16_supported():
        return torch.bfloat16
    if device.type == "cuda":
        return torch.float16
    return torch.float32


def safe_name(value: str) -> str:
    return "".join(ch if ch.isalnum() else "_" for ch in value).strip("_")


def tensor_stats(tensor: torch.Tensor) -> dict[str, Any]:
    values = tensor.detach().float().cpu()
    flat = values.reshape(-1)
    return {
        "shape": list(tensor.shape),
        "dtype": str(tensor.dtype).replace("torch.", ""),
        "numel": int(tensor.numel()),
        "mean": float(flat.mean().item()),
        "std": float(flat.std(unbiased=False).item()) if flat.numel() > 1 else 0.0,
        "min": float(flat.min().item()),
        "max": float(flat.max().item()),
        "absmax": float(flat.abs().max().item()),
        "l2_norm": float(torch.linalg.vector_norm(flat).item()),
    }


def write_stats_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = [
        "layer",
        "projection",
        "shape",
        "dtype",
        "numel",
        "mean",
        "std",
        "min",
        "max",
        "absmax",
        "l2_norm",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def base_model_and_layers(model: torch.nn.Module) -> tuple[torch.nn.Module, str, torch.nn.ModuleList]:
    base = getattr(model, "model", None) or getattr(model, "transformer", None)
    if base is None:
        raise RuntimeError("Could not find the decoder backbone on the loaded model.")

    for attr in ("layers", "h"):
        layers = getattr(base, attr, None)
        if layers is not None:
            return base, attr, layers
    raise RuntimeError("Could not find decoder layers on the loaded model.")


def projection_modules(layer: torch.nn.Module) -> dict[str, torch.nn.Module]:
    attn = getattr(layer, "self_attn", None) or getattr(layer, "attention", None)
    if attn is None:
        raise RuntimeError(f"Layer {layer.__class__.__name__} does not expose self attention.")
    modules = {}
    for projection in ("q_proj", "k_proj", "v_proj"):
        module = getattr(attn, projection, None)
        if module is None:
            raise RuntimeError(f"Attention block is missing {projection}.")
        modules[projection[0]] = module
    return modules


def main() -> None:
    args = parse_args()
    started = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = args.run_name or f"QKV_PREFILL_{safe_name(args.model)}_GPU_{started}"
    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested, but torch.cuda.is_available() is False.")

    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    dtype = select_dtype(args.dtype, device)

    model_local_dir = Path(args.local_root).expanduser() / args.model.replace("/", "--")
    print(f"run_name={run_name}", flush=True)
    print(f"model={args.model}", flush=True)
    print(f"cache_dir={args.cache_dir}", flush=True)
    print(f"local_model_dir={model_local_dir}", flush=True)
    print(f"out_dir={out_dir}", flush=True)

    snapshot_path = snapshot_download(
        repo_id=args.model,
        cache_dir=args.cache_dir,
        local_dir=str(model_local_dir),
        ignore_patterns=("*.h5", "*.msgpack", "*.ot", "flax_model.*", "tf_model.*"),
    )

    tokenizer = AutoTokenizer.from_pretrained(snapshot_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        snapshot_path,
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
    )
    model.eval()

    base, layers_attr, layers = base_model_and_layers(model)
    if args.layers < 1 or args.layers > len(layers):
        raise ValueError(f"--layers must be in [1, {len(layers)}], got {args.layers}")

    original_layer_count = len(layers)
    setattr(base, layers_attr, torch.nn.ModuleList(list(layers[: args.layers])))
    model.to(device)

    encoded = tokenizer(args.prompt, return_tensors="pt")
    if encoded["input_ids"].shape[1] > args.max_input_tokens:
        encoded = {key: value[:, : args.max_input_tokens] for key, value in encoded.items()}
    encoded = {key: value.to(device) for key, value in encoded.items()}

    captured: dict[tuple[int, str], torch.Tensor] = {}
    hooks: list[torch.utils.hooks.RemovableHandle] = []
    truncated_layers = getattr(base, layers_attr)

    for layer_index, layer in enumerate(truncated_layers):
        for projection_name, module in projection_modules(layer).items():
            def make_hook(index: int, name: str):
                def hook(_module: torch.nn.Module, _inputs: tuple[Any, ...], output: torch.Tensor) -> None:
                    if isinstance(output, tuple):
                        output_tensor = output[0]
                    else:
                        output_tensor = output
                    captured[(index, name)] = output_tensor.detach().cpu().contiguous()

                return hook

            hooks.append(module.register_forward_hook(make_hook(layer_index, projection_name)))

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize(device)
    t0 = time.perf_counter()
    with torch.inference_mode():
        outputs = base(
            input_ids=encoded["input_ids"],
            attention_mask=encoded.get("attention_mask"),
            use_cache=False,
            output_attentions=False,
            output_hidden_states=False,
        )
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elapsed_s = time.perf_counter() - t0

    for handle in hooks:
        handle.remove()

    last_hidden_state = outputs.last_hidden_state.detach().cpu().contiguous()
    expected = {(layer, projection) for layer in range(args.layers) for projection in ("q", "k", "v")}
    missing = sorted(expected.difference(captured))
    if missing:
        raise RuntimeError(f"Missing captured QKV tensors: {missing}")

    flat_tensors = {
        f"layer_{layer:02d}_{projection}": tensor
        for (layer, projection), tensor in sorted(captured.items())
    }
    flat_tensors["last_hidden_state_after_layer_02"] = last_hidden_state

    stats_rows: list[dict[str, Any]] = []
    stats_by_tensor: dict[str, dict[str, Any]] = {}
    for name, tensor in flat_tensors.items():
        stats = tensor_stats(tensor)
        stats_by_tensor[name] = stats
        if name.startswith("layer_"):
            _, layer_text, projection = name.split("_", maxsplit=2)
            stats_rows.append({"layer": int(layer_text), "projection": projection, **stats})

    qkv_pt_path = out_dir / f"{run_name}.pt"
    torch.save(
        {
            "metadata": {
                "run_name": run_name,
                "phase": "prefill",
                "artifact_name": "QKV_PREFILL",
                "reference_backend": "gpu",
                "crossbar_used": False,
                "model": args.model,
                "snapshot_path": str(snapshot_path),
                "layers_run": args.layers,
                "original_layer_count": original_layer_count,
                "prompt": args.prompt,
                "input_ids": encoded["input_ids"].detach().cpu(),
                "attention_mask": encoded.get("attention_mask", torch.ones_like(encoded["input_ids"])).detach().cpu(),
                "dtype": str(dtype).replace("torch.", ""),
                "device": str(device),
                "elapsed_s": elapsed_s,
            },
            "tensors": flat_tensors,
        },
        qkv_pt_path,
    )

    try:
        import numpy as np

        np_arrays = {name: tensor.float().numpy() for name, tensor in flat_tensors.items()}
        np.savez_compressed(out_dir / f"{run_name}.npz", **np_arrays)
    except Exception as exc:
        print(f"warning: NumPy export skipped: {exc}", flush=True)

    try:
        from safetensors.torch import save_file

        save_file(flat_tensors, out_dir / f"{run_name}.safetensors")
    except Exception as exc:
        print(f"warning: safetensors export skipped: {exc}", flush=True)

    write_stats_csv(out_dir / "qkv_stats.csv", stats_rows)

    metadata = {
        "run_name": run_name,
        "artifact_name": "QKV_PREFILL",
        "phase": "prefill",
        "reference_backend": "gpu",
        "crossbar_used": False,
        "model": args.model,
        "snapshot_path": str(snapshot_path),
        "layers_run": args.layers,
        "original_layer_count": original_layer_count,
        "prompt": args.prompt,
        "prompt_sha256": hashlib.sha256(args.prompt.encode("utf-8")).hexdigest(),
        "input_shape": list(encoded["input_ids"].shape),
        "input_ids": encoded["input_ids"].detach().cpu().tolist(),
        "attention_mask": encoded.get("attention_mask", torch.ones_like(encoded["input_ids"])).detach().cpu().tolist(),
        "dtype": str(dtype).replace("torch.", ""),
        "device": str(device),
        "cuda_available": torch.cuda.is_available(),
        "cuda_version": torch.version.cuda,
        "gpu_name": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "gpu_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
        "peak_cuda_memory_bytes": torch.cuda.max_memory_allocated(device) if device.type == "cuda" else 0,
        "elapsed_s": elapsed_s,
        "python": platform.python_version(),
        "platform": platform.platform(),
        "hostname": platform.node(),
        "stats": stats_by_tensor,
        "artifacts": {
            "torch": qkv_pt_path.name,
            "numpy": f"{run_name}.npz",
            "safetensors": f"{run_name}.safetensors",
            "stats_csv": "qkv_stats.csv",
            "metadata": "metadata.json",
        },
    }
    (out_dir / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    (out_dir / "README.txt").write_text(
        "QKV_PREFILL GPU reference\n"
        "=========================\n\n"
        "This folder is a GPU-only prefill reference for later crossbar verification.\n"
        "It loads Qwen/Qwen3-1.7B from Hugging Face, truncates the decoder stack to the\n"
        "first three transformer layers, runs one full-prompt prefill pass on CUDA, and\n"
        "captures the q_proj, k_proj, and v_proj outputs for layers 0, 1, and 2.\n\n"
        "Primary artifact: the .pt file contains metadata plus exact PyTorch tensors.\n"
        "Convenience artifacts: .npz stores float32 NumPy copies, .safetensors stores\n"
        "flat tensors, qkv_stats.csv stores lightweight numerical summaries, and\n"
        "metadata.json records the prompt/input IDs, GPU, dtype, and timing.\n\n"
        "crossbar_used=false by design.\n",
        encoding="utf-8",
    )
    print(f"wrote={out_dir}", flush=True)
    print(f"torch_artifact={qkv_pt_path}", flush=True)
    print(f"elapsed_s={elapsed_s:.4f}", flush=True)


if __name__ == "__main__":
    main()
