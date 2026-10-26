from __future__ import annotations

import argparse
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
    "GPU reference for Qwen3-1.7B transformer block one. Capture prefill and decode "
    "outputs for later crossbar verification."
)
DEFAULT_DECODE_TEXT = " next"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Capture one full Qwen3 transformer block on GPU for prefill and decode."
    )
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--decode-text", default=DEFAULT_DECODE_TEXT)
    parser.add_argument("--block-index", type=int, default=0, help="Zero-based transformer block index.")
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


def base_model_and_layers(model: torch.nn.Module) -> tuple[torch.nn.Module, str, torch.nn.ModuleList]:
    base = getattr(model, "model", None) or getattr(model, "transformer", None)
    if base is None:
        raise RuntimeError("Could not find the decoder backbone on the loaded model.")
    for attr in ("layers", "h"):
        layers = getattr(base, attr, None)
        if layers is not None:
            return base, attr, layers
    raise RuntimeError("Could not find decoder layers on the loaded model.")


def tensor_from_output(output: Any) -> torch.Tensor:
    if isinstance(output, tuple):
        output = output[0]
    if not torch.is_tensor(output):
        raise TypeError(f"Expected a tensor output, got {type(output)!r}")
    return output


def capture_tensor(tensor: torch.Tensor) -> torch.Tensor:
    return tensor.detach().to(torch.bfloat16).cpu().contiguous()


def tensor_stats(tensor: torch.Tensor) -> dict[str, Any]:
    values = tensor.float().reshape(-1)
    return {
        "shape": list(tensor.shape),
        "dtype": str(tensor.dtype).replace("torch.", ""),
        "numel": int(tensor.numel()),
        "mean": float(values.mean().item()),
        "std": float(values.std(unbiased=False).item()) if values.numel() > 1 else 0.0,
        "min": float(values.min().item()),
        "max": float(values.max().item()),
        "absmax": float(values.abs().max().item()),
    }


def unravel_index(flat_index: int, shape: list[int]) -> tuple[int, ...]:
    coords = []
    remaining = int(flat_index)
    for dim in reversed(shape):
        coords.append(remaining % int(dim))
        remaining //= int(dim)
    return tuple(reversed(coords))


def bf16_words(tensor: torch.Tensor) -> list[int]:
    bf16 = tensor.to(torch.bfloat16).contiguous()
    f32 = bf16.float().contiguous()
    words = ((f32.view(torch.int32) >> 16) & 0xFFFF).reshape(-1).to(torch.int32)
    return [int(value) for value in words.tolist()]


def write_bf16_text(path: Path, captures: dict[str, dict[str, torch.Tensor]], metadata: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write("# Qwen3-1.7B_Transformer_block_1 GPU BF16 reference values\n")
        handle.write("# This is GPU-only reference data; crossbar_used=false.\n")
        handle.write("# bf16_hex is the exact raw BF16 16-bit word stored for comparison.\n")
        handle.write("# bf16_value_as_float32 is the exact BF16 value widened to float32 for readability.\n")
        handle.write(f"# metadata_json={json.dumps(metadata, sort_keys=True)}\n")
        handle.write("phase\ttensor\tflat_index\tindex\tbf16_hex\tbf16_value_as_float32\n")

        total_values = 0
        for phase in ("prefill", "decode"):
            for name, tensor in captures[phase].items():
                shape = list(tensor.shape)
                words = bf16_words(tensor)
                values = tensor.float().reshape(-1).tolist()
                handle.write(f"# phase={phase} tensor={name} dtype=BF16 shape={shape} numel={tensor.numel()}\n")
                for flat_index, (word, value) in enumerate(zip(words, values)):
                    coords = unravel_index(flat_index, shape)
                    handle.write(f"{phase}\t{name}\t{flat_index}\t{coords}\t0x{word:04x}\t{value:.9g}\n")
                total_values += int(tensor.numel())
        handle.write(f"# total_values={total_values}\n")


def write_summary_text(path: Path, metadata: dict[str, Any], captures: dict[str, dict[str, torch.Tensor]]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write("Qwen3-1.7B_Transformer_block_1 GPU Reference Summary\n")
        handle.write("====================================================\n\n")
        for key in [
            "run_name",
            "model",
            "phase_list",
            "reference_backend",
            "crossbar_used",
            "original_layer_count",
            "block_index_zero_based",
            "block_index_one_based",
            "dtype",
            "device",
            "gpu_name",
            "prefill_input_shape",
            "decode_input_shape",
            "elapsed_prefill_s",
            "elapsed_decode_s",
        ]:
            handle.write(f"{key}: {metadata.get(key)}\n")
        handle.write("\nTensor summaries\n")
        handle.write("phase\ttensor\tshape\tdtype\tnumel\tmean\tstd\tmin\tmax\tabsmax\n")
        for phase in ("prefill", "decode"):
            for name, tensor in captures[phase].items():
                stats = tensor_stats(tensor)
                handle.write(
                    f"{phase}\t{name}\t{stats['shape']}\t{stats['dtype']}\t{stats['numel']}\t"
                    f"{stats['mean']:.9g}\t{stats['std']:.9g}\t{stats['min']:.9g}\t"
                    f"{stats['max']:.9g}\t{stats['absmax']:.9g}\n"
                )


def add_hooks(layer: torch.nn.Module, captures: dict[str, dict[str, torch.Tensor]], phase_ref: dict[str, str]) -> list[Any]:
    hooks = []

    def save(name: str, tensor: torch.Tensor) -> None:
        captures[phase_ref["phase"]][name] = capture_tensor(tensor)

    def pre_hook(name: str):
        def hook(_module: torch.nn.Module, inputs: tuple[Any, ...]) -> None:
            save(name, tensor_from_output(inputs[0]))

        return hook

    def out_hook(name: str):
        def hook(_module: torch.nn.Module, _inputs: tuple[Any, ...], output: Any) -> None:
            save(name, tensor_from_output(output))

        return hook

    hooks.append(layer.register_forward_pre_hook(pre_hook("layer_00_block_input")))
    hooks.append(layer.register_forward_hook(out_hook("layer_00_block_output")))

    input_norm = getattr(layer, "input_layernorm", None)
    post_norm = getattr(layer, "post_attention_layernorm", None)
    attn = getattr(layer, "self_attn", None) or getattr(layer, "attention", None)
    mlp = getattr(layer, "mlp", None)
    if input_norm is None or post_norm is None or attn is None or mlp is None:
        raise RuntimeError("Qwen decoder layer is missing expected block modules.")

    hooks.append(input_norm.register_forward_hook(out_hook("layer_00_input_layernorm_output")))
    hooks.append(post_norm.register_forward_hook(out_hook("layer_00_post_attention_layernorm_output")))

    for attr, name in [
        ("q_proj", "layer_00_q_proj_output"),
        ("k_proj", "layer_00_k_proj_output"),
        ("v_proj", "layer_00_v_proj_output"),
        ("o_proj", "layer_00_o_proj_output"),
    ]:
        module = getattr(attn, attr, None)
        if module is None:
            raise RuntimeError(f"Attention block is missing {attr}.")
        hooks.append(module.register_forward_hook(out_hook(name)))
    hooks.append(attn.o_proj.register_forward_pre_hook(pre_hook("layer_00_attn_context_before_o_proj")))

    for attr, name in [
        ("gate_proj", "layer_00_mlp_gate_proj_output"),
        ("up_proj", "layer_00_mlp_up_proj_output"),
        ("down_proj", "layer_00_mlp_down_proj_output"),
    ]:
        module = getattr(mlp, attr, None)
        if module is None:
            raise RuntimeError(f"MLP block is missing {attr}.")
        hooks.append(module.register_forward_hook(out_hook(name)))
    hooks.append(mlp.down_proj.register_forward_pre_hook(pre_hook("layer_00_mlp_gated_activation_before_down_proj")))

    return hooks


def load_model(model_path: str, dtype: torch.dtype) -> torch.nn.Module:
    kwargs = {
        "low_cpu_mem_usage": True,
        "trust_remote_code": True,
    }
    try:
        return AutoModelForCausalLM.from_pretrained(model_path, dtype=dtype, **kwargs)
    except TypeError:
        return AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=dtype, **kwargs)


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested, but torch.cuda.is_available() is False.")

    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    dtype = select_dtype(args.dtype, device)

    model_local_dir = Path(args.local_root).expanduser() / args.model.replace("/", "--")
    print(f"run_name={args.run_name}", flush=True)
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
    model = load_model(snapshot_path, dtype=dtype)
    model.eval()

    base, layers_attr, layers = base_model_and_layers(model)
    original_layer_count = len(layers)
    if args.block_index < 0 or args.block_index >= original_layer_count:
        raise ValueError(f"--block-index must be in [0, {original_layer_count - 1}], got {args.block_index}")

    selected_layer = layers[args.block_index]
    setattr(base, layers_attr, torch.nn.ModuleList([selected_layer]))
    model.to(device)

    encoded = tokenizer(args.prompt, return_tensors="pt")
    if encoded["input_ids"].shape[1] > args.max_input_tokens:
        encoded = {key: value[:, : args.max_input_tokens] for key, value in encoded.items()}
    encoded = {key: value.to(device) for key, value in encoded.items()}

    decode_ids = tokenizer(args.decode_text, add_special_tokens=False, return_tensors="pt")["input_ids"]
    if decode_ids.numel() < 1:
        raise RuntimeError("--decode-text did not produce any tokens.")
    decode_ids = decode_ids[:, :1].to(device)

    captures: dict[str, dict[str, torch.Tensor]] = {"prefill": {}, "decode": {}}
    phase_ref = {"phase": "prefill"}
    hooks = add_hooks(selected_layer, captures, phase_ref)

    try:
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
            torch.cuda.synchronize(device)

        t0 = time.perf_counter()
        with torch.inference_mode():
            prefill_outputs = base(
                input_ids=encoded["input_ids"],
                attention_mask=encoded.get("attention_mask"),
                use_cache=True,
                output_attentions=False,
                output_hidden_states=False,
            )
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        elapsed_prefill_s = time.perf_counter() - t0

        phase_ref["phase"] = "decode"
        decode_attention_mask = torch.cat(
            [
                encoded.get("attention_mask", torch.ones_like(encoded["input_ids"])),
                torch.ones((encoded["input_ids"].shape[0], 1), dtype=encoded["input_ids"].dtype, device=device),
            ],
            dim=1,
        )
        t1 = time.perf_counter()
        with torch.inference_mode():
            base(
                input_ids=decode_ids,
                attention_mask=decode_attention_mask,
                past_key_values=prefill_outputs.past_key_values,
                use_cache=True,
                output_attentions=False,
                output_hidden_states=False,
            )
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        elapsed_decode_s = time.perf_counter() - t1
    finally:
        for handle in hooks:
            handle.remove()

    expected = {
        "layer_00_block_input",
        "layer_00_input_layernorm_output",
        "layer_00_q_proj_output",
        "layer_00_k_proj_output",
        "layer_00_v_proj_output",
        "layer_00_attn_context_before_o_proj",
        "layer_00_o_proj_output",
        "layer_00_post_attention_layernorm_output",
        "layer_00_mlp_gate_proj_output",
        "layer_00_mlp_up_proj_output",
        "layer_00_mlp_gated_activation_before_down_proj",
        "layer_00_mlp_down_proj_output",
        "layer_00_block_output",
    }
    for phase in ("prefill", "decode"):
        missing = sorted(expected.difference(captures[phase]))
        if missing:
            raise RuntimeError(f"Missing {phase} captures: {missing}")

    metadata = {
        "run_name": args.run_name,
        "model": args.model,
        "snapshot_path": str(snapshot_path),
        "phase_list": ["prefill", "decode"],
        "reference_backend": "gpu",
        "crossbar_used": False,
        "original_layer_count": original_layer_count,
        "block_index_zero_based": args.block_index,
        "block_index_one_based": args.block_index + 1,
        "dtype": str(dtype).replace("torch.", ""),
        "device": str(device),
        "cuda_available": torch.cuda.is_available(),
        "cuda_version": torch.version.cuda,
        "gpu_name": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "prefill_prompt": args.prompt,
        "prefill_prompt_sha256": hashlib.sha256(args.prompt.encode("utf-8")).hexdigest(),
        "prefill_input_shape": list(encoded["input_ids"].shape),
        "prefill_input_ids": encoded["input_ids"].detach().cpu().tolist(),
        "decode_text": args.decode_text,
        "decode_input_shape": list(decode_ids.shape),
        "decode_input_ids": decode_ids.detach().cpu().tolist(),
        "elapsed_prefill_s": elapsed_prefill_s,
        "elapsed_decode_s": elapsed_decode_s,
        "peak_cuda_memory_bytes": torch.cuda.max_memory_allocated(device) if device.type == "cuda" else 0,
        "python": platform.python_version(),
        "platform": platform.platform(),
        "hostname": platform.node(),
    }

    values_txt = out_dir / f"{args.run_name}_bf16_values.txt"
    summary_txt = out_dir / f"{args.run_name}_summary.txt"
    metadata_txt = out_dir / f"{args.run_name}_metadata.txt"
    metadata_json = out_dir / "metadata.json"

    write_bf16_text(values_txt, captures, metadata)
    write_summary_text(summary_txt, metadata, captures)
    metadata_txt.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    metadata_json.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")

    print(f"wrote={out_dir}", flush=True)
    print(f"values_txt={values_txt}", flush=True)
    print(f"summary_txt={summary_txt}", flush=True)
    print(f"elapsed_prefill_s={elapsed_prefill_s:.4f}", flush=True)
    print(f"elapsed_decode_s={elapsed_decode_s:.4f}", flush=True)


if __name__ == "__main__":
    main()
