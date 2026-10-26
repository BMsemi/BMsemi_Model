"""Continue Qwen3-1.7B prefill after FPGA transformer block 0.

The runner consumes ``fpga_block1_prefill.npz`` from the verified X2 handoff,
executes transformer blocks 1..27 with the FP4LLM software-crossbar runtime,
then applies the framework final norm and language-model head.  It deliberately
does not rerun embedding or transformer block 0 and does not perform decode.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import platform
import shlex
import shutil
import socket
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from transformers import AutoTokenizer

from src import qwen_crossbar_runtime as runtime


EXPECTED_MODEL_REVISION = "70d244cc86ccca08cf5af4e1e306ecf908b1ad5e"
EXPECTED_PROMPT = "Hello"
EXPECTED_TOKEN_IDS = [9707]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Continue verified FPGA block-0 prefill through Qwen blocks 1..27 on software crossbars."
    )
    parser.add_argument("--input-package", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--model", default="Qwen/Qwen3-1.7B")
    parser.add_argument("--local-root", type=Path, default=Path.home() / "hf_models")
    parser.add_argument("--prefill-mapping", type=Path, default=Path("mappings/Qwen3_1_7B_prefill.m"))
    parser.add_argument("--decode-mapping", type=Path, default=Path("mappings/Qwen3_1_7B_decode.m"))
    parser.add_argument("--start-block-index", type=int, default=1)
    parser.add_argument("--end-block-index", type=int, default=27)
    parser.add_argument("--backend", choices=("wrapper", "ideal_int"), default="wrapper")
    parser.add_argument("--weight-bits", type=int, default=7)
    parser.add_argument("--act-bits", type=int, default=7)
    parser.add_argument("--block-size", type=int, default=16)
    parser.add_argument("--dtype", choices=("bfloat16", "float32"), default="bfloat16")
    parser.add_argument("--threads", type=int, default=0, help="0 uses the Slurm CPU allocation/affinity.")
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--disable-mapped-attention-gemms", action="store_true")
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f"Cannot serialize {type(value)!r}")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields: list[str] = []
    for row in rows:
        for field in row:
            if field not in fields:
                fields.append(field)
    with path.open("w", encoding="utf-8", newline="") as handle:
        if not fields:
            return
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def configure_threads(requested: int) -> int:
    affinity = len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else (os.cpu_count() or 1)
    slurm = int(os.environ.get("SLURM_CPUS_PER_TASK", "0") or 0)
    threads = int(requested) if requested > 0 else (slurm or affinity)
    threads = max(1, min(threads, affinity))
    torch.set_num_threads(threads)
    try:
        torch.set_num_interop_threads(max(1, min(4, threads)))
    except RuntimeError:
        pass
    return threads


def validate_source(package: Path) -> tuple[Path, dict[str, Any]]:
    hidden_path = package / "fpga_block1_prefill.npz"
    source_path = package / "hello.source.json"
    if not hidden_path.is_file() or not source_path.is_file():
        raise FileNotFoundError("Input package must contain fpga_block1_prefill.npz and hello.source.json")
    source = json.loads(source_path.read_text(encoding="utf-8"))
    checkpoint = str(source.get("checkpoint", ""))
    if EXPECTED_MODEL_REVISION not in checkpoint:
        raise RuntimeError(f"Unexpected checkpoint provenance: {checkpoint!r}")
    if source.get("prompt") != EXPECTED_PROMPT or source.get("token_ids") != EXPECTED_TOKEN_IDS:
        raise RuntimeError("The handoff is not the verified single-token 'Hello' input.")
    return hidden_path, source


def capture_output(value: Any) -> torch.Tensor:
    tensor = runtime.tensor_from_output(value)
    return tensor.detach().to(dtype=torch.float32, device="cpu").contiguous()


def make_readme(metadata: dict[str, Any]) -> str:
    command = metadata["command"]
    timing = metadata["timing"]
    next_token = metadata["next_token"]
    peak_resident = metadata["tile_lifecycle"].get("peak_resident_blocks", "not applicable")
    return f"""# Qwen3-1.7B mixed FPGA + software-crossbar prefill

## Completed scope

- Prompt: `Hello` (token ID `9707`, position 0).
- Transformer block 0: consumed from the verified physical X2 FPGA artifact; it was **not rerun**.
- Transformer blocks {metadata['start_block_index']} through {metadata['end_block_index']}: executed in order by the Python FP4LLM software-crossbar `CrossbarLinear` wrapper.
- Final RMS normalization and language-model head: executed by PyTorch after block 27.
- Resulting next token: ID `{next_token['id']}`, decoded as `{next_token['text']!r}`.

This is a mixed-arithmetic prefill experiment. The FPGA block used INT8 linear arithmetic, while the continuation used W{metadata['weight_bits']}/A{metadata['act_bits']} blockwise quantization with BF16 boundaries. It is not an all-FPGA run. Decode was not executed because a correct decode path must merge the FPGA block-0 K/V cache with caches from blocks 1–27.

## Parallel execution

Slurm job `{metadata['slurm']['job_id']}` on `{metadata['hostname']}` reserved {metadata['threads']['slurm_cpus_per_task']} CPU cores and {metadata['slurm']['memory']} of RAM. The blocks remain sequential because each block consumes the preceding block's hidden state; thread parallelism is used inside PyTorch/NumPy crossbar kernels. Peak resident software-crossbar blocks: {peak_resident}.

## Timing

- Blocks 1–27 plus final norm: {timing['continuation_seconds']:.6f} s
- Language-model head: {timing['lm_head_seconds']:.6f} s
- Total measured execution: {timing['total_seconds']:.6f} s

Per-block timings are in `block_timings.csv`.

## Reproduction steps

1. Check out project commit `7f8bcee0cf9cf4e6e24687359b614c1abbf3b829`.
2. Use Qwen/Qwen3-1.7B revision `{EXPECTED_MODEL_REVISION}`.
3. Reserve the compute node, for example:

   ```bash
   srun --pty --nodelist=bose01 --cpus-per-task=32 --mem=128G --time=12:00:00 bash
   ```

   X11 is not required; this run is headless.

4. Activate the environment and constrain numerical libraries to the allocation:

   ```bash
   source ~/venvs/fp4llm-qwen3/bin/activate
   export OMP_NUM_THREADS=32 MKL_NUM_THREADS=32 OPENBLAS_NUM_THREADS=32 NUMEXPR_NUM_THREADS=32
   ```

5. Run the continuation from the project root:

   ```bash
   {command}
   ```

6. Verify every artifact with `sha256sum -c SHA256SUMS.txt`.

## Artifacts

- `continuation_outputs.npz`: FPGA block-0 output, each software block output, final normalized hidden state, and logits.
- `metadata.json`: provenance, configuration, versions, shapes, timings, and next-token result.
- `block_timings.csv`: one timing row per software-executed transformer block.
- `linear_crossbar_modules.csv`: mapping and tile metadata for each crossbar-backed projection.
- `mapped_attention_gemms.csv`: executed QK^T and attention-value crossbar operations.
- `run.log`: captured console output from the execution.
- `run_qwen_fpga_continuation.py`: exact runner used.
- `fpga_block1_prefill.npz`, `hello.source.json`, `accuracy.json`, `fpga_summary.json`: copied source provenance.
- `SHA256SUMS.txt`: checksums for the folder contents.
"""


def main() -> None:
    args = parse_args()
    started = time.perf_counter()
    torch.manual_seed(args.seed)
    threads = configure_threads(args.threads)

    package = args.input_package.expanduser().resolve()
    out_dir = args.out_dir.expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    hidden_path, source = validate_source(package)

    with np.load(hidden_path, allow_pickle=False) as payload:
        if "output" not in payload:
            raise KeyError(f"{hidden_path} has no 'output' array")
        fpga_output = np.asarray(payload["output"], dtype=np.float32)
    if fpga_output.shape != (1, 2048):
        raise RuntimeError(f"Expected FPGA output shape (1, 2048), got {fpga_output.shape}")

    model_dir = runtime.resolve_local_model_dir(args.local_root, args.model)
    if not model_dir.is_dir():
        raise FileNotFoundError(f"Local model directory not found: {model_dir}")
    dtype = runtime.dtype_from_name(args.dtype)
    tokenizer = AutoTokenizer.from_pretrained(str(model_dir), trust_remote_code=True, local_files_only=True)
    model = runtime.load_model(str(model_dir), dtype=dtype).eval()
    base, layers_attr, layers = runtime.base_model_and_layers(model)
    original_layers = list(layers)
    original_count = len(original_layers)
    if original_count != 28:
        raise RuntimeError(f"Expected 28 transformer blocks, found {original_count}")
    if args.start_block_index != 1 or args.end_block_index != 27:
        if not (1 <= args.start_block_index <= args.end_block_index < original_count):
            raise ValueError("Continuation block range must be within 1..27")

    phase_ref = {"phase": "prefill"}
    streaming = args.backend == "wrapper"
    residency = {
        "resident_blocks": 0,
        "peak_resident_blocks": 0,
        "block_program_events": 0,
        "block_release_events": 0,
    }
    blocks: list[dict[str, Any]] = []
    linear_rows: list[dict[str, Any]] = []
    timing_rows: list[dict[str, Any]] = []
    block_outputs: dict[str, torch.Tensor] = {}
    hooks: list[Any] = []

    for index in range(args.start_block_index, args.end_block_index + 1):
        layer = original_layers[index]
        mapping_ctx = runtime.load_mapping_context(args.prefill_mapping, args.decode_mapping, index)
        checks = runtime.validate_sequence_mapping_capacity(mapping_ctx, prefill_tokens=1, decode_tokens=1)
        rows = runtime.replace_block_linears(
            layer,
            mapping_ctx=mapping_ctx,
            backend=args.backend,
            weight_bits=args.weight_bits,
            act_bits=args.act_bits,
            block_size=args.block_size,
            float_format="bf16" if args.dtype == "bfloat16" else "fp32",
            defer_programming=streaming,
        )
        for row in rows:
            row["block_index"] = index
            linear_rows.append(row)
        attention_runner = None
        if not args.disable_mapped_attention_gemms:
            attention_runner = runtime.install_mapped_attention_gemms(
                layer,
                mapping_ctx=mapping_ctx,
                phase_ref=phase_ref,
                backend=args.backend,
                weight_bits=args.weight_bits,
                act_bits=args.act_bits,
                block_size=args.block_size,
                float_format="bf16" if args.dtype == "bfloat16" else "fp32",
                saturation_config={"enabled": False},
                hardware_effect_config={"enabled": False},
            )

        timing_state: dict[str, float] = {}

        def start_timing(_module: nn.Module, _inputs: tuple[Any, ...], *, state=timing_state) -> None:
            state["start"] = time.perf_counter()

        def finish_timing(
            _module: nn.Module,
            _inputs: tuple[Any, ...],
            output: Any,
            *,
            state=timing_state,
            block_index=index,
        ) -> None:
            elapsed = time.perf_counter() - state["start"]
            timing_rows.append({"block_index": block_index, "block_one_based": block_index + 1, "seconds": elapsed})
            block_outputs[f"software_block_{block_index:02d}_output"] = capture_output(output)
            print(f"software_block_complete={block_index} seconds={elapsed:.6f}", flush=True)

        hooks.append(layer.register_forward_pre_hook(start_timing))
        hooks.append(layer.register_forward_hook(finish_timing))
        lifecycle_hooks = (
            runtime.install_block_tile_lifecycle(
                layer,
                block_index=index,
                phase_ref=phase_ref,
                residency=residency,
            )
            if streaming
            else []
        )
        hooks.extend(lifecycle_hooks)
        blocks.append(
            {
                "index": index,
                "layer": layer,
                "mapping_ctx": mapping_ctx,
                "mapping_validation": checks,
                "attention_runner": attention_runner,
            }
        )
        print(f"software_block_ready={index}", flush=True)

    selected_layers = [block["layer"] for block in blocks]
    setattr(base, layers_attr, nn.ModuleList(selected_layers))
    original_num_hidden_layers = int(base.config.num_hidden_layers)
    original_layer_types = list(base.config.layer_types)
    base.config.num_hidden_layers = len(selected_layers)
    base.config.layer_types = original_layer_types[args.start_block_index : args.end_block_index + 1]

    hidden = torch.from_numpy(fpga_output).unsqueeze(0).to(dtype=dtype)
    attention_mask = torch.ones((1, 1), dtype=torch.long)
    position_ids = torch.zeros((1, 1), dtype=torch.long)
    try:
        continuation_start = time.perf_counter()
        with torch.inference_mode():
            output = base(
                inputs_embeds=hidden,
                attention_mask=attention_mask,
                position_ids=position_ids,
                use_cache=False,
            )
        continuation_seconds = time.perf_counter() - continuation_start
        final_hidden = output.last_hidden_state
        lm_start = time.perf_counter()
        with torch.inference_mode():
            logits = model.lm_head(final_hidden.to(dtype=model.lm_head.weight.dtype))
        lm_seconds = time.perf_counter() - lm_start
    finally:
        for hook in hooks:
            hook.remove()
        setattr(base, layers_attr, nn.ModuleList(original_layers))
        base.config.num_hidden_layers = original_num_hidden_layers
        base.config.layer_types = original_layer_types

    next_id = int(torch.argmax(logits[:, -1, :], dim=-1).item())
    next_text = tokenizer.decode([next_id], skip_special_tokens=False)
    attention_rows = [
        dict(record, block_index=block["index"])
        for block in blocks
        if block["attention_runner"] is not None
        for record in block["attention_runner"].records
    ]
    mapping_validation = [
        dict(check, block_index=block["index"])
        for block in blocks
        for check in block["mapping_validation"]
    ]

    arrays: dict[str, np.ndarray] = {
        "fpga_block_00_output": fpga_output,
        "final_norm_output": final_hidden.detach().float().cpu().numpy(),
        "logits": logits.detach().float().cpu().numpy(),
    }
    arrays.update({name: tensor.numpy() for name, tensor in block_outputs.items()})
    np.savez_compressed(out_dir / "continuation_outputs.npz", **arrays)
    write_csv(out_dir / "block_timings.csv", sorted(timing_rows, key=lambda row: int(row["block_index"])))
    write_csv(out_dir / "linear_crossbar_modules.csv", linear_rows)
    write_csv(out_dir / "mapped_attention_gemms.csv", attention_rows)
    write_csv(out_dir / "mapping_validation.csv", mapping_validation)

    for name in (
        "fpga_block1_prefill.npz",
        "hello.source.json",
        "accuracy.json",
        "fpga_summary.json",
        "board_hello_run.txt",
    ):
        source_file = package / name
        if source_file.is_file():
            shutil.copy2(source_file, out_dir / name)
    shutil.copy2(Path(__file__).resolve(), out_dir / "run_qwen_fpga_continuation.py")

    total_seconds = time.perf_counter() - started
    affinity = len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else (os.cpu_count() or 1)
    metadata = {
        "status": "complete",
        "scope": "FPGA block 0 plus software-crossbar blocks 1..27 prefill; final norm and LM head",
        "commit": "7f8bcee0cf9cf4e6e24687359b614c1abbf3b829",
        "model": args.model,
        "model_revision": EXPECTED_MODEL_REVISION,
        "model_dir": str(model_dir),
        "prompt": EXPECTED_PROMPT,
        "token_ids": EXPECTED_TOKEN_IDS,
        "fpga_source": source,
        "fpga_input_file_sha256": sha256(hidden_path),
        "start_block_index": args.start_block_index,
        "end_block_index": args.end_block_index,
        "executed_block_count": len(blocks),
        "block_zero_recomputed": False,
        "prefill_only": True,
        "decode_executed": False,
        "crossbar_backend": args.backend,
        "crossbar_implementation": (
            "reram_crossbar.py:CrossbarArray.compute" if args.backend == "wrapper" else "numpy:int32_matmul"
        ),
        "mapped_attention_gemms_enabled": not args.disable_mapped_attention_gemms,
        "weight_bits": args.weight_bits,
        "act_bits": args.act_bits,
        "block_size": args.block_size,
        "dtype": args.dtype,
        "arithmetic_note": "mixed FPGA INT8 block 0 and software W7/A7 continuation with BF16 boundaries",
        "input_shape": list(hidden.shape),
        "final_hidden_shape": list(final_hidden.shape),
        "logits_shape": list(logits.shape),
        "next_token": {"id": next_id, "text": next_text},
        "timing": {
            "continuation_seconds": continuation_seconds,
            "lm_head_seconds": lm_seconds,
            "total_seconds": total_seconds,
            "per_block": sorted(timing_rows, key=lambda row: int(row["block_index"])),
        },
        "tile_lifecycle": residency if streaming else {"mode": "eager"},
        "threads": {
            "torch_intraop": torch.get_num_threads(),
            "torch_interop": torch.get_num_interop_threads(),
            "affinity_cpus": affinity,
            "slurm_cpus_per_task": int(os.environ.get("SLURM_CPUS_PER_TASK", "0") or 0),
            "configured": threads,
            "OMP_NUM_THREADS": os.environ.get("OMP_NUM_THREADS"),
            "MKL_NUM_THREADS": os.environ.get("MKL_NUM_THREADS"),
            "OPENBLAS_NUM_THREADS": os.environ.get("OPENBLAS_NUM_THREADS"),
        },
        "slurm": {
            "job_id": os.environ.get("SLURM_JOB_ID", ""),
            "job_node_list": os.environ.get("SLURM_JOB_NODELIST", ""),
            "cpus_per_task": os.environ.get("SLURM_CPUS_PER_TASK", ""),
            "memory": os.environ.get("SLURM_MEM_PER_NODE", "128G"),
        },
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "numpy": np.__version__,
        "command": shlex.join(["python", str(Path(__file__).resolve()), *sys.argv[1:]]),
        "outputs": sorted(arrays),
    }
    (out_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2, default=json_default) + "\n", encoding="utf-8"
    )
    (out_dir / "README.md").write_text(make_readme(metadata), encoding="utf-8")

    checksum_lines = []
    for path in sorted(out_dir.iterdir(), key=lambda item: item.name):
        if path.is_file() and path.name != "SHA256SUMS.txt":
            checksum_lines.append(f"{sha256(path)}  {path.name}")
    (out_dir / "SHA256SUMS.txt").write_text("\n".join(checksum_lines) + "\n", encoding="utf-8")
    print(f"status=complete out_dir={out_dir}", flush=True)
    print(f"next_token_id={next_id} next_token_text={next_text!r}", flush=True)
    print(f"total_seconds={total_seconds:.6f}", flush=True)


if __name__ == "__main__":
    main()
