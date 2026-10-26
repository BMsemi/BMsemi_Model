import json
import math
import re
import struct
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


CONFIG_PATH = Path(r"C:\Users\Avadh\Downloads\config.json")
MODEL_PATH = Path(r"C:\Users\Avadh\Downloads\model.safetensors")
TEXT_OUT = Path(r"C:\Users\Avadh\Downloads\qwen_model_full_info.txt")
JSON_OUT = Path(r"C:\Users\Avadh\Downloads\qwen_model_full_info.json")


DTYPE_BYTES = {
    "BF16": 2,
    "F16": 2,
    "F32": 4,
    "F64": 8,
    "I8": 1,
    "I16": 2,
    "I32": 4,
    "I64": 8,
    "U8": 1,
    "U16": 2,
    "U32": 4,
    "U64": 8,
    "BOOL": 1,
}


PE_SUPPORTED_SUFFIXES = {
    "self_attn.q_proj.weight": "linear projection: hidden_size -> hidden_size",
    "self_attn.k_proj.weight": "linear projection: hidden_size -> kv_heads * head_dim",
    "self_attn.v_proj.weight": "linear projection: hidden_size -> kv_heads * head_dim",
    "self_attn.o_proj.weight": "linear projection: hidden_size -> hidden_size",
    "mlp.gate_proj.weight": "linear projection: hidden_size -> intermediate_size",
    "mlp.up_proj.weight": "linear projection: hidden_size -> intermediate_size",
    "mlp.down_proj.weight": "linear projection: intermediate_size -> hidden_size",
}


OUTSIDE_PE_OPS = [
    {
        "op": "embedding lookup",
        "config_or_tensor": "model.embed_tokens.weight",
        "reason": "Token ID address lookup; no multiply-accumulate operation.",
    },
    {
        "op": "RMSNorm",
        "config_or_tensor": "input_layernorm.weight, post_attention_layernorm.weight",
        "reason": "Needs square, reduction, reciprocal square root, and scale.",
    },
    {
        "op": "q_norm / k_norm",
        "config_or_tensor": "self_attn.q_norm.weight, self_attn.k_norm.weight",
        "reason": "RMSNorm-style normalization on attention head vectors.",
    },
    {
        "op": "RoPE",
        "config_or_tensor": "rope_theta, max_position_embeddings",
        "reason": "Sin/cos positional rotation of Q/K values; not a weight-stationary MAC.",
    },
    {
        "op": "causal mask",
        "config_or_tensor": "attention mask/control path",
        "reason": "Comparison/indexing/control logic before softmax.",
    },
    {
        "op": "softmax",
        "config_or_tensor": "attention probability path",
        "reason": "Needs exp, sum reduction, and divide.",
    },
    {
        "op": "SiLU",
        "config_or_tensor": "hidden_act=silu",
        "reason": "Nonlinear sigmoid-based activation.",
    },
    {
        "op": "SwiGLU multiply",
        "config_or_tensor": "silu(gate_proj(x)) * up_proj(x)",
        "reason": "Elementwise multiply after two PE-supported linear projections.",
    },
    {
        "op": "residual add",
        "config_or_tensor": "transformer block residual path",
        "reason": "Vector addition; no learned matrix.",
    },
    {
        "op": "KV-cache indexing/control",
        "config_or_tensor": "use_cache=true",
        "reason": "Memory management for token positions and cache updates.",
    },
    {
        "op": "sampling / argmax / top-k / top-p",
        "config_or_tensor": "generation control path",
        "reason": "Comparison, selection, filtering, and random sampling logic.",
    },
    {
        "op": "native BF16 arithmetic",
        "config_or_tensor": "torch_dtype=bfloat16",
        "reason": "Your PE path uses quantized crossbar MACs, not BF16 floating-point ALUs.",
    },
]


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_safetensors_header(path: Path) -> dict[str, Any]:
    with path.open("rb") as handle:
        header_size = struct.unpack("<Q", handle.read(8))[0]
        return json.loads(handle.read(header_size))


def product(values: list[int]) -> int:
    out = 1
    for value in values:
        out *= int(value)
    return out


def gb(num_bytes: int) -> float:
    return float(num_bytes) / (1024.0**3)


def tensor_role(name: str) -> str:
    if name == "model.embed_tokens.weight":
        return "embedding"
    if name == "model.norm.weight":
        return "final_rmsnorm"
    if name.endswith(".input_layernorm.weight"):
        return "input_rmsnorm"
    if name.endswith(".post_attention_layernorm.weight"):
        return "post_attention_rmsnorm"
    if name.endswith(".self_attn.q_norm.weight"):
        return "q_norm"
    if name.endswith(".self_attn.k_norm.weight"):
        return "k_norm"
    for suffix in PE_SUPPORTED_SUFFIXES:
        if name.endswith(suffix):
            return suffix.replace(".weight", "")
    if name == "lm_head.weight":
        return "lm_head"
    return "other"


def pe_support_for_tensor(name: str) -> tuple[str, str]:
    if name == "lm_head.weight":
        return "partial", "Large linear output projection; PE can compute MACs, but vocab projection is huge."
    for suffix, reason in PE_SUPPORTED_SUFFIXES.items():
        if name.endswith(suffix):
            return "yes", reason
    return "no", "Not a 2-D learned linear projection suitable for direct crossbar MAC mapping."


def make_tensor_entries(header: dict[str, Any]) -> list[dict[str, Any]]:
    entries = []
    for name, info in header.items():
        if name == "__metadata__":
            continue
        shape = [int(x) for x in info.get("shape", [])]
        dtype = str(info.get("dtype", "UNKNOWN"))
        param_count = product(shape) if shape else 1
        dtype_bytes = DTYPE_BYTES.get(dtype)
        storage_bytes = param_count * dtype_bytes if dtype_bytes is not None else None
        pe_support, pe_reason = pe_support_for_tensor(name)
        entries.append(
            {
                "name": name,
                "role": tensor_role(name),
                "shape": shape,
                "dtype": dtype,
                "parameter_count": param_count,
                "storage_bytes_estimated": storage_bytes,
                "data_offsets": info.get("data_offsets"),
                "pe_support": pe_support,
                "pe_reason": pe_reason,
            }
        )
    return entries


def summarize_tensors(entries: list[dict[str, Any]]) -> dict[str, Any]:
    by_dtype: dict[str, dict[str, int]] = defaultdict(lambda: {"tensors": 0, "parameters": 0, "bytes": 0})
    by_role: dict[str, dict[str, int]] = defaultdict(lambda: {"tensors": 0, "parameters": 0, "bytes": 0})
    pe_support = Counter()
    total_parameters = 0
    total_estimated_bytes = 0

    for entry in entries:
        params = int(entry["parameter_count"])
        bytes_est = int(entry["storage_bytes_estimated"] or 0)
        total_parameters += params
        total_estimated_bytes += bytes_est
        by_dtype[entry["dtype"]]["tensors"] += 1
        by_dtype[entry["dtype"]]["parameters"] += params
        by_dtype[entry["dtype"]]["bytes"] += bytes_est
        by_role[entry["role"]]["tensors"] += 1
        by_role[entry["role"]]["parameters"] += params
        by_role[entry["role"]]["bytes"] += bytes_est
        pe_support[entry["pe_support"]] += 1

    return {
        "num_tensors": len(entries),
        "total_parameters": total_parameters,
        "total_estimated_bytes": total_estimated_bytes,
        "total_estimated_gib": gb(total_estimated_bytes),
        "by_dtype": dict(sorted(by_dtype.items())),
        "by_role": dict(sorted(by_role.items())),
        "pe_support_tensor_counts": dict(pe_support),
    }


def group_layers(entries: list[dict[str, Any]]) -> list[dict[str, Any]]:
    layer_re = re.compile(r"^model\.layers\.(\d+)\.(.+)$")
    grouped: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for entry in entries:
        match = layer_re.match(entry["name"])
        if match:
            grouped[int(match.group(1))].append(entry)

    layers = []
    for layer_idx in sorted(grouped):
        tensors = sorted(grouped[layer_idx], key=lambda x: x["name"])
        layers.append(
            {
                "layer_index": layer_idx,
                "num_tensors": len(tensors),
                "parameters": sum(int(t["parameter_count"]) for t in tensors),
                "estimated_bytes": sum(int(t["storage_bytes_estimated"] or 0) for t in tensors),
                "tensors": tensors,
            }
        )
    return layers


def inferred_architecture(config: dict[str, Any]) -> dict[str, Any]:
    hidden_size = int(config["hidden_size"])
    intermediate_size = int(config["intermediate_size"])
    num_heads = int(config["num_attention_heads"])
    num_kv_heads = int(config["num_key_value_heads"])
    head_dim = int(config["head_dim"])
    return {
        "architecture": config.get("architectures", []),
        "model_type": config.get("model_type"),
        "num_hidden_layers": int(config["num_hidden_layers"]),
        "hidden_size": hidden_size,
        "intermediate_size": intermediate_size,
        "num_attention_heads": num_heads,
        "num_key_value_heads": num_kv_heads,
        "head_dim": head_dim,
        "query_projection_dim": num_heads * head_dim,
        "key_value_projection_dim": num_kv_heads * head_dim,
        "gqa_fanout": num_heads // max(num_kv_heads, 1),
        "vocab_size": int(config["vocab_size"]),
        "max_position_embeddings": int(config["max_position_embeddings"]),
        "hidden_act": config.get("hidden_act"),
        "torch_dtype": config.get("torch_dtype"),
        "attention_bias": config.get("attention_bias"),
        "tie_word_embeddings": config.get("tie_word_embeddings"),
        "use_cache": config.get("use_cache"),
        "rms_norm_eps": config.get("rms_norm_eps"),
        "rope_theta": config.get("rope_theta"),
        "rope_scaling": config.get("rope_scaling"),
        "use_sliding_window": config.get("use_sliding_window"),
        "sliding_window": config.get("sliding_window"),
    }


def module_inventory(config: dict[str, Any], tensor_lookup: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    arch = inferred_architecture(config)
    hidden = arch["hidden_size"]
    inter = arch["intermediate_size"]
    q_dim = arch["query_projection_dim"]
    kv_dim = arch["key_value_projection_dim"]
    vocab = arch["vocab_size"]
    layers = arch["num_hidden_layers"]

    modules = [
        {
            "module": "embed_tokens",
            "count": 1,
            "operation": "token_id -> embedding vector",
            "shape_expected": [vocab, hidden],
            "pe_support": "no",
            "reason": "Memory lookup, not MAC.",
        },
        {
            "module": "input_layernorm",
            "count": layers,
            "operation": "RMSNorm(hidden)",
            "shape_expected": [hidden],
            "pe_support": "no",
            "reason": "Reduction, reciprocal sqrt, scale.",
        },
        {
            "module": "q_proj",
            "count": layers,
            "operation": "Linear(hidden -> q_heads * head_dim)",
            "shape_expected": [q_dim, hidden],
            "pe_support": "yes",
            "reason": "Learned 2-D matrix projection.",
        },
        {
            "module": "k_proj",
            "count": layers,
            "operation": "Linear(hidden -> kv_heads * head_dim)",
            "shape_expected": [kv_dim, hidden],
            "pe_support": "yes",
            "reason": "Learned 2-D matrix projection.",
        },
        {
            "module": "v_proj",
            "count": layers,
            "operation": "Linear(hidden -> kv_heads * head_dim)",
            "shape_expected": [kv_dim, hidden],
            "pe_support": "yes",
            "reason": "Learned 2-D matrix projection.",
        },
        {
            "module": "q_norm",
            "count": layers,
            "operation": "RMSNorm(head_dim)",
            "shape_expected": [arch["head_dim"]],
            "pe_support": "no",
            "reason": "Normalization, not MAC.",
        },
        {
            "module": "k_norm",
            "count": layers,
            "operation": "RMSNorm(head_dim)",
            "shape_expected": [arch["head_dim"]],
            "pe_support": "no",
            "reason": "Normalization, not MAC.",
        },
        {
            "module": "RoPE",
            "count": layers,
            "operation": "sin/cos rotate Q and K",
            "shape_expected": [arch["head_dim"]],
            "pe_support": "no",
            "reason": "Elementwise trig rotation.",
        },
        {
            "module": "QK^T",
            "count": layers,
            "operation": "attention score matmul",
            "shape_expected": ["tokens", "tokens", arch["head_dim"]],
            "pe_support": "partial",
            "reason": "MAC is PE-compatible, but K-cache storage/indexing and mask are digital.",
        },
        {
            "module": "causal_mask",
            "count": layers,
            "operation": "mask future attention positions",
            "shape_expected": ["tokens", "tokens"],
            "pe_support": "no",
            "reason": "Control/indexing operation.",
        },
        {
            "module": "softmax",
            "count": layers,
            "operation": "exp/sum/div attention probabilities",
            "shape_expected": ["tokens", "tokens"],
            "pe_support": "no",
            "reason": "Nonlinear reduction and divide.",
        },
        {
            "module": "Attention @ V",
            "count": layers,
            "operation": "attention value matmul",
            "shape_expected": ["tokens", arch["head_dim"]],
            "pe_support": "partial",
            "reason": "MAC is PE-compatible, but V-cache and post-softmax activations are external.",
        },
        {
            "module": "o_proj",
            "count": layers,
            "operation": "Linear(hidden -> hidden)",
            "shape_expected": [hidden, hidden],
            "pe_support": "yes",
            "reason": "Learned 2-D matrix projection.",
        },
        {
            "module": "post_attention_layernorm",
            "count": layers,
            "operation": "RMSNorm(hidden)",
            "shape_expected": [hidden],
            "pe_support": "no",
            "reason": "Reduction, reciprocal sqrt, scale.",
        },
        {
            "module": "gate_proj",
            "count": layers,
            "operation": "Linear(hidden -> intermediate)",
            "shape_expected": [inter, hidden],
            "pe_support": "yes",
            "reason": "Learned 2-D matrix projection.",
        },
        {
            "module": "up_proj",
            "count": layers,
            "operation": "Linear(hidden -> intermediate)",
            "shape_expected": [inter, hidden],
            "pe_support": "yes",
            "reason": "Learned 2-D matrix projection.",
        },
        {
            "module": "SiLU",
            "count": layers,
            "operation": "silu(gate)",
            "shape_expected": [inter],
            "pe_support": "no",
            "reason": "Sigmoid-based nonlinear activation.",
        },
        {
            "module": "SwiGLU multiply",
            "count": layers,
            "operation": "silu(gate) * up",
            "shape_expected": [inter],
            "pe_support": "no",
            "reason": "Elementwise vector multiply.",
        },
        {
            "module": "down_proj",
            "count": layers,
            "operation": "Linear(intermediate -> hidden)",
            "shape_expected": [hidden, inter],
            "pe_support": "yes",
            "reason": "Learned 2-D matrix projection.",
        },
        {
            "module": "residual_add",
            "count": layers * 2,
            "operation": "x + sublayer_output",
            "shape_expected": [hidden],
            "pe_support": "no",
            "reason": "Vector addition, no learned matrix.",
        },
        {
            "module": "final_norm",
            "count": 1,
            "operation": "RMSNorm(hidden)",
            "shape_expected": [hidden],
            "pe_support": "no",
            "reason": "Final normalization before logits.",
        },
        {
            "module": "lm_head",
            "count": 1,
            "operation": "Linear(hidden -> vocab)",
            "shape_expected": [vocab, hidden],
            "pe_support": "partial",
            "reason": "Tied to embeddings in this config; PE can compute MACs if materialized/mapped.",
        },
    ]

    for module in modules:
        suffix = module["module"]
        matched = [
            t
            for name, t in tensor_lookup.items()
            if name.endswith(f".{suffix}.weight") or name == f"model.{suffix}.weight" or name == f"{suffix}.weight"
        ]
        module["matching_tensors_found"] = len(matched)
    return modules


def write_text_report(report: dict[str, Any], path: Path) -> None:
    config = report["config"]
    arch = report["derived_architecture"]
    summary = report["tensor_summary"]

    with path.open("w", encoding="utf-8") as out:
        out.write("# Qwen Model Full Information Report\n\n")
        out.write("NOTE: This is not an ONNX graph export. It is a full static report inferred from config.json and model.safetensors.\n")
        out.write("An exact ONNX file requires loading the Transformers model code and exporting the computation graph.\n\n")

        out.write("## Files\n")
        for key, value in report["files"].items():
            out.write(f"- {key}: {value}\n")
        out.write("\n")

        out.write("## Config\n")
        for key in sorted(config):
            out.write(f"- {key}: {config[key]}\n")
        out.write("\n")

        out.write("## Derived Architecture\n")
        for key, value in arch.items():
            out.write(f"- {key}: {value}\n")
        out.write("\n")

        out.write("## Tensor Summary\n")
        out.write(f"- num_tensors: {summary['num_tensors']}\n")
        out.write(f"- total_parameters: {summary['total_parameters']:,}\n")
        out.write(f"- total_estimated_bytes: {summary['total_estimated_bytes']:,}\n")
        out.write(f"- total_estimated_gib: {summary['total_estimated_gib']:.6f}\n")
        out.write("- pe_support_tensor_counts: " + json.dumps(summary["pe_support_tensor_counts"]) + "\n\n")

        out.write("## Summary By DType\n")
        for dtype, row in summary["by_dtype"].items():
            out.write(
                f"- {dtype}: tensors={row['tensors']}, parameters={row['parameters']:,}, "
                f"bytes={row['bytes']:,}, GiB={gb(row['bytes']):.6f}\n"
            )
        out.write("\n")

        out.write("## Summary By Role\n")
        for role, row in summary["by_role"].items():
            out.write(
                f"- {role}: tensors={row['tensors']}, parameters={row['parameters']:,}, "
                f"bytes={row['bytes']:,}\n"
            )
        out.write("\n")

        out.write("## Inferred Module / Operation Inventory\n")
        out.write(f"{'module':30s} {'count':>6s} {'pe':>8s}  operation\n")
        out.write("-" * 120 + "\n")
        for module in report["module_inventory"]:
            out.write(
                f"{module['module']:30s} {str(module['count']):>6s} "
                f"{module['pe_support']:>8s}  {module['operation']} | {module['reason']}\n"
            )
        out.write("\n")

        out.write("## Outside-PE Operations\n")
        for op in report["outside_pe_operations"]:
            out.write(f"- {op['op']}: {op['reason']} ({op['config_or_tensor']})\n")
        out.write("\n")

        out.write("## Per-Layer Tensor Groups\n")
        for layer in report["layers"]:
            out.write(
                f"\n### Layer {layer['layer_index']}: tensors={layer['num_tensors']}, "
                f"parameters={layer['parameters']:,}, bytes={layer['estimated_bytes']:,}\n"
            )
            for t in layer["tensors"]:
                out.write(
                    f"- {t['name']}: shape={t['shape']}, dtype={t['dtype']}, "
                    f"params={t['parameter_count']:,}, pe={t['pe_support']}\n"
                )

        out.write("\n## Complete Tensor Inventory\n")
        out.write(f"{'tensor_name':80s} {'shape':24s} {'dtype':8s} {'params':>16s} {'pe':>8s} offsets\n")
        out.write("-" * 160 + "\n")
        for t in report["tensors"]:
            out.write(
                f"{t['name']:80s} {str(t['shape']):24s} {t['dtype']:8s} "
                f"{t['parameter_count']:16,d} {t['pe_support']:>8s} {t['data_offsets']}\n"
            )


def main() -> None:
    config = read_json(CONFIG_PATH)
    header = read_safetensors_header(MODEL_PATH)
    tensors = make_tensor_entries(header)
    tensor_lookup = {t["name"]: t for t in tensors}

    report = {
        "files": {
            "config_json": str(CONFIG_PATH),
            "model_safetensors": str(MODEL_PATH),
            "model_safetensors_size_bytes": MODEL_PATH.stat().st_size,
            "safetensors_metadata": header.get("__metadata__", {}),
        },
        "config": config,
        "derived_architecture": inferred_architecture(config),
        "tensor_summary": summarize_tensors(tensors),
        "module_inventory": module_inventory(config, tensor_lookup),
        "outside_pe_operations": OUTSIDE_PE_OPS,
        "layers": group_layers(tensors),
        "tensors": tensors,
    }

    JSON_OUT.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    write_text_report(report, TEXT_OUT)

    print(f"Wrote text report: {TEXT_OUT}")
    print(f"Wrote JSON report: {JSON_OUT}")
    print(f"Tensors: {report['tensor_summary']['num_tensors']}")
    print(f"Parameters: {report['tensor_summary']['total_parameters']:,}")


if __name__ == "__main__":
    main()
