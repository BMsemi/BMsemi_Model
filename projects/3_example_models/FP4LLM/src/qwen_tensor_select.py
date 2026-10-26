from __future__ import annotations

from collections.abc import Iterable


DEFAULT_QWEN_TENSOR_ROLE = "mlp_output"

QWEN_TENSOR_ROLE_SUFFIXES: dict[str, tuple[str, ...]] = {
    "mlp_output": ("mlp.down_proj.weight",),
    "mlp_gate": ("mlp.gate_proj.weight",),
    "mlp_up": ("mlp.up_proj.weight",),
    "attention_q": ("self_attn.q_proj.weight",),
    "q_proj": ("self_attn.q_proj.weight",),
    "legacy_projection": (
        "self_attn.q_proj.weight",
        "mlp.down_proj.weight",
        "mlp.gate_proj.weight",
    ),
}

QWEN_TENSOR_ROLE_DESCRIPTIONS: dict[str, str] = {
    "mlp_output": "MLP output projection only (down_proj).",
    "mlp_gate": "MLP gate projection only (gate_proj).",
    "mlp_up": "MLP up projection only (up_proj).",
    "attention_q": "Attention query projection only (q_proj).",
    "q_proj": "Attention query projection only (q_proj).",
    "legacy_projection": "Legacy mixed projection search used by earlier smoke runs.",
}


def qwen_tensor_role_choices() -> tuple[str, ...]:
    return tuple(QWEN_TENSOR_ROLE_SUFFIXES)


def qwen_tensor_role_suffixes(tensor_role: str) -> tuple[str, ...]:
    role = str(tensor_role).strip().lower()
    try:
        return QWEN_TENSOR_ROLE_SUFFIXES[role]
    except KeyError as exc:
        choices = ", ".join(qwen_tensor_role_choices())
        raise ValueError(f"Unknown Qwen tensor role {tensor_role!r}. Expected one of: {choices}.") from exc


def qwen_tensor_role_description(tensor_role: str) -> str:
    role = str(tensor_role).strip().lower()
    qwen_tensor_role_suffixes(role)
    return QWEN_TENSOR_ROLE_DESCRIPTIONS[role]


def candidate_qwen_tensor_names(
    tensor_names: Iterable[str],
    tensor_role: str = DEFAULT_QWEN_TENSOR_ROLE,
) -> list[str]:
    suffixes = qwen_tensor_role_suffixes(tensor_role)
    return sorted({name for name in tensor_names if name.endswith(suffixes)})


def select_qwen_tensor_name(
    tensor_names: Iterable[str],
    tensor_role: str = DEFAULT_QWEN_TENSOR_ROLE,
) -> str:
    role = str(tensor_role).strip().lower()
    candidates = candidate_qwen_tensor_names(tensor_names, role)
    if not candidates:
        suffix_text = ", ".join(qwen_tensor_role_suffixes(role))
        raise RuntimeError(f"No Qwen tensor found for role {role!r}. Expected suffix: {suffix_text}.")
    return candidates[0]


def qwen_tensor_role_for_name(tensor_name: str) -> str:
    for role, suffixes in QWEN_TENSOR_ROLE_SUFFIXES.items():
        if role == "legacy_projection":
            continue
        if tensor_name.endswith(suffixes):
            return role
    return "unknown"


def qwen_tensor_selection_metadata(
    tensor_name: str,
    tensor_role: str = DEFAULT_QWEN_TENSOR_ROLE,
) -> dict[str, str]:
    role = str(tensor_role).strip().lower()
    suffixes = qwen_tensor_role_suffixes(role)
    matched_suffix = next((suffix for suffix in suffixes if tensor_name.endswith(suffix)), "")
    return {
        "tensor_role": role,
        "tensor_role_description": qwen_tensor_role_description(role),
        "tensor_role_inferred_from_name": qwen_tensor_role_for_name(tensor_name),
        "tensor_suffix": matched_suffix,
    }
