from __future__ import annotations

import json
from pathlib import Path
from urllib.parse import quote
from urllib.request import urlopen

from .mapping import LayerSpec


def read_qwen_config(source: str | Path, *, revision: str = "main") -> dict[str, object]:
    """Read Qwen config from a local path or directly from Hugging Face.

    Local paths are useful for offline runs. A Hugging Face repo id such as
    ``Qwen/Qwen3-1.7B`` is the portable source of truth for generated mappings.
    """

    source_text = str(source)
    config_path = Path(source_text)
    if config_path.exists():
        if config_path.is_dir():
            config_path = config_path / "config.json"
        return json.loads(config_path.read_text(encoding="utf-8"))

    if "/" not in source_text:
        raise FileNotFoundError(f"Qwen config path does not exist: {source_text}")

    model_id = quote(source_text.strip("/"), safe="/")
    rev = quote(str(revision).strip("/") or "main", safe="")
    url = f"https://huggingface.co/{model_id}/resolve/{rev}/config.json"
    with urlopen(url, timeout=30) as handle:
        return json.loads(handle.read().decode("utf-8"))


def _layer(
    phase: str,
    network: str,
    idx: int,
    suffix: str,
    *,
    k: int,
    c: int,
    y: int,
    c_tile: int,
    c_stride: int,
    y_map: str,
    y_tile: int,
    y_stride: int,
) -> LayerSpec:
    return LayerSpec(
        phase=phase,
        network=network,
        name=f"L{idx:02d}_{suffix}",
        layer_type="CONV",
        role={
            "Q_Proj": "q_proj",
            "K_Proj": "k_proj",
            "V_Proj": "v_proj",
            "Attn_Scores_QKt": "attn_scores_qkt",
            "Attn_AV": "attn_av",
            "Out_Proj": "out_proj",
            "FFN_Gate": "ffn_up",
            "FFN_Up": "ffn_up",
            "FFN_Down": "ffn_down",
            "FFN_1": "ffn_up",
            "FFN_2": "ffn_down",
        }[suffix],
        k=int(k),
        c=int(c),
        r=1,
        s=1,
        y=int(y),
        x=1,
        stride_x=1,
        stride_y=1,
        k_map="SpatialMap",
        k_tile=1,
        k_stride=1,
        c_map="TemporalMap" if suffix in {"Attn_Scores_QKt", "Attn_AV"} else "SpatialMap",
        c_tile=int(c_tile),
        c_stride=int(c_stride),
        y_map=y_map,
        y_tile=int(y_tile),
        y_stride=int(y_stride),
    )


def qwen_phase_layers(
    config_path: str | Path,
    *,
    phase: str,
    seq_len: int = 512,
    decode_context: int = 513,
    revision: str = "main",
) -> tuple[str, list[LayerSpec]]:
    cfg = read_qwen_config(config_path, revision=revision)
    hidden = int(cfg["hidden_size"])
    intermediate = int(cfg["intermediate_size"])
    layers = int(cfg["num_hidden_layers"])
    head_dim = int(cfg.get("head_dim", hidden // int(cfg["num_attention_heads"])))
    kv_dim = int(cfg["num_key_value_heads"]) * head_dim
    if phase == "prefill":
        tokens = int(seq_len)
        attn_tokens = int(seq_len)
        proj_c_tile = 32
        ffn_c_tile = 32
        y_map = "SpatialMap"
        proj_y_tile = min(256, tokens)
        attn_score_y_tile = min(16, attn_tokens)
        attn_av_y_tile = min(8, attn_tokens)
        network = f"Qwen3_{hidden}H_{layers}L_Seq{seq_len}_GeneratedPrefill"
    elif phase == "decode":
        tokens = 1
        attn_tokens = int(decode_context)
        proj_c_tile = 128
        ffn_c_tile = 32
        y_map = "TemporalMap"
        proj_y_tile = 1
        attn_score_y_tile = min(512, attn_tokens)
        attn_av_y_tile = min(512, attn_tokens)
        network = f"Qwen3_{hidden}H_{layers}L_Ctx{decode_context}_GeneratedDecode"
    else:
        raise ValueError("phase must be 'prefill' or 'decode'.")

    specs: list[LayerSpec] = []
    idx = 1
    for _ in range(layers):
        specs.append(_layer(phase, network, idx, "Q_Proj", k=hidden, c=hidden, y=tokens, c_tile=proj_c_tile, c_stride=64 if phase == "decode" else 32, y_map=y_map, y_tile=proj_y_tile, y_stride=proj_y_tile))
        idx += 1
        specs.append(_layer(phase, network, idx, "K_Proj", k=kv_dim, c=hidden, y=tokens, c_tile=proj_c_tile, c_stride=64 if phase == "decode" else 32, y_map=y_map, y_tile=proj_y_tile, y_stride=proj_y_tile))
        idx += 1
        specs.append(_layer(phase, network, idx, "V_Proj", k=kv_dim, c=hidden, y=tokens, c_tile=proj_c_tile, c_stride=64 if phase == "decode" else 32, y_map=y_map, y_tile=proj_y_tile, y_stride=proj_y_tile))
        idx += 1
        specs.append(_layer(phase, network, idx, "Attn_Scores_QKt", k=attn_tokens, c=head_dim, y=attn_tokens, c_tile=8, c_stride=8, y_map="TemporalMap", y_tile=attn_score_y_tile, y_stride=attn_score_y_tile))
        idx += 1
        specs.append(_layer(phase, network, idx, "Attn_AV", k=head_dim, c=attn_tokens, y=attn_tokens, c_tile=16, c_stride=16, y_map="TemporalMap", y_tile=attn_av_y_tile, y_stride=attn_av_y_tile))
        idx += 1
        specs.append(_layer(phase, network, idx, "Out_Proj", k=hidden, c=hidden, y=tokens, c_tile=proj_c_tile, c_stride=64 if phase == "decode" else 32, y_map=y_map, y_tile=proj_y_tile, y_stride=proj_y_tile))
        idx += 1
        specs.append(_layer(phase, network, idx, "FFN_Gate", k=intermediate, c=hidden, y=tokens, c_tile=ffn_c_tile, c_stride=32, y_map=y_map, y_tile=proj_y_tile, y_stride=proj_y_tile))
        idx += 1
        specs.append(_layer(phase, network, idx, "FFN_Up", k=intermediate, c=hidden, y=tokens, c_tile=ffn_c_tile, c_stride=32, y_map=y_map, y_tile=proj_y_tile, y_stride=proj_y_tile))
        idx += 1
        specs.append(_layer(phase, network, idx, "FFN_Down", k=hidden, c=intermediate, y=tokens, c_tile=ffn_c_tile, c_stride=32, y_map=y_map, y_tile=proj_y_tile, y_stride=proj_y_tile))
        idx += 1
    return network, specs
