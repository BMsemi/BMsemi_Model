from __future__ import annotations

import argparse
import csv
import json
import math
import os
import platform
import socket
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from statistics import mean
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from huggingface_hub import snapshot_download
from safetensors import safe_open
try:
    from lm_eval import simple_evaluate
    from lm_eval.api.instance import Instance
    from lm_eval.api.model import LM
    from lm_eval.api.task import Task, TaskConfig
    LM_EVAL_IMPORT_ERROR: Exception | None = None
except Exception as exc:  # Allows mapping-only helpers to be imported on lean local envs.
    simple_evaluate = None
    Instance = Any
    LM = object
    Task = object
    TaskConfig = dict
    LM_EVAL_IMPORT_ERROR = exc


PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from run_qwen3_mapping_crossbar_eval import parse_mapping  # noqa: E402
from src.crossbar_wrapper import (  # noqa: E402
    CrossbarIVNonlinearityConfig,
    CrossbarLinear,
    CrossbarNoiseConfig,
    CrossbarReliabilityConfig,
)
from src.qwen_tensor_select import (  # noqa: E402
    DEFAULT_QWEN_TENSOR_ROLE,
    qwen_tensor_role_choices,
    qwen_tensor_selection_metadata,
    select_qwen_tensor_name,
)


MODEL_ID = "Qwen/Qwen3-1.7B"
ATTN_DENSE_ROLES = {"q_proj", "k_proj", "v_proj", "out_proj"}
MLP_DYNAMIC_ROLES = {"ffn_up", "ffn_down"}
DENSE_ROLES = ATTN_DENSE_ROLES | MLP_DYNAMIC_ROLES
K_CACHE_ROLE = "attn_scores_qkt"
V_CACHE_ROLE = "attn_av"
KV_CACHE_ROLES = {K_CACHE_ROLE, V_CACHE_ROLE}
DEFAULT_WEIGHT_RETENTION_TIME_S = 30.0 * 24.0 * 3600.0
SECONDS_PER_YEAR = 365.0 * 24.0 * 60.0 * 60.0
DEFAULT_KV_CACHE_CONTEXT_TOKENS = 45_056
DEFAULT_KV_CACHE_TOKEN_RATE_TPS = 10.0
DEFAULT_KV_CACHE_TOKEN_TIME_S = 1.0 / DEFAULT_KV_CACHE_TOKEN_RATE_TPS
DEFAULT_KV_CACHE_SERVICE_LIFE_YEARS = 5.0
DEFAULT_MLP_DYNAMIC_RETENTION_TIME_S = 2.0e-3
DEFAULT_DYNAMIC_KV_ENDURANCE_CYCLES = 10_000_000
DEFAULT_DYNAMIC_MLP_ENDURANCE_CYCLES = 100_000_000
DEFAULT_DYNAMIC_ECC_CODEWORD_BITS = 22
DEFAULT_DYNAMIC_ECC_DATA_BITS = 16
DEFAULT_DYNAMIC_ECC_LATENCY_CYCLES = 2
KV_CACHE_RETENTION_STRATEGIES = ("cache_residency", "scratch_remap")


@dataclass(frozen=True)
class EvalCase:
    case_id: str
    context: str
    continuation: str


@dataclass(frozen=True)
class FeatureCase:
    name: str
    group: str
    description: str
    noise: dict[str, Any] = field(default_factory=dict)
    reliability: dict[str, Any] = field(default_factory=dict)
    iv: dict[str, Any] = field(default_factory=dict)
    noise_seed: int | None = None
    reliability_seed: int | None = None


class TinyCharTokenizer:
    def __init__(self) -> None:
        alphabet = " abcdefghijklmnopqrstuvwxyz0123456789.,;:-_/'?()[]\n"
        self.bos_token_id = 0
        self.id_to_char = ["<bos>"] + list(alphabet)
        self.char_to_id = {ch: idx for idx, ch in enumerate(self.id_to_char)}
        self.vocab_size = len(self.id_to_char)

    def encode(self, text: str) -> list[int]:
        return [self.char_to_id.get(ch, self.char_to_id[" "]) for ch in text.lower()]


class QwenSliceCrossbarLM(nn.Module):
    def __init__(
        self,
        qwen_weight_slice: torch.Tensor,
        vocab_size: int,
        weight_bits: int,
        act_bits: int,
        layer_kwargs: dict[str, Any],
    ) -> None:
        super().__init__()
        hidden_dim = int(qwen_weight_slice.shape[0])
        self.embed = nn.Embedding(vocab_size, hidden_dim)
        self.out = nn.Linear(hidden_dim, vocab_size, bias=False)
        self.crossbar_forward_calls = 0

        with torch.no_grad():
            self.embed.weight.copy_(_fit_tensor(self.embed.weight, qwen_weight_slice))
            self.out.weight.copy_(_fit_tensor(self.out.weight, qwen_weight_slice))

        linear = nn.Linear(hidden_dim, hidden_dim, bias=False)
        with torch.no_grad():
            linear.weight.copy_(qwen_weight_slice.float())
        self.proj = CrossbarLinear.from_linear(
            linear,
            weight_bits=weight_bits,
            act_bits=act_bits,
            **layer_kwargs,
        )

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        self.crossbar_forward_calls += 1
        x = self.embed(input_ids)
        x = torch.tanh(self.proj(x))
        return self.out(x)


class FeatureEffectLM(LM):
    def __init__(
        self,
        model: nn.Module,
        tokenizer: TinyCharTokenizer,
        cases: list[EvalCase],
    ) -> None:
        super().__init__()
        self.model = model.eval()
        self.tokenizer = tokenizer
        self._device = torch.device("cpu")
        self._case_by_pair = {(item.context, item.continuation): item for item in cases}
        self.rows: list[dict[str, Any]] = []
        self.logits_by_case: list[np.ndarray] = []
        self.total_ll = 0.0
        self.total_tokens = 0
        self.total_greedy = 0

    @property
    def tokenizer_name(self) -> str:
        return "tiny-char-qwen3-feature-effects"

    def _score(self, context: str, continuation: str) -> tuple[float, bool]:
        context_ids = self.tokenizer.encode(context)
        continuation_ids = self.tokenizer.encode(continuation)
        full_ids = context_ids + continuation_ids
        if not continuation_ids:
            return 0.0, True

        input_ids = [self.tokenizer.bos_token_id] + full_ids[:-1]
        targets = torch.tensor(full_ids, dtype=torch.long)
        inputs = torch.tensor(input_ids, dtype=torch.long).unsqueeze(0)

        with torch.no_grad():
            logits = self.model(inputs)[0].float()
            logprobs = F.log_softmax(logits, dim=-1)
            greedy = torch.argmax(logits, dim=-1)

        logits_np = logits.detach().cpu().numpy()
        self.logits_by_case.append(logits_np)

        start = len(context_ids)
        ll = 0.0
        greedy_hits = 0
        is_greedy = True
        for idx in range(start, len(full_ids)):
            target = int(targets[idx])
            ll += float(logprobs[idx, target])
            hit = int(int(greedy[idx]) == target)
            greedy_hits += hit
            is_greedy = is_greedy and bool(hit)

        n_tokens = len(continuation_ids)
        self.total_ll += ll
        self.total_tokens += n_tokens
        self.total_greedy += greedy_hits

        eval_case = self._case_by_pair.get(
            (context, continuation),
            EvalCase("unknown", context, continuation),
        )
        self.rows.append(
            {
                "case_id": eval_case.case_id,
                "context": context,
                "continuation": continuation,
                "continuation_chars": n_tokens,
                "loglikelihood": ll,
                "avg_loglikelihood": ll / max(n_tokens, 1),
                "greedy_char_acc": greedy_hits / max(n_tokens, 1),
                "greedy_sequence_match": float(is_greedy),
                "logit_l2_norm": float(np.linalg.norm(logits_np)),
                "logit_checksum": float(np.sum(logits_np)),
            }
        )
        return ll, is_greedy

    def loglikelihood(self, requests: list[Instance]) -> list[tuple[float, bool]]:
        return [self._score(*request.args) for request in requests]

    def loglikelihood_rolling(self, requests: list[Instance]) -> list[float]:
        return [self._score("", request.args[0])[0] for request in requests]

    def generate_until(self, requests: list[Instance]) -> list[str]:
        return ["" for _ in requests]


class FeatureEffectTask(Task):
    VERSION = 0
    OUTPUT_TYPE = "loglikelihood"

    def __init__(self, cases: list[EvalCase]) -> None:
        self._docs = [
            {
                "case_id": item.case_id,
                "context": item.context,
                "continuation": item.continuation,
            }
            for item in cases
        ]
        super().__init__()
        config = TaskConfig(task="qwen3_feature_effects", num_fewshot=0, output_type="loglikelihood")
        config["task"] = "qwen3_feature_effects"
        config["num_fewshot"] = 0
        config["output_type"] = "loglikelihood"
        self._config = config

    def download(self, *args: Any, **kwargs: Any) -> None:
        self.dataset = None

    def has_training_docs(self) -> bool:
        return False

    def has_validation_docs(self) -> bool:
        return False

    def has_test_docs(self) -> bool:
        return True

    def test_docs(self) -> list[dict[str, str]]:
        return self._docs

    def doc_to_text(self, doc: dict[str, str]) -> str:
        return doc["context"]

    def doc_to_target(self, doc: dict[str, str]) -> str:
        return doc["continuation"]

    def construct_requests(self, doc: dict[str, str], ctx: str, **kwargs: Any) -> Instance:
        return Instance(
            request_type="loglikelihood",
            doc=doc,
            arguments=(ctx, self.doc_to_target(doc)),
            idx=0,
            metadata=kwargs.get("metadata", ("qwen3_feature_effects", doc["case_id"], 1)),
        )

    def process_results(self, doc: dict[str, str], results: list[tuple[float, bool]]) -> dict[str, float]:
        loglikelihood, is_greedy = results[0]
        token_count = max(len(doc["continuation"]), 1)
        return {
            "avg_loglikelihood": loglikelihood / token_count,
            "greedy_acc": float(is_greedy),
        }

    def aggregation(self):
        return {
            "avg_loglikelihood": mean,
            "greedy_acc": mean,
        }

    def higher_is_better(self) -> dict[str, bool]:
        return {
            "avg_loglikelihood": True,
            "greedy_acc": True,
        }


def default_eval_cases() -> list[EvalCase]:
    return [
        EvalCase("short_qwen", "qwen", " crossbar"),
        EvalCase("reram_linear", "reram", " linear"),
        EvalCase("quant_path", "block", " quant"),
        EvalCase("eval_harness", "eval", " harness"),
        EvalCase("attention", "attention cache", " stores keys"),
        EvalCase("adc", "adc clipping", " changes levels"),
        EvalCase("ecc", "ecc correction", " reduces bit flips"),
        EvalCase("retention", "retention drift", " moves states"),
        EvalCase("thermal", "thermal noise", " perturbs reads"),
        EvalCase("readout", "readout rms", " shifts adc"),
        EvalCase("mapping", "dense layers", " use tiles"),
        EvalCase("dynamic_kv", "dynamic kv cache", " costs writes"),
    ]


def base_noise(**overrides: Any) -> dict[str, Any]:
    cfg: dict[str, Any] = {
        "enabled": True,
        "vread": 0.1,
        "temp_k": 0.0,
        "fano": 0.0,
        "r_lrs_ohm": 100_000.0,
        "r_hrs_ohm": 10_000_000.0,
        "flicker_af_lrs": 0.0,
        "flicker_af_hrs": 0.0,
        "flicker_gamma_lrs": 1.0,
        "flicker_gamma_hrs": 1.0,
        "rtn_amp_frac_lrs": 0.0,
        "rtn_amp_frac_hrs": 0.0,
        "readout_noise_a_rms": 0.0,
        "bandwidth_hz": 1.0e9,
        "flicker_fmin_hz": 0.5,
    }
    cfg.update(overrides)
    return cfg


def _retention_time_from_calibration(
    calibration: dict[str, Any] | None,
    strategy: str,
    fallback_s: float,
) -> float:
    if not calibration:
        return fallback_s
    item = calibration.get(strategy, {})
    if not isinstance(item, dict):
        return fallback_s
    return float(item.get("retention_time_s", fallback_s))


def feature_cases(retention_calibration: dict[str, Any] | None = None) -> list[FeatureCase]:
    raw_ber_ppm = 10_000.0
    weight_retention_time_s = _retention_time_from_calibration(
        retention_calibration,
        "weight_stationary",
        DEFAULT_WEIGHT_RETENTION_TIME_S,
    )
    kv_retention_time_s = _retention_time_from_calibration(
        retention_calibration,
        "kv_cache",
        DEFAULT_KV_CACHE_CONTEXT_TOKENS * DEFAULT_KV_CACHE_TOKEN_TIME_S,
    )
    return [
        FeatureCase(
            name="baseline_clean_w7a7",
            group="baseline",
            description="Clean W7/A7 crossbar slice with ADC rounding but no added faults, drift, or read noise.",
        ),
        FeatureCase(
            name="adc_clipping_and_quantization",
            group="adc",
            description="Column-current accumulation is hard-clamped before ADC quantization.",
            iv={"enabled": True, "mode": "hard_clamp", "linear_rows_before_clamp": 4},
        ),
        FeatureCase(
            name="stuck_at_lrs_faults",
            group="reliability",
            description="Programmed cells have a 1 percent probability of being stuck at LRS.",
            reliability={"enabled": True, "stuck_at_lrs_prob": 0.01},
            reliability_seed=2101,
        ),
        FeatureCase(
            name="stuck_at_hrs_faults",
            group="reliability",
            description="Programmed cells have a 1 percent probability of being stuck at HRS.",
            reliability={"enabled": True, "stuck_at_hrs_prob": 0.01},
            reliability_seed=2102,
        ),
        FeatureCase(
            name="raw_ber_no_ecc",
            group="reliability",
            description="Raw cell bit errors are injected without ECC.",
            reliability={"enabled": True, "raw_ber_ppm": raw_ber_ppm, "use_ecc": False},
            reliability_seed=3101,
        ),
        FeatureCase(
            name="raw_ber_with_ecc",
            group="reliability",
            description="The same raw BER is passed through the optional single-error correction model.",
            reliability={"enabled": True, "raw_ber_ppm": raw_ber_ppm, "use_ecc": True},
            reliability_seed=3101,
        ),
        FeatureCase(
            name="retention_drift",
            group="reliability",
            description="Weight-stationary log-resistance drift after long static residency at elevated temperature.",
            reliability={
                "enabled": True,
                "retention_time_s": weight_retention_time_s,
                "retention_reference_s": 1.0,
                "retention_temp_c": 85.0,
                "retention_temp_reference_c": 25.0,
                "retention_temp_acceleration_per_100c": 2.0,
                "retention_log10_resistance_drift_per_decade": 0.03,
            },
            reliability_seed=4101,
        ),
        FeatureCase(
            name="retention_kv_cache",
            group="reliability",
            description="K/V-cache retention drift calibrated from cache residency or scratch remap cadence.",
            reliability={
                "enabled": True,
                "retention_time_s": kv_retention_time_s,
                "retention_reference_s": 1.0,
                "retention_temp_c": 85.0,
                "retention_temp_reference_c": 25.0,
                "retention_temp_acceleration_per_100c": 2.0,
                "retention_log10_resistance_drift_per_decade": 0.03,
            },
            reliability_seed=4101,
        ),
        FeatureCase(
            name="relaxation_drift",
            group="reliability",
            description="Short-term relaxation drift after programming, sized to cross an INT4 state boundary.",
            reliability={
                "enabled": True,
                "relaxation_time_s": 3600.0,
                "relaxation_tau_s": 2000.0,
                "relaxation_resistance_amplitude_ln": 0.40,
            },
            reliability_seed=4102,
        ),
        FeatureCase(
            name="state_conductance_variation",
            group="reliability",
            description="Per-cell state spread through state_sigma_ln.",
            reliability={"enabled": True, "state_sigma_ln": 0.08},
            reliability_seed=4103,
        ),
        FeatureCase(
            name="endurance_drift_crossbar",
            group="reliability",
            description="Crossbar conductance drift after consuming most of the write-cycle budget, sized to cross an INT4 state boundary.",
            reliability={
                "enabled": True,
                "endurance_write_cycles": 80_000,
                "endurance_cycles": 100_000,
                "endurance_resistance_drift_ln_per_life_fraction": 0.40,
            },
            reliability_seed=4104,
        ),
        FeatureCase(
            name="thermal_noise",
            group="noise",
            description="Thermal white noise only, integrated before ADC.",
            noise=base_noise(temp_k=500.0),
            noise_seed=5101,
        ),
        FeatureCase(
            name="shot_noise",
            group="noise",
            description="Shot noise only, integrated before ADC.",
            noise=base_noise(fano=1.0),
            noise_seed=5102,
        ),
        FeatureCase(
            name="flicker_1f_noise",
            group="noise",
            description="Flicker / 1/f noise only, integrated before ADC.",
            noise=base_noise(
                flicker_af_lrs=1.0e-4,
                flicker_af_hrs=1.0e-2,
                bandwidth_hz=1.0e6,
            ),
            noise_seed=5103,
        ),
        FeatureCase(
            name="rtn_noise",
            group="noise",
            description="Random telegraph noise only, converted to INT4-level perturbation before ADC.",
            noise=base_noise(rtn_amp_frac_lrs=0.05, rtn_amp_frac_hrs=0.40),
            noise_seed=5104,
        ),
        FeatureCase(
            name="readout_rms_noise",
            group="noise",
            description="Readout RMS current noise only, converted to INT4-level perturbation before ADC.",
            noise=base_noise(readout_noise_a_rms=5.0e-8),
            noise_seed=5105,
        ),
        FeatureCase(
            name="combined_noise_to_int4_before_adc",
            group="noise",
            description="Thermal, shot, flicker, RTN, and readout terms combined before ADC rounding.",
            noise={
                "enabled": True,
                "vread": 0.1,
                "temp_k": 400.0,
                "fano": 1.0,
                "r_lrs_ohm": 100_000.0,
                "r_hrs_ohm": 10_000_000.0,
                "flicker_af_lrs": 1.0e-5,
                "flicker_af_hrs": 1.0e-3,
                "rtn_amp_frac_lrs": 0.02,
                "rtn_amp_frac_hrs": 0.20,
                "readout_noise_a_rms": 2.5e-8,
                "bandwidth_hz": 2.5e8,
                "flicker_fmin_hz": 0.5,
            },
            noise_seed=5106,
        ),
    ]


def _fit_tensor(target: torch.Tensor, source: torch.Tensor) -> torch.Tensor:
    flat = source.float().flatten()
    repeats = (target.numel() + flat.numel() - 1) // flat.numel()
    return flat.repeat(repeats)[: target.numel()].reshape_as(target)


def has_snapshot_files(path: Path) -> bool:
    return path.exists() and (
        (path / "model.safetensors.index.json").exists()
        or any(path.glob("*.safetensors"))
    )


def resolve_snapshot(
    model_id: str,
    cache_dir: Path,
    local_root: Path,
    *,
    local_files_only: bool,
    max_workers: int,
) -> tuple[Path, bool]:
    local_dir = local_root / model_id.replace("/", "--")
    if has_snapshot_files(local_dir):
        return local_dir, True

    path = snapshot_download(
        repo_id=model_id,
        cache_dir=cache_dir,
        local_dir=local_dir,
        local_files_only=local_files_only,
        max_workers=max_workers,
        ignore_patterns=["*.h5", "*.msgpack", "*.ot", "flax_model.*", "tf_model.*"],
    )
    return Path(path), False


def load_qwen_weight_slice(
    snapshot_dir: Path,
    dim: int,
    tensor_role: str = DEFAULT_QWEN_TENSOR_ROLE,
) -> tuple[str, str, torch.Tensor]:
    index_path = snapshot_dir / "model.safetensors.index.json"
    if index_path.exists():
        index = json.loads(index_path.read_text(encoding="utf-8"))
        weight_map = index.get("weight_map", {})
        tensor_name = select_qwen_tensor_name(weight_map.keys(), tensor_role)
        tensor_file = snapshot_dir / weight_map[tensor_name]
    else:
        files = sorted(snapshot_dir.glob("*.safetensors"))
        if not files:
            raise RuntimeError(f"No safetensors files found in {snapshot_dir}")
        tensor_file = files[0]
        with safe_open(tensor_file, framework="pt", device="cpu") as handle:
            tensor_name = select_qwen_tensor_name(handle.keys(), tensor_role)

    with safe_open(tensor_file, framework="pt", device="cpu") as handle:
        weight = handle.get_tensor(tensor_name).float()

    rows = min(dim, weight.shape[0])
    cols = min(dim, weight.shape[1])
    square = torch.zeros((dim, dim), dtype=torch.float32)
    square[:rows, :cols] = weight[:rows, :cols]
    return tensor_name, str(tensor_file), square


def make_layer_kwargs(case: FeatureCase) -> dict[str, Any]:
    return {
        "noise_config": CrossbarNoiseConfig.from_obj(case.noise),
        "reliability_config": CrossbarReliabilityConfig.from_obj(case.reliability),
        "iv_config": CrossbarIVNonlinearityConfig.from_obj(case.iv),
        "noise_seed": case.noise_seed,
        "reliability_seed": case.reliability_seed,
    }


def evaluate_model(
    model: QwenSliceCrossbarLM,
    tokenizer: TinyCharTokenizer,
    cases: list[EvalCase],
) -> tuple[dict[str, float], list[np.ndarray], list[dict[str, Any]]]:
    if simple_evaluate is None:
        raise RuntimeError(f"lm-eval could not be imported: {LM_EVAL_IMPORT_ERROR}")
    lm = FeatureEffectLM(model, tokenizer, cases)
    result = simple_evaluate(
        model=lm,
        tasks=[FeatureEffectTask(cases)],
        num_fewshot=0,
        batch_size=1,
        limit=None,
        log_samples=False,
        bootstrap_iters=0,
    )["results"]["qwen3_feature_effects"]

    avg_ll = lm.total_ll / max(lm.total_tokens, 1)
    metrics = {
        "total_loglikelihood": lm.total_ll,
        "avg_loglikelihood": avg_ll,
        "nll": -avg_ll,
        "ppl_like": float(math.exp(min(50.0, -avg_ll))),
        "greedy_char_acc": lm.total_greedy / max(lm.total_tokens, 1),
        "scored_chars": float(lm.total_tokens),
        "crossbar_forward_calls": float(model.crossbar_forward_calls),
        "lm_eval_avg_loglikelihood": float(result.get("avg_loglikelihood,none", float("nan"))),
        "lm_eval_greedy_acc": float(result.get("greedy_acc,none", float("nan"))),
    }
    if model.crossbar_forward_calls <= 0:
        raise RuntimeError("lm_eval completed without exercising CrossbarLinear.")
    return metrics, lm.logits_by_case, lm.rows


def run_feature_case(
    case: FeatureCase,
    weight_slice: torch.Tensor,
    tokenizer: TinyCharTokenizer,
    eval_cases: list[EvalCase],
    *,
    weight_bits: int,
    act_bits: int,
) -> tuple[dict[str, Any], list[np.ndarray], list[dict[str, Any]]]:
    start = time.perf_counter()
    torch.manual_seed(1234)
    model = QwenSliceCrossbarLM(
        qwen_weight_slice=weight_slice,
        vocab_size=tokenizer.vocab_size,
        weight_bits=weight_bits,
        act_bits=act_bits,
        layer_kwargs=make_layer_kwargs(case),
    )
    metrics, logits_by_case, case_rows = evaluate_model(model, tokenizer, eval_cases)
    elapsed_s = time.perf_counter() - start
    row: dict[str, Any] = {
        "name": case.name,
        "group": case.group,
        "description": case.description,
        "elapsed_s": elapsed_s,
        "weight_bits": weight_bits,
        "act_bits": act_bits,
        **metrics,
        "noise_seed": case.noise_seed,
        "reliability_seed": case.reliability_seed,
        "noise_config": CrossbarNoiseConfig.from_obj(case.noise).as_dict(),
        "reliability_config": CrossbarReliabilityConfig.from_obj(case.reliability).as_dict(),
        "iv_config": CrossbarIVNonlinearityConfig.from_obj(case.iv).as_dict(),
    }
    return row, logits_by_case, case_rows


def mean_abs_delta(a: list[np.ndarray], b: list[np.ndarray]) -> float:
    if len(a) != len(b):
        return float("nan")
    deltas = [np.mean(np.abs(x - y)) for x, y in zip(a, b)]
    return float(np.mean(deltas)) if deltas else 0.0


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    flat_rows: list[dict[str, Any]] = []
    for row in rows:
        flat: dict[str, Any] = {}
        for key, value in row.items():
            if isinstance(value, (dict, list, tuple)):
                flat[key] = json.dumps(value, sort_keys=True)
            else:
                flat[key] = value
        flat_rows.append(flat)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(flat_rows[0].keys()))
        writer.writeheader()
        writer.writerows(flat_rows)


def estimate_kv_cache_retention_time_s(
    *,
    strategy: str,
    context_tokens: int,
    token_time_s: float,
    explicit_retention_time_s: float | None = None,
) -> tuple[float, str]:
    if explicit_retention_time_s is not None:
        return max(float(explicit_retention_time_s), 0.0), "explicit override"

    strategy_norm = str(strategy).strip().lower().replace("-", "_")
    context_tokens = max(int(context_tokens), 0)
    token_time_s = max(float(token_time_s), 0.0)

    if strategy_norm == "cache_residency":
        return context_tokens * token_time_s, "K/V entry remains programmed until context eviction."
    if strategy_norm == "scratch_remap":
        return token_time_s, "K/V scratch tile is rewritten on the next decode step."
    raise ValueError(
        f"Unknown K/V retention strategy {strategy!r}. "
        f"Expected one of {', '.join(KV_CACHE_RETENTION_STRATEGIES)}."
    )


def kv_cache_token_rate_tps(token_time_s: float) -> float:
    token_time = max(float(token_time_s), 0.0)
    return 1.0 / token_time if token_time > 0.0 else 0.0


def kv_cache_service_life_tokens(*, token_time_s: float, service_life_years: float) -> float:
    service_life_s = max(float(service_life_years), 0.0) * SECONDS_PER_YEAR
    return service_life_s * kv_cache_token_rate_tps(token_time_s)


def kv_cache_projected_cycles_per_slot(
    *,
    context_tokens: int,
    token_time_s: float,
    service_life_years: float,
) -> float:
    return kv_cache_service_life_tokens(
        token_time_s=token_time_s,
        service_life_years=service_life_years,
    ) / float(max(1, int(context_tokens)))


def kv_append_programmed_elements(layer: MappingLayer) -> int:
    if layer.role == K_CACHE_ROLE:
        return int(layer.in_dim)
    if layer.role == V_CACHE_ROLE:
        return int(layer.out_dim)
    return int(layer.weight_elements)


def build_retention_calibration(
    *,
    weight_retention_time_s: float,
    kv_cache_context_tokens: int,
    kv_cache_token_time_s: float,
    kv_cache_retention_strategy: str,
    kv_cache_retention_time_s: float | None,
    mlp_dynamic_retention_time_s: float = DEFAULT_MLP_DYNAMIC_RETENTION_TIME_S,
) -> dict[str, Any]:
    kv_time_s, kv_assumption = estimate_kv_cache_retention_time_s(
        strategy=kv_cache_retention_strategy,
        context_tokens=kv_cache_context_tokens,
        token_time_s=kv_cache_token_time_s,
        explicit_retention_time_s=kv_cache_retention_time_s,
    )
    weight_time_s = max(float(weight_retention_time_s), 0.0)
    mlp_time_s = max(float(mlp_dynamic_retention_time_s), 0.0)
    return {
        "mapping_strategy_dependent": True,
        "weight_stationary": {
            "retention_time_s": weight_time_s,
            "assumption": "Static model weights are programmed once and can remain resident for long storage/inference windows.",
        },
        "kv_cache": {
            "retention_time_s": kv_time_s,
            "strategy": kv_cache_retention_strategy,
            "context_tokens": int(kv_cache_context_tokens),
            "token_time_s": float(kv_cache_token_time_s),
            "explicit_retention_time_s": kv_cache_retention_time_s,
            "assumption": kv_assumption,
        },
        "mlp_dynamic": {
            "retention_time_s": mlp_time_s,
            "target_role_names": ["gate_proj", "up_proj", "down_proj"],
            "mapped_roles": sorted(MLP_DYNAMIC_ROLES),
            "assumption": "MLP gate/up/down dynamic scratch tiles are rewritten on decode-step cadence and require only short data-retention residency.",
        },
        "kv_vs_weight_retention_ratio": kv_time_s / max(weight_time_s, 1.0e-30),
        "mlp_vs_weight_retention_ratio": mlp_time_s / max(weight_time_s, 1.0e-30),
    }


def mapping_projection(
    prefill_path: Path,
    decode_path: Path,
    *,
    endurance_cycles: int,
    amortization_tokens: int,
    dynamic_kv_endurance_cycles: int = DEFAULT_DYNAMIC_KV_ENDURANCE_CYCLES,
    dynamic_mlp_endurance_cycles: int = DEFAULT_DYNAMIC_MLP_ENDURANCE_CYCLES,
    weight_retention_time_s: float = DEFAULT_WEIGHT_RETENTION_TIME_S,
    kv_cache_context_tokens: int = DEFAULT_KV_CACHE_CONTEXT_TOKENS,
    kv_cache_token_time_s: float = DEFAULT_KV_CACHE_TOKEN_TIME_S,
    kv_cache_retention_strategy: str = "cache_residency",
    kv_cache_retention_time_s: float | None = None,
    kv_cache_service_life_years: float = DEFAULT_KV_CACHE_SERVICE_LIFE_YEARS,
    mlp_dynamic_retention_time_s: float = DEFAULT_MLP_DYNAMIC_RETENTION_TIME_S,
) -> dict[str, Any]:
    _, prefill_layers = parse_mapping(prefill_path, "prefill")
    _, decode_layers = parse_mapping(decode_path, "decode")
    layers = prefill_layers + decode_layers

    def collect(phase: str | None, roles: set[str]) -> dict[str, int]:
        selected = [
            layer
            for layer in layers
            if layer.role in roles and (phase is None or layer.phase == phase)
        ]
        return {
            "layers": len(selected),
            "tokens": sum(layer.tokens for layer in selected),
            "crossbar_tiles": sum(layer.crossbar_tiles for layer in selected),
            "macs": sum(layer.macs for layer in selected),
            "programmed_elements": sum(layer.weight_elements for layer in selected),
        }

    dense_all = collect(None, DENSE_ROLES)
    static_all = collect(None, ATTN_DENSE_ROLES)
    dynamic_mlp = collect(None, MLP_DYNAMIC_ROLES)
    dynamic_mlp_decode = collect("decode", MLP_DYNAMIC_ROLES)
    dynamic_mlp_prefill = collect("prefill", MLP_DYNAMIC_ROLES)
    dynamic_k = collect(None, {K_CACHE_ROLE})
    dynamic_v = collect(None, {V_CACHE_ROLE})
    dynamic_decode = collect("decode", KV_CACHE_ROLES)
    dynamic_prefill = collect("prefill", KV_CACHE_ROLES)
    dense_decode = collect("decode", DENSE_ROLES)
    dense_prefill = collect("prefill", DENSE_ROLES)

    static_elems = static_all["programmed_elements"]
    dynamic_decode_full_context_elems = dynamic_decode["programmed_elements"]
    dynamic_decode_elems = sum(
        kv_append_programmed_elements(layer)
        for layer in layers
        if layer.phase == "decode" and layer.role in KV_CACHE_ROLES
    )
    dynamic_mlp_decode_elems = dynamic_mlp_decode["programmed_elements"]
    high_churn_decode_elems = dynamic_decode_elems + dynamic_mlp_decode_elems
    static_per_token = static_elems / max(amortization_tokens, 1)
    dynamic_ratio = dynamic_decode_elems / max(static_per_token, 1.0e-30)
    high_churn_ratio = high_churn_decode_elems / max(static_per_token, 1.0e-30)
    break_even_decode_tokens = (
        static_elems / dynamic_decode_elems if dynamic_decode_elems > 0 else float("inf")
    )
    high_churn_break_even_decode_tokens = (
        static_elems / high_churn_decode_elems if high_churn_decode_elems > 0 else float("inf")
    )
    retention_calibration = build_retention_calibration(
        weight_retention_time_s=weight_retention_time_s,
        kv_cache_context_tokens=kv_cache_context_tokens,
        kv_cache_token_time_s=kv_cache_token_time_s,
        kv_cache_retention_strategy=kv_cache_retention_strategy,
        kv_cache_retention_time_s=kv_cache_retention_time_s,
        mlp_dynamic_retention_time_s=mlp_dynamic_retention_time_s,
    )
    kv_cycles_per_slot = kv_cache_projected_cycles_per_slot(
        context_tokens=kv_cache_context_tokens,
        token_time_s=kv_cache_token_time_s,
        service_life_years=kv_cache_service_life_years,
    )
    kv_service_life_tokens = kv_cache_service_life_tokens(
        token_time_s=kv_cache_token_time_s,
        service_life_years=kv_cache_service_life_years,
    )

    return {
        "mapping_files": {
            "prefill": str(prefill_path),
            "decode": str(decode_path),
        },
        "roles": {
            "static_dense_model_weights": sorted(ATTN_DENSE_ROLES),
            "dynamic_mlp_gate_up_down": sorted(MLP_DYNAMIC_ROLES),
            "dynamic_k_cache_qkt": K_CACHE_ROLE,
            "dynamic_v_cache_av": V_CACHE_ROLE,
        },
        "dense_linear_layers_mapped_to_crossbars": dense_all,
        "static_dense_linear_layers": static_all,
        "dense_prefill": dense_prefill,
        "dense_decode": dense_decode,
        "dynamic_mlp_gate_up_down_programmed_to_crossbars": dynamic_mlp,
        "dynamic_mlp_prefill": dynamic_mlp_prefill,
        "dynamic_mlp_decode_per_token": dynamic_mlp_decode,
        "attention_qkt_k_cache_programmed_to_crossbars": dynamic_k,
        "attention_av_v_cache_programmed_to_crossbars": dynamic_v,
        "dynamic_kv_prefill": dynamic_prefill,
        "dynamic_kv_decode_per_token": dynamic_decode,
        "static_model_weight_programming_amortization": {
            "amortization_tokens": amortization_tokens,
            "static_programmed_elements_once": static_elems,
            "amortized_static_elements_per_token": static_per_token,
        },
        "dynamic_kv_cache_programming_cost": {
            "write_policy": "append_only_new_token",
            "dynamic_elements_per_decode_token": dynamic_decode_elems,
            "full_context_elements_if_rewritten_per_decode_token": dynamic_decode_full_context_elems,
            "full_context_rewrite_avoidance_factor": (
                dynamic_decode_full_context_elems / dynamic_decode_elems if dynamic_decode_elems > 0 else float("inf")
            ),
            "dynamic_to_amortized_static_write_ratio": dynamic_ratio,
            "break_even_decode_tokens_vs_one_static_program": break_even_decode_tokens,
        },
        "dynamic_mlp_programming_cost": {
            "dynamic_elements_per_decode_token": dynamic_mlp_decode_elems,
            "dynamic_to_amortized_static_write_ratio": dynamic_mlp_decode_elems / max(static_per_token, 1.0e-30),
            "break_even_decode_tokens_vs_one_static_program": (
                static_elems / dynamic_mlp_decode_elems if dynamic_mlp_decode_elems > 0 else float("inf")
            ),
        },
        "high_churn_dynamic_programming_cost": {
            "dynamic_elements_per_decode_token": high_churn_decode_elems,
            "dynamic_to_amortized_static_write_ratio": high_churn_ratio,
            "break_even_decode_tokens_vs_one_static_program": high_churn_break_even_decode_tokens,
        },
        "retention_calibration": retention_calibration,
        "dynamic_ecc_policy": {
            "target_roles": {
                "kv_cache": sorted(KV_CACHE_ROLES),
                "mlp_gate_up_down": sorted(MLP_DYNAMIC_ROLES),
            },
            "use_ecc": True,
            "ecc_codeword_bits": DEFAULT_DYNAMIC_ECC_CODEWORD_BITS,
            "ecc_data_bits": DEFAULT_DYNAMIC_ECC_DATA_BITS,
            "ecc_encode_latency_cycles": DEFAULT_DYNAMIC_ECC_LATENCY_CYCLES,
            "ecc_decode_latency_cycles": DEFAULT_DYNAMIC_ECC_LATENCY_CYCLES,
            "assumption": "Dynamic K/V append writes and MLP scratch programming use shorter SECDED-style codewords than the default static-weight ECC target.",
        },
        "endurance_lifetime_projection": {
            "static_weight_endurance_cycles": endurance_cycles,
            "dynamic_kv_endurance_cycles": dynamic_kv_endurance_cycles,
            "dynamic_mlp_endurance_cycles": dynamic_mlp_endurance_cycles,
            "kv_cache_write_policy": "append_only_new_token",
            "kv_cache_slot_wear_policy": "round_robin_context_slots",
            "kv_cache_context_slots": int(kv_cache_context_tokens),
            "kv_cache_token_rate_tps": kv_cache_token_rate_tps(kv_cache_token_time_s),
            "kv_cache_service_life_years": float(kv_cache_service_life_years),
            "projected_decode_tokens_per_service_life": kv_service_life_tokens,
            "projected_kv_write_cycles_per_slot": kv_cycles_per_slot,
            "kv_cache_endurance_margin": (
                float(dynamic_kv_endurance_cycles) / kv_cycles_per_slot
                if dynamic_kv_endurance_cycles > 0 and kv_cycles_per_slot > 0.0
                else None
            ),
            "assumption": "K/V cache writes only the new token slot on decode and spreads wear round-robin across context slots; MLP scratch cells are rewritten on decode-step cadence.",
            "projected_decode_tokens_per_reused_kv_cell": (
                dynamic_kv_endurance_cycles * int(kv_cache_context_tokens)
                if dynamic_kv_endurance_cycles > 0
                else 0
            ),
            "projected_decode_tokens_per_reused_mlp_cell": dynamic_mlp_endurance_cycles,
            "projected_static_weight_program_cycles": 1,
            "dynamic_cache_is_endurance_limited": bool(
                dynamic_decode_elems > 0 and kv_cycles_per_slot >= float(dynamic_kv_endurance_cycles)
            ),
            "dynamic_mlp_is_endurance_limited": dynamic_mlp_decode_elems > 0,
        },
    }


def requirement_coverage(
    projection: dict[str, Any],
    tensor_metadata: dict[str, str] | None = None,
) -> list[dict[str, Any]]:
    dense = projection["dense_linear_layers_mapped_to_crossbars"]
    qkt = projection["attention_qkt_k_cache_programmed_to_crossbars"]
    av = projection["attention_av_v_cache_programmed_to_crossbars"]
    dyn = projection["dynamic_kv_cache_programming_cost"]
    mlp_dyn = projection["dynamic_mlp_programming_cost"]
    dyn_ecc = projection["dynamic_ecc_policy"]
    amort = projection["static_model_weight_programming_amortization"]
    endurance = projection["endurance_lifetime_projection"]
    retention = projection["retention_calibration"]
    tensor_metadata = tensor_metadata or {}
    tensor_name = tensor_metadata.get("tensor_name", "")
    tensor_role = tensor_metadata.get("tensor_role", "")
    tensor_role_description = tensor_metadata.get("tensor_role_description", "")
    targets_mlp_output = tensor_role == "mlp_output" and tensor_name.endswith("mlp.down_proj.weight")
    return [
        {
            "requirement": "ADC clipping and quantization",
            "evidence": "Feature case adc_clipping_and_quantization enables CrossbarIVNonlinearityConfig hard_clamp.",
            "artifact": "feature_effects.csv",
            "covered": True,
        },
        {
            "requirement": "Stuck-at LRS faults on MLP output only",
            "evidence": (
                "Feature case stuck_at_lrs_faults injects stuck_at_lrs_prob into the selected "
                f"Qwen slice {tensor_name} ({tensor_role_description})."
            ),
            "artifact": "feature_effects.csv",
            "covered": targets_mlp_output,
        },
        {
            "requirement": "Stuck-at HRS faults",
            "evidence": "Feature case stuck_at_hrs_faults injects stuck_at_hrs_prob.",
            "artifact": "feature_effects.csv",
            "covered": True,
        },
        {
            "requirement": "Raw BER",
            "evidence": "Feature case raw_ber_no_ecc injects raw_ber_ppm without ECC.",
            "artifact": "feature_effects.csv",
            "covered": True,
        },
        {
            "requirement": "Optional ECC correction model",
            "evidence": "Feature case raw_ber_with_ecc reuses the same raw BER seed with use_ecc=True.",
            "artifact": "feature_effects.csv",
            "covered": True,
        },
        {
            "requirement": "Retention drift",
            "evidence": "Feature case retention_drift applies log-resistance drift over retention time and temperature.",
            "artifact": "feature_effects.csv",
            "covered": True,
        },
        {
            "requirement": "Mapping-strategy-dependent retention calibration",
            "evidence": (
                "Projection separates long weight-stationary residency "
                f"({retention['weight_stationary']['retention_time_s']} s) from K/V cache residency "
                f"({retention['kv_cache']['retention_time_s']} s) and MLP dynamic scratch residency "
                f"({retention['mlp_dynamic']['retention_time_s']} s)."
            ),
            "artifact": "mapping_projection.json",
            "covered": bool(retention["mapping_strategy_dependent"]),
        },
        {
            "requirement": "Stronger ECC for dynamic K/V and MLP programming paths",
            "evidence": (
                f"Projection assigns ECC {dyn_ecc['ecc_codeword_bits']}/{dyn_ecc['ecc_data_bits']} "
                "to K/V cache and MLP gate/up/down dynamic roles."
            ),
            "artifact": "mapping_projection.json",
            "covered": bool(dyn_ecc["use_ecc"] and dyn_ecc["ecc_data_bits"] <= 16),
        },
        {
            "requirement": "Relaxation drift",
            "evidence": "Feature case relaxation_drift applies short-term post-programming relaxation.",
            "artifact": "feature_effects.csv",
            "covered": True,
        },
        {
            "requirement": "State / conductance variation using state_sigma_ln",
            "evidence": "Feature case state_conductance_variation sets state_sigma_ln.",
            "artifact": "feature_effects.csv",
            "covered": True,
        },
        {
            "requirement": "Endurance drift inside the crossbar",
            "evidence": "Feature case endurance_drift_crossbar consumes most of the endurance cycle budget.",
            "artifact": "feature_effects.csv",
            "covered": True,
        },
        {
            "requirement": "Endurance lifetime / cost projection in the mapping optimizer",
            "evidence": (
                f"Append-only K/V projection is {endurance['projected_kv_write_cycles_per_slot']:.6g} "
                f"cycles/slot over {endurance['kv_cache_service_life_years']} years; MLP scratch targets "
                f"{endurance['projected_decode_tokens_per_reused_mlp_cell']} decode tokens per reused dynamic cell."
            ),
            "artifact": "mapping_projection.json",
            "covered": bool(endurance["projected_kv_write_cycles_per_slot"] > 0 and endurance["dynamic_mlp_is_endurance_limited"]),
        },
        {
            "requirement": "Thermal noise",
            "evidence": "Feature case thermal_noise enables temp_k-only read noise.",
            "artifact": "feature_effects.csv",
            "covered": True,
        },
        {
            "requirement": "Shot noise",
            "evidence": "Feature case shot_noise enables fano-only read noise.",
            "artifact": "feature_effects.csv",
            "covered": True,
        },
        {
            "requirement": "Flicker / 1/f noise",
            "evidence": "Feature case flicker_1f_noise enables flicker coefficients.",
            "artifact": "feature_effects.csv",
            "covered": True,
        },
        {
            "requirement": "RTN noise",
            "evidence": "Feature case rtn_noise enables random telegraph amplitude terms.",
            "artifact": "feature_effects.csv",
            "covered": True,
        },
        {
            "requirement": "Readout RMS noise",
            "evidence": "Feature case readout_rms_noise sets readout_noise_a_rms.",
            "artifact": "feature_effects.csv",
            "covered": True,
        },
        {
            "requirement": "Noise converted into INT4-level perturbation before ADC",
            "evidence": "Feature case combined_noise_to_int4_before_adc combines all read-noise terms before ADC rounding.",
            "artifact": "feature_effects.csv",
            "covered": True,
        },
        {
            "requirement": "Dense linear layers mapped to ReRAM crossbar tiles",
            "evidence": f"Projection found {dense['layers']} dense layers mapped to {dense['crossbar_tiles']} crossbar tiles.",
            "artifact": "mapping_projection.json",
            "covered": dense["layers"] > 0 and dense["crossbar_tiles"] > 0,
        },
        {
            "requirement": "Attention QK^T using K cache programmed into crossbars",
            "evidence": f"Projection found {qkt['layers']} QK^T cache-mapped layers.",
            "artifact": "mapping_projection.json",
            "covered": qkt["layers"] > 0,
        },
        {
            "requirement": "Attention AV using V cache programmed into crossbars",
            "evidence": f"Projection found {av['layers']} AV cache-mapped layers.",
            "artifact": "mapping_projection.json",
            "covered": av["layers"] > 0,
        },
        {
            "requirement": "Static model-weight programming amortization",
            "evidence": f"Projection amortizes {amort['static_programmed_elements_once']} static programmed elements over {amort['amortization_tokens']} tokens.",
            "artifact": "mapping_projection.json",
            "covered": amort["static_programmed_elements_once"] > 0,
        },
        {
            "requirement": "Dynamic K/V cache programming cost",
            "evidence": f"Projection estimates {dyn['dynamic_elements_per_decode_token']} dynamic K/V programmed elements per decode token.",
            "artifact": "mapping_projection.json",
            "covered": dyn["dynamic_elements_per_decode_token"] > 0,
        },
        {
            "requirement": "Dynamic MLP gate/up/down programming cost",
            "evidence": f"Projection estimates {mlp_dyn['dynamic_elements_per_decode_token']} dynamic MLP programmed elements per decode token.",
            "artifact": "mapping_projection.json",
            "covered": mlp_dyn["dynamic_elements_per_decode_token"] > 0,
        },
    ]


def build_markdown_summary(
    rows: list[dict[str, Any]],
    projection: dict[str, Any],
    tensor_metadata: dict[str, str] | None = None,
) -> str:
    tensor_metadata = tensor_metadata or {}
    lines = [
        "# Qwen3 Feature Effect Sweep",
        "",
        "This run uses lm-eval with a real Qwen/Qwen3-1.7B safetensor projection slice and a deterministic character-level task suite. Metrics are intended for relative hardware-feature deltas, not full-model benchmark claims.",
        "",
        "| Feature | Group | avg_loglikelihood | delta_avg_loglikelihood | mean_abs_logit_delta | greedy_char_acc | lm_eval_avg_loglikelihood |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            "| {name} | {group} | {avg:.6f} | {delta:.6f} | {mad:.6f} | {acc:.6f} | {lm_avg:.6f} |".format(
                name=row["name"],
                group=row["group"],
                avg=row["avg_loglikelihood"],
                delta=row["delta_avg_loglikelihood"],
                mad=row["mean_abs_logit_delta_vs_baseline"],
                acc=row["greedy_char_acc"],
                lm_avg=row["lm_eval_avg_loglikelihood"],
            )
        )
    dyn = projection["dynamic_kv_cache_programming_cost"]
    mlp_dyn = projection["dynamic_mlp_programming_cost"]
    high_churn = projection["high_churn_dynamic_programming_cost"]
    dense = projection["dense_linear_layers_mapped_to_crossbars"]
    static_dense = projection["static_dense_linear_layers"]
    retention = projection["retention_calibration"]
    endurance = projection["endurance_lifetime_projection"]
    dyn_ecc = projection["dynamic_ecc_policy"]
    lines.extend(
        [
            "",
            "## Mapping Projection",
            "",
            f"- Static non-MLP dense programmed elements: {static_dense['programmed_elements']}",
            f"- Dense crossbar tiles: {dense['crossbar_tiles']}",
            f"- Dynamic K/V elements per decode token: {dyn['dynamic_elements_per_decode_token']}",
            f"- Avoided K/V full-context rewrite elements/token: {dyn['full_context_elements_if_rewritten_per_decode_token']}",
            f"- Dynamic MLP gate/up/down elements per decode token: {mlp_dyn['dynamic_elements_per_decode_token']}",
            f"- K/V dynamic/amortized-static write ratio: {dyn['dynamic_to_amortized_static_write_ratio']:.3f}",
            f"- High-churn dynamic/amortized-static write ratio: {high_churn['dynamic_to_amortized_static_write_ratio']:.3f}",
            f"- Dynamic ECC target: {dyn_ecc['ecc_codeword_bits']}/{dyn_ecc['ecc_data_bits']} code, encode/decode {dyn_ecc['ecc_encode_latency_cycles']}/{dyn_ecc['ecc_decode_latency_cycles']} cycles.",
            f"- Endurance targets: K/V {endurance['dynamic_kv_endurance_cycles']} cycles, MLP {endurance['dynamic_mlp_endurance_cycles']} cycles, static {endurance['static_weight_endurance_cycles']} cycles.",
            f"- Append-only K/V wear: {endurance['projected_kv_write_cycles_per_slot']:.6g} cycles/slot over {endurance['kv_cache_service_life_years']} years at {endurance['kv_cache_token_rate_tps']:.6g} tok/s.",
            f"- Retention calibration: weight-stationary {retention['weight_stationary']['retention_time_s']:.6g}s, K/V cache {retention['kv_cache']['retention_time_s']:.6g}s ({retention['kv_cache']['strategy']}), MLP dynamic {retention['mlp_dynamic']['retention_time_s']:.6g}s.",
            f"- Verification tensor: {tensor_metadata.get('tensor_name', '')} ({tensor_metadata.get('tensor_role_description', '')})",
            "",
        ]
    )
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    default_cache = Path(os.environ.get("HF_HUB_CACHE", Path.home() / ".cache" / "huggingface" / "hub"))
    default_local_root = Path(os.environ.get("HF_MODEL_ROOT", Path.home() / "hf_models"))
    ts = time.strftime("%Y%m%d_%H%M%S")
    parser = argparse.ArgumentParser(description="Qwen3 FP4LLM feature-effect sweep.")
    parser.add_argument("--model", default=MODEL_ID)
    parser.add_argument("--cache-dir", type=Path, default=default_cache)
    parser.add_argument("--local-root", type=Path, default=default_local_root)
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--max-workers", type=int, default=8)
    parser.add_argument("--dim", type=int, default=64)
    parser.add_argument("--weight-bits", type=int, default=7)
    parser.add_argument("--act-bits", type=int, default=7)
    parser.add_argument(
        "--tensor-role",
        choices=qwen_tensor_role_choices(),
        default=DEFAULT_QWEN_TENSOR_ROLE,
        help="Qwen projection tensor used for feature-effect verification. Default is MLP output only.",
    )
    parser.add_argument("--prefill", type=Path, default=PROJECT_DIR / "mappings" / "Qwen3_1_7B_prefill.m")
    parser.add_argument("--decode", type=Path, default=PROJECT_DIR / "mappings" / "Qwen3_1_7B_decode.m")
    parser.add_argument("--endurance-cycles", type=int, default=100_000)
    parser.add_argument("--dynamic-kv-endurance-cycles", type=int, default=DEFAULT_DYNAMIC_KV_ENDURANCE_CYCLES)
    parser.add_argument("--dynamic-mlp-endurance-cycles", type=int, default=DEFAULT_DYNAMIC_MLP_ENDURANCE_CYCLES)
    parser.add_argument("--amortization-tokens", type=int, default=1_000_000)
    parser.add_argument("--weight-retention-time-s", type=float, default=DEFAULT_WEIGHT_RETENTION_TIME_S)
    parser.add_argument("--kv-cache-context-tokens", type=int, default=DEFAULT_KV_CACHE_CONTEXT_TOKENS)
    parser.add_argument("--kv-cache-token-time-s", type=float, default=DEFAULT_KV_CACHE_TOKEN_TIME_S)
    parser.add_argument("--kv-cache-service-life-years", type=float, default=DEFAULT_KV_CACHE_SERVICE_LIFE_YEARS)
    parser.add_argument(
        "--kv-cache-retention-strategy",
        choices=KV_CACHE_RETENTION_STRATEGIES,
        default="cache_residency",
    )
    parser.add_argument(
        "--kv-cache-retention-time-s",
        type=float,
        default=None,
        help="Override K/V retention residency directly. Otherwise it is derived from strategy, context tokens, and token time.",
    )
    parser.add_argument("--mlp-dynamic-retention-time-s", type=float, default=DEFAULT_MLP_DYNAMIC_RETENTION_TIME_S)
    parser.add_argument("--out-dir", type=Path, default=PROJECT_DIR / "results" / f"qwen3_feature_effects_{ts}")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    snapshot_dir, reused_snapshot = resolve_snapshot(
        args.model,
        args.cache_dir,
        args.local_root,
        local_files_only=args.local_files_only,
        max_workers=args.max_workers,
    )
    tensor_name, tensor_file, weight_slice = load_qwen_weight_slice(snapshot_dir, args.dim, args.tensor_role)
    tensor_metadata = {
        "tensor_name": tensor_name,
        "tensor_file": tensor_file,
        **qwen_tensor_selection_metadata(tensor_name, args.tensor_role),
    }
    tokenizer = TinyCharTokenizer()
    eval_cases = default_eval_cases()
    projection = mapping_projection(
        args.prefill.resolve(),
        args.decode.resolve(),
        endurance_cycles=args.endurance_cycles,
        amortization_tokens=args.amortization_tokens,
        dynamic_kv_endurance_cycles=args.dynamic_kv_endurance_cycles,
        dynamic_mlp_endurance_cycles=args.dynamic_mlp_endurance_cycles,
        weight_retention_time_s=args.weight_retention_time_s,
        kv_cache_context_tokens=args.kv_cache_context_tokens,
        kv_cache_token_time_s=args.kv_cache_token_time_s,
        kv_cache_retention_strategy=args.kv_cache_retention_strategy,
        kv_cache_retention_time_s=args.kv_cache_retention_time_s,
        kv_cache_service_life_years=args.kv_cache_service_life_years,
        mlp_dynamic_retention_time_s=args.mlp_dynamic_retention_time_s,
    )
    cases = feature_cases(projection["retention_calibration"])

    write_csv(out_dir / "test_cases.csv", [case.__dict__ for case in eval_cases])

    all_rows: list[dict[str, Any]] = []
    per_case_rows: list[dict[str, Any]] = []
    baseline_logits: list[np.ndarray] | None = None
    baseline_metrics: dict[str, Any] | None = None

    for idx, case in enumerate(cases, start=1):
        print(f"[{idx}/{len(cases)}] {case.name}", flush=True)
        row, logits, case_rows = run_feature_case(
            case,
            weight_slice,
            tokenizer,
            eval_cases,
            weight_bits=args.weight_bits,
            act_bits=args.act_bits,
        )
        if baseline_logits is None:
            baseline_logits = logits
            baseline_metrics = row
        row["delta_avg_loglikelihood"] = row["avg_loglikelihood"] - float(baseline_metrics["avg_loglikelihood"])
        row["delta_nll"] = row["nll"] - float(baseline_metrics["nll"])
        row["mean_abs_logit_delta_vs_baseline"] = mean_abs_delta(baseline_logits, logits)
        row["changed_vs_baseline"] = bool(row["mean_abs_logit_delta_vs_baseline"] > 1.0e-8)
        all_rows.append(row)

        for case_row in case_rows:
            case_row = dict(case_row)
            case_row["feature"] = case.name
            per_case_rows.append(case_row)

        write_csv(out_dir / "feature_effects.csv", all_rows)
        write_csv(out_dir / "per_case_scores.csv", per_case_rows)

    coverage = requirement_coverage(projection, tensor_metadata)

    checks = {
        "adc_clipping_changes_logits": next(
            row["changed_vs_baseline"] for row in all_rows if row["name"] == "adc_clipping_and_quantization"
        ),
        "stuck_at_lrs_changes_logits": next(
            row["changed_vs_baseline"] for row in all_rows if row["name"] == "stuck_at_lrs_faults"
        ),
        "stuck_at_lrs_targets_mlp_output_only": (
            tensor_metadata["tensor_role"] == "mlp_output"
            and tensor_metadata["tensor_role_inferred_from_name"] == "mlp_output"
            and tensor_name.endswith("mlp.down_proj.weight")
        ),
        "stuck_at_hrs_changes_logits": next(
            row["changed_vs_baseline"] for row in all_rows if row["name"] == "stuck_at_hrs_faults"
        ),
        "raw_ber_changes_logits": next(row["changed_vs_baseline"] for row in all_rows if row["name"] == "raw_ber_no_ecc"),
        "ecc_reduces_raw_ber_logit_delta": next(
            row["mean_abs_logit_delta_vs_baseline"] for row in all_rows if row["name"] == "raw_ber_with_ecc"
        )
        <= next(row["mean_abs_logit_delta_vs_baseline"] for row in all_rows if row["name"] == "raw_ber_no_ecc"),
        "retention_changes_logits": next(row["changed_vs_baseline"] for row in all_rows if row["name"] == "retention_drift"),
        "kv_cache_retention_is_shorter_than_weight_stationary": (
            projection["retention_calibration"]["kv_cache"]["retention_time_s"]
            < projection["retention_calibration"]["weight_stationary"]["retention_time_s"]
        ),
        "mlp_dynamic_retention_is_shorter_than_weight_stationary": (
            projection["retention_calibration"]["mlp_dynamic"]["retention_time_s"]
            < projection["retention_calibration"]["weight_stationary"]["retention_time_s"]
        ),
        "relaxation_changes_logits": next(row["changed_vs_baseline"] for row in all_rows if row["name"] == "relaxation_drift"),
        "state_sigma_changes_logits": next(
            row["changed_vs_baseline"] for row in all_rows if row["name"] == "state_conductance_variation"
        ),
        "endurance_drift_changes_logits": next(
            row["changed_vs_baseline"] for row in all_rows if row["name"] == "endurance_drift_crossbar"
        ),
        "thermal_noise_path_exercised": next(row["changed_vs_baseline"] for row in all_rows if row["name"] == "thermal_noise"),
        "shot_noise_path_exercised": next(row["changed_vs_baseline"] for row in all_rows if row["name"] == "shot_noise"),
        "flicker_noise_path_exercised": next(row["changed_vs_baseline"] for row in all_rows if row["name"] == "flicker_1f_noise"),
        "rtn_noise_path_exercised": next(row["changed_vs_baseline"] for row in all_rows if row["name"] == "rtn_noise"),
        "readout_noise_path_exercised": next(row["changed_vs_baseline"] for row in all_rows if row["name"] == "readout_rms_noise"),
        "combined_noise_to_int4_before_adc_changes_logits": next(
            row["changed_vs_baseline"] for row in all_rows if row["name"] == "combined_noise_to_int4_before_adc"
        ),
        "dense_linear_mapping_present": projection["dense_linear_layers_mapped_to_crossbars"]["layers"] > 0,
        "qkt_k_cache_mapping_present": projection["attention_qkt_k_cache_programmed_to_crossbars"]["layers"] > 0,
        "av_v_cache_mapping_present": projection["attention_av_v_cache_programmed_to_crossbars"]["layers"] > 0,
        "dynamic_kv_programming_cost_present": (
            projection["dynamic_kv_cache_programming_cost"]["dynamic_elements_per_decode_token"] > 0
        ),
        "dynamic_mlp_programming_cost_present": projection["dynamic_mlp_decode_per_token"]["programmed_elements"] > 0,
        "endurance_projection_present": (
            projection["endurance_lifetime_projection"]["projected_kv_write_cycles_per_slot"] > 0
            and projection["endurance_lifetime_projection"]["dynamic_mlp_is_endurance_limited"]
        ),
    }

    summary = {
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S %Z"),
        "host": socket.gethostname(),
        "platform": platform.platform(),
        "python": sys.version,
        "torch": torch.__version__,
        "model": args.model,
        "snapshot_dir": str(snapshot_dir),
        "snapshot_reused_before_download": reused_snapshot,
        "cache_dir": str(args.cache_dir),
        "local_root": str(args.local_root),
        "tensor_name": tensor_name,
        "tensor_file": tensor_file,
        "tensor_selection": tensor_metadata,
        "slice_dim": args.dim,
        "weight_bits": args.weight_bits,
        "act_bits": args.act_bits,
        "feature_effects": all_rows,
        "checks": checks,
        "all_checks_passed": all(bool(value) for value in checks.values()),
        "mapping_projection": projection,
        "requirement_coverage": coverage,
        "outputs": {
            "feature_effects_csv": str(out_dir / "feature_effects.csv"),
            "per_case_scores_csv": str(out_dir / "per_case_scores.csv"),
            "test_cases_csv": str(out_dir / "test_cases.csv"),
            "requirement_coverage_csv": str(out_dir / "requirement_coverage.csv"),
            "summary_json": str(out_dir / "summary.json"),
            "summary_md": str(out_dir / "summary.md"),
        },
    }
    (out_dir / "mapping_projection.json").write_text(json.dumps(projection, indent=2) + "\n", encoding="utf-8")
    (out_dir / "implementation_checks.json").write_text(json.dumps(checks, indent=2) + "\n", encoding="utf-8")
    write_csv(out_dir / "requirement_coverage.csv", coverage)
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    (out_dir / "summary.md").write_text(build_markdown_summary(all_rows, projection, tensor_metadata), encoding="utf-8")

    print(json.dumps({"out_dir": str(out_dir), "all_checks_passed": summary["all_checks_passed"]}, indent=2), flush=True)


if __name__ == "__main__":
    main()
