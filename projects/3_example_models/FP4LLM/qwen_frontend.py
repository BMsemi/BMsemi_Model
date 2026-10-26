"""Matched Qwen inference through FP4LLM's actual ReRAM linear executor.

all:crossbar maps every nn.Linear (including lm_head) and attention QK/AV.
Normalization, embeddings and nonlinearities remain framework operations.
Reconstructs tiles deterministically: fixed device faults, fresh noise per read.
"""
from __future__ import annotations

import argparse
import copy
import gc
import json
import time
import types
from pathlib import Path

import numpy as np
import torch
from torch import nn
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.models.qwen3.modeling_qwen3 import apply_rotary_pos_emb, repeat_kv

from src.crossbar_wrapper import weight_block16
from llm_reram_executor import ReRAMLinearExecutor, derive_seed, round_float_format, quantize_symmetric, quantize_block16
from reram_crossbar import CrossbarArray, CrossbarNoiseConfig, CrossbarReliabilityConfig


class VectorizedReadCrossbar(CrossbarArray):
    """Batch row operations while preserving the scalar simulator's RNG order."""
    _sigma_tables = {}

    def _log_resistance_to_levels(self, log_resistance):
        values = np.asarray(log_resistance)
        grid = self._state_log_resistance
        if grid[0] == grid[-1] or not np.isfinite(values).all():
            return super()._log_resistance_to_levels(values)
        decreasing = grid[0] > grid[-1]
        position = np.searchsorted(-grid if decreasing else grid, -values if decreasing else values)
        left = np.clip(position - 1, 0, len(grid) - 1)
        right = np.clip(position, 0, len(grid) - 1)
        # Compare the two neighbors explicitly to retain argmin's tie behavior.
        return np.where(np.abs(values - grid[left]) <= np.abs(values - grid[right]), left, right).astype(np.uint8)

    def _read_noise_level_sigma(self, levels):
        key = (self.noise_config, self._max_cell_level(), tuple(self._state_log_resistance))
        table = self._sigma_tables.get(key)
        if table is None:
            table = super()._read_noise_level_sigma(np.arange(self._max_cell_level() + 1))
            self._sigma_tables[key] = table
        return table[np.asarray(levels, dtype=np.int16)]

    def _compute_pwm_row_serial_int8(self, input_vector):
        if self.iv_config.enabled or self.tdc_config.enabled:
            return super()._compute_pwm_row_serial_int8(input_vector)
        a = self._validate_pwm_input_vector(np.asarray(input_vector))
        a_u = a.astype(np.int32) & 255
        low, high = a_u & 15, a_u >> 4
        self._ecc_compute_latency_cycles += self._ecc_latency_cycles_for_bits(
            self._pwm_ecc_read_bits_for_activation(low, high),
            self.reliability_config.ecc_decode_latency_cycles,
        )
        levels = np.stack((self._w_pos0, self._w_neg0, self._w_pos1, self._w_neg1,
                           self._w_pos0, self._w_neg0, self._w_pos1, self._w_neg1), axis=1)
        counts = np.stack((low, low, low, low, high, high, high, high), axis=1)
        products = (counts[..., None] * levels).astype(np.float64)
        active = counts != 0
        if self.noise_config.enabled:
            sigma = self._read_noise_level_sigma(levels[active]) * np.sqrt(counts[active, None])
            products[active] += self._noise_rng.normal(0.0, sigma, size=sigma.shape)
        products = self._adc_u(products)
        p00 = products[:, 0] - products[:, 1]
        p01 = products[:, 2] - products[:, 3]
        p10 = products[:, 4] - products[:, 5]
        p11 = products[:, 6] - products[:, 7]
        if self.slice_gain0 != 1.0:
            p00 = np.rint(p00 * self.slice_gain0).astype(np.int32)
            p10 = np.rint(p10 * self.slice_gain0).astype(np.int32)
        if self.slice_gain1 != 1.0:
            p01 = np.rint(p01 * self.slice_gain1).astype(np.int32)
            p11 = np.rint(p11 * self.slice_gain1).astype(np.int32)
        out = np.zeros(self.output_size, dtype=np.int32)
        out[:self.output_size // 2] = (p00 + (p01 << 4) + (p10 << 4) + (p11 << 8)).sum(axis=0, dtype=np.int32)
        return out


class TileReads:
    """Re-iterable tile view: reconstruct fixed faults, resample read noise."""
    def __init__(self, executor):
        self.executor = executor
        self.read_index = 0

    def __iter__(self):
        read_index = self.read_index
        self.read_index += 1
        for row in range(self.executor.row_tiles):
            yield self.columns(row, read_index)

    def columns(self, row, read_index):
        e = self.executor
        rs = row * e.row_tile
        for col in range(e.col_tiles):
            cs = col * e.col_tile
            values = e.weight_q[rs:rs + e.row_tile, cs:cs + e.col_tile]
            tile = np.zeros((e.row_tile, e.col_tile), dtype=np.int8)
            tile[:values.shape[0], :values.shape[1]] = values
            with e._quiet():
                crossbar = VectorizedReadCrossbar(
                    input_size=e.row_tile, output_size=e.col_tile * 2,
                    noise_config=e.noise_config, reliability_config=e.reliability_config,
                    iv_config=e.iv_config, tdc_config=e.tdc_config,
                    noise_seed=derive_seed(e.noise_seed, e.name, row, col, read_index),
                    reliability_seed=derive_seed(e.reliability_seed, e.name, row, col),
                )
                crossbar.program_weights(tile)
            yield crossbar, None


class ReplayExecutor(ReRAMLinearExecutor):
    def __init__(self, name, weight_float, **kwargs):
        # Independent columns can be evaluated together: no cross-column
        # coupling exists for read noise or cell faults. Keep the original
        # tile width when timing, saturation or ECC models are requested.
        coupled = any(kwargs.get(key, {}).get("enabled", False)
                      for key in ("iv_config", "tdc_config"))
        coupled = coupled or kwargs.get("reliability_config", {}).get("use_ecc", False)
        self.col_tile = 32 if coupled else min(1024, ((weight_float.shape[1] + 31) // 32) * 32)
        super().__init__(name, weight_float, **kwargs)

    def program_tiles(self):
        if not self._programmed:
            self._programmed = TileReads(self)

    def __call__(self, x_float):
        if self.iv_config.enabled or self.tdc_config.enabled:
            return super().__call__(x_float)
        original_shape = x_float.shape[:-1]
        x = round_float_format(x_float.reshape(-1, x_float.shape[-1]), self.float_format)
        x = quantize_block16(x, bits=self.act_bits, block_size=self.act_block_size)
        codes, scale = quantize_symmetric(x)
        if not self.noise_config.enabled and not self.reliability_config.enabled:
            # Exact sum of the clean tile products; int64 prevents accumulation overflow.
            integers = np.empty((codes.shape[0], self.out_dim), dtype=np.int64)
            codes = codes.astype(np.int64)
            for start in range(0, self.out_dim, 256):
                integers[:, start:start + 256] = codes @ self.weight_q[:, start:start + 256].astype(np.int64)
        else:
            view = self._programmed
            first_read = view.read_index
            view.read_index += len(codes)
            integers = np.zeros((len(codes), self.out_dim), dtype=np.int64)
            for row in range(self.row_tiles):
                rs = row * self.row_tile
                inputs = np.full((len(codes), self.row_tile), 128, dtype=np.uint8)
                width = min(self.row_tile, self.in_dim - rs)
                inputs[:, :width] = (codes[:, rs:rs + width].astype(np.int16) + 128).astype(np.uint8)
                for col, (crossbar, _) in enumerate(view.columns(row, first_read)):
                    cs = col * self.col_tile
                    width = min(self.col_tile, self.out_dim - cs)
                    offset = crossbar.pwm_zero_point_output(128)[:width].astype(np.int64)
                    for token, activation in enumerate(inputs):
                        if token:
                            crossbar._noise_rng = np.random.default_rng(derive_seed(
                                self.noise_seed, self.name, row, col, first_read + token))
                        raw = crossbar.compute(activation)[:width].astype(np.int64)
                        integers[token, cs:cs + width] += raw - offset
        output = (integers.astype(np.float32) * scale * self.weight_scale).reshape(*original_shape, self.out_dim)
        return round_float_format(output, self.float_format)


class Projection(nn.Module):
    def __init__(self, source, config, seed):
        super().__init__()
        settings = copy.deepcopy(config)
        bits = settings.pop("weight_bits", 7)
        block_size = settings.pop("block_size", 16)
        settings["act_block_size"] = block_size
        settings.setdefault("noise_seed", seed + 1_000_003)
        settings.setdefault("reliability_seed", seed + 2_000_003)
        weight = round_float_format(source.weight.detach().float().cpu().numpy().T,
                                    settings.get("float_format", "bf16"))
        if bits < 8:
            weight = weight_block16(weight, bits=bits, block_size=block_size)
        self.crossbar = ReplayExecutor("crossbar_linear", weight, **settings)
        self.label = "projection"
        self.register_buffer("bias", None if source.bias is None else source.bias.detach().clone())

    def forward(self, x):
        started = time.monotonic()
        y = torch.from_numpy(self.crossbar(x.detach().float().cpu().numpy())).to(device=x.device, dtype=x.dtype)
        if self.label:
            print(f"Computed {self.label}: {tuple(x.shape)}, {time.monotonic() - started:.1f}s", flush=True)
        return y if self.bias is None else y + self.bias


def map_linears(model, config, seed):
    names = []
    # Snapshot before replacement so new crossbars are not traversed.
    linear_names = [name for name, module in model.named_modules() if isinstance(module, nn.Linear)]
    for name in linear_names:
        module = model.get_submodule(name)
        parent_name, _, leaf = name.rpartition(".")
        parent = model.get_submodule(parent_name) if parent_name else model
        print(f"Programming {name}", flush=True)
        projection = Projection(module, config, seed + len(names))
        projection.label = name
        setattr(parent, leaf, projection)
        names.append(name)
    if not names:
        raise ValueError("No linear modules found")
    return names


def install_attention(model, config, seed):
    """Route both dynamic attention GEMMs through the same device simulator."""
    for layer_index, layer in enumerate(model.model.layers):
        attn = layer.self_attn
        attn.crossbar_seed = seed + 10_000 + layer_index
        attn.crossbar_reads = 0

        def forward(self, hidden_states, position_embeddings, attention_mask=None,
                    past_key_values=None, **kwargs):
            shape = hidden_states.shape[:-1]
            shaped = (*shape, -1, self.head_dim)
            query = self.q_norm(self.q_proj(hidden_states).view(shaped)).transpose(1, 2)
            key = self.k_norm(self.k_proj(hidden_states).view(shaped)).transpose(1, 2)
            value = self.v_proj(hidden_states).view(shaped).transpose(1, 2)
            query, key = apply_rotary_pos_emb(query, key, *position_embeddings)
            cache = past_key_values if past_key_values is not None else kwargs.get("past_key_value")
            if cache is not None:
                key, value = cache.update(key, value, self.layer_idx)
            key = repeat_kv(key, self.num_key_value_groups)
            value = repeat_kv(value, self.num_key_value_groups)

            def multiply(a, b, role):
                result = []
                for batch in range(a.shape[0]):
                    heads = []
                    for head in range(a.shape[1]):
                        settings = copy.deepcopy(config)
                        settings["noise_seed"] = derive_seed(self.crossbar_seed, role, batch, head, self.crossbar_reads)
                        settings["reliability_seed"] = derive_seed(self.crossbar_seed, role, batch, head)
                        source = types.SimpleNamespace(weight=b[batch, head].T, bias=None)
                        projection = Projection(source, settings, self.crossbar_seed)
                        projection.label = None
                        heads.append(projection(a[batch, head]))
                    result.append(torch.stack(heads))
                return torch.stack(result)

            scores = multiply(query, key.transpose(-2, -1), "qk") * self.scaling
            if attention_mask is not None:
                scores = scores + attention_mask[:, :, :, :key.shape[-2]]
            probabilities = scores.softmax(dim=-1, dtype=torch.float32).to(query.dtype)
            output = multiply(probabilities, value, "av")
            self.crossbar_reads += 1
            output = output.transpose(1, 2).contiguous().reshape(*shape, -1)
            return self.o_proj(output), probabilities

        attn.forward = types.MethodType(forward, attn)


def compare_logits(clean, noisy, ids):
    clean, noisy = clean.float(), noisy.float()
    if clean.shape != noisy.shape or clean.shape[:2] != ids.shape:
        raise ValueError("Logits and reference token shapes must match")
    if not torch.isfinite(clean).all() or not torch.isfinite(noisy).all():
        raise ValueError("Non-finite logits")
    a, b = clean.log_softmax(-1), noisy.log_softmax(-1)
    target = ids[:, 1:, None]
    clean_nll = -a[:, :-1].gather(-1, target).mean()
    noisy_nll = -b[:, :-1].gather(-1, target).mean()
    delta = noisy - clean
    return {
        "logit_rmse": delta.square().mean().sqrt().item(),
        "logit_relative_l2": (delta.norm() / clean.norm().clamp_min(1e-12)).item(),
        "next_token_agreement": (clean[:, :-1].argmax(-1) == noisy[:, :-1].argmax(-1)).float().mean().item(),
        "mean_kl_clean_to_noisy": (a.exp() * (a - b)).sum(-1).mean().item(),
        "clean_reference_nll": clean_nll.item(),
        "noisy_reference_nll": noisy_nll.item(),
        "clean_reference_perplexity": clean_nll.exp().item(),
        "noisy_reference_perplexity": noisy_nll.exp().item(),
        "perplexity_change_percent": (100 * ((noisy_nll - clean_nll).exp() - 1)).item(),
    }


def run(args):
    torch.set_num_threads(getattr(args, "threads", 4))
    config = json.loads(args.crossbar_config.read_text())
    if not config.get("noise_config", {}).get("enabled") or not config.get("reliability_config", {}).get("enabled"):
        raise ValueError("Comparison config must enable both noise and reliability")
    config["noise_config"] = CrossbarNoiseConfig.from_obj(config["noise_config"]).as_dict()
    config["reliability_config"] = CrossbarReliabilityConfig.from_obj(config["reliability_config"]).as_dict()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=args.local_files_only)
    reference = tokenizer(args.reference_text, return_tensors="pt")["input_ids"]
    if reference.shape[1] < 2:
        raise ValueError("Reference text needs at least two tokens")
    prompt = tokenizer(args.prompt, return_tensors="pt")["input_ids"] if args.raw_prompt else tokenizer.apply_chat_template(
        [{"role": "user", "content": args.prompt}], tokenize=True,
        add_generation_prompt=True, enable_thinking=False, return_tensors="pt", return_dict=False,
    )
    outputs, logits = {}, {}
    run_settings = {
        "model": args.model, "plan": args.plan, "config": config,
        "seed": args.seed, "prompt": args.prompt, "raw_prompt": args.raw_prompt,
        "reference_text": args.reference_text, "max_new_tokens": args.max_new_tokens,
        "torch_version": torch.__version__,
        "column_batch_width": 1024, "physical_tile_shape": [64, 32],
    }
    settings_path = args.out_dir / "run_settings.json"
    modes = ("clean", "noisy")
    if getattr(args, "resume_clean", False):
        previous = json.loads(settings_path.read_text())
        for key in ("model", "plan", "config", "seed", "prompt", "raw_prompt", "reference_text", "max_new_tokens"):
            if previous[key] != run_settings[key]:
                raise ValueError(f"Cannot reuse clean run: {key} changed")
        outputs["clean"] = json.loads((args.out_dir / "clean.json").read_text())
        logits["clean"] = torch.load(args.out_dir / "clean_logits.pt", weights_only=True)
        modes = ("noisy",)
    settings_path.write_text(json.dumps(run_settings, indent=2), encoding="utf-8")
    for mode in modes:
        print(f"Starting {mode} run", flush=True)
        torch.manual_seed(args.seed)
        settings = copy.deepcopy(config)
        if mode == "clean":
            settings["noise_config"]["enabled"] = False
            settings["reliability_config"]["enabled"] = False
        model = AutoModelForCausalLM.from_pretrained(
            args.model, local_files_only=args.local_files_only,
            dtype=torch.bfloat16, attn_implementation="eager",
        ).eval()
        names = map_linears(model, settings, args.seed)
        install_attention(model, settings, args.seed)
        with torch.inference_mode():
            same_prompt = torch.equal(prompt, reference)
            reference_result = model(reference, use_cache=same_prompt)
            logits[mode] = reference_result.logits.cpu()
            tokens = prompt.clone()
            cache = None
            for step in range(args.max_new_tokens):
                result = reference_result if same_prompt and step == 0 else model(
                    tokens if cache is None else tokens[:, -1:], past_key_values=cache, use_cache=True)
                cache = result.past_key_values
                next_id = result.logits[:, -1].argmax(-1, keepdim=True)
                tokens = torch.cat((tokens, next_id), dim=1)
                if next_id.item() == tokenizer.eos_token_id:
                    break
        outputs[mode] = {"config": settings, "mapped_modules": names,
                         "attention_crossbar_layers": len(model.model.layers),
                         "generated_token_ids": tokens[0, prompt.shape[1]:].tolist(),
                         "text": tokenizer.decode(tokens[0, prompt.shape[1]:], skip_special_tokens=True)}
        (args.out_dir / f"{mode}.json").write_text(json.dumps(outputs[mode], indent=2), encoding="utf-8")
        torch.save(logits[mode], args.out_dir / f"{mode}_logits.pt")
        del model, result, reference_result, cache
        gc.collect()
    report = {"model": args.model, "plan": args.plan, "seed": args.seed,
              "scope": __doc__, "prompt": args.prompt, "reference_text": args.reference_text,
              "reference_token_ids": reference[0].tolist(),
              "metrics": compare_logits(logits["clean"], logits["noisy"], reference),
              "outputs": outputs,
              "interpretation": "Single reference-text sensitivity measurement, not a public benchmark. Clean disables both noise and reliability; quantization and all other settings are matched."}
    (args.out_dir / "comparison.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report["metrics"], indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="Qwen/Qwen3-1.7B")
    parser.add_argument("--plan", choices=["all:crossbar"], required=True)
    parser.add_argument("--crossbar-config", type=Path, required=True)
    parser.add_argument("--prompt", default="Explain how ReRAM crossbars perform matrix multiplication.")
    parser.add_argument("--raw-prompt", action="store_true", help="Use raw completion tokens instead of the chat template")
    parser.add_argument("--resume-clean", action="store_true", help="Reuse an existing clean result with matching run settings")
    parser.add_argument("--reference-text", default="The capital of France is Paris. Two plus two equals four.")
    parser.add_argument("--max-new-tokens", type=int, default=16)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--out-dir", type=Path, default=Path("results/qwen_noise_comparison"))
    args = parser.parse_args()
    if args.max_new_tokens < 1 or args.threads < 1:
        parser.error("--max-new-tokens and --threads must be positive")
    run(args)


if __name__ == "__main__":
    main()
