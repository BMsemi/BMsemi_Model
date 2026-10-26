import copy
import json
import unittest
import tempfile
import argparse
from pathlib import Path

import torch
import numpy as np
from transformers import Qwen3Config, Qwen3ForCausalLM, PreTrainedTokenizerFast
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace

from qwen_frontend import Projection, VectorizedReadCrossbar, compare_logits, map_linears, install_attention, run
from reram_crossbar import CrossbarArray
from llm_reram_executor import ReRAMLinearExecutor
from src.crossbar_wrapper import CrossbarLinear


class FrontendTests(unittest.TestCase):
    def test_batched_tokens_preserve_scalar_executor_reads(self):
        linear = torch.nn.Linear(70, 40, bias=False)
        cfg = json.loads((Path(__file__).parent / "configs/qwen_crossbar_noise.json").read_text())
        cfg["noise_config"]["readout_noise_a_rms"] = 1e-7
        cfg["reliability_config"]["state_sigma_ln"] = 0.2
        scalar, batched = Projection(linear, cfg, 8), Projection(linear, cfg, 8)
        for _ in range(2):
            x = torch.randn(3, 70)
            expected = ReRAMLinearExecutor.__call__(scalar.crossbar, x.numpy())
            np.testing.assert_array_equal(expected, batched(x).numpy())

    def test_nearest_resistance_matches_original_at_boundaries(self):
        original = CrossbarArray(input_size=64, output_size=64)
        optimized = VectorizedReadCrossbar(input_size=64, output_size=64)
        grid = original._state_log_resistance
        boundaries = (grid[:-1] + grid[1:]) / 2
        values = np.concatenate((grid, boundaries, np.nextafter(boundaries, np.inf),
                                 np.nextafter(boundaries, -np.inf), np.linspace(grid.min()-1, grid.max()+1, 10000)))
        np.testing.assert_array_equal(original._log_resistance_to_levels(values), optimized._log_resistance_to_levels(values))

    def test_complete_local_comparison_artifacts(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            tokenizer = Tokenizer(WordLevel({"[UNK]": 0, "[EOS]": 1, "Hello": 2, "world": 3}, unk_token="[UNK]"))
            tokenizer.pre_tokenizer = Whitespace()
            PreTrainedTokenizerFast(tokenizer_object=tokenizer, unk_token="[UNK]", eos_token="[EOS]").save_pretrained(root)
            Qwen3ForCausalLM(Qwen3Config(vocab_size=32, hidden_size=16, intermediate_size=32,
                num_hidden_layers=1, num_attention_heads=2, num_key_value_heads=1, head_dim=8)).save_pretrained(root)
            arguments = argparse.Namespace(model=str(root), crossbar_config=Path(__file__).parent / "configs/qwen_crossbar_noise.json",
                local_files_only=True, out_dir=root / "results", reference_text="Hello world",
                prompt="Hello world", raw_prompt=True, plan="all:crossbar", seed=7, max_new_tokens=2)
            run(arguments)
            report = json.loads((root / "results/comparison.json").read_text())
            self.assertEqual(len(report["outputs"]["noisy"]["mapped_modules"]), 8)
            self.assertTrue((root / "results/noisy_logits.pt").exists())
            arguments.resume_clean = True
            run(arguments)
            repeated = json.loads((root / "results/comparison.json").read_text())
            self.assertEqual(report["metrics"], repeated["metrics"])
            arguments.seed += 1
            with self.assertRaisesRegex(ValueError, "seed changed"):
                run(arguments)

    def test_vectorized_reads_match_original_rng_and_arithmetic(self):
        rng = np.random.default_rng(123)
        weights = rng.integers(-128, 128, (64, 32), dtype=np.int16).astype(np.int8)
        for noise in (False, True):
            kwargs = dict(input_size=64, output_size=64,
                          noise_config={"enabled": noise, "readout_noise_a_rms": 1e-7},
                          reliability_config={"enabled": True, "raw_ber_ppm": 10000},
                          iv_config={"enabled": False}, tdc_config={"enabled": False},
                          noise_seed=9, reliability_seed=10)
            original, vector = CrossbarArray(**kwargs), VectorizedReadCrossbar(**kwargs)
            original.program_weights(weights)
            vector.program_weights(weights)
            for _ in range(3):
                x = rng.integers(0, 256, 64, dtype=np.uint8)
                x[:4] = [0, 16, 15, 255]
                np.testing.assert_array_equal(original.compute(x), vector.compute(x))

    def test_replay_matches_eager_clean_and_preserves_faults(self):
        torch.manual_seed(3)
        linear = torch.nn.Linear(17, 35, bias=False)
        cfg = json.loads((Path(__file__).parent / "configs/qwen_crossbar_noise.json").read_text())
        cfg["noise_config"]["enabled"] = False
        cfg["reliability_config"]["enabled"] = False
        eager = CrossbarLinear.from_linear(linear, seed=5, **cfg)
        replay = Projection(linear, cfg, 5)
        x = torch.randn(2, 17)
        self.assertTrue(torch.equal(eager(x), replay(x)))
        cfg["reliability_config"].update(enabled=True, raw_ber_ppm=10000)
        replay = Projection(linear, cfg, 5)
        self.assertTrue(torch.equal(replay(x), replay(x)))

    def test_actual_crossbar_forward_and_matched_baseline(self):
        torch.manual_seed(7)
        model = Qwen3ForCausalLM(Qwen3Config(
            vocab_size=32, hidden_size=16, intermediate_size=32,
            num_hidden_layers=1, num_attention_heads=2,
            num_key_value_heads=1, head_dim=8,
        )).eval()
        noisy = copy.deepcopy(model)
        cfg = json.loads((Path(__file__).parent / "configs/qwen_crossbar_noise.json").read_text())
        # Stress settings make propagation detectable in this tiny network;
        # the production JSON remains unchanged.
        cfg["reliability_config"]["raw_ber_ppm"] = 10000.0
        cfg["noise_config"]["readout_noise_a_rms"] = 1e-7
        clean_cfg = copy.deepcopy(cfg)
        clean_cfg["noise_config"]["enabled"] = False
        clean_cfg["reliability_config"]["enabled"] = False
        self.assertEqual(len(map_linears(model, clean_cfg, 12)), 8)
        map_linears(noisy, cfg, 12)
        install_attention(model, clean_cfg, 12)
        install_attention(noisy, cfg, 12)
        ids = torch.tensor([[1, 2, 3]])
        with torch.inference_mode():
            a, b = model(ids).logits, noisy(ids).logits
            repeat = model(ids).logits
            first = noisy(ids, use_cache=True)
            decoded = noisy(torch.tensor([[4]]), past_key_values=first.past_key_values, use_cache=True)
        self.assertEqual(tuple(decoded.logits.shape), (1, 1, 32))
        self.assertTrue(torch.equal(a, repeat))
        metrics = compare_logits(a, b, ids)
        self.assertGreater(metrics["logit_rmse"], 0)
        identity = compare_logits(a, a, ids)
        self.assertEqual(identity["logit_rmse"], 0)
        self.assertEqual(identity["perplexity_change_percent"], 0)


if __name__ == "__main__":
    unittest.main()
