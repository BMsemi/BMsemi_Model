import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from typing import Any
import unittest
import weakref
from unittest.mock import patch

import numpy as np
import torch
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from transformers import PreTrainedTokenizerFast, Qwen3Config, Qwen3ForCausalLM

import main as entrypoint
from llm_reram_executor import ReRAMLinearExecutor
from src import qwen_crossbar_runtime as runner


ROOT = Path(__file__).resolve().parent


class PromptCrossbarTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)

    def fixture(self, blocks):
        checkpoint = self.root / "model"
        torch.manual_seed(3)
        config = Qwen3Config(
            hidden_size=16, intermediate_size=32, num_hidden_layers=blocks,
            num_attention_heads=2, num_key_value_heads=1, head_dim=8,
            vocab_size=32, bos_token_id=1, eos_token_id=2,
        )
        model = Qwen3ForCausalLM(config).eval()
        model.save_pretrained(checkpoint)
        tokenizer = Tokenizer(WordLevel({"<unk>": 0, "one": 1, "two": 2, "three": 3, "next": 4}, unk_token="<unk>"))
        tokenizer.pre_tokenizer = Whitespace()
        PreTrainedTokenizerFast(tokenizer_object=tokenizer, unk_token="<unk>").save_pretrained(checkpoint)
        roles = [
            ("Q_Proj", 16, 16), ("K_Proj", 16, 8), ("V_Proj", 16, 8),
            ("Attn_Scores_QKt", 8, 8), ("Attn_AV", 8, 8),
            ("Out_Proj", 16, 16), ("FFN_1", 16, 32), ("FFN_2", 32, 16),
        ]
        for phase in ("prefill", "decode"):
            entries = []
            for block in range(blocks):
                for offset, (name, channels, outputs) in enumerate(roles):
                    tokens = 8 if phase == "prefill" or name.startswith("Attn_") else 1
                    entries.append(
                        f"Layer L{block * 8 + offset + 1:03d}_{name} {{\n"
                        "Type: CONV\nStride { X:1, Y:1 }\n"
                        f"Dimensions {{ K:{outputs}, C:{channels}, R:1, S:1, Y:{tokens}, X:1 }}\n"
                        "Dataflow { SpatialMap(1,1) K; TemporalMap(16,16) C; SpatialMap(1,1) Y; }\n}"
                    )
            (self.root / f"{phase}.m").write_text("Network Tiny {\n" + "\n".join(entries) + "\n}\n", encoding="utf-8")
        return checkpoint

    def run_main(self, blocks, *, single=None, audits=False, execution="block_streamed"):
        checkpoint = self.fixture(blocks)
        output = self.root / "results"
        command = [
            sys.executable, "-B", str(ROOT / "main.py"), "--model", str(checkpoint),
            "--prompt", "one two three", "--decode-text", " next", "--max-candidates", "1",
            "--prefill", str(self.root / "prefill.m"), "--decode", str(self.root / "decode.m"),
            "--out-dir", str(output), "--run-name", "tiny",
            "--crossbar-execution", execution,
        ]
        if single is not None:
            command.extend(["--block-index", str(single)])
        if not audits:
            command.extend(["--disable-saturation-audit", "--disable-hardware-effects-audit"])
        result = subprocess.run(
            command, cwd=ROOT, capture_output=True, text=True, timeout=120,
            env=dict(os.environ, PYTHONDONTWRITEBYTECODE="1"),
        )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        summary = json.loads((output / "main_summary.json").read_text(encoding="utf-8"))
        metadata = json.loads((output / "software_crossbar" / "metadata.json").read_text(encoding="utf-8"))
        return summary, metadata

    def test_main_runs_all_28_blocks(self):
        summary, metadata = self.run_main(28)
        self.assertEqual(metadata["crossbar_backend"], "wrapper")
        self.assertEqual(metadata["crossbar_execution"], "block_streamed")
        self.assertEqual(metadata["crossbar_tile_lifecycle"], {
            "resident_blocks": 0, "peak_resident_blocks": 1,
            "block_program_events": 56, "block_release_events": 56,
        })
        self.assertEqual(summary["software_crossbar_run"]["crossbar_execution"], "block_streamed")
        self.assertEqual(summary["software_crossbar_run"]["crossbar_implementation"], "reram_crossbar.py:CrossbarArray.compute")
        self.assertTrue(all(row["crossbar_backend"] == "wrapper" for row in metadata["linear_crossbar_modules"]))
        self.assertEqual(metadata["executed_block_indices"], list(range(28)))
        self.assertEqual(len(metadata["linear_crossbar_modules"]), 196)
        records = metadata["mapped_attention_gemm_records"]
        self.assertEqual(len(records), 112)
        self.assertTrue(all(row["crossbar_backend"] == "wrapper" for row in records))
        for index in range(28):
            actual = {(row["phase"], row["mapped_role"]) for row in records if row["block_index"] == index}
            self.assertEqual(actual, {(phase, role) for phase in ("prefill", "decode") for role in ("attn_scores_qkt", "attn_av")})
        self.assertEqual(metadata["decode_input_ids"], [[4]])
        self.assertEqual(summary["software_crossbar_run"]["executed_block_count"], 28)
        self.assertEqual(summary["software_crossbar_run"]["scope"], "all transformer blocks")
        self.assertIsInstance(metadata["prefill_next_token_id"], int)
        self.assertNotIn("selected_block_prefill_next_token_id", metadata)
        values = (self.root / "results" / "software_crossbar" / "tiny_bf16_values.txt").read_text(encoding="utf-8")
        self.assertIn("layer_27_block_output", values)
        self.assertIn("layer_00_block_output", values)
        self.assertEqual(metadata["endurance_summary"]["dynamic_kv_elements_observed"], 4 * 8 * 2 * 28)

    def test_all_blocks_audits(self):
        _, metadata = self.run_main(2, audits=True)
        _, eager = self.run_main(2, audits=True, execution="eager")
        for field in ("saturation_records", "hardware_effect_records"):
            self.assertEqual({row["block_index"] for row in metadata[field]}, {0, 1})
            self.assertEqual(len(metadata[field]), 36)
            self.assertEqual(metadata[field], eager[field])

    def test_block_streaming_matches_eager_prefill_and_decode(self):
        checkpoint = self.fixture(3)
        eager_model = runner.load_model(str(checkpoint), dtype=torch.bfloat16).eval()
        streamed_model = runner.load_model(str(checkpoint), dtype=torch.bfloat16).eval()
        phase_ref = {"phase": "prefill"}
        options: dict[str, Any] = dict(
            prefill_path=self.root / "prefill.m", decode_path=self.root / "decode.m",
            block_index=None, phase_ref=phase_ref, backend="wrapper",
            weight_bits=7, act_bits=7, block_size=16, float_format="bf16",
        )
        runner.prepare_crossbar_blocks(eager_model, **options)
        blocks = runner.prepare_crossbar_blocks(streamed_model, crossbar_execution="block_streamed", **options)
        executors = [
            [runner.get_replaced_linear(block["layer"], name)._executor for name in runner.LINEAR_ROLE_BY_MODULE]
            for block in blocks
        ]
        tile_references = []
        observed_blocks = []

        def check_released(module, inputs, output=None):
            self.assertTrue(all(not executor._programmed for group in executors for executor in group))
            self.assertTrue(all(reference() is None for reference in tile_references))

        check_released(None, None)
        for block in blocks:
            index = block["index"]

            def check_active(module, inputs, index=index):
                for other_index, group in enumerate(executors):
                    self.assertTrue(all(bool(executor._programmed) == (index == other_index) for executor in group))
                tile_references.extend(
                    weakref.ref(tile) for executor in executors[index]
                    for row in executor._programmed for tile, _ in row
                )
                observed_blocks.append(index)

            handles = [
                block["layer"].register_forward_pre_hook(check_released, prepend=True),
                block["layer"].register_forward_pre_hook(check_active),
                block["layer"].register_forward_hook(check_released),
            ]
            for handle in handles + block["lifecycle_hooks"]:
                self.addCleanup(handle.remove)

        eager_cache = streamed_cache = None
        for step, token_ids in enumerate(([1, 2, 3], [4], [2])):
            phase_ref["phase"] = "prefill" if step == 0 else "decode"
            inputs = torch.tensor([token_ids])
            with torch.inference_mode():
                eager_output = eager_model.model(input_ids=inputs, past_key_values=eager_cache, use_cache=True)
                streamed_output = streamed_model.model(input_ids=inputs, past_key_values=streamed_cache, use_cache=True)
                torch.testing.assert_close(streamed_output.last_hidden_state, eager_output.last_hidden_state, rtol=0, atol=0)
                torch.testing.assert_close(
                    streamed_model.lm_head(streamed_output.last_hidden_state.to(streamed_model.lm_head.weight.dtype)),
                    eager_model.lm_head(eager_output.last_hidden_state.to(eager_model.lm_head.weight.dtype)), rtol=0, atol=0,
                )
            eager_cache = eager_output.past_key_values
            streamed_cache = streamed_output.past_key_values
            for index in range(3):
                self.assertEqual(streamed_cache.get_seq_length(index), 3 + step)
                torch.testing.assert_close(streamed_cache.layers[index].keys, eager_cache.layers[index].keys, rtol=0, atol=0)
                torch.testing.assert_close(streamed_cache.layers[index].values, eager_cache.layers[index].values, rtol=0, atol=0)
        self.assertEqual(observed_blocks, [0, 1, 2] * 3)
        self.assertEqual(blocks[0]["tile_lifecycle"], {
            "resident_blocks": 0, "peak_resident_blocks": 1,
            "block_program_events": 9, "block_release_events": 9,
        })

    def test_block_streaming_releases_tiles_on_failure(self):
        checkpoint = self.fixture(1)
        model = runner.load_model(str(checkpoint), dtype=torch.bfloat16).eval()
        blocks = runner.prepare_crossbar_blocks(
            model, prefill_path=self.root / "prefill.m", decode_path=self.root / "decode.m",
            block_index=None, phase_ref={"phase": "prefill"}, backend="wrapper",
            weight_bits=7, act_bits=7, block_size=16, float_format="bf16",
            crossbar_execution="block_streamed",
        )
        layer = blocks[0]["layer"]
        executors = [runner.get_replaced_linear(layer, name)._executor for name in runner.LINEAR_ROLE_BY_MODULE]
        for handle in blocks[0]["lifecycle_hooks"]:
            self.addCleanup(handle.remove)
        for target, method in ((executors[1], "_program_tiles"), (layer.mlp, "forward")):
            with self.subTest(method=method):
                with patch.object(target, method, side_effect=RuntimeError("injected failure")):
                    with self.assertRaisesRegex(RuntimeError, "injected failure"), torch.inference_mode():
                        model.model(input_ids=torch.tensor([[1, 2]]), use_cache=False)
                self.assertTrue(all(not executor._programmed for executor in executors))
                self.assertEqual(blocks[0]["tile_lifecycle"]["resident_blocks"], 0)
        with torch.inference_mode():
            output = model.model(input_ids=torch.tensor([[1, 2]]), use_cache=True)
        self.assertEqual(output.past_key_values.get_seq_length(), 2)
        self.assertTrue(all(not executor._programmed for executor in executors))

    def test_executor_tile_release_preserves_results(self):
        generator = np.random.default_rng(7)
        weights = generator.integers(-3, 4, size=(65, 33)).astype(np.float32)
        inputs = generator.integers(-3, 4, size=(3, 65)).astype(np.float32)
        options: dict[str, Any] = dict(
            noise_config={"enabled": False}, reliability_config={"enabled": False},
            iv_config={"enabled": False}, tdc_config={"enabled": False},
        )
        eager = ReRAMLinearExecutor("test", weights, **options)
        streamed = ReRAMLinearExecutor("test", weights, defer_programming=True, **options)
        self.assertEqual(streamed._programmed, [])
        expected = eager(inputs)
        for _ in range(2):
            np.testing.assert_array_equal(streamed(inputs), expected)
            tiles = [weakref.ref(tile) for row in streamed._programmed for tile, _ in row]
            streamed.release_tiles()
            self.assertEqual(streamed._programmed, [])
            self.assertTrue(all(reference() is None for reference in tiles))
        with self.assertRaisesRegex(ValueError, "streaming requires"):
            ReRAMLinearExecutor("unsafe", weights, defer_programming=True)

    def test_projection_calls_reram_compute(self):
        generator = torch.Generator().manual_seed(7)
        inputs = torch.randint(-3, 4, (3, 65), generator=generator).float()
        linear = torch.nn.Linear(65, 33, bias=False)
        with torch.no_grad():
            linear.weight.copy_(torch.randint(-3, 4, (33, 65), generator=generator).float())
        wrapper = runner.CrossbarLinear.from_linear(
            linear, weight_bits=8, act_bits=7, block_size=16,
            noise_config={"enabled": False}, reliability_config={"enabled": False},
            iv_config={"enabled": False}, tdc_config={"enabled": False},
        )
        expected = runner.IdealIntCrossbarLinear(
            linear, weight_bits=8, act_bits=7, block_size=16, float_format="bf16",
        )(inputs)
        with patch.object(runner.CrossbarArray, "compute", autospec=True, side_effect=runner.CrossbarArray.compute) as compute:
            actual = wrapper(inputs)
        self.assertEqual(compute.call_count, 3 * 2 * 2)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    def test_attention_calls_reram_compute(self):
        generator = torch.Generator().manual_seed(11)
        inputs = torch.randint(-3, 4, (1, 2, 3, 65), generator=generator).float()
        weights = torch.randint(-3, 4, (1, 2, 65, 33), generator=generator).float()
        options = dict(weight_bits=8, act_bits=7, block_size=16, float_format="bf16")
        expected = runner.crossbar_matmul(inputs, weights, backend="ideal_int", **options)
        with patch.object(runner.CrossbarArray, "compute", autospec=True, side_effect=runner.CrossbarArray.compute) as compute:
            actual = runner.crossbar_matmul(inputs, weights, backend="wrapper", **options)
        self.assertEqual(compute.call_count, 2 * 3 * 2 * 2)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    def test_single_block_override_and_cache_index(self):
        summary, metadata = self.run_main(2, single=1)
        self.assertEqual(metadata["executed_block_indices"], [1])
        self.assertEqual(len(metadata["linear_crossbar_modules"]), 7)
        self.assertEqual(len(metadata["mapped_attention_gemm_records"]), 4)
        self.assertEqual(metadata["block_index_zero_based"], 1)
        self.assertEqual(summary["software_crossbar_run"]["scope"], "selected transformer block")
        self.assertEqual(metadata["selected_block_decode_next_token_id"], metadata["decode_next_token_id"])

    def test_scope_forwarding(self):
        for extra, flag in (([], "--all-blocks"), (["--block-index", "1"], "--block-index")):
            args = entrypoint.build_parser().parse_args(["--prompt", "test", *extra])
            with patch.object(runner, "run") as launch, patch.object(Path, "exists", return_value=True):
                entrypoint.run_prompt_crossbar(args, {"prefill": Path("prefill.m"), "decode": Path("decode.m")}, self.root)
            runtime_args = launch.call_args.args[0]
            self.assertEqual(runtime_args.all_blocks, flag == "--all-blocks")
            self.assertEqual(runtime_args.block_index, 0 if flag == "--all-blocks" else 1)
            self.assertEqual(runtime_args.crossbar_backend, "wrapper")
            self.assertEqual(runtime_args.crossbar_execution, "block_streamed")
        args = entrypoint.build_parser().parse_args(["--prompt", "test", "--crossbar-execution", "eager"])
        with patch.object(runner, "run") as launch, patch.object(Path, "exists", return_value=True):
            entrypoint.run_prompt_crossbar(args, {"prefill": Path("prefill.m"), "decode": Path("decode.m")}, self.root)
        self.assertEqual(launch.call_args.args[0].crossbar_execution, "eager")

    def test_missing_block_mapping_rejected_before_conversion(self):
        checkpoint = self.fixture(1)
        model = Qwen3ForCausalLM(Qwen3Config(
            hidden_size=16, intermediate_size=32, num_hidden_layers=2,
            num_attention_heads=2, num_key_value_heads=1, head_dim=8, vocab_size=32,
        ))
        with self.assertRaisesRegex(ValueError, "cannot provide block 2"):
            runner.prepare_crossbar_blocks(
                model, prefill_path=self.root / "prefill.m", decode_path=self.root / "decode.m",
                block_index=None, phase_ref={"phase": "prefill"}, backend="ideal_int",
                weight_bits=7, act_bits=7, block_size=16, float_format="bf16",
            )
        self.assertTrue(all(isinstance(layer.self_attn.q_proj, torch.nn.Linear) for layer in model.model.layers))


if __name__ == "__main__":
    unittest.main()