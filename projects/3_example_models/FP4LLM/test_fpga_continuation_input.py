"""Prompt/sequence validation without requiring inference libraries or hardware."""
import copy
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

import numpy as np

import run_qwen_fpga_continuation as runner


ROOT = Path(__file__).resolve().parent


class ContinuationInputTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.package = Path(self.temp.name)
        self.summary = json.loads((ROOT / 'results/qwen_model_execution_on_x2/fpga_summary.json').read_text())

    def fixture(self, count=3):
        summary = copy.deepcopy(self.summary)
        summary['tokens'] = count
        source = summary['bundle_metadata']
        source.update(prompt='a different prompt', token_ids=list(range(100, 100+count)),
                      position_ids=list(range(count)))
        self.write_summary(summary)
        np.savez(self.package / 'fpga_block1_prefill.npz', output=np.zeros((count, 2048), np.float32))
        return summary

    def write_summary(self, summary):
        (self.package / 'fpga_summary.json').write_text(json.dumps(summary))

    def test_sequence_sizes_and_positions(self):
        for count in (1, 3, 32, 128):
            with self.subTest(count=count):
                self.fixture(count)
                path, source = runner.validate_source(self.package)
                self.assertEqual(runner.load_prefill_input(path, source).shape, (count, 2048))
                mask, positions = runner.prefill_layout(source)
                np.testing.assert_array_equal(mask, np.ones((1, count), dtype=np.int64))
                np.testing.assert_array_equal(positions, np.arange(count)[None, :])

    def test_existing_physical_hello_package(self):
        path, source = runner.validate_source(ROOT / 'results/qwen_model_execution_on_x2')
        self.assertEqual(source['prompt'], 'Hello')
        self.assertEqual(runner.load_prefill_input(path, source).shape, (1, 2048))
        self.assertEqual(runner.checkpoint_revision(source), '70d244cc86ccca08cf5af4e1e306ecf908b1ad5e')

    def test_invalid_metadata(self):
        changes = [
            lambda s: s.update(physical_hardware=False),
            lambda s: s.update(phase='decode'),
            lambda s: s.update(tokens=2),
            lambda s: s['bundle_metadata'].update(position_ids=[0, 0, 0]),
            lambda s: s['bundle_metadata'].update(token_ids=[-1, 2, 3]),
            lambda s: s['bundle_metadata'].update(token_ids=[True, 2, 3]),
            lambda s: s['bundle_metadata'].update(token_ids=[151936, 2, 3]),
            lambda s: s['bundle_metadata'].update(block_index=1),
        ]
        for change in changes:
            summary = self.fixture()
            change(summary)
            self.write_summary(summary)
            with self.assertRaises(ValueError):
                runner.validate_source(self.package)
        for count in (0, 129):
            self.fixture(count)
            with self.assertRaises(ValueError):
                runner.validate_source(self.package)

    def test_bad_arrays(self):
        self.fixture()
        path, source = runner.validate_source(self.package)
        for output in (np.zeros((1, 2048)), np.zeros((3, 1024)), np.full((3, 2048), np.nan)):
            np.savez(path, output=output)
            with self.assertRaises(ValueError):
                runner.load_prefill_input(path, source)

    def test_stale_hello_sidecar_rejected(self):
        self.fixture()
        (self.package / 'hello.source.json').write_text(json.dumps({'prompt': 'Hello'}))
        with self.assertRaisesRegex(ValueError, 'disagrees'):
            runner.validate_source(self.package)

    def test_tokenizer_and_config_mismatch(self):
        self.fixture()
        _, source = runner.validate_source(self.package)
        tokenizer = SimpleNamespace(encode=lambda prompt, add_special_tokens: source['token_ids'])
        config = SimpleNamespace(**source['config'])
        runner.validate_model_input(source, tokenizer, config)
        config.rope_theta = 1
        with self.assertRaisesRegex(ValueError, 'rope_theta'):
            runner.validate_model_input(source, tokenizer, config)
        config = SimpleNamespace(**source['config'])
        tokenizer.encode = lambda *args, **kwargs: [9707]
        with self.assertRaisesRegex(ValueError, 'tokenizer'):
            runner.validate_model_input(source, tokenizer, config)


if __name__ == '__main__':
    unittest.main()
