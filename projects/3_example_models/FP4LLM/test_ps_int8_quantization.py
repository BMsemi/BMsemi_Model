"""CPU validation of the PS bridge; no FPGA execution is claimed."""
import importlib.util
from pathlib import Path
import sys
import unittest

import numpy as np

from src.ps_int8_quantization import prepare_mvm, round_bf16, unpack_bf16


class PSQuantizationTests(unittest.TestCase):
    def test_bf16_words_and_ties(self):
        words = np.array([0x3f80, 0xbf00, 0, 1], dtype=np.uint16)
        expected = np.array([1, -.5, 0, 2**-133], dtype=np.float32)
        np.testing.assert_array_equal(unpack_bf16(words), expected)
        ties = np.array([0x3f808000, 0x3f818000], dtype=np.uint32).view(np.float32)
        np.testing.assert_array_equal(round_bf16(ties).view(np.uint32), [0x3f800000, 0x3f820000])

    def test_zero_and_per_row_scales(self):
        p = prepare_mvm(np.array([[0, 0], [1, -1]], np.float32),
                        np.array([[2, -2], [.5, -.5], [0, 0]], np.float32))
        np.testing.assert_array_equal(p.activations, [[0, 0], [127, -127]])
        np.testing.assert_allclose(p.activation_scales, [1, 1/127])
        np.testing.assert_allclose(p.weight_scales, [2/127, .5/127, 1])
        sums = p.activations.astype(np.int32) @ p.weights.astype(np.int32)
        np.testing.assert_array_equal(p.reconstruct(sums), [[0, 0, 0], [4, 1, 0]])

    def test_tiled_reduction_and_four_engine_partition(self):
        rng = np.random.default_rng(6)
        p = prepare_mvm(rng.normal(size=(35, 65)).astype(np.float32),
                        rng.normal(size=(145, 65)).astype(np.float32))
        summed = np.zeros((35, 145), np.int64)
        seen = set()
        for tile in p.tiles():
            t, c, k = tile.token_start, tile.channel_start, tile.output_start
            n, width = tile.activations.shape[0], tile.weights.shape[1]
            self.assertLessEqual(max(*tile.activations.shape, *tile.weights.shape), 32)
            self.assertEqual(tile.engine_id, (k//32) % 4)
            seen.add(tile.engine_id)
            # Test-only stand-in for hardware results, never part of the bridge.
            summed[t:t+n, k:k+width] += tile.activations.astype(np.int64) @ tile.weights.astype(np.int64)
        self.assertEqual(seen, {0, 1, 2, 3})
        direct = p.activations.astype(np.int64) @ p.weights.astype(np.int64)
        np.testing.assert_array_equal(summed, direct)
        np.testing.assert_array_equal(p.reconstruct(summed), p.reconstruct(direct))

    def test_existing_fpga_dma_packer(self):
        driver = Path(__file__).resolve().parents[2] / 'TCAM_Search_FPGA/zcu106_tcam_rag_flow/xforge_zcu106/x2_zcu106/runtime/axi_x2_mvm_driver.py'
        spec = importlib.util.spec_from_file_location('_x2_driver_ps_test', driver)
        mod = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = mod
        spec.loader.exec_module(mod)
        p = prepare_mvm(np.array([[1, -1, 0], [.5, -.5, .25]], np.float32),
                        np.array([[1, 2, 3], [-1, 0, 1]], np.float32))
        tile = next(p.tiles())
        packed = mod.pack_mvm_batch_stream(tile.activations, tile.weights)
        raw = packed.astype('<u4').tobytes()
        self.assertEqual(raw[:6], tile.weights.tobytes())
        self.assertEqual(raw[6:8], b'\0\0')
        self.assertEqual(raw[8:11], tile.activations[0].tobytes())
        self.assertEqual(raw[12:15], tile.activations[1].tobytes())
        self.assertEqual(raw[11], 0)
        self.assertEqual(raw[15], 0)

    def test_invalid_operands(self):
        with self.assertRaises(TypeError):
            prepare_mvm(np.ones((1, 2), np.uint16), np.ones((1, 2), np.float32))
        with self.assertRaises(ValueError):
            prepare_mvm(np.array([[np.nan]], np.float32), np.ones((1, 1), np.float32))
        with self.assertRaises(ValueError):
            prepare_mvm(np.ones((1, 3), np.float32), np.ones((2, 4), np.float32))


if __name__ == '__main__':
    unittest.main()
