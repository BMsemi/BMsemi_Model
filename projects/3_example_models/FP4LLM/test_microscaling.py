"""Arithmetic checks independent of the unavailable PWM implementation."""
import unittest
import numpy as np
from src.microscaling import (
    ScaledWeights, ScaledInputs, differential_cells, encode_weight_scale,
    execute, quantize_inputs, quantize_weights, requantize_output,
)


class MicroscalingTests(unittest.TestCase):
    def test_cycle_reconstruction_matches_dense_reference(self):
        rng = np.random.default_rng(7)
        w = rng.normal(size=(35, 19))
        w[16:32] *= 0.005
        x = rng.normal(size=35)
        x[16:32] *= 20
        qw, qx = quantize_weights(w), quantize_inputs(x)
        reference = qx.reconstruct() @ qw.reconstruct()
        for bits in (1, 2, 3, 4, 5, 6):
            with self.subTest(dac_bits=bits):
                y, trace = execute(qw, qx, bits)
                np.testing.assert_allclose(y, reference, rtol=1e-13, atol=1e-13)
                self.assertEqual(len(trace), 6)
                for block in trace:
                    i, j = block['input_block'] * 16, block['output_block'] * 16
                    expected = qx.values[i:i+16] @ qw.values[i:i+16, j:j+16]
                    np.testing.assert_array_equal(block['integer_partial'], expected)

    def test_different_scales_are_applied_before_reduction(self):
        qw = ScaledWeights(np.array([[10], [10]]), np.array([[0.5], [2.0]]),
                           np.array([[48], [64]], dtype=np.uint8), 0, 1)
        qx = ScaledInputs(np.array([10, 10]), np.array([1, 2], dtype=np.uint16), 0, 1)
        y, _ = execute(qw, qx)
        np.testing.assert_array_equal(y, [450.0])

    def test_differential_mapping(self):
        q = np.array([-15, -1, 0, 1, 15])
        p, n = differential_cells(q)
        np.testing.assert_array_equal(p - n, q)
        self.assertTrue(np.all((p >= 0) & (p <= 15) & (n >= 0) & (n <= 15)))

    def test_zero_blocks(self):
        qw, qx = quantize_weights(np.zeros((17, 3))), quantize_inputs(np.zeros(17))
        y, _ = execute(qw, qx)
        np.testing.assert_array_equal(y, np.zeros(3))
        np.testing.assert_array_equal(qw.scale_codes, 0)
        np.testing.assert_array_equal(qx.scale_codes, 0)

    def test_e4m3_boundaries_and_ties(self):
        self.assertEqual(encode_weight_scale(448), (126, 448.0))
        self.assertEqual(encode_weight_scale(1), (56, 1.0))
        self.assertEqual(encode_weight_scale(1.0625), (56, 1.0))
        self.assertEqual(encode_weight_scale(1.1875), (58, 1.25))
        self.assertEqual(encode_weight_scale(1e-12), (1, 2 ** -9))

    def test_signed_endpoints(self):
        x = np.array([-31., 31.])
        qx = quantize_inputs(x, fractional_bits=8)
        np.testing.assert_array_equal(qx.values, [-31, 31])
        qw = quantize_weights(np.array([[-15., 15.], [15., -15.]]))
        y, _ = execute(qw, qx)
        np.testing.assert_allclose(y, qx.reconstruct() @ qw.reconstruct())

    def test_output_rounding_and_saturation(self):
        codes, values, count = requantize_output(np.array([-40000., -1.5, 2.5, 40000.]), 0)
        np.testing.assert_array_equal(codes, [-32768, -2, 2, 32767])
        np.testing.assert_array_equal(values, codes)
        self.assertEqual(count, 2)

    def test_reject_invalid_inputs(self):
        for x in ([], [np.nan], [np.inf]):
            with self.assertRaises(ValueError):
                quantize_inputs(x)
        with self.assertRaises(ValueError):
            quantize_inputs([1e6], fractional_bits=16)
        with self.assertRaises(ValueError):
            quantize_weights([[1]], block_size=0)
        with self.assertRaises(ValueError):
            execute(quantize_weights([[1]]), quantize_inputs([1, 2]))
        with self.assertRaises(ValueError):
            execute(quantize_weights([[1]]), quantize_inputs([1]), dac_bits=0)
        with self.assertRaises(ValueError):
            execute(quantize_weights([[1]]), quantize_inputs([1]), read=lambda *args: [np.nan])


if __name__ == '__main__':
    unittest.main()
