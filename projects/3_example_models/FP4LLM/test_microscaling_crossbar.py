"""Integration checks against the project's programmed cell / ADC model."""
import unittest
import numpy as np

from reram_crossbar import CrossbarArray
from src.microscaling import execute, quantize_inputs, quantize_weights
from src.microscaling_crossbar import MicroscalingCrossbar


class CrossbarMicroscalingTests(unittest.TestCase):
    def adapter(self, **kwargs):
        return MicroscalingCrossbar(noise_config={"enabled": False},
                                    iv_config={"enabled": False}, **kwargs)

    def test_31_terms_and_partial_output_tiles(self):
        rng = np.random.default_rng(19)
        w, x = rng.normal(size=(31, 19)), rng.normal(size=31)
        w[16:] *= 0.05
        qw, qx = quantize_weights(w), quantize_inputs(x)
        for bits in (1, 2, 3, 4, 5, 6):
            with self.subTest(dac_bits=bits):
                cb = self.adapter(dac_bits=bits, adc_bits=16)
                y, _ = execute(qw, qx, bits, read=cb)
                np.testing.assert_allclose(y, qx.reconstruct() @ qw.reconstruct(), atol=1e-12)
                report = cb.report()
                self.assertEqual(report['programmed_arrays'], 4)
                self.assertEqual(report['programmed_cells'], 4 * 32 * 32)
                self.assertEqual(report['adc_clip_events'], 0)
                for array, _ in cb.tiles.values():
                    self.assertEqual(array._w_pos1.size, 0)
                    self.assertEqual(array.get_info()['pwm_stored_weight_bits'], 32 * 32 * 4)
                # Repeated execution keeps programmed cells and does not accumulate prior outputs.
                y2, _ = execute(qw, qx, bits, read=cb)
                np.testing.assert_array_equal(y, y2)
                self.assertEqual(cb.report()['programmed_cells'], report['programmed_cells'])

    def test_adc_clips_per_bank_not_after_differential(self):
        cb = self.adapter(adc_bits=8)
        p = np.full((16, 1), 15, dtype=np.int64)
        n = np.zeros_like(p)
        cb.prepare_tile((0, 0), p, n)
        np.testing.assert_array_equal(cb(p, n, np.full(16, 3)), [255])
        self.assertEqual(cb.report()['adc_clip_events'], 1)

    def test_iv_transfer_is_used(self):
        cb = MicroscalingCrossbar(noise_config={"enabled": False},
              iv_config={"enabled": True, "adc_full_scale_levels": 100})
        p, n = np.full((16, 1), 15), np.zeros((16, 1), dtype=int)
        cb.prepare_tile((0, 0), p, n)
        np.testing.assert_array_equal(cb(p, n, np.full(16, 3)), [100])
        self.assertGreater(cb.report()['tiles'][0]['iv_saturation_stats']['upper_saturation_events'], 0)

    def test_persistent_cell_defects_are_used(self):
        cb = self.adapter(reliability_config={"enabled": True, "stuck_at_hrs_prob": 1.0})
        p, n = np.full((16, 1), 15), np.zeros((16, 1), dtype=int)
        cb.prepare_tile((0, 0), p, n)
        np.testing.assert_array_equal(cb(p, n, np.full(16, 3)), [0])
        self.assertEqual(cb.report()['tiles'][0]['reliability_stats']['stuck_at_hrs_cells'], 1024)

    def test_noise_is_seeded_and_changes_results(self):
        w = quantize_weights(np.arange(31 * 3).reshape(31, 3) / 20 - 2)
        x = quantize_inputs(np.linspace(-1, 1, 31))
        cfg = dict(noise_config={"enabled": True, "readout_noise_a_rms": 1e-6},
                   iv_config={"enabled": False}, adc_bits=16, seed=2)
        a, b = MicroscalingCrossbar(**cfg), MicroscalingCrossbar(**cfg)
        ya, _ = execute(w, x, read=a)
        yb, _ = execute(w, x, read=b)
        np.testing.assert_array_equal(ya, yb)
        self.assertFalse(np.allclose(ya, x.reconstruct() @ w.reconstruct()))

    def test_capacity_and_pulse_validation(self):
        cb = self.adapter()
        with self.assertRaises(ValueError):
            cb.prepare_tile((0, 0), np.ones((16, 17), dtype=int), np.zeros((16, 17), dtype=int))
        p, n = np.ones((16, 1), dtype=int), np.zeros((16, 1), dtype=int)
        cb.prepare_tile((0, 0), p, n)
        with self.assertRaises(ValueError):
            cb(p, n, np.full(16, 4))
        with self.assertRaises(ValueError):
            cb.prepare_tile((0, 0), p * 2, n)

    def test_int8_programming_still_works_after_int5(self):
        cb = CrossbarArray(input_size=2, output_size=4, noise_config={"enabled": False},
                          iv_config={"enabled": False})
        cb.program_int5_weights(np.array([[15, -15], [-1, 1]]))
        np.testing.assert_array_equal(cb.compute_int5_pulses(np.array([3, 2])), [43, -43])
        w = np.array([[-128, 127], [23, -44]])
        cb.program_weights(w)
        np.testing.assert_array_equal(cb.compute(np.array([255, 19]))[:2], np.array([255, 19]) @ w)
        self.assertEqual(cb.get_mode(), cb.MODE_PWM_INT8_ROW_SERIAL)


if __name__ == '__main__':
    unittest.main()
