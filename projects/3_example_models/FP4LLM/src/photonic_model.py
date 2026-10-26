"""
Photonic MAC accumulator precision model.
Captures output distributions from linear layers and estimates
required ENOB, SNR, and accumulator bit-width.
"""
import torch
import torch.nn as nn
import numpy as np
from collections import defaultdict
from .fp4_utils import classify_layer, FP4_MAX


# ── Distribution capture hook ─────────────────────────────────────────────────

class LinearOutputCapture:
    """
    Hooks all nn.Linear layers to record output statistics.
    Call .remove() when done. Results in .stats dict.
    """
    def __init__(self, model, max_samples: int = 64):
        self.max_samples = max_samples
        self.stats = defaultdict(lambda: {
            'samples': [], 'layer_type': 'other', 'in_features': 0, 'out_features': 0
        })
        self.hooks = []

        for name, module in model.named_modules():
            if not isinstance(module, nn.Linear):
                continue
            self.stats[name]['layer_type'] = classify_layer(name)
            self.stats[name]['in_features'] = module.in_features
            self.stats[name]['out_features'] = module.out_features
            h = module.register_forward_hook(self._make_hook(name))
            self.hooks.append(h)

    def _make_hook(self, name):
        def hook(module, inputs, output):
            st = self.stats[name]
            if len(st['samples']) >= self.max_samples:
                return
            # Store flattened output values (detached, float32, CPU)
            vals = output.detach().float().flatten().cpu()
            st['samples'].append(vals)
        return hook

    def remove(self):
        for h in self.hooks:
            h.remove()
        self.hooks = []

    def compute_stats(self) -> dict:
        """
        Aggregate captured samples into per-layer precision requirements.
        Returns dict: {layer_name: {...stats...}}
        """
        results = {}
        for name, st in self.stats.items():
            if not st['samples']:
                continue
            vals = torch.cat(st['samples'])
            amax = vals.abs().max().item()
            amin_nonzero = vals[vals != 0].abs().min().item() if (vals != 0).any() else amax

            if amin_nonzero == 0 or amax == 0:
                continue

            dynamic_range = amax / amin_nonzero
            enob = float(np.log2(dynamic_range)) if dynamic_range > 1 else 1.0
            required_bits = int(np.ceil(np.log2(dynamic_range))) + 1   # +1 for sign

            # SNR: signal = amax, noise = amax / 2^bits
            snr_db = 6.02 * required_bits + 1.76

            # torch.quantile fails above ~16M elements — subsample first
            abs_vals = vals.abs()
            MAX_Q = 2_000_000
            if abs_vals.numel() > MAX_Q:
                idx = torch.randperm(abs_vals.numel(), device=abs_vals.device)[:MAX_Q]
                abs_sample = abs_vals[idx]
            else:
                abs_sample = abs_vals

            results[name] = {
                'layer_type':     st['layer_type'],
                'in_features':    st['in_features'],
                'out_features':   st['out_features'],
                'max_abs':        amax,
                'min_abs_nonzero': amin_nonzero,
                'dynamic_range':  dynamic_range,
                'enob':           enob,
                'required_bits':  required_bits,
                'snr_db_needed':  snr_db,
                'p01':            float(torch.quantile(abs_sample, 0.01)),
                'p99':            float(torch.quantile(abs_sample, 0.99)),
                'p999':           float(torch.quantile(abs_sample, 0.999)),
                'std':            float(vals.std()),
                'mean_abs':       float(abs_vals.mean()),
            }
        return results


# ── Precision-limiting hook (for layer sensitivity sweep) ────────────────────

class AccumulatorPrecisionLimiter:
    """
    Quantizes the *output* of targeted linear layers to N-bit fixed-point,
    simulating a photonic ADC with finite resolution.
    """
    def __init__(self, model, n_bits: int, target_layer: str = None):
        """
        n_bits: ADC bit-width (4-16).
        target_layer: if None, apply to all linear layers; else only that layer name.
        """
        self.n_bits = n_bits
        self.target_layer = target_layer
        self.hooks = []

        for name, module in model.named_modules():
            if not isinstance(module, nn.Linear):
                continue
            if target_layer is not None and name != target_layer:
                continue
            h = module.register_forward_hook(self._make_hook(n_bits))
            self.hooks.append(h)

    @staticmethod
    def _make_hook(n_bits):
        levels = 2 ** (n_bits - 1)   # half-range for signed

        def hook(module, inputs, output):
            y = output.float()
            amax = y.abs().max().clamp(min=1e-38)
            step = amax / levels
            y_q = (y / step).round().clamp(-levels, levels - 1) * step
            return y_q.to(output.dtype)
        return hook

    def remove(self):
        for h in self.hooks:
            h.remove()
        self.hooks = []


# ── Analytical accumulator bit-width estimate ─────────────────────────────────

def required_accumulator_bits(k: int,
                               w_max: float = FP4_MAX,
                               a_max: float = FP4_MAX,
                               smallest_product: float = 0.25) -> dict:
    """
    Analytical estimate of required accumulator bits for FP4 GEMM.

    k: number of elements in inner product (layer input dim)
    w_max, a_max: max FP4 values (both 6.0)
    smallest_product: minimum non-zero product of two FP4 values (0.5×0.5 = 0.25)

    Returns: dict with required bits and ENOB.
    """
    max_partial_sum = k * w_max * a_max       # worst case: all elements at max
    dynamic_range = max_partial_sum / smallest_product
    bits = int(np.ceil(np.log2(dynamic_range))) + 1   # +1 for sign
    enob = np.log2(dynamic_range)
    snr_db = 6.02 * bits + 1.76
    return {
        'k': k,
        'max_partial_sum': max_partial_sum,
        'dynamic_range': dynamic_range,
        'required_bits': bits,
        'enob': enob,
        'snr_db': snr_db,
    }
