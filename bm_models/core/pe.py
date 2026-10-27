"""Bounded platform adapter for the pinned kanna cycle-stepped PE model."""
import json
from pathlib import Path

import numpy as np

from ..vendor.neuromorphic_x1.neuromorphic_x2 import NeuromorphicX2, ChipTiming
from ..vendor.neuromorphic_x1.energy import EnergyCalculator

SOURCE = json.loads((Path(__file__).parents[1] / 'vendor/neuromorphic_x1/SOURCE.json').read_text())


def integer(p, name, default, minimum, maximum):
    value = p.get(name, default)
    if isinstance(value, bool) or not isinstance(value, int) or not minimum <= value <= maximum:
        raise ValueError(f'{name} must be an integer from {minimum} to {maximum}')
    return value


def simulate(p):
    rows = integer(p, 'rows', 512, 1, 512)
    cols = integer(p, 'cols', 512, 1, 512)
    iterations = integer(p, 'iterations', 0, 0, 8)
    timing_params = p.get('timing', {})
    if not isinstance(timing_params, dict):
        raise ValueError('timing must be an object')
    allowed = {'program_cells_per_cycle', 'program_cycles_per_cell', 'input_words_per_cycle', 'output_words_per_cycle', 'read_words_per_cycle'}
    if set(timing_params) - allowed:
        raise ValueError('Unknown timing option; compute latency is fixed at 288 cycles')
    timing = ChipTiming(**{k: integer(timing_params, k, 1, 1, 100000) for k in timing_params})
    if 'weights' in p:
        weights = np.asarray(p['weights'], dtype=float)
    else:
        mode = p.get('weight_mode', 'identity')
        if mode not in ('identity', 'ones', 'zeros'):
            raise ValueError('weight_mode must be identity, ones or zeros, or supply weights')
        weights = np.eye(rows, cols) if mode == 'identity' else np.full((rows, cols), int(mode == 'ones'), dtype=float)
    inputs = np.asarray(p.get('input', [1] * rows), dtype=float)
    if weights.shape != (rows, cols) or not np.isfinite(weights).all():
        raise ValueError(f'weights must be a finite {rows}x{cols} matrix')
    if inputs.shape != (rows,) or not np.isfinite(inputs).all():
        raise ValueError(f'input must contain {rows} finite numbers')
    # Keep an API request bounded even with caller-supplied architectural timings.
    ticks = ((rows * cols + timing.program_cells_per_cycle - 1) // timing.program_cells_per_cycle) * timing.program_cycles_per_cell
    ticks += (iterations + 1) * (288 + (cols + timing.output_words_per_cycle - 1) // timing.output_words_per_cycle)
    ticks += iterations * ((cols + timing.input_words_per_cycle - 1) // timing.input_words_per_cycle)
    ticks += (cols + timing.read_words_per_cycle - 1) // timing.read_words_per_cycle
    if ticks > 15_000_000:
        raise ValueError('Requested simulation exceeds the 15 million cycle request limit')
    calc = EnergyCalculator(
        output_buffer_store_energy_per_buffer=p.get('output_buffer_store_energy_J'),
        output_buffer_read_energy_per_buffer=p.get('output_buffer_read_energy_J'))
    chip = NeuromorphicX2(iterations=iterations, input_size=rows, output_size=cols, timing=timing)
    chip.program_weights(weights)
    chip.load_inputs(inputs)
    chip.command(chip.COMPUTE)
    chip.run_until_idle(max_cycles=15_000_000)
    output = chip.read_outputs()
    if not np.isfinite(output).all():
        raise ValueError('Arithmetic overflow: use smaller weights and input values')
    words = chip.output_buffer_words
    energy = {
        'output_buffer_store_J': calc.output_buffer_store_energy(words['STORE'], cols),
        'output_buffer_read_J': calc.output_buffer_read_energy(words['READ'] + words['FEEDBACK'], cols),
    }
    binary = bool(np.isin(weights, (0, 1)).all() and np.isin(inputs, (0, 1)).all() and iterations == 0)
    if binary:
        energy.update(program_J=float(calc.program_phase_energy(weights)),
                      array_read_J=float(calc.read_phase_energy(weights)),
                      compute_J=float(calc.compute_phase_energy(inputs, weights)))
    else:
        energy.update(program_J=None, array_read_J=None, compute_J=None)
    complete = all(v is not None for v in energy.values())
    energy['modeled_total_J'] = sum(energy.values()) if complete else None
    energy['configured_subtotal_J'] = sum(v for k, v in energy.items() if k != 'modeled_total_J' and v is not None)
    energy['status'] = 'MODELED TOTAL' if complete else 'PARTIAL MODELED TOTAL'
    energy['note'] = ('Array energy is reported only for binary weights/input without temporal feedback. '
                      'Unset buffer costs are unknown, not zero. Array READ is charged once; FIFO READ uses transferred words. '
                      'These are uncalibrated model estimates, not measured board energy.')
    return {'backend': 'behavioral_simulator', 'physical_hardware': False,
            'shape': [rows, cols], 'output': output.tolist(),
            'initial_output': chip.initial_output_vector.tolist(),
            'stage_cycles': chip.stage_cycles, 'total_cycles': chip.cycle,
            'output_buffer_words': words, 'status': chip.status, 'energy': energy,
            'source_commit': SOURCE['commit'], 'source_repository': SOURCE['repository'],
            'timing_note': 'One tick at a time; ideal arithmetic. Input staging is untimed; COMPUTE is 288 cycles per pass. No clock frequency or measured silicon timing assumed.'}
