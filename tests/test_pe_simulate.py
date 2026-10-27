import numpy as np
import pytest

from bm_models.core.pe import simulate, ChipTiming, NeuromorphicX2


def test_matrix_orientation_and_clock_counts():
    r = simulate({'rows': 2, 'cols': 3, 'weights': [[1, 0, 1], [0, 1, 1]], 'input': [1, 0],
                  'output_buffer_store_energy_J': 2e-9, 'output_buffer_read_energy_J': 3e-9})
    assert r['output'] == [1, 0, 1]
    assert r['stage_cycles'] == {'PROGRAM': 300, 'COMPUTE': 288, 'STORE': 3, 'FEEDBACK': 0, 'READ': 3}
    assert r['total_cycles'] == 594 and r['physical_hardware'] is False
    e = r['energy']
    periphery = 6*27e-12 + 3*2.67e-9
    assert e['program_J'] == pytest.approx(4*10.7882e-9 + 2*6.994e-9 + periphery)
    assert e['compute_J'] == pytest.approx(2*284.484e-15 + 37.37e-15 + periphery)
    assert e['modeled_total_J'] == pytest.approx(2*e['program_J'] + e['compute_J'] + 5e-9)


def test_unconfigured_and_nonbinary_energy_are_not_false_totals():
    r = simulate({'rows': 2, 'cols': 2})
    assert r['energy']['modeled_total_J'] is None
    assert r['energy']['output_buffer_read_J'] is None
    r = simulate({'rows': 2, 'cols': 2, 'input': [2, -1]})
    assert r['output'] == [2, -1]
    assert r['energy']['compute_J'] is None


def test_temporal_feedback_counts_and_arithmetic():
    r = simulate({'rows': 2, 'cols': 2, 'input': [1, 0], 'iterations': 1})
    expected = np.exp([1., 0.]); expected /= expected.sum()
    assert r['output'] == pytest.approx(expected)
    assert r['stage_cycles']['COMPUTE'] == 576
    assert r['output_buffer_words'] == {'STORE': 4, 'READ': 2, 'FEEDBACK': 2}


def test_busy_and_partial_read_upstream_protocol():
    chip = NeuromorphicX2(input_size=2, output_size=2, timing=ChipTiming(program_cycles_per_cell=1))
    chip.load_weights(np.eye(2)); chip.command(chip.PROGRAM)
    with pytest.raises(RuntimeError, match='busy'): chip.command(chip.COMPUTE)
    chip.run_until_idle(); chip.load_inputs([1, 0]); chip.command(chip.COMPUTE); chip.run_until_idle()
    assert chip.read_outputs(1).tolist() == [1]
    chip.load_inputs([1, 0])
    with pytest.raises(RuntimeError, match='pending'): chip.command(chip.COMPUTE)
    assert chip.read_outputs(1).tolist() == [0]
    assert chip.output_buffer_words['READ'] == 2


@pytest.mark.parametrize('p', [
    {'rows': 0}, {'rows': True}, {'cols': 513}, {'rows': 2, 'cols': 3, 'iterations': 1},
    {'rows': 2, 'cols': 2, 'input': [1]}, {'rows': 2, 'cols': 2, 'input': [float('nan'), 0]},
    {'timing': {'compute_cycles': 1}}, {'timing': {'program_cycles_per_cell': 100000}},
    {'output_buffer_read_energy_J': -1},
])
def test_invalid_parameters(p):
    with pytest.raises(ValueError): simulate(p)
