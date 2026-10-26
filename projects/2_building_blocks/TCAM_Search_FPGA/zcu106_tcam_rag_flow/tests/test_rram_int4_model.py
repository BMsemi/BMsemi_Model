from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path


def _load_model_module():
    root = Path(__file__).resolve().parents[1]
    script = root / "scripts" / "model_rram_int4_programming_accuracy.py"
    spec = importlib.util.spec_from_file_location("rram_int4_model", script)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_int4_targets_and_deterministic_summary():
    model = _load_model_module()
    targets = model.int4_targets(5.0, 95.0)
    assert len(targets) == 16
    assert targets[0] == 5.0
    assert targets[-1] == 95.0

    args = argparse.Namespace(
        trials_per_code=5,
        g_min_us=5.0,
        g_max_us=95.0,
        read_noise_sigma_us=0.75,
        pulse_cv=0.20,
        max_pulses=24,
        sample_records=0,
        seed=7,
    )
    result = model.run_model(args)
    summary = result["summary"]
    assert summary["total_trials"] == 80
    assert 0.0 <= summary["accuracy"] <= 1.0
    assert len(summary["by_code"]) == 16
    assert result["model"]["target_conductance_us"] == [float(v) for v in targets]
