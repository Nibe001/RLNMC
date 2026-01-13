# tests/test_metrics.py
import numpy as np

from metrics.metrics import (
    hamming_distance,
    residual_energy,
    residual_energy_curve,
    tts_curve,
    diversity_curve,
    diversity_integral_score,
)

def test_hamming_distance():
    x = [0,1,0,1]
    y = [0,0,0,1]
    assert hamming_distance(x,y,normalize=False) == 1.0
    assert abs(hamming_distance(x,y,normalize=True) - 0.25) < 1e-9

def test_residual_energy_and_curve():
    runs = [
        {"energies": [5,4,4,3]},
        {"energies": [6,6,5,5]},
    ]
    r = residual_energy(runs, optimum_energy=0.0, agg="mean")
    assert abs(r - ((3 + 5)/2)) < 1e-9

    t, c = residual_energy_curve(runs, optimum_energy=0.0, agg="median")
    assert t[0] == 0
    assert len(t) == len(c)

def test_tts_curve_basic():
    # 2 runs, success at different times
    runs = [
        {"energies": [2,1,0]},
        {"energies": [2,2,2]},
    ]
    budgets = [0,1,2]
    out = tts_curve(runs, budgets, target_energy=0.0, confidence=0.99, with_ci=False)
    assert len(out) == 3
    assert out[-1].p_success == 0.5  # at budget=2, 1/2 succeeded

def test_diversity():
    sols = [
        np.array([0,0,0,0], dtype=np.int8),
        np.array([1,0,0,0], dtype=np.int8),
        np.array([1,1,0,0], dtype=np.int8),
    ]
    d_grid, div = diversity_curve(sols, thresholds=[0.1,0.5,1.0], normalize=True)
    assert d_grid.shape == div.shape
    score = diversity_integral_score(sols, n_thresholds=5, normalize=True)
    assert 0.0 <= score <= 1.0