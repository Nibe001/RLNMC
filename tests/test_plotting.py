# tests/test_plotting.py
import matplotlib
matplotlib.use("Agg")  # headless

import numpy as np
from plots.plotting import plot_energy_vs_sweeps, plot_tts_vs_sweeps, plot_diversity_curve

def test_plot_energy_smoke():
    stats = {
        "sa": {"x": np.array([1,2,3]), "median": np.array([10,5,2]), "p25": np.array([9,4,2]), "p75": np.array([11,6,3])}
    }
    fig, ax = plot_energy_vs_sweeps(stats)
    assert fig is not None and ax is not None

def test_plot_tts_smoke():
    stats = {
        "sa": {"x": np.array([100,200]), "tts": np.array([1000,400]), "low": np.array([900,350]), "high": np.array([1200,500])}
    }
    fig, ax = plot_tts_vs_sweeps(stats)
    assert fig is not None and ax is not None

def test_plot_diversity_smoke():
    stats = {
        "sa": {"R": np.array([0.1,0.2,0.3]), "D": np.array([0.8,0.6,0.4])}
    }
    fig, ax = plot_diversity_curve(stats)
    assert fig is not None and ax is not None