# tests/test_simulated_annealing.py
import numpy as np

from algorithms.simulated_annealing import SimulatedAnnealing, make_linear_beta_schedule

def test_sa_runs_and_returns_valid_result(tiny_problem, rng, random_x0):
    betas = make_linear_beta_schedule(0.1, 5.0, n_sweeps=50)
    sa = SimulatedAnnealing(problem=tiny_problem, beta_schedule=betas, seed=123)

    res = sa.run(x0=random_x0)

    assert isinstance(res.best_energy, int)
    assert res.best_x.shape == (tiny_problem.n_vars,)
    assert res.energy_trace.shape == (50,)
    assert res.best_trace.shape == (50,)
    assert res.n_update_attempts == tiny_problem.n_vars * 50

    # best_trace is non-increasing
    assert np.all(np.diff(res.best_trace) <= 0)

    # consistency: best_energy equals energy(best_x)
    e_check = int(tiny_problem.energy(res.best_x))
    assert e_check == int(res.best_energy)

def test_sa_reproducible(tiny_problem, random_x0):
    betas = make_linear_beta_schedule(0.1, 5.0, n_sweeps=30)

    sa1 = SimulatedAnnealing(problem=tiny_problem, beta_schedule=betas, seed=7)
    sa2 = SimulatedAnnealing(problem=tiny_problem, beta_schedule=betas, seed=7)

    r1 = sa1.run(x0=random_x0)
    r2 = sa2.run(x0=random_x0)

    assert np.array_equal(r1.best_x, r2.best_x)
    assert r1.best_energy == r2.best_energy
    assert np.array_equal(r1.best_trace, r2.best_trace)