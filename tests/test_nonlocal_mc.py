# tests/test_nonlocal_mc.py
import numpy as np

from algorithms.nonlocal_mc import nmc_one_cycle, NonlocalMonteCarloAnnealing
from env.sat_env import linear_beta_schedule

def test_nmc_one_cycle_binary_and_shape(tiny_problem, rng, random_x0):
    x = random_x0.copy()
    backbone = np.zeros_like(x, dtype=bool)
    backbone[:2] = True

    x2 = nmc_one_cycle(
        problem=tiny_problem,
        x=x,
        beta=3.0,
        backbone_mask=backbone,
        rng=rng,
        n_relax_sweeps=1,
        n_full_sweeps=2,
    )
    assert x2.shape == x.shape
    assert set(np.unique(x2)).issubset({0, 1})

def test_nmc_annealing_runs(tiny_problem):
    betas = linear_beta_schedule(0.1, 6.0, 40)
    solver = NonlocalMonteCarloAnnealing(
        beta_schedule=betas,
        beta_nmc=3.0,
        backbone_threshold=0.5,
        n_nmc_steps=5,
        n_cycles=2,
        n_sw_per_cycle=10,
        seed=0,
    )

    res = solver.run(problem=tiny_problem, x0=None)

    assert isinstance(res.best_energy, int)
    assert res.best_x.shape == (tiny_problem.n_vars,)
    assert res.energy_trace.shape == (40,)
    assert res.best_trace.shape == (40,)
    assert res.backbone_sizes.shape == (40,)
    assert res.n_nmc_jumps >= 0
    assert np.all(np.diff(res.best_trace) <= 0)

def test_nmc_no_jump_if_beta_never_reached(tiny_problem):
    # beta_schedule always below beta_nmc
    betas = linear_beta_schedule(0.1, 0.2, 30)
    solver = NonlocalMonteCarloAnnealing(
        beta_schedule=betas,
        beta_nmc=10.0,
        backbone_threshold=0.5,
        n_nmc_steps=10,
        n_cycles=2,
        n_sw_per_cycle=10,
        seed=0,
    )
    res = solver.run(problem=tiny_problem, x0=None)
    assert int(res.n_nmc_jumps) == 0
    assert np.all(res.backbone_sizes == 0)