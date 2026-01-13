# tests/test_rl_nmc.py
import numpy as np

from algorithms.rl_nmc import RLNMCInference
from env.sat_env import linear_beta_schedule

class DummyPolicy:
    def __init__(self, n_vars: int, mode: str = "zeros"):
        self.n_vars = n_vars
        self.mode = mode

    def predict(self, obs, deterministic=False):
        # obs ignored; returns MultiBinary(N)
        if self.mode == "zeros":
            a = np.zeros(self.n_vars, dtype=np.int8)
        elif self.mode == "ones":
            a = np.ones(self.n_vars, dtype=np.int8)
        else:
            # random but deterministic from obs sum for stability
            s = int(np.sum(obs) * 1e6) % 2
            a = np.full(self.n_vars, s, dtype=np.int8)
        return a, None

def test_rlnmc_runs_with_dummy_policy(tiny_problem):
    betas = linear_beta_schedule(0.1, 5.0, 25)
    solver = RLNMCInference(
        beta_schedule=betas,
        n_steps=5,
        n_cycles=2,
        n_sw_per_cycle=8,
        rl_start_idx=0,
        include_assignment_in_obs=False,
        seed=0,
        model_path=None,
    )
    solver.model = DummyPolicy(tiny_problem.n_vars, mode="zeros")

    res = solver.run(problem=tiny_problem, x0=None)

    assert res.best_x.shape == (tiny_problem.n_vars,)
    assert res.energy_trace.shape == (25,)
    assert res.best_trace.shape == (25,)
    assert res.backbone_sizes.shape == (25,)
    assert res.n_rl_jumps >= 0
    assert np.all(np.diff(res.best_trace) <= 0)

def test_rlnmc_action_shape_mismatch_raises(tiny_problem):
    betas = linear_beta_schedule(0.1, 5.0, 10)
    solver = RLNMCInference(betas, n_steps=2, n_cycles=1, n_sw_per_cycle=5, seed=0)
    solver.model = DummyPolicy(n_vars=tiny_problem.n_vars + 1)  # wrong size

    try:
        solver.run(problem=tiny_problem, x0=None)
        assert False, "Expected ValueError due to wrong action shape"
    except ValueError:
        pass