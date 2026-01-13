# tests/test_run_experiments.py
import json
from pathlib import Path
import numpy as np

from training.run_experiments import run_experiments

def test_run_experiments_smoke(tmp_path, tiny_problem):
    # on fabrique une "instance list" minimale compatible avec run_experiments
    # si ton CNFInstance est importable et a ces champs, utilise-le plutôt.
    class DummyCNF:
        def __init__(self, n_vars, clauses):
            self.n_vars = n_vars
            self.clauses = clauses

    instances = [DummyCNF(tiny_problem.n_vars, tiny_problem.clauses)]
    out_path = tmp_path / "results.json"

    run_experiments(
        instances=instances,
        instance_type="uniform",
        algos=["sa"],
        budgets=[20],
        replicas=2,
        out_path=str(out_path),
        seed=0,
        target_energy=0,
        save_energy_trace=True,
        beta_start=0.1,
        beta_end=3.0,
        beta_nmc=2.0,
        backbone_threshold=0.5,
        n_nmc_steps=2,
        nmc_cycles=1,
        nmc_sw_per_cycle=5,
        rlnmc_model=None,
        n_rl_steps=2,
        rl_start_frac=0.0,
        include_assignment_in_obs=False,
        rlnmc_deterministic=False,
    )

    payload = json.loads(Path(out_path).read_text(encoding="utf-8"))
    assert "meta" in payload and "runs" in payload
    assert len(payload["runs"]) == 1 * 1 * 2 * 1  # inst * budget * replicas * algos

    r0 = payload["runs"][0]
    assert "best_x" in r0
    assert "energies" in r0
    assert isinstance(r0["best_energy"], int)