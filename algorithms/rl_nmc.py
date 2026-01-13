"""algorithms/rl_nmc.py

Reinforcement Learning Nonlocal Monte Carlo (RLNMC).

This module implements the *inference-time* variant of RLNMC, i.e. how a
trained reinforcement-learning policy is used to guide nonlocal Monte Carlo
moves.

Key idea (paper recap)
----------------------
- Classical NMC selects the backbone using a handcrafted heuristic
  (|H_i| > threshold).
- RLNMC replaces this heuristic by a learned policy π_θ.
- At each NMC step, the policy outputs a MultiBinary(N) backbone mask.

Training is handled elsewhere via a Gym environment (RLNMCEnv) and PPO
(stable-baselines3). This file focuses on:

- Loading a trained policy (SB3 PPO)
- Running SA + RL-guided NMC jumps
- Logging energies and backbone statistics

Important consistency note
--------------------------
The observation built here MUST match the one used during training in
`env/sat_env.RLNMCEnv._build_obs()`.

By default, RLNMCEnv uses:
    obs = concat([|H|, E/M, E_best/M, beta/beta_max])
Optionally (if include_assignment_in_obs=True during training):
    obs = concat([|H|, x, E/M, E_best/M, beta/beta_max])

Author: (your name)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from env.sat_env import SATProblem, metropolis_sweep
from algorithms.nonlocal_mc import nmc_jump


# -----------------------------------------------------------------------------
# Result container
# -----------------------------------------------------------------------------


@dataclass
class RLNMCResult:
    """Container for RLNMC inference outputs."""

    best_energy: int
    best_x: np.ndarray

    energy_trace: np.ndarray
    best_trace: np.ndarray
    backbone_sizes: np.ndarray

    n_rl_jumps: int


# -----------------------------------------------------------------------------
# RL-guided NMC inference
# -----------------------------------------------------------------------------


class RLNMCInference:
    """Run Nonlocal Monte Carlo guided by a trained RL policy.

    Parameters
    ----------
    beta_schedule:
        Inverse-temperature schedule (one value per sweep).
    n_steps:
        Number of RL-guided NMC jumps along the schedule.
        These jumps are spaced uniformly between `rl_start_idx` and the end.
    n_cycles:
        Number of excitation–relaxation cycles per NMC jump.
    n_sw_per_cycle:
        Total sweep budget *per cycle* (paper-style accounting). We map it to:
          - n_relax_sweeps = 1
          - n_full_sweeps  = max(1, n_sw_per_cycle - 1)
        (You can override these by passing explicit values.)
    rl_start_idx:
        First schedule index at which RL-guided jumps are allowed.
        Use this to mimic an SA warm-up phase (local minimum) before RL acts.
    include_assignment_in_obs:
        MUST match the flag used in the training environment.
    seed:
        RNG seed.
    model_path:
        Optional path to a trained PPO model (stable-baselines3 .zip file).
    """

    def __init__(
        self,
        beta_schedule: np.ndarray,
        n_steps: int,
        n_cycles: int,
        n_sw_per_cycle: int,
        rl_start_idx: int = 0,
        include_assignment_in_obs: bool = False,
        seed: int = 0,
        model_path: Optional[str] = None,
        n_relax_sweeps: Optional[int] = None,
        n_full_sweeps: Optional[int] = None,
    ):
        self.beta_schedule = np.asarray(beta_schedule, dtype=np.float32)
        if self.beta_schedule.ndim != 1:
            raise ValueError("beta_schedule must be a 1D array")

        self.n_steps = int(n_steps)
        self.n_cycles = int(n_cycles)
        self.n_sw_per_cycle = int(n_sw_per_cycle)
        self.rl_start_idx = int(rl_start_idx)
        self.include_assignment_in_obs = bool(include_assignment_in_obs)

        # Default mapping consistent with our NMC wrapper.
        self.n_relax_sweeps = int(1 if n_relax_sweeps is None else n_relax_sweeps)
        self.n_full_sweeps = int(max(1, self.n_sw_per_cycle - 1) if n_full_sweeps is None else n_full_sweeps)

        self.rng = np.random.default_rng(int(seed))

        self.model = None
        if model_path is not None:
            self.load(model_path)

    # ------------------------------------------------------------------
    # Model handling
    # ------------------------------------------------------------------

    def load(self, model_path: str) -> None:
        """Load a trained PPO policy from disk."""
        from stable_baselines3 import PPO

        self.model = PPO.load(model_path)

    # ------------------------------------------------------------------
    # Observation builder (must match RLNMCEnv)
    # ------------------------------------------------------------------

    def _build_observation(
        self,
        problem: SATProblem,
        x: np.ndarray,
        E: int,
        E_best: int,
        beta: float,
    ) -> np.ndarray:
        """Build observation vector consistent with `RLNMCEnv._build_obs()`."""
        H = problem.local_fields(x)
        absH = np.abs(H).astype(np.float32)

        M = float(problem.n_clauses)
        E_norm = float(E) / M
        Ebest_norm = float(E_best) / M
        beta_norm = float(beta) / float(np.max(self.beta_schedule) + 1e-8)

        parts = [absH]
        if self.include_assignment_in_obs:
            parts.append(x.astype(np.float32))
        parts.append(np.array([E_norm, Ebest_norm, beta_norm], dtype=np.float32))
        return np.concatenate(parts, axis=0).astype(np.float32)

    # ------------------------------------------------------------------
    # Main inference loop
    # ------------------------------------------------------------------

    def run(self, problem: SATProblem, x0: Optional[np.ndarray] = None) -> RLNMCResult:
        """Run RLNMC inference on a given SAT problem.

        Parameters
        ----------
        problem:
            SATProblem instance.
        x0:
            Optional initial assignment. If None, starts from a random assignment.

        Returns
        -------
        RLNMCResult
            Best configuration found and diagnostic traces.
        """
        if self.model is None:
            raise RuntimeError("No RL policy loaded. Call load(model_path) first.")

        N = int(problem.n_vars)
        n_sweeps = int(len(self.beta_schedule))
        if not (0 <= self.rl_start_idx < n_sweeps):
            raise ValueError("rl_start_idx out of range")

        # Initialize
        if x0 is None:
            x = problem.random_assignment(self.rng)
        else:
            x = np.asarray(x0, dtype=np.int8).copy()
            if x.shape != (N,):
                raise ValueError(f"x0 must have shape ({N},), got {x.shape}")

        E = int(problem.energy(x))
        best_E = int(E)
        best_x = x.copy()

        energy_trace = np.zeros(n_sweeps, dtype=np.int32)
        best_trace = np.zeros(n_sweeps, dtype=np.int32)
        backbone_sizes = np.zeros(n_sweeps, dtype=np.int32)

        # Indices where RL-guided NMC jumps are applied
        if self.n_steps > 0:
            rl_indices = set(np.linspace(self.rl_start_idx, n_sweeps - 1, num=self.n_steps, dtype=int).tolist())
        else:
            rl_indices = set()

        n_rl_jumps = 0

        for t, beta in enumerate(self.beta_schedule):
            beta_f = float(beta)

            if t in rl_indices:
                # ------------------------------------------------------
                # RL-guided nonlocal move
                # ------------------------------------------------------
                obs = self._build_observation(problem, x, E, best_E, beta_f)

                # MultiBinary action (mask)
                action, _ = self.model.predict(obs, deterministic=False)
                backbone_mask = np.asarray(action, dtype=np.int8).reshape(-1)
                if backbone_mask.shape[0] != N:
                    raise ValueError(f"Policy returned action of shape {backbone_mask.shape}, expected ({N},)")

                backbone_bool = backbone_mask.astype(bool)
                backbone_sizes[t] = int(backbone_bool.sum())

                x = nmc_jump(
                    problem=problem,
                    x=x,
                    beta=beta_f,
                    backbone_mask=backbone_bool,
                    n_cycles=self.n_cycles,
                    n_relax_sweeps=self.n_relax_sweeps,
                    n_full_sweeps=self.n_full_sweeps,
                    rng=self.rng,
                )
                n_rl_jumps += 1

                # Recompute energy after nonlocal transition
                E = int(problem.energy(x))
            else:
                # ------------------------------------------------------
                # Standard SA sweep between RL jumps
                # ------------------------------------------------------
                x, _ = metropolis_sweep(problem, x, beta=beta_f, rng=self.rng)
                E = int(problem.energy(x))

            # Track best
            if E < best_E:
                best_E = int(E)
                best_x = x.copy()

            # Logging
            energy_trace[t] = int(E)
            best_trace[t] = int(best_E)

        return RLNMCResult(
            best_energy=int(best_E),
            best_x=best_x,
            energy_trace=energy_trace,
            best_trace=best_trace,
            backbone_sizes=backbone_sizes,
            n_rl_jumps=int(n_rl_jumps),
        )