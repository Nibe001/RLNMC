

"""env/sat_env.py

SAT / MAX-k-SAT environment and problem utilities.

This module provides two core components:

1) SATProblem
   - Holds a k-CNF instance (variables + clauses)
   - Computes energy E(x) = number of violated clauses
   - Supports efficient local computations for MCMC-style solvers:
       * per-clause satisfaction
       * local flip delta-energy ΔE_i
       * local fields H_i := (E(x_i -> 1-x_i) - E(x)) / 2  (used by NMC/RLNMC)

2) RLNMCEnv (Gymnasium-compatible)
   - A lightweight RL environment for training a backbone-selection policy.
   - Observation: per-variable local fields |H_i| plus optional globals
   - Action: MultiBinary(N) backbone mask (which vars to excite)
   - Transition: executes one NMC-style nonlocal move (excite + relax + full optimize)
   - Reward: improvement of best-so-far energy (paper-style shaping)

Notes
-----
- We keep this environment intentionally simple and CPU-friendly.
- The original RLNMC paper uses a GNN+GRU policy and heavy GPU vectorization.
  Here we target a PyTorch + stable-baselines3 PPO baseline with an MLP policy.

- This file does NOT implement SA/NMC/RLNMC fully; those live in `algorithms/`.
  However, the environment needs to be able to execute a *single* NMC-like step
  to define RL transitions. We implement that step using utilities in this file.

- For MAX-k-SAT we minimize the number of unsatisfied clauses.

Author: (your name)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple, Dict, Any

import numpy as np

# Gymnasium is the maintained fork of gym; stable-baselines3 supports it.
try:
    import gymnasium as gym
    from gymnasium import spaces
except Exception:  # pragma: no cover
    import gym
    from gym import spaces


Clause = np.ndarray  # shape (k,2): (var_index, sign) with sign in {+1,-1}


# -----------------------------------------------------------------------------
# SATProblem
# -----------------------------------------------------------------------------


@dataclass
class SATProblem:
    """A k-CNF / MAX-k-SAT problem instance.

    Parameters
    ----------
    n_vars:
        Number of variables N.
    clauses:
        List of clauses. Each clause is a (k,2) int array of (var_index, sign).
        - var_index is in [0, N-1]
        - sign = +1 means literal is x_i
        - sign = -1 means literal is ¬x_i

    Notes
    -----
    The energy is:
        E(x) = # of violated clauses

    We maintain a few auxiliary structures for fast delta-energy updates:
    - var_to_clauses[v] : list of clause indices where variable v appears
    - clause_vars[c]    : variable indices in clause c
    - clause_signs[c]   : corresponding signs in clause c

    This is enough to compute ΔE for a single-variable flip without
    re-scanning all clauses.
    """

    n_vars: int
    clauses: List[Clause]

    def __post_init__(self) -> None:
        self.n_vars = int(self.n_vars)
        self.n_clauses = int(len(self.clauses))
        if self.n_clauses == 0:
            raise ValueError("clauses list is empty")

        # Clause size k inferred from the first clause
        k0 = int(self.clauses[0].shape[0])
        self.k = k0

        # Normalize dtype and basic checks
        norm_clauses: List[Clause] = []
        for idx, cl in enumerate(self.clauses):
            cl = np.asarray(cl, dtype=np.int32)
            if cl.shape != (self.k, 2):
                raise ValueError(f"Clause {idx} has shape {cl.shape}, expected {(self.k,2)}")
            # Ensure variable indices are within range
            if np.any(cl[:, 0] < 0) or np.any(cl[:, 0] >= self.n_vars):
                raise ValueError(f"Clause {idx} has out-of-range variable indices")
            # Ensure signs are ±1
            if not np.all(np.isin(cl[:, 1], np.array([1, -1], dtype=np.int32))):
                raise ValueError(f"Clause {idx} has invalid signs; expected ±1")
            norm_clauses.append(cl)
        self.clauses = norm_clauses

        # Precompute var->clauses adjacency
        self.var_to_clauses: List[List[int]] = [[] for _ in range(self.n_vars)]
        self._clause_vars = np.zeros((self.n_clauses, self.k), dtype=np.int32)
        self._clause_signs = np.zeros((self.n_clauses, self.k), dtype=np.int32)

        for c_idx, cl in enumerate(self.clauses):
            vs = cl[:, 0].astype(np.int32)
            ss = cl[:, 1].astype(np.int32)
            self._clause_vars[c_idx] = vs
            self._clause_signs[c_idx] = ss
            for v in vs.tolist():
                self.var_to_clauses[int(v)].append(c_idx)

    # ----------------------
    # Clause satisfaction
    # ----------------------

    def clause_satisfied(self, x: np.ndarray, c_idx: int) -> bool:
        """Return True iff clause c_idx is satisfied by assignment x."""
        vs = self._clause_vars[c_idx]
        ss = self._clause_signs[c_idx]

        # Clause is satisfied if any literal evaluates to True.
        # literal is x[v] when sign=+1, else ¬x[v].
        xv = x[vs]
        # True literal mask
        # sign=+1: literal true when xv==1
        # sign=-1: literal true when xv==0
        return bool(np.any(((ss == 1) & (xv == 1)) | ((ss == -1) & (xv == 0))))
    
    def violated_mask(self, x: np.ndarray) -> np.ndarray:
        """Return boolean array of length M indicating which clauses are violated."""
        violated = np.zeros(self.n_clauses, dtype=bool)
        for c in range(self.n_clauses):
            violated[c] = not self.clause_satisfied(x, c)
        return violated

    # ----------------------
    # Energy and deltas
    # ----------------------

    def energy(self, x: np.ndarray) -> int:
        """Energy E(x) = number of violated clauses."""
        return int(np.sum(self.violated_mask(x)))

    def _clause_sat_given_value(self, v: int, new_val: int, x: np.ndarray, c_idx: int) -> bool:
        """Check clause satisfaction if variable v were set to new_val.

        This helper is used to compute ΔE_i efficiently.
        """
        vs = self._clause_vars[c_idx]
        ss = self._clause_signs[c_idx]

        # We only need to change the value for occurrences of v inside the clause.
        xv = x[vs].copy()
        xv[vs == v] = new_val

        return bool(np.any(((ss == 1) & (xv == 1)) | ((ss == -1) & (xv == 0))))

    def delta_energy_flip(self, x: np.ndarray, v: int) -> int:
        """Compute ΔE = E(x') - E(x) if we flip variable v (x_v -> 1-x_v).

        Complexity: O(deg(v) * k), where deg(v) is number of clauses containing v.
        This is fast enough for moderate N, and keeps the implementation readable.
        """
        v = int(v)
        old_val = int(x[v])
        new_val = 1 - old_val

        dE = 0
        for c_idx in self.var_to_clauses[v]:
            sat_before = self.clause_satisfied(x, c_idx)
            sat_after = self._clause_sat_given_value(v, new_val, x, c_idx)

            # If clause changes from satisfied -> violated, energy increases by +1.
            if sat_before and not sat_after:
                dE += 1
            # If clause changes from violated -> satisfied, energy decreases by -1.
            elif (not sat_before) and sat_after:
                dE -= 1

        return int(dE)

    def local_fields(self, x: np.ndarray) -> np.ndarray:
        """Compute local fields H_i for all variables.

        We use the paper's simplified definition:
            H_i = (E(x_i -> 1-x_i) - E(x)) / 2

        Since E(x_i -> 1-x_i) - E(x) is exactly ΔE_i, we get:
            H_i = ΔE_i / 2

        Returns
        -------
        H : np.ndarray shape (N,), dtype float32
        """
        H = np.zeros(self.n_vars, dtype=np.float32)
        for i in range(self.n_vars):
            H[i] = 0.5 * float(self.delta_energy_flip(x, i))
        return H

    # ----------------------
    # Sampling helpers
    # ----------------------

    def random_assignment(self, rng: np.random.Generator) -> np.ndarray:
        """Sample x ~ Uniform({0,1}^N)."""
        return rng.integers(0, 2, size=self.n_vars, dtype=np.int8)


# -----------------------------------------------------------------------------
# One-step MCMC utilities (Metropolis)
# -----------------------------------------------------------------------------


def metropolis_sweep(
    problem: SATProblem,
    x: np.ndarray,
    beta: float,
    rng: np.random.Generator,
    n_attempts: Optional[int] = None,
) -> Tuple[np.ndarray, int]:
    """Perform one Metropolis sweep.

    A "sweep" means we do ~N single-variable update attempts.

    Parameters
    ----------
    problem:
        SATProblem.
    x:
        Current assignment (modified copy returned).
    beta:
        Inverse temperature (beta = 1/T).
    rng:
        NumPy RNG.
    n_attempts:
        Number of flip proposals in this sweep. If None, defaults to N.

    Returns
    -------
    x_new:
        Updated assignment.
    n_accepted:
        Number of accepted flips.
    """
    x = x.copy()
    N = problem.n_vars
    if n_attempts is None:
        n_attempts = N

    accepted = 0
    for _ in range(int(n_attempts)):
        v = int(rng.integers(0, N))
        dE = problem.delta_energy_flip(x, v)
        if dE <= 0:
            x[v] = 1 - x[v]
            accepted += 1
        else:
            # Metropolis acceptance
            if rng.random() < float(np.exp(-beta * float(dE))):
                x[v] = 1 - x[v]
                accepted += 1

    return x, accepted


# -----------------------------------------------------------------------------
# RLNMCEnv
# -----------------------------------------------------------------------------


class RLNMCEnv(gym.Env):
    """Gymnasium environment for training a backbone-selection policy.

    State and action design (MLP-friendly)
    --------------------------------------
    We keep an observation that is a fixed-size vector:
        obs = concat([
            |H_1|, ..., |H_N|,
            E(x)/M,
            E_best/M,
            beta/beta_max
        ])

    This is intentionally simple for stable-baselines3 MlpPolicy.

    Action
    ------
    MultiBinary(N) mask a ∈ {0,1}^N.
      a_i = 1 means variable i is included in the backbone (excited).

    Transition
    ----------
    One environment step executes a simplified NMC move:
      1) Excite backbone: randomize x_i for i in B
      2) Relax non-backbone: 1 low-T sweep over NB while B fixed
      3) Full optimize: (n_full_sweeps) sweeps over all vars

    Reward
    ------
    Paper-style shaping: reward is the improvement of best-so-far energy.
      r_t = max(0, E_best_prev - E_best_new)

    Episode ends when:
      - reached max_steps, or
      - reached energy 0 if `terminate_on_solution=True`.

    Important
    ---------
    This environment is designed for *training* a policy; it is not meant
    to be the fastest possible SAT simulator.
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        problem: SATProblem,
        beta_schedule: np.ndarray,
        nmc_beta_start_idx: int,
        n_cycles: int = 1,
        n_relax_sweeps: int = 1,
        n_full_sweeps: int = 5,
        seed: int = 0,
        terminate_on_solution: bool = True,
        include_assignment_in_obs: bool = False,
    ) -> None:
        super().__init__()

        self.problem = problem
        self.beta_schedule = np.asarray(beta_schedule, dtype=np.float32)
        if self.beta_schedule.ndim != 1:
            raise ValueError("beta_schedule must be 1D")

        self.n_steps_total = int(len(self.beta_schedule))
        self.nmc_beta_start_idx = int(nmc_beta_start_idx)
        if not (0 <= self.nmc_beta_start_idx < self.n_steps_total):
            raise ValueError("nmc_beta_start_idx out of range")

        self.n_cycles = int(n_cycles)
        self.n_relax_sweeps = int(n_relax_sweeps)
        self.n_full_sweeps = int(n_full_sweeps)
        self.terminate_on_solution = bool(terminate_on_solution)
        self.include_assignment_in_obs = bool(include_assignment_in_obs)

        self.rng = np.random.default_rng(int(seed))

        # Action: select backbone mask
        self.action_space = spaces.MultiBinary(self.problem.n_vars)

        # Observation: |H| plus some globals.
        # Optionally include the raw assignment x (0/1) for additional signal.
        obs_dim = self.problem.n_vars
        if self.include_assignment_in_obs:
            obs_dim += self.problem.n_vars
        obs_dim += 3  # E_norm, Ebest_norm, beta_norm

        self.observation_space = spaces.Box(
            low=0.0,
            high=np.inf, # 1e10,
            shape=(obs_dim,),
            dtype=np.float32,
        )

        # Internal episode state
        self._t = 0
        self._x: Optional[np.ndarray] = None
        self._E: Optional[int] = None
        self._E_best: Optional[int] = None

    # ----------------------
    # Gym API
    # ----------------------

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        if seed is not None:
            self.rng = np.random.default_rng(int(seed))

        self._t = int(self.nmc_beta_start_idx)

        # Start from a random assignment, then "warm up" to reach a local-ish minimum
        x = self.problem.random_assignment(self.rng)

        # Warm-up: run SA up to nmc_beta_start_idx
        for j in range(self.nmc_beta_start_idx):
            beta = float(self.beta_schedule[j])
            x, _ = metropolis_sweep(self.problem, x, beta=beta, rng=self.rng)

        E = self.problem.energy(x)
        self._x = x
        self._E = int(E)
        self._E_best = int(E)

        obs = self._build_obs()
        info: Dict[str, Any] = {
            "t": self._t,
            "energy": self._E,
            "best_energy": self._E_best,
        }
        return obs, info

    def step(self, action):
        if self._x is None or self._E is None or self._E_best is None:
            raise RuntimeError("Call reset() before step().")

        a = np.asarray(action, dtype=np.int8).reshape(-1)
        if a.shape[0] != self.problem.n_vars:
            raise ValueError("Action has wrong shape")

        x = self._x
        E_best_prev = int(self._E_best)

        # Current beta for this NMC step
        beta = float(self.beta_schedule[self._t])

        # Execute (possibly multiple) NMC cycles
        x_new = x
        for _ in range(self.n_cycles):
            x_new = self._nmc_one_cycle(x_new, backbone_mask=a, beta=beta)

        E_new = self.problem.energy(x_new)
        self._x = x_new
        self._E = int(E_new)

        if E_new < self._E_best:
            self._E_best = int(E_new)

        # Reward: improvement of best-so-far energy
        reward = float(max(0, E_best_prev - int(self._E_best)))

        # Advance time index along the schedule
        self._t = min(self._t + 1, self.n_steps_total - 1)

        terminated = False
        if self.terminate_on_solution and self._E_best == 0:
            terminated = True

        truncated = False
        # Episode ends when we reach the end of the schedule
        if self._t >= self.n_steps_total - 1:
            truncated = True

        obs = self._build_obs()
        info: Dict[str, Any] = {
            "t": self._t,
            "energy": self._E,
            "best_energy": self._E_best,
            "beta": beta,
        }
        return obs, reward, terminated, truncated, info

    # ----------------------
    # Observation
    # ----------------------

    def _build_obs(self) -> np.ndarray:
        assert self._x is not None
        assert self._E is not None
        assert self._E_best is not None

        H = self.problem.local_fields(self._x)
        absH = np.abs(H).astype(np.float32)

        # Normalize energies by number of clauses (stable scaling)
        M = float(self.problem.n_clauses)
        E_norm = float(self._E) / M
        Ebest_norm = float(self._E_best) / M

        beta = float(self.beta_schedule[self._t])
        beta_norm = beta / float(np.max(self.beta_schedule) + 1e-8)

        parts = [absH]
        if self.include_assignment_in_obs:
            parts.append(self._x.astype(np.float32))
        parts.append(np.array([E_norm, Ebest_norm, beta_norm], dtype=np.float32))

        obs = np.concatenate(parts, axis=0).astype(np.float32)
        return obs

    # ----------------------
    # NMC step used by the environment
    # ----------------------

    def _nmc_one_cycle(self, x: np.ndarray, backbone_mask: np.ndarray, beta: float) -> np.ndarray:
        """One simplified NMC cycle.

        Stage 1 (excite backbone):
            Randomize backbone variables. This is equivalent to sampling those
            variables at infinite temperature: all flips are accepted, producing
            a random configuration on that subset.

        Stage 2 (relax non-backbone):
            Run a few sweeps updating ONLY non-backbone variables at low T.
            Backbone vars remain fixed.

        Stage 3 (full optimize):
            Run a few low-T sweeps over all variables.

        Parameters
        ----------
        x:
            Current assignment.
        backbone_mask:
            Binary mask of shape (N,).
        beta:
            Inverse temperature for the low-T sweeps.

        Returns
        -------
        x_new:
            Updated assignment after the NMC cycle.
        """
        x = x.copy()
        N = self.problem.n_vars

        # Backbone set
        B = np.where(backbone_mask.astype(bool))[0]
        if B.size > 0:
            x[B] = self.rng.integers(0, 2, size=B.size, dtype=np.int8)

        # Relaxation: update only NB variables
        NB = np.where(~backbone_mask.astype(bool))[0]

        # Relaxation: only if there are non-backbone variables
        if NB.size > 0:
            for _ in range(self.n_relax_sweeps):
                n_attempts = int(NB.size)
                for _a in range(n_attempts):
                    v = int(NB[int(self.rng.integers(0, NB.size))])
                    dE = int(self.problem.delta_energy_flip(x, v))
                    if dE <= 0 or self.rng.random() < np.exp(-beta * float(dE)):
                        x[v] = 1 - x[v]

        # Full optimization: sweep over all variables
        for _ in range(self.n_full_sweeps):
            x, _ = metropolis_sweep(self.problem, x, beta=beta, rng=self.rng)

        return x


# -----------------------------------------------------------------------------
# Convenience: schedule builder
# -----------------------------------------------------------------------------


def linear_beta_schedule(beta_start: float, beta_end: float, n_steps: int) -> np.ndarray:
    """Linear schedule in beta (inverse temperature), as used in the paper."""
    return np.linspace(float(beta_start), float(beta_end), int(n_steps), dtype=np.float32)
