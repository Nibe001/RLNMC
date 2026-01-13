"""algorithms/simulated_annealing.py

Simulated Annealing (SA) for MAX-k-SAT / energy minimization.

This module provides a clean, reusable implementation of SA using Metropolis
single-variable flip updates.

Key conventions used throughout the project
------------------------------------------
- A *configuration* is a binary vector x in {0,1}^N.
- The *energy* E(x) is the number of violated clauses (so we want to MINIMIZE it).
- A *flip* is x_i <- 1 - x_i.
- A *sweep* is N attempted flips (N = number of variables). During one sweep,
  the same variable may be proposed multiple times and some variables may not be
  proposed at all (when sampling indices uniformly at random). This is standard
  in MCMC implementations and keeps the code simple and unbiased.

The code is written to be:
- correct and well documented,
- deterministic under a fixed RNG seed,
- compatible with the SATProblem utilities in `env/sat_env.py`.

"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from env.sat_env import SATProblem


# -----------------------------------------------------------------------------
# Schedules
# -----------------------------------------------------------------------------

def make_linear_beta_schedule(beta_i: float, beta_f: float, n_sweeps: int) -> np.ndarray:
    """Create a linear schedule in inverse temperature beta.

    Parameters
    ----------
    beta_i:
        Initial inverse temperature (low beta = high temperature).
    beta_f:
        Final inverse temperature (high beta = low temperature).
    n_sweeps:
        Number of sweeps (length of the schedule).

    Returns
    -------
    beta_schedule:
        Array of shape (n_sweeps,) with dtype float32.
    """
    if n_sweeps <= 0:
        raise ValueError("n_sweeps must be positive")
    return np.linspace(beta_i, beta_f, num=n_sweeps, dtype=np.float32)


def make_geometric_beta_schedule(beta_i: float, beta_f: float, n_sweeps: int) -> np.ndarray:
    """Create a geometric schedule in beta.

    This is sometimes used when you want multiplicative changes in temperature.
    """
    if n_sweeps <= 0:
        raise ValueError("n_sweeps must be positive")
    if beta_i <= 0 or beta_f <= 0:
        raise ValueError("beta_i and beta_f must be > 0 for geometric schedule")
    # Avoid numerical issues when n_sweeps=1
    if n_sweeps == 1:
        return np.array([beta_f], dtype=np.float32)
    ratio = (beta_f / beta_i) ** (1.0 / (n_sweeps - 1))
    betas = beta_i * (ratio ** np.arange(n_sweeps, dtype=np.float32))
    return betas.astype(np.float32)


# -----------------------------------------------------------------------------
# Metropolis accept rule
# -----------------------------------------------------------------------------

def metropolis_accept(delta_e: int, beta: float, rng: np.random.Generator) -> bool:
    """Metropolis acceptance rule.

    For a proposed move x -> x' with energy change ΔE = E(x') - E(x):
      - if ΔE <= 0: always accept
      - if ΔE > 0 : accept with probability exp(-beta * ΔE)

    Notes
    -----
    - beta is the inverse temperature.
    - This rule satisfies detailed balance for a fixed beta.

    Parameters
    ----------
    delta_e:
        Energy change (integer for SAT energy).
    beta:
        Inverse temperature.
    rng:
        NumPy random generator.

    Returns
    -------
    accept:
        True if the move is accepted.
    """
    if delta_e <= 0:
        return True
    # exp(-beta * ΔE) can underflow for large beta*ΔE; that is fine (-> 0 prob).
    return rng.random() < np.exp(-beta * float(delta_e))


# -----------------------------------------------------------------------------
# Results container
# -----------------------------------------------------------------------------

@dataclass
class SAResult:
    """Container for SA outputs."""

    best_energy: int
    best_x: np.ndarray

    # Traces (length = n_sweeps)
    energy_trace: np.ndarray  # energy after each sweep
    best_trace: np.ndarray    # best energy so far after each sweep

    # Optional: number of attempted updates
    n_update_attempts: int


# -----------------------------------------------------------------------------
# Main SA class
# -----------------------------------------------------------------------------

class SimulatedAnnealing:
    """Simulated Annealing for SAT energy minimization.

    Parameters
    ----------
    problem:
        SATProblem instance holding the clauses / structure.
    beta_schedule:
        Array of inverse temperatures, one per sweep.
    seed:
        RNG seed for reproducibility.
    """

    def __init__(self, problem: SATProblem, beta_schedule: np.ndarray, seed: int = 0):
        self.problem = problem
        self.beta_schedule = np.asarray(beta_schedule, dtype=np.float32)
        if self.beta_schedule.ndim != 1:
            raise ValueError("beta_schedule must be a 1D array")
        self.rng = np.random.default_rng(seed)

    def run(self, x0: Optional[np.ndarray] = None) -> SAResult:
        """Run simulated annealing.

        Parameters
        ----------
        x0:
            Optional initial assignment. If None, a random assignment is used.

        Returns
        -------
        SAResult
            Best configuration encountered and traces.
        """
        N = int(self.problem.n_vars)

        # Initialize configuration
        if x0 is None:
            x = self.problem.random_assignment(self.rng)
        else:
            x = np.asarray(x0, dtype=np.int8).copy()
            if x.shape != (N,):
                raise ValueError(f"x0 must have shape ({N},), got {x.shape}")

        # Current and best energies
        E = int(self.problem.energy(x))
        best_E = int(E)
        best_x = x.copy()

        n_sweeps = int(len(self.beta_schedule))
        energy_trace = np.zeros(n_sweeps, dtype=np.int32)
        best_trace = np.zeros(n_sweeps, dtype=np.int32)

        n_update_attempts = 0

        # Main loop over sweeps
        for t, beta in enumerate(self.beta_schedule):
            beta_f = float(beta)

            # One sweep = N attempted flips
            for _ in range(N):
                n_update_attempts += 1

                # Choose variable index uniformly at random
                i = int(self.rng.integers(0, N))

                # Compute ΔE if we flip i
                delta_e = int(self.problem.delta_energy_flip(x, i))

                # Metropolis accept/reject
                if metropolis_accept(delta_e, beta_f, self.rng):
                    x[i] = 1 - x[i]
                    E += delta_e

                    # Track best as soon as we improve
                    if E < best_E:
                        best_E = int(E)
                        best_x = x.copy()

            # Log after each sweep
            energy_trace[t] = int(E)
            best_trace[t] = int(best_E)

        return SAResult(
            best_energy=int(best_E),
            best_x=best_x,
            energy_trace=energy_trace,
            best_trace=best_trace,
            n_update_attempts=n_update_attempts,
        )

    def run_with_budget(
        self,
        n_sweeps: int,
        beta_i: float,
        beta_f: float,
        x0: Optional[np.ndarray] = None,
        schedule: str = "linear",
    ) -> SAResult:
        """Convenience wrapper when you want to specify the schedule by endpoints.

        This is useful in quick experiments when you want to scan budgets.
        """
        if schedule == "linear":
            betas = make_linear_beta_schedule(beta_i, beta_f, n_sweeps)
        elif schedule == "geometric":
            betas = make_geometric_beta_schedule(beta_i, beta_f, n_sweeps)
        else:
            raise ValueError(f"Unknown schedule: {schedule}")

        # Temporarily override schedule for this run
        old = self.beta_schedule
        self.beta_schedule = betas
        try:
            return self.run(x0=x0)
        finally:
            self.beta_schedule = old
