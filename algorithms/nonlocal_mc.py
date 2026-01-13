"""algorithms/nonlocal_mc.py

Nonlocal Monte Carlo (NMC) for MAX-k-SAT.

This module extends Simulated Annealing by introducing *nonlocal jumps*
that explicitly break rigid (backbone) variables in order to escape deep
local minima.

We implement the simplified NMC scheme used in the RLNMC paper:

- Rigid variables are detected via local fields H_i = ΔE_i / 2.
- Backbone variables are those with |H_i| > θ (fixed threshold).
- A nonlocal move ("NMC jump") consists of:
    1) Backbone excitation (infinite-temperature randomization)
    2) Relaxation of non-backbone variables at low temperature
    3) Full low-temperature optimization

This file provides:
- `nmc_jump`: a single nonlocal transition primitive
- `NonlocalMonteCarloAnnealing`: SA + periodic NMC jumps

The implementation is deliberately explicit and pedagogical, prioritizing
clarity and correctness over extreme performance.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional
import numpy as np

from env.sat_env import SATProblem, metropolis_sweep


# -----------------------------------------------------------------------------
# Core NMC jump primitive
# -----------------------------------------------------------------------------

def nmc_one_cycle(
    problem: SATProblem,
    x: np.ndarray,
    beta: float,
    backbone_mask: np.ndarray,
    rng: np.random.Generator,
    n_relax_sweeps: int = 1,
    n_full_sweeps: int = 5,
) -> np.ndarray:
    """Perform ONE simplified NMC cycle.

    Stage 1 (excite backbone):
        Randomize variables in the backbone subset (infinite-temperature sampling).
    Stage 2 (relax non-backbone):
        Run low-T Metropolis attempts updating ONLY non-backbone variables.
    Stage 3 (full optimize):
        Run a few low-T Metropolis sweeps over all variables.

    Parameters
    ----------
    problem:
        SATProblem instance.
    x:
        Current assignment (0/1 vector), will not be modified in-place.
    beta:
        Inverse temperature used for relaxation/optimization.
    backbone_mask:
        Boolean mask of shape (N,) indicating backbone variables.
    rng:
        NumPy RNG.
    n_relax_sweeps:
        Number of relaxation sweeps on non-backbone variables.
    n_full_sweeps:
        Number of full sweeps on all variables.

    Returns
    -------
    x_new:
        Updated assignment after the NMC cycle.
    """
    x = np.asarray(x, dtype=np.int8).copy()
    N = int(problem.n_vars)

    backbone_mask = np.asarray(backbone_mask, dtype=bool)
    B = np.where(backbone_mask)[0]
    if B.size > 0:
        # Infinite-temperature excitation: assign fresh random values to backbone vars.
        x[B] = rng.integers(0, 2, size=B.size, dtype=np.int8)

    NB = np.where(~backbone_mask)[0]

    # Relaxation: update only non-backbone variables (backbone fixed).
    # If NB is empty (e.g. policy selected all variables as backbone), we skip relaxation.
    if NB.size > 0:
        for _ in range(int(n_relax_sweeps)):
            # A relaxation "sweep" is ~|NB| attempted flips on the non-backbone subset.
            n_attempts = int(max(1, NB.size))
            for _a in range(n_attempts):
                v = int(NB[int(rng.integers(0, NB.size))])
                dE = int(problem.delta_energy_flip(x, v))
                if dE <= 0:
                    x[v] = 1 - x[v]
                else:
                    if rng.random() < float(np.exp(-beta * float(dE))):
                        x[v] = 1 - x[v]

    # Full optimization: standard Metropolis sweeps over all variables
    for _ in range(int(n_full_sweeps)):
        x, _ = metropolis_sweep(problem, x, beta=beta, rng=rng)

    return x


def nmc_jump(
    problem: SATProblem,
    x: np.ndarray,
    beta: float,
    backbone_mask: np.ndarray,
    n_cycles: int,
    n_relax_sweeps: int,
    n_full_sweeps: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Perform a nonlocal Monte Carlo (NMC) *jump* consisting of several cycles.

    We return the best assignment (lowest energy) encountered across cycles.

    Parameters
    ----------
    problem:
        SATProblem instance.
    x:
        Current assignment (0/1 vector).
    beta:
        Inverse temperature at which relaxation/optimization is performed.
    backbone_mask:
        Boolean array of shape (N,), where True indicates a backbone variable.
    n_cycles:
        Number of excitation–relaxation cycles.
    n_relax_sweeps:
        Relaxation sweeps per cycle (non-backbone only).
    n_full_sweeps:
        Full sweeps per cycle (all variables).
    rng:
        NumPy random generator.

    Returns
    -------
    x_best:
        Best assignment found during the jump.

    Notes
    -----
    - Backbone excitation corresponds to sampling those variables at *infinite temperature*.
    - Keeping the backbone fixed during relaxation prevents immediate return to the original basin.
    """
    x_curr = np.asarray(x, dtype=np.int8).copy()
    x_best = x_curr.copy()
    E_best = int(problem.energy(x_best))

    for _ in range(int(n_cycles)):
        x_curr = nmc_one_cycle(
            problem=problem,
            x=x_curr,
            beta=beta,
            backbone_mask=backbone_mask,
            rng=rng,
            n_relax_sweeps=n_relax_sweeps,
            n_full_sweeps=n_full_sweeps,
        )
        E_curr = int(problem.energy(x_curr))
        if E_curr < E_best:
            E_best = E_curr
            x_best = x_curr.copy()

    return x_best


# -----------------------------------------------------------------------------
# SA + NMC annealing wrapper
# -----------------------------------------------------------------------------

@dataclass
class NMCResult:
    """Container for NMC outputs."""

    best_energy: int
    best_x: np.ndarray

    energy_trace: np.ndarray
    best_trace: np.ndarray
    backbone_sizes: np.ndarray

    n_nmc_jumps: int


class NonlocalMonteCarloAnnealing:
    """Simulated Annealing with periodic Nonlocal Monte Carlo jumps.

    The algorithm follows a standard SA schedule, but once the inverse
    temperature exceeds `beta_nmc`, nonlocal jumps are periodically inserted.

    Backbone variables are selected using the heuristic:
        |H_i| > backbone_threshold
    where H_i = ΔE_i / 2 is the local field.

    n_sw_per_cycle: total sweeps per cycle (relaxation + full optimization).
    """

    def __init__(
        self,
        beta_schedule: np.ndarray,
        beta_nmc: float,
        backbone_threshold: float,
        n_nmc_steps: int,
        n_cycles: int,
        n_sw_per_cycle: int,
        seed: int = 0,
    ):
        self.beta_schedule = np.asarray(beta_schedule, dtype=np.float32)
        self.beta_nmc = float(beta_nmc)
        self.backbone_threshold = float(backbone_threshold)
        self.n_nmc_steps = int(n_nmc_steps)
        self.n_cycles = int(n_cycles)
        self.n_sw_per_cycle = int(n_sw_per_cycle)
        self.rng = np.random.default_rng(seed)

    def run(self, problem: SATProblem, x0: Optional[np.ndarray] = None) -> NMCResult:
        """Run SA augmented with periodic NMC jumps.

        Parameters
        ----------
        problem:
            SATProblem instance.
        x0:
            Optional initial assignment. If None, starts from a random assignment.

        Returns
        -------
        NMCResult
            Best solution found and diagnostic traces.
        """
        N = int(problem.n_vars)
        n_sweeps = int(len(self.beta_schedule))

        # Initialize
        if x0 is None:
            x = problem.random_assignment(self.rng)
        else:
            x = np.asarray(x0, dtype=np.int8).copy()
            if x.shape != (N,):
                raise ValueError(f"x0 must have shape ({N},), got {x.shape}")

        E = int(problem.energy(x))
        best_x = x.copy()
        best_E = E

        energy_trace = np.zeros(n_sweeps, dtype=np.int32)
        best_trace = np.zeros(n_sweeps, dtype=np.int32)
        backbone_sizes = np.zeros(n_sweeps, dtype=np.int32)

        # Determine when to apply NMC jumps.
        # Important: if beta_nmc is never reached, we must NOT schedule any NMC jumps.
        mask = self.beta_schedule >= self.beta_nmc
        if (self.n_nmc_steps > 0) and bool(np.any(mask)):
            idx_start = int(np.argmax(mask))
            nmc_indices = set(np.linspace(idx_start, n_sweeps - 1, num=self.n_nmc_steps, dtype=int))
        else:
            nmc_indices = set()

        n_nmc_jumps = 0

        for t, beta in enumerate(self.beta_schedule):
            beta_f = float(beta)

            if t in nmc_indices:
                # Nonlocal step: compute backbone via |H_i| > threshold
                H = problem.local_fields(x)
                backbone = (np.abs(H) > float(self.backbone_threshold))
                backbone_sizes[t] = int(backbone.sum())

                x = nmc_jump(
                    problem=problem,
                    x=x,
                    beta=beta_f,
                    backbone_mask=backbone,
                    n_cycles=self.n_cycles,
                    # Interpret n_sw_per_cycle as the total sweep budget per cycle: 1 relax sweep (NB only) + (n_sw_per_cycle-1) full sweeps.
                    n_relax_sweeps=1,
                    n_full_sweeps=max(1, int(self.n_sw_per_cycle) - 1),
                    rng=self.rng,
                )
                n_nmc_jumps += 1
            else:
                # Standard SA sweep
                x, _ = metropolis_sweep(problem, x, beta=beta_f, rng=self.rng)

            E = int(problem.energy(x))
            energy_trace[t] = E
            if E < best_E:
                best_E = E
                best_x = x.copy()
            best_trace[t] = best_E

        return NMCResult(
            best_energy=int(best_E),
            best_x=best_x,
            energy_trace=energy_trace,
            best_trace=best_trace,
            backbone_sizes=backbone_sizes,
            n_nmc_jumps=int(n_nmc_jumps),
        )