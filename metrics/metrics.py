

"""metrics/metrics.py

Metrics for comparing SA / NMC / RLNMC runs.

This module is intentionally *algorithm-agnostic*: it only consumes the raw outputs
produced by your experiment scripts (energies over time, final/best solutions, etc.)

Implemented metrics (aligned with the paper's evaluation protocol)
---------------------------------------------------------------
1) Residual energy
   - For each run, take the minimum energy encountered during the run.
   - Aggregate across runs/instances via mean/median.
   - Optionally subtract a known optimum (e.g., 0 for planted satisfiable instances).

2) Time-To-Solution (TTS_99)
   - Let p_succ(t) be the probability that a run succeeds (energy <= target)
     within a time budget t.
   - Then the number of independent runs required to reach confidence c is:
       n = ceil( log(1-c) / log(1-p_succ) )
     and TTS_c(t) = n * t.

   We also provide a Wilson confidence interval for p_succ and propagate it to
   get a band for TTS.

3) Diversity of solutions
   - Paper definition (high level): build a graph where nodes are solutions and
     edges connect "too-close" solutions (Hamming distance < d). The diversity at
     threshold d is the size of a Maximum Independent Set (MIS) of that graph.

   Computing MIS exactly is NP-hard. For a reproducible *approximation* we use:
   - a greedy maximal independent set heuristic (fast)

   We then compute diversity(d) for a grid of thresholds d in [0,1] and return:
   - the curve diversity(d)
   - an "integral diversity" score = area under the curve (trapezoidal rule)

Expected run record format
--------------------------
Your experiment runner can store each run as a dictionary with (at least) the fields:

    {
      "energies": [e_0, e_1, ..., e_T],     # energy trace vs time steps (sweeps)
      "time": T,                           # optional; defaults to len(energies)-1
      "best_energy": min(energies),        # optional
      "best_x": [0/1]*N,                   # optional best solution bitstring
      "success": bool,                     # optional; derived from target_energy if absent
    }

All functions below try to be flexible: if a field is missing we infer it.

Dependencies
------------
- numpy is required.
- networkx is optional; we implement the greedy MIS ourselves to keep deps light.

"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import math
import numpy as np


# -----------------------------------------------------------------------------
# Utilities
# -----------------------------------------------------------------------------


def _as_1d_int_array(x: Any) -> np.ndarray:
    """Convert an assignment (list/np array/tuple) to a 1D int8 numpy array."""
    arr = np.asarray(x)
    if arr.ndim != 1:
        raise ValueError(f"Expected 1D assignment, got shape {arr.shape}")
    # Accept {0,1} or {-1,+1}; normalize to {0,1}
    # Use unique()+isin() for robustness across dtypes (int/float/object).
    if arr.dtype == bool:
        return arr.astype(np.int8)

    vals = np.unique(arr)
    if np.all(np.isin(vals, [0, 1])):
        return arr.astype(np.int8)
    if np.all(np.isin(vals, [-1, 1])):
        return ((arr.astype(np.int8) + 1) // 2).astype(np.int8)

    raise ValueError("Assignment must be in {0,1}, {-1,+1}, or bool.")


def hamming_distance(x: Any, y: Any, normalize: bool = True) -> float:
    """Compute Hamming distance between two binary assignments.

    Parameters
    ----------
    x, y:
        Assignments as list/tuple/np.ndarray, either in {0,1} or {-1,+1}.
    normalize:
        If True, return distance / N in [0,1]. Otherwise return integer distance.
    """
    xa = _as_1d_int_array(x)
    ya = _as_1d_int_array(y)
    if xa.shape != ya.shape:
        raise ValueError(f"Shapes mismatch: {xa.shape} vs {ya.shape}")
    d = int(np.count_nonzero(xa != ya))
    return d / float(xa.size) if normalize else float(d)


def _get_energy_trace(run: Mapping[str, Any]) -> np.ndarray:
    """Extract energies trace as a 1D numpy array."""
    if "energies" in run and run["energies"] is not None:
        e = np.asarray(run["energies"], dtype=np.float64)
        if e.ndim != 1:
            raise ValueError("run['energies'] must be 1D")
        return e
    # Fallback: allow storing only best_energy
    if "best_energy" in run:
        return np.asarray([float(run["best_energy"])], dtype=np.float64)
    raise KeyError("Run record must contain either 'energies' or 'best_energy'.")


def _get_time(run: Mapping[str, Any]) -> int:
    """Extract run time budget in sweeps / steps."""
    if "time" in run and run["time"] is not None:
        return int(run["time"])
    e = _get_energy_trace(run)
    return int(max(0, e.size - 1))


# -----------------------------------------------------------------------------
# Residual energy
# -----------------------------------------------------------------------------


def best_energy_per_run(runs: Sequence[Mapping[str, Any]]) -> np.ndarray:
    """Return an array of per-run minimum energies."""
    best = []
    for r in runs:
        e = _get_energy_trace(r)
        if e.size == 0:
            # Skip empty traces (should not happen, but avoids crashes).
            continue
        best.append(float(np.min(e)))
    return np.asarray(best, dtype=np.float64)


def residual_energy(
    runs: Sequence[Mapping[str, Any]],
    optimum_energy: float = 0.0,
    agg: str = "mean",
) -> float:
    """Compute residual energy for a set of runs.

    residual = agg( min_energy(run) - optimum_energy )

    Parameters
    ----------
    runs:
        List of run records.
    optimum_energy:
        Known optimum energy for the instance. For planted SAT, this is 0.
        For MAX-k-SAT it may be unknown; set to 0 and interpret as "absolute" energy.
    agg:
        Aggregation: "mean" or "median".
    """
    vals = best_energy_per_run(runs) - float(optimum_energy)
    if agg == "mean":
        return float(np.mean(vals))
    if agg == "median":
        return float(np.median(vals))
    raise ValueError("agg must be one of {'mean','median'}")


def residual_energy_curve(
    runs: Sequence[Mapping[str, Any]],
    optimum_energy: float = 0.0,
    agg: str = "median",
    max_time: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Aggregate *best-so-far* energy vs time.

    For each run we compute the trace:
        b_t = min_{s<=t} E_s
    then aggregate b_t across runs.

    Returns
    -------
    times:  array of time indices 0..T
    curve:  aggregated residual energy at each time
    """
    traces: List[np.ndarray] = []
    T = 0
    for r in runs:
        e = _get_energy_trace(r)
        if e.size == 0:
            continue
        b = np.minimum.accumulate(e) - float(optimum_energy)
        traces.append(b)
        T = max(T, b.size - 1)

    if max_time is not None:
        T = min(T, int(max_time))

    # Pad to same length with last value (common in learning curves)
    padded = []
    for b in traces:
        if b.size < (T + 1):
            pad_len = (T + 1) - b.size
            b2 = np.concatenate([b, np.full(pad_len, float(b[-1]), dtype=np.float64)])
        else:
            b2 = b[: T + 1]
        padded.append(b2)

    if len(padded) == 0:
        times = np.arange(T + 1, dtype=np.int32)
        return times, np.full_like(times, fill_value=np.nan, dtype=np.float64)

    A = np.vstack(padded)  # shape (n_runs, T+1)
    if agg == "mean":
        curve = np.mean(A, axis=0)
    elif agg == "median":
        curve = np.median(A, axis=0)
    else:
        raise ValueError("agg must be one of {'mean','median'}")

    times = np.arange(T + 1, dtype=np.int32)
    return times, curve


# -----------------------------------------------------------------------------
# TTS_99
# -----------------------------------------------------------------------------


@dataclass
class TTSResult:
    """Container for a TTS curve point."""

    time_budget: int
    p_success: float
    tts: float
    # Optional CI
    p_low: Optional[float] = None
    p_high: Optional[float] = None
    tts_low: Optional[float] = None
    tts_high: Optional[float] = None


def _wilson_interval(k: int, n: int, z: float = 1.96) -> Tuple[float, float]:
    """Wilson score interval for a Bernoulli proportion."""
    if n <= 0:
        return 0.0, 1.0
    phat = k / n
    denom = 1.0 + (z * z) / n
    center = (phat + (z * z) / (2 * n)) / denom
    half = (z / denom) * math.sqrt((phat * (1 - phat) + (z * z) / (4 * n)) / n)
    return max(0.0, center - half), min(1.0, center + half)


def tts_from_psuccess(
    p_success: float,
    time_budget: float,
    confidence: float = 0.99,
    eps: float = 1e-12,
) -> float:
    """Compute TTS at a given success probability.

    If p_success == 0 => TTS = +inf
    If p_success == 1 => TTS = time_budget
    """
    p_success = float(p_success)
    time_budget = float(time_budget)
    if p_success <= 0.0:
        return float("inf")
    if p_success >= 1.0:
        return time_budget

    # n = ceil( log(1-c) / log(1-p) )
    denom = math.log(max(eps, 1.0 - p_success))
    numer = math.log(max(eps, 1.0 - float(confidence)))
    n_runs = int(math.ceil(numer / denom))
    n_runs = max(1, n_runs)
    return n_runs * time_budget


def estimate_psuccess_at_budget(
    runs: Sequence[Mapping[str, Any]],
    time_budget: int,
    target_energy: float = 0.0,
) -> Tuple[int, int]:
    """Return (k_success, n_total) at a given time budget.

    Success means: min_{t<=budget} E_t <= target_energy.

    Notes
    -----
    - We interpret the trace indices as "sweeps".
    - If a run is shorter than time_budget, we use its full trace.
    """
    # Count only runs that actually have a non-empty trace.
    n_total = 0
    k_success = 0

    for r in runs:
        e = _get_energy_trace(r)
        if e.size == 0:
            continue
        n_total += 1

        t = min(int(time_budget), int(e.size - 1))
        best_until_t = float(np.min(e[: t + 1]))
        if best_until_t <= float(target_energy):
            k_success += 1

    return k_success, n_total


def tts_curve(
    runs: Sequence[Mapping[str, Any]],
    budgets: Sequence[int],
    target_energy: float = 0.0,
    confidence: float = 0.99,
    with_ci: bool = True,
    z: float = 1.96,
) -> List[TTSResult]:
    """Compute a TTS curve over a list of time budgets.

    Parameters
    ----------
    runs:
        Independent runs on the *same instance* (or same distribution).
    budgets:
        Time budgets (sweeps). Each budget produces one TTSResult.
    target_energy:
        Success threshold. For planted SAT: 0.
    confidence:
        Desired overall success probability (default 0.99).
    with_ci:
        If True, compute Wilson CI for p_success and a corresponding band for TTS.
    z:
        z-score for the interval (1.96 ~ 95%).
    """
    out: List[TTSResult] = []
    for b in budgets:
        k_succ, n_total = estimate_psuccess_at_budget(runs, int(b), target_energy=target_energy)
        p_succ = k_succ / n_total if n_total > 0 else 0.0
        tts = tts_from_psuccess(p_succ, time_budget=float(b), confidence=confidence)

        if with_ci:
            p_low, p_high = _wilson_interval(k_succ, n_total, z=z)
            # Conservative: tts_low uses p_high (best case), tts_high uses p_low (worst case)
            tts_low = tts_from_psuccess(p_high, float(b), confidence=confidence)
            tts_high = tts_from_psuccess(p_low, float(b), confidence=confidence)
            out.append(
                TTSResult(
                    time_budget=int(b),
                    p_success=float(p_succ),
                    tts=float(tts),
                    p_low=float(p_low),
                    p_high=float(p_high),
                    tts_low=float(tts_low),
                    tts_high=float(tts_high),
                )
            )
        else:
            out.append(TTSResult(time_budget=int(b), p_success=float(p_succ), tts=float(tts)))

    return out


# -----------------------------------------------------------------------------
# Diversity (MIS approximation)
# -----------------------------------------------------------------------------


def _pairwise_hamming_matrix(solutions: Sequence[Any]) -> np.ndarray:
    """Return pairwise normalized Hamming distances (n x n)."""
    sols = [_as_1d_int_array(s) for s in solutions]
    n = len(sols)
    if n == 0:
        return np.zeros((0, 0), dtype=np.float64)
    N = sols[0].size
    for s in sols:
        if s.size != N:
            raise ValueError("All solutions must have same dimension")

    # Vectorized computation: dist(i,j) = mean(xor)
    X = np.stack(sols, axis=0).astype(np.int8)  # (n, N)
    # Using broadcasting: (n,1,N) != (1,n,N) -> (n,n,N)
    D = (X[:, None, :] != X[None, :, :]).mean(axis=2).astype(np.float64)
    return D


def greedy_independent_set_size(adj: List[List[int]]) -> int:
    """Greedy maximal independent set size for an undirected graph.

    Parameters
    ----------
    adj:
        Adjacency list (0..n-1). Graph is assumed undirected.

    Returns
    -------
    size of a maximal (not maximum) independent set.

    Notes
    -----
    This is a standard heuristic: repeatedly pick a node of smallest degree
    and remove it and its neighbors.

    It is fast and deterministic (given the same adjacency list).
    """
    n = len(adj)
    remaining = set(range(n))
    indep = 0

    # Precompute degrees; update lazily.
    while remaining:
        # Select node with smallest (current) degree among remaining
        u = min(remaining, key=lambda i: sum((v in remaining) for v in adj[i]))
        indep += 1
        # Remove u and its neighbors
        to_remove = {u}
        for v in adj[u]:
            if v in remaining:
                to_remove.add(v)
        remaining.difference_update(to_remove)

    return indep


def diversity_curve(
    solutions: Sequence[Any],
    thresholds: Sequence[float],
    normalize: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute diversity(d) for a grid of Hamming-distance thresholds.

    Parameters
    ----------
    solutions:
        Collection of solutions (bitstrings). Typically: best solutions from many runs.
    thresholds:
        Iterable of d values in [0,1]. Two solutions are connected if dist < d.
    normalize:
        If True, returns diversity values normalized by number of solutions.

    Returns
    -------
    d_grid: (m,) thresholds as numpy array
    div:    (m,) diversity(d) values (MIS approx) possibly normalized
    """
    sols = list(solutions)
    n = len(sols)
    if n == 0:
        d_grid = np.asarray(list(thresholds), dtype=np.float64)
        return d_grid, np.zeros_like(d_grid)

    D = _pairwise_hamming_matrix(sols)

    d_grid = np.asarray(list(thresholds), dtype=np.float64)
    div_vals: List[float] = []
    for d in d_grid:
        # Build adjacency list for edges with dist < d
        adj = [[] for _ in range(n)]
        for i in range(n):
            for j in range(i + 1, n):
                if D[i, j] < float(d):
                    adj[i].append(j)
                    adj[j].append(i)
        mis_size = greedy_independent_set_size(adj)
        div_vals.append(mis_size / n if normalize else float(mis_size))

    return d_grid, np.asarray(div_vals, dtype=np.float64)


def diversity_integral_score(
    solutions: Sequence[Any],
    n_thresholds: int = 21,
    normalize: bool = True,
) -> float:
    """Compute an integral diversity score (area under diversity-vs-threshold curve).

    We use thresholds equally spaced on [0,1]. The returned value is in:
    - [0,1] if normalize=True
    - [0,n_solutions] if normalize=False
    """
    thresholds = np.linspace(0.0, 1.0, int(n_thresholds), dtype=np.float64)
    d_grid, div = diversity_curve(solutions, thresholds=thresholds, normalize=normalize)
    # trapezoidal integration over d
    return float(np.trapezoid(div, d_grid))


# -----------------------------------------------------------------------------
# Convenience aggregators (multi-instance)
# -----------------------------------------------------------------------------


def group_runs_by_instance(records: Sequence[Mapping[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
    """Group run records by instance_id.

    This is optional sugar for your analysis script.

    We look for record['instance_id'].
    """
    grouped: Dict[str, List[Dict[str, Any]]] = {}
    for r in records:
        iid = str(r.get("instance_id", "unknown"))
        grouped.setdefault(iid, []).append(dict(r))
    return grouped


def summarize_residual_energy_over_instances(
    grouped_runs: Mapping[str, Sequence[Mapping[str, Any]]],
    optimum_energy: float = 0.0,
    per_instance_agg: str = "mean",
    across_instances_agg: str = "median",
) -> float:
    """Compute a single scalar residual energy summary over many instances.

    Steps:
      1) For each instance, compute residual_energy(runs_i, agg=per_instance_agg)
      2) Aggregate these instance-level values with across_instances_agg

    This matches common paper protocols (median over instances).
    """
    per_inst = []
    for iid, runs in grouped_runs.items():
        _ = iid  # unused
        per_inst.append(residual_energy(runs, optimum_energy=optimum_energy, agg=per_instance_agg))
    vals = np.asarray(per_inst, dtype=np.float64)
    if vals.size == 0:
        return float("nan")

    if across_instances_agg == "mean":
        return float(np.mean(vals))
    if across_instances_agg == "median":
        return float(np.median(vals))
    raise ValueError("across_instances_agg must be one of {'mean','median'}")