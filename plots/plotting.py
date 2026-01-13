"""plots/plotting.py

Plotting utilities for SA / NMC / RLNMC experiments.

This module is intentionally lightweight and backend-agnostic:
- It consumes the JSON outputs written by `training/run_experiments.py`.
- It can also consume *aggregated* statistics prepared by analysis scripts.

Compatibility note
------------------
The current experiment pipeline logs runs in a stable schema (see
`training/run_experiments.py`), where each run may contain:
  - "algo": "sa" | "nmc" | "rlnmc"
  - "instance_type": "uniform" | "scalefree" | ...
  - "best_energy": int
  - "best_x": list[int] | None
  - "energies": list[int] | None   (typically a best-so-far trace)

Older logs may use aliases like "best_solution" or "energy_trace"; we
canonicalize these automatically.

Figures supported
-----------------
1) Energy vs sweeps (median + optional percentile band)
2) TTS_99 vs sweeps (with optional Wilson band)
3) Diversity curves D(R) and the diversity integral

Design notes
------------
- Uses matplotlib only.
- No global style is imposed.
- Plotting functions return (fig, ax) so callers can further customize.

"""

from __future__ import annotations

from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import matplotlib.pyplot as plt

from metrics.metrics import diversity_curve, diversity_integral_score, tts_curve


# -----------------------------------------------------------------------------
# Small helpers
# -----------------------------------------------------------------------------


def _ensure_1d(a: np.ndarray) -> np.ndarray:
    a = np.asarray(a)
    return a.reshape(-1)


def _default_label_map() -> Dict[str, str]:
    return {
        "sa": "SA",
        "nmc": "NMC",
        "rlnmc": "RLNMC",
        "rlnmc_total": "RLNMC (total)",
    }


def canonicalize_run_keys(run: Mapping[str, Any]) -> Dict[str, Any]:
    """Normalize common run schemas to canonical keys.

    Canonical keys used by this module:
      - energies: Optional[list]
      - best_energy: Optional[float|int]
      - best_x: Optional[list]

    Accepted aliases:
      - energies <- energy_trace / best_energy_trace / best_trace
      - best_x  <- best_solution
      - best_energy <- best_e
    """
    r = dict(run)

    if r.get("energies", None) is None:
        for k in ("energies", "energy_trace", "best_energy_trace", "best_trace"):
            if k in r and r[k] is not None:
                r["energies"] = r[k]
                break

    if r.get("best_x", None) is None:
        for k in ("best_x", "best_solution", "best_assignment"):
            if k in r and r[k] is not None:
                r["best_x"] = r[k]
                break

    if r.get("best_energy", None) is None:
        for k in ("best_energy", "best_e", "bestE"):
            if k in r and r[k] is not None:
                r["best_energy"] = r[k]
                break

    if "algo" in r and r["algo"] is not None:
        r["algo"] = str(r["algo"]).lower()
    if "instance_type" in r and r["instance_type"] is not None:
        r["instance_type"] = str(r["instance_type"]).lower()

    return r


def group_runs_by_algo(runs: Sequence[Mapping[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
    """Group run dicts by r['algo'] (lowercased)."""
    out: Dict[str, List[Dict[str, Any]]] = {}
    for r0 in runs:
        r = canonicalize_run_keys(r0)
        algo = str(r.get("algo", "unknown")).lower()
        out.setdefault(algo, []).append(r)
    return out


def _best_so_far_trace(energies: np.ndarray) -> np.ndarray:
    if energies.size == 0:
        return energies
    return np.minimum.accumulate(energies)


def aggregate_energy_curves(
    runs: Sequence[Mapping[str, Any]],
    *,
    optimum_energy: float = 0.0,
    max_time: Optional[int] = None,
    percentiles: Tuple[float, float] = (25.0, 75.0),
) -> Dict[str, np.ndarray]:
    """Aggregate residual-energy curves across runs.

    Returns a dict with keys:
      - x: time indices 0..T
      - median: median best-so-far residual energy
      - pXX: optional percentile bands (e.g. p25/p75)

    Notes
    -----
    - This function is robust to runs storing either raw energies or best-so-far
      traces. We always apply a min-accumulate, which is idempotent.
    """
    traces: List[np.ndarray] = []
    T = 0

    for r0 in runs:
        r = canonicalize_run_keys(r0)
        e = r.get("energies", None)
        if e is None:
            be = r.get("best_energy", None)
            if be is None:
                continue
            arr = np.asarray([float(be)], dtype=np.float64)
        else:
            arr = np.asarray(e, dtype=np.float64)
        if arr.ndim != 1:
            arr = arr.reshape(-1)
        if arr.size == 0:
            continue

        bsf = _best_so_far_trace(arr) - float(optimum_energy)
        traces.append(bsf)
        T = max(T, bsf.size - 1)

    if max_time is not None:
        T = min(T, int(max_time))

    if len(traces) == 0:
        x = np.arange(T + 1, dtype=np.int32)
        return {"x": x, "median": np.full_like(x, np.nan, dtype=np.float64)}

    padded: List[np.ndarray] = []
    for b in traces:
        if b.size < (T + 1):
            pad_len = (T + 1) - b.size
            b2 = np.concatenate([b, np.full(pad_len, float(b[-1]), dtype=np.float64)])
        else:
            b2 = b[: T + 1]
        padded.append(b2)

    A = np.vstack(padded)  # (n_runs, T+1)
    x = np.arange(T + 1, dtype=np.int32)

    lo, hi = float(percentiles[0]), float(percentiles[1])
    out: Dict[str, np.ndarray] = {
        "x": x,
        "median": np.median(A, axis=0),
        f"p{int(round(lo))}": np.percentile(A, lo, axis=0),
        f"p{int(round(hi))}": np.percentile(A, hi, axis=0),
    }
    return out


def build_energy_stats(
    runs_by_algo: Mapping[str, Sequence[Mapping[str, Any]]],
    *,
    optimum_energy: float = 0.0,
    max_time: Optional[int] = None,
    percentiles: Tuple[float, float] = (25.0, 75.0),
) -> Dict[str, Dict[str, np.ndarray]]:
    """Convenience: energy_stats dict ready for plot_energy_vs_sweeps."""
    out: Dict[str, Dict[str, np.ndarray]] = {}
    for algo, runs in runs_by_algo.items():
        stats = aggregate_energy_curves(
            runs,
            optimum_energy=float(optimum_energy),
            max_time=max_time,
            percentiles=percentiles,
        )
        # Normalize band keys to p25/p75 for the plotting function
        lo, hi = int(round(percentiles[0])), int(round(percentiles[1]))
        out[algo] = {
            "x": stats["x"],
            "median": stats["median"],
            "p25": stats.get(f"p{lo}", None) if lo == 25 else stats.get(f"p{lo}", None),
            "p75": stats.get(f"p{hi}", None) if hi == 75 else stats.get(f"p{hi}", None),
        }
        # Drop None keys if percentiles aren't present
        out[algo] = {k: v for k, v in out[algo].items() if v is not None}
    return out


def build_tts_stats(
    runs_by_algo: Mapping[str, Sequence[Mapping[str, Any]]],
    *,
    budgets: Sequence[int],
    target_energy: float = 0.0,
    confidence: float = 0.99,
    with_ci: bool = True,
) -> Dict[str, Dict[str, np.ndarray]]:
    """Build tts_stats dict ready for plot_tts_vs_sweeps."""
    out: Dict[str, Dict[str, np.ndarray]] = {}
    for algo, runs in runs_by_algo.items():
        res_list = tts_curve(
            runs,
            budgets=list(budgets),
            target_energy=float(target_energy),
            confidence=float(confidence),
            with_ci=bool(with_ci),
        )
        x = np.asarray([r.time_budget for r in res_list], dtype=np.float64)
        y = np.asarray([r.tts for r in res_list], dtype=np.float64)
        stats: Dict[str, np.ndarray] = {"x": x, "tts": y}
        if with_ci:
            low = np.asarray([np.nan if r.tts_low is None else float(r.tts_low) for r in res_list], dtype=np.float64)
            high = np.asarray([np.nan if r.tts_high is None else float(r.tts_high) for r in res_list], dtype=np.float64)
            if np.any(np.isfinite(low)) and np.any(np.isfinite(high)):
                stats["low"] = low
                stats["high"] = high
        out[algo] = stats
    return out


def build_diversity_stats(
    runs_by_algo: Mapping[str, Sequence[Mapping[str, Any]]],
    *,
    thresholds: Sequence[float],
    only_success_solutions: bool = True,
    target_energy: float = 0.0,
) -> Tuple[Dict[str, Dict[str, np.ndarray]], Dict[str, float]]:
    """Build diversity curve stats and integral scores.

    Returns
    -------
    (diversity_stats, integrals)

    diversity_stats[algo] has keys:
      - R: thresholds
      - D: diversity values

    integrals[algo] is the diversity integral (area under curve).
    """
    d_stats: Dict[str, Dict[str, np.ndarray]] = {}
    integrals: Dict[str, float] = {}

    thr = np.asarray(list(thresholds), dtype=np.float64)

    for algo, runs in runs_by_algo.items():
        sols: List[np.ndarray] = []
        for r0 in runs:
            r = canonicalize_run_keys(r0)
            x = r.get("best_x", None)
            if x is None:
                continue
            if only_success_solutions:
                be = r.get("best_energy", np.inf)
                if float(be) > float(target_energy):
                    continue
            sols.append(np.asarray(x, dtype=np.int8).reshape(-1))

        if len(sols) < 2:
            continue

        R, D = diversity_curve(sols, thresholds=thr, normalize=True)
        d_stats[algo] = {"R": np.asarray(R, dtype=np.float64), "D": np.asarray(D, dtype=np.float64)}
        integrals[algo] = float(diversity_integral_score(sols, n_thresholds=int(len(thr)), normalize=True))

    return d_stats, integrals


# -----------------------------------------------------------------------------
# Energy vs sweeps
# -----------------------------------------------------------------------------


def plot_energy_vs_sweeps(
    energy_stats: Dict[str, Dict[str, np.ndarray]],
    *,
    title: str = "Residual energy vs sweeps",
    xlabel: str = "Sweeps (MCS)",
    ylabel: str = "Residual energy",
    logx: bool = True,
    logy: bool = True,
    show_band: bool = True,
    label_map: Optional[Dict[str, str]] = None,
    ax: Optional[plt.Axes] = None,
) -> Tuple[plt.Figure, plt.Axes]:
    """Plot energy curves for multiple algorithms.

    Parameters
    ----------
    energy_stats:
        Dict algo_name -> dict with keys:
          - "x": sweeps array shape (T,)
          - "median": median energy at each sweep (T,)
          - optionally "p25" and "p75" for an interquartile band
          - optionally "mean" / "std" for mean +/- std

    Returns
    -------
    (fig, ax)
    """
    label_map = _default_label_map() if label_map is None else label_map

    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.figure

    for algo, stats in energy_stats.items():
        x = _ensure_1d(stats["x"])
        y = _ensure_1d(stats["median"])
        ax.plot(x, y, label=label_map.get(algo, algo))

        if show_band and ("p25" in stats) and ("p75" in stats):
            ax.fill_between(x, _ensure_1d(stats["p25"]), _ensure_1d(stats["p75"]), alpha=0.2)
        elif show_band and ("mean" in stats) and ("std" in stats):
            ax.fill_between(
                x,
                _ensure_1d(stats["mean"]) - _ensure_1d(stats["std"]),
                _ensure_1d(stats["mean"]) + _ensure_1d(stats["std"]),
                alpha=0.2,
            )

    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.legend()

    if logx:
        ax.set_xscale("log")
    if logy:
        ax.set_yscale("log")

    ax.grid(True, which="both", linestyle=":", linewidth=0.5)
    return fig, ax


# -----------------------------------------------------------------------------
# TTS vs sweeps
# -----------------------------------------------------------------------------


def plot_tts_vs_sweeps(
    tts_stats: Dict[str, Dict[str, np.ndarray]],
    *,
    title: str = r"$\mathrm{TTS}_{99}$ vs sweeps",
    xlabel: str = "Sweeps per run (MCS)",
    ylabel: str = r"$\mathrm{TTS}_{99}$ (MCS)",
    logx: bool = True,
    logy: bool = True,
    show_band: bool = True,
    label_map: Optional[Dict[str, str]] = None,
    ax: Optional[plt.Axes] = None,
) -> Tuple[plt.Figure, plt.Axes]:
    """Plot TTS_99 curves."""
    label_map = _default_label_map() if label_map is None else label_map

    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.figure

    for algo, stats in tts_stats.items():
        x = _ensure_1d(stats["x"])
        y = _ensure_1d(stats["tts"])
        ax.plot(x, y, label=label_map.get(algo, algo))
        if show_band and ("low" in stats) and ("high" in stats):
            ax.fill_between(x, _ensure_1d(stats["low"]), _ensure_1d(stats["high"]), alpha=0.2)

    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.legend()

    if logx:
        ax.set_xscale("log")
    if logy:
        ax.set_yscale("log")

    ax.grid(True, which="both", linestyle=":", linewidth=0.5)
    return fig, ax


# -----------------------------------------------------------------------------
# Diversity
# -----------------------------------------------------------------------------


def plot_diversity_curve(
    diversity_stats: Dict[str, Dict[str, np.ndarray]],
    *,
    title: str = "Diversity curve D(R)",
    xlabel: str = "R (Hamming distance threshold)",
    ylabel: str = "D(R)",
    show_band: bool = False,
    label_map: Optional[Dict[str, str]] = None,
    ax: Optional[plt.Axes] = None,
) -> Tuple[plt.Figure, plt.Axes]:
    """Plot diversity curves D(R) for multiple algorithms."""
    label_map = _default_label_map() if label_map is None else label_map

    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.figure

    for algo, stats in diversity_stats.items():
        R = _ensure_1d(stats["R"])
        D = _ensure_1d(stats["D"])
        ax.plot(R, D, label=label_map.get(algo, algo))
        if show_band and ("low" in stats) and ("high" in stats):
            ax.fill_between(R, _ensure_1d(stats["low"]), _ensure_1d(stats["high"]), alpha=0.2)

    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.legend()
    ax.grid(True, linestyle=":", linewidth=0.5)
    return fig, ax


def plot_diversity_integral_bar(
    diversity_integrals: Dict[str, float],
    *,
    title: str = "Diversity integral",
    ylabel: str = "Integral D",
    label_map: Optional[Dict[str, str]] = None,
    ax: Optional[plt.Axes] = None,
) -> Tuple[plt.Figure, plt.Axes]:
    """Bar plot for diversity integrals per algorithm."""
    label_map = _default_label_map() if label_map is None else label_map

    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.figure

    algos = list(diversity_integrals.keys())
    vals = [float(diversity_integrals[a]) for a in algos]
    labels = [label_map.get(a, a) for a in algos]

    ax.bar(labels, vals)
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.grid(True, axis="y", linestyle=":", linewidth=0.5)
    return fig, ax


# -----------------------------------------------------------------------------
# CNF visualization
# -----------------------------------------------------------------------------


def cnf_incidence_matrix(n_vars: int, clauses) -> np.ndarray:
    """
    Build a (n_clauses, n_vars) incidence matrix for a CNF instance.

    We assume each clause is a numpy array of shape (k, 2),
    where:
      - col 0: variable index in [0, n_vars-1]
      - col 1: sign in {+1, -1}  (1 = positive literal, -1 = negative literal)

    Returns
    -------
    M : np.ndarray of shape (n_clauses, n_vars) with entries in {-1, 0, +1}
        M[i, j] = +1 if x_j appears positively in clause i
                  -1 if x_j appears negatively in clause i
                   0 otherwise
    """
    n_clauses = len(clauses)
    M = np.zeros((n_clauses, n_vars), dtype=np.int8)

    for ci, cl in enumerate(clauses):
        cl = np.asarray(cl)
        if cl.ndim != 2 or cl.shape[1] != 2:
            raise ValueError(f"Clause {ci} has shape {cl.shape}, expected (k, 2)")
        for var_idx, sign in cl:
            v = int(var_idx)
            if not (0 <= v < n_vars):
                raise ValueError(f"Variable index {v} out of range for n_vars={n_vars}")
            s = int(sign)
            if s not in (-1, 1):
                raise ValueError(f"Unexpected sign {s} in clause {ci}")
            M[ci, v] = s
    return M


def plot_cnf_incidence(
    n_vars: int,
    clauses,
    *,
    title: str = "CNF incidence matrix",
    ax: Optional[plt.Axes] = None,
) -> Tuple[plt.Figure, plt.Axes]:
    """
    Visualize a CNF formula as a clauses × variables incidence matrix.

    Parameters
    ----------
    n_vars : int
        Number of variables in the instance.
    clauses : sequence of np.ndarray
        Each clause must be an array of shape (k, 2) as in `cnf_incidence_matrix`.

    Returns
    -------
    (fig, ax)
    """
    M = cnf_incidence_matrix(n_vars, clauses)

    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.figure

    im = ax.imshow(M, aspect="auto", interpolation="nearest")

    # Build a simple discrete colormap: negative / zero / positive
    # We avoid specifying explicit colors so the global matplotlib style can decide.
    ax.set_xlabel("Variables (0..n_vars-1)")
    ax.set_ylabel("Clauses (0..n_clauses-1)")
    ax.set_title(title)
    ax.grid(False)

    # Add a colorbar for clarity
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    return fig, ax


# -----------------------------------------------------------------------------
# Convenience wrappers (save)
# -----------------------------------------------------------------------------


def save_figure(fig: plt.Figure, out_path: str, dpi: int = 200) -> None:
    """Save a figure with a sensible default (creates parent dirs)."""
    from pathlib import Path

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight", dpi=dpi)


def close_figure(fig: plt.Figure) -> None:
    """Close a matplotlib figure (helps in batch scripts)."""
    plt.close(fig)
