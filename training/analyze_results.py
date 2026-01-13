"""training/analyze_results.py

Analyze experiment logs and generate figures for SA / NMC / RLNMC.

This script is intentionally *framework-like* and conservative about assumptions:
- It accepts JSON logs produced by `training/run_experiments.py` (or similar) for each algorithm.
- It computes the key metrics used in the paper:
    * residual energy (best-so-far energy curve)
    * time-to-solution TTS_99 versus compute budget
    * diversity of solutions (Hamming-distance based, MIS greedy approximation)
- It generates Matplotlib figures into an output directory.

Expected input format (flexible)
-------------------------------
We support two common structures:

(A) A single JSON file containing a list of run dicts:
    [
      {
        "algo": "SA",
        "instance_id": "uf_0001",
        "n_vars": 250,
        "instance_type": "uniform" | "scalefree",
        "seed": 123,
        "energies": [..],              # energy trace after each sweep OR at fixed intervals
        "sweeps": [..] | null,        # optional x-axis for energies (same length)
        "best_energy": 0,
        "best_x": [0,1,0,...],        # optional solution vector
        "success": true | false       # optional; otherwise inferred from best_energy <= target
      }, ...
    ]

(B) A JSON dict with key "runs" containing the list above:
    {"runs": [ ... ]}

You can also pass multiple files, or a folder containing *.json files.

Usage examples
--------------
# Analyze a folder with logs:
python -m training.analyze_results --input results/ --out figures/

# Analyze specific files:
python -m training.analyze_results --input results/sa.json results/nmc.json results/rlnmc.json --out figures/

Notes
-----
- We keep plotting code here to avoid tight coupling with plots/plotting.py while the project is evolving.
- This file relies on `metrics/metrics.py` for the actual metric computations.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import matplotlib.pyplot as plt

from metrics.metrics import (
    residual_energy_curve,
    tts_curve,
    diversity_curve,
    diversity_integral_score,
)


# -----------------------------------------------------------------------------
# IO utilities
# -----------------------------------------------------------------------------


def _load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _normalize_runs(payload: Any, source_name: str) -> List[Dict[str, Any]]:
    """Convert different payload shapes into a list of run dicts."""
    if isinstance(payload, dict) and "runs" in payload:
        runs = payload["runs"]
    else:
        runs = payload

    if not isinstance(runs, list):
        raise ValueError(f"{source_name}: expected a list of runs or a dict with key 'runs'.")

    out: List[Dict[str, Any]] = []
    for i, r in enumerate(runs):
        if not isinstance(r, dict):
            raise ValueError(f"{source_name}: run #{i} is not a dict.")
        rr = dict(r)
        rr.setdefault("source", source_name)
        rr = _canonicalize_run_keys(rr)
        out.append(rr)
    return out


def _canonicalize_run_keys(run: Dict[str, Any]) -> Dict[str, Any]:
    """Normalize common run dict schemas to the canonical keys used by this project.

    Canonical keys expected downstream:
      - energies: list[float|int] or None
      - best_energy: float|int
      - best_x: list[int] (0/1) or None

    We accept several historical aliases:
      - energy_trace / best_energy_trace / best_trace -> energies
      - best_solution -> best_x
      - best_e -> best_energy

    Note: if the provided trace is a *best-so-far* trace (as in our experiments),
    metrics still work because min-accumulate(best_trace) == best_trace.
    """
    r = dict(run)

    # Energies trace
    if r.get("energies", None) is None:
        for k in ("energies", "energy_trace", "best_energy_trace", "best_trace"):
            if k in r and r[k] is not None:
                r["energies"] = r[k]
                break

    # Best energy
    if r.get("best_energy", None) is None:
        for k in ("best_energy", "best_e", "bestE"):
            if k in r and r[k] is not None:
                r["best_energy"] = r[k]
                break

    # Best solution
    if r.get("best_x", None) is None:
        for k in ("best_x", "best_solution", "best_assignment"):
            if k in r and r[k] is not None:
                r["best_x"] = r[k]
                break

    # Normalize algo name a bit for grouping (keep original too)
    if "algo" in r and r["algo"] is not None:
        r["algo"] = str(r["algo"]).lower()

    if "instance_type" in r and r["instance_type"] is not None:
        r["instance_type"] = str(r["instance_type"]).lower()

    return r


def load_runs_from_inputs(inputs: Sequence[str]) -> List[Dict[str, Any]]:
    """Load runs from a list of file/folder inputs."""
    paths: List[Path] = []
    for s in inputs:
        p = Path(s)
        if p.is_dir():
            paths.extend(sorted(p.glob("*.json")))
        else:
            paths.append(p)

    if not paths:
        raise FileNotFoundError("No JSON files found in --input.")

    all_runs: List[Dict[str, Any]] = []
    for p in paths:
        payload = _load_json(p)
        all_runs.extend(_normalize_runs(payload, source_name=p.name))

    return all_runs


# -----------------------------------------------------------------------------
# Run grouping helpers
# -----------------------------------------------------------------------------


def _get_algo_name(run: Dict[str, Any], fallback: str = "UNKNOWN") -> str:
    # common keys used across scripts
    for k in ("algo", "algorithm", "name"):
        if k in run and run[k] is not None:
            return str(run[k])
    return fallback


def _get_instance_type(run: Dict[str, Any]) -> str:
    for k in ("instance_type", "problem_type", "type"):
        if k in run and run[k] is not None:
            return str(run[k])
    return "unknown"


def group_runs(runs: List[Dict[str, Any]]) -> Dict[Tuple[str, str], List[Dict[str, Any]]]:
    """Group by (algo, instance_type)."""
    groups: Dict[Tuple[str, str], List[Dict[str, Any]]] = {}
    for r in runs:
        key = (_get_algo_name(r), _get_instance_type(r))
        groups.setdefault(key, []).append(r)
    return groups


# -----------------------------------------------------------------------------
# Plotting helpers
# -----------------------------------------------------------------------------


def _ensure_outdir(outdir: Path) -> None:
    outdir.mkdir(parents=True, exist_ok=True)


def _savefig(fig: plt.Figure, outpath: Path) -> None:
    outpath.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(outpath, dpi=200)
    plt.close(fig)


def plot_residual_energy(
    grouped: Dict[Tuple[str, str], List[Dict[str, Any]]],
    outdir: Path,
    optimum_energy: float,
    title_prefix: str = "Residual energy",
) -> None:
    """Produce best-so-far residual energy curves (median across runs)."""
    for (algo, inst_type), runs in grouped.items():
        t, curve = residual_energy_curve(
            runs,
            optimum_energy=optimum_energy,
            agg="median",
        )

        fig = plt.figure()
        plt.plot(t, curve)
        plt.xscale("log")
        plt.yscale("log")
        plt.xlabel("Compute budget (sweeps or steps)")
        plt.ylabel("Residual energy (best-so-far)")
        plt.title(f"{title_prefix} — {algo} — {inst_type}")

        _savefig(fig, outdir / f"residual_energy__{inst_type}__{algo}.png")


def plot_tts(
    grouped: Dict[Tuple[str, str], List[Dict[str, Any]]],
    outdir: Path,
    budgets: Sequence[int],
    target_energy: float,
    confidence: float = 0.99,
    with_ci: bool = True,
) -> None:
    """Produce TTS_99 vs budget curves."""
    for (algo, inst_type), runs in grouped.items():
        res_list = tts_curve(
            runs,
            budgets=list(budgets),
            target_energy=target_energy,
            confidence=confidence,
            with_ci=with_ci,
        )

        x_budget = np.asarray([r.time_budget for r in res_list], dtype=np.float64)
        y_tts = np.asarray([r.tts for r in res_list], dtype=np.float64)

        fig = plt.figure()
        plt.plot(x_budget, y_tts, marker="o")

        if with_ci:
            y_low = np.asarray([np.nan if r.tts_low is None else r.tts_low for r in res_list], dtype=np.float64)
            y_high = np.asarray([np.nan if r.tts_high is None else r.tts_high for r in res_list], dtype=np.float64)
            if np.any(np.isfinite(y_low)) and np.any(np.isfinite(y_high)):
                plt.fill_between(x_budget, y_low, y_high, alpha=0.2)

        plt.xscale("log")
        plt.yscale("log")
        plt.xlabel("Budget per run (sweeps)")
        plt.ylabel(f"TTS_{int(confidence*100)}")
        plt.title(f"TTS curve — {algo} — {inst_type}")

        _savefig(fig, outdir / f"tts__{inst_type}__{algo}.png")


def plot_diversity(
    grouped: Dict[Tuple[str, str], List[Dict[str, Any]]],
    outdir: Path,
    thresholds: Sequence[float],
    only_success_solutions: bool = True,
    target_energy: float = 0.0,
) -> None:
    """Plot diversity curve D(R) and report integral score."""
    summary_lines: List[str] = []

    for (algo, inst_type), runs in grouped.items():
        # Collect one representative solution per run.
        sols: List[np.ndarray] = []
        for r in runs:
            x = r.get("best_x", None)
            if x is None:
                continue
            if only_success_solutions:
                be = float(r.get("best_energy", np.inf))
                if be > target_energy:
                    continue
            sols.append(np.asarray(x, dtype=np.int8))

        if len(sols) < 2:
            summary_lines.append(f"{algo}/{inst_type}: not enough solutions for diversity (n={len(sols)}).")
            continue

        d_grid, d_vals = diversity_curve(sols, thresholds=np.asarray(list(thresholds), dtype=np.float64))
        score = diversity_integral_score(sols, n_thresholds=len(thresholds))

        fig = plt.figure()
        plt.plot(d_grid, d_vals, marker="o")
        plt.xlabel("Hamming threshold R")
        plt.ylabel("D(R) (greedy MIS approx)")
        plt.title(f"Diversity — {algo} — {inst_type} (integral={score:.3f})")
        _savefig(fig, outdir / f"diversity_curve__{inst_type}__{algo}.png")

        summary_lines.append(f"{algo}/{inst_type}: diversity_integral={score:.6f} over {len(sols)} solutions")

    # Write a small text summary for convenience
    (outdir / "diversity_summary.txt").write_text("\n".join(summary_lines) + "\n", encoding="utf-8")


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Analyze RLNMC project results and generate figures.")
    p.add_argument(
        "--input",
        nargs="+",
        required=True,
        help="One or more JSON files OR directories containing *.json logs.",
    )
    p.add_argument(
        "--out",
        type=str,
        default="figures",
        help="Output folder for generated figures.",
    )

    # Metrics params
    p.add_argument("--target_energy", type=float, default=0.0, help="Energy threshold for success.")
    p.add_argument("--optimum_energy", type=float, default=0.0, help="Optimum energy for residual energy.")

    # Budgets used for TTS curve
    p.add_argument(
        "--budgets",
        type=int,
        nargs="+",
        default=[200, 500, 1000, 2000, 5000, 10000],
        help="List of per-run budgets (sweeps) for TTS estimation.",
    )
    p.add_argument("--confidence", type=float, default=0.99, help="Confidence level for TTS, e.g. 0.99.")

    # Diversity params
    p.add_argument(
        "--div_thresholds",
        type=float,
        nargs="+",
        default=[0.02, 0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5],
        help="Threshold grid R for diversity curve D(R).",
    )
    p.add_argument(
        "--div_only_success",
        action="store_true",
        help="If set, compute diversity only from successful solutions (best_energy <= target_energy).",
    )

    return p.parse_args()


def main() -> None:
    args = parse_args()
    outdir = Path(args.out)
    _ensure_outdir(outdir)

    runs = load_runs_from_inputs(args.input)
    groups = group_runs(runs)

    # Produce figures
    plot_residual_energy(groups, outdir, optimum_energy=float(args.optimum_energy))
    plot_tts(
        groups,
        outdir,
        budgets=args.budgets,
        target_energy=float(args.target_energy),
        confidence=float(args.confidence),
        with_ci=True,
    )
    plot_diversity(
        groups,
        outdir,
        thresholds=args.div_thresholds,
        only_success_solutions=bool(args.div_only_success),
        target_energy=float(args.target_energy),
    )

    # Small console summary
    print(f"Loaded {len(runs)} runs.")
    print(f"Generated figures into: {outdir.resolve()}")
    print("Groups:")
    for (algo, inst_type), rs in sorted(groups.items()):
        print(f"  - {algo:10s} | {inst_type:10s} | n_runs={len(rs)}")


if __name__ == "__main__":
    main()
