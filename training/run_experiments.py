"""training/run_experiments.py

Run SA / NMC / RLNMC experiments and write results to JSON.

This script is intentionally *framework-like*:
- it can (optionally) generate benchmark instances on the fly
- it can also load pre-generated instances from JSON
- it supports multiple algorithms and multiple budgets (sweeps)
- it logs *raw per-run data* that downstream scripts (e.g. analyze_results.py) can aggregate

Expected output format (single JSON file)
----------------------------------------
{
  "meta": {...},
  "runs": [
     {
       "algo": "sa"|"nmc"|"rlnmc",
       "instance_type": "uniform"|"scalefree",
       "instance_id": int,
       "seed": int,
       "budget_sweeps": int,
       "best_energy": int,
       "success": bool,
       "best_x": [0,1,...] | null,
       "energies": [e0,e1,...] | null,
       "n_vars": int,
       "n_clauses": int
     },
     ...
  ]
}

Design notes
------------
- We log a *best-so-far* energy trace so residual-energy curves can be reproduced.
- We log solutions (0/1 vectors) when available, so diversity can be computed.

Assumptions about project modules
--------------------------------
This file assumes the following APIs exist:

- data.generate_instances:
    - CNFInstance dataclass with fields (n_vars, k, clauses, planted_assignment)
    - generate_uniform_kcnf(...)
    - generate_scalefree_kcnf(...)
    - load_instances(path)
    - save_instances(instances, path)
    - set_global_seed(seed) -> np.random.Generator

- env.sat_env:
    - SATProblem(n_vars: int, clauses: List[np.ndarray])
    - linear_beta_schedule(beta_i, beta_f, n_sweeps)

- algorithms.simulated_annealing:
    - SimulatedAnnealing(problem: SATProblem, beta_schedule: np.ndarray, seed: int)
      with:
        - run(x0=None) -> SAResult with fields:
            best_energy, best_x, energy_trace, best_trace

- algorithms.nonlocal_mc:
    - NonlocalMonteCarloAnnealing(beta_schedule, beta_nmc, backbone_threshold,
                                 n_nmc_steps, n_cycles, n_sw_per_cycle, seed)
      with:
        - run(problem, x0=None) -> NMCResult with fields:
            best_energy, best_x, energy_trace, best_trace, backbone_sizes, n_nmc_jumps

- algorithms.rl_nmc:
    - RLNMCInference(beta_schedule, n_steps, n_cycles, n_sw_per_cycle,
                    rl_start_idx, include_assignment_in_obs, seed, model_path)
      with:
        - run(problem, x0=None) -> RLNMCResult with fields:
            best_energy, best_x, energy_trace, best_trace, backbone_sizes, n_rl_jumps

This script adapts solver results to a stable JSON schema used by metrics.

CLI examples
------------
Generate + run uniform benchmark:
  python -m training.run_experiments --instance_type uniform --n_vars 250 --k 4 --clause_ratio 9.2 \
    --n_instances 20 --replicas 50 --budgets 200 500 1000 2000 \
    --algos sa nmc --out results/uniform_sa_nmc.json

Load instances + run RLNMC:
  python -m training.run_experiments --instances_path data/instances/scalefree.json \
    --algos rlnmc --rlnmc_model training/checkpoints/best_model.zip \
    --replicas 30 --budgets 500 1000 2000 --out results/scalefree_rlnmc.json
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from data.generate_instances import (
    CNFInstance,
    generate_uniform_kcnf,
    generate_scalefree_kcnf,
    load_instances,
    save_instances,
    set_global_seed,
)
from env.sat_env import SATProblem, linear_beta_schedule

from algorithms.simulated_annealing import SimulatedAnnealing
from algorithms.nonlocal_mc import NonlocalMonteCarloAnnealing
from algorithms.rl_nmc import RLNMCInference


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------


def _ensure_dir(path: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)


def _as_int_list(x: Optional[np.ndarray]) -> Optional[List[int]]:
    if x is None:
        return None
    return [int(v) for v in x.astype(np.int8).tolist()]


def _now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S")


def _success_from_energy(best_e: int, target_energy: int) -> bool:
    return int(best_e) <= int(target_energy)


def _maybe_get_trace(trace: Optional[Sequence[Any]]) -> Optional[List[int]]:
    if trace is None:
        return None
    return [int(v) for v in list(trace)]


# -----------------------------------------------------------------------------
# Solver adapters (single run)
# -----------------------------------------------------------------------------


def _run_sa(
    problem: SATProblem,
    budget_sweeps: int,
    seed: int,
    beta_start: float,
    beta_end: float,
) -> Dict[str, Any]:
    """Run Simulated Annealing for a given sweep budget."""
    beta_schedule = linear_beta_schedule(float(beta_start), float(beta_end), int(budget_sweeps))
    solver = SimulatedAnnealing(problem=problem, beta_schedule=beta_schedule, seed=int(seed))
    res = solver.run(x0=None)
    return {
        "best_energy": int(res.best_energy),
        "best_x": res.best_x.copy(),
        "best_trace": res.best_trace.astype(int).tolist(),
    }


def _run_nmc(
    problem: SATProblem,
    budget_sweeps: int,
    seed: int,
    beta_start: float,
    beta_end: float,
    beta_nmc: float,
    backbone_threshold: float,
    n_nmc_steps: int,
    nmc_cycles: int,
    nmc_sw_per_cycle: int,
) -> Dict[str, Any]:
    """Run NMC (SA + periodic nonlocal jumps) for a given sweep budget."""
    beta_schedule = linear_beta_schedule(float(beta_start), float(beta_end), int(budget_sweeps))
    solver = NonlocalMonteCarloAnnealing(
        beta_schedule=beta_schedule,
        beta_nmc=float(beta_nmc),
        backbone_threshold=float(backbone_threshold),
        n_nmc_steps=int(n_nmc_steps),
        n_cycles=int(nmc_cycles),
        n_sw_per_cycle=int(nmc_sw_per_cycle),
        seed=int(seed),
    )
    res = solver.run(problem=problem, x0=None)
    return {
        "best_energy": int(res.best_energy),
        "best_x": res.best_x.copy(),
        "best_trace": res.best_trace.astype(int).tolist(),
        "backbone_sizes": res.backbone_sizes.astype(int).tolist(),
        "n_nmc_jumps": int(res.n_nmc_jumps),
    }


def _run_rlnmc(
    problem: SATProblem,
    budget_sweeps: int,
    seed: int,
    beta_start: float,
    beta_end: float,
    n_rl_steps: int,
    nmc_cycles: int,
    nmc_sw_per_cycle: int,
    model_path: str,
    include_assignment_in_obs: bool,
    rl_start_idx: int,
    deterministic_policy: bool,
) -> Dict[str, Any]:
    """Run RLNMC inference for a given sweep budget."""
    beta_schedule = linear_beta_schedule(float(beta_start), float(beta_end), int(budget_sweeps))

    # Note: RLNMCInference currently uses deterministic=False internally for SB3 predict().
    # We keep the flag for CLI/metadata compatibility.
    _ = deterministic_policy

    solver = RLNMCInference(
        beta_schedule=beta_schedule,
        n_steps=int(n_rl_steps),
        n_cycles=int(nmc_cycles),
        n_sw_per_cycle=int(nmc_sw_per_cycle),
        rl_start_idx=int(rl_start_idx),
        include_assignment_in_obs=bool(include_assignment_in_obs),
        seed=int(seed),
        model_path=str(model_path),
    )
    res = solver.run(problem=problem, x0=None)
    return {
        "best_energy": int(res.best_energy),
        "best_x": res.best_x.copy(),
        "best_trace": res.best_trace.astype(int).tolist(),
        "backbone_sizes": res.backbone_sizes.astype(int).tolist(),
        "n_rl_jumps": int(res.n_rl_jumps),
    }


# -----------------------------------------------------------------------------
# Main experiment loop
# -----------------------------------------------------------------------------


def run_experiments(
    instances: List[CNFInstance],
    instance_type: str,
    algos: Sequence[str],
    budgets: Sequence[int],
    replicas: int,
    out_path: str,
    seed: int,
    target_energy: int,
    save_energy_trace: bool,
    # SA params
    beta_start: float,
    beta_end: float,
    # NMC params
    beta_nmc: float,
    backbone_threshold: float,
    n_nmc_steps: int,
    nmc_cycles: int,
    nmc_sw_per_cycle: int,
    # RLNMC params
    rlnmc_model: Optional[str],
    n_rl_steps: int,
    rl_start_frac: float,
    include_assignment_in_obs: bool,
    rlnmc_deterministic: bool,
) -> None:
    """Run a grid of (instance, budget, replica, algo) and write JSON logs."""
    algos = [a.lower() for a in algos]
    for a in algos:
        if a not in {"sa", "nmc", "rlnmc"}:
            raise ValueError(f"Unknown algo '{a}'. Choose from sa/nmc/rlnmc.")

    if "rlnmc" in algos and not rlnmc_model:
        raise ValueError("--rlnmc_model is required when running RLNMC.")

    budgets = sorted(set(int(b) for b in budgets))
    rng = set_global_seed(seed)

    runs: List[Dict[str, Any]] = []

    t0 = time.time()
    for inst_id, cnf in enumerate(instances):
        problem = SATProblem(n_vars=int(cnf.n_vars), clauses=cnf.clauses)

        inst_seed_base = int(rng.integers(0, 2**31 - 1))

        for budget in budgets:
            for rep in range(int(replicas)):
                run_seed = int((inst_seed_base + 1000003 * budget + 97 * rep) % (2**31 - 1))

                for algo in algos:
                    if algo == "sa":
                        out = _run_sa(
                            problem=problem,
                            budget_sweeps=budget,
                            seed=run_seed,
                            beta_start=beta_start,
                            beta_end=beta_end,
                        )
                    elif algo == "nmc":
                        out = _run_nmc(
                            problem=problem,
                            budget_sweeps=budget,
                            seed=run_seed,
                            beta_start=beta_start,
                            beta_end=beta_end,
                            beta_nmc=beta_nmc,
                            backbone_threshold=backbone_threshold,
                            n_nmc_steps=n_nmc_steps,
                            nmc_cycles=nmc_cycles,
                            nmc_sw_per_cycle=nmc_sw_per_cycle,
                        )
                    else:
                        rl_start_idx = int(round(float(rl_start_frac) * float(max(0, budget - 1))))
                        rl_start_idx = int(np.clip(rl_start_idx, 0, max(0, budget - 1)))

                        out = _run_rlnmc(
                            problem=problem,
                            budget_sweeps=budget,
                            seed=run_seed,
                            beta_start=beta_start,
                            beta_end=beta_end,
                            n_rl_steps=n_rl_steps,
                            nmc_cycles=nmc_cycles,
                            nmc_sw_per_cycle=nmc_sw_per_cycle,
                            model_path=str(rlnmc_model),
                            include_assignment_in_obs=bool(include_assignment_in_obs),
                            rl_start_idx=int(rl_start_idx),
                            deterministic_policy=bool(rlnmc_deterministic),
                        )

                    best_e = int(out["best_energy"])
                    best_x = out.get("best_x", None)
                    trace = out.get("best_trace", None)

                    runs.append(
                        {
                            "algo": algo,
                            "instance_type": instance_type,
                            "instance_id": int(inst_id),
                            "seed": int(run_seed),
                            "budget_sweeps": int(budget),
                            "best_energy": int(best_e),
                            "success": bool(_success_from_energy(best_e, target_energy)),
                            "best_x": _as_int_list(best_x),
                            "energies": _maybe_get_trace(trace) if save_energy_trace else None,
                            "n_vars": int(problem.n_vars),
                            "n_clauses": int(problem.n_clauses),
                        }
                    )

    meta: Dict[str, Any] = {
        "created_at": _now_iso(),
        "instance_type": instance_type,
        "n_instances": int(len(instances)),
        "replicas": int(replicas),
        "budgets": budgets,
        "algos": algos,
        "target_energy": int(target_energy),
        "save_energy_trace": bool(save_energy_trace),
        "sa": {"beta_start": float(beta_start), "beta_end": float(beta_end)},
        "nmc": {
            "beta_nmc": float(beta_nmc),
            "backbone_threshold": float(backbone_threshold),
            "n_nmc_steps": int(n_nmc_steps),
            "nmc_cycles": int(nmc_cycles),
            "nmc_sw_per_cycle": int(nmc_sw_per_cycle),
        },
        "rlnmc": {
            "model": None if rlnmc_model is None else str(rlnmc_model),
            "n_rl_steps": int(n_rl_steps),
            "rl_start_frac": float(rl_start_frac),
            "include_assignment_in_obs": bool(include_assignment_in_obs),
            "deterministic": bool(rlnmc_deterministic),
        },
        "wall_time_sec": float(time.time() - t0),
        "seed": int(seed),
    }

    payload = {"meta": meta, "runs": runs}

    _ensure_dir(out_path)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f)

    print(f"Saved results: {out_path}")
    print(f"Runs: {len(runs)}")


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------


def _build_instances_from_args(args: argparse.Namespace) -> Tuple[List[CNFInstance], str]:
    """Return (instances, instance_type) based on args."""
    if args.instances_path:
        instances = load_instances(args.instances_path)
        instance_type = args.instance_type
        if instance_type is None:
            name = Path(args.instances_path).name.lower()
            if "scale" in name or "scalefree" in name:
                instance_type = "scalefree"
            elif "uniform" in name:
                instance_type = "uniform"
            else:
                instance_type = "uniform"
        return instances, instance_type

    if args.instance_type is None:
        raise ValueError("Either --instances_path or --instance_type must be provided.")

    instance_type = args.instance_type.lower()
    if instance_type not in {"uniform", "scalefree"}:
        raise ValueError("--instance_type must be one of: uniform, scalefree")

    rng = set_global_seed(args.seed)
    n_clauses = int(round(float(args.clause_ratio) * int(args.n_vars)))
    instances: List[CNFInstance] = []

    for _ in range(int(args.n_instances)):
        if instance_type == "uniform":
            inst = generate_uniform_kcnf(
                n_vars=int(args.n_vars),
                n_clauses=n_clauses,
                k=int(args.k),
                planted=bool(args.planted),
                rng=rng,
                avoid_duplicates=not bool(args.allow_duplicates),
            )
        else:
            inst = generate_scalefree_kcnf(
                n_vars=int(args.n_vars),
                n_clauses=n_clauses,
                k=int(args.k),
                planted=bool(args.planted),
                rng=rng,
                alpha=float(args.alpha),
                b=args.b,
                avoid_duplicates=not bool(args.allow_duplicates),
            )
        instances.append(inst)

    if args.save_instances_path:
        _ensure_dir(args.save_instances_path)
        save_instances(instances, args.save_instances_path)
        print(f"Saved generated instances to: {args.save_instances_path}")

    return instances, instance_type


def main() -> None:
    p = argparse.ArgumentParser(description="Run SA/NMC/RLNMC experiments and log results to JSON")

    # Instances: load OR generate
    p.add_argument(
        "--instances_path",
        type=str,
        default=None,
        help="Path to JSON instances (generated by data/generate_instances.py)",
    )
    p.add_argument(
        "--save_instances_path",
        type=str,
        default=None,
        help="If generating instances, optionally save them here (JSON).",
    )

    p.add_argument("--instance_type", type=str, default=None, choices=["uniform", "scalefree"], help="If generating instances, choose type.")
    p.add_argument("--n_vars", type=int, default=250)
    p.add_argument("--k", type=int, default=4)
    p.add_argument("--clause_ratio", type=float, default=9.2)
    p.add_argument("--n_instances", type=int, default=20)
    p.add_argument("--alpha", type=float, default=2.6, help="Scale-free exponent (used if instance_type=scalefree and b not set)")
    p.add_argument("--b", type=float, default=None, help="Scale-free alternative parameterization")
    p.add_argument("--planted", action="store_true", help="Generate planted satisfiable instances (guarantees target_energy=0 is reachable).")
    p.add_argument("--allow_duplicates", action="store_true", help="Allow duplicate clauses in generated instances.")

    # Experiment grid
    p.add_argument("--algos", nargs="+", default=["sa", "nmc", "rlnmc"], help="Subset of: sa nmc rlnmc")
    p.add_argument("--budgets", nargs="+", type=int, default=[200, 500, 1000, 2000], help="Sweep budgets")
    p.add_argument("--replicas", type=int, default=20, help="Independent runs per (instance,budget,algo)")

    # Output
    p.add_argument("--out", type=str, required=True, help="Output JSON file")
    p.add_argument("--seed", type=int, default=12345)
    p.add_argument("--target_energy", type=int, default=0, help="Success threshold: best_energy <= target_energy")
    p.add_argument("--save_energy_trace", action="store_true", help="Store best-energy trace per run (bigger JSON)")

    # SA hyperparams
    p.add_argument("--beta_start", type=float, default=2.0)
    p.add_argument("--beta_end", type=float, default=8.0)

    # NMC / RLNMC hyperparams
    p.add_argument("--beta_nmc", type=float, default=5.0)
    p.add_argument("--backbone_threshold", type=float, default=4.5)
    p.add_argument("--n_nmc_steps", type=int, default=10, help="Number of NMC jumps along the schedule")
    p.add_argument("--nmc_cycles", type=int, default=3, help="Number of cycles inside one (NMC/RLNMC) jump")
    p.add_argument("--nmc_sw_per_cycle", type=int, default=50, help="Total sweeps per cycle inside a nonlocal jump")

    # RLNMC model
    p.add_argument("--rlnmc_model", type=str, default=None, help="Path to SB3 PPO model (.zip) for RLNMC inference")
    p.add_argument("--rlnmc_deterministic", action="store_true", help="Use deterministic policy at test time")
    p.add_argument("--n_rl_steps", type=int, default=10, help="Number of RL-guided nonlocal jumps along the schedule")
    p.add_argument("--rl_start_frac", type=float, default=0.0, help="Fraction of the schedule before RL is allowed to act (warm-up)")
    p.add_argument(
        "--include_assignment_in_obs",
        action="store_true",
        help="Must match the flag used during policy training (RLNMCEnv include_assignment_in_obs).",
    )

    args = p.parse_args()

    instances, instance_type = _build_instances_from_args(args)

    run_experiments(
        instances=instances,
        instance_type=instance_type,
        algos=args.algos,
        budgets=args.budgets,
        replicas=int(args.replicas),
        out_path=args.out,
        seed=int(args.seed),
        target_energy=int(args.target_energy),
        save_energy_trace=bool(args.save_energy_trace),
        beta_start=float(args.beta_start),
        beta_end=float(args.beta_end),
        beta_nmc=float(args.beta_nmc),
        backbone_threshold=float(args.backbone_threshold),
        n_nmc_steps=int(args.n_nmc_steps),
        nmc_cycles=int(args.nmc_cycles),
        nmc_sw_per_cycle=int(args.nmc_sw_per_cycle),
        rlnmc_model=args.rlnmc_model,
        n_rl_steps=int(args.n_rl_steps),
        rl_start_frac=float(args.rl_start_frac),
        include_assignment_in_obs=bool(args.include_assignment_in_obs),
        rlnmc_deterministic=bool(args.rlnmc_deterministic),
    )


if __name__ == "__main__":
    main()