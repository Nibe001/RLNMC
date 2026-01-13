"""
data/generate_instances.py

Dataset generation for uniform random and scale-free (power-law) k-SAT / MAX-k-SAT.

We generate k-CNF instances of the form:
    C1 ∧ C2 ∧ ... ∧ CM
where each clause C is an OR of k literals, and each literal is either x_i or ¬x_i.

Representation (internal)
-------------------------
- We store a clause as a NumPy array of shape (k, 2).
  Each row is (var_index, sign):
    - var_index ∈ {0, ..., N-1}
    - sign ∈ {+1, -1}
      * +1 means the literal is x_i
      * -1 means the literal is ¬x_i

- A full instance is a CNFInstance dataclass:
    n_vars, k, clauses, planted_assignment (optional)

Planted satisfiable mode
------------------------
If planted=True:
- Sample a "planted" assignment x* ∈ {0,1}^N.
- Generate clauses but reject any clause that is UNSAT under x*.
  => guarantees E*(x*) = 0 for the energy E(x) = #violated clauses.

This is very convenient for:
- time-to-solution where success is reaching energy 0,
- reproducibility tests and debugging,
- small-scale reproductions.

Scale-free mode
---------------
We implement a practical scale-free generator using power-law sampling of variable indices.
We provide two equivalent parametrizations:
- alpha (power-law exponent, common in practice)
- b (Ansótegui-style parameter, used in some scale-free SAT literature)
You can choose either; internally we build a probability vector p_i for selecting variables.

CLI usage
---------
Uniform:
  python -m data.generate_instances --type uniform --n_vars 250 --clause_ratio 9.2 --n_instances 100 --k 4 --seed 123 --out data/instances/uniform.json --planted

Scale-free:
  python -m data.generate_instances --type scalefree --n_vars 250 --clause_ratio 9.2 --n_instances 100 --k 4 --alpha 2.6 --seed 123 --out data/instances/scalefree.json --planted
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict, Any
from pathlib import Path
import json
import numpy as np

# Clause type alias: shape (k,2), dtype int32
Clause = np.ndarray


# -----------------------------------------------------------------------------
# Utilities
# -----------------------------------------------------------------------------

def set_global_seed(seed: int) -> np.random.Generator:
    """
    Create and return a NumPy Generator with a fixed seed.
    Use this generator everywhere to keep generation reproducible.
    """
    return np.random.default_rng(int(seed))


def clause_satisfied(x: np.ndarray, clause: Clause) -> bool:
    """
    Check if a clause is satisfied by assignment x.

    clause rows: (var_index, sign)
    - sign=+1 => literal is x[var_index]
    - sign=-1 => literal is ¬x[var_index]
    """
    for (v, s) in clause:
        v = int(v)
        s = int(s)
        if s == 1 and x[v] == 1:
            return True
        if s == -1 and x[v] == 0:
            return True
    return False


def make_clause_key(clause: Clause) -> Tuple[int, ...]:
    """
    Create a hashable key for a clause to avoid duplicates.
    Key is simply flattened integers (v0,s0,v1,s1,...).
    """
    return tuple(int(z) for z in clause.flatten().tolist())


def sample_k_distinct_indices(
    n: int,
    k: int,
    rng: np.random.Generator,
    p: Optional[np.ndarray] = None
) -> np.ndarray:
    """
    Sample k distinct indices from {0,...,n-1}.
    - If p is None: uniform without replacement
    - If p is provided: we sample with replacement and reject duplicates (simple & robust)
      (because numpy choice without replacement with p is available, but we keep this explicit
       and easy to reason about).
    """
    if p is None:
        return rng.choice(n, size=k, replace=False).astype(np.int32)

    chosen = set()
    while len(chosen) < k:
        idx = int(rng.choice(n, p=p))
        chosen.add(idx)
    return np.array(sorted(chosen), dtype=np.int32)


# -----------------------------------------------------------------------------
# Data container
# -----------------------------------------------------------------------------

@dataclass
class CNFInstance:
    """
    A k-CNF instance.

    Attributes
    ----------
    n_vars:
        Number of boolean variables.
    k:
        Clause size.
    clauses:
        List of clauses; each clause is a (k,2) array.
    planted_assignment:
        Optional planted satisfying assignment (0/1 vector length n_vars).
    """
    n_vars: int
    k: int
    clauses: List[Clause]
    planted_assignment: Optional[np.ndarray] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "n_vars": int(self.n_vars),
            "k": int(self.k),
            "clauses": [cl.astype(np.int32).tolist() for cl in self.clauses],
            "planted_assignment": None if self.planted_assignment is None else self.planted_assignment.astype(int).tolist(),
        }

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> "CNFInstance":
        planted = d.get("planted_assignment", None)
        return CNFInstance(
            n_vars=int(d["n_vars"]),
            k=int(d["k"]),
            clauses=[np.array(cl, dtype=np.int32) for cl in d["clauses"]],
            planted_assignment=None if planted is None else np.array(planted, dtype=np.int8),
        )


# -----------------------------------------------------------------------------
# Uniform generator
# -----------------------------------------------------------------------------

def generate_uniform_kcnf(
    n_vars: int,
    n_clauses: int,
    k: int = 4,
    planted: bool = True,
    rng: Optional[np.random.Generator] = None,
    avoid_duplicates: bool = True,
) -> CNFInstance:
    """
    Generate uniform random k-CNF.

    Each clause:
    - choose k distinct variables uniformly
    - choose literal signs uniformly in {+1,-1}
    If planted=True, reject clauses not satisfied by x*.

    Parameters
    ----------
    n_vars, n_clauses, k:
        Instance size parameters.
    planted:
        If True, generate planted satisfiable formula.
    rng:
        NumPy Generator. If None, creates a default one (non-reproducible).
    avoid_duplicates:
        If True, reject repeated clauses.

    Returns
    -------
    CNFInstance
    """
    rng = np.random.default_rng() if rng is None else rng
    n_vars = int(n_vars)
    n_clauses = int(n_clauses)
    k = int(k)

    if planted:
        x_star = rng.integers(0, 2, size=n_vars, dtype=np.int8)
    else:
        x_star = None

    clauses: List[Clause] = []
    seen = set()

    # Safety: rejection can loop too long if constraints are too strict
    max_tries = max(1000, 200 * n_clauses)
    tries = 0

    while len(clauses) < n_clauses:
        tries += 1
        if tries > max_tries:
            raise RuntimeError(
                f"Failed to generate enough clauses ({len(clauses)}/{n_clauses}). "
                f"Try increasing max_tries or relaxing constraints."
            )

        vars_ = sample_k_distinct_indices(n_vars, k, rng, p=None)
        signs = rng.choice(np.array([1, -1], dtype=np.int8), size=k, replace=True)
        clause = np.stack([vars_, signs.astype(np.int32)], axis=1).astype(np.int32)

        if planted and not clause_satisfied(x_star, clause):
            continue

        if avoid_duplicates:
            key = make_clause_key(clause)
            if key in seen:
                continue
            seen.add(key)

        clauses.append(clause)

    return CNFInstance(n_vars=n_vars, k=k, clauses=clauses, planted_assignment=x_star)


# -----------------------------------------------------------------------------
# Scale-free generator
# -----------------------------------------------------------------------------

def powerlaw_probs_alpha(n_vars: int, alpha: float) -> np.ndarray:
    """
    Build probability vector p_i for i=0..n_vars-1 following a power law.

    A simple discrete form:
        p_i ∝ (i+1)^(-1/(alpha-1))

    This yields a heavy-tailed selection distribution (hubs).
    """
    n_vars = int(n_vars)
    alpha = float(alpha)
    if alpha <= 1.0:
        raise ValueError("alpha must be > 1 for a valid power-law.")
    i = np.arange(1, n_vars + 1, dtype=np.float64)
    w = i ** (-1.0 / (alpha - 1.0))
    w /= w.sum()
    return w.astype(np.float64)


def powerlaw_probs_b(n_vars: int, b: float) -> np.ndarray:
    """
    Alternative parametrization used in some scale-free SAT literature.

    One common form (consistent with power-law variable frequency models) is:
        p_i ∝ (N/(i+1))^(1/(b-1))

    where b>2 typically yields a scale-free-like distribution.
    """
    n_vars = int(n_vars)
    b = float(b)
    if b <= 1.0:
        raise ValueError("b must be > 1")
    i = np.arange(1, n_vars + 1, dtype=np.float64)
    w = (n_vars / i) ** (1.0 / (b - 1.0))
    w /= w.sum()
    return w.astype(np.float64)


def generate_scalefree_kcnf(
    n_vars: int,
    n_clauses: int,
    k: int = 4,
    planted: bool = True,
    rng: Optional[np.random.Generator] = None,
    alpha: float = 2.6,
    b: Optional[float] = None,
    avoid_duplicates: bool = True,
) -> CNFInstance:
    """
    Generate a scale-free random k-CNF.

    Variables are sampled from a power-law distribution (hubs),
    then k distinct variables are used per clause.

    Parameters
    ----------
    n_vars, n_clauses, k:
        Instance size parameters.
    planted:
        If True, planted satisfiable via rejection of UNSAT clauses under x*.
    alpha:
        Power-law exponent parameter (used if b is None).
    b:
        Optional alternative parameterization. If provided, overrides alpha.
    avoid_duplicates:
        If True, reject repeated clauses.

    Returns
    -------
    CNFInstance
    """
    rng = np.random.default_rng() if rng is None else rng
    n_vars = int(n_vars)
    n_clauses = int(n_clauses)
    k = int(k)

    if planted:
        x_star = rng.integers(0, 2, size=n_vars, dtype=np.int8)
    else:
        x_star = None

    p = powerlaw_probs_b(n_vars, b) if b is not None else powerlaw_probs_alpha(n_vars, alpha)

    clauses: List[Clause] = []
    seen = set()

    max_tries = max(2000, 500 * n_clauses)
    tries = 0

    while len(clauses) < n_clauses:
        tries += 1
        if tries > max_tries:
            raise RuntimeError(
                f"Failed to generate enough scale-free clauses ({len(clauses)}/{n_clauses}). "
                f"Try increasing max_tries or adjusting alpha/b."
            )

        vars_ = sample_k_distinct_indices(n_vars, k, rng, p=p)
        signs = rng.choice(np.array([1, -1], dtype=np.int8), size=k, replace=True)
        clause = np.stack([vars_, signs.astype(np.int32)], axis=1).astype(np.int32)

        if planted and not clause_satisfied(x_star, clause):
            continue

        if avoid_duplicates:
            key = make_clause_key(clause)
            if key in seen:
                continue
            seen.add(key)

        clauses.append(clause)

    return CNFInstance(n_vars=n_vars, k=k, clauses=clauses, planted_assignment=x_star)


# -----------------------------------------------------------------------------
# Save / load (JSON)
# -----------------------------------------------------------------------------

def save_instances(instances: List[CNFInstance], out_path: str) -> None:
    """
    Save a list of CNFInstance objects to JSON.
    """
    out_path = str(out_path)
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    payload = [ins.to_dict() for ins in instances]
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f)


def load_instances(path: str) -> List[CNFInstance]:
    """
    Load instances saved by save_instances().
    """
    with open(path, "r", encoding="utf-8") as f:
        arr = json.load(f)
    return [CNFInstance.from_dict(d) for d in arr]


# -----------------------------------------------------------------------------
# Optional: DIMACS export (CNF)
# -----------------------------------------------------------------------------

def write_dimacs(instance: CNFInstance, out_path: str) -> None:
    """
    Write a CNFInstance to DIMACS CNF format.

    DIMACS uses variables numbered from 1..N and literals as:
      +i for x_i, -i for ¬x_i
    """
    out_path = str(out_path)
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(f"p cnf {instance.n_vars} {len(instance.clauses)}\n")
        for cl in instance.clauses:
            lits = []
            for (v, s) in cl:
                v = int(v) + 1
                s = int(s)
                lit = v if s == 1 else -v
                lits.append(str(lit))
            f.write(" ".join(lits) + " 0\n")


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------

def main() -> None:
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("--type", choices=["uniform", "scalefree"], required=True)
    p.add_argument("--n_vars", type=int, default=250)
    p.add_argument("--k", type=int, default=4)
    p.add_argument("--clause_ratio", type=float, default=9.2)
    p.add_argument("--n_instances", type=int, default=50)
    p.add_argument("--seed", type=int, default=12345)
    p.add_argument("--out", type=str, default="data/instances/instances.json")

    # scale-free params
    p.add_argument("--alpha", type=float, default=2.6, help="Power-law exponent (used if --b is not set).")
    p.add_argument("--b", type=float, default=None, help="Alternative scale-free parameterization (overrides alpha).")

    # options
    p.add_argument("--planted", action="store_true", help="Use planted satisfiable generation.")
    p.add_argument("--allow_duplicates", action="store_true", help="Allow duplicate clauses.")
    p.add_argument("--dimacs_out", type=str, default=None, help="Optional DIMACS output file.")

    args = p.parse_args()

    rng = set_global_seed(args.seed)
    n_clauses = int(round(args.clause_ratio * args.n_vars))
    avoid_duplicates = not bool(args.allow_duplicates)

    insts: List[CNFInstance] = []
    for _ in range(args.n_instances):
        if args.type == "uniform":
            inst = generate_uniform_kcnf(
                n_vars=args.n_vars,
                n_clauses=n_clauses,
                k=args.k,
                planted=args.planted,
                rng=rng + _,
                avoid_duplicates=avoid_duplicates,
            )
        else:
            inst = generate_scalefree_kcnf(
                n_vars=args.n_vars,
                n_clauses=n_clauses,
                k=args.k,
                planted=args.planted,
                rng=rng + _,
                alpha=args.alpha,
                b=args.b,
                avoid_duplicates=avoid_duplicates,
            )
        insts.append(inst)

    save_instances(insts, args.out)
    print(f"Saved {len(insts)} instances to {args.out}")

    if args.dimacs_out is not None:
        # write only the first instance to DIMACS (common practice)
        write_dimacs(insts[0], args.dimacs_out)
        print(f"Wrote first instance to DIMACS: {args.dimacs_out}")


if __name__ == "__main__":
    main()