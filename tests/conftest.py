# tests/conftest.py
# tests/conftest.py
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import pytest

from env.sat_env import SATProblem

@pytest.fixture
def rng():
    return np.random.default_rng(0)

@pytest.fixture
def tiny_problem(rng):
    """
    Petit SATProblem jouet.
    On suppose que SATProblem accepte (n_vars, clauses) où clauses est une liste de np.ndarray.
    Adapte si ton format clause est différent.
    """
    n_vars = 8
    # Exemple de clauses (format typique CNF: literals ±(index+1))
    # Si ton SATProblem attend un autre format, modifie ici UNE fois.
    def lit_clause_to_pairs(lits: np.ndarray) -> np.ndarray:
        # lits: array shape (k,) in ±(idx+1)
        pairs = np.zeros((lits.size, 2), dtype=np.int32)
        vars0 = np.abs(lits) - 1          # 0-based var index
        signs = np.sign(lits).astype(np.int32)  # +1 or -1
        pairs[:, 0] = vars0
        pairs[:, 1] = signs
        return pairs

    clauses_lits = [
        np.array([+1, -2, +3], dtype=np.int32),
        np.array([-1, +4, -5], dtype=np.int32),
        np.array([+2, +6, -7], dtype=np.int32),
        np.array([-3, -4, +8], dtype=np.int32),
        np.array([+5, +7, +8], dtype=np.int32),
        np.array([-6, -7, -8], dtype=np.int32),
    ]
    clauses = [lit_clause_to_pairs(c) for c in clauses_lits]

    return SATProblem(n_vars=n_vars, clauses=clauses)

@pytest.fixture
def random_x0(tiny_problem, rng):
    return tiny_problem.random_assignment(rng)

def assert_binary(x):
    x = np.asarray(x)
    assert x.ndim == 1
    assert set(np.unique(x)).issubset({0, 1})