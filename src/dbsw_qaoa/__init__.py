"""DBSW-QAOA public API."""

from .core import evaluate_energy, normalize_qubo
from .io import read_problem
from .solver import solve_tabu_qaoa
from .types import SolveResult, SolverConfig

__all__ = [
    "SolverConfig",
    "SolveResult",
    "evaluate_energy",
    "normalize_qubo",
    "read_problem",
    "solve_tabu_qaoa",
]

__version__ = "0.1.0"
