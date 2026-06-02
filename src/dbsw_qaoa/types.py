"""Shared data structures for DBSW-QAOA."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Dict, List, Literal, Tuple

QuboKey = Tuple[int, int]
Qubo = Dict[QuboKey, float]
ModelName = Literal["qubo", "maxcut"]
BackendName = Literal["auto", "exact", "qcover"]


@dataclass(frozen=True)
class SolverConfig:
    """Configuration for the Tabu-QAOA hybrid solver."""

    iterations: int = 1
    submatrix: int = 10
    p: int = 1
    backend: BackendName = "auto"
    model: ModelName = "qubo"
    find_max: bool = True
    qlen: int = 20
    target: float = 0.0
    target_set: bool = False
    seed: int | None = None
    exact_limit: int = 20
    progress_check: int = 12
    submatrix_span: float = 0.214
    initial_tabu_factor: int = 6500
    tabu_pass_factor: int = 1700


@dataclass(frozen=True)
class SolveResult:
    """Result returned by :func:`solve_tabu_qaoa`."""

    best_energy: float
    initial_energy: float
    solution: List[int]
    qaoa_calls: int
    total_changes: int
    iterations: int
    runtime_seconds: float
    backend: str
    model: str
    find_max: bool

    def to_dict(self) -> dict:
        """Return a JSON-serializable representation."""

        return asdict(self)
