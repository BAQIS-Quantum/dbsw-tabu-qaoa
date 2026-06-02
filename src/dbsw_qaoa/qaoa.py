"""Subproblem solvers used by the Tabu-QAOA loop."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from typing import Mapping, Sequence

from .core import evaluate_energy
from .types import Qubo


class MissingOptionalDependency(RuntimeError):
    """Raised when an optional QAOA simulator dependency is not installed."""


@dataclass(frozen=True)
class SubproblemResult:
    solution: list[int]
    backend: str


def solve_subproblem_exact(sub_qubo: Mapping[tuple[int, int], float], size: int, *, limit: int = 20) -> SubproblemResult:
    """Solve a small sub-QUBO exactly by exhaustive search."""

    if size > limit:
        raise ValueError(
            f"Exact subproblem backend refuses size {size}; set a smaller submatrix "
            f"or install the QAOA simulator extras and use backend='qcover'."
        )

    best_energy = float("-inf")
    best_solution = [0] * size
    for bits in product((0, 1), repeat=size):
        energy = evaluate_energy(bits, sub_qubo)
        if energy > best_energy:
            best_energy = energy
            best_solution = list(bits)
    return SubproblemResult(solution=best_solution, backend="exact")


def solve_subproblem_qcover(sub_qubo: Mapping[tuple[int, int], float], size: int, *, p: int = 1) -> SubproblemResult:
    """Solve a sub-QUBO with the bundled QCover/Qulacs QAOA simulator."""

    try:
        import networkx as nx
        from Qcover_quark.backends import CircuitByQulacs
        from Qcover_quark.core import Qcovermini
        from Qcover_quark.optimizers import COBYLA
    except ImportError as exc:
        raise MissingOptionalDependency(
            "QCover backend requires optional dependencies. Install with "
            "`pip install -e .[qaoa]` or use backend='exact'."
        ) from exc

    graph = nx.Graph()
    for i in range(size):
        graph.add_node(i, weight=sub_qubo.get((i, i), 0.0))
    for (i, j), value in sub_qubo.items():
        if i != j:
            graph.add_edge(i, j, weight=0.5 * value)

    qaoa = Qcovermini(
        graph,
        p=p,
        optimizer=COBYLA(options={"tol": 1e-1, "disp": False}),
        backend=CircuitByQulacs(),
    )
    result = qaoa.run()
    counts = qaoa.backend.get_result_counts(result["Optimal parameter value"])
    if not counts:
        raise RuntimeError("QCover returned no measurement counts.")

    most_common = max(counts, key=counts.get)
    bit_string = bin(most_common)[2:].zfill(size)
    return SubproblemResult(solution=[int(bit) for bit in bit_string], backend="qcover")


def solve_subproblem_auto(
    sub_qubo: Mapping[tuple[int, int], float],
    size: int,
    *,
    p: int = 1,
    exact_limit: int = 20,
) -> SubproblemResult:
    """Prefer QCover/Qulacs when installed, otherwise fall back to exact search."""

    try:
        return solve_subproblem_qcover(sub_qubo, size, p=p)
    except MissingOptionalDependency:
        return solve_subproblem_exact(sub_qubo, size, limit=exact_limit)


def make_subproblem_solver(backend: str, *, p: int, exact_limit: int):
    """Build a callable subproblem solver for the configured backend."""

    if backend == "exact":
        return lambda sub_qubo, size: solve_subproblem_exact(sub_qubo, size, limit=exact_limit)
    if backend == "qcover":
        return lambda sub_qubo, size: solve_subproblem_qcover(sub_qubo, size, p=p)
    if backend == "auto":
        return lambda sub_qubo, size: solve_subproblem_auto(sub_qubo, size, p=p, exact_limit=exact_limit)
    raise ValueError(f"Unsupported backend: {backend}")
