"""Input helpers for QUBO and Max-Cut problem files."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

from .core import normalize_qubo
from .types import Qubo


def _clean_lines(path: str | Path) -> list[str]:
    lines: list[str] = []
    for raw_line in Path(path).read_text(encoding="utf-8").splitlines():
        line = raw_line.split("#", 1)[0].strip()
        if line:
            lines.append(line)
    return lines


def read_edge_file(path: str | Path) -> tuple[int | None, list[tuple[int, int, float]]]:
    """Read lines of ``i j value`` with optional ``problem_size N``."""

    size: int | None = None
    edges: list[tuple[int, int, float]] = []
    for line in _clean_lines(path):
        tokens = line.replace(",", " ").split()
        if tokens[0].lower() in {"problem_size", "size", "n"}:
            size = int(tokens[1])
            continue
        if len(tokens) != 3:
            raise ValueError(f"Expected three columns in {path!s}: {line!r}")
        i, j = int(tokens[0]), int(tokens[1])
        value = float(tokens[2])
        edges.append((i, j, value))
    return size, edges


def edges_to_qubo(edges: Iterable[tuple[int, int, float]], size: int | None = None) -> tuple[int, Qubo]:
    qubo: Qubo = {}
    max_node = -1
    for i, j, value in edges:
        max_node = max(max_node, i, j)
        qubo[(i, j)] = qubo.get((i, j), 0.0) + float(value)
    if size is None:
        size = max_node + 1
    return size, normalize_qubo(qubo, size)


def maxcut_edges_to_qubo(edges: Iterable[tuple[int, int, float]], size: int | None = None) -> tuple[int, Qubo]:
    qubo: Qubo = {}
    max_node = -1
    for i, j, weight in edges:
        if i == j:
            continue
        max_node = max(max_node, i, j)
        a, b = (i, j) if i < j else (j, i)
        qubo[(a, a)] = qubo.get((a, a), 0.0) + weight
        qubo[(b, b)] = qubo.get((b, b), 0.0) + weight
        qubo[(a, b)] = qubo.get((a, b), 0.0) - 2.0 * weight
    if size is None:
        size = max_node + 1
    return size, normalize_qubo(qubo, size)


def read_matrix(path: str | Path) -> tuple[int, Qubo]:
    rows: list[list[float]] = []
    for line in _clean_lines(path):
        tokens = line.replace(",", " ").split()
        rows.append([float(token) for token in tokens])
    if not rows:
        raise ValueError(f"Matrix file is empty: {path!s}")
    size = len(rows)
    if any(len(row) != size for row in rows):
        raise ValueError("Dense matrix input must be square.")

    qubo: Qubo = {}
    for i, row in enumerate(rows):
        for j, value in enumerate(row):
            if value != 0.0:
                qubo[(i, j)] = value
    return size, normalize_qubo(qubo, size)


def read_problem(path: str | Path, *, problem_format: str = "qubo") -> tuple[int, Qubo]:
    """Read a QUBO problem from disk.

    Supported formats are ``qubo`` for coefficient edge lists, ``maxcut`` for
    weighted graph edge lists, and ``matrix`` for a dense QUBO matrix.
    """

    problem_format = problem_format.lower()
    if problem_format == "matrix":
        return read_matrix(path)
    if problem_format not in {"qubo", "maxcut"}:
        raise ValueError("problem_format must be one of: qubo, maxcut, matrix.")

    size, edges = read_edge_file(path)
    if problem_format == "maxcut":
        return maxcut_edges_to_qubo(edges, size)
    return edges_to_qubo(edges, size)


def write_result_json(path: str | Path, payload: dict) -> None:
    import json

    Path(path).write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
