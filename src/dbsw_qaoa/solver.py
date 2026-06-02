"""Public Tabu-QAOA solver wrapper."""

from __future__ import annotations

import random
import time
from typing import Mapping

from .core import (
    apply_subproblem_solution,
    evaluate_energy,
    flip_solution_by_index,
    make_subqubo,
    normalize_qubo,
    randomize_solution,
    tabu_search,
    val_index_sort,
)
from .qaoa import make_subproblem_solver
from .types import Qubo, SolveResult, SolverConfig


def _validate_config(config: SolverConfig) -> None:
    if config.iterations < 0:
        raise ValueError("iterations must be non-negative.")
    if config.submatrix <= 0:
        raise ValueError("submatrix must be positive.")
    if config.qlen <= 0:
        raise ValueError("qlen must be positive.")
    if config.model not in {"qubo", "maxcut"}:
        raise ValueError("model must be 'qubo' or 'maxcut'.")


def solve_tabu_qaoa(
    qubo: Mapping[tuple[int, int], float],
    size: int | None = None,
    config: SolverConfig | None = None,
) -> SolveResult:
    """Solve a QUBO using the packaged Tabu-QAOA workflow."""

    config = config or SolverConfig()
    _validate_config(config)
    start = time.perf_counter()

    working_qubo = normalize_qubo(qubo, size)
    if size is None:
        size = max(max(i, j) for i, j in working_qubo) + 1
    if not config.find_max:
        working_qubo = {key: -value for key, value in working_qubo.items()}

    rng = random.Random(config.seed)
    sub_solver = make_subproblem_solver(config.backend, p=config.p, exact_limit=config.exact_limit)

    solution = [0] * size
    tabu_solution = [0] * size
    flip_cost = [0.0] * size
    tabu_k = [0] * size
    index = list(range(size))
    best_solution = [0] * size
    bit_flips = 0
    qaoa_calls = 0
    total_changes = 0
    backend_used = config.backend

    randomize_solution(tabu_solution, rng)
    initial_iter_max = bit_flips + max(400, config.initial_tabu_factor * size)
    energy, solution, index, flip_cost, bit_flips = tabu_search(
        solution,
        tabu_solution,
        working_qubo,
        size,
        flip_cost,
        tabu_k,
        index,
        bit_flips,
        initial_iter_max,
        rng,
        target=config.target,
        target_set=config.target_set,
        find_max=True,
    )

    initial_energy = energy
    best_energy = energy
    best_solution = list(solution)
    repeat_pass = 0
    no_progress = 0
    max_nodes_sub = int(max(config.submatrix + 1, config.submatrix_span * size))

    for _ in range(config.iterations):
        if size > 10 and config.submatrix < size:
            index = val_index_sort(index, flip_cost, rng)
            l_max = min(size - config.submatrix, max_nodes_sub)

            if no_progress % config.progress_check == config.progress_check - 1:
                randomize_solution(solution, rng)
            else:
                for start_index in range(0, l_max, config.submatrix):
                    selected = sorted(index[start_index : start_index + config.submatrix])
                    if len(selected) < config.submatrix:
                        continue
                    sub_qubo = make_subqubo(selected, working_qubo, size, solution)
                    sub_result = sub_solver(sub_qubo, config.submatrix)
                    backend_used = sub_result.backend
                    qaoa_calls += 1
                    change = apply_subproblem_solution(selected, solution, sub_result.solution)
                    total_changes += change
                    if change <= 2:
                        flip_solution_by_index(solution, start_index, index, rng)

        iter_max = bit_flips + config.tabu_pass_factor * size
        index = val_index_sort(index, flip_cost, rng)
        energy, solution, index, flip_cost, bit_flips = tabu_search(
            solution,
            tabu_solution,
            working_qubo,
            size,
            flip_cost,
            tabu_k,
            index,
            bit_flips,
            iter_max,
            rng,
            target=config.target,
            target_set=config.target_set,
            find_max=True,
        )

        if energy > best_energy:
            best_energy = energy
            best_solution = list(solution)
            repeat_pass = 0
        else:
            repeat_pass += 1
            no_progress += 1

        if config.target_set and best_energy >= config.target:
            break
        if repeat_pass >= max(50, config.iterations):
            break

    reported_best = best_energy if config.find_max else -best_energy
    reported_initial = initial_energy if config.find_max else -initial_energy
    runtime = time.perf_counter() - start

    return SolveResult(
        best_energy=reported_best,
        initial_energy=reported_initial,
        solution=best_solution,
        qaoa_calls=qaoa_calls,
        total_changes=total_changes,
        iterations=config.iterations,
        runtime_seconds=runtime,
        backend=backend_used,
        model=config.model,
        find_max=config.find_max,
    )


def score_solution(
    solution: list[int],
    qubo: Mapping[tuple[int, int], float],
    *,
    find_max: bool = True,
) -> float:
    """Evaluate a solution in the caller's original objective direction."""

    energy = evaluate_energy(solution, normalize_qubo(qubo))
    return energy if find_max else -energy
