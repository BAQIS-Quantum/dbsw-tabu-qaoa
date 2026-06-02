"""Core Tabu search primitives used by the DBSW-QAOA wrapper."""

from __future__ import annotations

import random
from typing import Callable, Iterable, Mapping, Sequence

from .types import Qubo

BIG_NEGATIVE_FLOAT = -1.225e308


def infer_size(qubo: Mapping[tuple[int, int], float]) -> int:
    """Infer the number of binary variables in a QUBO dictionary."""

    if not qubo:
        raise ValueError("QUBO is empty.")
    return max(max(i, j) for i, j in qubo) + 1


def normalize_qubo(
    qubo: Mapping[tuple[int, int], float],
    size: int | None = None,
    *,
    duplicate_tolerance: float = 1e-12,
) -> Qubo:
    """Return an upper-triangular QUBO dictionary.

    Symmetric duplicate entries with the same value are kept once. If the two
    directions disagree, their values are summed, which matches x^T Q x input.
    """

    if size is None:
        size = infer_size(qubo)

    grouped: dict[tuple[int, int], dict[tuple[int, int], float]] = {}
    for (raw_i, raw_j), raw_value in qubo.items():
        i, j = int(raw_i), int(raw_j)
        if i < 0 or j < 0 or i >= size or j >= size:
            raise ValueError(f"QUBO key {(raw_i, raw_j)} is outside size {size}.")
        value = float(raw_value)
        if i <= j:
            key = (i, j)
        else:
            key = (j, i)
        grouped.setdefault(key, {})[(i, j)] = value

    normalized: Qubo = {}
    for key, entries in grouped.items():
        if key[0] == key[1]:
            normalized[key] = sum(entries.values())
            continue

        values = list(entries.values())
        if len(values) == 1:
            normalized[key] = values[0]
        elif max(values) - min(values) <= duplicate_tolerance:
            normalized[key] = entries.get(key, values[0])
        else:
            normalized[key] = sum(values)

    return {key: value for key, value in normalized.items() if value != 0.0}


def evaluate_energy(solution: Sequence[int], qubo: Mapping[tuple[int, int], float]) -> float:
    """Evaluate the QUBO energy for a binary solution."""

    energy = 0.0
    for (i, j), value in qubo.items():
        if solution[i] and solution[j]:
            energy += value
    return energy


def make_subqubo(
    selected: Sequence[int],
    qubo: Mapping[tuple[int, int], float],
    size: int,
    solution: Sequence[int],
) -> Qubo:
    """Clamp variables outside ``selected`` and return the induced sub-QUBO."""

    selected = sorted(selected)
    selected_set = set(selected)
    local_index = {global_index: local for local, global_index in enumerate(selected)}
    sub_qubo: Qubo = {}

    for (i, j), value in qubo.items():
        i_selected = i in selected_set
        j_selected = j in selected_set
        if i_selected and j_selected:
            local_i = local_index[i]
            local_j = local_index[j]
            if local_i <= local_j:
                key = (local_i, local_j)
            else:
                key = (local_j, local_i)
            sub_qubo[key] = sub_qubo.get(key, 0.0) + value
        elif i_selected and solution[j]:
            local_i = local_index[i]
            sub_qubo[(local_i, local_i)] = sub_qubo.get((local_i, local_i), 0.0) + value
        elif j_selected and solution[i]:
            local_j = local_index[j]
            sub_qubo[(local_j, local_j)] = sub_qubo.get((local_j, local_j), 0.0) + value

    sub_size = len(selected)
    for i in range(sub_size):
        sub_qubo.setdefault((i, i), 0.0)
    return sub_qubo


def randomize_solution(solution: list[int], rng: random.Random) -> list[int]:
    for i in range(len(solution)):
        solution[i] = rng.randint(0, 1)
    return solution


def flip_solution_by_index(solution: list[int], nbits: int, indices: Sequence[int], rng: random.Random) -> list[int]:
    for i in range(min(nbits, len(indices))):
        bit = indices[i]
        if solution[bit] == 1 and rng.randint(0, 1) == 1:
            solution[bit] = 0
        else:
            solution[bit] = 1
    return solution


def evaluate_with_flip_cost(
    solution: list[int],
    size: int,
    qubo: Mapping[tuple[int, int], float],
    flip_cost: list[float],
) -> tuple[float, list[int], list[float]]:
    energy = evaluate_energy(solution, qubo)
    for bit in range(size):
        solution[bit] = 1 - solution[bit]
        flip_cost[bit] = evaluate_energy(solution, qubo) - energy
        solution[bit] = 1 - solution[bit]
    return energy, solution, flip_cost


def evaluate_1bit(
    old_energy: float,
    bit: int,
    solution: list[int],
    size: int,
    qubo: Mapping[tuple[int, int], float],
    flip_cost: list[float],
) -> tuple[float, list[int], list[float]]:
    result = old_energy + flip_cost[bit]
    solution[bit] = 1 - solution[bit]
    for next_bit in range(size):
        solution[next_bit] = 1 - solution[next_bit]
        flip_cost[next_bit] = evaluate_energy(solution, qubo) - result
        solution[next_bit] = 1 - solution[next_bit]
    return result, solution, flip_cost


def shuffle_index(indices: list[int], rng: random.Random) -> list[int]:
    rng.shuffle(indices)
    return indices


def val_index_sort(index: list[int], val: Sequence[float], rng: random.Random, shuffle: bool = True) -> list[int]:
    for i in range(len(index)):
        index[i] = i
    if shuffle:
        shuffle_index(index, rng)
    index.sort(key=lambda item: val[item], reverse=True)
    return index


def local_search_1bit(
    energy: float,
    solution: list[int],
    size: int,
    qubo: Mapping[tuple[int, int], float],
    flip_cost: list[float],
    bit_flips: int,
    rng: random.Random,
) -> tuple[float, list[int], list[float], int]:
    index = list(range(size))
    improve = True
    reverse = True

    while improve:
        improve = False
        if reverse:
            shuffle_index(index, rng)
            iterator: Iterable[int] = reversed(index)
        else:
            iterator = index
        reverse = not reverse

        for bit in iterator:
            bit_flips += 1
            if flip_cost[bit] > 0.0:
                energy, solution, flip_cost = evaluate_1bit(energy, bit, solution, size, qubo, flip_cost)
                improve = True

    return energy, solution, flip_cost, bit_flips


def local_search(
    solution: list[int],
    size: int,
    qubo: Mapping[tuple[int, int], float],
    flip_cost: list[float],
    bit_flips: int,
    rng: random.Random,
) -> tuple[float, list[int], list[float], int]:
    energy, solution, flip_cost = evaluate_with_flip_cost(solution, size, qubo, flip_cost)
    return local_search_1bit(energy, solution, size, qubo, flip_cost, bit_flips, rng)


def tabu_tenure(size: int) -> int:
    if size < 100:
        return 10
    if size < 250:
        return 12
    if size < 500:
        return 13
    if size < 1000:
        return 21
    if size < 2500:
        return 29
    if size < 8000:
        return 34
    return 35


def tabu_search(
    solution: list[int],
    best: list[int],
    qubo: Mapping[tuple[int, int], float],
    size: int,
    flip_cost: list[float],
    tabu_k: list[int],
    index: list[int],
    bit_flips: int,
    iter_max: int,
    rng: random.Random,
    *,
    target: float = 0.0,
    target_set: bool = False,
    find_max: bool = True,
) -> tuple[float, list[int], list[int], list[float], int]:
    tenure = tabu_tenure(size)
    sign = 1 if find_max else -1
    best_energy, solution, flip_cost, bit_flips = local_search(solution, size, qubo, flip_cost, bit_flips, rng)
    index = val_index_sort(index, flip_cost, rng)

    this_iter = max(1, iter_max - bit_flips)
    increase_iter = this_iter / 2
    last_energy = best_energy
    last_bit = 0
    bit_cycle_1 = size
    bit_cycle_2 = size
    bit_cycle = 0
    num_increase = 900

    for i in range(size):
        best[i] = solution[i]
        tabu_k[i] = 0

    reverse = True
    while bit_flips < iter_max:
        neighbour_best = BIG_NEGATIVE_FLOAT
        broke_for_improvement = False
        iterator = reversed(index) if reverse else iter(index)
        reverse = not reverse

        for bit in iterator:
            if tabu_k[bit] != 0:
                continue

            bit_flips += 1
            new_energy = last_energy + flip_cost[bit]
            if new_energy > best_energy and bit != bit_cycle_1:
                broke_for_improvement = True
                last_bit = bit
                delta_energy = new_energy - best_energy
                new_energy, solution, flip_cost = evaluate_1bit(last_energy, bit, solution, size, qubo, flip_cost)
                last_energy, solution, flip_cost, bit_flips = local_search_1bit(
                    new_energy, solution, size, qubo, flip_cost, bit_flips, rng
                )
                index = val_index_sort(index, flip_cost, rng, shuffle=False)
                best_energy = last_energy

                for i in range(size):
                    best[i] = solution[i]

                if target_set and last_energy >= sign * target:
                    break

                if delta_energy <= 1e-8:
                    bit_cycle += 1
                if bit_cycle_2 == bit_cycle_1:
                    bit_cycle += 1
                if bit_cycle_2 == last_bit:
                    bit_cycle += 1
                if bit_cycle > 4:
                    break

                bit_cycle_2 = bit_cycle_1
                bit_cycle_1 = last_bit
                if (iter_max - bit_flips) / this_iter < 0.80 and num_increase > 0:
                    iter_max += increase_iter
                    this_iter += increase_iter
                    num_increase -= 1
                break

            if new_energy > neighbour_best:
                last_bit = bit
                neighbour_best = new_energy

        if target_set and last_energy >= sign * target:
            break
        if bit_cycle > 6:
            break
        if not broke_for_improvement:
            last_energy, solution, flip_cost = evaluate_1bit(last_energy, last_bit, solution, size, qubo, flip_cost)

        for i in range(size):
            tabu_k[i] = max(0, tabu_k[i] - 1)
        tabu_k[last_bit] = tenure + 1 if solution[size - 1] == 0 else max(1, tenure - 1)

    for i in range(size):
        solution[i] = best[i]
    final_energy, solution, flip_cost = evaluate_with_flip_cost(solution, size, qubo, flip_cost)
    index = val_index_sort(index, flip_cost, rng)
    return final_energy, solution, index, flip_cost, bit_flips


def apply_subproblem_solution(
    selected: Sequence[int],
    solution: list[int],
    sub_solution: Sequence[int],
) -> int:
    change = 0
    for local_bit, global_bit in enumerate(selected):
        new_value = int(sub_solution[local_bit])
        if solution[global_bit] != new_value:
            change += 1
        solution[global_bit] = new_value
    return change


SubproblemSolver = Callable[[Qubo, int], Sequence[int]]
