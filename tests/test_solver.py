from dbsw_qaoa import SolverConfig, evaluate_energy, normalize_qubo, solve_tabu_qaoa
from dbsw_qaoa.qaoa import solve_subproblem_exact


def test_exact_subproblem_solver_finds_best_solution() -> None:
    qubo = normalize_qubo({(0, 0): 1.0, (1, 1): 1.0, (0, 1): 2.0})

    result = solve_subproblem_exact(qubo, 2)

    assert result.solution == [1, 1]
    assert result.backend == "exact"


def test_solve_tabu_qaoa_returns_consistent_energy() -> None:
    qubo = {
        (0, 0): 1.0,
        (1, 1): 1.0,
        (2, 2): -1.0,
        (0, 1): 2.0,
    }
    config = SolverConfig(
        iterations=1,
        submatrix=2,
        backend="exact",
        seed=11,
        initial_tabu_factor=20,
        tabu_pass_factor=10,
    )

    result = solve_tabu_qaoa(qubo, size=3, config=config)

    assert len(result.solution) == 3
    assert result.best_energy == evaluate_energy(result.solution, normalize_qubo(qubo))
    assert result.backend in {"exact", "auto"}
