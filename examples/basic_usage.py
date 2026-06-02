from dbsw_qaoa import SolverConfig, solve_tabu_qaoa


def main() -> None:
    qubo = {
        (0, 0): 1.0,
        (1, 1): 1.0,
        (0, 1): 2.0,
    }
    config = SolverConfig(iterations=1, submatrix=2, backend="exact", seed=3)
    result = solve_tabu_qaoa(qubo, size=2, config=config)

    print(result.to_dict())


if __name__ == "__main__":
    main()
