from pathlib import Path

from dbsw_qaoa.io import maxcut_edges_to_qubo, read_problem


def test_read_qubo_edge_list(tmp_path: Path) -> None:
    path = tmp_path / "problem.txt"
    path.write_text(
        "\n".join(
            [
                "problem_size 3",
                "0 0 1.0",
                "0 1 2.0",
                "1 0 2.0",
                "2 2 -1.0",
            ]
        ),
        encoding="utf-8",
    )

    size, qubo = read_problem(path)

    assert size == 3
    assert qubo[(0, 0)] == 1.0
    assert qubo[(0, 1)] == 2.0
    assert qubo[(2, 2)] == -1.0


def test_maxcut_edges_to_qubo() -> None:
    size, qubo = maxcut_edges_to_qubo([(0, 1, 1.0), (1, 2, 2.0)], size=3)

    assert size == 3
    assert qubo[(0, 0)] == 1.0
    assert qubo[(1, 1)] == 3.0
    assert qubo[(2, 2)] == 2.0
    assert qubo[(0, 1)] == -2.0
    assert qubo[(1, 2)] == -4.0
