from pathlib import Path

from dbsw_qaoa.cli import main


def test_cli_solve_exact(tmp_path: Path, capsys) -> None:
    path = tmp_path / "qubo.txt"
    path.write_text("problem_size 2\n0 0 1\n1 1 1\n0 1 2\n", encoding="utf-8")

    exit_code = main(["solve", str(path), "--backend", "exact", "--submatrix", "1", "--seed", "1"])
    output = capsys.readouterr().out

    assert exit_code == 0
    assert '"best_energy"' in output
