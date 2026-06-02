# dbsw-tabu-qaoa

`dbsw-tabu-qaoa` packages a Tabu-QAOA hybrid solver for binary QUBO and Max-Cut
instances. The project provides a clean Python API, a CLI, examples, tests, and
GitHub CI files so the repository can be pushed directly to GitHub.

The original working files are not modified. This package contains a clean
wrapper and a bundled copy of the QCover/Qulacs simulator components needed for
optional QAOA subproblem solving.

## Install

```bash
git clone https://github.com/your-org/dbsw-tabu-qaoa.git
cd dbsw-tabu-qaoa
python -m pip install -e ".[dev]"
```

For QCover/Qulacs-backed QAOA subproblems:

```bash
python -m pip install -e ".[qaoa]"
```

Without the optional simulator extras, the default `auto` backend falls back to
an exact subproblem solver for small submatrices.

## CLI

Solve a QUBO edge-list file:

```bash
dbsw-qaoa solve examples/qubo_4.txt --format qubo --backend exact --iterations 2 --submatrix 2 --seed 7
```

Solve a Max-Cut weighted edge-list file:

```bash
dbsw-qaoa solve examples/maxcut_triangle.txt --format maxcut --backend exact --iterations 2 --submatrix 2
```

The command prints JSON:

```json
{
  "backend": "exact",
  "best_energy": 4.0,
  "solution": [1, 1, 0, 0]
}
```

## Python API

```python
from dbsw_qaoa import SolverConfig, solve_tabu_qaoa

qubo = {
    (0, 0): 1.0,
    (1, 1): 1.0,
    (0, 1): 2.0,
}

config = SolverConfig(iterations=1, submatrix=2, backend="exact", seed=1)
result = solve_tabu_qaoa(qubo, size=2, config=config)

print(result.best_energy)
print(result.solution)
```

## Input Formats

QUBO edge list:

```text
problem_size 4
0 0 1.0
0 1 2.0
1 1 -1.0
```

Dense matrix:

```text
1.0 2.0 0.0
0.0 -1.0 0.5
0.0 0.0 2.0
```

Max-Cut edge list:

```text
problem_size 3
0 1 1.0
1 2 1.0
0 2 1.0
```

More details are in `docs/input-format.md`.

## Development

```bash
python -m pip install -e ".[dev]"
pytest
python -m dbsw_qaoa solve examples/qubo_4.txt --backend exact --submatrix 2
```

## Project Name

The GitHub repository name is `dbsw-tabu-qaoa`. The installable Python package
and command remain `dbsw-qaoa` for short, convenient usage.

## Notes

- No API tokens are stored in this repository.
- The base package has no mandatory third-party dependencies.
- The optional QCover/Qulacs simulator path is enabled with `.[qaoa]`.
