# Input Format

`dbsw-qaoa solve` accepts three input formats.

## QUBO Edge List

Use `--format qubo`.

```text
problem_size 4
0 0 1.0
0 1 2.0
1 1 -1.0
2 3 0.5
```

The optional `problem_size` line gives the number of binary variables. Each
remaining line is:

```text
i j coefficient
```

Variables are zero-indexed. If both `(i, j)` and `(j, i)` are present with the
same coefficient, the duplicate is kept once. If the two directions differ,
their values are summed.

## Dense Matrix

Use `--format matrix`.

```text
1.0 2.0 0.0
0.0 -1.0 0.5
0.0 0.0 2.0
```

Rows may be whitespace-separated or comma-separated. The matrix must be square.

## Max-Cut Edge List

Use `--format maxcut`.

```text
problem_size 3
0 1 1.0
1 2 1.0
0 2 1.0
```

Each non-comment line is an undirected weighted graph edge:

```text
u v weight
```

The loader converts the graph to a QUBO with diagonal node-degree terms and
off-diagonal `-2 * weight` terms.

## Comments

All formats allow comments beginning with `#`.
