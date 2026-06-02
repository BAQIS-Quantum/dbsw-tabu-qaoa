"""Command line interface for DBSW-QAOA."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .io import read_problem, write_result_json
from .solver import solve_tabu_qaoa
from .types import SolverConfig


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="dbsw-qaoa", description="Run the DBSW Tabu-QAOA solver.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    solve = subparsers.add_parser("solve", help="Solve a QUBO or Max-Cut instance.")
    solve.add_argument("input", type=Path, help="Problem file path.")
    solve.add_argument("--format", choices=["qubo", "maxcut", "matrix"], default="qubo", help="Input format.")
    solve.add_argument("--backend", choices=["auto", "exact", "qcover"], default="auto", help="Subproblem backend.")
    solve.add_argument("--iterations", type=int, default=1, help="Number of hybrid Tabu-QAOA passes.")
    solve.add_argument("--submatrix", type=int, default=10, help="Subproblem size.")
    solve.add_argument("--p", type=int, default=1, help="QAOA depth for the qcover backend.")
    solve.add_argument("--minimize", action="store_true", help="Minimize instead of maximize.")
    solve.add_argument("--seed", type=int, default=None, help="Random seed.")
    solve.add_argument("--exact-limit", type=int, default=20, help="Largest subproblem accepted by exact backend.")
    solve.add_argument("--output", type=Path, default=None, help="Optional JSON output path.")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "solve":
        size, qubo = read_problem(args.input, problem_format=args.format)
        config = SolverConfig(
            iterations=args.iterations,
            submatrix=args.submatrix,
            p=args.p,
            backend=args.backend,
            model="maxcut" if args.format == "maxcut" else "qubo",
            find_max=not args.minimize,
            seed=args.seed,
            exact_limit=args.exact_limit,
        )
        result = solve_tabu_qaoa(qubo, size=size, config=config)
        payload = result.to_dict()
        text = json.dumps(payload, indent=2, sort_keys=True)
        print(text)
        if args.output:
            write_result_json(args.output, payload)
        return 0

    parser.error(f"Unsupported command: {args.command}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
