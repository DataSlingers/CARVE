"""Command-line entry point: python -m benchmarks.run.

Compute only. Nothing here imports matplotlib or renders anything; figures
and tables are produced separately from the artifacts this writes.
"""

import argparse
import sys
import warnings
from pathlib import Path

from ._artifacts import promote
from ._registry import PUBLISHED_RANDOM_STATE, SCENARIOS
from ._run import run_scenario

DEFAULT_ROOT = Path("results/runs")
DEFAULT_PUBLISHED_ROOT = Path("results/published")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m benchmarks.run",
        description="Run CARVE benchmark scenarios and write versioned artifacts.",
    )
    parser.add_argument("--scenario", help="Name of a single scenario to run.")
    parser.add_argument("--all", action="store_true", help="Run every scenario.")
    parser.add_argument("--list", action="store_true", help="List scenario names.")
    parser.add_argument("--promote", help="Publish the run directory at this path.")
    parser.add_argument(
        "--tables",
        help="Write manuscript .tex table fragments to this directory.",
    )
    parser.add_argument("--root", default=str(DEFAULT_ROOT), help="Working run root.")
    parser.add_argument(
        "--published-root",
        default=str(DEFAULT_PUBLISHED_ROOT),
        help="Destination root for promoted runs.",
    )
    parser.add_argument("--n-seeds", type=int, default=None)
    parser.add_argument("--n-resamples", type=int, default=100)
    parser.add_argument("--n-jobs", type=int, default=1)
    parser.add_argument("--random-state", type=int, default=PUBLISHED_RANDOM_STATE)
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument("--verbose", action="count", default=0)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)

    if args.list:
        for name in sorted(SCENARIOS):
            print(name)
        return 0

    if args.promote:
        try:
            out = promote(Path(args.promote), Path(args.published_root))
        except FileNotFoundError as exc:
            print(str(exc), file=sys.stderr)
            return 1
        print(f"Promoted to {out}")
        return 0

    if args.tables:
        from ._artifacts import read_run
        from .tables import write_all_tables

        frames = {}
        for name in sorted(SCENARIOS):
            candidates = sorted(Path(args.root).glob(f"{name}/*/"))
            if not candidates:
                continue
            if len(candidates) > 1:
                warnings.warn(
                    f"{name}: {len(candidates)} run directories found under "
                    f"{args.root!r}; using {candidates[-1]} and ignoring the "
                    "rest.",
                    stacklevel=2,
                )
            frames[name] = read_run(candidates[-1])
        paths = write_all_tables(frames, Path(args.tables))
        for path in paths:
            print(path)
        return 0

    if args.scenario:
        if args.scenario not in SCENARIOS:
            print(
                f"Unknown scenario {args.scenario!r}. "
                f"Valid names: {sorted(SCENARIOS)}.",
                file=sys.stderr,
            )
            return 2
        names = [args.scenario]
    elif args.all:
        names = sorted(SCENARIOS)
    else:
        print(
            "Nothing to do. Pass --scenario, --all, --list, or --promote.",
            file=sys.stderr,
        )
        return 2

    for name in names:
        rd = run_scenario(
            SCENARIOS[name],
            root=Path(args.root),
            n_jobs=args.n_jobs,
            random_state=args.random_state,
            n_seeds=args.n_seeds,
            n_resamples=args.n_resamples,
            resume=not args.no_resume,
            verbose=args.verbose,
        )
        print(f"{name}: {rd}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
