"""Command-line entry point: python -m benchmarks.run.

Compute only. Nothing here imports matplotlib or renders anything; figures
and tables are produced separately from the artifacts this writes.
"""

import argparse
import sys
import warnings
from pathlib import Path

from ._ablation import run_ablation
from ._ablation_cells import timing_units
from ._artifacts import promote
from ._registry import ABLATIONS, PUBLISHED_RANDOM_STATE, SCENARIOS
from ._run import run_scenario

DEFAULT_ROOT = Path("results/runs")
DEFAULT_PUBLISHED_ROOT = Path("results/published")


def _resolve_n_jobs(args: argparse.Namespace) -> int:
    """Scenarios keep one worker by default: the scaling scenarios time their
    fits, and concurrent workers change those timings. An ablation runs on
    every core."""
    if args.n_jobs is not None:
        return int(args.n_jobs)
    return -1 if args.ablation else 1


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m benchmarks.run",
        description="Run CARVE benchmark scenarios and write versioned artifacts.",
    )
    parser.add_argument("--scenario", help="Name of a single scenario to run.")
    parser.add_argument(
        "--ablation",
        help="Name of an ablation to run (see --list). Exclusive with --scenario and --all.",
    )
    parser.add_argument("--all", action="store_true", help="Run every scenario.")
    parser.add_argument("--list", action="store_true", help="List scenario names.")
    parser.add_argument(
        "--scale",
        default=None,
        help="Ablation scale to run; defaults to the ablation's own default scale.",
    )
    parser.add_argument(
        "--timing-batch",
        action="store_true",
        help="With --ablation: run only its fixed timing batch, without resume, "
        "to choose --n-jobs.",
    )
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
    parser.add_argument(
        "--n-jobs",
        type=int,
        default=None,
        help="Defaults to 1 for scenarios and -1 for an ablation.",
    )
    parser.add_argument("--random-state", type=int, default=PUBLISHED_RANDOM_STATE)
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument(
        "--allow-code-change",
        action="store_true",
        help="Resume into a run directory recorded at a different git_sha. Off by "
        "default, because resuming across code versions mixes their checkpoints.",
    )
    parser.add_argument("--verbose", action="count", default=0)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)

    if args.list:
        for name in sorted(SCENARIOS):
            print(name)
        for name in sorted(ABLATIONS):
            print(f"ablation:{name}")
        return 0

    if args.promote:
        try:
            out = promote(Path(args.promote), Path(args.published_root))
        except FileNotFoundError as exc:
            print(str(exc), file=sys.stderr)
            return 1
        print(f"Promoted to {out}")
        return 0

    if args.ablation:
        if args.scenario or args.all:
            print("--ablation is exclusive with --scenario and --all.", file=sys.stderr)
            return 2
        if args.ablation not in ABLATIONS:
            print(
                f"Unknown ablation {args.ablation!r}. Valid names: {sorted(ABLATIONS)}.",
                file=sys.stderr,
            )
            return 2
        ablation = ABLATIONS[args.ablation]
        scale = args.scale if args.scale is not None else ablation.default_scale
        units = timing_units(ablation, scale) if args.timing_batch else None
        rd = run_ablation(
            ablation,
            scale=scale,
            root=Path(args.root),
            n_jobs=_resolve_n_jobs(args),
            resume=(not args.no_resume) and not args.timing_batch,
            verbose=args.verbose,
            units=units,
        )
        print(f"{args.ablation} ({scale}): {rd}")
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
            "Nothing to do. Pass --scenario, --all, --ablation, --list, or --promote.",
            file=sys.stderr,
        )
        return 2

    for name in names:
        rd = run_scenario(
            SCENARIOS[name],
            root=Path(args.root),
            n_jobs=_resolve_n_jobs(args),
            random_state=args.random_state,
            n_seeds=args.n_seeds,
            n_resamples=args.n_resamples,
            resume=not args.no_resume,
            allow_code_change=args.allow_code_change,
            verbose=args.verbose,
        )
        print(f"{name}: {rd}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
