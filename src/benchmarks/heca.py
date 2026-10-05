"""Command-line entry point: python -m benchmarks.heca.

The full-scale hECA run in stages, one SLURM job each: embed, calibrate and
fit, plus status for a fit in progress. Every stage reads and writes one run
directory. Runbook: slurm/heca/README.md. Design:
docs/superpowers/specs/2026-10-01-heca-longleaf-runtime-design.md.
"""

import argparse
import sys
from pathlib import Path

from ._heca_calibration import run_calibrate
from ._heca_stages import (
    StageOutputExists,
    fit_status,
    format_status,
    run_embed,
    run_fit,
)

STAGES: tuple[str, ...] = ("embed", "calibrate", "fit", "status")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m benchmarks.heca",
        description="Run the full-scale hECA case study in stages.",
    )
    parser.add_argument("stage", choices=STAGES)
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument(
        "--force", action="store_true", help="Overwrite this stage's outputs."
    )
    parser.add_argument(
        "--n-jobs",
        type=int,
        default=None,
        help="fit only: worker count. Default: one per physical core, capped "
        "by node memory over calibration's per-worker peak.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        if args.stage == "embed":
            record = run_embed(args.run_dir, force=args.force)
            print(
                f"Embedded {record['n_cells']} cells "
                f"({record['wall_clock_s'] / 3600:.2f} h, cached={record['cached']})."
            )
        elif args.stage == "calibrate":
            record = run_calibrate(args.run_dir, force=args.force)
            if record["grid_error"] is not None:
                print(f"No grid proposed: {record['grid_error']}", file=sys.stderr)
                return 1
            grid = ", ".join(f"{value:g}" for value in record["proposed_grid"])
            print(f"Proposed grid: {grid}")
            print(f"Projected core-hours: {record['projection']['core_hours']:.0f}")
            for n_jobs, hours in record["projection"]["wall_clock_hours"].items():
                print(f"  {n_jobs:>4} workers: {hours:.1f} h")
        elif args.stage == "fit":
            record = run_fit(args.run_dir, n_jobs=args.n_jobs, force=args.force)
            print(
                f"Fit {record['n_cells']} cells on {record['n_jobs']} workers in "
                f"{record['wall_clock_s'] / 3600:.2f} h."
            )
        else:
            print(format_status(fit_status(args.run_dir)))
    except StageOutputExists as error:
        print(error, file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
