"""Cross-language equivalence driver for the CARVE R port.

Runs the Python `carve` package on a small set of fixed-seed datasets
and emits per-configuration metrics plus selection results to CSV
files in a target directory. The R vignette `cross-validation.Rmd`
reads those CSVs and compares them against the R side under the
documented tolerances.

This script is also runnable standalone:

    python -m carve.sim ...        # not used here
    python inst/python/cross_validation.py [output-dir]

The script is intentionally dependency-light: only `numpy`, `pandas`,
`scikit-learn`, and the Python `carve` package are required.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.datasets import make_blobs, make_moons


def make_dataset(name: str, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Reproduce the four toy datasets used in the R vignette.

    The R vignette generates the *same* datasets with base-R RNG
    using the recipe documented in the vignette, so do not change
    any of the magic numbers below without updating the R side.
    """
    if name == "blobs_well_separated":
        return make_blobs(
            n_samples=300,
            n_features=10,
            centers=3,
            cluster_std=1.0,
            random_state=seed,
        )
    if name == "blobs_ambiguous":
        return make_blobs(
            n_samples=300,
            n_features=10,
            centers=3,
            cluster_std=4.0,
            random_state=seed,
        )
    if name == "two_moons":
        return make_moons(n_samples=300, noise=0.08, random_state=seed)
    raise ValueError(f"Unknown dataset name: {name}")


def fit_one(X: np.ndarray, *, n_resamples: int, random_state: int) -> "CARVE":  # type: ignore[name-defined]
    from carve import CARVE

    carve = CARVE(
        n_clusters=list(range(2, 7)),
        n_resamples=n_resamples,
        subsample_ratio=0.8,
        estimator_param_grids="light",
        random_state=random_state,
        n_jobs=1,
        verbose=0,
    )
    carve.fit(X)
    return carve


def metrics_table(carve) -> pd.DataFrame:
    cols = [
        "estimator",
        "n_clusters",
        "ari_stability",
        "consensus_pac_stability",
        "consensus_gini_stability",
        "consensus_ce_stability",
        "ari_generalizability",
        "accuracy_generalizability",
    ]
    df = carve.estimator_results_.copy()
    keep = [c for c in cols if c in df.columns]
    return df[keep]


def selection_table(carve, dataset: str) -> pd.DataFrame:
    rows = []
    for measure in ("stability", "generalizability"):
        for rule in ("max", "1se"):
            try:
                k = int(carve.get_k(measure=measure, rule=rule))
            except Exception:  # noqa: BLE001
                k = -1
            rows.append(
                {"dataset": dataset, "measure": measure, "rule": rule, "k_python": k}
            )
    return pd.DataFrame(rows)


def main(out_dir: Path, n_resamples: int, seeds: list[int]) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    datasets = ("blobs_well_separated", "blobs_ambiguous", "two_moons")
    metrics_rows: list[pd.DataFrame] = []
    selection_rows: list[pd.DataFrame] = []

    for ds in datasets:
        for seed in seeds:
            X, _ = make_dataset(ds, seed=seed)
            carve = fit_one(X, n_resamples=n_resamples, random_state=seed)

            m = metrics_table(carve)
            m.insert(0, "seed", seed)
            m.insert(0, "dataset", ds)
            metrics_rows.append(m)

            selection_rows.append(selection_table(carve, ds).assign(seed=seed))

    metrics = pd.concat(metrics_rows, ignore_index=True)
    selection = pd.concat(selection_rows, ignore_index=True)

    metrics.to_csv(out_dir / "metrics_python.csv", index=False)
    selection.to_csv(out_dir / "selection_python.csv", index=False)

    print(f"Wrote {len(metrics)} metric rows to {out_dir / 'metrics_python.csv'}")
    print(
        f"Wrote {len(selection)} selection rows to {out_dir / 'selection_python.csv'}"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "out_dir",
        nargs="?",
        default=os.environ.get("CARVE_R_CV_OUT", "cross_validation_out"),
        help="Directory to write metrics_python.csv / selection_python.csv",
    )
    parser.add_argument("--n-resamples", type=int, default=30)
    parser.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3, 4, 5])
    args = parser.parse_args()

    sys.exit(main(Path(args.out_dir), n_resamples=args.n_resamples, seeds=args.seeds))
