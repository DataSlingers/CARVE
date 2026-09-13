"""Regression gate for the non-randomized path.

Captures what CARVE computes on a fixed dataset under a fixed seed with
randomize_preprocessing=False, so a change to the randomized path can be
checked against it. tests/test_api.py::TestNonRandomizedRegressionGate
refits the same model and compares.

The fixture was written from commit 1dea71d, before the 2026-09-13
randomized-preprocessing redesign. Regenerate it only on purpose, from a
commit whose non-randomized numbers are known good:

    .venv/bin/python -m tests.fixtures.nonrandomized_gate
"""

from pathlib import Path

import numpy as np
from sklearn.cluster import AgglomerativeClustering, KMeans

from carve import CARVE

GATE_PATH = Path(__file__).with_name("nonrandomized_gate.npz")

GATE_KEYS = (
    "results",
    "consensus",
    "consensus_generalizability",
    "gini",
    "ce",
    "generalizability",
)


def gate_dataset() -> np.ndarray:
    """Three Gaussian clusters with boundary points, 60 x 5.

    Offsets of 3 rather than 6 leave points near the boundaries, so a change
    in the feature space the clusterers see moves at least some labels. A
    fully separated dataset gives the same partition under almost any
    preprocessing and could not detect the mutation the gate exists for.
    """
    rng = np.random.RandomState(7)
    return np.vstack(
        [
            rng.randn(20, 5) + [3, 0, 0, 0, 0],
            rng.randn(20, 5) + [0, 3, 0, 0, 0],
            rng.randn(20, 5) + [0, 0, 3, 0, 0],
        ]
    )


def gate_model(**kwargs) -> CARVE:
    """The gate's model. kwargs override constructor arguments."""
    options = dict(
        n_clusters=np.array([2, 3, 4]),
        n_resamples=6,
        subsample_ratio=0.618,
        estimator_param_grids=[
            (KMeans, {"n_clusters": [2, 3, 4], "n_init": [3]}),
            (AgglomerativeClustering, {"n_clusters": [2, 3, 4], "linkage": ["ward"]}),
        ],
        n_jobs=1,
        random_state=0,
        verbose=0,
    )
    options.update(kwargs)
    return CARVE(**options)


def gate_arrays(model: CARVE) -> dict[str, np.ndarray]:
    """Every number the gate compares, keyed for np.savez."""
    df = model.estimator_results_
    numeric = df.select_dtypes(include="number")
    return {
        "columns": np.asarray(list(numeric.columns)),
        "method_labels": np.asarray(df["method_label"].astype(str).tolist()),
        "results": numeric.to_numpy(dtype=float),
        "consensus": np.stack(model.consensus_matrices_),
        "consensus_generalizability": np.stack(
            model.consensus_generalizability_matrices_
        ),
        "gini": np.asarray(model.stability_gini_scores_, dtype=float),
        "ce": np.asarray(model.stability_ce_scores_, dtype=float),
        "generalizability": np.vstack(model.generalizability_scores_),
    }


def fit_gate(**fit_kwargs) -> dict[str, np.ndarray]:
    """Fit the gate model on the gate dataset and extract its arrays."""
    return gate_arrays(gate_model().fit(gate_dataset(), **fit_kwargs))


if __name__ == "__main__":
    np.savez_compressed(GATE_PATH, **fit_gate())
    print(f"wrote {GATE_PATH}")
