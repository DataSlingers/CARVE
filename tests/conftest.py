"""Shared fixtures for CARVE test suite."""

import numpy as np
import pandas as pd
import pytest
from sklearn.cluster import KMeans

from carve import CARVE


@pytest.fixture()
def rng():
    """Seeded NumPy random state."""
    return np.random.RandomState(42)


@pytest.fixture()
def X_two_clusters(rng):
    """Well-separated 2-cluster dataset (60 samples, 5 features)."""
    return np.vstack(
        [
            rng.randn(30, 5) + [4, 0, 0, 0, 0],
            rng.randn(30, 5) + [0, 4, 0, 0, 0],
        ]
    )


@pytest.fixture()
def X_three_clusters(rng):
    """Well-separated 3-cluster dataset (90 samples, 5 features)."""
    return np.vstack(
        [
            rng.randn(30, 5) + [6, 0, 0, 0, 0],
            rng.randn(30, 5) + [0, 6, 0, 0, 0],
            rng.randn(30, 5) + [0, 0, 6, 0, 0],
        ]
    )


@pytest.fixture(scope="module")
def fitted_carve_module():
    """Module-scoped fitted CARVE instance for read-only tests."""
    rng = np.random.RandomState(42)
    X = np.vstack(
        [
            rng.randn(30, 5) + [4, 0, 0, 0, 0],
            rng.randn(30, 5) + [0, 4, 0, 0, 0],
        ]
    )
    carve = CARVE(
        n_clusters=2,
        n_resamples=5,
        subsample_ratio=0.8,
        estimator_param_grids=[(KMeans, {"n_clusters": [2]})],
        normalization_options=[],
        dim_reduction_options=[],
        n_jobs=1,
        random_state=0,
        verbose=0,
    )
    carve.fit(X)
    return carve


@pytest.fixture()
def results_df():
    """Synthetic estimator results DataFrame for selection tests."""
    return pd.DataFrame(
        {
            "estimator": ["KMeans"] * 4,
            "n_clusters": [2, 3, 4, 5],
            "ari_stability": [0.9, 0.85, 0.7, 0.6],
            "ari_stability_se": [0.02, 0.03, 0.05, 0.06],
            "ari_stability_upper": [0.92, 0.88, 0.75, 0.66],
            "ari_stability_lower": [0.88, 0.82, 0.65, 0.54],
            "ari_generalizability": [0.85, 0.80, 0.65, 0.55],
            "ari_generalizability_se": [0.03, 0.04, 0.06, 0.07],
            "ari_generalizability_upper": [0.88, 0.84, 0.71, 0.62],
            "ari_generalizability_lower": [0.82, 0.76, 0.59, 0.48],
            "ari_average": [0.875, 0.825, 0.675, 0.575],
            "ari_average_se": [0.025, 0.035, 0.055, 0.065],
            "ari_average_upper": [0.90, 0.86, 0.73, 0.64],
            "ari_average_lower": [0.85, 0.79, 0.62, 0.51],
            "consensus_pac_stability": [0.95, 0.90, 0.80, 0.70],
            "consensus_gini_stability": [0.92, 0.87, 0.75, 0.65],
            "consensus_ce_stability": [0.91, 0.86, 0.74, 0.64],
            "accuracy_generalizability": [0.88, 0.83, 0.68, 0.58],
        }
    )
