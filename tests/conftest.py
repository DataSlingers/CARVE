"""Shared fixtures and hooks for the CARVE test suite."""

import importlib.util

import matplotlib
import numpy as np
import pandas as pd
import pytest

from tests._helpers import with_sweep_cols

# Select the file-backed backend once, before any test module imports
# pyplot. conftest.py is imported before collection reaches any test file.
matplotlib.use("Agg", force=True)

# Leiden and Louvain live behind the optional [graph] extra. Tests that need
# them carry @pytest.mark.requires_graph; the hook below skips them when the
# extra is absent.
_HAS_GRAPH = all(
    importlib.util.find_spec(m) is not None for m in ("igraph", "leidenalg")
)


def pytest_collection_modifyitems(config, items):
    if _HAS_GRAPH:
        return
    skip = pytest.mark.skip(reason="requires the [graph] extra (igraph + leidenalg)")
    for item in items:
        if "requires_graph" in item.keywords:
            item.add_marker(skip)


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


@pytest.fixture()
def results_df():
    """Synthetic k-mode estimator results DataFrame for selection tests."""
    return with_sweep_cols(
        pd.DataFrame(
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
        ),
        param="n_clusters",
        method_label="KMeans",
        observed=[2.0, 3.0, 4.0, 5.0],
    )


@pytest.fixture()
def resolution_results_df():
    """Synthetic resolution-mode results table for selection tests.

    ``sweep_rank`` runs coarse -> fine and ``config_id`` runs 0..n-1,
    matching what the runner writes.
    """
    return with_sweep_cols(
        pd.DataFrame(
            {
                "estimator": ["LeidenClustering"] * 4,
                "resolution": [0.25, 0.5, 1.0, 2.0],
                "n_neighbors": [15] * 4,
                "ari_stability": [0.9, 0.85, 0.7, 0.6],
                "ari_stability_se": [0.02, 0.03, 0.05, 0.06],
                "ari_stability_upper": [0.92, 0.88, 0.75, 0.66],
                "ari_stability_lower": [0.88, 0.82, 0.65, 0.54],
            }
        ),
        param="resolution",
        method_label="LeidenClustering, n_neighbors=15",
        observed=[2.0, 3.0, 5.0, 9.0],
        observed_se=[0.0, 0.1, 0.3, 0.5],
    )


@pytest.fixture()
def min_cluster_size_results_df():
    """Synthetic min_cluster_size results table (descending sweep_rank).

    ``min_cluster_size`` is the one registry parameter where larger values
    yield *fewer* clusters, so ``sweep_rank`` runs 2 -> 0 as the size grows.
    """
    return with_sweep_cols(
        pd.DataFrame(
            {
                "estimator": ["HDBSCAN"] * 3,
                "min_cluster_size": [5, 10, 25],
                "cluster_selection_method": ["eom"] * 3,
                "ari_stability": [0.70, 0.85, 0.88],
                "ari_stability_se": [0.05, 0.03, 0.02],
                "ari_stability_upper": [0.75, 0.88, 0.90],
                "ari_stability_lower": [0.65, 0.82, 0.86],
            }
        ),
        param="min_cluster_size",
        method_label="HDBSCAN, cluster_selection_method=eom",
        observed=[9.0, 5.0, 3.0],
        noise_fraction=[0.05, 0.10, 0.20],
    )


# -----------------------------------------------------------------------
# AnnData fixtures
# -----------------------------------------------------------------------


@pytest.fixture
def adata_three_clusters(X_three_clusters):
    """AnnData with three well-separated clusters, X_pca and X_umap."""
    import anndata as ad

    adata = ad.AnnData(np.asarray(X_three_clusters, dtype=np.float32))
    adata.obs_names = [f"cell{i}" for i in range(adata.n_obs)]
    adata.var_names = [f"gene{j}" for j in range(adata.n_vars)]
    adata.obsm["X_pca"] = np.asarray(X_three_clusters[:, :3], dtype=np.float32)
    adata.obsm["X_umap"] = np.asarray(X_three_clusters[:, :2], dtype=np.float32)
    adata.layers["counts"] = adata.X.copy()
    adata.obs["cell_type"] = pd.Categorical(["a"] * 30 + ["b"] * 30 + ["c"] * 30)
    return adata


@pytest.fixture
def adata_sparse(X_three_clusters):
    """AnnData whose ``.X`` is CSR, as a typical scRNA-seq object would be."""
    import anndata as ad
    from scipy import sparse

    adata = ad.AnnData(
        sparse.csr_matrix(np.asarray(X_three_clusters, dtype=np.float32))
    )
    adata.obsm["X_pca"] = np.asarray(X_three_clusters[:, :3], dtype=np.float32)
    return adata


@pytest.fixture(scope="module")
def fitted_adata():
    """A small AnnData carrying results from ``tl.carve``.

    Module-scoped: fitting is the expensive part of these tests, and none of
    the consumers mutate it.
    """
    import anndata as ad

    import carve

    rng = np.random.RandomState(0)
    X = np.vstack(
        [
            rng.randn(30, 6) + 8,
            rng.randn(30, 6) - 8,
            rng.randn(30, 6) * 0.4,
        ]
    ).astype(np.float32)
    adata = ad.AnnData(X)
    adata.obsm["X_pca"] = X[:, :4]
    adata.obsm["X_umap"] = X[:, :2]
    carve.tl.carve(
        adata,
        use_rep="X_pca",
        n_clusters=range(2, 5),
        n_resamples=5,
        random_state=0,
    )
    return adata


@pytest.fixture(autouse=True)
def _isolated_matplotlib_state():
    """Undo every rcParams change a test makes and close its figures.

    rc_context restores each rcParam changed inside the block on exit, so a
    test that calls apply_theme() or assigns plt.rcParams[...] cannot leak
    into the next test.
    """
    import matplotlib.pyplot as plt

    with matplotlib.rc_context():
        yield
    plt.close("all")
