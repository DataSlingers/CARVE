"""Plotting helpers for simulations."""

import numpy as np
import matplotlib.pyplot as plt
import warnings
from sklearn.decomposition import PCA


def _plot_simulation(X: np.ndarray, y: np.ndarray, *, random_state: int | None):
    """Plot a 2D PCA projection of the simulated dataset.

    Parameters
    ----------
    X : ndarray of shape (n_samples, n_features)
        Data matrix.
    y : ndarray of shape (n_samples,)
        Labels array; cluster labels or -1 for outliers.
    random_state : int or None
        RNG seed used for PCA initialization.

    Returns
    -------
    None
        Displays a Matplotlib figure.
    """
    if X.shape[0] < 2 or X.shape[1] < 2:
        warnings.warn("Skipping plot: need at least 2 samples and 2 features for PCA.")
        return
    pca = PCA(n_components=2, random_state=random_state)
    X_pca = pca.fit_transform(X)

    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(
        X_pca[:, 0], X_pca[:, 1], c=y, cmap="tab10", edgecolors="k", alpha=0.8
    )
    plt.title("PCA projection of simulated clusters")
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.legend(
        *scatter.legend_elements(),
        title="Clusters",
        bbox_to_anchor=(1.05, 1),
        loc="upper left",
    )
    plt.tight_layout()
    plt.show()
